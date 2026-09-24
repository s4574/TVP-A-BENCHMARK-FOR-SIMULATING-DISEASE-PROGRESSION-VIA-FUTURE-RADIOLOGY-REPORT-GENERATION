"""
Unified sealed-test scorer for the temporal benchmark: scores ANY method with the SAME metric
(per-ns per-concept F1 macro + instance micro-F1 + bootstrap CI), so all rows of the main results
table are comparable. Modes:
  --mode zeroshot                       (LLM zero-shot via API)
  --mode ml --method logreg|rf|xgb|mlp  (structured ML, trained on train split)
  --mode sft --ckpt /path/to/ckpt       (fine-tuned LLM: generate on test text, parse findings)
Default test set = data/temporal_query_test.jsonl (sealed, subject-split). Vocab from cohort_rad_findings.
Use --test data/temporal_episode_query_test.jsonl for episode-aware disease-evolution evaluation.
Usage: python3 eval_temporal_unified.py --mode <...> [--method ..] [--ckpt ..] [--test ..] [--n 0(all)]
"""
import sys, json, argparse, collections, re
import numpy as np
sys.path.insert(0, "${VP_ROOT}/analysis")

TEST = "${VP_ROOT}/data/temporal_query_test.jsonl"
FIND = "${VP_ROOT}/analysis/cohort_rad_findings.jsonl"
POS = {"present", "uncertain"}

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["zeroshot", "ml", "sft"])
ap.add_argument("--method", default="logreg")
ap.add_argument("--ckpt", default="")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--gpu", default="0")
ap.add_argument("--test", default=TEST)
ap.add_argument("--agent-data", default="${VP_ROOT}/analysis/agent_ml_data.pkl")
args = ap.parse_args()

# ---- per-ns concept vocab (question space) ----
nsc = collections.defaultdict(collections.Counter)
for l in open(FIND):
    try: o = json.loads(l)
    except: continue
    if o.get("error"): continue
    for a in o.get("attrs", {}):
        nsc[o["ns"]][a.split(".", 2)[-1]] += 1
VOCAB = {ns: [c for c, n in cnt.most_common() if n >= 5][:15] for ns, cnt in nsc.items()}

rows = [json.loads(l) for l in open(args.test) if l.strip()]
if args.n: rows = rows[:args.n]

def gold_pos(rec):
    g = {}
    for a, p in rec["gold"].items(): g[a.split(".", 2)[-1]] = p
    return {c for c in VOCAB.get(rec["ns"], []) if g.get(c) in POS}

# ---- predictors -> return set of predicted-present concepts (within ns vocab) ----
def make_predictor():
    if args.mode == "zeroshot":
        from llm_client import chat_full
        SYS = "You predict which findings an unperformed exam would show, given other data. Output ONLY JSON."
        P = 'PATIENT DATA:\n%s\n\nFor exam "%s", predict present|absent|uncertain for EACH candidate.\nCandidates: %s\nJSON {"findings":{"<c>":"present|absent|uncertain"}}'
        def pred(rec):
            cands = VOCAB.get(rec["ns"], [])
            if not cands: return set()
            out, _ = chat_full([{"role": "system", "content": SYS}, {"role": "user", "content": P % (rec["context"][:4000], rec.get("exam_name", rec["ns"]), ", ".join(cands))}], model="gpt-5.4-mini", max_tokens=600)
            try: o = json.loads(out[out.find("{"):out.rfind("}") + 1]).get("findings", {})
            except: o = {}
            return {c for c in cands if o.get(c) in POS}
        return pred
    if args.mode == "sft":
        import os; os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.ckpt); tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
        SYS = ("You are a clinical world model. Given a patient's observations before a requested radiology exam, "
               "predict that exam's structured findings as JSON {\"findings\": {\"<attr>\": \"present|absent|uncertain\"}} "
               "using the exam's namespaced attributes.")
        USER = "PATIENT OBSERVATIONS BEFORE TARGET EXAM:\n%s\n\nREQUESTED EXAM (predict findings): %s (%s)\nJSON:"
        def pred(rec):
            cands = VOCAB.get(rec["ns"], []);
            if not cands: return set()
            msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": USER % (rec["context"], rec.get("exam_name", ""), rec["ns"])}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=3500).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id)
            txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            try: o = json.loads(txt[txt.find("{"):txt.rfind("}") + 1]).get("findings", {})
            except: o = {}
            # o keys may be full namespaced or bare concept
            got = set()
            for c in cands:
                for key in (c, f"{rec['ns']}.{c}"):
                    if o.get(key) in POS: got.add(c)
            return got
        return pred
    if args.mode == "ml":
        # train structured ML per-ns per-concept on train split (reuse precomputed features)
        import pickle
        d = pickle.load(open(args.agent_data, "rb"))
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from xgboost import XGBClassifier
        clfs = {}  # (ns) -> (per-concept models, prep stats)
        for ns in d["top_ns"]:
            Xtr, Ytr = d["data"][ns]["Xtr"], d["data"][ns]["Ytr"]
            med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
            Xi = np.where(np.isnan(Xtr), med, Xtr); mu, sd = Xi.mean(0), Xi.std(0) + 1e-8
            models = []
            for k in range(Ytr.shape[1]):
                y = Ytr[:, k]
                if y.sum() < 5 or y.sum() == len(y): models.append(None); continue
                m = {"logreg": LogisticRegression(max_iter=300, class_weight="balanced"),
                     "rf": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=4),
                     "xgb": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, n_jobs=4, verbosity=0),
                     "mlp": MLPClassifier(hidden_layer_sizes=(64,), max_iter=200)}[args.method]
                m.fit((Xi - mu) / sd, y); models.append(m)
            clfs[ns] = (models, med, mu, sd, d["ns_vocab"][ns])
        # NOTE: ML uses structured features -> needs test features; here we score ML on its OWN test arrays (aligned)
        # so ML mode reports on the pkl test split (same subject split as temporal_query_test).
        print(f"[note] ML mode scores on precomputed arrays from {args.agent_data}.")
        return ("ml_special", clfs, d)
    raise SystemExit("bad mode")

def score_instances(insts):  # insts: list of (ns, gold_set, pred_set)
    stats = collections.defaultdict(collections.Counter); itp = ifp = ifn = 0
    for ns, gp, pp in insts:
        for c in VOCAB.get(ns, []):
            if c in gp and c in pp: stats[(ns, c)]["tp"] += 1
            elif c in pp: stats[(ns, c)]["fp"] += 1
            elif c in gp: stats[(ns, c)]["fn"] += 1
        itp += len(gp & pp); ifp += len(pp - gp); ifn += len(gp - pp)
    def f1(c):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]; P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
        return 2*P*R/(P+R) if P+R else 0.0
    by_ns = collections.defaultdict(list)
    for (ns, c), cc in stats.items(): by_ns[ns].append(f1(cc))
    macro = float(np.mean([np.mean(v) for v in by_ns.values()])) if by_ns else 0
    P = itp/(itp+ifp) if itp+ifp else 0; R = itp/(itp+ifn) if itp+ifn else 0
    micro = 2*P*R/(P+R) if P+R else 0
    # bootstrap CI on micro
    rng = np.random.default_rng(0); n = len(insts)
    def micro_of(idx):
        tp=fp=fn=0
        for i in idx: _, gp, pp = insts[i]; tp+=len(gp&pp); fp+=len(pp-gp); fn+=len(gp-pp)
        P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0; return 2*P*R/(P+R) if P+R else 0
    boot = [micro_of(rng.integers(0, n, n)) for _ in range(300)] if n else [0]
    return macro, micro, (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))), by_ns

predictor = make_predictor()
if isinstance(predictor, tuple) and predictor[0] == "ml_special":
    _, clfs, d = predictor; insts = []
    for ns in d["top_ns"]:
        Xte, Yte = d["data"][ns]["Xte"], d["data"][ns]["Yte"]; models, med, mu, sd, voc = clfs[ns]
        if len(Xte) == 0: continue
        Xi = np.where(np.isnan(Xte), med, Xte); Xs = (Xi - mu) / sd
        for i in range(len(Xte)):
            gp = {voc[k] for k in range(len(voc)) if Yte[i, k] == 1}
            pp = set()
            for k, m in enumerate(models):
                if m is not None and m.predict(Xs[i:i+1])[0] == 1: pp.add(voc[k])
            insts.append((ns, gp & set(VOCAB.get(ns, [])), pp & set(VOCAB.get(ns, []))))
else:
    insts = []
    import concurrent.futures as cf
    if args.mode == "zeroshot":
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            preds = list(ex.map(lambda r: (r["ns"], gold_pos(r), predictor(r)), rows))
        insts = [p for p in preds if VOCAB.get(p[0])]
    else:  # sft (sequential generation)
        for i, r in enumerate(rows):
            if not VOCAB.get(r["ns"]): continue
            insts.append((r["ns"], gold_pos(r), predictor(r)))
            if (i + 1) % 200 == 0: print(f"  {i+1}/{len(rows)}", flush=True)

macro, micro, ci, by_ns = score_instances(insts)
tag = args.mode + (f":{args.method}" if args.mode == "ml" else (f":{args.ckpt.split('/')[-2] if args.ckpt else ''}" if args.mode == "sft" else ""))
out = [f"=== UNIFIED sealed-test score [{tag}]  n={len(insts)} ===",
       f"macro-F1 (per-ns per-concept): {macro:.3f}",
       f"instance micro-F1: {micro:.3f}  95%CI[{ci[0]:.3f},{ci[1]:.3f}]",
       "per-ns macro-F1:"]
for ns, v in sorted(by_ns.items(), key=lambda x: -len(x[1]))[:10]:
    out.append(f"   {ns:22s} {np.mean(v):.3f} ({len(v)} concepts)")
rep = "\n".join(out); print(rep)
safe_test = args.test.rsplit("/", 1)[-1].replace(".jsonl", "")
open(f"${VP_ROOT}/analysis/unified_{safe_test}_{args.mode}_{args.method if args.mode=='ml' else 'x'}.txt", "w").write(rep + "\n")
