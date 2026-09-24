"""Pilot: probability-annotated (C) section on the error subset (end-to-end right, fullctx-advisory wrong).
(C) lists every target-ns schema concept with the state predictor's P(positive), sorted descending, plus
reading guidance (>=0.5 strong prior; 0.3-0.5 moderate; <0.3 weak -- mention only if (A) supports).
Everything else identical to the fullctx advisory arm. Scored per-case vs the flat-list advisory output."""
import json, sys, threading
import numpy as np

sys.path.insert(0, "."); sys.path.insert(0, "analysis")
import run_direct_vp_report_smoke as R
import run_state_pipeline_smoke as SP
import run_temporal_llm_smoke_v2 as sm
import render_enriched as RE

vocab = R.build_vocab_from_schema("analysis/eval_schema_auto_train.json")
full_rows = {}
for l in open("data/direct_vp_full_ehr_raw_reports_test_stratified_1500.jsonl"):
    r = json.loads(l); full_rows[(str(r["hadm_id"]), r["charttime"], r["ns"])] = r

# --- error subset: e2e perfect, fullctx advisory wrong (same criteria as case-study selection) ---
import glob
f = [x for x in glob.glob("analysis/direct_vp_report_gpt-5.4-mini*records_n1500.jsonl") if "v3_full_ehr" in x][0]
e2e = {(str(r["hadm_id"]), r["charttime"], r["ns"]): r for r in (json.loads(l) for l in open(f)) if not r.get("gen_error")}
parse_cache = R.load_cache("analysis/direct_vp_report_parse_gpt-5.4-mini_%s.jsonl" % R.PARSE_PROMPT_VERSION)
rc_old = R.load_cache("analysis/rendered_report_gpt-5.4-mini_enriched_advisory_fullctx_mini.jsonl")
subset = []
for k, r in full_rows.items():
    if not vocab.get(r["ns"]):
        continue
    ro = rc_old.get(SP.skey("gpt-5.4-mini", r, "enriched_advisory_fullctx_mini_render"))
    if not ro or ro.get("error"):
        continue
    report, _, _ = R.parse_report_wrapper(ro["text"])
    r2 = dict(r); r2["_gen_model"] = "enriched_advisory_fullctx_mini_gpt-5.4-mini"
    o = parse_cache.get(R.parse_cache_key("gpt-5.4-mini", r2["_gen_model"], r2, report))
    if not o or o.get("error"):
        continue
    obj = R.extract_json_obj(o.get("text", "")) or {"observations": []}
    attrs = {f"{r['ns']}.{(ob.get('concept') or '').strip()}": {"presence": ob.get("presence", "present")}
             for ob in obj.get("observations", []) if (ob.get("concept") or "").strip()}
    pp = R.pred_pos_from_attrs(attrs, r["ns"], vocab); gp = R.gold_pos(r, vocab)
    e = e2e.get(k)
    if not e or not gp:
        continue
    if (not e["fp"] and not e["fn"]) and (len(pp - gp) + len(gp - pp)) >= 2:
        subset.append((k, gp, pp))
print(f"error subset: {len(subset)} cases, ns set: {sorted({k[2] for k,_,_ in subset})}", flush=True)

# --- per-concept probabilities via committed-config replay (only needed namespaces) ---
import importlib, os
os.environ["VP_AGENT_ML_DATA"] = "analysis/agent_ml_data_episode_condtrans_union_stratval.pkl"
import playground.env_temporal as ET
importlib.reload(ET)
best = json.load(open("analysis/agentB_decision_condtrans_mini.json"))["best_config"]
env = ET.TemporalEnv(compute_units=10**6, eval_queries=10**6)
env.recipe = best["recipe"]; env.thr_mode = best["threshold_mode"]; env.min_pos = best["min_pos"]
env.hparams = best["hparams"] or {}; env.active_groups = list(best["feature_groups"])
methods = env._methods_of(best["model"])

def _f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    P = tp / (tp + fp) if tp + fp else 0; Rc = tp / (tp + fn) if tp + fn else 0
    return 2 * P * Rc / (P + Rc) if P + Rc else 0.0

need_ns = sorted({k[2] for k, _, _ in subset})
key2probs = {}
for ns in need_ns:
    Xtr, Ytr = env.work[ns]["Xtr"], env.work[ns]["Ytr"]
    if len(Xtr) > ET.CAP:
        sel = np.random.default_rng(0).choice(len(Xtr), ET.CAP, replace=False); Xtr, Ytr = Xtr[sel], Ytr[sel]
    Xval, Yval, Xte = env.work[ns]["Xval"], env.work[ns]["Yval"], env.data[ns]["Xte"]
    cols = env._cols(ns)
    if cols is not None:
        Xtr, Xval, Xte = Xtr[:, cols], Xval[:, cols], Xte[:, cols]
    Xtrs, Xvals = env._prep(Xtr, Xval); _, Xtes = env._prep(Xtr, Xte)
    concepts = env.ns_vocab[ns]
    probs = np.full((len(Xte), len(concepts)), np.nan)
    for c in range(Ytr.shape[1]):
        ytr = Ytr[:, c]
        if ytr.sum() < env.min_pos or ytr.sum() == len(ytr):
            continue
        probs[:, c] = np.mean([env._proba_one(m, Xtrs, ytr, Xtes) for m in methods], axis=0)
    for i, key in enumerate(env.te_keys[ns]):
        key2probs[tuple(str(x) for x in key[:3])] = {concepts[c]: (None if np.isnan(probs[i, c]) else round(float(probs[i, c]), 2))
                                                     for c in range(len(concepts))}
    print(f"probs done: {ns}", flush=True)

# --- probability-annotated prompt (advisory base, modified C) ---
def prob_prompt(rec, prior_state, probs, vocab_ns):
    context = rec.get("context", "") or ""
    mod, region = RE.ns_human(rec.get("ns", ""))
    prior_lines = "\n".join(f"- {c}: {s}" for c, s in sorted(prior_state.items())) or "- (none documented)"
    ranked = sorted(((c, p) for c, p in probs.items() if p is not None), key=lambda x: -x[1])
    plines = "\n".join(f"- {c}: {p:.2f}" for c, p in ranked)
    skipped = [c for c, p in probs.items() if p is None]
    sys_msg = ("You are acting as a virtual patient simulator for a clinical forecasting benchmark. "
               "Given observations available before a requested radiology exam, plus two structured state summaries "
               "from an upstream prediction pipeline, generate the radiology report text that would plausibly be "
               "documented for that exam. Output ONLY JSON.")
    user = (
        "TASK:\nThe target radiology report is hidden. Generate the free-text radiology report that would plausibly be "
        "documented for the requested target exam, using only patient information available before that exam. "
        "This is a forecasting/simulation task, not extraction from the hidden report.\n\n"
        "You are given THREE information sources:\n"
        "(A) RAW PATIENT CONTEXT: prior admissions, prior radiology reports (verbatim), diagnoses, medications, "
        "procedures and ED events documented before the target exam.\n"
        "(B) CURRENT STRUCTURED STATE: finding concepts extracted from this patient's PRIOR imaging reports, "
        "each labeled present/absent/uncertain as of the most recent prior exams; unlisted concepts were never "
        "mentioned. It summarizes what is already known and contains NO information about the future.\n"
        "(C) PREDICTED PROBABILITIES AT TARGET TIME: a learned transition model's per-finding probability that "
        "each schema concept will be POSITIVE in the target exam. The model conditioned on the time gap and "
        "interventions. READ THE NUMBERS: >=0.50 is strong prior evidence; 0.30-0.50 is moderate -- weigh against "
        "(A); <0.30 is weak -- mention it only if (A) itself supports it. These probabilities are NOT equally "
        "reliable claims; do not treat them as a checklist.\n"
        "(B) and (C) are ADVISORY, not ground truth. Write the report primarily from your own clinical reasoning "
        "over (A); override (B)/(C) whenever (A) clearly indicates otherwise.\n\n"
        "REPORT REQUIREMENTS:\n"
        "- Concise radiology-report style with EXAMINATION, FINDINGS, and IMPRESSION sections.\n"
        "- Include findings the target exam would likely state, including normal/negative findings when normally "
        "documented.\n- No checklists or structured labels inside the report.\n"
        "- Prior reports are prior evidence only; do not copy them as the target report.\n\n"
        "=== (A) RAW PATIENT CONTEXT (before the target exam) ===\n"
        f"{context}\n\n"
        "=== (B) CURRENT STRUCTURED STATE (from prior reports; before target time) ===\n"
        f"{prior_lines}\n\n"
        f"=== (C) PREDICTED PROBABILITIES AT TARGET TIME (exam type {rec.get('ns','')}) ===\n"
        f"{plines}\n"
        + (f"(no reliable estimate for: {', '.join(skipped)})\n" if skipped else "")
        + "\n=== REQUESTED TARGET EXAM ===\n"
        f"Exam name: {rec.get('exam_name', rec.get('ns', ''))}\n"
        f"Modality: {mod}; body region: {region}\n"
        f"Exam namespace/type: {rec.get('ns', '')}\n"
        f"Target time: {rec.get('charttime', '')}\n\n"
        'Output JSON exactly as {"report":"<free-text radiology report>"}.'
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]

# --- render + score the subset ---
key2prior = RE.prior_states("analysis/agent_ml_data_episode_condtrans_union_stratval.pkl")
cfgs = sm.load_model_config("llm_api/gpt_56_sol.yaml")
cfg = dict(cfgs["gpt-5.4-mini"]); cfg["max_retry"] = 4
tag = "enriched_probC_fullctx_mini"
rcp = f"analysis/rendered_report_gpt-5.4-mini_{tag}.jsonl"
rcache = R.load_cache(rcp); rl = threading.Lock(); pl = threading.Lock()
res = []
for k, gp, pp_old in subset:
    r = full_rows[k]
    k4 = tuple(str(x) for x in [r["hadm_id"], r["charttime"], r["ns"], r["exam_name"]])
    rk = SP.skey("gpt-5.4-mini", r, tag + "_render")
    if rk in rcache and not rcache[rk].get("error"):
        ro = rcache[rk]
    else:
        try:
            text, usage, lat = sm.call_chat(prob_prompt(r, key2prior.get(k4, {}), key2probs[k], vocab[r["ns"]]), cfg, 1200, 0.1)
            ro = {"key": rk, "text": text, "usage": usage, "latency_sec": lat, "error": None}
        except Exception as e:
            ro = {"key": rk, "text": "", "usage": {}, "latency_sec": None, "error": str(e)}
        with rl:
            open(rcp, "a").write(json.dumps(ro, ensure_ascii=False) + "\n"); rcache[rk] = ro
    report, _, _ = R.parse_report_wrapper(ro.get("text", "")) if not ro.get("error") else ("", None, False)
    r2 = dict(r); r2["_gen_model"] = f"{tag}_gpt-5.4-mini"
    attrs, pok, _ = R.generated_attrs_from_report(report, r2, "gpt-5.4-mini", cfg, parse_cache,
        "analysis/direct_vp_report_parse_gpt-5.4-mini_%s.jsonl" % R.PARSE_PROMPT_VERSION, pl, False) if report else ({}, False, {})
    pp_new = R.pred_pos_from_attrs(attrs, r["ns"], vocab)
    def cf1(pp):
        d = 2 * len(gp & pp) + len(pp - gp) + len(gp - pp)
        return 2 * len(gp & pp) / d if d else 1.0
    res.append((k, sorted(gp), sorted(pp_old), sorted(pp_new), cf1(pp_old), cf1(pp_new)))
    print(f"{k[0]}|{k[2][4:]}: gold={sorted(gp)} | flat-advisory={sorted(pp_old)} F1={cf1(pp_old):.2f} -> probC={sorted(pp_new)} F1={cf1(pp_new):.2f}", flush=True)

import statistics
old = [x[4] for x in res]; new = [x[5] for x in res]
print(f"\n=== 试点汇总(n={len(res)},全部为 flat-advisory 出错 & 端到端全对的 case) ===")
print(f"per-case F1: flat-advisory mean {statistics.mean(old):.3f} -> probC mean {statistics.mean(new):.3f}")
print(f"改善 case 数: {sum(1 for a,b in zip(old,new) if b>a)} / 持平 {sum(1 for a,b in zip(old,new) if b==a)} / 变差 {sum(1 for a,b in zip(old,new) if b<a)}")
json.dump([{ 'key':list(k),'gold':g,'flat':o,'probC':n,'f1_flat':a,'f1_prob':b} for k,g,o,n,a,b in res],
          open('analysis/pilot_probC_results.json','w'), ensure_ascii=False, indent=1)
