"""
Render the auto-research committed four-state config to report-score (path B endpoint).

Uses the four-state pkl: per ns, trains ensemble(rf+xgb) on the train pool, calibrates per-concept thresholds
on internal val, predicts the sealed te -> predicted-present concepts per case (aligned via te_keys). Then
renders each predicted state to a report (shared render_prompt) and scores with the SAME fixed extractor as the
end-to-end baseline. Two reads: state-direct (from ML) and report-score (primary, comparable to LLM-module
0.473 and end-to-end 0.555).

Anti-leakage: thresholds calibrated on internal val only; sealed te used only for final scoring. Render+extract
cost API (start with --n small).
"""
import argparse, concurrent.futures as cf, json, os, pickle, sys, threading, time
import numpy as np

sys.path.insert(0, "${VP_ROOT}/analysis")
import run_direct_vp_report_smoke as R
import run_state_pipeline_smoke as SP
import run_temporal_llm_smoke_v2 as sm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT = "${VP_ROOT}"
NJOBS = int(os.environ.get("VP_AGENT_NJOBS", "2"))
THR = np.round(np.linspace(0.1, 0.9, 17), 3)
POS = {"present", "uncertain"}


def _f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    P = tp / (tp + fp) if tp + fp else 0; Rc = tp / (tp + fn) if tp + fn else 0
    return 2 * P * Rc / (P + Rc) if P + Rc else 0.0


def predict_sealed(pkl):
    """Return key -> set(predicted-present concepts) using ensemble(rf+xgb)+calibrated, and per-key gold-direct."""
    d = pickle.load(open(pkl, "rb"))
    key2pred = {}
    for ns in d["top_ns"]:
        D = d["data"][ns]; concepts = d["ns_vocab"][ns]; keys = d["te_keys"][ns]
        Xtr, Ytr, Xval, Yval, Xte = D["Xtr"], D["Ytr"], D["Xval"], D["Yval"], D["Xte"]
        if len(Xtr) == 0 or len(Xte) == 0:
            continue
        pred = np.zeros((len(Xte), len(concepts)))
        for k in range(Ytr.shape[1]):
            ytr = Ytr[:, k]
            if ytr.sum() < 5 or ytr.sum() == len(ytr):
                continue
            probas_val = []; probas_te = []
            for M in (RandomForestClassifier(n_estimators=100, n_jobs=NJOBS),
                      XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, n_jobs=NJOBS, verbosity=0)):
                try:
                    M.fit(Xtr, ytr)
                    probas_val.append(M.predict_proba(Xval)[:, 1] if len(Xval) else np.zeros(0))
                    probas_te.append(M.predict_proba(Xte)[:, 1])
                except Exception:
                    pass
            if not probas_te:
                continue
            pv = np.mean(probas_val, axis=0) if len(Xval) and probas_val else None
            pt = np.mean(probas_te, axis=0)
            thr = 0.5
            if pv is not None and len(pv):
                yv = Yval[:, k]; best = -1
                for t in THR:
                    f = _f1(yv, (pv >= t).astype(int))
                    if f > best:
                        best, thr = f, t
            pred[:, k] = (pt >= thr).astype(int)
        for i, key in enumerate(keys):
            key2pred[tuple(str(x) for x in key)] = {concepts[k] for k in range(len(concepts)) if pred[i, k] == 1}
    return key2pred


def predict_sealed_from_decision(pkl, decision_path):
    """Replay an auto-research committed config (agentB_decision.json best_config) through TemporalEnv and
    return key -> set(predicted-present concepts). Supports sklearn/ensemble arms and the conditional
    transition arm; thresholds are calibrated on internal val per the committed threshold_mode."""
    os.environ["VP_AGENT_ML_DATA"] = pkl
    sys.path.insert(0, ROOT)
    import importlib
    import playground.env_temporal as ET
    importlib.reload(ET)
    best = json.load(open(decision_path))["best_config"]
    env = ET.TemporalEnv(compute_units=10**6, eval_queries=10**6)
    if best.get("recipe"): env.recipe = best["recipe"]
    if best.get("threshold_mode"): env.thr_mode = best["threshold_mode"]
    if best.get("min_pos"): env.min_pos = int(best["min_pos"])
    env.hparams = best.get("hparams") or {}
    if best.get("feature_groups"): env.active_groups = list(best["feature_groups"])
    if best.get("state_concepts"):
        env.active_concepts = list(best["state_concepts"]); env.active_groups = None
        env.concept_mode = best.get("state_concepts_mode") or "evidence"
    model = best["model"]
    print(f"replaying committed config: {json.dumps({k: best.get(k) for k in ('model','recipe','threshold_mode','min_pos','hparams','feature_groups','state_concepts','state_concepts_mode','transition_config')}, ensure_ascii=False)}")
    if model == "transition":
        r = env.train_transition(best.get("transition_config") or {})
        assert r.get("ok"), r
        m, preds = env._score_transition("te", return_preds=True)
    else:
        env._score(model, "val")            # fit + calibrate thresholds on internal val
        m, preds = env._score(model, "te", return_preds=True)
    print(f"replay sealed state-direct: {m}")
    key2pred = {}
    for ns in env.top_ns:
        concepts = env.ns_vocab[ns]
        for i, key in enumerate(env.te_keys[ns]):
            key2pred[tuple(str(x) for x in key)] = {concepts[k] for k in range(len(concepts)) if preds[ns][i, k] == 1}
    return key2pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=f"{ROOT}/analysis/agent_ml_data_episode_fourstate.pkl")
    ap.add_argument("--decision", default=None,
                    help="agentB_decision.json to replay (committed config incl. transition arm); "
                         "otherwise the legacy hardcoded ensemble(rf+xgb)+calibrated is used")
    ap.add_argument("--test", default=f"{ROOT}/data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl")
    ap.add_argument("--schema", default=f"{ROOT}/analysis/eval_schema_auto_train.json")
    ap.add_argument("--config", default=f"{ROOT}/llm_api/gpt_56_sol.yaml")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--parser-model", default="gpt-5.4-mini")
    ap.add_argument("--tag", default="fourstate_autoresearch")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--api-retries", type=int, default=3)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    cfgs = sm.load_model_config(args.config)
    vocab = R.build_vocab_from_schema(args.schema)
    key2pred = (predict_sealed_from_decision(args.pkl, args.decision) if args.decision
                else predict_sealed(args.pkl))
    print(f"predicted sealed cases={len(key2pred)}")
    te_rows = [json.loads(l) for l in open(args.test) if l.strip()]
    te_rows = [r for r in te_rows if vocab.get(r.get("ns")) and tuple(str(x) for x in
               [r.get("hadm_id"), r.get("charttime"), r.get("ns"), r.get("exam_name")]) in key2pred]
    te_rows = te_rows[: args.n] if args.n > 0 else te_rows
    print(f"scorable+matched rows={len(te_rows)}")
    if not args.run:
        # state-direct only (no API)
        insts = []
        for r in te_rows:
            key = tuple(str(x) for x in [r.get("hadm_id"), r.get("charttime"), r.get("ns"), r.get("exam_name")])
            insts.append((R.gold_pos(r, vocab), key2pred[key]))
        tp = sum(len(g & p) for g, p in insts); fp = sum(len(p - g) for g, p in insts); fn = sum(len(g - p) for g, p in insts)
        P = tp / (tp + fp) if tp + fp else 0; Rc = tp / (tp + fn) if tp + fn else 0
        print(json.dumps({"dry_run_state_direct": {"micro_f1": round(2 * P * Rc / (P + Rc) if P + Rc else 0, 4),
                          "precision": round(P, 4), "recall": round(Rc, 4)}}, ensure_ascii=False))
        return

    cfg = dict(cfgs[args.model]); cfg["max_retry"] = args.api_retries
    parser_cfg = dict(cfgs[args.parser_model]); parser_cfg["max_retry"] = args.api_retries
    render_cache_path = f"{ROOT}/analysis/rendered_report_{args.model}_{args.tag}.jsonl"
    parse_cache_path = f"{ROOT}/analysis/direct_vp_report_parse_{args.parser_model}_{R.PARSE_PROMPT_VERSION}.jsonl"
    render_cache = R.load_cache(render_cache_path); parse_cache = R.load_cache(parse_cache_path)
    r_lock = threading.Lock(); p_lock = threading.Lock()

    def one(r):
        ns = r["ns"]; key = tuple(str(x) for x in [r.get("hadm_id"), r.get("charttime"), ns, r.get("exam_name")])
        pp_state = key2pred[key]; concepts = vocab[ns]
        pred_state = {c: ("present" if c in pp_state else "not_mentioned") for c in concepts}
        gp = R.gold_pos(r, vocab)
        rk = SP.skey(args.model, r, args.tag + "_render")
        if rk in render_cache and not (args.retry_errors and render_cache[rk].get("error")):
            ro = render_cache[rk]
        else:
            try:
                t0 = time.time(); text, usage, lat = sm.call_chat(SP.render_prompt(r, pred_state), cfg, 900, 0.1)
                ro = {"key": rk, "text": text, "usage": usage, "latency_sec": lat if lat is not None else time.time() - t0, "error": None}
            except Exception as e:
                ro = {"key": rk, "text": "", "usage": {}, "latency_sec": None, "error": str(e)}
            with r_lock:
                open(render_cache_path, "a").write(json.dumps(ro, ensure_ascii=False) + "\n"); render_cache[rk] = ro
        report, _, rok = R.parse_report_wrapper(ro.get("text", "")) if not ro.get("error") else ("", None, False)
        r2 = dict(r); r2["_gen_model"] = f"{args.tag}_{args.model}"
        attrs, pok, _ = R.generated_attrs_from_report(report, r2, args.parser_model, parser_cfg, parse_cache, parse_cache_path, p_lock, args.retry_errors) if report else ({}, False, {})
        return {"gp": gp, "pp_state": pp_state, "pp_report": R.pred_pos_from_attrs(attrs, ns, vocab), "rok": rok, "pok": pok}

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(one, te_rows))

    def micro(pairs):
        tp = sum(len(g & p) for g, p in pairs); fp = sum(len(p - g) for g, p in pairs); fn = sum(len(g - p) for g, p in pairs)
        P = tp / (tp + fp) if tp + fp else 0; Rc = tp / (tp + fn) if tp + fn else 0
        return {"precision": round(P, 4), "recall": round(Rc, 4), "micro_f1": round(2 * P * Rc / (P + Rc) if P + Rc else 0, 4)}
    summary = {"n": len(res), "tag": args.tag, "model": args.model,
               "report_score_primary": micro([(r["gp"], r["pp_report"]) for r in res]),
               "state_dict_direct": micro([(r["gp"], r["pp_state"]) for r in res]),
               "report_parse_ok": sum(r["pok"] for r in res), "wall_sec": round(time.time() - t0, 1),
               "baselines": {"llm_module_report": 0.473, "end2end_mini_report": 0.555}}
    out = f"{ROOT}/analysis/fourstate_render_{args.tag}_summary.json"
    json.dump(summary, open(out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2)); print(f"-> {out}")


if __name__ == "__main__":
    main()
