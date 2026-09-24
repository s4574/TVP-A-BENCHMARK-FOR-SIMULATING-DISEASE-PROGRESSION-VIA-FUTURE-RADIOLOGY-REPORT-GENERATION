"""
Plan B pipeline (path B) smoke: state-prediction module -> report renderer -> SAME extractor/scorer as the
end-to-end baseline (run_direct_vp_report_smoke). This is the LLM-API-as-state-module arm:

  1. STATE MODULE: given pre-target context + target exam + target time (+ pre-target interventions already in
     context), an API LLM predicts the post-target STRUCTURED STATE over the ns's schema concepts
     (present|absent|uncertain|not_mentioned). It does NOT write a free-text report -> this is what distinguishes
     it from the end-to-end baseline.
  2. RENDERER: an API LLM renders that predicted state dict into a free-text radiology report.
  3. SCORE: reuse run_direct_vp_report_smoke.generated_attrs_from_report (fixed gpt-5.4-mini +
     cohort_rad_extract_prompt_v1) + the same schema vocab -> identical scoring path as the baseline.

Two reads per case are recorded: state-dict-direct F1 (diagnostic/upper bound) and report-score (primary,
apples-to-apples with the end-to-end baseline). All intermediates (predicted state, rendered report, extracted
attrs) are cached so new metrics can be added later without re-calling the API.

Resource note: shared/oversubscribed host. Start workers=1, ramp only after a clean small smoke.
"""
import argparse, collections, concurrent.futures as cf, json, os, sys, threading, time

sys.path.insert(0, "${VP_ROOT}/analysis")
import run_temporal_llm_smoke_v2 as sm            # noqa: E402  (config loader + call_chat + load_rows)
import run_direct_vp_report_smoke as R            # noqa: E402  (extractor + scorer + vocab, reused verbatim)

ROOT = "${VP_ROOT}"
POS = {"present", "uncertain"}
STATE_CLASSES = ["present", "absent", "uncertain", "not_mentioned"]


def state_module_prompt(rec, concepts):
    """Ask the LLM to predict the post-target structured state over the fixed concept list (schema-aware)."""
    context = rec.get("context", "") or ""
    sys_msg = ("You are a clinical state-forecasting module in a fixed pipeline. Given only information available "
               "BEFORE a requested future radiology exam, predict the patient's state AT the target exam time for a "
               "fixed list of findings. You do NOT write a report. Output ONLY JSON.")
    user = (
        "TASK: For the requested future exam, predict each listed finding's status at the target time using only "
        "pre-target information (labs, meds, interventions, prior imaging). This is forecasting, not extraction.\n\n"
        f"PATIENT CONTEXT BEFORE TARGET EXAM:\n{context}\n\n"
        f"REQUESTED TARGET EXAM: {rec.get('exam_name', rec.get('ns',''))}  (type {rec.get('ns','')})\n"
        f"TARGET TIME: {rec.get('charttime','')}\n\n"
        "FINDINGS TO PREDICT (use EXACTLY these keys):\n" + json.dumps(concepts, ensure_ascii=False) + "\n\n"
        "For each finding output one of: present | absent | uncertain | not_mentioned. Use 'present' only if you "
        "predict the target report would document it, 'absent' if it would be explicitly negated, 'not_mentioned' "
        "if the target report would likely not address it.\n"
        'Output JSON exactly as {"state":{"<finding>":"present|absent|uncertain|not_mentioned", ...},'
        '"confidence":"<one of ' + " / ".join(sm.CONFIDENCE_CLASSES) + '>"}.'
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def render_prompt(rec, pred_state):
    """Render a free-text report from the predicted structured state (shared renderer for all pipeline arms)."""
    present = [c for c, s in pred_state.items() if s == "present"]
    uncertain = [c for c, s in pred_state.items() if s == "uncertain"]
    absent = [c for c, s in pred_state.items() if s == "absent"]
    sys_msg = ("You are the report-writer stage of a clinical pipeline. Convert a predicted structured finding state "
               "into a plausible free-text radiology report. Output ONLY JSON.")
    user = (
        "Write a concise radiology-report-style text (EXAMINATION, FINDINGS, IMPRESSION) that faithfully expresses "
        "the predicted state below. State present findings as positive; state explicitly-absent findings as negative "
        "when normally documented; do not introduce findings not implied by the state.\n\n"
        f"TARGET EXAM: {rec.get('exam_name', rec.get('ns',''))}  (type {rec.get('ns','')})\n"
        f"PREDICTED PRESENT: {present}\nPREDICTED UNCERTAIN: {uncertain}\nPREDICTED ABSENT: {absent}\n\n"
        'Output JSON exactly as {"report":"<free-text radiology report>"}.'
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def parse_state(text, concepts):
    try:
        obj = R.extract_json_obj(text) or {}
        st = obj.get("state", {}) if isinstance(obj, dict) else {}
    except Exception:
        st = {}
    out = {}
    for c in concepts:
        v = st.get(c)
        out[c] = v if v in STATE_CLASSES else "not_mentioned"
    return out


def skey(model, rec, tag):
    import hashlib
    h = hashlib.md5(f"{tag}|{model}|{rec.get('hadm_id')}|{rec.get('charttime')}|{rec.get('ns')}|{rec.get('exam_name')}".encode()).hexdigest()[:16]
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=f"{ROOT}/data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl")
    ap.add_argument("--schema", default=f"{ROOT}/analysis/eval_schema_auto_train.json")
    ap.add_argument("--config", default=f"{ROOT}/llm_api/gpt_56_sol.yaml")
    ap.add_argument("--model", default="gpt-5.4-mini", help="state-module + renderer model")
    ap.add_argument("--parser-model", default="gpt-5.4-mini")
    ap.add_argument("--tag", default="statepipe_v1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--api-retries", type=int, default=3)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--cache-dir", default=f"{ROOT}/analysis")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    cfgs = sm.load_model_config(args.config)
    for m in (args.model, args.parser_model):
        if m not in cfgs:
            raise SystemExit(f"model {m} not in {args.config}; available={sorted(cfgs)}")
    vocab = R.build_vocab_from_schema(args.schema)
    rows = [r for r in sm.load_rows(args.test, args.n) if vocab.get(r.get("ns"))]
    print(f"loaded scorable rows={len(rows)} model={args.model} parser={args.parser_model} tag={args.tag} workers={args.workers}")
    if not args.run:
        print("dry-run (pass --run to call the API)"); return

    cfg = dict(cfgs[args.model]); cfg["max_retry"] = args.api_retries
    parser_cfg = dict(cfgs[args.parser_model]); parser_cfg["max_retry"] = args.api_retries
    os.makedirs(args.cache_dir, exist_ok=True)
    state_cache_path = os.path.join(args.cache_dir, f"state_module_{args.model}_{args.tag}.jsonl")
    render_cache_path = os.path.join(args.cache_dir, f"rendered_report_{args.model}_{args.tag}.jsonl")
    parse_cache_path = os.path.join(args.cache_dir, f"direct_vp_report_parse_{args.parser_model}_{R.PARSE_PROMPT_VERSION}.jsonl")
    state_cache = R.load_cache(state_cache_path); render_cache = R.load_cache(render_cache_path)
    parse_cache = R.load_cache(parse_cache_path)
    s_lock = threading.Lock(); r_lock = threading.Lock(); p_lock = threading.Lock()

    def cached_call(messages, cache, cache_path, lock, key, mtok):
        if key in cache and not (args.retry_errors and cache[key].get("error")):
            return cache[key]
        try:
            t0 = time.time()
            text, usage, latency = sm.call_chat(messages, cfg, mtok, 0.1)
            o = {"key": key, "text": text, "usage": usage, "latency_sec": latency if latency is not None else time.time() - t0, "error": None}
        except Exception as e:
            o = {"key": key, "text": "", "usage": {}, "latency_sec": None, "error": str(e)}
        with lock:
            with open(cache_path, "a") as w:
                w.write(json.dumps(o, ensure_ascii=False) + "\n")
            cache[key] = o
        return o

    def one(rec):
        ns = rec.get("ns"); concepts = vocab.get(ns, [])
        gp = R.gold_pos(rec, vocab)
        # 1) state module
        so = cached_call(state_module_prompt(rec, concepts), state_cache, state_cache_path, s_lock, skey(args.model, rec, args.tag + "_state"), 640)
        pred_state = parse_state(so.get("text", ""), concepts) if not so.get("error") else {c: "not_mentioned" for c in concepts}
        pp_state = {c for c in concepts if pred_state.get(c) in POS}
        # 2) renderer
        ro = cached_call(render_prompt(rec, pred_state), render_cache, render_cache_path, r_lock, skey(args.model, rec, args.tag + "_render"), 900)
        report, _, report_json_ok = R.parse_report_wrapper(ro.get("text", "")) if not ro.get("error") else ("", None, False)
        # 3) reuse SAME extractor + scorer as end-to-end baseline
        rec2 = dict(rec); rec2["_gen_model"] = f"{args.tag}_{args.model}"
        attrs, parse_ok, _ = R.generated_attrs_from_report(report, rec2, args.parser_model, parser_cfg, parse_cache, parse_cache_path, p_lock, args.retry_errors) if report else ({}, False, {})
        pp_report = R.pred_pos_from_attrs(attrs, ns, vocab)
        return {"key": skey(args.model, rec, args.tag), "ns": ns, "gold": sorted(gp),
                "pred_state_pos": sorted(pp_state), "pred_report_pos": sorted(pp_report),
                "state_err": bool(so.get("error")), "render_err": bool(ro.get("error")),
                "report_json_ok": report_json_ok, "report_parse_ok": parse_ok,
                "gp": gp, "pp_state": pp_state, "pp_report": pp_report}

    t0 = time.time()
    results = []
    if args.workers <= 1:
        for r in rows:
            results.append(one(r))
    else:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(one, rows))

    def micro(pairs):
        tp = fp = fn = 0
        for gp, pp in pairs:
            tp += len(gp & pp); fp += len(pp - gp); fn += len(gp - pp)
        P = tp / (tp + fp) if tp + fp else 0.0; Rc = tp / (tp + fn) if tp + fn else 0.0
        return {"precision": round(P, 4), "recall": round(Rc, 4), "micro_f1": round(2 * P * Rc / (P + Rc) if P + Rc else 0.0, 4),
                "hallucination": round(1 - P, 4) if tp + fp else 0.0}
    report_metrics = micro([(r["gp"], r["pp_report"]) for r in results])
    state_metrics = micro([(r["gp"], r["pp_state"]) for r in results])
    state_errs = sum(r["state_err"] for r in results); render_errs = sum(r["render_err"] for r in results)
    summary = {"n": len(results), "model": args.model, "parser": args.parser_model, "tag": args.tag,
               "report_score_primary": report_metrics, "state_dict_direct": state_metrics,
               "state_module_errors": state_errs, "render_errors": render_errs,
               "report_json_ok": sum(r["report_json_ok"] for r in results), "report_parse_ok": sum(r["report_parse_ok"] for r in results),
               "wall_sec": round(time.time() - t0, 1)}
    out_summary = os.path.join(args.cache_dir, f"state_pipeline_{args.model}_{args.tag}_summary.json")
    out_records = os.path.join(args.cache_dir, f"state_pipeline_{args.model}_{args.tag}_records_n{len(results)}.jsonl")
    json.dump(summary, open(out_summary, "w"), ensure_ascii=False, indent=2)
    with open(out_records, "w") as w:
        for r in results:
            w.write(json.dumps({k: r[k] for k in ("key", "ns", "gold", "pred_state_pos", "pred_report_pos",
                     "state_err", "render_err", "report_json_ok", "report_parse_ok")}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary -> {out_summary}\nrecords -> {out_records}")


if __name__ == "__main__":
    main()
