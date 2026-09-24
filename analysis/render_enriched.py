"""
Enriched render (path B, render-side lever): the renderer gets the SAME raw patient context as the
end-to-end baseline (same 4000-char truncation -> apples-to-apples vs end2end mini 0.555), PLUS two
structured supplements from our pipeline, each with explicit semantics:

  1) CURRENT STATE (predictor INPUT, "原始态"): four-state findings extracted from the patient's prior
     imaging reports before the target time (any modality), decoded from the sealed Xte state block.
  2) PREDICTED STATE (predictor OUTPUT, "预测态"): the committed auto-research config's forecast of the
     target exam's findings at target time -- the model additionally saw the time gap and interventions
     (meds/procedures/ED), i.e. temporal-evolution signal the renderer cannot read from raw text alone.

Target exam modality/body-region, both states' meanings, and the forecasting task are all stated explicitly
in the prompt (experiment spec 2026-09-24). Scored with the SAME fixed extractor; compares against
end2end mini 0.555, state-only render 0.5041, LLM-module 0.473.
"""
import argparse, concurrent.futures as cf, json, os, pickle, sys, threading, time
import numpy as np

ROOT = "${VP_ROOT}"
sys.path.insert(0, f"{ROOT}/analysis")
import run_direct_vp_report_smoke as R
import run_state_pipeline_smoke as SP
import run_temporal_llm_smoke_v2 as sm
import cond_transition as CT
from render_committed_config import predict_sealed_from_decision

MODALITY = {"xray": "X-ray (radiograph)", "ct": "CT", "mr": "MRI", "us": "ultrasound"}


def ns_human(ns):
    """'rad.xray_chest' -> ('X-ray (radiograph)', 'chest')"""
    tail = ns.split(".", 1)[-1]
    mod, _, region = tail.partition("_")
    return MODALITY.get(mod, mod), (region or "unspecified").replace("_", " ")


def prior_states(pkl):
    """key -> {concept: state} decoded from the sealed te INPUT state block (only concepts that were
    mentioned in prior reports; everything else is not_mentioned)."""
    d = pickle.load(open(pkl, "rb"))
    uc = d["meta"]["union_concepts"]; sb = d["meta"]["state_block"]
    key2prior = {}
    for ns in d["top_ns"]:
        D = d["data"][ns]; keys = d["te_keys"][ns]
        if not len(D["Xte"]):
            continue
        lab = CT.state_to_labels(D["Xte"], sb["n_concepts"])
        for i, key in enumerate(keys):
            key2prior[tuple(str(x) for x in key)] = {uc[j]: CT.FOURSTATE[lab[i, j]]
                                                     for j in range(len(uc)) if lab[i, j] != 3}
    return key2prior


def enriched_prompt(rec, prior_state, pred_pos, vocab_ns, input_max_chars=4000, style="anchored"):
    context = rec.get("context", "") or ""
    if input_max_chars and input_max_chars > 0 and len(context) > input_max_chars:
        context = context[:input_max_chars].rstrip() + "\n[context truncated by runner]"
    mod, region = ns_human(rec.get("ns", ""))
    prior_lines = "\n".join(f"- {c}: {s}" for c, s in sorted(prior_state.items())) or "- (none documented)"
    pos = sorted(pred_pos); neg = sorted(set(vocab_ns) - pred_pos)
    if style == "anchored":
        fusion = ("Use all three sources. When they conflict, reason clinically: (C) is a statistically calibrated "
                  "forecast of WHICH findings will be positive; (A) carries the descriptive detail and nuance for "
                  "HOW to word them.\n\n")
    elif style == "radiologist":
        # advisory framing + explicit mention-selection judgment (targets omission-type FPs: prior-positive
        # findings carried forward that the actual radiologist would not restate in THIS report)
        fusion = ("(B) and (C) are ADVISORY state evidence from an upstream pipeline, not ground truth. "
                  "You are the REPORTING RADIOLOGIST for this exam. Reporting is selective: a real report does "
                  "not restate everything that is true of the patient. After deciding what is clinically present "
                  "(from (A)-(C)), apply reporting judgment to decide WHAT BELONGS IN THIS REPORT:\n"
                  "- MENTION: findings that are new or changed; findings relevant to the exam's indication/clinical "
                  "question; pertinent negatives a radiologist would document for this indication.\n"
                  "- OMIT: stable chronic findings unrelated to the indication that a radiologist would not restate "
                  "in this exam's report; incidental prior findings outside this exam's scope; anything you would "
                  "only be repeating from old reports without current relevance.\n"
                  "When unsure whether a stable prior finding would be restated, prefer OMITTING it unless it is "
                  "directly relevant to the indication.\n\n")
    else:  # advisory: (A) is primary, (B)/(C) are supplements the renderer may override
        fusion = ("(B) and (C) are ADVISORY supplements from an upstream pipeline, not ground truth. Write the "
                  "report primarily from your own clinical reasoning over (A). Consult (C) as a second opinion "
                  "about temporal evolution -- it conditioned on the time gap and interventions, which are hard "
                  "to weigh from raw text -- especially where (A) leaves a finding ambiguous. Override (B)/(C) "
                  "whenever (A) clearly indicates otherwise.\n\n")
    sys_msg = (
        "You are acting as a virtual patient simulator for a clinical forecasting benchmark. "
        "Given observations available before a requested radiology exam, plus two structured state summaries "
        "from an upstream prediction pipeline, generate the radiology report text that would plausibly be "
        "documented for that exam. Output ONLY JSON."
    )
    user = (
        "TASK:\n"
        "The target radiology report is hidden. Generate the free-text radiology report that would plausibly be "
        "documented for the requested target exam, using only patient information available before that exam. "
        "This is a forecasting/simulation task, not extraction from the hidden report.\n\n"
        "You are given THREE information sources:\n"
        "(A) RAW PATIENT CONTEXT: prior admissions, prior radiology reports (verbatim), diagnoses, medications, "
        "procedures and ED events documented before the target exam.\n"
        "(B) CURRENT STRUCTURED STATE: finding concepts extracted from this patient's PRIOR imaging reports "
        "(any modality/body region), each labeled present/absent/uncertain as of the most recent prior exams. "
        "Concepts not listed were not mentioned in any prior report. This summarizes what is already known -- "
        "it contains NO information about the future.\n"
        "(C) PREDICTED STATE AT TARGET TIME: a learned conditional transition model's forecast of the target "
        "exam's findings. Unlike you, that model explicitly conditioned on the TIME GAP between the prior exams "
        "and the target time, and on the interventions (medications/procedures/ED events) in between -- so it "
        "carries temporal-evolution signal that is hard to read from raw text. It only covers the listed schema "
        "concepts for the target exam type.\n"
        + fusion +
        "REPORT REQUIREMENTS:\n"
        "- Write in concise radiology-report style, preferably with EXAMINATION, FINDINGS, and IMPRESSION sections.\n"
        "- Include findings the target exam would likely state, including normal/negative findings when normally "
        "documented.\n"
        "- Do not output a candidate finding checklist or structured labels inside the report.\n"
        "- If prior radiology reports are provided, use them as prior evidence only; do not simply copy them.\n\n"
        "=== (A) RAW PATIENT CONTEXT (before the target exam) ===\n"
        f"{context}\n\n"
        "=== (B) CURRENT STRUCTURED STATE (from prior reports; before target time; any modality) ===\n"
        f"{prior_lines}\n\n"
        f"=== (C) PREDICTED STATE AT TARGET TIME (schema concepts for exam type {rec.get('ns','')}) ===\n"
        f"Predicted POSITIVE at target time: {pos if pos else '(none)'}\n"
        f"Predicted NOT positive at target time: {neg if neg else '(none)'}\n\n"
        "=== REQUESTED TARGET EXAM ===\n"
        f"Exam name: {rec.get('exam_name', rec.get('ns', ''))}\n"
        f"Modality: {mod}; body region: {region}\n"
        f"Exam namespace/type: {rec.get('ns', '')}\n"
        f"Target time: {rec.get('charttime', '')}\n\n"
        'Output JSON exactly as {"report":"<free-text radiology report>"}.'
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=f"{ROOT}/analysis/agent_ml_data_episode_condtrans_union_stratval.pkl")
    ap.add_argument("--decision", default=f"{ROOT}/analysis/agentB_decision_condtrans_mini.json")
    ap.add_argument("--test", default=f"{ROOT}/data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl")
    ap.add_argument("--schema", default=f"{ROOT}/analysis/eval_schema_auto_train.json")
    ap.add_argument("--config", default=f"{ROOT}/llm_api/gpt_56_sol.yaml")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--parser-model", default="gpt-5.4-mini")
    ap.add_argument("--tag", default="enriched_condtrans_mini")
    ap.add_argument("--style", default="anchored", choices=["anchored", "advisory", "radiologist"])
    ap.add_argument("--input-max-chars", type=int, default=4000,
                    help="context truncation for the render prompt; 0 disables (full-EHR parity with end-to-end)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--api-retries", type=int, default=3)
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args()

    cfgs = sm.load_model_config(args.config)
    vocab = R.build_vocab_from_schema(args.schema)
    key2pred = predict_sealed_from_decision(args.pkl, args.decision)
    key2prior = prior_states(args.pkl)
    te_rows = [json.loads(l) for l in open(args.test) if l.strip()]
    te_rows = [r for r in te_rows if vocab.get(r.get("ns")) and tuple(str(x) for x in
               [r.get("hadm_id"), r.get("charttime"), r.get("ns"), r.get("exam_name")]) in key2pred]
    te_rows = te_rows[: args.n] if args.n > 0 else te_rows
    print(f"scorable+matched rows={len(te_rows)}")

    cfg = dict(cfgs[args.model]); cfg["max_retry"] = args.api_retries
    parser_cfg = dict(cfgs[args.parser_model]); parser_cfg["max_retry"] = args.api_retries
    render_cache_path = f"{ROOT}/analysis/rendered_report_{args.model}_{args.tag}.jsonl"
    parse_cache_path = f"{ROOT}/analysis/direct_vp_report_parse_{args.parser_model}_{R.PARSE_PROMPT_VERSION}.jsonl"
    render_cache = R.load_cache(render_cache_path); parse_cache = R.load_cache(parse_cache_path)
    r_lock = threading.Lock(); p_lock = threading.Lock()

    def one(r):
        ns = r["ns"]; key = tuple(str(x) for x in [r.get("hadm_id"), r.get("charttime"), ns, r.get("exam_name")])
        pp_state = key2pred[key]
        gp = R.gold_pos(r, vocab)
        rk = SP.skey(args.model, r, args.tag + "_render")
        if rk in render_cache and not (args.retry_errors and render_cache[rk].get("error")):
            ro = render_cache[rk]
        else:
            try:
                t0 = time.time()
                text, usage, lat = sm.call_chat(
                    enriched_prompt(r, key2prior.get(key, {}), pp_state, vocab[ns],
                                    input_max_chars=args.input_max_chars, style=args.style), cfg, 1200, 0.1)
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
               "baselines": {"end2end_mini_report": 0.555, "state_only_render": 0.5041, "llm_module_report": 0.473}}
    out = f"{ROOT}/analysis/fourstate_render_{args.tag}_summary.json"
    json.dump(summary, open(out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2)); print(f"-> {out}")


if __name__ == "__main__":
    main()
