"""Parametrized probC render runner: --model (renderer), --mode fullctx|stateonly, --probs (key2probs json),
--workers. Extraction stays frozen gpt-5.4-mini. Caches are keyed by renderer model + tag, so reruns resume."""
import argparse, json, os, sys, threading, time
import concurrent.futures as cf

sys.path.insert(0, "${VP_ROOT}"); sys.path.insert(0, "${VP_ROOT}/analysis")
os.chdir("${VP_ROOT}")
import run_direct_vp_report_smoke as R
import run_state_pipeline_smoke as SP
import run_temporal_llm_smoke_v2 as sm
import render_enriched as RE
from render_enriched import ns_human

def fullctx_prompt(rec, prior_state, probs, input_max_chars=0):
    context = rec.get("context", "") or ""
    if input_max_chars and len(context) > input_max_chars:
        context = context[:input_max_chars].rstrip() + "\n[context truncated by runner]"
    mod, region = ns_human(rec.get("ns", ""))
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

def stateonly_prompt(rec, probs):
    mod, region = ns_human(rec.get("ns", ""))
    ranked = sorted(((c, p) for c, p in probs.items() if p is not None), key=lambda x: -x[1])
    plines = "\n".join(f"- {c}: {p:.2f}" for c, p in ranked)
    skipped = [c for c, p in probs.items() if p is None]
    sys_msg = ("You are the report-writer stage of a clinical forecasting pipeline. From a state predictor's "
               "per-finding probabilities, generate the radiology report text that would plausibly be documented "
               "for the requested exam. Output ONLY JSON.")
    user = (
        "TASK:\nThe target radiology report is hidden. Using ONLY the predicted probabilities below, write the "
        "free-text radiology report that would plausibly be documented for the requested exam. This is a "
        "forecasting/simulation task.\n\n"
        "PREDICTED PROBABILITIES that each finding will be POSITIVE in the target exam (from a learned model "
        "conditioned on the patient's prior imaging state, the time gap, and interventions):\n"
        f"{plines}\n"
        + (f"(no reliable estimate for: {', '.join(skipped)})\n" if skipped else "")
        + "\nREAD THE NUMBERS: >=0.50 = strong evidence, state it as a finding; 0.30-0.50 = moderate, mention "
        "with hedged/soft wording or as possible; <0.30 = weak, usually OMIT or state as negative when normally "
        "documented. These are not equally reliable claims; do not treat them as a checklist.\n\n"
        "REPORT REQUIREMENTS:\n"
        "- Concise radiology-report style with EXAMINATION, FINDINGS, and IMPRESSION sections.\n"
        "- Include normal/negative findings when normally documented for this exam type.\n"
        "- No checklists or structured labels inside the report.\n\n"
        "REQUESTED TARGET EXAM:\n"
        f"Exam name: {rec.get('exam_name', rec.get('ns', ''))}\n"
        f"Modality: {mod}; body region: {region}\n"
        f"Exam namespace/type: {rec.get('ns', '')}\n\n"
        'Output JSON exactly as {"report":"<free-text radiology report>"}.'
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["fullctx", "stateonly"], required=True)
    ap.add_argument("--probs", default="analysis/key2probs_committed_mini.json")
    ap.add_argument("--config", default="llm_api/gpt_56_sol.yaml")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"probC_{args.mode}_{args.model.replace('.', '')}"

    vocab = R.build_vocab_from_schema("analysis/eval_schema_auto_train.json")
    key2probs = json.load(open(args.probs))
    src = ("data/direct_vp_full_ehr_raw_reports_test_stratified_1500.jsonl" if args.mode == "fullctx"
           else "data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl")
    rows = [json.loads(l) for l in open(src)]
    rows = [r for r in rows if vocab.get(r["ns"]) and "|".join([str(r["hadm_id"]), r["charttime"], r["ns"]]) in key2probs]
    if args.n > 0: rows = rows[: args.n]
    print(f"mode={args.mode} model={args.model} tag={tag} scorable={len(rows)} workers={args.workers}", flush=True)
    key2prior = RE.prior_states("analysis/agent_ml_data_episode_condtrans_union_stratval.pkl") if args.mode == "fullctx" else {}

    cfgs = sm.load_model_config(args.config)
    cfg = dict(cfgs[args.model]); cfg["max_retry"] = 5
    pcfg = dict(cfgs["gpt-5.4-mini"]); pcfg["max_retry"] = 4
    rcp = f"analysis/rendered_report_{args.model}_{tag}.jsonl"
    pcp = "analysis/direct_vp_report_parse_gpt-5.4-mini_%s.jsonl" % R.PARSE_PROMPT_VERSION
    rcache = R.load_cache(rcp); pcache = R.load_cache(pcp)
    rl = threading.Lock(); pl = threading.Lock()
    done = [0]; errs = [0]; t0 = time.time()

    def one(r):
        k = "|".join([str(r["hadm_id"]), r["charttime"], r["ns"]])
        k4 = tuple([str(r["hadm_id"]), r["charttime"], r["ns"], r["exam_name"]])
        rk = SP.skey(args.model, r, tag + "_render")
        if rk in rcache and not rcache[rk].get("error"):
            ro = rcache[rk]
        else:
            try:
                msgs = (fullctx_prompt(r, key2prior.get(k4, {}), key2probs[k]) if args.mode == "fullctx"
                        else stateonly_prompt(r, key2probs[k]))
                text, usage, lat = sm.call_chat(msgs, cfg, 1600, 0.1)
                ro = {"key": rk, "text": text, "usage": usage, "latency_sec": lat, "error": None}
            except Exception as e:
                ro = {"key": rk, "text": "", "usage": {}, "latency_sec": None, "error": str(e)}; errs[0] += 1
            with rl:
                open(rcp, "a").write(json.dumps(ro, ensure_ascii=False) + "\n"); rcache[rk] = ro
        report, _, _ = R.parse_report_wrapper(ro.get("text", "")) if not ro.get("error") else ("", None, False)
        r2 = dict(r); r2["_gen_model"] = f"{tag}_{args.model}"
        attrs, pok, _ = R.generated_attrs_from_report(report, r2, "gpt-5.4-mini", pcfg, pcache, pcp, pl, False) if report else ({}, False, {})
        done[0] += 1
        if done[0] % 100 == 0:
            print(f"[{done[0]}/{len(rows)}] errs={errs[0]} {time.time()-t0:.0f}s", flush=True)
        return R.gold_pos(r, vocab), R.pred_pos_from_attrs(attrs, r["ns"], vocab), bool(ro.get("error"))

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(one, rows))
    ok = [(g, p) for g, p, e in res if not e]
    tp = sum(len(g & p) for g, p in ok); fp = sum(len(p - g) for g, p in ok); fn = sum(len(g - p) for g, p in ok)
    fn += sum(len(g) for g, p, e in res if e)
    P = tp / max(tp + fp, 1); Rc = tp / max(tp + fn, 1)
    out = {"n": len(res), "gen_errors": errs[0], "model": args.model, "mode": args.mode,
           "precision": round(P, 4), "recall": round(Rc, 4),
           "micro_f1": round(2 * P * Rc / max(P + Rc, 1e-9), 4),
           "mini_refs": {"probC_fullctx": 0.5461, "stateonly_probC": 0.5236, "end2end_mini": 0.555,
                         "end2end_astra": 0.5614}}
    json.dump(out, open(f"analysis/fourstate_render_{tag}_summary.json", "w"), indent=1)
    print(json.dumps(out))

if __name__ == "__main__":
    main()
