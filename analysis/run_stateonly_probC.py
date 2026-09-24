"""State-only + probability render: the renderer sees ONLY (i) the target exam identity and (ii) the state
predictor's per-concept probabilities with reading guidance -- no raw context, no prior-state list.
Isolates whether probability annotation helps absent any other evidence (vs flat state-only 0.5041)."""
import json, sys, threading, time
import concurrent.futures as cf

sys.path.insert(0, "${VP_ROOT}"); sys.path.insert(0, "${VP_ROOT}/analysis")
import os
os.chdir("${VP_ROOT}")
import run_direct_vp_report_smoke as R
import run_state_pipeline_smoke as SP
import run_temporal_llm_smoke_v2 as sm
from render_enriched import ns_human

vocab = R.build_vocab_from_schema("analysis/eval_schema_auto_train.json")
key2probs = json.load(open("analysis/key2probs_committed_mini.json"))
rows = [json.loads(l) for l in open("data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl")]
rows = [r for r in rows if vocab.get(r["ns"]) and "|".join([str(r["hadm_id"]), r["charttime"], r["ns"]]) in key2probs]
print("scorable:", len(rows), flush=True)

def prompt(rec, probs):
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

cfgs = sm.load_model_config("llm_api/gpt_56_sol.yaml")
cfg = dict(cfgs["gpt-5.4-mini"]); cfg["max_retry"] = 4
tag = "stateonly_probC_mini"
rcp = f"analysis/rendered_report_gpt-5.4-mini_{tag}.jsonl"
pcp = "analysis/direct_vp_report_parse_gpt-5.4-mini_%s.jsonl" % R.PARSE_PROMPT_VERSION
rcache = R.load_cache(rcp); pcache = R.load_cache(pcp)
rl = threading.Lock(); pl = threading.Lock()
done = [0]; errs = [0]; t0 = time.time()

def one(r):
    k = "|".join([str(r["hadm_id"]), r["charttime"], r["ns"]])
    rk = SP.skey("gpt-5.4-mini", r, tag + "_render")
    if rk in rcache and not rcache[rk].get("error"):
        ro = rcache[rk]
    else:
        try:
            text, usage, lat = sm.call_chat(prompt(r, key2probs[k]), cfg, 1200, 0.1)
            ro = {"key": rk, "text": text, "usage": usage, "latency_sec": lat, "error": None}
        except Exception as e:
            ro = {"key": rk, "text": "", "usage": {}, "latency_sec": None, "error": str(e)}; errs[0] += 1
        with rl:
            open(rcp, "a").write(json.dumps(ro, ensure_ascii=False) + "\n"); rcache[rk] = ro
    report, _, _ = R.parse_report_wrapper(ro.get("text", "")) if not ro.get("error") else ("", None, False)
    r2 = dict(r); r2["_gen_model"] = f"{tag}_gpt-5.4-mini"
    attrs, pok, _ = R.generated_attrs_from_report(report, r2, "gpt-5.4-mini", cfg, pcache, pcp, pl, False) if report else ({}, False, {})
    done[0] += 1
    if done[0] % 200 == 0:
        print(f"[{done[0]}/{len(rows)}] errs={errs[0]} {time.time()-t0:.0f}s", flush=True)
    return R.gold_pos(r, vocab), R.pred_pos_from_attrs(attrs, r["ns"], vocab), bool(ro.get("error"))

with cf.ThreadPoolExecutor(max_workers=12) as ex:
    res = list(ex.map(one, rows))
ok = [(g, p) for g, p, e in res if not e]
tp = sum(len(g & p) for g, p in ok); fp = sum(len(p - g) for g, p in ok); fn = sum(len(g - p) for g, p in ok)
fn += sum(len(g) for g, p, e in res if e)
P = tp / (tp + fp); Rc = tp / (tp + fn)
out = {"n": len(res), "gen_errors": errs[0], "precision": round(P, 4), "recall": round(Rc, 4),
       "micro_f1": round(2 * P * Rc / (P + Rc), 4),
       "baselines": {"stateonly_flat": 0.5041, "end2end_mini_4k": 0.5031, "probC_fullctx": 0.5461}}
json.dump(out, open("analysis/fourstate_render_stateonly_probC_summary.json", "w"), indent=1)
print(json.dumps(out))
