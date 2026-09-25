"""Paired case-level bootstrap significance for headline micro-F1 comparisons (cmt_v1 #8).
All arms scored on the same 1,419 sealed scorable queries; per-case (gold_pos, pred_pos) loaded from
saved records; each arm's aggregate is verified against the published number before testing.
Also reconstructs the omission-tolerant reading per case (forgive FP iff prior state 'present' and the
target gold extraction does not mark the concept absent) and verifies it against published aggregates.
Paired bootstrap: resample cases with replacement (B=10000), micro-F1 difference, 95% percentile CI,
two-sided p = 2*min(P(diff<=0), P(diff>=0)).
"""
import json, pickle, sys
import numpy as np

sys.path.insert(0, "/workspace/siyan/virtualPatient"); sys.path.insert(0, "/workspace/siyan/virtualPatient/analysis")
import os; os.chdir("/workspace/siyan/virtualPatient")
import run_direct_vp_report_smoke as R
import render_enriched as RE

vocab = R.build_vocab_from_schema("analysis/eval_schema_auto_train.json")
rows = [json.loads(l) for l in open("data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl")]
rows = [r for r in rows if vocab.get(r["ns"])]
KEYS = ["|".join([str(r["hadm_id"]), r["charttime"], r["ns"]]) for r in rows]
key2row = dict(zip(KEYS, rows))
assert len(KEYS) == 1419, len(KEYS)

def load_e2e(path):
    out = {}
    for l in open(path):
        o = json.loads(l)
        k = "|".join([str(o["hadm_id"]), o["charttime"], o["ns"]])
        if k in key2row:
            out[k] = (set(o["gold_pos"]), set(o["pred_pos"]))
    return out

def load_pathB(path):
    out = {}
    for l in open(path):
        o = json.loads(l)
        k = "|".join([str(o["hadm_id"]), o["charttime"], o["ns"]])
        if k in key2row:
            out[k] = (set(o["gp"]), set(o["pp"]))
    return out

def load_klist(path):
    out = {}
    for o in json.load(open(path)):
        if o["k"] in key2row:
            out[o["k"]] = (set(o["gp"]), set(o["pp"]))
    return out

A = "analysis/"
ARMS = {
    "e2e_mini": load_e2e(A + "direct_vp_report_gpt-5.4-mini_direct_vp_report_v3_full_ehr_ed_proc_raw_reports_json_confidence_records_n1500.jsonl"),
    "e2e_astra": load_e2e(A + "direct_vp_report_gpt-6-astra_direct_vp_report_v3_full_ehr_ed_proc_raw_reports_json_confidence_records_n1500.jsonl"),
    "e2e_opus": load_e2e(A + "direct_vp_report_claude-opus-4-7_direct_vp_report_v3_full_ehr_ed_proc_raw_reports_json_confidence_records_n1500.jsonl"),
    "e2e_glm": load_e2e(A + "direct_vp_report_glm-5.3-flash_direct_vp_report_v3_full_ehr_ed_proc_raw_reports_json_confidence_records_n1500.jsonl"),
    "e2e_4k": load_e2e(A + "direct_vp_report_gpt-5.4-mini_direct_vp_report_v3compressed4k_json_confidence_records_n1500.jsonl"),
    "e2e_ablate": load_e2e(A + "direct_vp_report_gpt-5.4-mini_direct_vp_report_v3ablate_noicd_cc_triage_dcsum_json_confidence_records_n1500.jsonl"),
    "advisory4k": load_pathB(A + "pathB_records_enriched_advisory_condtrans_mini.jsonl"),
    "stateonly_render": load_pathB(A + "pathB_records_condtrans_autoresearch_mini.jsonl"),
    "probC_fullctx": load_klist(A + "probC_percase_records.json"),
    "copy_prior": load_klist(A + "copy_baseline_percase_records.json"),
}

def micro(arm, keys=KEYS):
    tp = fp = fn = 0
    for k in keys:
        gp, pp = arm.get(k, (set(key2row[k] and R.gold_pos(key2row[k], vocab)), set()))
        tp += len(gp & pp); fp += len(pp - gp); fn += len(gp - pp)
    P = tp / max(tp + fp, 1); Rc = tp / max(tp + fn, 1)
    return 2 * P * Rc / max(P + Rc, 1e-9)

print("== aggregate verification (must match paper) ==")
PUB = {"e2e_mini": 0.555, "e2e_astra": 0.5614, "e2e_opus": 0.5375, "e2e_glm": 0.5363, "e2e_4k": 0.503,
       "e2e_ablate": 0.546, "advisory4k": 0.523, "stateonly_render": 0.504, "probC_fullctx": 0.5461,
       "copy_prior": 0.4947}
for name, arm in ARMS.items():
    m = micro(arm)
    flag = "OK" if abs(m - PUB[name]) < 0.006 else "MISMATCH"
    print(f"{name:18s} n={len(arm):4d} micro={m:.4f} pub={PUB[name]} {flag}")

# ---- omission-tolerant reconstruction (original rule: prior present/uncertain via findings index,
# target row's embedded gold four-state != absent; matches the 2026-09-24 computation) ----
import score_ml_fourstate as F4
IDX4 = F4.load_findings_index()
key2prior = {}
key2goldstate = {}
for r in rows:
    k = "|".join([str(r["hadm_id"]), r["charttime"], r["ns"]])
    key2prior[k] = F4.prior_state(r, IDX4)[0] or {}
    key2goldstate[k] = {kk.split(".", 2)[-1]: v for kk, v in (r.get("gold") or {}).items()}

def omtol_counts(arm, k):
    gp, pp = arm.get(k, (R.gold_pos(key2row[k], vocab), set()))
    prior = key2prior.get(k, {})
    gold_state = key2goldstate.get(k, {})
    fp_set = pp - gp
    forgiven = {c for c in fp_set if prior.get(c) in ("present", "uncertain") and gold_state.get(c) != "absent"}
    return len(gp & pp), len(fp_set - forgiven), len(gp - pp)

def micro_omtol(arm):
    tp = fp = fn = 0
    for k in KEYS:
        a, b, c = omtol_counts(arm, k)
        tp += a; fp += b; fn += c
    P = tp / max(tp + fp, 1); Rc = tp / max(tp + fn, 1)
    return 2 * P * Rc / max(P + Rc, 1e-9)

print("\n== omission-tolerant verification ==")
# NOTE: the previously-published mini omtol 0.628 was computed on the WRONG records file
# (v1-prompt run picked up by glob); the correct v3-records value is 0.6657. Verified 2026-09-25.
PUB_OT = {"e2e_astra": 0.668, "e2e_opus": 0.651, "e2e_glm": 0.6483, "e2e_mini": 0.6657, "advisory4k": 0.638}
for name in PUB_OT:
    m = micro_omtol(ARMS[name])
    flag = "OK" if abs(m - PUB_OT[name]) < 0.006 else "MISMATCH"
    print(f"{name:18s} omtol={m:.4f} pub={PUB_OT[name]} {flag}")

# ---- paired bootstrap ----
def cellmat(arm, omtol=False):
    M = np.zeros((len(KEYS), 3), dtype=np.int32)
    for i, k in enumerate(KEYS):
        if omtol:
            M[i] = omtol_counts(arm, k)
        else:
            gp, pp = arm.get(k, (R.gold_pos(key2row[k], vocab), set()))
            M[i] = (len(gp & pp), len(pp - gp), len(gp - pp))
    return M

BOOT = 10000
rng = np.random.default_rng(0)
IDX = rng.integers(0, len(KEYS), size=(BOOT, len(KEYS)))

def boot_diff(mA, mB):
    def f1(M):
        s = M[IDX].sum(axis=1)  # (B,3)
        tp, fp, fn = s[:, 0].astype(float), s[:, 1], s[:, 2]
        P = tp / np.maximum(tp + fp, 1); Rc = tp / np.maximum(tp + fn, 1)
        return 2 * P * Rc / np.maximum(P + Rc, 1e-9)
    d = f1(mA) - f1(mB)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(max(p, 1.0 / BOOT))

PAIRS = [
    ("advisory4k", "e2e_4k", "mention"),
    ("probC_fullctx", "e2e_mini", "mention"),
    ("e2e_mini", "e2e_ablate", "mention"),
    ("e2e_astra", "e2e_mini", "mention"),
    ("stateonly_render", "copy_prior", "mention"),
    ("e2e_mini", "copy_prior", "mention"),
    ("e2e_opus", "e2e_mini", "omtol"),
    ("e2e_glm", "e2e_mini", "omtol"),
    ("e2e_astra", "e2e_mini", "omtol"),
    ("e2e_opus", "e2e_mini", "mention"),
    ("e2e_astra", "probC_fullctx", "mention"),
    ("e2e_astra", "probC_fullctx", "omtol"),
    ("probC_fullctx", "e2e_mini", "omtol"),
    ("stateonly_render", "copy_prior", "omtol"),
]
print("\n== paired bootstrap (B=10000) ==")
results = []
CACHE = {}
for a, b, metric in PAIRS:
    ka, kb = (a, metric), (b, metric)
    if ka not in CACHE: CACHE[ka] = cellmat(ARMS[a], metric == "omtol")
    if kb not in CACHE: CACHE[kb] = cellmat(ARMS[b], metric == "omtol")
    d, lo, hi, p = boot_diff(CACHE[ka], CACHE[kb])
    sig = "SIG" if (lo > 0 or hi < 0) else "ns"
    print(f"{a} - {b} [{metric}]: diff={d:+.4f} CI=[{lo:+.4f},{hi:+.4f}] p={p:.4f} {sig}")
    results.append({"a": a, "b": b, "metric": metric, "diff": round(d, 4),
                    "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "p": round(p, 4), "sig": sig == "SIG"})
json.dump(results, open("analysis/paired_bootstrap_results.json", "w"), indent=1)
print("\nsaved analysis/paired_bootstrap_results.json")
