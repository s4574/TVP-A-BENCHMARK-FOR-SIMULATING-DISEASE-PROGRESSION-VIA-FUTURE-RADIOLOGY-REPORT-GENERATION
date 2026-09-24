"""GT-fix gate (2026-09-24): does denoised/densified TRAIN supervision move sealed state-direct?
Grid: pkl {orig (control), maskA (silence-masked), denseB (3d union + silence-masked)}
    x arm {transition-mlp(concat,ss.5), pooled-xgb, pooled-logreg, legacy per-ns xgb}.
All calibrated thresholds on internal val (mention-aligned); sealed te scored once per cell.
Frontier to beat: 0.499-0.502."""
import importlib, json, os, sys

ROOT = "${VP_ROOT}"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/analysis")
PKLS = [
    ("orig", f"{ROOT}/analysis/agent_ml_data_episode_condtrans_union_stratval.pkl"),
    ("maskA", f"{ROOT}/analysis/agent_ml_data_episode_condtrans_maskA_stratval.pkl"),
    ("denseB", f"{ROOT}/analysis/agent_ml_data_episode_condtrans_denseB_stratval.pkl"),
]
ARMS = [
    ("trans_mlp", {"cond_mode": "concat", "selfsup_ratio": 0.5}),
    ("pooled_xgb", {"learner": "xgb", "selfsup_ratio": 0.25}),
    ("pooled_logreg", {"learner": "logreg", "selfsup_ratio": 0.25}),
]
out = {}
for pname, pkl in PKLS:
    os.environ["VP_AGENT_ML_DATA"] = pkl
    import playground.env_temporal as ET
    importlib.reload(ET)
    env = ET.TemporalEnv(compute_units=10**6, eval_queries=10**6)
    env.set_threshold_mode("calibrated")
    for aname, cfg in ARMS:
        r = env.train_transition(dict(cfg))
        assert r.get("ok"), (pname, aname, r)
        te = env._score_transition("te")
        out[f"{pname}|{aname}"] = {"val": r["internal_val"], "te": te}
        print(pname, aname, "val", r["internal_val"], "te", te, flush=True)
    r = env.train("xgb")
    te = env._score("xgb", "te")
    out[f"{pname}|legacy_ns_xgb"] = {"val": r["internal_val"], "te": te}
    print(pname, "legacy_ns_xgb val", r["internal_val"], "te", te, flush=True)
json.dump(out, open(f"{ROOT}/analysis/gt_fix_gate.json", "w"), indent=2)
print("-> analysis/gt_fix_gate.json")
