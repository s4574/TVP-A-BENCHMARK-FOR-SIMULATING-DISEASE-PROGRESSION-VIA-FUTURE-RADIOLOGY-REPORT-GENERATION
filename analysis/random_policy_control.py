"""Random-policy control for the auto-research protocol (reviewer #7 / cmt_v1):
a uniform-random policy over the SAME typed action space, same budgets (24 iters / 30 compute),
same env + pkl as the committed runs. No LLM. Usage: python3 random_policy_control.py <seed>
Then score the committed config with render_committed_config.predict_sealed_from_decision (state-direct).
"""
import json, os, random, sys, types

SEED = int(sys.argv[1])
PKL = "/workspace/siyan/virtualPatient/analysis/agent_ml_data_episode_condtrans_union_stratval.pkl"
DEC = f"/workspace/siyan/virtualPatient/analysis/agentB_decision_RANDOMPOLICY_seed{SEED}.json"
os.environ["VP_AGENT_ML_DATA"] = PKL
os.environ["VP_AGENT_DECISION"] = DEC

sys.path.insert(0, "/workspace/siyan/virtualPatient")
sys.path.insert(0, "/workspace/siyan/virtualPatient/analysis")
from cond_transition import MENUS

rng = random.Random(SEED)

def random_action():
    r = rng.random()
    if r < 0.15:
        return rng.choice([
            {"action": "set_threshold_mode", "args": {"mode": rng.choice(["fixed", "calibrated"])}},
            {"action": "set_recipe", "args": {"recipe": rng.choice(["default", "balanced", "oversample"])}},
            {"action": "set_min_pos", "args": {"n": rng.choice([3, 5, 8])}},
        ])
    if r < 0.55:
        return {"action": "train", "args": {"method": rng.choice(["logreg", "rf", "xgb", "mlp", "base-rate"])}}
    if r < 0.75:
        ms = rng.sample(["logreg", "rf", "xgb", "mlp"], k=rng.choice([2, 3]))
        return {"action": "train_ensemble", "args": {"methods": ms}}
    cfg = {k: rng.choice(v) for k, v in MENUS.items()}
    return {"action": "train_transition", "args": {"config": cfg}}

def chat_full(messages, model="m", temperature=0.0, max_tokens=0, response_json=False):
    a = random_action()
    a["thinking"] = "uniform-random policy"
    a["why"] = "random"
    return json.dumps(a), {"prompt_tokens": 1, "completion_tokens": 1}

mock = types.ModuleType("llm_client")
mock.chat_full = chat_full
sys.modules["llm_client"] = mock
sys.argv = ["agentB_train", "24", f"random-policy-seed{SEED}"]

import playground.agentB_train as A
try:
    A.main()
except SystemExit:
    pass

dec = json.load(open(DEC))
print(f"seed={SEED} best_config:", json.dumps(dec["best_config"], ensure_ascii=False)[:400])

from render_committed_config import predict_sealed_from_decision
import run_direct_vp_report_smoke as R
vocab = R.build_vocab_from_schema("/workspace/siyan/virtualPatient/analysis/eval_schema_auto_train.json")
key2pred = predict_sealed_from_decision(PKL, DEC)
rows = [json.loads(l) for l in open("/workspace/siyan/virtualPatient/data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl")]
rows = [r for r in rows if vocab.get(r["ns"])]
tp = fp = fn = 0
for r in rows:
    k = tuple([str(r["hadm_id"]), r["charttime"], r["ns"], r["exam_name"]])
    k3 = tuple([str(r["hadm_id"]), r["charttime"], r["ns"]])
    pred = key2pred.get(k, key2pred.get(k3, set()))
    gp = R.gold_pos(r, vocab)
    tp += len(gp & pred); fp += len(pred - gp); fn += len(gp - pred)
P = tp / max(tp + fp, 1); Rc = tp / max(tp + fn, 1)
out = {"seed": SEED, "sealed_micro_f1_1419": round(2 * P * Rc / max(P + Rc, 1e-9), 4),
       "precision": round(P, 4), "recall": round(Rc, 4),
       "internal_val_best": dec["best_config"].get("internal_val") or dec["best_config"].get("micro"),
       "termination": dec.get("termination_reason")}
print("RANDOM POLICY SEALED:", json.dumps(out))
json.dump(out, open(f"/workspace/siyan/virtualPatient/analysis/random_policy_sealed_seed{SEED}.json", "w"), indent=1)
