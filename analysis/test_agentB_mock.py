"""Offline mock-policy test of the agentB_train driver loop (no API): exercises train_transition dispatch,
flat-args form, off-menu rejection, config inheritance, set_state_concepts (valid + unknown-name rejection),
balanced recipe on the transition arm, and best_config recording incl. transition_config/state_concepts."""
import json, os, pickle, sys, types

PKL = "${VP_ROOT}/analysis/agent_ml_data_episode_condtrans_union_stratval.pkl"
os.environ["VP_AGENT_ML_DATA"] = PKL
os.environ["VP_AGENT_DECISION"] = "${VP_ROOT}/analysis/agentB_decision_MOCKTEST.json"
uc = pickle.load(open(PKL, "rb"))["meta"]["union_concepts"]

ACTIONS = [
    {"thinking": "t", "action": "set_threshold_mode", "args": {"mode": "calibrated"}, "why": "w"},
    {"thinking": "t", "action": "set_state_concepts", "args": {"names": ["NOT_A_CONCEPT"]}, "why": "unknown name"},
    {"thinking": "t", "action": "set_state_concepts", "args": {"names": uc[:10]}, "why": "w"},
    {"thinking": "t", "action": "train", "args": {"method": "logreg"}, "why": "w"},
    {"thinking": "t", "action": "set_recipe", "args": {"recipe": "balanced"}, "why": "w"},
    {"thinking": "t", "action": "train_transition", "args": {"config": {"cond_mode": "concat", "selfsup_ratio": 0.5}}, "why": "balanced CE"},
    {"thinking": "t", "action": "train_transition", "args": {"cond_mode": "film"}, "why": "flat args; inherits selfsup 0.5"},
    {"thinking": "t", "action": "train_transition", "args": {"config": {"hidden": 999}}, "why": "off-menu"},
    {"thinking": "t", "action": "set_state_concepts", "args": {"names": uc[:10], "mode": "state"}, "why": "tied state space"},
    {"thinking": "t", "action": "train", "args": {"method": "logreg"}, "why": "state-mode sklearn"},
    {"thinking": "t", "action": "train_transition", "args": {"config": {"cond_mode": "concat"}}, "why": "state-mode transition"},
    {"thinking": "t", "action": "train_transition", "args": {"config": {"learner": "xgb", "selfsup_ratio": 0.25}}, "why": "pooled full-state sklearn learner"},
    {"thinking": "t", "action": "stop", "args": {}, "why": "done"},
]
it = iter(ACTIONS)

mock = types.ModuleType("llm_client")
mock.chat_full = lambda messages, model="m", temperature=0.0, max_tokens=0, response_json=False: (json.dumps(next(it)), {"prompt_tokens": 1, "completion_tokens": 1})
sys.modules["llm_client"] = mock
sys.path.insert(0, "${VP_ROOT}")
sys.path.insert(0, "${VP_ROOT}/analysis")
sys.argv = ["agentB_train", "14", "mock-policy"]

import playground.agentB_train as A
try:
    A.main()
except SystemExit:
    pass

dec = json.load(open(os.environ["VP_AGENT_DECISION"]))
best = dec["best_config"]; tr = dec["trace"]
print("\nbest_config:", json.dumps(best, ensure_ascii=False))
assert dec["termination_reason"] == "model_stop"
assert len(tr) == 13, len(tr)
assert "error" in tr[1]["obs"], tr[1]["obs"]                       # unknown concept rejected
assert tr[2]["obs"].get("n_selected") == 10, tr[2]["obs"]          # valid concept subset accepted (evidence mode)
assert tr[5]["obs"].get("recipe") == "balanced", tr[5]["obs"]      # balanced reaches transition arm
cfg7 = tr[6]["obs"].get("config")                                  # flat args + inheritance
assert cfg7 and cfg7["cond_mode"] == "film" and cfg7["selfsup_ratio"] == 0.5, cfg7
assert "error" in tr[7]["obs"], tr[7]["obs"]                       # off-menu rejected
assert tr[8]["obs"].get("mode") == "state", tr[8]["obs"]           # tied state-space mode accepted
ev9, ev10 = tr[9]["obs"].get("internal_val"), tr[10]["obs"].get("internal_val")
assert ev9 and ev10, (ev9, ev10)                                   # both arms train under state mode
cfg11 = tr[11]["obs"].get("config")                                # pooled sklearn learner in transition pathway
assert cfg11 and cfg11["learner"] == "xgb" and tr[11]["obs"].get("internal_val"), tr[11]["obs"]
assert best["state_concepts"] == uc[:10], best["state_concepts"]
assert best["state_concepts_mode"] in ("evidence", "state"), best
print("MOCK DRIVER TEST PASSED")
