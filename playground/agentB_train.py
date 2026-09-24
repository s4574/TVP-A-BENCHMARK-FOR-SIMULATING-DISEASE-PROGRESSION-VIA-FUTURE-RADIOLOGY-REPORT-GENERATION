"""
Auto-research driver (Plan B, no SFT): an LLM policy (astra, mini fallback) explores ML methods x data recipes
x ensembles x per-concept threshold calibration x hyperparams x internal train/val re-split, on INTERNAL VAL
ONLY (no sealed test -> anti-hack), under finite iters + compute budget, and emits the best config it found.
No SFT-4B arm (dropped in Plan B); the sealed-test score for the paper table is produced separately by a
budgeted request_eval driver. Usage: python3 -m playground.agentB_train [max_iters] [policy_model]
"""
import os, sys, json, time
sys.path.insert(0, "${VP_ROOT}/analysis")
from llm_client import chat_full
from playground.env_temporal import TemporalEnv

POLICY = sys.argv[2] if len(sys.argv) > 2 else "gpt-6-astra"
DEC = os.environ.get("VP_AGENT_DECISION", "${VP_ROOT}/analysis/agentB_decision.json")
COMPUTE = int(os.environ.get("VP_AGENT_COMPUTE", "30"))    # training compute budget (train=1, ensemble=n, transition=2)
RECENT_K = int(os.environ.get("VP_AGENT_RECENT", "6"))     # actions shown in RECENT (before LAST); 0=none, -1=full history
SYS = """You are an autonomous ML-research POLICY for a clinical exam-finding state-prediction benchmark.
Each turn you FIRST think briefly, THEN output ONE action. Respond with a single JSON object:
{"thinking":"<2-5 sentences: read STATE/RECENT/best_so_far, list which options remain untried, weigh which is
most likely to raise micro-F1, and why>","action":"...","args":{...},"why":"<one-line justification>"}.
You search over model families, data recipes, ensembles, per-concept threshold calibration, and hyperparameters
on an INTERNAL validation split to find the best configuration. You NEVER see the sealed test set.

OBJECTIVE: maximize internal-val instance MICRO-F1 (the primary reported metric). macro-F1 is a secondary
fairness diagnostic only -- never trade away micro-F1 to chase macro-F1.

CONTEXT each turn: STATE (best_so_far config+score, remaining compute/iteration budget, current recipe/
threshold), LAST (the previous action's FULL result), and RECENT (recent actions BEFORE the last one, with
their internal-val scores; older history is only visible via best_so_far). Always use RECENT and best_so_far:
never re-measure a configuration you already tried (same model+recipe+threshold_mode+hparams+feature selection);
each action should build on what scored best.

ACTIONS (choose exactly one per turn):
- {"action":"train","args":{"method":"base-rate|logreg|rf|xgb|mlp"}}
    Train one model family and measure internal-val. Cheap (1 compute unit). base-rate = majority baseline.
- {"action":"train_ensemble","args":{"methods":["rf","xgb"]}}
    Soft-vote (average predicted probabilities) of >=2 of logreg|rf|xgb|mlp. Often beats single models.
    Cost = number of methods.
- {"action":"set_recipe","args":{"recipe":"default|balanced|oversample"}}
    Training data distribution. default = as-is. balanced = weight classes inversely to frequency.
    oversample = duplicate positive examples. Prefer balanced/oversample when concepts are rare/imbalanced.
    Applies to the NEXT train; you must retrain to measure its effect.
- {"action":"set_threshold_mode","args":{"mode":"fixed|calibrated"}}
    fixed = 0.5 probability cutoff. calibrated = per-concept decision threshold tuned on internal-val to
    maximize F1. calibrated usually gives the single largest micro-F1 gain on imbalanced multilabel findings;
    turn it on once you have a strong model, then retrain.
- {"action":"set_min_pos","args":{"n":5}}
    Minimum positive training examples required to fit a concept's classifier (otherwise it predicts all-absent
    for that concept). Raise (e.g. 10-20) to skip noisy ultra-rare concepts; lower (e.g. 2-3) to attempt more
    concepts. Default 5. Retrain to measure.
- {"action":"set_hparams","args":{"method":"rf","params":{"n_estimators":300}}}
    Override a family's hyperparameters, then retrain that family. Use to tune the BEST family after the family
    comparison. Suggested ranges:
      logreg: {"C":0.1-10, "max_iter":100-500}
      rf:     {"n_estimators":100-400, "max_depth":null or 6-20, "min_samples_leaf":1-5}
      xgb:    {"n_estimators":100-400, "max_depth":3-8, "learning_rate":0.03-0.3}
      mlp:    {"hidden":[64] or [128,64], "max_iter":200-500}
- {"action":"resplit_val","args":{"val_frac":0.1,"seed":0}}
    Re-draw the internal train/val split from the NON-sealed pool (val_frac 0.05-0.3). Use if internal-val looks
    too small/noisy, or to check that a config is robust across a different split. NEVER touches the sealed test.
- {"action":"set_feature_groups","args":{"groups":["self"]}}
    Select which STATE feature groups feed the model. 'self' = the target exam's own concepts' prior four-state
    (compact, usually strongest); 'cross' = all other pool concepts' prior four-state (cross-region context,
    often just noise/dilution). Use ["self"] or ["self","cross"]. Only when the dataset exposes feature groups.
    Condition features (time-gap Delta + interventions), when the dataset has them, are ALWAYS fed regardless
    of this selection -- they are clinically mandatory and cannot be excluded. Retrain to measure.
- {"action":"train_transition","args":{"config":{"hidden":128,"depth":2,"dropout":0.0,"cond_mode":"concat","selfsup_ratio":0.5,"learner":"mlp"}}}
    Train the conditional-transition WORLD MODEL: full prior state + conditions (time-gap Delta,
    interventions: meds/procedures/ED) -> full next state over the SAME concept space, then read out positive
    concepts. This is the only arm that models the FULL state (train() families are per-target-exam-type
    conditional models). Knobs are bounded menus:
    learner {"mlp","logreg","rf","xgb"}: "mlp" = torch net (hidden {64,128,256}, depth {1,2}, dropout {0.0,0.2},
    cond_mode {"concat","film"} apply, and it can model condition-x-state interactions); "logreg"/"rf"/"xgb" =
    ONE pooled per-concept classifier over the full state, trained across ALL exam types (the full-state
    version of the sklearn families; mlp-only knobs are ignored). selfsup_ratio {0,0.25,0.5,1.0} = fraction of
    Delta=0 identity self-supervised samples mixed in (anchors state continuity), applies to every learner.
    Unspecified knobs INHERIT your previous train_transition config (first call uses defaults
    {"hidden":128,"depth":2,"dropout":0.0,"cond_mode":"concat","selfsup_ratio":0.25,"learner":"mlp"}); the
    observation echoes the resolved full config. recipe 'balanced' applies here as class weighting on the
    training loss ('oversample' is sklearn-only, ignored); feature selection and threshold calibration apply.
    Cost 2 compute. Available only when STATE shows transition_arm.
- {"action":"set_state_concepts","args":{"names":["pleural_effusion","..."],"mode":"evidence"}}
    Choose WHICH concepts (any non-empty subset of STATE's state_concepts list) represent the patient state.
    mode "evidence" (default): the subset limits only the INPUT evidence; every concept is still supervised
    and predicted. mode "state": the subset IS the state space at both time points -- concepts outside it are
    neither supervised nor predicted (forced not_mentioned), so their gold positives become guaranteed misses;
    a good subset must cover the concepts that actually occur. Overrides set_feature_groups (last one wins);
    state extras + condition features are always kept. Use it to test hypotheses like "only clinically
    persistent findings carry signal" or to prune noise/dilution concepts; your step and compute budgets bound
    how many subsets you can try. Retrain to measure. Only when STATE shows state_concepts.
- {"action":"stop","args":{}}
    End the search. Use when no untried, low-cost action is likely to beat best_so_far.

RECOMMENDED STRATEGY:
(1) First compare model FAMILIES under one reasonable recipe: train logreg, rf, and xgb, then train_ensemble of
    the best two (xgb and ensembles are often strongest).
(2) Turn on calibrated thresholds and retrain the best config (usually the largest single gain).
(3) If STATE shows transition_arm, explore train_transition seriously: try both cond_mode values and 2-3
    selfsup_ratio values -- condition-x-state interaction is where untapped gain is expected.
(4) If STATE shows state_concepts, spend a few iterations on set_state_concepts: e.g. drop concepts you
    suspect are noise for the current pool, or try mode "state" with a compact high-prevalence subset --
    feature/state-space choice is a first-class research axis here, not an afterthought.
(5) THEN tune the best family only: try set_hparams; if concepts are imbalanced try balanced/oversample;
    optionally set_min_pos or resplit_val to probe robustness.
(6) Never change a setting without a following train to measure its effect.
(7) STOP once no untried, low-cost action is likely to beat best_so_far -- do not waste iterations."""


def ask(state, last, hist, model):
    win = [] if RECENT_K == 0 else (hist[:-1] if RECENT_K < 0 else hist[-(RECENT_K + 1):-1])
    u = f"STATE:\n{json.dumps(state,ensure_ascii=False)}\n\nLAST:\n{json.dumps(last,ensure_ascii=False)}\n\nRECENT:{json.dumps(win,ensure_ascii=False)}\n\nOne action, JSON only."
    out, usage = chat_full([{"role": "system", "content": SYS}, {"role": "user", "content": u}], model=model, temperature=0.2, max_tokens=800)
    return json.loads(out[out.find("{"):out.rfind("}") + 1]), (usage or {})


def main():
    mi = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    env = TemporalEnv(compute_units=COMPUTE, eval_queries=0, max_iters=mi)  # eval_queries=0: no sealed test in this driver
    best = {"model": None, "recipe": None, "threshold_mode": None, "min_pos": None, "hparams": None,
            "feature_groups": None, "state_concepts": None, "state_concepts_mode": None,
            "transition_config": None, "macro": -1, "micro": -1}
    trace = []; history = []; last = {"note": "start"}; hist = []; model = POLICY
    term_reason = "max_iters"
    n_calls = 0; tok_prompt = 0; tok_completion = 0; t_start = time.time(); fail_streak = 0

    def record_best(model_name):
        v = last["internal_val"]
        if v["micro_f1"] > best["micro"]:   # primary objective = instance micro-F1 (matches report-score headline)
            best.update({"model": model_name, "recipe": env.recipe, "threshold_mode": env.thr_mode,
                         "min_pos": env.min_pos, "hparams": json.loads(json.dumps(env.hparams)),
                         "feature_groups": list(env.active_groups) if env.active_groups else None,
                         "state_concepts": list(env.active_concepts) if env.active_concepts else None,
                         "state_concepts_mode": (env.concept_mode if env.active_concepts else None),
                         "transition_config": (dict(env.trans_cfg) if model_name == "transition" and env.trans_cfg else None),
                         "macro": v["macro_f1"], "micro": v["micro_f1"]})

    def snap(step, a):
        iv = last.get("internal_val") if isinstance(last, dict) else None
        history.append({"step": step, "action": a, "model": (env.cur[0] if env.cur else None),
                        "recipe": env.recipe, "threshold_mode": env.thr_mode, "min_pos": env.min_pos,
                        "val_macro": iv["macro_f1"] if iv else None, "val_micro": iv["micro_f1"] if iv else None,
                        "best_macro_so_far": (best["macro"] if best["macro"] >= 0 else None),
                        "compute_left": env.compute_units, "iters_used": step + 1})

    for step in range(mi):
        env.iters = step; st = env.summary()
        st["best_so_far"] = ({k: best[k] for k in ("model", "recipe", "threshold_mode", "min_pos", "macro", "micro")}
                             if best["macro"] >= 0 else None)
        try:
            act, usage = ask(st, last, hist, model)
            fail_streak = 0
            n_calls += 1
            tok_prompt += int(usage.get("prompt_tokens", 0) or 0)
            tok_completion += int(usage.get("completion_tokens", 0) or 0)
        except Exception as e:
            fail_streak += 1
            if model != "gpt-5.4-mini":
                print(f"[{step}] policy {model} failed ({str(e)[:60]}); fallback gpt-5.4-mini"); model = "gpt-5.4-mini"; time.sleep(5); continue
            if fail_streak >= 3:
                print(f"[{step}] policy err x{fail_streak} {e}; stop"); term_reason = "policy_failure"; break
            print(f"[{step}] policy err {e}; retry in 30s ({fail_streak}/3)"); time.sleep(30); continue
        a = act.get("action"); args = act.get("args", {}) or {}; why = act.get("why", ""); think = act.get("thinking", "")
        if a == "set_recipe": last = env.set_recipe(**args)
        elif a == "set_threshold_mode": last = env.set_threshold_mode(**args)
        elif a == "set_min_pos": last = env.set_min_pos(**args)
        elif a == "set_hparams": last = env.set_hparams(**args)
        elif a == "resplit_val": last = env.resplit_val(**args)
        elif a == "set_feature_groups": last = env.set_feature_groups(args.get("groups", []))
        elif a == "set_state_concepts": last = env.set_state_concepts(args.get("names", []), args.get("mode", "evidence"))
        elif a == "train":
            last = env.train(args.get("method", ""))
            if isinstance(last, dict) and last.get("internal_val"): record_best(args.get("method", ""))
        elif a == "train_ensemble":
            last = env.train_ensemble(args.get("methods", []))
            if isinstance(last, dict) and last.get("internal_val"): record_best(last.get("model", "ensemble"))
        elif a == "train_transition":
            cfg = args.get("config") if isinstance(args.get("config"), dict) else {k: v for k, v in args.items() if k != "config"}
            last = env.train_transition(cfg or {})
            if isinstance(last, dict) and last.get("internal_val"): record_best("transition")
        elif a == "stop": last = {"ok": True}; trace.append({"step": step, "action": a, "why": why}); snap(step, a); term_reason = "model_stop"; break
        else: last = {"error": f"unknown {a}"}
        trace.append({"step": step, "action": a, "args": args, "why": why, "thinking": think, "obs": last})
        snap(step, a)
        hist.append({"step": step, "action": a, "args": args,
                     "val": (last.get("internal_val") if isinstance(last, dict) else None),
                     "note": (last.get("note") or last.get("error")) if isinstance(last, dict) else None})
        print(f"[{step}] {a}({args}) -> {json.dumps(last,ensure_ascii=False)[:160]} | {why[:60]}", flush=True)
        if a == "stop": break

    decision = {"best_config": best, "trajectory_length": len(trace), "termination_reason": term_reason,
                "budgets": {"max_iters": mi, "compute_units": COMPUTE, "recent_window": RECENT_K,
                            "show_progress": os.environ.get("VP_AGENT_SHOW_PROGRESS", "0") == "1"},
                "policy_model_start": POLICY, "policy_model_final": model, "history": history, "trace": trace,
                "policy_usage": {"policy_model": POLICY, "api_calls": n_calls, "prompt_tokens": tok_prompt,
                                 "completion_tokens": tok_completion, "total_tokens": tok_prompt + tok_completion,
                                 "wall_sec": round(time.time() - t_start, 1)}}
    json.dump(decision, open(DEC, "w"), ensure_ascii=False, indent=2)
    print(f"\n=== auto-research decision | trajectory={len(trace)} term={term_reason} ===\n" + json.dumps(best, ensure_ascii=False, indent=2))
    print("=== policy cost ===\n" + json.dumps(decision["policy_usage"], ensure_ascii=False, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
