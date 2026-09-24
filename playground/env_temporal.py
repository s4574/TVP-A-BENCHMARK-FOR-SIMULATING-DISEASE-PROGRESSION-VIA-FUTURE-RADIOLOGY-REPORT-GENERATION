"""
Temporal multimodal auto-research env (Plan B, no SFT): agent auto-selects among ML methods x data recipes,
model ENSEMBLES, per-concept THRESHOLD calibration, HYPERPARAMS, and internal train/val RE-SPLIT, under
budget + anti-hack. Sealed test (subject-split) judged only via budgeted request_eval; agent searches on
internal val. The SFT-4B arm is intentionally removed here (Plan B); if episode SFT training returns it can
be re-added as an ablation behind VP_ENABLE_SFT_ARM.

Resource note: this box is shared/oversubscribed. Keep sklearn parallelism low: VP_AGENT_NJOBS (default 2),
VP_AGENT_CAP (default 6000) subsamples train per-ns for fast, light agent-loop iteration.
"""
import os, pickle, copy, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
sys.path.insert(0, "${VP_ROOT}/analysis")
try:
    import cond_transition as CT     # torch conditional transition arm (needs condtrans pkl with meta)
except Exception:
    CT = None

PKL = os.environ.get("VP_AGENT_ML_DATA", "${VP_ROOT}/analysis/agent_ml_data.pkl")
NJOBS = int(os.environ.get("VP_AGENT_NJOBS", "2"))      # keep low: shared/oversubscribed host
CAP = int(os.environ.get("VP_AGENT_CAP", "6000"))       # subsample train per ns for fast agent-loop iteration
SHOW_PROGRESS = os.environ.get("VP_AGENT_SHOW_PROGRESS", "0") == "1"   # expose elapsed/max steps in STATE
BASE_METHODS = ["base-rate", "logreg", "rf", "xgb", "mlp"]
RECIPES = ["default", "balanced", "oversample"]
THR = np.round(np.linspace(0.1, 0.9, 17), 3)            # per-concept threshold grid


def _f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
    return 2 * P * R / (P + R) if P + R else 0.0


class TemporalEnv:
    def __init__(self, compute_units=30, eval_queries=6, max_iters=12, ship_macro=0.15):
        d = pickle.load(open(PKL, "rb"))
        self.data = d["data"]; self.ns_vocab = d["ns_vocab"]; self.top_ns = d["top_ns"]
        self.te_keys = d.get("te_keys")     # sealed-case alignment for downstream render tooling
        self.compute_units = compute_units; self.eval_used = 0; self.eval_queries = eval_queries
        self.max_iters = max_iters; self.iters = 0; self.ship_macro = ship_macro
        self.recipe = "default"; self.thr_mode = "fixed"; self.min_pos = 5
        self.hparams = {}                       # optional per-method overrides
        self.active_groups = None               # feature-group selection (None=all cols); e.g. ["self"], ["self","cross"]
        self.active_concepts = None             # explicit union-concept subset (overrides feature groups when set)
        self.concept_mode = "evidence"          # 'evidence': subset limits input only; 'state': subset IS the state space (both ends)
        self.ns_spec = {}                       # per-namespace model/ensemble override (falls back to the global spec)
        self.ns_hparams = {}                    # per-namespace per-method hparam overrides
        self.last_val_by_ns = None              # per-ns internal-val micro from the most recent val scoring
        self.has_groups = any("groups" in self.data[ns] for ns in self.top_ns)
        self.cur = None; self.cur_val = None; self.cur_thr = {}; self.last_eval = None; self.shipped = None
        # conditional transition arm (world model): available iff pkl carries meta + four-state labels
        self.meta = d.get("meta")
        self.union_concepts = (self.meta or {}).get("union_concepts")
        self.has_transition = bool(CT is not None and self.meta
                                   and all("Y4tr" in self.data[ns] for ns in self.top_ns))
        self.trans_model = None; self.trans_cfg = None
        # working train/val pool per ns (sealed te never touched); supports resplit_val
        self.work = {}
        for ns in self.top_ns:
            D = self.data[ns]
            self.work[ns] = {"Xtr": D["Xtr"], "Ytr": D["Ytr"], "Xval": D["Xval"], "Yval": D["Yval"]}
            if "Y4tr" in D:
                self.work[ns]["Y4tr"] = D["Y4tr"]; self.work[ns]["Y4val"] = D["Y4val"]
            for k in ("LM4tr", "LM4val"):
                if k in D:
                    self.work[ns][k] = D[k]

    # ---- helpers ----
    def _prep(self, Xtr, Xev):
        med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xev = np.where(np.isnan(Xev), med, Xev)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        return (Xtr - mu) / sd, (Xev - mu) / sd

    def _cols(self, ns):
        """Column indices for the active feature selection (None = all columns).
        Explicit concept subset (set_state_concepts) takes precedence over feature groups; state extras and
        the 'cond' group (Delta + interventions) are a clinical prior: ALWAYS included, not agent-selectable."""
        g = self.data[ns].get("groups")
        if self.active_concepts is not None and self.union_concepts:
            sb = self.meta["state_block"]; pc = sb["per_concept"]
            cols = []
            for c in self.active_concepts:
                i = self.union_concepts.index(c)
                cols += range(pc * i, pc * i + pc)
            cols += range(pc * sb["n_concepts"], pc * sb["n_concepts"] + sb["extras"])  # extras always kept
            if g:
                cols = set(cols) | set(g.get("cond", []))
            return sorted(cols)
        if not g:
            return None
        if not self.active_groups:
            return None          # all columns (conditions included)
        cols = []
        for grp in self.active_groups:
            cols += g.get(grp, [])
        cols = sorted(set(cols) | set(g.get("cond", [])))
        return cols if cols else None

    def _mk(self, method, ns=None):
        cw = "balanced" if self.recipe == "balanced" else None
        hp = dict(self.hparams.get(method, {}))
        if ns is not None:
            hp.update(self.ns_hparams.get(ns, {}).get(method, {}))
        if method == "logreg":
            return LogisticRegression(max_iter=int(hp.get("max_iter", 100)), C=float(hp.get("C", 1.0)),
                                      class_weight=cw, solver="liblinear")
        if method == "rf":
            return RandomForestClassifier(n_estimators=int(hp.get("n_estimators", 100)),
                                          max_depth=hp.get("max_depth", None), class_weight=cw, n_jobs=NJOBS)
        if method == "xgb":
            return XGBClassifier(n_estimators=int(hp.get("n_estimators", 150)), max_depth=int(hp.get("max_depth", 4)),
                                 learning_rate=float(hp.get("learning_rate", 0.1)), n_jobs=NJOBS, verbosity=0)
        if method == "mlp":
            return MLPClassifier(hidden_layer_sizes=tuple(hp.get("hidden", (64,))), max_iter=int(hp.get("max_iter", 200)))
        return None

    def _proba_one(self, method, xtr, ytr, xev, ns=None):
        """Return P(y=1) on xev for a single concept; base-rate handled by caller."""
        clf = self._mk(method, ns)
        try:
            clf.fit(xtr, ytr)
            if hasattr(clf, "predict_proba"):
                return clf.predict_proba(xev)[:, 1]
            return clf.predict(xev).astype(float)
        except Exception:
            return np.zeros(len(xev))

    def _methods_of(self, spec):
        return spec.split(":", 1)[1].split("+") if spec.startswith("ensemble:") else [spec]

    # ---- conditional transition arm (world model) ----
    def _pool_transition(self, split):
        """Pooled (S, C, Y4, ns_of_row, LM) for the transition arm. State columns outside the active feature
        groups are ZEROED per ns (concept-subset knob); condition columns are always fed (clinical prior).
        LM = per-ROW label mask: dense per-case masks (LM4tr/LM4val, from a dense-GT pkl) when present,
        else the ns's static label_mask tiled."""
        sb = self.meta["state_block"]
        n_state = sb["per_concept"] * sb["n_concepts"] + sb["extras"]
        S, C, Y4, ns_of, LM = [], [], [], [], []
        for ns in self.top_ns:
            src = self.data[ns] if split == "te" else self.work[ns]
            X = src[f"X{split}" if split != "te" else "Xte"]
            if not len(X):
                continue
            Xs = X[:, :n_state]
            cols = self._cols(ns)
            if cols is not None:
                keep = np.zeros(n_state, dtype=bool)
                for c in cols:
                    if c < n_state:
                        keep[c] = True
                Xs = np.where(keep[None, :], Xs, 0.0)
            S.append(Xs); C.append(X[:, n_state:])
            Y4.append(src[f"Y4{split}" if split != "te" else "Y4te"].astype(np.int64))
            lm = src.get(f"LM4{split}") if split != "te" else src.get("LM4te")
            LM.append(np.asarray(lm, dtype=np.float32) if lm is not None
                      else np.tile(self.data[ns]["label_mask"].astype(np.float32), (len(X), 1)))
            ns_of += [ns] * len(X)
        return np.vstack(S), np.vstack(C), np.vstack(Y4), np.array(ns_of), np.vstack(LM)

    def _score_transition(self, split, return_preds=False):
        """Score the cached transition model on val/te with the same threshold + metric machinery."""
        Sv, Cv, _, nsv, _ = self._pool_transition("val")
        P_val = self.trans_model.proba_pos(Sv, Cv)
        if split == "te":
            Se, Ce, _, nse, _ = self._pool_transition("te")
            P_ev, ns_ev = self.trans_model.proba_pos(Se, Ce), nse
        else:
            P_ev, ns_ev = P_val, nsv
        uc = self.meta["union_concepts"]
        keep = self._state_keep()
        if keep is not None:      # 'state' mode: concepts outside the subset read out as not-positive
            P_val = P_val * keep[None, :]
            P_ev = P_ev * keep[None, :]
        calibrate = (split == "val" and self.thr_mode == "calibrated")
        if split == "val":
            self.cur_thr = {}
        ns_macros = []; itp = ifp = ifn = 0; preds_by_ns = {}
        for ns in self.top_ns:
            sel = [uc.index(c) for c in self.ns_vocab[ns]]
            Yev = (self.data[ns]["Yte"] if split == "te" else self.work[ns]["Yval"])
            pev = P_ev[ns_ev == ns][:, sel]
            if not len(Yev):
                continue
            preds = np.zeros_like(Yev); f1s = []
            pval = P_val[nsv == ns][:, sel]
            Yval = self.work[ns]["Yval"]
            for k in range(Yev.shape[1]):
                if calibrate and len(Yval):
                    best_t, best_f = 0.5, -1.0
                    for t in THR:
                        f = _f1(Yval[:, k], (pval[:, k] >= t).astype(int))
                        if f > best_f: best_f, best_t = f, t
                    self.cur_thr[(ns, k)] = float(best_t)
                t = self.cur_thr.get((ns, k), 0.5)
                p = (pev[:, k] >= t).astype(int)
                preds[:, k] = p; f1s.append(_f1(Yev[:, k], p))
            ns_macros.append(np.mean(f1s))
            preds_by_ns[ns] = preds
            for i in range(len(Yev)):
                gp = set(np.where(Yev[i] == 1)[0]); pp = set(np.where(preds[i] == 1)[0])
                itp += len(gp & pp); ifp += len(pp - gp); ifn += len(gp - pp)
        macro = float(np.mean(ns_macros)) if ns_macros else 0.0
        P = itp / (itp + ifp) if itp + ifp else 0; R = itp / (itp + ifn) if itp + ifn else 0
        m = {"macro_f1": round(macro, 3), "micro_f1": round(2 * P * R / (P + R) if P + R else 0, 3)}
        return (m, preds_by_ns) if return_preds else m


    def _score(self, spec, split, return_preds=False):  # split in {'val','te'}; spec = method or "ensemble:m1+m2"
        calibrate_any = (split == "val" and self.thr_mode == "calibrated")
        if split == "val":
            self.cur_thr = {}
            self._val_by_ns = {}
        ns_macros = []; itp = ifp = ifn = 0; preds_by_ns = {}
        Kdrop = (set(self.active_concepts) if (self.active_concepts is not None and self.concept_mode == "state")
                 else None)   # 'state' mode: concepts outside the subset are not trained/predicted (forced negative)
        for ns in self.top_ns:
            spec_ns = self.ns_spec.get(ns, spec)
            methods = self._methods_of(spec_ns)
            is_baserate = (spec_ns == "base-rate")
            calibrate = calibrate_any and not is_baserate
            Xtr, Ytr = self.work[ns]["Xtr"], self.work[ns]["Ytr"]
            if len(Xtr) > CAP:
                sel = np.random.default_rng(0).choice(len(Xtr), CAP, replace=False); Xtr, Ytr = Xtr[sel], Ytr[sel]
            if split == "val":
                Xev, Yev = self.work[ns]["Xval"], self.work[ns]["Yval"]
            else:
                Xev, Yev = self.data[ns]["Xte"], self.data[ns]["Yte"]
            if len(Xev) == 0 or len(Xtr) == 0:
                continue
            cols = self._cols(ns)
            if cols is not None:
                Xtr = Xtr[:, cols]; Xev = Xev[:, cols]
            Xtrs, Xevs = self._prep(Xtr, Xev); preds = np.zeros_like(Yev); f1s = []
            for k in range(Ytr.shape[1]):
                ytr, yev = Ytr[:, k], Yev[:, k]
                if Kdrop is not None and self.ns_vocab[ns][k] not in Kdrop:
                    p = np.zeros(len(yev))
                elif ytr.sum() < self.min_pos or ytr.sum() == len(ytr):
                    p = np.zeros(len(yev))
                elif is_baserate:
                    p = np.full(len(yev), 1.0 if ytr.mean() >= 0.5 else 0.0)
                else:
                    xtr2, y2 = Xtrs, ytr
                    if self.recipe == "oversample" and 0 < ytr.mean() < 0.5:  # duplicate positives
                        pos = np.where(ytr == 1)[0]; reps = int(1 / max(ytr.mean(), 0.05)) - 1
                        if reps > 0:
                            xtr2 = np.vstack([Xtrs] + [Xtrs[pos]] * reps); y2 = np.concatenate([ytr] + [ytr[pos]] * reps)
                    proba = np.mean([self._proba_one(m, xtr2, y2, Xevs, ns) for m in methods], axis=0)
                    if calibrate:
                        best_t, best_f = 0.5, -1.0
                        for t in THR:
                            f = _f1(yev, (proba >= t).astype(int))
                            if f > best_f: best_f, best_t = f, t
                        self.cur_thr[(ns, k)] = float(best_t); t = best_t
                    else:
                        t = self.cur_thr.get((ns, k), 0.5)
                    p = (proba >= t).astype(int)
                preds[:, k] = p; f1s.append(_f1(yev, p))
            ns_macros.append(np.mean(f1s))
            preds_by_ns[ns] = preds
            ntp = nfp = nfn = 0
            for i in range(len(Yev)):
                gp = set(np.where(Yev[i] == 1)[0]); pp = set(np.where(preds[i] == 1)[0])
                ntp += len(gp & pp); nfp += len(pp - gp); nfn += len(gp - pp)
            itp += ntp; ifp += nfp; ifn += nfn
            if split == "val":
                nP = ntp / (ntp + nfp) if ntp + nfp else 0; nR = ntp / (ntp + nfn) if ntp + nfn else 0
                self._val_by_ns[ns] = round(2 * nP * nR / (nP + nR) if nP + nR else 0.0, 3)
        if split == "val":
            self.last_val_by_ns = dict(self._val_by_ns)
        macro = float(np.mean(ns_macros)) if ns_macros else 0.0
        P = itp / (itp + ifp) if itp + ifp else 0; R = itp / (itp + ifn) if itp + ifn else 0
        m = {"macro_f1": round(macro, 3), "micro_f1": round(2 * P * R / (P + R) if P + R else 0, 3)}
        return (m, preds_by_ns) if return_preds else m

    # ---- actions ----
    def summary(self):
        extra = ["set_recipe", "set_threshold_mode(fixed|calibrated)", "set_min_pos(int)",
                 "set_hparams(method,{...})", "resplit_val(val_frac,seed)",
                 "set_feature_groups(['self'] or ['self','cross'])",
                 "train(method)", "train_ensemble([methods])", "stop"]
        extra += ["inspect_data (free, per-ns sizes + last per-ns val)",
                  "set_ns_model(ns, spec|'global')", "set_ns_hparams(ns, method, {...})"]
        out = {"iters_left": self.max_iters - self.iters, "compute_left": self.compute_units,
               "eval_left": self.eval_queries - self.eval_used, "recipe": self.recipe,
               "threshold_mode": self.thr_mode, "min_pos": self.min_pos, "hparams": self.hparams,
               "feature_groups": self.active_groups, "feature_groups_available": self.has_groups,
               "ns_model_overrides": dict(self.ns_spec), "ns_hparam_overrides": {k: list(v) for k, v in self.ns_hparams.items()},
               "state_concepts_selected": self.active_concepts,
               "state_concepts_mode": (self.concept_mode if self.active_concepts is not None else None),
               "current_model": self.cur, "current_internal_val": self.cur_val,
               "last_sealed_eval": self.last_eval, "shipped": self.shipped,
               "methods": BASE_METHODS, "recipes": RECIPES}
        if SHOW_PROGRESS:
            out["iters_used"] = self.iters; out["max_iters"] = self.max_iters
        if self.union_concepts:
            extra.insert(-1, "set_state_concepts([concept names]) free subset of state_concepts")
            out["state_concepts"] = self.union_concepts
        if self.has_transition:
            extra.insert(-1, "train_transition({hidden,depth,dropout,cond_mode,selfsup_ratio}) cost=2")
            out["transition_arm"] = {"available": True, "menus": CT.MENUS,
                                     "note": ("world model: full state(t-) + conditions -> full state(t), same "
                                              "concept space; conditions (Delta+interventions) always fed, "
                                              "not removable; selfsup_ratio mixes Delta=0 identity samples; "
                                              "unspecified knobs inherit your previous transition config "
                                              "(defaults first time); recipe 'balanced' = class-weighted train "
                                              "loss here; 'oversample' is sklearn-only (ignored)")}
        out["extra_actions"] = extra
        return out

    def set_recipe(self, recipe):
        if recipe not in RECIPES: return {"error": "unknown recipe"}
        self.recipe = recipe; return {"ok": True, "recipe": recipe, "note": "retrain to apply"}

    def set_feature_groups(self, groups):
        """Select which STATE feature groups to use: 'self' = target-namespace concepts' prior four-state,
        'cross' = all other pool concepts' prior four-state. Only available on a union-pool dataset.
        Condition features (Delta + interventions), when present, are always kept regardless of selection."""
        if not self.has_groups:
            return {"error": "this dataset has a single feature block; feature groups not available"}
        valid = {"self", "cross"}
        if not isinstance(groups, list) or not groups or not set(groups) <= valid:
            return {"error": "groups must be a non-empty subset of ['self','cross']"}
        self.active_groups = list(dict.fromkeys(groups)); self.active_concepts = None
        return {"ok": True, "feature_groups": self.active_groups, "note": "retrain to apply"}

    def set_state_concepts(self, names, mode="evidence"):
        """Free (but validated) selection of WHICH union concepts represent the patient state: any non-empty
        subset of union_concepts. mode='evidence': subset limits the INPUT evidence only; every concept is
        still supervised and predicted. mode='state': the subset IS the state space at both time points
        (§12 locked design) -- concepts outside it are neither supervised nor predicted (read out as
        not_mentioned; their gold positives become guaranteed misses). Overrides set_feature_groups (last one
        wins). State extras + condition features are always kept."""
        if not self.union_concepts:
            return {"error": "concept-level selection unavailable (dataset lacks union_concepts meta)"}
        if mode not in ("evidence", "state"):
            return {"error": "mode must be 'evidence' or 'state'"}
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            return {"error": "names must be a non-empty list of concept strings from state_concepts"}
        bad = [n for n in names if n not in self.union_concepts]
        if bad:
            return {"error": f"unknown concepts (not in state_concepts): {bad[:8]}"}
        self.active_concepts = list(dict.fromkeys(names)); self.active_groups = None; self.concept_mode = mode
        return {"ok": True, "n_selected": len(self.active_concepts), "mode": mode,
                "note": "retrain to apply; overrides feature_groups; extras+conditions always kept"}

    def _state_keep(self):
        """Bool keep-vector over union concepts under 'state' mode, else None (all concepts live)."""
        if self.active_concepts is None or self.concept_mode != "state" or not self.union_concepts:
            return None
        sel = set(self.active_concepts)
        return np.array([c in sel for c in self.union_concepts])

    def set_threshold_mode(self, mode):
        if mode not in ("fixed", "calibrated"): return {"error": "mode must be fixed|calibrated"}
        self.thr_mode = mode; return {"ok": True, "threshold_mode": mode, "note": "retrain to apply"}

    def set_min_pos(self, n):
        try: n = int(n)
        except Exception: return {"error": "min_pos must be int"}
        if not (1 <= n <= 100): return {"error": "min_pos out of range 1..100"}
        self.min_pos = n; return {"ok": True, "min_pos": n, "note": "retrain to apply"}

    def inspect_data(self):
        """Read-only dataset/train-val diagnostics (NO sealed information): per-ns pool sizes, concept counts,
        supervision density, and the per-ns internal-val micro-F1 of the most recent training pass."""
        out = {}
        for ns in self.top_ns:
            D = self.data[ns]
            npos = float(self.work[ns]["Ytr"].sum())
            out[ns] = {"n_train": int(len(self.work[ns]["Xtr"])), "n_val": int(len(self.work[ns]["Xval"])),
                       "n_concepts": int(D["Ytr"].shape[1]),
                       "guarded_concepts": int((self.work[ns]["Ytr"].sum(0) < self.min_pos).sum()),
                       "train_positives": int(npos),
                       "val_micro_last_train": (self.last_val_by_ns or {}).get(ns)}
        return {"ok": True, "note": ("micro-F1 pools all namespaces; each contributes ~proportionally to its "
                                     "positives. Small n_val slices give NOISY per-ns feedback."),
                "per_ns": out}

    def set_ns_model(self, ns, spec):
        """Per-namespace model override: this ns uses `spec` (a family or 'ensemble:m1+m2') instead of the
        globally trained spec. Pass spec='global' to clear. Retrain to apply."""
        if ns not in self.top_ns:
            return {"error": f"unknown ns; choose from {self.top_ns}"}
        if spec in (None, "global", ""):
            self.ns_spec.pop(ns, None)
            return {"ok": True, "ns": ns, "note": "override cleared; retrain to apply"}
        base = spec.split(":", 1)[1].split("+") if isinstance(spec, str) and spec.startswith("ensemble:") else [spec]
        if not all(m in BASE_METHODS for m in base):
            return {"error": f"spec must be one of {BASE_METHODS} or 'ensemble:m1+m2'"}
        self.ns_spec[ns] = spec
        return {"ok": True, "ns": ns, "spec": spec, "note": "retrain to apply"}

    def set_ns_hparams(self, ns, method, params):
        """Per-namespace hyperparameter override for one family (merged over the global hparams)."""
        if ns not in self.top_ns:
            return {"error": f"unknown ns; choose from {self.top_ns}"}
        if method not in ("logreg", "rf", "xgb", "mlp"):
            return {"error": "hparams only for logreg|rf|xgb|mlp"}
        if not isinstance(params, dict):
            return {"error": "params must be a dict"}
        self.ns_hparams.setdefault(ns, {})[method] = {**self.ns_hparams.get(ns, {}).get(method, {}), **params}
        return {"ok": True, "ns": ns, "method": method, "hparams": self.ns_hparams[ns][method], "note": "retrain to apply"}

    def set_hparams(self, method, params):
        if method not in ("logreg", "rf", "xgb", "mlp"): return {"error": "hparams only for logreg|rf|xgb|mlp"}
        if not isinstance(params, dict): return {"error": "params must be a dict"}
        self.hparams[method] = {**self.hparams.get(method, {}), **params}
        return {"ok": True, "method": method, "hparams": self.hparams[method], "note": "retrain to apply"}

    def resplit_val(self, val_frac=0.1, seed=0):
        """Re-split the internal train/val from the combined non-sealed pool. Never touches sealed te."""
        try:
            val_frac = float(val_frac); seed = int(seed)
        except Exception:
            return {"error": "val_frac float, seed int"}
        if not (0.02 <= val_frac <= 0.5): return {"error": "val_frac out of range 0.02..0.5"}
        for ns in self.top_ns:
            D = self.data[ns]
            Xall = np.vstack([D["Xtr"], D["Xval"]]); Yall = np.vstack([D["Ytr"], D["Yval"]])
            n = len(Xall); idx = np.random.default_rng(seed).permutation(n); nval = max(1, int(n * val_frac))
            vi, ti = idx[:nval], idx[nval:]
            self.work[ns] = {"Xtr": Xall[ti], "Ytr": Yall[ti], "Xval": Xall[vi], "Yval": Yall[vi]}
            if "Y4tr" in D:
                Y4all = np.vstack([D["Y4tr"], D["Y4val"]])
                self.work[ns]["Y4tr"] = Y4all[ti]; self.work[ns]["Y4val"] = Y4all[vi]
            if "LM4tr" in D and "LM4val" in D:
                LMall = np.vstack([D["LM4tr"], D["LM4val"]])
                self.work[ns]["LM4tr"] = LMall[ti]; self.work[ns]["LM4val"] = LMall[vi]
        return {"ok": True, "val_frac": val_frac, "seed": seed,
                "note": ("internal split changed; sealed te untouched; retrain"
                         + ("; WARNING: dense-GT pkl -- resplit mixes dense-labeled and mention-labeled rows"
                            if any("LM4tr" in self.data[ns] for ns in self.top_ns) else ""))}

    def train(self, method):
        if method not in BASE_METHODS: return {"error": f"unknown method; use {BASE_METHODS} or train_ensemble"}
        if self.compute_units < 1: return {"error": "out of compute"}
        self.compute_units -= 1
        v = self._score(method, "val"); self.cur = (method, self.recipe); self.cur_val = v
        return {"ok": True, "model": method, "recipe": self.recipe, "threshold_mode": self.thr_mode,
                "internal_val": v, "compute_left": self.compute_units}

    def train_ensemble(self, methods):
        base = [m for m in methods if m in ("logreg", "rf", "xgb", "mlp")]
        if len(base) < 2: return {"error": "ensemble needs >=2 of logreg|rf|xgb|mlp"}
        cost = len(base)
        if self.compute_units < cost: return {"error": f"need {cost} compute for ensemble"}
        self.compute_units -= cost
        spec = "ensemble:" + "+".join(base)
        v = self._score(spec, "val"); self.cur = (spec, self.recipe); self.cur_val = v
        return {"ok": True, "model": spec, "recipe": self.recipe, "threshold_mode": self.thr_mode,
                "internal_val": v, "compute_left": self.compute_units}

    def _concept_vis(self, ns):
        """Bool vector over union concepts: which concepts' prior four-state is VISIBLE in the state input
        for rows of this ns under the current feature selection (concept subset or feature groups)."""
        n = self.meta["state_block"]["n_concepts"]
        if self.active_concepts is not None:
            sel = set(self.active_concepts)
            return np.array([c in sel for c in self.union_concepts])
        cols = self._cols(ns)
        if cols is None:
            return np.ones(n, dtype=bool)
        pc = self.meta["state_block"]["per_concept"]
        cs = set(cols)
        return np.array([(pc * i) in cs for i in range(n)])

    def train_transition(self, config=None):
        """Train the conditional-transition world model (torch MLP): full state(t-) + conditions -> full
        state(t) in the same union concept space. Conditions (Delta + interventions) are always fed.
        config knobs are bounded discrete menus (see summary); unspecified knobs inherit the previous
        transition config (defaults on first call). recipe 'balanced' applies as inverse state-frequency
        class weights on the train loss; 'oversample' is sklearn-only (ignored here). cost 2 compute."""
        if not self.has_transition:
            return {"error": "transition arm unavailable (dataset lacks four-state labels/meta)"}
        cfg, err = CT.validate_config(config or {}, base=self.trans_cfg)
        if err:
            return {"error": err, "menus": CT.MENUS}
        if self.compute_units < 2: return {"error": "need 2 compute for transition training"}
        self.compute_units -= 2
        sb = self.meta["state_block"]
        n_state = sb["per_concept"] * sb["n_concepts"] + sb["extras"]
        S, C, Y4, ns_tr, M = self._pool_transition("tr")
        Sv, Cv, Y4v, nsv, Mv = self._pool_transition("val")
        self.trans_model = CT.make_model(n_state, C.shape[1], sb["n_concepts"],
                                         len(self.meta["delta_buckets"]), cfg, n_jobs=NJOBS)
        keep = self._state_keep()
        row_vis = None
        if self.active_concepts is not None or self.active_groups:   # any input masking active
            vis = {ns: self._concept_vis(ns) for ns in self.top_ns}
            row_vis = np.stack([vis[n] for n in ns_tr]).astype(np.float32)
        self.trans_model.fit(S, C, Y4, M, Sv, Cv, Y4v, Mv, balanced=(self.recipe == "balanced"),
                             concept_keep=(keep.astype(np.float32) if keep is not None else None),
                             selfsup_row_keep=row_vis)
        self.trans_cfg = cfg
        v = self._score_transition("val")
        self.cur = ("transition", self.recipe); self.cur_val = v
        return {"ok": True, "model": "transition", "config": cfg, "recipe": self.recipe,
                "val_ce": round(self.trans_model.val_ce, 4),
                "threshold_mode": self.thr_mode, "internal_val": v, "compute_left": self.compute_units}

    def request_eval(self):
        if self.cur is None: return {"error": "train first"}
        if self.eval_used >= self.eval_queries: return {"error": "out of eval queries"}
        self.eval_used += 1; self.recipe = self.cur[1]
        if self.cur[0] == "transition":
            if self.trans_model is None: return {"error": "transition model gone; retrain"}
            v = self._score_transition("te")   # reuses self.cur_thr calibrated on val
            self.last_eval = {"model": "transition", "config": self.trans_cfg, "recipe": self.cur[1],
                              "threshold_mode": self.thr_mode, **v,
                              "eval_left": self.eval_queries - self.eval_used}
            return self.last_eval
        v = self._score(self.cur[0], "te")   # reuses self.cur_thr calibrated on val
        self.last_eval = {"model": self.cur[0], "recipe": self.cur[1], "threshold_mode": self.thr_mode,
                          **v, "eval_left": self.eval_queries - self.eval_used}
        return self.last_eval

    def package(self):
        if not self.last_eval: return {"error": "no eval"}
        if self.last_eval["macro_f1"] < self.ship_macro: return {"ok": False, "reason": "below ship bar"}
        self.shipped = self.last_eval; return {"ok": True, "shipped": self.shipped}

    def stop(self): return {"ok": True}
