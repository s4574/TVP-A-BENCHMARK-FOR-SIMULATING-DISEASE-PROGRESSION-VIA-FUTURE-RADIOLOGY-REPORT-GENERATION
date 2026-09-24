"""
Conditional transition "world model" (Plan B §12 locked design), torch MLP arm.

  input  = full state s(t-) (union four-state block, 3/concept + 3 extras) + conditions c (Delta + interventions)
  output = full state s(t) logits over the SAME union concept space, 4 classes/concept
           (present / absent / uncertain / not_mentioned); conditions never appear in the output.

Supervision: masked CE — each case only has labels on its observed (target-ns) concepts.
Self-supervision: Delta=0 identity anchors the continuity assumption: (s, Delta=0, no interventions) -> s,
mixed in at selfsup_ratio (bounded menu).

Bounded discrete menus (auto-research knobs): hidden, depth, dropout, cond_mode (concat|film), selfsup_ratio.
Fixed (not knobs): four-state encoding, same-space in/out, conditions always fed, lr/epochs/seed.
"""
import numpy as np
import torch
import torch.nn as nn

FOURSTATE = ["present", "absent", "uncertain", "not_mentioned"]
MENUS = {
    "hidden": [64, 128, 256],
    "depth": [1, 2],
    "dropout": [0.0, 0.2],
    "cond_mode": ["concat", "film"],
    "selfsup_ratio": [0.0, 0.25, 0.5, 1.0],
    "learner": ["mlp", "logreg", "rf", "xgb"],
}
DEFAULTS = {"hidden": 128, "depth": 2, "dropout": 0.0, "cond_mode": "concat", "selfsup_ratio": 0.25,
            "learner": "mlp"}
LR, EPOCHS, BATCH, PATIENCE, SEED = 1e-3, 60, 256, 8, 0


def validate_config(cfg, base=None):
    """Clamp a config dict to the bounded menus; unknown keys / off-menu values are errors.
    Unspecified knobs inherit `base` (e.g. the previous committed config) when given, else DEFAULTS."""
    out = dict(base) if base else dict(DEFAULTS)
    for k, v in (cfg or {}).items():
        if k not in MENUS:
            return None, f"unknown knob '{k}'; knobs: {sorted(MENUS)}"
        if v not in MENUS[k]:
            return None, f"{k}={v} off-menu; allowed: {MENUS[k]}"
        out[k] = v
    return out, None


def zero_cond_vector(cond_dim, n_delta_buckets):
    """Canonical Delta=0 / no-intervention condition: [log_gap=0, is_zero=1, all gap<=b buckets=1, zeros...]."""
    z = np.zeros(cond_dim, dtype=np.float32)
    z[1] = 1.0
    z[2:2 + n_delta_buckets] = 1.0
    return z


def state_to_labels(S, n_concepts):
    """Map the input state block back to 4-state labels per concept (for Delta=0 identity targets).
    Per concept the 3 dims are one-hot present/absent/uncertain; all-zero = not_mentioned."""
    B = S[:, : 3 * n_concepts].reshape(len(S), n_concepts, 3)
    lab = np.full((len(S), n_concepts), 3, dtype=np.int64)
    has = B.max(axis=2) > 0
    lab[has] = B.argmax(axis=2)[has]
    return lab


class TransitionNet(nn.Module):
    def __init__(self, state_dim, cond_dim, n_concepts, hidden, depth, dropout, cond_mode):
        super().__init__()
        self.n_concepts, self.cond_mode = n_concepts, cond_mode
        in_dim = state_dim + (cond_dim if cond_mode == "concat" else 0)
        self.layers = nn.ModuleList()
        self.films = nn.ModuleList() if cond_mode == "film" else None
        d = in_dim
        for _ in range(depth):
            self.layers.append(nn.Linear(d, hidden))
            if self.films is not None:
                self.films.append(nn.Linear(cond_dim, 2 * hidden))
            d = hidden
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, n_concepts * 4)

    def forward(self, s, c):
        h = s if self.cond_mode == "film" else torch.cat([s, c], dim=-1)
        for i, lin in enumerate(self.layers):
            h = lin(h)
            if self.films is not None:
                g, b = self.films[i](c).chunk(2, dim=-1)
                h = g * h + b
            h = self.drop(self.act(h))
        return self.head(h).view(-1, self.n_concepts, 4)


def _masked_ce(logits, y, m, w=None):
    ce = nn.functional.cross_entropy(logits.reshape(-1, 4), y.reshape(-1), weight=w, reduction="none").view_as(y.float())
    return (ce * m).sum() / m.sum().clamp(min=1.0)


class TransitionModel:
    """Train/predict wrapper. All arrays are pooled across ns; the caller applies any per-ns state-column
    masking BEFORE pooling and projects predictions back to per-ns concepts AFTER."""

    def __init__(self, state_dim, cond_dim, n_concepts, n_delta_buckets, config, device=None):
        cfg, err = validate_config(config)
        assert err is None, err
        self.cfg = cfg
        self.state_dim, self.cond_dim, self.n_concepts = state_dim, cond_dim, n_concepts
        self.zero_cond = zero_cond_vector(cond_dim, n_delta_buckets)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(SEED)
        self.net = TransitionNet(state_dim, cond_dim, n_concepts,
                                 cfg["hidden"], cfg["depth"], cfg["dropout"], cfg["cond_mode"]).to(self.device)

    def _selfsup(self, S, rng, row_keep=None):
        r = self.cfg["selfsup_ratio"]
        if r <= 0:
            return None
        k = int(round(r * len(S)))
        sel = rng.choice(len(S), size=k, replace=k > len(S))
        S_ss = S[sel]
        C_ss = np.tile(self.zero_cond, (k, 1))
        Y_ss = state_to_labels(S_ss, self.n_concepts)
        # identity supervision is only defined on concepts VISIBLE in the (possibly masked) input row:
        # a masked concept reads as all-zero, so its identity label would be a spurious 'not_mentioned'
        # conflicting with the real samples' supervision.
        M_ss = np.ones((k, self.n_concepts), dtype=np.float32)
        if row_keep is not None:
            M_ss = M_ss * row_keep[sel]
        return S_ss, C_ss, Y_ss, M_ss

    def fit(self, S, C, Y4, M, Sv, Cv, Y4v, Mv, balanced=False, concept_keep=None, selfsup_row_keep=None,
            verbose=False):
        rng = np.random.default_rng(SEED)
        ss = self._selfsup(S, rng, selfsup_row_keep)
        if ss is not None:
            S = np.vstack([S, ss[0]]); C = np.vstack([C, ss[1]])
            Y4 = np.vstack([Y4, ss[2]]); M = np.vstack([M, ss[3]])
        # 'state'-mode concept subset: no supervision outside the subset (train, selfsup, and val alike)
        if concept_keep is not None:
            M = M * concept_keep[None, :]; Mv = Mv * concept_keep[None, :]
        # balanced recipe = inverse state-frequency class weights on the TRAIN loss only
        # (val_ce stays unweighted so early stopping / val_ce are comparable across configs)
        w_tr = None
        if balanced:
            cnt = np.bincount(Y4[M > 0].astype(np.int64).ravel(), minlength=4).astype(np.float64)
            w_np = cnt.sum() / (4.0 * np.maximum(cnt, 1.0))
            w_tr = torch.as_tensor(w_np / w_np.mean(), dtype=torch.float32, device=self.device)
        # z-score standardization from the (selfsup-augmented) train pool, matching the sklearn arms' _prep
        self.mu_s, self.sd_s = S.mean(0), S.std(0) + 1e-8
        self.mu_c, self.sd_c = C.mean(0), C.std(0) + 1e-8
        S = (S - self.mu_s) / self.sd_s; Sv = (Sv - self.mu_s) / self.sd_s
        C = (C - self.mu_c) / self.sd_c; Cv = (Cv - self.mu_c) / self.sd_c
        t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=self.device)
        S, C, Y4, M = t(S), t(C), t(Y4, torch.long), t(M)
        Sv, Cv, Y4v, Mv = t(Sv), t(Cv), t(Y4v, torch.long), t(Mv)
        opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        best, best_state, bad = float("inf"), None, 0
        n = len(S)
        for ep in range(EPOCHS):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, BATCH):
                b = perm[i:i + BATCH]
                opt.zero_grad()
                loss = _masked_ce(self.net(S[b], C[b]), Y4[b], M[b], w_tr)
                loss.backward(); opt.step()
                tot += float(loss) * len(b)
            self.net.eval()
            with torch.no_grad():
                vl = float(_masked_ce(self.net(Sv, Cv), Y4v, Mv))
            if verbose:
                print(f"ep{ep} train_ce={tot / n:.4f} val_ce={vl:.4f}")
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.val_ce = best
        return self

    @torch.no_grad()
    def proba_pos(self, S, C):
        """P(positive concept) = P(present) + P(uncertain) per union concept, shape (N, n_concepts)."""
        self.net.eval()
        S = (S - self.mu_s) / self.sd_s; C = (C - self.mu_c) / self.sd_c
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        out = []
        for i in range(0, len(S), 4096):
            p = torch.softmax(self.net(t(S[i:i + 4096]), t(C[i:i + 4096])), dim=-1)
            out.append((p[..., 0] + p[..., 2]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.n_concepts), dtype=np.float32)

    @torch.no_grad()
    def predict_state(self, S, C):
        """Full four-state argmax prediction, shape (N, n_concepts), values index FOURSTATE."""
        self.net.eval()
        S = (S - self.mu_s) / self.sd_s; C = (C - self.mu_c) / self.sd_c
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        out = []
        for i in range(0, len(S), 4096):
            out.append(self.net(t(S[i:i + 4096]), t(C[i:i + 4096])).argmax(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.n_concepts), dtype=np.int64)


class PooledSklearnModel:
    """FULL-STATE sklearn arm: ONE binary (positive=present|uncertain) classifier per union concept, trained
    on rows pooled across ALL namespaces where that concept is label-masked observed -- the state-predictor
    view (predict the whole state after Delta), not the per-ns query-conditioned ladder. Same fit/proba_pos
    interface as TransitionModel so the env's pooled pathway (feature masking, per-row label masks, selfsup,
    threshold calibration, readout projection) applies unchanged. mlp-specific knobs are ignored."""

    MIN_POS = 5

    def __init__(self, state_dim, cond_dim, n_concepts, n_delta_buckets, config, n_jobs=2):
        cfg, err = validate_config(config)
        assert err is None, err
        self.cfg = cfg; self.n_concepts = n_concepts; self.n_jobs = n_jobs
        self.zero_cond = zero_cond_vector(cond_dim, n_delta_buckets)
        self.models = {}; self.val_ce = float("nan")

    def _mk(self, balanced):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        cw = "balanced" if balanced else None
        l = self.cfg["learner"]
        if l == "logreg":
            return LogisticRegression(max_iter=200, class_weight=cw, solver="liblinear", random_state=SEED)
        if l == "rf":
            return RandomForestClassifier(n_estimators=150, class_weight=cw, n_jobs=self.n_jobs, random_state=SEED)
        return XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, n_jobs=self.n_jobs, verbosity=0,
                             random_state=SEED)

    def _selfsup(self, S, rng, row_keep=None):
        r = self.cfg["selfsup_ratio"]
        if r <= 0:
            return None
        k = int(round(r * len(S)))
        sel = rng.choice(len(S), size=k, replace=k > len(S))
        S_ss = S[sel]
        C_ss = np.tile(self.zero_cond, (k, 1))
        Y_ss = state_to_labels(S_ss, self.n_concepts)
        M_ss = np.ones((k, self.n_concepts), dtype=np.float32)
        if row_keep is not None:
            M_ss = M_ss * row_keep[sel]
        return S_ss, C_ss, Y_ss, M_ss

    def fit(self, S, C, Y4, M, Sv, Cv, Y4v, Mv, balanced=False, concept_keep=None, selfsup_row_keep=None,
            verbose=False):
        rng = np.random.default_rng(SEED)
        ss = self._selfsup(S, rng, selfsup_row_keep)
        if ss is not None:
            S = np.vstack([S, ss[0]]); C = np.vstack([C, ss[1]])
            Y4 = np.vstack([Y4, ss[2]]); M = np.vstack([M, ss[3]])
        if concept_keep is not None:
            M = M * concept_keep[None, :]; Mv = Mv * concept_keep[None, :]
        X = np.hstack([S, C]); Xv = np.hstack([Sv, Cv])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        X = (X - self.mu) / self.sd; Xv = (Xv - self.mu) / self.sd
        Ypos = np.isin(Y4, (0, 2)).astype(int); Yvpos = np.isin(Y4v, (0, 2)).astype(int)
        ces = []
        for j in range(self.n_concepts):
            rows = M[:, j] > 0
            y = Ypos[rows, j]
            if rows.sum() < 2 * self.MIN_POS or y.sum() < self.MIN_POS or y.sum() == len(y):
                continue
            clf = self._mk(balanced)
            try:
                clf.fit(X[rows], y)
                self.models[j] = clf
            except Exception:
                continue
            vr = Mv[:, j] > 0
            if vr.any():
                p = np.clip(clf.predict_proba(Xv[vr])[:, 1], 1e-6, 1 - 1e-6)
                yv = Yvpos[vr, j]
                ces.append(float(-(yv * np.log(p) + (1 - yv) * np.log(1 - p)).mean()))
        self.val_ce = float(np.mean(ces)) if ces else float("nan")
        return self

    def proba_pos(self, S, C):
        X = (np.hstack([S, C]) - self.mu) / self.sd
        out = np.zeros((len(X), self.n_concepts), dtype=np.float32)
        for j, clf in self.models.items():
            out[:, j] = clf.predict_proba(X)[:, 1]
        return out


def make_model(state_dim, cond_dim, n_concepts, n_delta_buckets, config, n_jobs=2, device=None):
    """Factory over the 'learner' knob: 'mlp' -> torch TransitionModel; else pooled per-concept sklearn."""
    cfg, err = validate_config(config)
    assert err is None, err
    if cfg["learner"] == "mlp":
        return TransitionModel(state_dim, cond_dim, n_concepts, n_delta_buckets, cfg, device=device)
    return PooledSklearnModel(state_dim, cond_dim, n_concepts, n_delta_buckets, cfg, n_jobs=n_jobs)
