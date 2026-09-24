"""
Four-state structured featurizer via cohort_rad_findings JOIN (no string matching, no new API).

For each case, the prior imaging state is looked up from cohort_rad_findings.jsonl (the already-done mini
four-state extraction) by hadm+charttime:
  prior reports = findings rows with hadm_id in {case hadm} U prior_related_hadms, charttime < target time.
Aggregate per concept the MOST-RECENT prior presence (present / absent / uncertain; not_mentioned = never
stated) -> clean four-state finding-persistence features + region-imaged flag. Optionally adds labs (parsed
numeric + miss) and/or bag-of-words. Compares xgb/logreg state-direct on the same 1419 vs bow-only (0.431)
and string-structured (both 0.460).

Anti-leakage: cohort join uses only PRE-target reports; models fit on the keyed train pool only. Pure CPU.
"""
import argparse, ast, collections, json, os, re, sys
import numpy as np

sys.path.insert(0, "${VP_ROOT}/analysis")
import precompute_agent_ml_data_from_sft as PC   # noqa: E402
import run_direct_vp_report_smoke as R           # noqa: E402
import score_ml_structured as S                  # noqa: E402  (parse_context for labs)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

NJOBS = int(os.environ.get("VP_AGENT_NJOBS", "2"))
ROOT = "${VP_ROOT}"
POS = {"present", "uncertain"}
FIND = f"{ROOT}/analysis/cohort_rad_findings.jsonl"


def concept_name(attr):
    return str(attr).split(".", 2)[-1]


def region_of(ns):
    tail = ns.split(".")[-1]
    return tail.split("_", 1)[1] if "_" in tail else tail


ICD_RE = re.compile(r"ICD-\d+:\s*([A-Za-z0-9.]+)")
DELTA_BUCKETS = [1, 3, 7, 30, 90, 365]   # days


def parse_interventions(text):
    """Parse pre-target interventions from context: medication names, ICD procedure codes, ED presence."""
    meds, procs = set(), set()
    has_ed = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line.startswith("Medications:"):
            meds.update(x.strip() for x in line[len("Medications:"):].split(",") if x.strip())
        elif line.startswith("ED medication reconciliation") or line.startswith("ED medications dispensed"):
            meds.update(x.strip() for x in line.split(":", 1)[-1].split(",") if x.strip())
        elif line.startswith("Procedures/interventions:"):
            procs.update(ICD_RE.findall(line))
        if line.lower().startswith("ed ") or "ed information before target" in line.lower():
            has_ed = True
    return {"meds": meds, "procs": procs, "has_ed": has_ed}


def gap_days(row):
    """Most-recent prior gap in days. prior_gap_days is a LIST (one gap per prior hadm); min = latest prior.
    The original float() cast raised TypeError on the list and silently returned 0.0 for every row, so the
    earlier delta sweep effectively fed constant-zero Delta features."""
    g = row.get("prior_gap_days")
    if isinstance(g, list):
        vals = [float(x) for x in g if x is not None]
        return min(vals) if vals else 0.0
    try:
        return float(g or 0.0)
    except (TypeError, ValueError):
        return 0.0


def delta_features(row):
    """Time-gap (Delta) features from prior_gap_days: log value + coarse one-hot buckets."""
    g = gap_days(row)
    feats = [np.log1p(max(g, 0.0)) / 6.0, 1.0 if g <= 0 else 0.0]   # ~normalized log + is-zero flag
    for b in DELTA_BUCKETS:
        feats.append(1.0 if g <= b else 0.0)
    return feats


def load_findings_index():
    """hadm_id -> list of (charttime, ns, attrs_dict) sorted by charttime."""
    idx = collections.defaultdict(list)
    with open(FIND) as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = str(o.get("hadm_id") or "")
            raw = o.get("attrs")
            if not h or not raw:
                continue
            try:
                attrs = ast.literal_eval(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
            idx[h].append((o.get("charttime") or "", o.get("ns") or "", attrs))
    for h in idx:
        idx[h].sort(key=lambda e: e[0])
    return idx


def prior_state(row, idx):
    """Aggregate most-recent prior presence per bare concept + regions imaged, before target time."""
    target_t = row.get("charttime") or ""
    hadms = {str(row.get("hadm_id"))} | {str(x) for x in (row.get("prior_related_hadms") or [])}
    reports = []
    for h in hadms:
        for ct, ns, attrs in idx.get(h, []):
            if ct and ct < target_t:
                reports.append((ct, ns, attrs))
    reports.sort(key=lambda e: e[0])
    status = {}; regions = set()
    for ct, ns, attrs in reports:
        regions.add(region_of(ns))
        for k, v in (attrs or {}).items():
            status[concept_name(k)] = (v or {}).get("presence", "present")
    return status, regions, len(reports)


def featurize(row, ns, feat_concepts, concepts, idx, groups, lab_vocab, lab_stats, fidx, med_vocab=None, proc_vocab=None):
    status, regions, n_reports = prior_state(row, idx)
    imaged = 1.0 if region_of(ns) in regions else 0.0
    feats = []
    if "prior4state" in groups:
        for c in feat_concepts:
            s = status.get(c, "not_mentioned")
            feats += [float(s == "present"), float(s == "absent"), float(s == "uncertain")]
        feats += [imaged, float(n_reports), float(len(status))]
    if "delta" in groups:
        feats += delta_features(row)
    if "interventions" in groups:
        iv = parse_interventions(row.get("context", ""))
        for m in (med_vocab or []):
            feats.append(1.0 if m in iv["meds"] else 0.0)
        for p in (proc_vocab or []):
            feats.append(1.0 if p in iv["procs"] else 0.0)
        feats += [float(len(iv["meds"])), float(len(iv["procs"])), 1.0 if iv["has_ed"] else 0.0]
    if "labs" in groups:
        labs = S.parse_context(row.get("context", ""))["labs"]
        for name in lab_vocab:
            if name in labs:
                mu, sd = lab_stats[name]; feats += [(labs[name] - mu) / sd, 0.0]
            else:
                feats += [0.0, 1.0]
    if "bow" in groups:
        bow = np.zeros(len(fidx), dtype=np.float32)
        for t in set(PC.tokens(row.get("context", ""))):
            if t in fidx:
                bow[fidx[t]] = 1.0
        feats += bow.tolist()
    return np.array(feats, dtype=np.float32)


def _f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    P = tp / (tp + fp) if tp + fp else 0; Rc = tp / (tp + fn) if tp + fn else 0
    return 2 * P * Rc / (P + Rc) if P + Rc else 0.0


def _mk(method, cw):
    return {"logreg": LogisticRegression(max_iter=200, class_weight=cw, solver="liblinear"),
            "rf": RandomForestClassifier(n_estimators=100, class_weight=cw, n_jobs=NJOBS),
            "xgb": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, n_jobs=NJOBS, verbosity=0)}[method]


def load_keyed(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-keyed", default=f"{ROOT}/data/temporal_episode_keyed_query_train_keyed.jsonl")
    ap.add_argument("--test", default=f"{ROOT}/data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl")
    ap.add_argument("--schema", default=f"{ROOT}/analysis/eval_schema_auto_train.json")
    ap.add_argument("--groups", default="prior4state", help="comma of prior4state,labs,bow")
    ap.add_argument("--pool", default="ns", choices=["ns", "union"], help="ns=target-ns concepts only; union=all schema concepts")
    ap.add_argument("--methods", default="logreg,rf,xgb")
    ap.add_argument("--recipe", default="default", choices=["default", "balanced"])
    ap.add_argument("--lab-vocab", type=int, default=80)
    ap.add_argument("--med-vocab", type=int, default=60)
    ap.add_argument("--proc-vocab", type=int, default=60)
    ap.add_argument("--bow-features", type=int, default=600)
    ap.add_argument("--out", default=f"{ROOT}/analysis/ml_state_direct_fourstate.json")
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    train_rows = load_keyed(args.train_keyed)
    te_rows = [json.loads(l) for l in open(args.test) if l.strip()]
    vocab = R.build_vocab_from_schema(args.schema)
    union_concepts = sorted({c for cs in vocab.values() for c in cs})
    print(f"train_keyed={len(train_rows)} test={len(te_rows)} groups={groups} pool={args.pool} union_concepts={len(union_concepts)}")
    idx = load_findings_index()
    print(f"findings index hadms={len(idx)}")

    lab_vocab, lab_stats = [], {}
    if "labs" in groups:
        freq = collections.Counter()
        for r in train_rows:
            for name in S.parse_context(r.get("context", ""))["labs"]:
                freq[name] += 1
        lab_vocab = [n for n, _ in freq.most_common(args.lab_vocab)]
        vals = collections.defaultdict(list)
        for r in train_rows:
            for n, v in S.parse_context(r.get("context", ""))["labs"].items():
                if n in lab_vocab:
                    vals[n].append(v)
        lab_stats = {n: (float(np.mean(vals[n])), float(np.std(vals[n]) + 1e-6)) for n in lab_vocab if vals[n]}
        lab_vocab = [n for n in lab_vocab if n in lab_stats]
    fidx = None
    if "bow" in groups:
        tok = collections.Counter()
        for r in train_rows:
            tok.update(set(PC.tokens(r.get("context", ""))))
        fidx = {w: i for i, w in enumerate([t for t, _ in tok.most_common(args.bow_features)])}
    med_vocab, proc_vocab = [], []
    if "interventions" in groups:
        mf, pf = collections.Counter(), collections.Counter()
        for r in train_rows:
            iv = parse_interventions(r.get("context", ""))
            mf.update(iv["meds"]); pf.update(iv["procs"])
        med_vocab = [m for m, _ in mf.most_common(args.med_vocab)]
        proc_vocab = [p for p, _ in pf.most_common(args.proc_vocab)]
        print(f"med_vocab={len(med_vocab)} proc_vocab={len(proc_vocab)}")

    def grp(rows):
        d = collections.defaultdict(list)
        for r in rows:
            d[r["ns"]].append(r)
        return d
    tr, te = grp(train_rows), grp(te_rows)
    cw = "balanced" if args.recipe == "balanced" else None
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    def label(row, concepts):
        bare = {k.split(".", 2)[-1]: v for k, v in row["gold"].items()}
        return np.array([1.0 if bare.get(c) in POS else 0.0 for c in concepts], dtype=np.float32)

    results = {}; te_used = None; ndim = None
    for method in methods:
        itp = ifp = ifn = 0; macros = []; used = 0
        for ns, concepts in vocab.items():
            trr, terr = tr.get(ns, []), te.get(ns, [])
            if not trr or not terr:
                continue
            feat_concepts = union_concepts if args.pool == "union" else concepts
            Xtr = np.array([featurize(r, ns, feat_concepts, concepts, idx, groups, lab_vocab, lab_stats, fidx, med_vocab, proc_vocab) for r in trr])
            Ytr = np.array([label(r, concepts) for r in trr])
            Xte = np.array([featurize(r, ns, feat_concepts, concepts, idx, groups, lab_vocab, lab_stats, fidx, med_vocab, proc_vocab) for r in terr])
            Yte = np.array([label(r, concepts) for r in terr])
            ndim = Xtr.shape[1]; used += len(terr)
            preds = np.zeros_like(Yte); f1s = []
            for k in range(Ytr.shape[1]):
                ytr, yte = Ytr[:, k], Yte[:, k]
                if ytr.sum() < 5 or ytr.sum() == len(ytr):
                    p = np.zeros(len(yte))
                else:
                    try:
                        clf = _mk(method, cw); clf.fit(Xtr, ytr); p = clf.predict(Xte)
                    except Exception:
                        p = np.zeros(len(yte))
                preds[:, k] = p; f1s.append(_f1(yte, p))
            macros.append(float(np.mean(f1s)))
            for i in range(len(Yte)):
                gp = set(np.where(Yte[i] == 1)[0]); pp = set(np.where(preds[i] == 1)[0])
                itp += len(gp & pp); ifp += len(pp - gp); ifn += len(gp - pp)
        P = itp / (itp + ifp) if itp + ifp else 0.0; Rc = itp / (itp + ifn) if itp + ifn else 0.0
        results[method] = {"micro_f1": round(2 * P * Rc / (P + Rc) if P + Rc else 0.0, 4),
                           "precision": round(P, 4), "recall": round(Rc, 4),
                           "macro_f1": round(float(np.mean(macros)) if macros else 0.0, 4)}
        te_used = used
    out = {"groups": groups, "recipe": args.recipe, "feature_dim_example": ndim,
           "lab_vocab_size": len(lab_vocab), "te_cases_scored": te_used, "results": results,
           "baselines": {"bow_only_xgb": 0.431, "string_structured_both_logreg": 0.460}}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
