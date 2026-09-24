"""
Build a four-state (finding-persistence) agent_ml_data pkl for the auto-research env (TemporalEnv).

Replaces the bag-of-words features with the clean target-ns four-state features (the 0.4575 winner):
per target-ns concept -> prior present/absent/uncertain (from cohort_rad_findings join, charttime<target) +
region-imaged flag. Output = same per-ns concept labels. Includes te_keys aligned to the fixed 1419 subset.

Pure CPU, no API. Consumed by env_temporal via VP_AGENT_ML_DATA.
"""
import argparse, json, os, pickle, sys
import numpy as np

sys.path.insert(0, "${VP_ROOT}/analysis")
import run_direct_vp_report_smoke as R           # noqa: E402
import score_ml_fourstate as F4                   # noqa: E402  (prior_state / featurize / index)

ROOT = "${VP_ROOT}"
POS = {"present", "uncertain"}


def label(row, concepts):
    bare = {k.split(".", 2)[-1]: v for k, v in row["gold"].items()}
    return np.array([1.0 if bare.get(c) in POS else 0.0 for c in concepts], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-keyed", default=f"{ROOT}/data/temporal_episode_keyed_query_train_keyed.jsonl")
    ap.add_argument("--test", default=f"{ROOT}/data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl")
    ap.add_argument("--schema", default=f"{ROOT}/analysis/eval_schema_auto_train.json")
    ap.add_argument("--pool", default="ns", choices=["ns", "union"])
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-ns", type=int, default=13)
    ap.add_argument("--out", default=f"{ROOT}/analysis/agent_ml_data_episode_fourstate.pkl")
    args = ap.parse_args()

    train_rows = [json.loads(l) for l in open(args.train_keyed) if l.strip()]
    te_rows = [json.loads(l) for l in open(args.test) if l.strip()]
    vocab = R.build_vocab_from_schema(args.schema)
    union_concepts = sorted({c for cs in vocab.values() for c in cs})
    idx = F4.load_findings_index()
    print(f"train={len(train_rows)} test={len(te_rows)} findings_hadms={len(idx)} pool={args.pool}")

    def grp(rows):
        d = {}
        for i, r in enumerate(rows):
            d.setdefault(r["ns"], []).append((i, r))
        return d
    tr_by_ns = grp(train_rows)
    # proportional namespace-stratified internal val: within each ns take val_frac as val, rest as train.
    # This makes val follow the same proportional-by-ns protocol used to draw the sealed test; val and test
    # independently follow the same sampling protocol (val never peeks at test).
    rng = np.random.default_rng(0)
    val_ids = set()
    for ns in sorted(tr_by_ns):
        idxs = [i for i, _ in tr_by_ns[ns]]
        perm = rng.permutation(len(idxs))
        k = max(1, int(round(args.val_frac * len(idxs)))) if len(idxs) > 1 else 0
        for j in perm[:k]:
            val_ids.add(idxs[j])
    te_by_ns = {}
    for r in te_rows:
        te_by_ns.setdefault(r["ns"], []).append(r)

    # namespaces present in both, ordered by train count
    ns_list = [ns for ns, _ in sorted(tr_by_ns.items(), key=lambda x: -len(x[1]))
               if vocab.get(ns) and te_by_ns.get(ns)][: args.max_ns]

    def feat(row, ns, concepts):
        fc = union_concepts if args.pool == "union" else concepts
        return F4.featurize(row, ns, fc, concepts, idx, ["prior4state"], [], {}, None)

    data, ns_vocab, te_keys = {}, {}, {}
    groups_by_ns = {}
    for ns in ns_list:
        concepts = vocab[ns]
        ns_vocab[ns] = concepts
        Xtr, Ytr, Xval, Yval = [], [], [], []
        for i, r in tr_by_ns[ns]:
            x = feat(r, ns, concepts); y = label(r, concepts)
            if i in val_ids:
                Xval.append(x); Yval.append(y)
            else:
                Xtr.append(x); Ytr.append(y)
        Xte = [feat(r, ns, concepts) for r in te_by_ns[ns]]
        Yte = [label(r, concepts) for r in te_by_ns[ns]]
        te_keys[ns] = [[r.get("hadm_id"), r.get("charttime"), r.get("ns"), r.get("exam_name")] for r in te_by_ns[ns]]
        data[ns] = {"Xtr": np.array(Xtr, dtype=np.float32), "Ytr": np.array(Ytr, dtype=np.float32),
                    "Xval": np.array(Xval, dtype=np.float32), "Yval": np.array(Yval, dtype=np.float32),
                    "Xte": np.array(Xte, dtype=np.float32), "Yte": np.array(Yte, dtype=np.float32)}
        # feature-group column split: self = this ns's own concepts (+shared extras), cross = other pool concepts
        dim = data[ns]["Xtr"].shape[1] if len(Xtr) else (data[ns]["Xte"].shape[1] if len(Xte) else 0)
        if args.pool == "union":
            selfset = set(concepts); self_cols = []
            for i, c in enumerate(union_concepts):
                if c in selfset:
                    self_cols += [3 * i, 3 * i + 1, 3 * i + 2]
            extras = [3 * len(union_concepts), 3 * len(union_concepts) + 1, 3 * len(union_concepts) + 2]
            self_cols = self_cols + extras
            cross_cols = [c for c in range(dim) if c not in set(self_cols)]
        else:
            self_cols, cross_cols = list(range(dim)), []
        groups_by_ns[ns] = {"self": self_cols, "cross": cross_cols}
        data[ns]["groups"] = groups_by_ns[ns]

    fdim = data[ns_list[0]]["Xtr"].shape[1] if ns_list else 0
    obj = {"data": data, "ns_vocab": ns_vocab, "top_ns": ns_list, "fdim": fdim, "pool": args.pool,
           "feature_vocab": [f"fourstate_pool={args.pool}"], "te_keys": te_keys, "groups_by_ns": groups_by_ns}
    pickle.dump(obj, open(args.out, "wb"))
    print(json.dumps({"out": args.out, "top_ns": ns_list, "fdim_example": fdim,
                      "sizes": {ns: {k: int(len(v)) for k, v in data[ns].items()} for ns in ns_list}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
