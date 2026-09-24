"""
Build the CONDITIONAL-TRANSITION agent_ml_data pkl (Plan B world-model, stage 1 data).

Design (logs/2026-09-23-plan-b-no-sft-state-prediction.md §12 final locked design):
  input  = [current full state: four-state per union concept] + [conditions: Delta buckets + typed interventions]
  output = [future full state: four-state, SAME union concept space]  (conditions never in output)
Per-case supervision exists only on the target-ns concepts (observed projection); everything else is masked.

The pkl is a SUPERSET of agent_ml_data_episode_fourstate_union_stratval.pkl so env_temporal keeps working:
  data[ns]: Xtr/Ytr/Xval/Yval/Xte/Yte  (X = union prior4state block + cond block; Y = binary positive labels)
            groups: {"self","cross","cond"}  (cond = Delta+interventions columns, clinically mandatory)
  NEW per ns: Y4tr/Y4val/Y4te  int8 (N, C_union) four-state labels 0=present 1=absent 2=uncertain 3=not_mentioned
              label_mask float32 (C_union,) 1 on this ns's observed concepts
  NEW meta: state_block {n_concepts, per_concept:3, extras:3}, cond_cols, cond_names, med_vocab, proc_vocab,
            union_concepts, fourstate_values.

Uses the FIXED delta featurizer (prior_gap_days is a list; min = most-recent prior gap). Pure CPU, no API.
"""
import argparse, json, pickle, sys
import numpy as np

sys.path.insert(0, "${VP_ROOT}/analysis")
import run_direct_vp_report_smoke as R           # noqa: E402
import score_ml_fourstate as F4                   # noqa: E402

ROOT = "${VP_ROOT}"
POS = {"present", "uncertain"}
FOURSTATE = ["present", "absent", "uncertain", "not_mentioned"]
S4 = {v: i for i, v in enumerate(FOURSTATE)}
GROUPS = ["prior4state", "delta", "interventions"]


def label_bin(row, concepts):
    bare = {k.split(".", 2)[-1]: v for k, v in row["gold"].items()}
    return np.array([1.0 if bare.get(c) in POS else 0.0 for c in concepts], dtype=np.float32)


def label_four(row, union_concepts):
    """Four-state label over the union space; concepts outside the case's gold ns are masked by label_mask.
    Missing from gold within the observed ns = not_mentioned (extractor read the whole report).
    Rare non-standard gold values (e.g. 'unchanged', 1 instance in train) -> not_mentioned."""
    bare = {k.split(".", 2)[-1]: v for k, v in row["gold"].items()}
    return np.array([S4.get(bare.get(c, "not_mentioned"), S4["not_mentioned"]) for c in union_concepts],
                    dtype=np.int8)


def _pt(t):
    from datetime import datetime
    try:
        return datetime.fromisoformat(t) if t else None
    except ValueError:
        return None


def label_four_dense(row, union_concepts, vocab, idx, window_days, mask_silent_definite, prior_status):
    """Dense TRAIN label (GT-noise fix, 2026-09-24 log §14):
    - dense union: findings extracted from ANY of the patient's reports within +/-window_days of target time
      (farthest applied first, so the nearest mention of a concept wins; the target report's own gold wins last).
    - per-case mask: concepts mentioned by any window report, plus the target ns's schema concepts.
    - mask_silent_definite: entries with a DEFINITE prior input state (present/absent/uncertain) that are still
      not_mentioned after the union are UNSUPERVISED (silence is reporting behavior ~51% of the time even at
      Delta<=7d, not a state transition -- see flicker diagnosis).
    Returns (y4 int8 (C,), mask float32 (C,))."""
    y = {}
    t = _pt(row.get("charttime"))
    if window_days > 0 and t is not None:
        hadms = {str(row.get("hadm_id"))} | {str(x) for x in (row.get("prior_related_hadms") or [])}
        wins = []
        for h in hadms:
            for ct, ns_r, attrs in idx.get(h, []):
                c = _pt(ct)
                if c is not None and abs((c - t).total_seconds()) <= window_days * 86400:
                    wins.append((abs((c - t).total_seconds()), attrs))
        wins.sort(key=lambda e: -e[0])          # farthest first; nearer overwrites
        for _, attrs in wins:
            for k, v in (attrs or {}).items():
                y[F4.concept_name(k)] = (v or {}).get("presence", "present")
    bare = {k.split(".", 2)[-1]: v for k, v in row["gold"].items()}
    y.update(bare)                               # target report gold has final say
    y4 = np.array([S4.get(y.get(c, "not_mentioned"), S4["not_mentioned"]) for c in union_concepts], dtype=np.int8)
    ns_set = set(vocab.get(row["ns"], []))
    mask = np.array([1.0 if (c in y or c in ns_set) else 0.0 for c in union_concepts], dtype=np.float32)
    if mask_silent_definite:
        for j, c in enumerate(union_concepts):
            if mask[j] and y4[j] == S4["not_mentioned"] and prior_status.get(c) in ("present", "absent", "uncertain"):
                mask[j] = 0.0
    return y4, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-keyed", default=f"{ROOT}/data/temporal_episode_keyed_query_train_keyed.jsonl")
    ap.add_argument("--test", default=f"{ROOT}/data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl")
    ap.add_argument("--schema", default=f"{ROOT}/analysis/eval_schema_auto_train.json")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-ns", type=int, default=13)
    ap.add_argument("--med-vocab", type=int, default=60)
    ap.add_argument("--proc-vocab", type=int, default=60)
    ap.add_argument("--dense-window", type=float, default=0.0,
                    help="days; >0 = dense TRAIN labels from all reports within +/-window of target time")
    ap.add_argument("--mask-silent-definite", action="store_true",
                    help="unsupervise TRAIN entries with definite prior state but not_mentioned label")
    ap.add_argument("--out", default=f"{ROOT}/analysis/agent_ml_data_episode_condtrans_union_stratval.pkl")
    args = ap.parse_args()

    train_rows = [json.loads(l) for l in open(args.train_keyed) if l.strip()]
    te_rows = [json.loads(l) for l in open(args.test) if l.strip()]
    vocab = R.build_vocab_from_schema(args.schema)
    union_concepts = sorted({c for cs in vocab.values() for c in cs})
    idx = F4.load_findings_index()
    print(f"train={len(train_rows)} test={len(te_rows)} findings_hadms={len(idx)} union_concepts={len(union_concepts)}")

    # intervention vocab from the train pool only (never test)
    import collections
    mf, pf = collections.Counter(), collections.Counter()
    for r in train_rows:
        iv = F4.parse_interventions(r.get("context", ""))
        mf.update(iv["meds"]); pf.update(iv["procs"])
    med_vocab = [m for m, _ in mf.most_common(args.med_vocab)]
    proc_vocab = [p for p, _ in pf.most_common(args.proc_vocab)]
    print(f"med_vocab={len(med_vocab)} proc_vocab={len(proc_vocab)}")

    def grp(rows):
        d = {}
        for i, r in enumerate(rows):
            d.setdefault(r["ns"], []).append((i, r))
        return d
    tr_by_ns = grp(train_rows)
    # proportional namespace-stratified internal val (same protocol as the sealed test draw; val never peeks at test)
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
    ns_list = [ns for ns, _ in sorted(tr_by_ns.items(), key=lambda x: -len(x[1]))
               if vocab.get(ns) and te_by_ns.get(ns)][: args.max_ns]

    def feat(row, ns):
        return F4.featurize(row, ns, union_concepts, vocab[ns], idx, GROUPS, [], {}, None, med_vocab, proc_vocab)

    # column layout: state block (3 per concept + 3 extras) then cond block (delta + interventions)
    n_state = 3 * len(union_concepts) + 3
    probe = feat(train_rows[0], train_rows[0]["ns"])
    n_delta = 2 + len(F4.DELTA_BUCKETS)
    n_interv = len(med_vocab) + len(proc_vocab) + 3
    assert len(probe) == n_state + n_delta + n_interv, (len(probe), n_state, n_delta, n_interv)
    cond_cols = list(range(n_state, n_state + n_delta + n_interv))
    cond_names = (["delta_log_gap", "delta_is_zero"] + [f"delta_le_{b}d" for b in F4.DELTA_BUCKETS]
                  + [f"med:{m}" for m in med_vocab] + [f"proc:{p}" for p in proc_vocab]
                  + ["n_meds", "n_procs", "has_ed"])

    data, ns_vocab, te_keys, groups_by_ns = {}, {}, {}, {}
    dense = args.dense_window > 0 or args.mask_silent_definite
    dstat = {"entries": 0, "masked_silent": 0, "dense_upgraded": 0}
    for ns in ns_list:
        concepts = vocab[ns]
        ns_vocab[ns] = concepts
        ns_static_mask = np.array([1.0 if c in set(concepts) else 0.0 for c in union_concepts], dtype=np.float32)
        buf = {"tr": ([], [], [], []), "val": ([], [], [], [])}   # X, Ybin, Y4, LM4
        for i, r in tr_by_ns[ns]:
            is_val = i in val_ids
            X, Yb, Y4, LM = buf["val"] if is_val else buf["tr"]
            X.append(feat(r, ns)); Yb.append(label_bin(r, concepts))
            if dense and not is_val:
                # TRAIN rows: dense state labels; VAL stays single-report mention-aligned (same protocol as sealed)
                prior_status = F4.prior_state(r, idx)[0] if args.mask_silent_definite else {}
                y4_sparse = label_four(r, union_concepts)
                y4, lm = label_four_dense(r, union_concepts, vocab, idx, args.dense_window,
                                          args.mask_silent_definite, prior_status)
                dstat["entries"] += int(lm.sum())
                dstat["masked_silent"] += int(((ns_static_mask > 0) & (lm == 0)).sum())
                dstat["dense_upgraded"] += int(((y4 != 3) & (y4_sparse == 3) & (lm > 0)).sum())
                Y4.append(y4); LM.append(lm)
            else:
                Y4.append(label_four(r, union_concepts)); LM.append(ns_static_mask.copy())
        Xte = [feat(r, ns) for r in te_by_ns[ns]]
        Ybte = [label_bin(r, concepts) for r in te_by_ns[ns]]
        Y4te = [label_four(r, union_concepts) for r in te_by_ns[ns]]
        te_keys[ns] = [[r.get("hadm_id"), r.get("charttime"), r.get("ns"), r.get("exam_name")] for r in te_by_ns[ns]]

        selfset = set(concepts); self_cols = []
        for ci, c in enumerate(union_concepts):
            if c in selfset:
                self_cols += [3 * ci, 3 * ci + 1, 3 * ci + 2]
        self_cols += [3 * len(union_concepts), 3 * len(union_concepts) + 1, 3 * len(union_concepts) + 2]
        cross_cols = [c for c in range(n_state) if c not in set(self_cols)]
        groups_by_ns[ns] = {"self": self_cols, "cross": cross_cols, "cond": cond_cols}

        mask = np.array([1.0 if c in selfset else 0.0 for c in union_concepts], dtype=np.float32)
        data[ns] = {
            "Xtr": np.array(buf["tr"][0], dtype=np.float32), "Ytr": np.array(buf["tr"][1], dtype=np.float32),
            "Xval": np.array(buf["val"][0], dtype=np.float32), "Yval": np.array(buf["val"][1], dtype=np.float32),
            "Xte": np.array(Xte, dtype=np.float32), "Yte": np.array(Ybte, dtype=np.float32),
            "Y4tr": np.array(buf["tr"][2], dtype=np.int8), "Y4val": np.array(buf["val"][2], dtype=np.int8),
            "Y4te": np.array(Y4te, dtype=np.int8), "label_mask": mask, "groups": groups_by_ns[ns],
        }
        if dense:
            data[ns]["LM4tr"] = np.array(buf["tr"][3], dtype=np.float32)
            data[ns]["LM4val"] = np.array(buf["val"][3], dtype=np.float32)

    fdim = data[ns_list[0]]["Xtr"].shape[1]
    obj = {"data": data, "ns_vocab": ns_vocab, "top_ns": ns_list, "fdim": fdim, "pool": "union",
           "feature_vocab": [f"condtrans_union"], "te_keys": te_keys, "groups_by_ns": groups_by_ns,
           "meta": {"union_concepts": union_concepts, "fourstate_values": FOURSTATE,
                    "state_block": {"n_concepts": len(union_concepts), "per_concept": 3, "extras": 3},
                    "cond_cols": cond_cols, "cond_names": cond_names,
                    "med_vocab": med_vocab, "proc_vocab": proc_vocab,
                    "delta_buckets": F4.DELTA_BUCKETS,
                    "dense_labels": ({"window_days": args.dense_window,
                                      "mask_silent_definite": bool(args.mask_silent_definite)} if dense else None)}}
    pickle.dump(obj, open(args.out, "wb"))

    # sanity: cond features must be non-degenerate (the old sweep's delta was silently all-zero)
    Xall = np.vstack([data[ns]["Xtr"] for ns in ns_list])
    cond = Xall[:, cond_cols]
    nz = (np.abs(cond) > 0).mean(0)
    print(json.dumps({"out": args.out, "top_ns": ns_list, "fdim": fdim,
                      "n_state": n_state, "n_cond": len(cond_cols),
                      "dense": ({"window_days": args.dense_window, "mask_silent_definite": bool(args.mask_silent_definite),
                                 **dstat} if dense else None),
                      "cond_nonzero_rate": {"delta_log_gap": round(float(nz[0]), 3),
                                            "delta_buckets_mean": round(float(nz[2:2 + len(F4.DELTA_BUCKETS)].mean()), 3),
                                            "meds_mean": round(float(nz[n_delta:n_delta + len(med_vocab)].mean()), 3),
                                            "has_ed": round(float(nz[-1]), 3)},
                      "sizes": {ns: {"tr": int(len(data[ns]["Xtr"])), "val": int(len(data[ns]["Xval"])),
                                     "te": int(len(data[ns]["Xte"]))} for ns in ns_list}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
