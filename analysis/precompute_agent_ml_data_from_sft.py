"""
Precompute lightweight text-feature matrices for TemporalEnv from SFT/test JSON.

Unlike precompute_ml_data.py, this consumes an explicit ShareGPT SFT train file
and JSONL test file. It is used for episode-aware agent cheap arms, where the
input context is already composed by build_temporal_episode_dataset.py.

The feature and target vocabularies are built from train only. The test file is
encoded only after vocab construction so the cheap-arm sealed split does not
leak labels into training-time choices.
"""
import argparse
import collections
import json
import os
import pickle
import re

import numpy as np


POS = {"present", "uncertain"}
TOKEN_RE = re.compile(r"[a-z_][a-z_0-9]{2,}", re.I)
NS_RE = re.compile(r"[\[(](rad\.[a-z_]+)[\])]" )


def load_train(path):
    rows = []
    for rec in json.load(open(path)):
        try:
            user = rec["messages"][1]["content"]
            gold = json.loads(rec["messages"][2]["content"])["findings"]
        except Exception:
            continue
        m = NS_RE.search(user)
        if not m:
            continue
        rows.append({"ns": m.group(1), "text": user, "gold": gold})
    return rows


def load_test(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("ns") and rec.get("gold"):
                rows.append({"ns": rec["ns"],
                             "text": rec.get("context", "") + "\n" + rec.get("exam_name", ""),
                             "gold": rec["gold"],
                             "key": [rec.get("hadm_id"), rec.get("charttime"), rec.get("ns"), rec.get("exam_name")],
                             "subject": rec.get("subject")})
    return rows


def concept_name(attr):
    return str(attr).split(".", 2)[-1]


def tokens(text):
    return TOKEN_RE.findall((text or "").lower())


def build_vocabs(train_rows, max_features, max_targets, min_ns_train, max_ns):
    tok_freq = collections.Counter()
    ns_target_freq = collections.defaultdict(collections.Counter)
    ns_count = collections.Counter()
    for row in train_rows:
        ns = row["ns"]
        ns_count[ns] += 1
        tok_freq.update(set(tokens(row["text"])))
        for attr, presence in row["gold"].items():
            c = concept_name(attr)
            if presence in POS:
                ns_target_freq[ns][c] += 1
    feature_vocab = [t for t, _ in tok_freq.most_common(max_features)]
    ns_vocab = {}
    for ns, n in ns_count.most_common():
        if n < min_ns_train:
            continue
        concepts = [c for c, count in ns_target_freq[ns].most_common() if count >= 3][:max_targets]
        if concepts:
            ns_vocab[ns] = concepts
    top_ns = list(ns_vocab)[:max_ns]
    return feature_vocab, {ns: ns_vocab[ns] for ns in top_ns}, top_ns


def encode_rows(rows, feature_vocab, ns_vocab, top_ns):
    fidx = {w: i for i, w in enumerate(feature_vocab)}
    out = {ns: {k: [] for k in ["Xtr", "Ytr", "Xval", "Yval", "Xte", "Yte"]} for ns in top_ns}
    return out, fidx


def featurize(row, fidx):
    x = np.zeros(len(fidx) + 2, dtype=np.float32)
    toks = set(tokens(row["text"]))
    for t in toks:
        i = fidx.get(t)
        if i is not None:
            x[i] = 1.0
    x[len(fidx)] = min(len(row["text"]) / 4000.0, 10.0)
    x[len(fidx) + 1] = float("related prior admission" in row["text"].lower())
    return x


def label(row, concepts):
    bare = {concept_name(k): v for k, v in row["gold"].items()}
    return np.array([1.0 if bare.get(c) in POS else 0.0 for c in concepts], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="${VP_ROOT}/data/temporal_episode_query_sft.json")
    ap.add_argument("--test", default="${VP_ROOT}/data/temporal_episode_query_test.jsonl")
    ap.add_argument("--out", default="${VP_ROOT}/analysis/agent_ml_data_episode.pkl")
    ap.add_argument("--max-features", type=int, default=600)
    ap.add_argument("--max-targets", type=int, default=15)
    ap.add_argument("--min-ns-train", type=int, default=20)
    ap.add_argument("--max-ns", type=int, default=8)
    args = ap.parse_args()

    train_rows = load_train(args.train)
    test_rows = load_test(args.test)
    rng = np.random.default_rng(0)
    order = np.arange(len(train_rows))
    rng.shuffle(order)
    n_val = max(1, int(0.1 * len(order))) if len(order) else 0
    val_ids = set(order[:n_val].tolist())

    feature_vocab, ns_vocab, top_ns = build_vocabs(train_rows, args.max_features, args.max_targets, args.min_ns_train, args.max_ns)
    data, fidx = encode_rows(train_rows + test_rows, feature_vocab, ns_vocab, top_ns)
    te_keys = {ns: [] for ns in top_ns}
    te_subjects = {ns: [] for ns in top_ns}

    for i, row in enumerate(train_rows):
        ns = row["ns"]
        if ns not in data:
            continue
        split = "val" if i in val_ids else "tr"
        data[ns]["X" + split].append(featurize(row, fidx))
        data[ns]["Y" + split].append(label(row, ns_vocab[ns]))
    for row in test_rows:
        ns = row["ns"]
        if ns not in data:
            continue
        data[ns]["Xte"].append(featurize(row, fidx))
        data[ns]["Yte"].append(label(row, ns_vocab[ns]))
        te_keys[ns].append(row.get("key"))
        te_subjects[ns].append(row.get("subject"))

    for ns in top_ns:
        for k in data[ns]:
            data[ns][k] = np.array(data[ns][k], dtype=np.float32)

    n_te_covered = sum(len(v) for v in te_keys.values())
    obj = {"data": data, "ns_vocab": ns_vocab, "top_ns": top_ns, "fdim": len(feature_vocab) + 2,
           "feature_vocab": feature_vocab, "te_keys": te_keys, "te_subjects": te_subjects}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pickle.dump(obj, open(args.out, "wb"))
    print(json.dumps({
        "out": args.out,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "test_rows_covered_by_top_ns": n_te_covered,
        "top_ns": top_ns,
        "sizes": {ns: {k: int(len(v)) for k, v in data[ns].items()} for ns in top_ns},
        "fdim": len(feature_vocab) + 2,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
