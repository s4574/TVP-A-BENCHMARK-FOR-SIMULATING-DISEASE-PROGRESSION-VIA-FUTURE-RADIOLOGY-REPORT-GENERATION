"""
Summarize temporal sealed-test distribution and create a deterministic 1,500-row
namespace-stratified subset for frontier-LLM dry runs.

The temporal task is built as: context events strictly before target radiology
exam time -> predict target exam findings. For time-gap analysis, the anchor is
the latest observed input event before the target exam among labs, medications,
and prior radiology exams within the same admission.
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import random
import re
from datetime import datetime


ROOT = "${VP_ROOT}"
TEST = f"{ROOT}/data/temporal_query_test.jsonl"
FIND = f"{ROOT}/analysis/cohort_rad_findings.jsonl"
SHARDS = "${WORKSPACE}/open_data/mimic_iv/processed/mimic/merged_mimic_dataset_first_day_extracted_0202"
OUT_SUBSET = f"{ROOT}/data/temporal_query_test_stratified_1500.jsonl"
OUT_REPORT = f"{ROOT}/analysis/temporal_test_distribution.md"
OUT_JSON = f"{ROOT}/analysis/temporal_test_distribution.json"


def parse_time(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ns_parts(ns):
    if not ns.startswith("rad."):
        return "non_rad", "unknown", "unknown"
    body = ns.split(".", 1)[1]
    if "_" in body:
        modality, region = body.split("_", 1)
    else:
        modality, region = body, "unknown"
    return "imaging_report", modality, region


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def table(counter, denom, headers, limit=None):
    items = counter.most_common(limit)
    out = [f"| {headers[0]} | n | % |", "|---|---:|---:|"]
    for k, n in items:
        out.append(f"| {k} | {n} | {pct(n, denom):.2f} |")
    return "\n".join(out)


def stratified_sample(rows, n=1500, key="ns", seed=0):
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[r.get(key, "")].append((i, r))
    total = len(rows)
    if n >= total:
        return rows[:], {k: len(v) for k, v in groups.items()}

    allocation = {}
    remainders = []
    for k, vals in groups.items():
        exact = len(vals) * n / total
        base = min(len(vals), max(1, math.floor(exact)))
        allocation[k] = base
        remainders.append((exact - math.floor(exact), k))

    cur = sum(allocation.values())
    for _, k in sorted(remainders, reverse=True):
        if cur >= n:
            break
        if allocation[k] < len(groups[k]):
            allocation[k] += 1
            cur += 1

    if cur > n:
        for _, k in sorted(remainders):
            if cur <= n:
                break
            if allocation[k] > 1:
                allocation[k] -= 1
                cur -= 1

    rng = random.Random(seed)
    picked = []
    for k, vals in sorted(groups.items()):
        vals = vals[:]
        rng.shuffle(vals)
        picked.extend(vals[: allocation[k]])
    picked.sort(key=lambda x: x[0])
    return [r for _, r in picked], allocation


def load_rad_by_hadm():
    rad_by_h = collections.defaultdict(list)
    with open(FIND) as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("error") or not o.get("attrs"):
                continue
            h = str(o.get("hadm_id"))
            if not h:
                continue
            rad_by_h[h].append(o)
    for h in rad_by_h:
        rad_by_h[h].sort(key=lambda x: x.get("charttime") or "")
    return rad_by_h


def load_hosp_events(needed_hadm):
    hosp = {}
    for fp in sorted(glob.glob(SHARDS + "/*.jsonl")):
        with open(fp) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                h = str(d.get("hadm_id"))
                if h not in needed_hadm or h in hosp:
                    continue
                events = []
                for e in d.get("lab_vital_timeline") or []:
                    if e.get("time"):
                        events.append((e["time"], "lab"))
                for e in d.get("treatment_timeline") or []:
                    if e.get("time") and e.get("name"):
                        events.append((e["time"], "med"))
                hosp[h] = {
                    "subject": str(d.get("subject_id")),
                    "events": sorted(events),
                }
        if len(hosp) >= len(needed_hadm):
            break
    return hosp


BUCKETS = [
    ("0-1h", 0, 1),
    ("1-6h", 1, 6),
    ("6-24h", 6, 24),
    ("1-3d", 24, 72),
    ("3-7d", 72, 168),
    ("7-30d", 168, 720),
    (">30d", 720, None),
]


def bucket_hours(hours):
    if hours is None:
        return "no_prior_context"
    for name, lo, hi in BUCKETS:
        if hours >= lo and (hi is None or hours < hi):
            return name
    return "negative_or_invalid"


def annotate_time(rows, hosp, rad_by_h):
    out = []
    relation = collections.Counter()
    gap_bucket = collections.Counter()
    anchor_source = collections.Counter()
    for r in rows:
        h = str(r["hadm_id"])
        target = parse_time(r.get("charttime"))
        anchors = []
        if target:
            for ts, kind in hosp.get(h, {}).get("events", []):
                et = parse_time(ts)
                if et and et < target:
                    anchors.append((et, kind))
            for ex in rad_by_h.get(h, []):
                et = parse_time(ex.get("charttime"))
                if et and et < target:
                    anchors.append((et, "prior_rad"))
        if not anchors or not target:
            rel = "no_prior_context"
            hours = None
            src = "none"
        else:
            anchor_t = max(t for t, _ in anchors)
            srcs = sorted({k for t, k in anchors if t == anchor_t})
            hours = (target - anchor_t).total_seconds() / 3600.0
            src = "+".join(srcs)
            if hours < 0:
                rel = "earlier_than_input"
            elif hours == 0:
                rel = "equal_to_input"
            else:
                rel = "later_than_input"
        relation[rel] += 1
        gap_bucket[bucket_hours(hours)] += 1
        anchor_source[src] += 1
        out.append((rel, bucket_hours(hours), src, hours))
    return out, relation, gap_bucket, anchor_source


def summarize(rows, name, hosp, rad_by_h):
    n = len(rows)
    type_c = collections.Counter()
    mod_c = collections.Counter()
    region_c = collections.Counter()
    ns_c = collections.Counter()
    exam_c = collections.Counter()
    label_c = collections.Counter()
    presence_c = collections.Counter()
    attrs_per_record = []
    subjects = set()
    hadms = set()

    for r in rows:
        kind, mod, region = ns_parts(r.get("ns", ""))
        type_c[kind] += 1
        mod_c[mod] += 1
        region_c[region] += 1
        ns_c[r.get("ns", "")] += 1
        exam_c[r.get("exam_name", "")] += 1
        subjects.add(r.get("subject"))
        hadms.add(r.get("hadm_id"))
        gold = r.get("gold") or {}
        attrs_per_record.append(len(gold))
        for k, v in gold.items():
            label_c[k] += 1
            presence_c[v] += 1

    _, relation_c, gap_c, source_c = annotate_time(rows, hosp, rad_by_h)
    attrs_mean = sum(attrs_per_record) / n if n else 0.0
    attrs_sorted = sorted(attrs_per_record)
    attrs_med = attrs_sorted[n // 2] if n else 0

    return {
        "name": name,
        "n": n,
        "subjects": len(subjects),
        "hadms": len(hadms),
        "attrs_mean": attrs_mean,
        "attrs_median": attrs_med,
        "type": type_c,
        "modality": mod_c,
        "region": region_c,
        "ns": ns_c,
        "exam": exam_c,
        "label": label_c,
        "presence": presence_c,
        "relation": relation_c,
        "gap": gap_c,
        "anchor_source": source_c,
    }


def add_section(lines, s):
    n = s["n"]
    lines.append(f"## {s['name']}")
    lines.append("")
    lines.append(
        f"- records: {n}; subjects: {s['subjects']}; admissions: {s['hadms']}; "
        f"gold attrs/record mean={s['attrs_mean']:.2f}, median={s['attrs_median']}"
    )
    lines.append("")
    lines.append("### Target type")
    lines.append(table(s["type"], n, ["type"]))
    lines.append("")
    lines.append("### Imaging modality")
    lines.append(table(s["modality"], n, ["modality"]))
    lines.append("")
    lines.append("### Body region")
    lines.append(table(s["region"], n, ["region"]))
    lines.append("")
    lines.append("### Namespace")
    lines.append(table(s["ns"], n, ["namespace"]))
    lines.append("")
    lines.append("### Top exam names")
    lines.append(table(s["exam"], n, ["exam_name"], limit=30))
    lines.append("")
    lines.append("### Target time relative to latest input event")
    lines.append(table(s["relation"], n, ["relation"]))
    lines.append("")
    lines.append("### Target lead-time buckets")
    lines.append(table(s["gap"], n, ["bucket"]))
    lines.append("")
    lines.append("### Latest input event source")
    lines.append(table(s["anchor_source"], n, ["source"]))
    lines.append("")
    lines.append("### Gold presence labels")
    lines.append(table(s["presence"], sum(s["presence"].values()), ["presence"]))
    lines.append("")
    lines.append("### Top gold attributes")
    lines.append(table(s["label"], sum(s["label"].values()), ["attribute"], limit=40))
    lines.append("")


def serializable(s):
    out = {}
    for k, v in s.items():
        if isinstance(v, collections.Counter):
            out[k] = dict(v)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_jsonl(TEST)
    subset, allocation = stratified_sample(rows, n=args.n, key="ns", seed=args.seed)
    with open(OUT_SUBSET, "w") as w:
        for r in subset:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    needed = {str(r["hadm_id"]) for r in rows}
    rad_by_h = load_rad_by_hadm()
    hosp = load_hosp_events(needed)

    full = summarize(rows, "full temporal test", hosp, rad_by_h)
    sub = summarize(subset, f"namespace-stratified subset n={len(subset)} seed={args.seed}", hosp, rad_by_h)

    lines = [
        "# Temporal Test Distribution",
        "",
        "Scope: `data/temporal_query_test.jsonl`; deterministic subset written to "
        "`data/temporal_query_test_stratified_1500.jsonl`.",
        "",
        "Current temporal target type is radiology/imaging findings only. Structured labs/indicators and PE are "
        "available in the older `mm_query_*` static benchmark, but they are not targets in this temporal sealed test.",
        "",
        "Time anchor: latest observed input event strictly before the target exam among labs, medications, and prior "
        "radiology in the same admission. Because the dataset builder only includes pre-exam context (`event_time < "
        "target_charttime`), anchored targets should all be later than the input; rows with no prior event are "
        "reported separately.",
        "",
    ]
    add_section(lines, full)
    add_section(lines, sub)
    lines.append("## Stratified Subset Allocation")
    lines.append("")
    lines.append(table(collections.Counter(allocation), len(subset), ["namespace"]))
    lines.append("")

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w") as w:
        w.write("\n".join(lines))
    with open(OUT_JSON, "w") as w:
        json.dump({"full": serializable(full), "subset": serializable(sub), "allocation": allocation}, w, ensure_ascii=False, indent=2)

    print(f"wrote {OUT_REPORT}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_SUBSET}")
    print(f"full n={full['n']} subjects={full['subjects']} hadms={full['hadms']}")
    print(f"subset n={sub['n']} subjects={sub['subjects']} hadms={sub['hadms']}")
    print("full target types", dict(full["type"]))
    print("full relation", dict(full["relation"]))
    print("subset relation", dict(sub["relation"]))


if __name__ == "__main__":
    main()
