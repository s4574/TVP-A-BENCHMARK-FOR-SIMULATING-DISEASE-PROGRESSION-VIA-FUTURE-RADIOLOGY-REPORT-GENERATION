"""Analyze returned radiologist annotation JSONs (5-point version, up to 3 annotators).
Usage:
  python3 analyze_radiologist_answers.py --taskA a1.json a2.json a3.json --taskB b1.json b2.json b3.json
(any subset works; single-file also fine)

Task A (Q1/Q2/Q3 each 1-5):
  - per-annotator score distributions
  - inter-annotator agreement: pairwise linear-weighted Cohen's kappa (1-5) + Fleiss' kappa on
    collapsed 3-way categories; majority/median aggregate
  - comparison vs the author (LLM-assisted) 3-way audit via collapse maps:
      Q1: 5->full, 3-4->partial, 1-2->poor
      Q2: 1->none, 2-3->minor, 4-5->major
      Q3: 1->none, 2-3->some, 4-5->severe
Task B (persistence 1-5 + flagM/flagN):
  - control pass rates: CA (target explicitly absent) expects v<=2 or flagM; CP expects v>=4 or flagM
  - forgiven items: median-score distribution, likely-present share (median>=4) vs likely-resolved
    (median<=2), flag rates (M = target actually mentions -> our extraction FN; N = prior label
    unsupported -> prior extraction error), per-arm breakdown, inter-annotator weighted kappa.
"""
import argparse, collections, csv, itertools, json, statistics

ROOT = "/workspace/siyan/virtualPatient"

Q1MAP = {1: "poor", 2: "poor", 3: "partial", 4: "partial", 5: "full"}
Q2MAP = {1: "none", 2: "minor", 3: "minor", 4: "major", 5: "major"}
Q3MAP = {1: "none", 2: "some", 3: "some", 4: "severe", 5: "severe"}

def wkappa(pairs, k=5):
    """Linear-weighted Cohen's kappa for ordinal 1..k pairs."""
    if not pairs: return None
    n = len(pairs)
    w = [[1 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    obs = [[0] * k for _ in range(k)]
    for a, b in pairs: obs[a - 1][b - 1] += 1
    pa = [sum(obs[i]) / n for i in range(k)]
    pb = [sum(obs[i][j] for i in range(k)) / n for j in range(k)]
    po = sum(w[i][j] * obs[i][j] / n for i in range(k) for j in range(k))
    pe = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return (po - pe) / (1 - pe) if pe < 1 else None

def fleiss(items):
    """Fleiss' kappa. items: list of Counters(category->count) with equal total raters preferred."""
    items = [c for c in items if sum(c.values()) >= 2]
    if not items: return None
    cats = sorted({c for it in items for c in it})
    N = len(items)
    P_i, p_j = [], collections.Counter()
    for it in items:
        n = sum(it.values())
        P_i.append((sum(v * v for v in it.values()) - n) / (n * (n - 1)))
        for c, v in it.items(): p_j[c] += v
    total = sum(p_j.values())
    Pbar = statistics.mean(P_i)
    Pe = sum((p_j[c] / total) ** 2 for c in cats)
    return (Pbar - Pe) / (1 - Pe) if Pe < 1 else None

def cohen(pairs):
    if not pairs: return None
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pa = collections.Counter(a for a, _ in pairs); pb = collections.Counter(b for _, b in pairs)
    pe = sum(pa[c] * pb[c] for c in set(pa) | set(pb)) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else None

def load(paths):
    out = {}
    for p in paths:
        d = json.load(open(p))
        out[d.get("annotator") or p] = d["answers"]
    return out

def task_a(paths):
    ann = load(paths)
    names = sorted(ann)
    print(f"== Task A: annotators = {names} ==")
    for nm in names:
        done = [v for v in ann[nm].values() if v.get("q1")]
        print(f"  {nm}: answered {len(done)}; "
              + "; ".join(f"{q} mean={statistics.mean(int(v[q]) for v in done if v.get(q)):.2f}"
                          for q in ("q1", "q2", "q3") if any(v.get(q) for v in done)))
    # pairwise weighted kappa on shared items
    for q in ("q1", "q2", "q3"):
        for a, b in itertools.combinations(names, 2):
            shared = [c for c in ann[a] if c in ann[b] and ann[a][c].get(q) and ann[b][c].get(q)]
            pairs = [(int(ann[a][c][q]), int(ann[b][c][q])) for c in shared]
            if pairs:
                print(f"  {q} weighted-kappa {a} vs {b}: n={len(pairs)} k_w={wkappa(pairs):.3f}")
        # Fleiss on collapsed categories
        MAP = {"q1": Q1MAP, "q2": Q2MAP, "q3": Q3MAP}[q]
        cases = collections.defaultdict(collections.Counter)
        for nm in names:
            for c, v in ann[nm].items():
                if v.get(q): cases[c][MAP[int(v[q])]] += 1
        fk = fleiss(list(cases.values()))
        if fk is not None: print(f"  {q} Fleiss-kappa (collapsed 3-way, >=2 raters/case): {fk:.3f}")
    # vs author audit (collapsed, median across annotators)
    author = {}
    with open(f"{ROOT}/analysis/annotation_100/annotations_llm.csv") as f:
        for row in csv.DictReader(f): author[row["case_id"]] = row
    cols = {"q1": "q1_coverage(full/partial/poor)", "q2": "q2_contradiction(none/minor/major)",
            "q3": "q3_redundancy(none/some/severe)"}
    for q, col in cols.items():
        MAP = {"q1": Q1MAP, "q2": Q2MAP, "q3": Q3MAP}[q]
        pairs = []
        allc = {c for nm in names for c in ann[nm]}
        for c in allc:
            vals = [int(ann[nm][c][q]) for nm in names if c in ann[nm] and ann[nm][c].get(q)]
            au = author.get(c, {}).get(col, "").strip()
            if vals and au:
                med = int(statistics.median(vals))
                pairs.append((MAP[med], au))
        if pairs:
            agree = sum(1 for a, b in pairs if a == b) / len(pairs)
            print(f"  {q} median-of-radiologists vs author audit: n={len(pairs)} agree={agree:.3f} kappa={cohen(pairs):.3f}")

def task_b(paths):
    ann = load(paths)
    names = sorted(ann)
    mp = {m["id"]: m for m in json.load(open(f"{ROOT}/analysis/radiologist_taskB_mapping.json"))}
    print(f"\n== Task B: annotators = {names} ==")
    # controls per annotator
    for nm in names:
        a = ann[nm]
        ca = [(i, v) for i, v in a.items() if mp.get(i, {}).get("type") == "CA" and (v.get("v") or v.get("flagM") or v.get("flagN"))]
        cp = [(i, v) for i, v in a.items() if mp.get(i, {}).get("type") == "CP" and (v.get("v") or v.get("flagM") or v.get("flagN"))]
        cap = sum(1 for _, v in ca if v.get("flagM") or (v.get("v") and int(v["v"]) <= 2))
        cpp = sum(1 for _, v in cp if v.get("flagM") or (v.get("v") and int(v["v"]) >= 4))
        nans = sum(1 for v in a.values() if v.get("v") or v.get("flagM") or v.get("flagN"))
        print(f"  {nm}: answered {nans}; controls CA pass {cap}/{len(ca)}, CP pass {cpp}/{len(cp)}")
    # inter-annotator weighted kappa on scored (non-flag) forgiven items
    for a, b in itertools.combinations(names, 2):
        pairs = [(int(ann[a][i]["v"]), int(ann[b][i]["v"])) for i in ann[a]
                 if mp.get(i, {}).get("type") == "F" and i in ann[b]
                 and ann[a][i].get("v") and ann[b][i].get("v")
                 and not (ann[a][i].get("flagM") or ann[a][i].get("flagN") or ann[b][i].get("flagM") or ann[b][i].get("flagN"))]
        if pairs: print(f"  persistence weighted-kappa {a} vs {b}: n={len(pairs)} k_w={wkappa(pairs):.3f}")
    # calibration on forgiven items (median across annotators; flags: any annotator)
    allc = {i for nm in names for i in ann[nm] if mp.get(i, {}).get("type") == "F"}
    med, flagsM, flagsN = {}, set(), set()
    for i in allc:
        vals = [int(ann[nm][i]["v"]) for nm in names if i in ann[nm] and ann[nm][i].get("v")]
        if any(ann[nm].get(i, {}).get("flagM") for nm in names): flagsM.add(i)
        if any(ann[nm].get(i, {}).get("flagN") for nm in names): flagsN.add(i)
        if vals: med[i] = statistics.median(vals)
    scored = {i: v for i, v in med.items() if i not in flagsM and i not in flagsN}
    n = len(scored)
    if n:
        dist = collections.Counter(("present" if v >= 4 else "resolved" if v <= 2 else "uncertain") for v in scored.values())
        print(f"  forgiven items scored (median, flags excluded): n={n}; "
              f"likely-present {dist['present']} ({dist['present']/n*100:.0f}%), "
              f"likely-resolved {dist['resolved']} ({dist['resolved']/n*100:.0f}%), "
              f"uncertain {dist['uncertain']} ({dist['uncertain']/n*100:.0f}%)")
        dec = dist["present"] + dist["resolved"]
        if dec:
            print(f"  HEADLINE among decided: likely-still-present = {dist['present']/dec:.3f}; mean score = {statistics.mean(scored.values()):.2f}")
    print(f"  flag rates over answered forgiven items: M(target actually mentions)={len(flagsM)}, N(prior unsupported)={len(flagsN)}")
    per_arm = collections.defaultdict(list)
    for i, v in scored.items():
        for a in mp[i]["arms"]: per_arm[a].append(v)
    for a, vs in sorted(per_arm.items()):
        print(f"    arm {a}: n={len(vs)} mean={statistics.mean(vs):.2f} likely-present={sum(1 for v in vs if v>=4)/len(vs)*100:.0f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskA", nargs="*", default=[])
    ap.add_argument("--taskB", nargs="*", default=[])
    args = ap.parse_args()
    if args.taskA: task_a(args.taskA)
    if args.taskB: task_b(args.taskB)
