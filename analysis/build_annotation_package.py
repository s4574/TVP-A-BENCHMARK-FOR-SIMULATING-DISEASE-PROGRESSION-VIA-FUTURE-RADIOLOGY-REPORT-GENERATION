"""Package a stratified 100-case sample of the sealed set for HUMAN verification of the extraction-based
eval schema (paper §3 silver-GT audit). Per case: the REAL target report text (cohort_rad_corpus join),
the extractor's gold four-state attributes (what the benchmark scores against), and diagnosis (ICD) lines
from the pre-exam context as clinical reference.

Annotator questions (protocol agreed 2026-09-24):
  Q1 coverage:      do the extracted attributes cover ALL key findings of the real report?
  Q2 contradiction: does any extracted attribute CONTRADICT the report text?
  Q3 redundancy:    are there redundant/spurious attributes (stated but not meaningfully in the report)?

Outputs (analysis/annotation_100/): cases.md (readable, one section per case) + annotations.csv (to fill).
"""
import csv, json, os, re, sys
import numpy as np

ROOT = "${VP_ROOT}"
sys.path.insert(0, f"{ROOT}/analysis")
import run_direct_vp_report_smoke as R

TEST = f"{ROOT}/data/temporal_episode_query_test_stratified_1500_clean_supported.jsonl"
CORPUS = f"{ROOT}/analysis/cohort_rad_corpus.jsonl"
SCHEMA = f"{ROOT}/analysis/eval_schema_auto_train.json"
OUT = f"{ROOT}/analysis/annotation_100"
N, SEED = 100, 0


def main():
    vocab = R.build_vocab_from_schema(SCHEMA)
    rows = [json.loads(l) for l in open(TEST) if l.strip()]
    rows = [r for r in rows if vocab.get(r.get("ns"))]
    print(f"scorable pool={len(rows)}")

    # proportional ns-stratified sample (largest remainder, min 1), same spirit as the sealed draw
    by_ns = {}
    for r in rows:
        by_ns.setdefault(r["ns"], []).append(r)
    total = sum(len(v) for v in by_ns.values())
    quota = {ns: len(v) * N / total for ns, v in by_ns.items()}
    take = {ns: int(q) for ns, q in quota.items()}
    rem = N - sum(take.values())
    for ns in sorted(quota, key=lambda n: -(quota[n] - take[n]))[:rem]:
        take[ns] += 1
    rng = np.random.default_rng(SEED)
    sample = []
    for ns, k in take.items():
        idx = rng.permutation(len(by_ns[ns]))[:max(k, 1) if k else 0]
        sample += [by_ns[ns][i] for i in idx]
    sample = sample[:N]
    print(f"sampled={len(sample)} across {sum(1 for k in take.values() if k)} ns")

    # corpus join: (hadm, charttime, exam_name) -> report text
    corpus = {}
    want = {(str(r["hadm_id"]), r["charttime"], r["exam_name"]) for r in sample}
    with open(CORPUS) as f:
        for line in f:
            o = json.loads(line)
            k = (str(o.get("hadm_id")), o.get("charttime"), o.get("exam_name"))
            if k in want:
                corpus[k] = o.get("text", "")
    print(f"report text found for {len(corpus)}/{len(sample)}")

    os.makedirs(OUT, exist_ok=True)
    md = open(f"{OUT}/cases.md", "w")
    md.write(
        "# 抽取器 GT 人工核验(n=100,sealed 分层抽样,seed=0)\n\n"
        "**每例材料**:真实目标报告原文 + 基准打分所依据的抽取属性(四态)+ 上下文中的诊断(ICD)参考。\n\n"
        "**三个问题(填入 annotations.csv)**:\n"
        "- **Q1 coverage** — 抽取属性是否覆盖了报告的全部重点所见?`full`(全覆盖)/ `partial`(漏了次要点)/ `poor`(漏了主要点);漏掉的写进 q1_missed。\n"
        "- **Q2 contradiction** — 有无属性与报告事实矛盾?`none` / `minor`(定位/程度出入)/ `major`(有↔无颠倒);写明哪个概念。\n"
        "- **Q3 redundancy** — 有无报告中并不存在/无意义的冗余属性?`none` / `some` / `severe`;写明哪个概念。\n\n"
        "注意:只核对**该 ns 的 schema 概念**(打分只用它们);absent=报告明确否定;not_mentioned 不列出(=报告未提)。\n\n---\n\n")
    csvf = open(f"{OUT}/annotations.csv", "w", newline="")
    w = csv.writer(csvf)
    w.writerow(["case_id", "ns", "q1_coverage(full/partial/poor)", "q1_missed",
                "q2_contradiction(none/minor/major)", "q2_which",
                "q3_redundancy(none/some/severe)", "q3_which", "notes"])
    miss = 0
    for i, r in enumerate(sorted(sample, key=lambda x: (x["ns"], str(x["hadm_id"])))):
        key = (str(r["hadm_id"]), r["charttime"], r["exam_name"])
        cid = f"{key[0]}|{key[1]}|{r['ns']}"
        text = corpus.get(key)
        if not text:
            miss += 1
        icd = "\n".join(dict.fromkeys(m.strip() for m in re.findall(r"^.*ICD:.*$", r.get("context", ""), re.M)))[:1200]
        gold = {k.split(".", 2)[-1]: v for k, v in r["gold"].items()}
        md.write(f"## case {i+1:03d}  `{cid}`\n\n**Exam**: {r['exam_name']}  ({r['ns']})  @ {r['charttime']}\n\n")
        md.write(f"**诊断参考(pre-exam context 中的 ICD 行)**:\n```\n{icd or '(无)'}\n```\n\n")
        md.write(f"**真实报告原文**:\n```\n{(text or '【未在 corpus 中找到 —— 跳过此例】').strip()}\n```\n\n")
        md.write("**抽取属性(基准 GT,四态;not_mentioned 未列)**:\n\n| concept | state |\n|---|---|\n")
        for c in vocab[r["ns"]]:
            if c in gold:
                md.write(f"| {c} | **{gold[c]}** |\n")
        md.write(f"\n(该 ns schema 共 {len(vocab[r['ns']])} 概念;未列出的 = not_mentioned)\n\n---\n\n")
        w.writerow([cid, r["ns"], "", "", "", "", "", "", "" if text else "REPORT TEXT MISSING"])
    md.close(); csvf.close()
    print(f"-> {OUT}/cases.md + annotations.csv  (missing report text: {miss})")


if __name__ == "__main__":
    main()
