"""
Direct LLM virtual-patient radiology report baseline.

This is the end-to-end VP baseline: given only pre-target-exam context and the
requested exam, ask an API LLM to generate a free-text radiology report wrapped
in JSON. The generated report is then parsed with the same generic radiology
extraction prompt used to build cohort_rad_findings.jsonl, and scored against
held-out gold labels with deterministic hard metrics.

Example:
  python analysis/run_direct_vp_report_smoke.py --test data/temporal_episode_query_test.jsonl --run --n 10 --models gpt-5.4-mini --workers 1
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import time
import threading

sys.path.insert(0, "${VP_ROOT}/analysis")
import run_temporal_llm_smoke_v2 as sm  # noqa: E402
EXTRACT_SYS = "You extract structured findings from a radiology report. Output ONLY JSON."
EXTRACT_PROMPT = """Report modality=%s region=%s exam=%s.
Extract findings as JSON: {"observations":[{"concept":"<snake>","presence":"present|absent|uncertain","location":"<opt>"}]}.
STRICT rules:
- Output a concept ONLY if the report EXPLICITLY describes it (present/uncertain) or EXPLICITLY negates it (absent).
- The SEED list is just a preferred VOCABULARY for naming, NOT a checklist. Do NOT go through the seed list.
- Do NOT output concepts that are merely not mentioned (those are "not_mentioned", so OMIT them entirely - never mark them absent).
- Map a stated finding to the closest SEED concept if applicable; otherwise use "other:<snake_label>".
SEED(%s)=%s
REPORT:
\"\"\"%s\"\"\""""
SEED = {
    "Abdomen": ["appendicitis", "appendix_dilated", "periappendiceal_fluid", "gallstones", "cholecystitis",
        "gallbladder_wall_thickening", "pericholecystic_fluid", "cbd_dilation", "pancreatitis", "peripancreatic_fluid",
        "pancreatic_necrosis", "diverticulitis", "diverticulosis", "bowel_obstruction", "bowel_wall_thickening",
        "free_intraperitoneal_air", "free_fluid_ascites", "abscess", "fat_stranding", "hydronephrosis",
        "renal_calculus", "hepatic_lesion", "splenomegaly", "lymphadenopathy"],
    "Chest": ["pleural_effusion", "pneumothorax", "pulmonary_edema", "atelectasis", "consolidation", "opacity",
        "pneumonia", "cardiomegaly", "nodule"],
}

ROOT = "${VP_ROOT}"
DEFAULT_TEST = f"{ROOT}/data/temporal_episode_query_test.jsonl"
DEFAULT_FIND = f"{ROOT}/analysis/cohort_rad_findings.jsonl"
DEFAULT_CONFIG = f"{ROOT}/llm_api/gpt_56_sol.yaml"
DEFAULT_SCHEMA = f"{ROOT}/analysis/eval_schema_auto_train.json"
DEFAULT_CACHE_DIR = f"{ROOT}/analysis"
DEFAULT_PROMPT_VERSION = "direct_vp_report_v1_json_confidence"
PROMPT_VERSION = DEFAULT_PROMPT_VERSION
PARSE_PROMPT_VERSION = "cohort_rad_extract_prompt_v1"
POS = {"present", "uncertain"}
CONFIDENCE_CLASSES = sm.CONFIDENCE_CLASSES


def ns_to_mod_region(ns):
    tail = (ns or "rad.unknown_unknown").split(".", 1)[-1]
    if "_" in tail:
        mod, region = tail.split("_", 1)
    else:
        mod, region = tail, "unknown"
    return mod, region


def cache_key(model, rec):
    ctx_hash = hashlib.sha1(str(rec.get("context", "")).encode("utf-8")).hexdigest()[:12]
    return "|".join([
        model,
        PROMPT_VERSION,
        str(rec.get("hadm_id", "")),
        str(rec.get("charttime", "")),
        str(rec.get("ns", "")),
        str(rec.get("exam_name", "")),
        ctx_hash,
    ])


def parse_cache_key(parser_model, gen_model, rec, report):
    report_hash = hashlib.sha1((report or "").encode("utf-8")).hexdigest()[:12]
    return "|".join([
        parser_model,
        PARSE_PROMPT_VERSION,
        gen_model,
        str(rec.get("hadm_id", "")),
        str(rec.get("charttime", "")),
        str(rec.get("ns", "")),
        report_hash,
    ])


def load_cache(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("key"):
                out[o["key"]] = o
    return out


def extract_json_obj(text):
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", s).strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    return json.loads(s[a:b + 1])


def build_report_prompt(rec, input_max_chars=4000):
    context = rec.get("context", "") or ""
    context_truncated = False
    if input_max_chars and input_max_chars > 0 and len(context) > input_max_chars:
        context = context[:input_max_chars].rstrip() + "\n[context truncated by runner]"
        context_truncated = True
    sys_msg = (
        "You are acting as a virtual patient simulator for a clinical forecasting benchmark. "
        "Given only observations available before a requested radiology exam, generate the "
        "radiology report text that would plausibly be documented for that exam. Output ONLY JSON."
    )
    user = (
        "TASK:\n"
        "The target radiology report is hidden. Generate the free-text radiology report that would plausibly be documented "
        "for the requested target exam using only patient information available before that exam. "
        "This is an end-to-end virtual patient simulation/forecasting task, not extraction from the hidden report.\n\n"
        "REPORT REQUIREMENTS:\n"
        "- Write in concise radiology-report style, preferably with EXAMINATION, FINDINGS, and IMPRESSION sections.\n"
        "- Include findings that the target exam would likely state, including normal/negative findings when normally documented.\n"
        "- Do not output a candidate finding checklist or structured labels inside the report.\n"
        "- Do not copy unrelated history as imaging findings unless it would likely appear in the target report.\n"
        "- If prior radiology reports are provided, use them as prior evidence only; do not simply copy them as the target report.\n\n"
        "PATIENT CONTEXT AVAILABLE BEFORE THE TARGET EXAM:\n"
        f"{context}\n\n"
        "REQUESTED TARGET EXAM:\n"
        f"Exam name: {rec.get('exam_name', rec.get('ns', ''))}\n"
        f"Exam namespace/type: {rec.get('ns', '')}\n"
        f"Target time: {rec.get('charttime', '')}\n\n"
        "Finally, classify your confidence in the generated report into exactly one of these classes: "
        + "; ".join(CONFIDENCE_CLASSES)
        + ".\n\n"
        "Output JSON exactly as {\"report\":\"<free-text radiology report>\",\"confidence\":\"<confidence class>\"}."
    )
    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
    return messages, context_truncated


def parse_report_wrapper(text):
    try:
        obj = extract_json_obj(text)
    except Exception:
        obj = None
    if isinstance(obj, dict):
        report = obj.get("report")
        confidence = obj.get("confidence")
        if isinstance(report, str) and report.strip():
            return report.strip(), confidence if confidence in CONFIDENCE_CLASSES else None, True
    return (text or "").strip(), None, False


def generated_attrs_from_report(report, rec, parser_model, parser_cfg, parse_cache, parse_cache_path, parse_lock, retry_errors=False):
    key = parse_cache_key(parser_model, rec.get("_gen_model", ""), rec, report)
    if key in parse_cache and not (retry_errors and parse_cache[key].get("error")):
        o = parse_cache[key]
    else:
        mod, region = ns_to_mod_region(rec.get("ns"))
        region_name = region.capitalize()
        seed = SEED.get(region_name, [])
        try:
            t0 = time.time()
            raw, usage, latency = sm.call_chat([
                {"role": "system", "content": EXTRACT_SYS},
                {"role": "user", "content": EXTRACT_PROMPT % (mod, region, rec.get("exam_name", ""), region_name, seed, report[:6000])},
            ], parser_cfg, max_tokens=1024, temperature=0.1)
            o = {"key": key, "parser_model": parser_model, "text": raw, "usage": usage, "latency_sec": latency if latency is not None else time.time() - t0, "error": None}
        except Exception as e:
            o = {"key": key, "parser_model": parser_model, "text": "", "usage": {}, "latency_sec": None, "error": str(e)}
        with parse_lock:
            with open(parse_cache_path, "a") as w:
                w.write(json.dumps(o, ensure_ascii=False) + "\n")
            parse_cache[key] = o
    attrs = {}
    parse_ok = False
    if not o.get("error"):
        try:
            obj = extract_json_obj(o.get("text", "")) or {"observations": []}
            ns = rec.get("ns")
            for obs in obj.get("observations", []) or []:
                c = (obs.get("concept") or "").strip()
                if c:
                    attrs[f"{ns}.{c}"] = {"presence": obs.get("presence", "present"), "location": obs.get("location")}
            parse_ok = True
        except Exception as e:
            o["local_parse_error"] = str(e)
    return attrs, parse_ok, o


def pred_pos_from_attrs(attrs, ns, vocab):
    bare = {a.split(".", 2)[-1]: v.get("presence") for a, v in attrs.items() if a.startswith(ns + ".")}
    return {c for c in vocab.get(ns, []) if bare.get(c) in POS}


def gold_pos(rec, vocab):
    bare = {a.split(".", 2)[-1]: p for a, p in rec.get("gold", {}).items()}
    return {c for c in vocab.get(rec["ns"], []) if bare.get(c) in POS}


def build_vocab_from_schema(schema_path):
    obj = json.load(open(schema_path))
    out = {}
    for ns, d in (obj.get("namespaces") or {}).items():
        vals = d.get("included_presence_concepts") or []
        if vals:
            out[ns] = list(vals)
    return out


def score(insts):
    tp = fp = fn = 0
    gen_ok = parse_ok = 0
    conf_stats = {c: collections.Counter() for c in CONFIDENCE_CLASSES}
    for gp, pp, gen_parse_ok, report_parse_ok, conf in insts:
        tp += len(gp & pp)
        fp += len(pp - gp)
        fn += len(gp - pp)
        gen_ok += int(gen_parse_ok)
        parse_ok += int(report_parse_ok)
        if conf in conf_stats:
            conf_stats[conf]["n"] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(insts),
        "generation_json_parse_success": gen_ok / len(insts) if insts else 0.0,
        "report_parse_success": parse_ok / len(insts) if insts else 0.0,
        "precision": precision,
        "coverage_recall": recall,
        "hallucination_rate": 1.0 - precision if tp + fp else 0.0,
        "micro_f1": f1,
        "confidence_counts": {k: dict(v) for k, v in conf_stats.items() if v},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=DEFAULT_TEST)
    ap.add_argument("--find", default=DEFAULT_FIND)
    ap.add_argument("--schema", default=None, help="Optional fixed evaluation schema JSON with per-namespace included_presence_concepts.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--models", default="gpt-5.4-mini")
    ap.add_argument("--parser-model", default="gpt-5.4-mini")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--prompt-tag", default=DEFAULT_PROMPT_VERSION, help="Generation prompt/cache tag. Use a new tag when the input format changes.")
    ap.add_argument("--input-max-chars", type=int, default=4000, help="Maximum context characters inserted into the generation prompt; 0 disables truncation.")
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--api-retries", type=int, default=3, help="Finite per-request retry count for API errors such as 429/500.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not args.run:
        args.dry_run = True

    global PROMPT_VERSION
    PROMPT_VERSION = args.prompt_tag

    cfgs = sm.load_model_config(args.config)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    missing = [m for m in models if m not in cfgs]
    if missing:
        raise SystemExit(f"models not found in {args.config}: {missing}; available={sorted(cfgs)}")
    if args.parser_model not in cfgs:
        raise SystemExit(f"parser model {args.parser_model} not found in {args.config}; available={sorted(cfgs)}")

    rows = sm.load_rows(args.test, args.n)
    loaded_rows = list(rows)
    if args.schema:
        vocab = build_vocab_from_schema(args.schema)
        vocab_source = args.schema
    else:
        vocab = sm.build_vocab(args.find)
        vocab_source = args.find
    dropped_by_vocab = [
        {
            "row_in_loaded": i,
            "hadm_id": r.get("hadm_id"),
            "subject": r.get("subject"),
            "charttime": r.get("charttime"),
            "ns": r.get("ns"),
            "exam_name": r.get("exam_name"),
        }
        for i, r in enumerate(loaded_rows, 1)
        if not vocab.get(r.get("ns"))
    ]
    dropped_ns_counts = collections.Counter(x["ns"] for x in dropped_by_vocab)
    rows = [r for r in loaded_rows if vocab.get(r.get("ns"))]
    prompt_items = []
    trunc_by_key = {}
    for r in rows:
        messages, truncated = build_report_prompt(r, args.input_max_chars)
        prompt_items.append((r, messages))
        trunc_by_key[cache_key("__input__", r)] = truncated
    prompts = prompt_items
    est_in = sum(sm.estimate_tokens(msgs, args.max_tokens)[0] for _, msgs in prompts)
    context_lens = [len(r.get("context", "") or "") for r in rows]
    truncated_count = sum(1 for v in trunc_by_key.values() if v)
    print(f"rows={len(rows)} loaded_rows={len(loaded_rows)} dropped_by_vocab={len(dropped_by_vocab)} test={args.test}")
    if dropped_by_vocab:
        print(f"dropped_by_vocab_ns={dict(sorted(dropped_ns_counts.items()))}")
    print(f"dry_run={args.dry_run} models={models} prompt={PROMPT_VERSION} parser={args.parser_model} input_max_chars={args.input_max_chars} vocab_source={vocab_source}")
    print(f"estimated generation prompt tokens={est_in:,}; generation completion cap={len(rows)*args.max_tokens:,}; avg prompt={est_in/max(len(rows),1):.0f}; truncated_inputs={truncated_count}")
    if args.dry_run:
        return

    os.makedirs(args.cache_dir, exist_ok=True)
    parse_cache_path = os.path.join(args.cache_dir, f"direct_vp_report_parse_{args.parser_model}_{PARSE_PROMPT_VERSION}.jsonl")
    parse_cache = load_cache(parse_cache_path)
    parse_lock = threading.Lock()

    for model in models:
        safe_model = model.replace("/", "_")
        gen_cache_path = os.path.join(args.cache_dir, f"direct_vp_report_{safe_model}_{PROMPT_VERSION}.jsonl")
        gen_cache = load_cache(gen_cache_path)
        cfg = dict(cfgs[model])
        cfg["max_retry"] = args.api_retries
        parser_cfg = dict(cfgs[args.parser_model])
        parser_cfg["max_retry"] = args.api_retries
        gen_lock = threading.Lock()

        def one(item):
            rec, messages = item
            gen_key = cache_key(model, rec)
            if gen_key in gen_cache and not (args.retry_errors and gen_cache[gen_key].get("error")):
                gen_o = gen_cache[gen_key]
            else:
                try:
                    t0 = time.time()
                    text, usage, latency = sm.call_chat(messages, cfg, args.max_tokens, args.temperature)
                    gen_o = {
                        "key": gen_key,
                        "model": model,
                        "prompt": PROMPT_VERSION,
                        "hadm_id": rec.get("hadm_id"),
                        "subject": rec.get("subject"),
                        "charttime": rec.get("charttime"),
                        "ns": rec.get("ns"),
                        "exam_name": rec.get("exam_name"),
                        "text": text,
                        "usage": usage,
                        "latency_sec": latency if latency is not None else time.time() - t0,
                        "error": None,
                    }
                except Exception as e:
                    gen_o = {"key": gen_key, "model": model, "prompt": PROMPT_VERSION, "text": "", "usage": {}, "latency_sec": None, "error": str(e)}
                with gen_lock:
                    with open(gen_cache_path, "a") as w:
                        w.write(json.dumps(gen_o, ensure_ascii=False) + "\n")
                    gen_cache[gen_key] = gen_o

            if gen_o.get("error"):
                return gold_pos(rec, vocab), set(), False, False, None, gen_o, {}, None
            report, conf, gen_parse_ok = parse_report_wrapper(gen_o.get("text", ""))
            rec2 = dict(rec)
            rec2["_gen_model"] = model
            attrs, report_parse_ok, parse_o = generated_attrs_from_report(report, rec2, args.parser_model, parser_cfg, parse_cache, parse_cache_path, parse_lock, args.retry_errors)
            pred = pred_pos_from_attrs(attrs, rec.get("ns"), vocab)
            return gold_pos(rec, vocab), pred, gen_parse_ok, report_parse_ok, conf, gen_o, attrs, parse_o

        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            done = list(ex.map(one, prompts))
        metrics = score([(g, p, gok, pok, conf) for g, p, gok, pok, conf, *_ in done])
        usage_gen = collections.Counter()
        usage_parse = collections.Counter()
        gen_lat = []
        parse_lat = []
        errors_gen = 0
        errors_parse = 0
        records = []
        for rec, result in zip(rows, done):
            gp, pp, gen_ok, report_ok, conf, gen_o, attrs, parse_o = result
            errors_gen += int(bool(gen_o.get("error")))
            errors_parse += int(bool(parse_o and parse_o.get("error")))
            for k, v in (gen_o.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    usage_gen[k] += v
            if isinstance(gen_o.get("latency_sec"), (int, float)):
                gen_lat.append(gen_o["latency_sec"])
            if parse_o:
                for k, v in (parse_o.get("usage") or {}).items():
                    if isinstance(v, (int, float)):
                        usage_parse[k] += v
                if isinstance(parse_o.get("latency_sec"), (int, float)):
                    parse_lat.append(parse_o["latency_sec"])
            report, _, _ = parse_report_wrapper(gen_o.get("text", "")) if not gen_o.get("error") else ("", None, False)
            records.append({
                "hadm_id": rec.get("hadm_id"),
                "subject": rec.get("subject"),
                "charttime": rec.get("charttime"),
                "ns": rec.get("ns"),
                "exam_name": rec.get("exam_name"),
                "confidence": conf,
                "report": report,
                "gold_pos": sorted(gp),
                "pred_pos": sorted(pp),
                "tp": sorted(gp & pp),
                "fp": sorted(pp - gp),
                "fn": sorted(gp - pp),
                "parsed_attrs": attrs,
                "gen_error": gen_o.get("error"),
                "parse_error": parse_o.get("error") if parse_o else None,
                "context_char_len": len(rec.get("context", "") or ""),
                "input_truncated_by_runner": trunc_by_key.get(cache_key("__input__", rec), False),
                "input_mode": rec.get("input_mode"),
                "raw_context_audit": rec.get("raw_context_audit"),
            })
        rep = {
            "model": model,
            "prompt": PROMPT_VERSION,
            "parser_model": args.parser_model,
            "parse_prompt": PARSE_PROMPT_VERSION,
            "vocab_source": vocab_source,
            "test": args.test,
            "n_requested": args.n,
            "n_loaded_before_vocab_filter": len(loaded_rows),
            "n_scored": len(done),
            "dropped_by_vocab": len(dropped_by_vocab),
            "dropped_by_vocab_ns": dict(sorted(dropped_ns_counts.items())),
            "dropped_by_vocab_examples": dropped_by_vocab[:20],
            "input_max_chars": args.input_max_chars,
            "input_truncated_by_runner": truncated_count,
            "context_char_len": {
                "min": min(context_lens) if context_lens else None,
                "max": max(context_lens) if context_lens else None,
                "avg": sum(context_lens) / len(context_lens) if context_lens else None,
            },
            "wall_sec": time.time() - t0,
            "generation_errors": errors_gen,
            "report_parse_errors": errors_parse,
            "avg_generation_latency_sec": sum(gen_lat) / len(gen_lat) if gen_lat else None,
            "avg_parse_latency_sec": sum(parse_lat) / len(parse_lat) if parse_lat else None,
            "generation_usage": dict(usage_gen),
            "parse_usage": dict(usage_parse),
            "metrics": metrics,
            "generation_cache": gen_cache_path,
            "parse_cache": parse_cache_path,
        }
        summary_path = os.path.join(args.cache_dir, f"direct_vp_report_{safe_model}_{PROMPT_VERSION}_summary.json")
        records_path = os.path.join(args.cache_dir, f"direct_vp_report_{safe_model}_{PROMPT_VERSION}_records_n{args.n}.jsonl")
        md_path = os.path.join(args.cache_dir, f"direct_vp_report_{safe_model}_{PROMPT_VERSION}_examples_n{args.n}.md")
        with open(summary_path, "w") as w:
            json.dump(rep, w, ensure_ascii=False, indent=2)
        with open(records_path, "w") as w:
            for r in records:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        lines = [f"# Direct VP Report Smoke: {model}", "", f"prompt: `{PROMPT_VERSION}`", f"parser: `{args.parser_model}` / `{PARSE_PROMPT_VERSION}`", "", "## Summary", "", json.dumps(rep, ensure_ascii=False, indent=2), "", "## Examples"]
        for i, r in enumerate(records[:5], 1):
            lines += ["", f"### Case {i}: {r['ns']} | {r['exam_name']} | {r['charttime']}", f"confidence: `{r.get('confidence')}`", f"gold_pos: `{', '.join(r['gold_pos']) or 'none'}`", f"pred_pos: `{', '.join(r['pred_pos']) or 'none'}`", f"FP: `{', '.join(r['fp']) or 'none'}`", f"FN: `{', '.join(r['fn']) or 'none'}`", "", "```text", r.get("report", "")[:2000], "```"]
        with open(md_path, "w") as w:
            w.write("\n".join(lines) + "\n")
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        print(f"examples: {md_path}")


if __name__ == "__main__":
    main()
