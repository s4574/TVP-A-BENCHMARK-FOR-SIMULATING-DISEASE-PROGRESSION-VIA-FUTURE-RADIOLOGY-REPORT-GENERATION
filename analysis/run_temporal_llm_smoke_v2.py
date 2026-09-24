"""
LLM smoke runner for temporal radiology finding prediction with a clarified
task prompt and fixed confidence classes.

Primary scoring still uses only presence labels. Confidence is stored and
summarized separately for calibration/trustworthiness analysis.

Examples:
  python analysis/run_temporal_llm_smoke_v2.py --dry-run --n 20 --models gpt-5.4-mini
  python analysis/run_temporal_llm_smoke_v2.py --run --n 20 --models gpt-5.4-mini --workers 1
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import re
import time
import urllib.request


ROOT = "${VP_ROOT}"
DEFAULT_TEST = f"{ROOT}/data/temporal_query_test_stratified_1500.jsonl"
DEFAULT_FIND = f"{ROOT}/analysis/cohort_rad_findings.jsonl"
DEFAULT_CONFIG = f"{ROOT}/llm_api/gpt_56_sol.yaml"
DEFAULT_CACHE_DIR = f"{ROOT}/analysis"
POS = {"present", "uncertain"}
CONFIDENCE_CLASSES = [
    "Almost no chance",
    "Highly unlikely",
    "Chances are slight",
    "Unlikely",
    "Less than even",
    "Better than even",
    "Likely",
    "Very good chance",
    "Highly likely",
    "Almost certain",
]


def parse_scalar(v):
    v = v.strip().strip('"').strip("'")
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d*\.\d+", v):
        return float(v)
    return v


def load_model_config(path):
    try:
        import yaml
        obj = yaml.safe_load(open(path)) or {}
        if "models" in obj and isinstance(obj["models"], dict):
            return obj["models"]
        if isinstance(obj, dict):
            # Either a single OpenAI-style config or {model_name: config}.
            if {"api_key", "base_url"} <= set(obj):
                name = str(obj.get("name") or obj.get("model_name") or os.path.splitext(os.path.basename(path))[0])
                return {name: obj}
            models = {}
            for name, cfg in obj.items():
                if isinstance(cfg, dict) and {"api_key", "base_url"} <= set(cfg):
                    models[str(name)] = cfg
            if models:
                return models
    except Exception:
        pass

    models = {}
    cur = None
    top = {}
    in_models = False
    with open(path) as f:
        for raw in f:
            line = raw.rstrip()
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if re.match(r"^models:\s*$", line):
                in_models = True
                continue
            if in_models:
                m = re.match(r"^  ([\w.\-]+):\s*$", line)
                if m:
                    cur = m.group(1)
                    models[cur] = {}
                    continue
                m = re.match(r"^    ([\w_\-]+):\s*(.+?)\s*$", line)
                if m and cur:
                    models[cur][m.group(1)] = parse_scalar(m.group(2))
                    continue
            else:
                m = re.match(r"^([\w.\-]+):\s*$", line)
                if m:
                    cur = m.group(1)
                    models[cur] = {}
                    continue
                m = re.match(r"^  ([\w_\-]+):\s*(.+?)\s*$", line)
                if m and cur:
                    models[cur][m.group(1)] = parse_scalar(m.group(2))
                    continue
                m = re.match(r"^([\w_\-]+):\s*(.+?)\s*$", line)
                if m:
                    top[m.group(1)] = parse_scalar(m.group(2))
    if not models and {"api_key", "base_url"} <= set(top):
        name = str(top.get("name") or top.get("model_name") or os.path.splitext(os.path.basename(path))[0])
        models[name] = top
    return models

def load_rows(path, n):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if n and len(rows) >= n:
                break
    return rows


def build_vocab(find_path):
    nsc = collections.defaultdict(collections.Counter)
    with open(find_path) as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("error"):
                continue
            ns = o.get("ns")
            for attr in o.get("attrs", {}):
                nsc[ns][attr.split(".", 2)[-1]] += 1
    return {ns: [c for c, count in cnt.most_common() if count >= 5][:15] for ns, cnt in nsc.items()}


def gold_set(rec, vocab):
    bare = {a.split(".", 2)[-1]: p for a, p in rec.get("gold", {}).items()}
    return {c for c in vocab.get(rec["ns"], []) if bare.get(c) in POS}


def prompt_for(rec, cands, include_rationale=False):
    sys_msg = (
        "You are participating in a clinical forecasting benchmark. Given only observations "
        "available before a requested radiology exam, predict that exam's findings. Output ONLY JSON."
    )
    user = (
        "TASK:\n"
        "The target radiology report is hidden. This is not information extraction from a report. "
        "For each candidate finding, forecast what the hidden target exam report would explicitly state, "
        "using the pre-exam patient context and the requested exam type.\n\n"
        "LABEL SEMANTICS:\n"
        "- present: the target report would state or clearly support the finding.\n"
        "- absent: the target report would say no/not present, or would likely not mention the finding.\n"
        "- uncertain: the target report would explicitly hedge the finding (e.g., possible/equivocal/cannot exclude). "
        "Do not use uncertain for your own epistemic uncertainty; use the confidence field for that.\n"
        "- Do not carry forward a prior finding or device as present unless the target exam would likely still report it.\n\n"
        "PATIENT CONTEXT AVAILABLE BEFORE THE TARGET EXAM:\n"
        f"{rec['context'][:4000]}\n\n"
        f"REQUESTED TARGET EXAM:\n{rec.get('exam_name', rec['ns'])} ({rec['ns']})\n\n"
        "CANDIDATE FINDINGS TO LABEL:\n"
        f"{', '.join(cands)}\n\n"
        "For every candidate, output:\n"
        "- presence: exactly one of present, absent, uncertain, following the label semantics above.\n"
        "- confidence: your calibrated belief that this specific presence label is correct, using exactly one "
        "of the following classes: "
        + "; ".join(CONFIDENCE_CLASSES)
        + ".\n"
        + ("- evidence: a short phrase, <=12 words, naming the main pre-exam clue or `no direct evidence`.\n" if include_rationale else "")
        + "\n"
        + ('Output JSON exactly as {"findings":{"<candidate>":{"presence":"present|absent|uncertain",'
           '"confidence":"<confidence class>","evidence":"<short phrase>"}}}. ' if include_rationale else
           'Output JSON exactly as {"findings":{"<candidate>":{"presence":"present|absent|uncertain",'
           '"confidence":"<confidence class>"}}}. ')
        + "Use candidate keys exactly; do not use full namespaced attributes; do not omit candidates."
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def estimate_tokens(messages, max_tokens):
    chars = sum(len(m["content"]) for m in messages)
    return int(chars / 3.8) + 16, max_tokens


PROMPT_VERSION = "clarified_confidence_v3_labelsem_ctxhash"


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


def load_cache(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("key"):
                out[o["key"]] = o
    return out


def call_chat(messages, cfg, max_tokens, temperature):
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.get("model_name") or cfg.get("model") or cfg.get("name"),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if isinstance(cfg.get("extra_body"), dict):
        payload.update(cfg["extra_body"])
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + cfg["api_key"]}
    last = None
    for attempt in range(max(int(cfg.get("max_retry", 3)), 3)):
        try:
            req = urllib.request.Request(url, data=data, method="POST", headers=headers)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=int(cfg.get("timeout", 180))) as r:
                obj = json.loads(r.read().decode())
            return obj["choices"][0]["message"]["content"], obj.get("usage", {}), time.time() - t0
        except Exception as e:
            last = e
            time.sleep(min(30, 3 * (attempt + 1)))
    raise RuntimeError(f"chat failed after retries: {last}")


def parse_prediction(text, rec, cands):
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
        findings = obj.get("findings", {})
    except Exception:
        return set(), {}, False
    pred = set()
    confidence = {}
    for c in cands:
        val = findings.get(c, findings.get(f"{rec['ns']}.{c}"))
        if isinstance(val, dict):
            presence = val.get("presence")
            conf = val.get("confidence")
        else:
            presence = val
            conf = None
        if presence in POS:
            pred.add(c)
        if conf in CONFIDENCE_CLASSES:
            confidence[c] = conf
    return pred, confidence, True


def score(insts):
    tp = fp = fn = 0
    ok = 0
    conf_stats = {c: collections.Counter() for c in CONFIDENCE_CLASSES}
    conf_total = 0
    for gp, pp, parse_ok, conf, cands in insts:
        ok += int(parse_ok)
        tp += len(gp & pp)
        fp += len(pp - gp)
        fn += len(gp - pp)
        for c, klass in conf.items():
            if c not in cands:
                continue
            conf_total += 1
            conf_stats[klass]["n"] += 1
            conf_stats[klass]["correct"] += int((c in gp) == (c in pp))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    by_confidence = {}
    for klass in CONFIDENCE_CLASSES:
        c = conf_stats[klass]
        if c["n"]:
            by_confidence[klass] = {"n": c["n"], "accuracy": c["correct"] / c["n"]}
    return {
        "n": len(insts),
        "parse_success": ok / len(insts) if insts else 0.0,
        "precision": precision,
        "coverage_recall": recall,
        "hallucination_rate": 1.0 - precision if tp + fp else 0.0,
        "micro_f1": f1,
        "confidence_coverage": conf_total / sum(len(cands) for *_, cands in insts) if insts else 0.0,
        "confidence_by_class": by_confidence,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=DEFAULT_TEST)
    ap.add_argument("--find", default=DEFAULT_FIND)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--models", default="gpt-5.4-mini")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--rationale", action="store_true", help="Ask for a short evidence phrase per candidate for audit/debug.")
    ap.add_argument("--retry-errors", action="store_true", help="Retry cached rows whose previous API call ended in error.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not args.run:
        args.dry_run = True

    global PROMPT_VERSION
    if args.rationale:
        PROMPT_VERSION = "clarified_confidence_v4_evidence_ctxhash"

    cfgs = load_model_config(args.config)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    missing = [m for m in models if m not in cfgs]
    if missing:
        raise SystemExit(f"models not found in {args.config}: {missing}; available={sorted(cfgs)}")

    vocab = build_vocab(args.find)
    rows = [r for r in load_rows(args.test, args.n) if vocab.get(r.get("ns"))]
    prompts = [(r, prompt_for(r, vocab[r["ns"]], args.rationale)) for r in rows]
    est_in = sum(estimate_tokens(msgs, args.max_tokens)[0] for _, msgs in prompts)
    est_out_cap = len(prompts) * args.max_tokens
    print(f"rows={len(rows)} test={args.test}")
    print(f"dry_run={args.dry_run} models={models} prompt={PROMPT_VERSION}")
    print(f"estimated prompt tokens={est_in:,}; completion cap={est_out_cap:,}; avg prompt={est_in/max(len(rows),1):.0f}")

    if args.dry_run:
        return

    os.makedirs(args.cache_dir, exist_ok=True)
    for model in models:
        cfg = cfgs[model]
        safe_model = model.replace("/", "_")
        cache_path = os.path.join(args.cache_dir, f"llm_temporal_smoke_{safe_model}_{PROMPT_VERSION}.jsonl")
        cache = load_cache(cache_path)

        def one(item):
            rec, messages = item
            cands = vocab[rec["ns"]]
            key = cache_key(model, rec)
            if key in cache and not (args.retry_errors and cache[key].get("error")):
                o = cache[key]
            else:
                try:
                    text, usage, latency = call_chat(messages, cfg, args.max_tokens, args.temperature)
                    o = {
                        "key": key,
                        "model": model,
                        "prompt": PROMPT_VERSION,
                        "hadm_id": rec.get("hadm_id"),
                        "charttime": rec.get("charttime"),
                        "ns": rec.get("ns"),
                        "exam_name": rec.get("exam_name"),
                        "text": text,
                        "usage": usage,
                        "latency_sec": latency,
                        "error": None,
                    }
                except Exception as e:
                    o = {"key": key, "model": model, "prompt": PROMPT_VERSION, "error": str(e), "usage": {}, "latency_sec": None}
                with open(cache_path, "a") as w:
                    w.write(json.dumps(o, ensure_ascii=False) + "\n")
            if o.get("error"):
                pred, conf, parse_ok = set(), {}, False
            else:
                pred, conf, parse_ok = parse_prediction(o.get("text", ""), rec, cands)
            return gold_set(rec, vocab), pred, parse_ok, conf, set(cands), o

        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            done = list(ex.map(one, prompts))
        metrics = score([(g, p, ok, conf, cands) for g, p, ok, conf, cands, _ in done])
        usage = collections.Counter()
        latencies = []
        errors = 0
        for _, _, _, _, _, o in done:
            errors += int(bool(o.get("error")))
            for k, v in (o.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] += v
            if isinstance(o.get("latency_sec"), (int, float)):
                latencies.append(o["latency_sec"])
        rep = {
            "model": model,
            "prompt": PROMPT_VERSION,
            "cache": cache_path,
            "wall_sec": time.time() - t0,
            "errors": errors,
            "avg_latency_sec": sum(latencies) / len(latencies) if latencies else None,
            "usage": dict(usage),
            "metrics": metrics,
        }
        out_path = os.path.join(args.cache_dir, f"llm_temporal_smoke_{safe_model}_{PROMPT_VERSION}_summary.json")
        with open(out_path, "w") as w:
            json.dump(rep, w, ensure_ascii=False, indent=2)
        print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
