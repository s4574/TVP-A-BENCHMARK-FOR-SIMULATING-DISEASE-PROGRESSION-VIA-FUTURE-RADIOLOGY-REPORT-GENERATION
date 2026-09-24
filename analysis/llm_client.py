"""Minimal OpenAI-compatible client (stdlib urllib) for an OpenAI-compatible endpoint. Reads llm_api/gpt_56_sol.yaml."""
import json, re, urllib.request, urllib.error, time

CFG = "${VP_ROOT}/llm_api/gpt_56_sol.yaml"

def _load_cfg(path=CFG):
    # 极简 yaml 解析(仅本文件结构: models: -> <name>: -> key: value)
    models, cur = {}, None
    with open(path) as f:
        for line in f:
            if re.match(r"^\s*#", line) or not line.strip():
                continue
            m = re.match(r"^  ([\w.\-]+):\s*$", line)
            if m:
                cur = m.group(1); models[cur] = {}; continue
            m = re.match(r"^    ([\w_]+):\s*(.+?)\s*$", line)
            if m and cur:
                k, v = m.group(1), m.group(2)
                if re.fullmatch(r"-?\d+", v): v = int(v)
                elif re.fullmatch(r"-?\d*\.\d+", v): v = float(v)
                models[cur][k] = v
    return models

MODELS = _load_cfg()

def chat_full(messages, model="gpt-5.6-sol", temperature=0.1, max_tokens=4096, response_json=False):
    """Return (content, usage_dict). usage has prompt_tokens/completion_tokens[/prompt_tokens_details.cached_tokens]."""
    cfg = MODELS[model]
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = {"model": cfg.get("model_name", model), "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(max(cfg.get("max_retry", 3), 6)):
        try:
            req = urllib.request.Request(url, data=data, method="POST", headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + cfg["api_key"]})
            with urllib.request.urlopen(req, timeout=cfg.get("timeout", 120)) as r:
                obj = json.loads(r.read().decode())
            return obj["choices"][0]["message"]["content"], obj.get("usage", {})
        except Exception as e:
            last = e
            time.sleep(min(30, 3 * (attempt + 1)))   # longer backoff for 429/503
    raise RuntimeError(f"chat failed after retries: {last}")

def chat(messages, model="gpt-5.6-sol", temperature=0.1, max_tokens=4096, response_json=False):
    return chat_full(messages, model, temperature, max_tokens, response_json)[0]

if __name__ == "__main__":
    import sys
    mdl = sys.argv[1] if len(sys.argv) > 1 else "gpt-5.6-sol"
    print("models in cfg:", list(MODELS))
    print("ping", mdl, "->", chat([{"role": "user", "content": "Reply with exactly: PONG"}],
                                   model=mdl, max_tokens=16))
