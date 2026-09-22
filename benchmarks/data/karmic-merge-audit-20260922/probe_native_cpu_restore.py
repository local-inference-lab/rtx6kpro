"""Check Qwen text correctness and CPU KV offload after GPU eviction."""

import hashlib
import json
import os
import re
import time

import requests


BASE = os.getenv("OFFLOAD_BASE", "http://192.168.0.115:5074")
MODEL = "Qwen3.8-Flash-Next"
CLIENT = requests.Session()
CLIENT.trust_env = False


def metrics():
    raw = CLIENT.get(BASE + "/metrics", timeout=15).text
    names = (
        "vllm:external_prefix_cache_queries_total",
        "vllm:external_prefix_cache_hits_total",
        "vllm:prefix_cache_hits_total",
        "vllm:kv_cache_usage_perc",
    )
    return {
        name: float(line.rsplit(" ", 1)[1])
        for line in raw.splitlines()
        for name in names
        if line.startswith(name + "{")
    }


def system_text(marker):
    return f"The access code is {access_code(marker)}. Remember it exactly.\n" + "\n".join(
        f"Fact {index:04d} for {marker}: lanterns are blue and rivers are long."
        for index in range(340)
    )


def access_code(marker):
    if os.getenv("STRICT_RECALL") != "1" or marker == prime_marker:
        return "TEAL-428"
    # Every eviction has a different answer, so a stale state cannot satisfy
    # the restored request merely by recalling another request's access code.
    return "COPPER-" + hashlib.sha256(marker.encode()).hexdigest()[:8].upper()


def chat(marker):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_text(marker)},
            {"role": "user", "content": os.getenv("PROBE_QUESTION", "What is 11 plus 2? Answer with just the number.")},
        ],
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 256,
    }
    started = time.monotonic()
    response = CLIENT.post(BASE + "/v1/chat/completions", json=body, timeout=300)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    response.raise_for_status()
    answer = response.json()
    return {
        "elapsed_s": round(time.monotonic() - started, 3),
        "usage": answer.get("usage"),
        "message": answer["choices"][0].get("message"),
        "finish_reason": answer["choices"][0].get("finish_reason"),
    }


eviction_count = int(os.getenv("EVICT_COUNT", "2"))
prime_marker = os.getenv("PRIME_MARKER", "alpha")
evict_prefix = os.getenv("EVICT_PREFIX", "unique")
stages = [("prime", prime_marker)]
stages.extend((f"evict-{index}", f"{evict_prefix}-{index}") for index in range(eviction_count))
stages.append(("restore", prime_marker))
for stage, marker in stages:
    if stage == "restore" and os.getenv("RESET_NATIVE") == "1":
        reset = CLIENT.post(
            BASE + "/reset_prefix_cache?reset_external=false&reset_running_requests=false",
            json={}, timeout=30,
        )
        reset.raise_for_status()
        if not reset.json().get("success"):
            raise RuntimeError(reset.text)
        print(json.dumps({"stage": "reset", "result": reset.json()}), flush=True)
    before = metrics()
    output = chat(marker)
    time.sleep(2)
    after = metrics()
    expected_code = access_code(marker)
    content = output["message"].get("content") or ""
    recall_passed = expected_code in content and bool(re.search(r"\b13\b", content))
    print(json.dumps({"stage": stage, "prompt_tokens": output["usage"]["prompt_tokens"],
                      "expected_code": expected_code, "recall_passed": recall_passed,
                      "cached_tokens": output["usage"]["prompt_tokens_details"].get("cached_tokens"),
                      "content": output["message"].get("content"),
                      "reasoning": output["message"].get("reasoning"),
                      "finish_reason": output["finish_reason"],
                      "external_hits_delta": after["vllm:external_prefix_cache_hits_total"]-before["vllm:external_prefix_cache_hits_total"],
                      "native_hits_delta": after["vllm:prefix_cache_hits_total"]-before["vllm:prefix_cache_hits_total"]}), flush=True)
    if os.getenv("STRICT_RECALL") == "1" and not recall_passed:
        raise AssertionError(f"{stage}: expected {expected_code} and 13, got {content!r}")
    if stage == "restore" and os.getenv("STRICT_RECALL") == "1":
        assert after["vllm:external_prefix_cache_hits_total"] > before["vllm:external_prefix_cache_hits_total"]
        assert after["vllm:prefix_cache_hits_total"] == before["vllm:prefix_cache_hits_total"]
print(json.dumps({"health": CLIENT.get(BASE + "/health", timeout=15).ok}), flush=True)
