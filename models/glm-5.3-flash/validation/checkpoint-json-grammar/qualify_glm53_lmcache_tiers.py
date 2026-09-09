#!/usr/bin/env python3
"""Qualify vLLM APC, LMCache DRAM, and LMCache filesystem prefix hits.

One unique GLM-5.3 prompt is executed four times. The first request computes
the prompt, the second must use vLLM's local prefix cache, the third runs after
an APC reset and must use LMCache L1, and the fourth runs after both an APC
reset and an LMCache L1 clear and must recover the prefix from LMCache L2.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

_SOURCES = ("external_kv_transfer", "local_compute", "local_cache_hit")


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def reset_prefix_cache(base_url: str, timeout: float = 60.0) -> None:
    """Reset vLLM's local prefix cache after asynchronous stores release blocks."""
    deadline = time.monotonic() + timeout
    while True:
        result = request_json(f"{base_url}/reset_prefix_cache", {})
        if result.get("success") is True:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("vLLM did not release blocks for a prefix-cache reset")
        time.sleep(0.1)


def metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        text = response.read().decode()
    name = "vllm:prompt_tokens_by_source_total"
    result = {source: 0.0 for source in _SOURCES}
    for line in text.splitlines():
        if not line.startswith(name + "{"):
            continue
        for source in _SOURCES:
            if f'source="{source}"' in line:
                result[source] = float(line.rsplit(" ", 1)[1])
    return result


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, int]:
    return {source: round(after[source] - before[source]) for source in _SOURCES}


def wait_for_store(
    cache_url: str,
    timeout: float = 60.0,
    *,
    minimum_checkpoint_count: int | None = None,
) -> dict[str, Any]:
    """Wait for drained writes, optionally requiring atomic manifest publication."""
    deadline = time.monotonic() + timeout
    while True:
        status = request_json(f"{cache_url}/status")
        storage = status["storage_manager"]
        l1 = storage["l1_manager"]
        controller = storage["store_controller"]
        checkpoint = status.get("recurrent_checkpoints", {})
        checkpoint_ready = minimum_checkpoint_count is None or (
            checkpoint.get("published_generations", 0) >= minimum_checkpoint_count
            and checkpoint.get("pending_generations") == 0
            and checkpoint.get("store_leases") == 0
            and checkpoint.get("retrieve_leases") == 0
        )
        if (
            l1["write_locked_count"] == 0
            and l1["temporary_count"] == 0
            and controller["pending_keys_count"] == 0
            and controller["in_flight_task_count"] == 0
            and checkpoint_ready
        ):
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError("LMCache did not drain its pending stores")
        time.sleep(0.1)


def execute(
    base_url: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    before = metrics(base_url)
    response = request_json(f"{base_url}/v1/completions", payload)
    after = metrics(base_url)
    return response, metric_delta(before, after)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5051")
    parser.add_argument("--cache-url", default="http://127.0.0.1:8085")
    parser.add_argument("--model", default="GLM-5.3-Flash-NVFP4")
    parser.add_argument("--sentence-repetitions", type=int, default=1250)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nonce = time.time_ns()
    prompt = (
        f"LMCache tier qualification nonce {nonce}. "
        + "violet engines compare quiet harbor maps under winter stars. "
        * args.sentence_repetitions
        + "Select the single most likely next token:"
    )
    payload = {
        "model": args.model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 1,
        "seed": 29,
    }

    reset_prefix_cache(args.base_url)
    cold, cold_delta = execute(args.base_url, payload)
    status = wait_for_store(args.cache_url)

    apc, apc_delta = execute(args.base_url, payload)

    reset_prefix_cache(args.base_url)
    l1, l1_delta = execute(args.base_url, payload)

    reset_prefix_cache(args.base_url)
    request_json(f"{args.cache_url}/cache/clear", {})
    l2, l2_delta = execute(args.base_url, payload)

    chunk_size = int(status["chunk_size"])
    checks = {
        "cold_used_local_compute": cold_delta["local_compute"] >= chunk_size,
        "apc_used_local_cache": apc_delta["local_cache_hit"] >= chunk_size,
        "apc_avoided_external_cache": apc_delta["external_kv_transfer"] == 0,
        "l1_used_external_cache": l1_delta["external_kv_transfer"] >= chunk_size,
        "l2_used_external_cache": l2_delta["external_kv_transfer"] >= chunk_size,
    }
    passed = all(checks.values())
    result = {
        "status": "qualified" if passed else "failed",
        "conditions": {
            "model": args.model,
            "cache_chunk_tokens": chunk_size,
            "prompt_tokens": cold["usage"]["prompt_tokens"],
            "temperature": 0,
            "max_tokens": 1,
        },
        "checks": checks,
        "requests": {
            "cold_compute": {"id": cold["id"], "prompt_sources": cold_delta},
            "vllm_apc": {"id": apc["id"], "prompt_sources": apc_delta},
            "lmcache_l1": {"id": l1["id"], "prompt_sources": l1_delta},
            "lmcache_l2": {"id": l2["id"], "prompt_sources": l2_delta},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
