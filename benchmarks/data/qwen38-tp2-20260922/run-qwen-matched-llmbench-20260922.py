"""Run llmbench with identical explicit chat sampling on both engines."""

import os
import runpy

import httpx


original_build_request = httpx.AsyncClient.build_request


def build_request(self, method, url, **kwargs):
    body = kwargs.get("json")
    if str(url).endswith("/v1/chat/completions") and isinstance(body, dict):
        body = dict(body)
        body.update(temperature=1.0, top_p=0.95, top_k=-1, reasoning_effort="medium")
        kwargs["json"] = body
    return original_build_request(self, method, url, **kwargs)


httpx.AsyncClient.build_request = build_request
runpy.run_path(
    os.getenv("LLM_BENCH_PATH", "/root/llm-inference-bench/llm_decode_bench.py"),
    run_name="__main__",
)
