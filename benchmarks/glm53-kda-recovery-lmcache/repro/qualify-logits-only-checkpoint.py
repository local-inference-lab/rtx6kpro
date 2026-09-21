"""Exercise a one-token response from an exact external prompt checkpoint.

Use a dedicated endpoint with the development cache-reset API. RAM checks must
hit the complete prompt externally with zero GPU-cache hits before continuing
the truncated response. The unit test separately asserts that this logits-only
path reuses its checkpoint without exporting a duplicate or committing records.
"""

import argparse
import importlib.util
import json
import time
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--cache-metrics", required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = importlib.util.spec_from_file_location("cache_probe", args.helper)
    cache = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache)
    client = cache.Client(args.base, args.cache_metrics, 600)
    report = {"status": "running", "stages": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def request(name, payload, endpoint="/v1/chat/completions", minimum=0):
        before = client.snapshot()
        cache.require_idle(before["vllm"])
        start = time.monotonic()
        response = json.loads(client.call(args.base + endpoint, payload, post=True))
        after = client.completed_snapshot(before)
        hits = {
            key: cache.delta(before[scope], after[scope], key, required=False)
            for key, scope in (("gpu_hits", "vllm"), ("external_hits", "vllm"))
        }
        row = {
            "name": name,
            "response": response,
            "seconds": time.monotonic() - start,
            "hits": hits,
            "request_sha256": cache.fingerprint(payload),
        }
        report["stages"].append(row)
        save()
        if minimum and not (hits["gpu_hits"] == 0 and hits["external_hits"] >= minimum):
            raise AssertionError(f"Complete external prompt was not restored: {hits}")
        print(json.dumps({"name": name, "hits": hits}), flush=True)
        return response

    try:
        payload = {
            "model": "GLM-5.3-Flash",
            "messages": [
                {
                    "role": "system",
                    "content": "Reference catalog: AX = COBALT; BY = AMBER.\n"
                    + "These are ordinary fictional catalog identifiers. " * 1700,
                },
                {"role": "user", "content": "Look up AX. Return only the mapped word."},
            ],
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 20260921,
            "max_tokens": 1024,
            "return_token_ids": True,
            "cache_salt": uuid.uuid4().hex,
        }
        prime = request("cold_prompt", payload)
        cache.require_answer(prime, "COBALT")
        prompt = prime["prompt_token_ids"]
        assert len(prompt) > 10000
        client.reset_gpu()
        short = request(
            "exact_restore_one_token", dict(payload, max_tokens=1), minimum=len(prompt)
        )
        output = short["choices"][0]["token_ids"]
        assert len(output) == 1 and short["usage"]["completion_tokens"] == 1
        suffix = json.loads(
            client.call(
                args.base + "/tokenize",
                {
                    "model": payload["model"],
                    "prompt": "</think><|user|>Now look up BY. Return only the mapped word.<|assistant|><think>",
                    "add_special_tokens": False,
                },
                post=True,
            )
        )["tokens"]
        continuation = {
            key: value for key, value in payload.items() if key != "messages"
        }
        continuation["prompt"] = prompt + output + suffix
        client.reset_gpu()
        resumed = request(
            "truncated_response_continuation",
            continuation,
            "/v1/completions",
            len(prompt),
        )
        content = resumed["choices"][0]["text"].split("</think>")[-1]
        if "AMBER" not in content.upper():
            raise AssertionError(f"Unexpected continuation: {content!r}")
        report["status"] = "qualified"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
