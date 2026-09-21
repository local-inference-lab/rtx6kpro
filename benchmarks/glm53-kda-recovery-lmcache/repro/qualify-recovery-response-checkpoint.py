"""Prove external reuse of a generated response, not only its prompt checkpoint.

Run against a dedicated GLM endpoint with its cache metrics reachable. Token IDs
preserve the exact generated prefix even when the chat parser separates reasoning.
The restored continuation must answer a different factual question. A cold-salt
control verifies that the answer is independent of the restore path.
"""

import argparse
import copy
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
    parser.add_argument("--restore-from", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = importlib.util.spec_from_file_location("cache_probe", args.helper)
    cache = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache)
    client = cache.Client(args.base, args.cache_metrics, 900)
    report = {"status": "running", "stages": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def request(label, payload, endpoint, expected, min_external=0):
        before = client.snapshot()
        cache.require_idle(before["vllm"])
        started = time.monotonic()
        response = json.loads(client.call(args.base + endpoint, payload, post=True))
        after = client.completed_snapshot(before)
        choice = response["choices"][0]
        content = choice.get("message", {}).get("content")
        if content is None:
            content = choice.get("text", "").split("</think>")[-1]
        hits = {
            key: cache.delta(before[scope], after[scope], key, required=False)
            for key, scope in (
                ("gpu_hits", "vllm"),
                ("external_hits", "vllm"),
                ("l2_loaded", "cache"),
            )
        }
        row = {
            "name": label,
            "response": response,
            "seconds": time.monotonic() - started,
            "before": before,
            "after": after,
            "hits": hits,
        }
        report["stages"].append(row)
        save()
        if expected not in content.upper() or choice["finish_reason"] != "stop":
            raise AssertionError(f"Expected {expected}: {content!r}")
        if min_external and not (
            hits["external_hits"] >= min_external and hits["gpu_hits"] == 0
        ):
            raise AssertionError(f"The generated response was not restored: {hits}")
        print(json.dumps({k: row[k] for k in ("name", "seconds", "hits")}), flush=True)
        return response

    try:
        if args.restore_from:
            prime = json.loads(args.restore_from.read_text())
            assert prime["status"] == "qualified"
            continuation = prime["continuation"]
            minimum = prime["minimum_response_tokens"]
            report.update(continuation=continuation, minimum_response_tokens=minimum)
            request(
                "disk_response_continuation",
                continuation,
                "/v1/completions",
                "AMBER",
                minimum,
            )
            assert report["stages"][-1]["hits"]["l2_loaded"] > 0
        else:
            payload = {
                "model": "GLM-5.3-Flash",
                "messages": [
                    {
                        "role": "system",
                        "content": "Reference catalog: AX maps to COBALT; BY maps to AMBER.\n"
                        + "These are fictional item identifiers in a catalog. " * 1600,
                    },
                    {
                        "role": "user",
                        "content": "Look up AX. Return only the mapped word.",
                    },
                ],
                "temperature": 1.0,
                "top_p": 0.95,
                "seed": 20260921,
                "max_tokens": 1024,
                "return_token_ids": True,
                "cache_salt": uuid.uuid4().hex,
            }
            first = request(
                "cold_generated_response", payload, "/v1/chat/completions", "COBALT"
            )
            prompt = first["prompt_token_ids"]
            output = first["choices"][0]["token_ids"]
            assert (
                isinstance(prompt, list)
                and isinstance(output, list)
                and len(output) > 4
            )
            suffix = json.loads(
                client.call(
                    args.base + "/tokenize",
                    {
                        "model": payload["model"],
                        "prompt": "<|user|>Now look up BY. Return only the mapped word.<|assistant|><think>",
                        "add_special_tokens": False,
                    },
                    post=True,
                )
            )["tokens"]
            continuation = {k: v for k, v in payload.items() if k != "messages"}
            continuation["prompt"] = prompt + output + suffix
            # The final emitted token has not yet passed through target forward.
            minimum = len(prompt) + len(output) - 1
            report.update(continuation=continuation, minimum_response_tokens=minimum)
            client.reset_gpu()
            request(
                "ram_response_continuation",
                continuation,
                "/v1/completions",
                "AMBER",
                minimum,
            )
            cold = copy.deepcopy(continuation)
            cold["cache_salt"] = uuid.uuid4().hex
            request("cold_continuation_control", cold, "/v1/completions", "AMBER")
            assert report["stages"][-1]["hits"]["external_hits"] == 0
        report["status"] = "qualified"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
