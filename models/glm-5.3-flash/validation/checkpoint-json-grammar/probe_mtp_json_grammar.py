"""Record concurrent MTP JSON-schema outcomes with optional external replays.

The schema specifies output shape, not the lookup answer. Each round has a
fresh cache salt and four requests with shared instructions and distinct lookup
questions. HTTP failures and malformed or incorrect answers remain in the
report; an error stops the probe after the active round drains.
With --cache-url, every cold round is followed by GPU-cache-reset replays and
an independent-salt request. Admission failures remain valid probe outcomes;
this diagnostic does not require every prompt to have a published checkpoint.
"""

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qualification_devices import require_container_devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--cache-url")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    container = json.loads(
        subprocess.check_output(["docker", "inspect", args.container])
    )[0]
    require_container_devices(container)
    if not args.container.startswith("glm53-warmup-retention-"):
        raise ValueError("Use a dedicated qualification container")
    report = {
        "status": "running",
        "image_id": container["Image"],
        "container": args.container,
        "environment": [
            x
            for x in container["Config"]["Env"]
            if x.startswith(("TP=", "DCP=", "CACHE_MODE=", "MTP_DEPTH=", "SPECULATOR="))
        ],
        "rounds": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cache_url:
        from qualify_glm53_lmcache_tiers import (
            metric_delta,
            metrics,
            reset_prefix_cache,
            wait_for_store,
        )
    system = (
        "Read the literal document and return only the requested value.\n"
        + "\n".join(f"Record {i} has value VALUE_{i * 7 + 11}." for i in range(1200))
        + '\nReturn only a JSON object {"value":"VALUE_<recorded number>"}. Copy the requested record\'s value exactly, without explanations.'
    )
    document = "Reference notes:\n" + "\n".join(
        f"Reference {i} concerns ordinary warehouse inventory." for i in range(1200)
    )
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "literal_record_value",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "pattern": "^VALUE_[0-9]+$"}
                },
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }

    def run(position, salt):
        payload = {
            "model": "GLM-5.3-Flash-NVFP4",
            "temperature": 0,
            "max_tokens": 128,
            "cache_salt": salt,
            "response_format": schema,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": document + f"\nWhat is the value of record {position}?",
                },
            ],
        }
        request = urllib.request.Request(
            args.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        result = {
            "position": position,
            "expected": {"value": f"VALUE_{position * 7 + 11}"},
            "salt": salt,
        }
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result["http_status"] = response.status
                result["response"] = json.load(response)
            content = result["response"]["choices"][0]["message"].get("content") or ""
            try:
                result["exact_answer"] = json.loads(content) == result["expected"]
            except json.JSONDecodeError:
                result["exact_answer"] = False
        except urllib.error.HTTPError as error:
            result.update(
                http_status=error.code,
                error_body=error.read().decode(),
                exact_answer=False,
            )
        result["elapsed_seconds"] = time.monotonic() - started
        return result

    for index in range(args.rounds):
        salt = f"mtp-json-contract-{time.time_ns()}"
        if args.cache_url:
            wait_for_store(args.cache_url, minimum_checkpoint_count=0)
            reset_prefix_cache(args.base_url)
        with ThreadPoolExecutor(max_workers=4) as executor:
            jobs = [
                executor.submit(run, position, salt)
                for position in (111, 333, 777, 1111)
            ]
            results = [job.result() for job in jobs]
        for result in results:
            result["phase"] = "cold"
        report["rounds"].append({"index": index, "results": results})
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if args.cache_url and all(x["http_status"] == 200 for x in results):
            wait_for_store(args.cache_url, minimum_checkpoint_count=0)
            for position in (111, 333, 777, 1111, 111):
                isolated = len(results) == 8
                reset_prefix_cache(args.base_url)
                before = metrics(args.base_url)
                result = run(
                    position, salt + "-independent-tenant" if isolated else salt
                )
                result["phase"] = "isolated" if isolated else "replay"
                result["prompt_sources"] = metric_delta(before, metrics(args.base_url))
                results.append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                wait_for_store(args.cache_url, minimum_checkpoint_count=0)
                if result["http_status"] != 200:
                    break
        print(
            json.dumps(
                {
                    "round": index,
                    "outcomes": [
                        {
                            "position": x["position"],
                            "status": x["http_status"],
                            "exact": x["exact_answer"],
                        }
                        for x in results
                    ],
                }
            ),
            flush=True,
        )
        if any(x["http_status"] != 200 for x in results):
            break
    results = [x for r in report["rounds"] for x in r["results"]]
    report["http_errors"] = sum(x["http_status"] != 200 for x in results)
    report["incorrect_answers"] = sum(not x["exact_answer"] for x in results)
    report["status"] = "failed" if report["incorrect_answers"] else "qualified"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "http_errors": report["http_errors"],
                "requests": len(results),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
