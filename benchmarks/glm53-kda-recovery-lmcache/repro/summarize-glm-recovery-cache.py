"""Export compact, reproducible GLM recovery evidence without full metric dumps.

Every input receipt is content-addressed. The summary preserves all measured
decode windows and repetition flags; it does not promote smoke tests into a
general accuracy claim or discard slower samples.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    inputs = {}

    def read(path):
        raw = path.read_bytes()
        inputs[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def response_text(response):
        choices = response.get("choices", [])
        if not choices:
            return ""
        item = choices[0]
        return item.get("message", {}).get("content") or item.get("text") or ""

    caches = []
    for path in sorted(root.rglob("*.json")):
        if "serving" in str(path) or path.name.endswith(".resume.json"):
            continue
        receipt = read(path)
        if not isinstance(receipt, dict) or not isinstance(receipt.get("stages"), list):
            continue
        stages = []
        for stage in receipt["stages"]:
            stages.append(
                {
                    "name": stage["name"],
                    "status": stage.get("status"),
                    "cache": stage.get("evidence")
                    or stage.get("cache_delta")
                    or stage.get("hits"),
                    "answer": response_text(stage.get("response", {})),
                    "seconds": stage.get("seconds"),
                    "request_sha256": stage.get("request_sha256"),
                }
            )
        caches.append(
            {
                "file": str(path.relative_to(root)),
                "status": receipt.get("status"),
                "error": receipt.get("error"),
                "stages": stages,
            }
        )

    serving = []
    keys = (
        "concurrency",
        "context_tokens",
        "aggregate_tps",
        "server_steps_per_s",
        "server_accept_len_effective",
        "server_spec_accept_rate",
        "num_errors",
        "failure_reason",
        "loop_detected",
        "loop_diagnostics",
        "underfilled",
        "warmup_timed_out",
        "measurement_seconds",
        "measurement_wall_seconds",
        "request_count",
        "effective_concurrency",
    )
    for path in sorted(root.rglob("qualification.json")):
        qualification = read(path)
        if "verified_speculative_config" not in qualification:
            continue
        rows = []
        for sample in sorted(path.parent.glob("decode-*.json")):
            if sample.name.endswith(".resume.json"):
                continue
            data = read(sample)
            rows += [
                {
                    "file": str(sample.relative_to(root)),
                    **{key: cell.get(key) for key in keys},
                }
                for cell in data.get("results", [])
            ]
        summaries = {}
        for cc in sorted({row["concurrency"] for row in rows}):
            cells = [row for row in rows if row["concurrency"] == cc]
            summaries[str(cc)] = {
                key: statistics.median(
                    row[key] for row in cells if row[key] is not None
                )
                for key in (
                    "aggregate_tps",
                    "server_steps_per_s",
                    "server_accept_len_effective",
                )
            }
        prefills = []
        for sample in sorted(path.parent.glob("prefill*.json")):
            if sample.name.endswith(".resume.json"):
                continue
            data = read(sample)
            prefills.append(
                {"file": str(sample.relative_to(root)), "results": data.get("prefill")}
            )
        serving.append(
            {
                "file": str(path.relative_to(root)),
                "status": qualification.get("status"),
                "all_smoke_passed": qualification.get("all_smoke_passed"),
                "smoke": [
                    {
                        "name": item["name"],
                        "passed": item["passed"],
                        "answer": response_text(item.get("response", {})),
                    }
                    for item in qualification.get("smoke", [])
                ],
                "speculation": qualification["verified_speculative_config"],
                "logical_kv_capacity_tokens": qualification.get(
                    "logical_kv_capacity_tokens"
                ),
                "invalid_decode_cells": qualification.get("invalid_decode_cells"),
                "missing_decode_cells": qualification.get("missing_decode_cells"),
                "commands": qualification.get("commands", []),
                "bench_sha256": qualification.get("bench_sha256"),
                "decode_samples": rows,
                "decode_medians": summaries,
                "prefill": prefills,
            }
        )
    output = {
        "scope": "Bounded kernel and serving checks; not a general model accuracy evaluation.",
        "cache_probes": caches,
        "serving": serving,
        "raw_receipt_sha256": inputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cache_probes": len(caches),
                "serving_modes": len(serving),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
