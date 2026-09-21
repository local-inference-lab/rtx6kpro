"""Check factual answers and cache isolation across recurrent prompt-tail edits.

Use only a dedicated idle endpoint. Prime writes public synthetic request state
for a disk-restore check after both serving and cache processes restart.
"""

import argparse
import copy
import importlib.util
import json
import time
import uuid
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", required=True)
parser.add_argument("--cache-metrics", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--restore-from", type=Path)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
helper = Path("/root/vllm/worktrees/blackwell-platform-environment/tools/jovian_wheel_runtime/qualify_model_cache.py")
spec = importlib.util.spec_from_file_location("cache_probe", helper)
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
client = cache.Client(args.base, args.cache_metrics, 900)

if args.restore_from:
    prime = json.loads(args.restore_from.read_text())
    assert prime["status"] == "qualified"
    base = prime["request"]
else:
    base = {
        "model": "GLM-5.3-Flash",
        "messages": [{"role": "user", "content":
            "Reference table: AX = COBALT; BY = AMBER; CZ = MAGENTA; DQ = TURQUOISE.\n"
            "Read this fictional reference document. Answer with only the mapped word.\n"
            + "This document contains ordinary fictional item identifiers for a reference catalog. " * 3000
            + "\nLook up item AX. Return only its mapped word."}],
        "temperature": 1.0, "top_p": 0.95, "seed": 20260919,
        "max_tokens": 1024, "cache_salt": uuid.uuid4().hex,
    }
report = {"status": "running", "request": base, "stages": []}
args.output.parent.mkdir(parents=True, exist_ok=True)


def save():
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def payload(key):
    value = copy.deepcopy(base)
    value["messages"][0]["content"] = value["messages"][0]["content"].replace(
        "Look up item AX.", f"Look up item {key}.")
    return value


def run(name, value, expected, restore=None, reset=False):
    row = {"name": name, "status": "running", "request_sha256": cache.fingerprint(value)}
    report["stages"].append(row)
    if reset:
        row["reset"] = client.reset_gpu()
    before = client.snapshot()
    cache.require_idle(before["vllm"])
    row["before"] = before
    save()
    started = time.monotonic()
    response = json.loads(client.call(args.base + "/v1/chat/completions", value, post=True))
    after = client.completed_snapshot(before)
    row.update(seconds=time.monotonic() - started, response=response, after=after)
    content = response["choices"][0]["message"].get("content") or ""
    row["cache_delta"] = {
        key: cache.delta(before[scope], after[scope], key, required=False)
        for key, scope in (("gpu_hits", "vllm"), ("external_hits", "vllm"), ("l2_loaded", "cache"))
    }
    save()
    cache.require_answer(response, expected)
    hit = row["cache_delta"]
    if restore == "native":
        assert hit["gpu_hits"] > 10000, hit
    elif restore == "external":
        assert hit["external_hits"] > 10000 and hit["gpu_hits"] == 0, hit
    elif restore == "cold":
        assert hit["external_hits"] == hit["gpu_hits"] == 0, hit
    if args.restore_from:
        assert hit["l2_loaded"] > 0, hit
        for scope in ("vllm", "cache"):
            assert before[scope]["process_start_time_seconds"] != prime["stages"][0]["before"][scope]["process_start_time_seconds"]
    row["status"] = "qualified"
    save()
    print(json.dumps({k: row[k] for k in ("name", "seconds", "cache_delta", "status")}), flush=True)


try:
    if args.restore_from:
        run("disk_tail_DQ", payload("DQ"), "TURQUOISE", "external")
    else:
        run("cold_AX", base, "COBALT", "cold")
        run("native_tail_BY", payload("BY"), "AMBER", "native")
        run("external_tail_CZ", payload("CZ"), "MAGENTA", "external", reset=True)
        fresh = payload("CZ")
        fresh["cache_salt"] = uuid.uuid4().hex
        run("cold_control_CZ", fresh, "MAGENTA", "cold")
        early_edit = payload("AX")
        early_edit["messages"][0]["content"] = early_edit["messages"][0]["content"].replace("AX = COBALT", "AX = SILVER")
        run("early_edit_misses", early_edit, "SILVER", "cold", reset=True)
    report["status"] = "qualified"
except Exception as error:
    report.update(status="failed", error=repr(error))
    raise
finally:
    save()
