"""Qualify GLM recovery with text checkpoint RAM/disk restores and serving.

The named qualification container must already be started on remote GPUs 5–8.
Only that container is restarted. Its LMCache sidecar runs in the same container.
Image-bearing requests are checked for generation, not external checkpoint hits:
the existing boundary-checkpoint adapter excludes multimodal requests.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mtp2", "mtp4", "dflash4"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    helper = Path(
        "/root/vllm/worktrees/blackwell-tp4-memory-validation/tools/jovian_wheel_runtime/qualify_model_cache.py"
    )
    spec = importlib.util.spec_from_file_location(
        "handoff", scripts / "qualify-public-spark-handoff.py"
    )
    handoff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handoff)
    name = "glm53-recovery-cache-" + args.mode
    identity = json.loads(handoff.remote(["docker", "inspect", name]))[0]
    devices = set(identity["HostConfig"]["DeviceRequests"][0]["DeviceIDs"])
    expected = {"5", "6"} if args.mode == "mtp2" else {"5", "6", "7", "8"}
    if (
        devices != expected
        or identity["Config"]["Labels"].get("lil.qualification")
        != "glm53-kda-recovery-lmcache"
    ):
        raise ValueError("Container does not match the assigned qualification GPUs")
    container_id = identity["Id"]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / (
        "restore-controller.json" if args.restore_only else "controller.json"
    )
    if record_path.exists():
        raise FileExistsError(record_path)
    record = {
        "status": "running",
        "container_id": container_id,
        "mode": args.mode,
        "commands": [],
    }
    session = requests.Session()
    session.trust_env = False
    base = "http://192.168.0.115:5571"

    def save():
        record_path.write_text(json.dumps(record, indent=2) + "\n")

    def run(label, argv, **kwargs):
        print(json.dumps({"stage": label, "mode": args.mode}), flush=True)
        row = {"label": label, "argv": argv, "started": time.time()}
        record["commands"].append(row)
        save()
        with (out / (label + ".log")).open("w") as log:
            completed = subprocess.run(
                argv, stdout=log, stderr=subprocess.STDOUT, timeout=2400, **kwargs
            )
        row.update(returncode=completed.returncode, finished=time.time())
        save()
        completed.check_returncode()

    def ready():
        for attempt in range(480):
            state = json.loads(handoff.remote(["docker", "inspect", container_id]))[0]
            if not state["State"]["Running"]:
                raise RuntimeError("Qualification container stopped")
            try:
                if session.get(base + "/health", timeout=3).status_code == 200:
                    return
            except requests.RequestException:
                pass
            if attempt % 12 == 0:
                print(
                    json.dumps({"waiting": "model startup", "mode": args.mode}),
                    flush=True,
                )
            time.sleep(5)
        raise TimeoutError("Qualification endpoint did not become ready")

    env = dict(os.environ, DOCKER_HOST="ssh://root@192.168.66.4")
    try:
        ready()
        with handoff.private_cache_metrics(15572) as metrics:
            cache_common = [
                "--dedicated-endpoint",
                "--base-url",
                base,
                "--cache-metrics-url",
                metrics,
                "--output-dir",
                str(out / "cache-text"),
                "--timeout",
                "600",
            ]
            tail_common = [
                sys.executable,
                str(scripts / "probe-checkpoint-tail-correctness.py"),
                "--base",
                base,
                "--cache-metrics",
                metrics,
            ]
            response_common = [
                sys.executable,
                str(scripts / "qualify-recovery-response-checkpoint.py"),
                "--base",
                base,
                "--cache-metrics",
                metrics,
                "--helper",
                str(helper),
            ]
            if not args.restore_only:
                run(
                    "cache-prime",
                    [
                        sys.executable,
                        str(helper),
                        "prime",
                        *cache_common,
                        "--container",
                        container_id,
                        "--model",
                        "GLM-5.3-Flash",
                        "--persistent",
                    ],
                    env=env,
                )
                run(
                    "tail-prime",
                    tail_common + ["--output", str(out / "tail-prime.json")],
                )
                run(
                    "response-prime",
                    response_common + ["--output", str(out / "response-prime.json")],
                )
                run(
                    "restart",
                    [
                        "ssh",
                        "-o",
                        "HostKeyAlias=192.168.66.4",
                        "root@192.168.0.115",
                        "docker restart --timeout 60 " + container_id,
                    ],
                )
                ready()
            run(
                "cache-disk",
                [sys.executable, str(helper), "restore", *cache_common],
                env=env,
            )
            run(
                "tail-disk",
                tail_common
                + [
                    "--restore-from",
                    str(out / "tail-prime.json"),
                    "--output",
                    str(out / "tail-disk.json"),
                ],
            )
            # Each restore must prove external reuse rather than residual GPU APC.
            cache_spec = importlib.util.spec_from_file_location("cache_probe", helper)
            cache = importlib.util.module_from_spec(cache_spec)
            cache_spec.loader.exec_module(cache)
            cache.Client(base, metrics, 600).reset_gpu()
            run(
                "response-disk",
                response_common
                + [
                    "--restore-from",
                    str(out / "response-prime.json"),
                    "--output",
                    str(out / "response-disk.json"),
                ],
            )
        dflash = args.mode == "dflash4"
        run(
            "serving",
            [
                sys.executable,
                str(scripts / "qualify-model-endpoint.py"),
                "--container",
                container_id,
                "--docker-ssh-host",
                "root@192.168.66.4",
                "--base",
                base,
                "--model",
                "GLM-5.3-Flash",
                "--vision",
                "--expected-method",
                "dflash" if dflash else "mtp",
                "--expected-draft-tokens",
                "7" if dflash else "3",
                "--decode-runs",
                "3",
                "--prefill-runs",
                "1",
                "--concurrency",
                "1,4" if args.mode == "mtp2" else "1,8",
                "--output-dir",
                str(out / "serving"),
            ],
        )
        serving = json.loads((out / "serving/qualification.json").read_text())
        if (
            not serving["all_smoke_passed"]
            or serving["invalid_decode_cells"]
            or serving["missing_decode_cells"]
        ):
            raise AssertionError(
                "Serving qualification contains failed or missing cells"
            )
        record["status"] = "qualified"
    except Exception as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
