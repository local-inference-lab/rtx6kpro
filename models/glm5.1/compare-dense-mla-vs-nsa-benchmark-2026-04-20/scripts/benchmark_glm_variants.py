#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


CONTAINER = "bf9b07d70de2"
PORT = 8001
DEFAULT_RUNS = 30
PROMPT_FILE = "/root/glm/models/glm5/compare-dense-mla-vs-nsa-benchmark-2026-04-20/prompts/testLuke5.txt"
MAX_TOKENS = 40000


def sh(cmd: str, *, check: bool = True, capture: bool = True, timeout: int | None = None):
    return subprocess.run(
        cmd,
        shell=True,
        check=check,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )


def docker_bash(script: str, *, check: bool = True, capture: bool = True, timeout: int | None = None):
    quoted = script.replace("'", "'\"'\"'")
    return sh(
        f"docker exec {CONTAINER} bash -lc '{quoted}'",
        check=check,
        capture=capture,
        timeout=timeout,
    )


def stop_server():
    docker_bash(
        r"""
set +e
pids=$(ps -eo pid,args | grep "python3 -m sglang.launch_server" | grep -v grep | awk '{print $1}')
if [ -n "$pids" ]; then
  kill -TERM $pids
fi
for i in $(seq 1 120); do
  still=$(ps -eo pid,args | grep "python3 -m sglang.launch_server" | grep -v grep | awk '{print $1}')
  if [ -z "$still" ]; then
    break
  fi
  sleep 1
done
pids=$(ps -eo pid,args | grep "python3 -m sglang.launch_server" | grep -v grep | awk '{print $1}')
if [ -n "$pids" ]; then
  kill -KILL $pids
fi
""",
        check=True,
    )


def wait_healthy(port: int, timeout_s: int = 1800) -> bool:
    script = f"""
python3 - <<'PY'
import sys, time, urllib.request
deadline = time.time() + {timeout_s}
url = "http://127.0.0.1:{port}/health"
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if r.status == 200:
                print("healthy")
                sys.exit(0)
            last = f"status={{r.status}}"
    except Exception as e:
        last = repr(e)
    time.sleep(2)
print(last or "timeout")
sys.exit(1)
PY
"""
    r = docker_bash(script, check=False, capture=True, timeout=timeout_s + 30)
    return r.returncode == 0


def start_server(variant: dict, result_dir: Path) -> int:
    log_path = f"/tmp/{variant['name']}_server.log"
    pid_path = f"/tmp/{variant['name']}_server.pid"
    launcher_path = f"/tmp/{variant['name']}_launch.sh"
    launcher = "#!/usr/bin/env bash\nset -euo pipefail\ncd /opt/sglang\n" + variant["env"] + "\nexec " + variant["cmd"] + "\n"
    local_launcher = result_dir / f"{variant['name']}_launch.sh"
    local_launcher.write_text(launcher)
    os.chmod(local_launcher, 0o755)
    sh(f"docker cp {local_launcher} {CONTAINER}:{launcher_path}")
    docker_bash(
        f"chmod +x {launcher_path}; nohup {launcher_path} > {log_path} 2>&1 & echo $! > {pid_path}; cat {pid_path}",
        check=True,
    )
    if not wait_healthy(PORT):
        tail = docker_bash(f"tail -n 200 {log_path}", check=False).stdout
        (result_dir / f"{variant['name']}_startup_failure.log").write_text(tail)
        raise RuntimeError(f"{variant['name']} failed to become healthy")
    sh(f"docker cp {CONTAINER}:{log_path} {result_dir / (variant['name'] + '_server_initial.log')}", check=False)
    pid = int(docker_bash(f"cat {pid_path}").stdout.strip())
    return pid


def run_one_eval() -> dict:
    cmd = (
        f"python3 /root/glm/models/glm5/compare-dense-mla-vs-nsa-benchmark-2026-04-20/scripts/test.py --port {PORT} --model GLM-5 -f {PROMPT_FILE} --max-tokens {MAX_TOKENS} "
        "--no-overlay --quiet --json-summary -"
    )
    r = docker_bash(cmd, check=False, capture=True, timeout=7200)
    stdout = (r.stdout or "").strip().splitlines()
    summary = None
    for line in reversed(stdout):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                summary = json.loads(line)
                break
            except json.JSONDecodeError:
                pass
    return {
        "returncode": r.returncode,
        "stdout": r.stdout,
        "stderr": r.stderr,
        "summary": summary,
    }


def health_ok() -> bool:
    return wait_healthy(PORT, timeout_s=2)


def extract_correctness(text: str) -> bool:
    return bool(re.search(r"\bestonia\b", text or "", flags=re.IGNORECASE))


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return text.strip()


def summarize_runs(runs: list[dict]) -> dict:
    completed = [r for r in runs if r.get("ok")]
    if not completed:
        return {
            "attempted_runs": len(runs),
            "completed_runs": 0,
            "correct_runs": 0,
            "correct_rate": 0.0,
        }

    def vals(key):
        return [float(r[key]) for r in completed if r.get(key) is not None]

    comp_tokens = vals("completion_tokens")
    elapsed = vals("elapsed")
    gen_elapsed = vals("gen_elapsed")
    ttft = vals("ttft")
    e2e_tps = [r["completion_tokens"] / r["elapsed"] for r in completed if r["elapsed"] > 0 and r["completion_tokens"] > 0]
    gen_tps = [r["completion_tokens"] / r["gen_elapsed"] for r in completed if r["gen_elapsed"] > 0 and r["completion_tokens"] > 0]
    return {
        "attempted_runs": len(runs),
        "completed_runs": len(completed),
        "correct_runs": sum(1 for r in completed if r["correct"]),
        "correct_rate": sum(1 for r in completed if r["correct"]) / len(completed),
        "mean_completion_tokens": statistics.mean(comp_tokens) if comp_tokens else 0.0,
        "median_completion_tokens": statistics.median(comp_tokens) if comp_tokens else 0.0,
        "mean_elapsed_s": statistics.mean(elapsed) if elapsed else 0.0,
        "mean_gen_elapsed_s": statistics.mean(gen_elapsed) if gen_elapsed else 0.0,
        "mean_ttft_s": statistics.mean(ttft) if ttft else 0.0,
        "mean_e2e_tok_s": statistics.mean(e2e_tps) if e2e_tps else 0.0,
        "mean_gen_tok_s": statistics.mean(gen_tps) if gen_tps else 0.0,
        "min_gen_tok_s": min(gen_tps) if gen_tps else 0.0,
        "max_gen_tok_s": max(gen_tps) if gen_tps else 0.0,
        "server_restarts": sum(1 for r in runs if r.get("server_restarted_before_run")),
    }


def variant_definitions():
    dense_env = """export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_ENABLE_SPEC_V2=True
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_ENABLE_DEEP_GEMM=0
export NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export NCCL_MIN_NCHANNELS=8
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1"""
    dense_cmd = """python3 -m sglang.launch_server \
  --model-path lukealonso/GLM-5.1-NVFP4 \
  --served-model-name GLM-5 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --tensor-parallel-size 8 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype bfloat16 \
  --trust-remote-code \
  --disable-shared-experts-fusion \
  --attention-backend flashinfer \
  --moe-runner-backend b12x \
  --fp4-gemm-backend b12x \
  --cuda-graph-max-bs 30 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 4 \
  --speculative-num-draft-tokens 6 \
  --speculative-eagle-topk 1 \
  --chunked-prefill-size 8192 \
  --max-running-requests 30 \
  --mem-fraction-static 0.80 \
  --host 0.0.0.0 \
  --port 8001 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
  --enable-metrics \
  --enable-piecewise-cuda-graph"""

    nsa_env = """export CUTE_DSL_ARCH=sm_120a
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_ENABLE_SPEC_V2=True
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_ENABLE_DEEP_GEMM=0
export NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export NCCL_MIN_NCHANNELS=8
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1"""
    nsa_cmd = """python3 -m sglang.launch_server \
  --model-path lukealonso/GLM-5.1-NVFP4 \
  --served-model-name GLM-5 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --tensor-parallel-size 8 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --disable-shared-experts-fusion \
  --nsa-prefill-backend b12x \
  --nsa-decode-backend b12x \
  --page-size 64 \
  --attention-backend nsa \
  --moe-runner-backend b12x \
  --fp4-gemm-backend b12x \
  --cuda-graph-max-bs 4 \
  --enable-pcie-oneshot-allreduce \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 \
  --chunked-prefill-size 8192 \
  --max-running-requests 4 \
  --mem-fraction-static 0.76 \
  --host 0.0.0.0 \
  --port 8001 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
  --json-model-override-args '{"index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}' \
  --preferred-sampling-params '{"temperature": 1.0, "top_p": 0.95}' \
  --enable-metrics"""
    return [
        {"name": "dense_mla", "env": dense_env, "cmd": dense_cmd},
        {"name": "nsa", "env": nsa_env, "cmd": nsa_cmd},
    ]


def print_progress(variant: str, idx: int, total_runs: int, rec: dict):
    status = "OK" if rec.get("ok") else "FAIL"
    correct = "correct" if rec.get("correct") else "wrong"
    toks = rec.get("completion_tokens", 0)
    gen_tps = rec.get("gen_tok_s", 0.0)
    elapsed = rec.get("elapsed", 0.0)
    print(
        f"[{variant}] run {idx:02d}/{total_runs} {status} {correct} "
        f"| completion_tokens={toks} | gen_tok_s={gen_tps:.2f} | elapsed={elapsed:.2f}s",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark dense MLA vs NSA GLM variants.")
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of evaluation runs per variant (default: {DEFAULT_RUNS}).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["dense_mla", "nsa"],
        default=["dense_mla", "nsa"],
        help="Subset of variants to benchmark.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(f"/root/glm/benchmarks/glm_dense_vs_nsa_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    print(f"results_dir={out_dir}", flush=True)

    requested = set(args.variants)
    for variant in [v for v in variant_definitions() if v["name"] in requested]:
        variant_dir = out_dir / variant["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Starting variant: {variant['name']} ===", flush=True)
        stop_server()
        start_server(variant, variant_dir)

        runs = []
        jsonl_path = variant_dir / "runs.jsonl"
        for i in range(1, args.runs + 1):
            restarted = False
            if not health_ok():
                print(f"[{variant['name']}] server unhealthy before run {i}, restarting", flush=True)
                stop_server()
                start_server(variant, variant_dir)
                restarted = True

            raw = run_one_eval()
            summary = raw["summary"]
            rec = {
                "run": i,
                "server_restarted_before_run": restarted,
                "returncode": raw["returncode"],
                "ok": False,
                "correct": False,
                "completion_tokens": None,
                "elapsed": None,
                "gen_elapsed": None,
                "ttft": None,
                "gen_tok_s": 0.0,
                "e2e_tok_s": 0.0,
                "output_text": None,
                "reasoning_text": None,
                "content_text": None,
                "final_answer": None,
                "finish_reason": None,
                "stdout_path": None,
                "stderr_path": None,
            }

            stdout_path = variant_dir / f"run_{i:02d}.stdout.txt"
            stderr_path = variant_dir / f"run_{i:02d}.stderr.txt"
            stdout_path.write_text(raw["stdout"] or "")
            stderr_path.write_text(raw["stderr"] or "")
            rec["stdout_path"] = str(stdout_path)
            rec["stderr_path"] = str(stderr_path)

            if summary and summary.get("last_result"):
                last = summary["last_result"]
                output_text = last.get("output_text") or ""
                reasoning_text = last.get("reasoning_text") or ""
                content_text = last.get("content_text") or ""
                completion_tokens = int(last.get("completion_tokens") or 0)
                elapsed = float(last.get("elapsed") or 0.0)
                gen_elapsed = float(last.get("gen_elapsed") or 0.0)
                ttft = float(last.get("ttft") or 0.0)
                finish_reason = last.get("finish_reason")
                answer_text = content_text or output_text
                final_answer = extract_final_answer(answer_text)
                correct = extract_correctness(final_answer)
                rec.update(
                    {
                        "ok": True,
                        "completion_tokens": completion_tokens,
                        "elapsed": elapsed,
                        "gen_elapsed": gen_elapsed,
                        "ttft": ttft,
                        "output_text": output_text,
                        "reasoning_text": reasoning_text,
                        "content_text": content_text,
                        "final_answer": final_answer,
                        "finish_reason": finish_reason,
                        "correct": correct,
                        "e2e_tok_s": (completion_tokens / elapsed) if elapsed > 0 else 0.0,
                        "gen_tok_s": (completion_tokens / gen_elapsed) if gen_elapsed > 0 else 0.0,
                    }
                )

            runs.append(rec)
            with jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print_progress(variant["name"], i, args.runs, rec)

        stop_server()
        summary = summarize_runs(runs)
        all_summaries[variant["name"]] = summary
        (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    compare_rows = []
    for name, s in all_summaries.items():
        compare_rows.append(
            {
                "variant": name,
                "completed_runs": s["completed_runs"],
                "correct_runs": s["correct_runs"],
                "correct_rate": round(100.0 * s["correct_rate"], 2),
                "mean_completion_tokens": round(s["mean_completion_tokens"], 2),
                "mean_ttft_s": round(s["mean_ttft_s"], 3),
                "mean_elapsed_s": round(s["mean_elapsed_s"], 3),
                "mean_gen_tok_s": round(s["mean_gen_tok_s"], 3),
                "mean_e2e_tok_s": round(s["mean_e2e_tok_s"], 3),
                "server_restarts": s["server_restarts"],
            }
        )

    final_json = out_dir / "final_summary.json"
    final_csv = out_dir / "final_summary.csv"
    final_json.write_text(json.dumps(compare_rows, indent=2, ensure_ascii=False) + "\n")
    with final_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)

    print("\n=== FINAL SUMMARY ===", flush=True)
    for row in compare_rows:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"artifacts={out_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_server()
        raise
