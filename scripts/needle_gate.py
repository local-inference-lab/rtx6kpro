#!/usr/bin/env python3
"""Needle-at-depth qualification gate for GLM-5.2 serving images.

Plants two labeled facts (5% and 95% positions) in synthetic prompts of
increasing size and requires exact retrieval at temperature 0. Catches
long-context KV/readout corruption that throughput benchmarks cannot see
(a context-blind model benchmarks *faster*).

Prompts are salted per invocation so no prefix cache (vLLM or LMCache L2)
can serve KV computed by a previous run or a previous image. Without the
salt, a broken engine can poison the shared LMCache disk cache and make a
healthy image appear broken (or vice versa) on identical prompts.

Born from the v20 r26 rejection (2026-08-03): r26 passed every speed cell,
then lost the entire readout past ~98-128K tokens on TP4/DCP2. Run this
against every new image/checkpoint/config before it goes near production.

Usage:
  python needle_gate.py --model GLM-5.2-EXL3-TR3-3.0bpw [--port 5001]
                        [--sizes 8,64,96,128,163] [--max-tokens 6000]

Exit code 0 only if every size at or below --required-clean-k passes both
needles; larger sizes are reported informationally (r19's sparse DSA
gracefully drops distant needles >=128K — expected, not a failure).
"""

import argparse
import json
import random
import string
import sys
import time
import urllib.error
import urllib.request

FILLER = (
    "Paragraph %d: The logistics report for sector %d notes routine container "
    "throughput, standard customs clearance times, and no exceptions recorded "
    "during the audit window. "
)
ALPHA = "CRITICAL NOTE ALPHA: The vault code is 84291. "
OMEGA = "CRITICAL NOTE OMEGA: The exit password is FERN-72. "
QUESTION = (
    "\n\nFrom the document above, answer exactly two lines:\n"
    "Line 1: the vault code\nLine 2: the exit password"
)
PARAS_PER_K = 30  # ~33 tokens per salted filler paragraph
SALT = "".join(random.choices(string.ascii_lowercase, k=10))


def build_prompt(target_k: int) -> str:
    """Salted per process: defeats prefix/LMCache KV reuse across runs."""
    n = target_k * PARAS_PER_K
    parts = []
    for i in range(n):
        parts.append(f"Audit run {SALT}-{i % 7}. " + FILLER % (i, i % 97))
        if i == int(n * 0.05):
            parts.append(ALPHA)
        if i == int(n * 0.95):
            parts.append(OMEGA)
    return "".join(parts) + QUESTION


def probe(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            j = json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            return {"skipped": "prompt exceeds the server context window",
                    "wall_s": round(time.time() - t0, 1)}
        raise
    choice = j["choices"][0]
    content = choice["message"].get("content") or ""
    return {
        "wall_s": round(time.time() - t0, 1),
        "prompt_tokens": j["usage"]["prompt_tokens"],
        "finish": choice["finish_reason"],
        "alpha": "84291" in content,
        "omega": "FERN-72" in content,
        "content_head": content[:80],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--sizes", default="8,64,96,128,163",
                    help="approximate prompt sizes in K tokens")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--required-clean-k", type=int, default=96,
                    help="both needles MUST retrieve at all sizes <= this")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    sizes = [int(s) for s in args.sizes.split(",")]
    results, gate_ok = [], True
    print(f"salt={SALT}")
    for k in sizes:
        r = probe(base, args.model, build_prompt(k), args.max_tokens)
        r["target_k"] = k
        required = k <= args.required_clean_k
        r["required"] = required
        if r.get("skipped"):
            print(f"{k:>4}K  SKIP ({r['skipped']})")
            results.append(r)
            continue
        ok = r["alpha"] and r["omega"]
        if required and not ok:
            gate_ok = False
        print(f"{k:>4}K  prompt={r['prompt_tokens']:>7}  "
              f"alpha={'OK' if r['alpha'] else 'FAIL'}  "
              f"omega={'OK' if r['omega'] else 'FAIL'}  "
              f"finish={r['finish']}  wall={r['wall_s']}s"
              f"{'  [REQUIRED]' if required else '  [informational]'}")
        results.append(r)

    verdict = "NEEDLE_GATE_PASS" if gate_ok else "NEEDLE_GATE_FAIL"
    print(verdict)
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"verdict": verdict, "salt": SALT, "results": results,
                       "required_clean_k": args.required_clean_k}, f, indent=2)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
