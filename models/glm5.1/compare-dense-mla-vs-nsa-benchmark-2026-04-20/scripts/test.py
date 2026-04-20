#!/usr/bin/env python3
"""
LLM Streaming CJK Watchdog.

Runs streaming chat completions against any OpenAI-compatible endpoint
(SGLang, vLLM, llama.cpp, OpenAI, OpenRouter, Together, ...) and watches
the output for Chinese / CJK Han ideographs. Designed to catch model
drift, KV-cache corruption, quantization damage, and other failure
modes where an English task unexpectedly starts emitting Chinese tokens.

Features:
  - Single-shot or --loop mode (runs until the first CJK character appears)
  - Live two-row status overlay pinned to the bottom of the terminal:
      row 1: current iteration — precise tok/s, tokens, elapsed, CJK counter
      row 2: last completed iteration + cumulative totals across the loop
  - PRECISE tok/s from the server (uses stream_options.continuous_usage_stats
    when supported) — never estimates from chunk counts
  - Optional padding context to simulate long-input workloads
  - Graceful Ctrl+C, exit code 2 when CJK characters are detected

Usage:
    python3 llm_cjk_watchdog.py
    python3 llm_cjk_watchdog.py --port 5000 --loop
    python3 llm_cjk_watchdog.py --host https://api.together.xyz --api-key sk-... --model meta-llama/llama-3-70b
    python3 llm_cjk_watchdog.py --context-tokens 40000 --max-tokens 2000
    python3 llm_cjk_watchdog.py --loop --prompt "Explain how TCP fast retransmit works."
"""

import argparse
import json
import shutil
import sys
import time

import requests


VERSION = "0.1.0"

# Approximate ratio used only to size the padding generator — actual token
# accounting always comes from the server's usage block.
CHARS_PER_TOKEN = 4

DEFAULT_PROMPT = "Write a Python script that implements the Sieve of Eratosthenes."

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


# --------------------------------------------------------------------------- #
# CJK detection
# --------------------------------------------------------------------------- #

def count_chinese_chars(text):
    """Count CJK Han ideographs in ``text``.

    Covers CJK Unified Ideographs and extensions A–E plus Compatibility
    Ideographs — i.e. characters shared by Chinese, Japanese kanji, and
    Korean hanja. Does not count kana, hangul, or CJK punctuation.
    """
    count = 0
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or      # CJK Unified Ideographs
            0x3400 <= cp <= 0x4DBF or      # Extension A
            0x20000 <= cp <= 0x2A6DF or    # Extension B
            0x2A700 <= cp <= 0x2B73F or    # Extension C
            0x2B740 <= cp <= 0x2B81F or    # Extension D
            0x2B820 <= cp <= 0x2CEAF or    # Extension E
            0xF900 <= cp <= 0xFAFF):       # Compatibility Ideographs
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Terminal overlay: reserves the bottom 2 rows of the TTY for a live status
# line + a persistent stats line using the DEC scroll-region (DECSTBM) escape.
# When stdout is not a TTY (piped/redirected) the overlay is disabled and the
# script falls back to plain linear output.
# --------------------------------------------------------------------------- #

_overlay_active = False


def _term_size():
    return shutil.get_terminal_size((80, 24))


def start_overlay():
    global _overlay_active
    if not sys.stdout.isatty():
        return
    rows = _term_size().lines
    # Content area: rows 1 .. rows-2. Row rows-1 = LIVE, row rows = STATS.
    sys.stdout.write(f"\033[1;{rows - 2}r")
    sys.stdout.write(f"\033[{rows - 2};1H")
    sys.stdout.flush()
    _overlay_active = True


def stop_overlay():
    global _overlay_active
    if not _overlay_active:
        return
    rows = _term_size().lines
    sys.stdout.write(f"\033[{rows - 1};1H\033[2K")
    sys.stdout.write(f"\033[{rows};1H\033[2K")
    sys.stdout.write("\033[r")
    sys.stdout.write(f"\033[{rows};1H")
    sys.stdout.flush()
    _overlay_active = False


def _strip_ansi(s):
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\033" and i + 1 < len(s) and s[i + 1] == "[":
            j = i + 2
            while j < len(s) and not ("@" <= s[j] <= "~"):
                j += 1
            i = j + 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _write_overlay_row(row, text):
    if not _overlay_active:
        return
    cols = _term_size().columns
    if len(_strip_ansi(text)) > cols:
        # Fallback truncation — drops colors but keeps content readable.
        text = _strip_ansi(text)[:cols]
    sys.stdout.write("\0337")                    # DECSC save cursor
    sys.stdout.write(f"\033[{row};1H")           # move to status row
    sys.stdout.write("\033[2K")                  # clear line
    sys.stdout.write(text)
    sys.stdout.write("\0338")                    # DECRC restore cursor
    sys.stdout.flush()


def update_overlay_live(text):
    _write_overlay_row(_term_size().lines - 1, text)


def update_overlay_stats(text):
    _write_overlay_row(_term_size().lines, text)


# --------------------------------------------------------------------------- #
# Padding context — builds a synthetic user turn of roughly ``target_tokens``
# tokens so the model has something substantial to attend over. Used to
# reproduce long-context failure modes.
# --------------------------------------------------------------------------- #

PADDING_SENTENCES = [
    "The history of European architecture spans thousands of years and encompasses a wide variety of styles and movements.",
    "From the ancient Greek temples to the Gothic cathedrals of the Middle Ages, each era has left its distinctive mark on the built environment.",
    "The Renaissance brought a renewed interest in classical forms, while the Baroque period introduced dramatic ornamentation and grandeur.",
    "In the modern era, architects have experimented with new materials such as steel, glass, and reinforced concrete.",
    "The development of skyscrapers in the late 19th century transformed urban landscapes around the world.",
    "Sustainable architecture has become increasingly important as societies grapple with climate change and resource depletion.",
    "The principles of good design include functionality, durability, and aesthetic appeal.",
    "Urban planning plays a crucial role in shaping how cities develop and how their inhabitants experience daily life.",
    "Public spaces such as parks, plazas, and waterfronts contribute significantly to the quality of urban living.",
    "The integration of technology into building design has opened up new possibilities for energy efficiency and comfort.",
    "Historical preservation efforts seek to maintain the cultural heritage embodied in older structures.",
    "The relationship between architecture and nature has been explored by many influential designers throughout history.",
    "Building codes and regulations ensure that structures meet minimum standards for safety and accessibility.",
    "The choice of materials in construction affects not only the appearance of a building but also its environmental impact.",
    "Innovative structural engineering techniques have made it possible to create buildings of unprecedented scale and complexity.",
    "The study of vernacular architecture reveals how different cultures have adapted their building practices to local conditions.",
    "Interior design complements architecture by addressing the arrangement and decoration of interior spaces.",
    "Landscape architecture deals with the design of outdoor areas, landmarks, and structures to achieve environmental or aesthetic outcomes.",
    "The concept of smart cities integrates information technology with urban infrastructure to improve efficiency and quality of life.",
    "Affordable housing remains one of the most pressing challenges facing urban planners and policymakers worldwide.",
]


def generate_padding_text(target_tokens):
    target_chars = target_tokens * CHARS_PER_TOKEN
    lines = []
    current_chars = 0
    idx = 0
    while current_chars < target_chars:
        sentence = PADDING_SENTENCES[idx % len(PADDING_SENTENCES)]
        lines.append(sentence)
        current_chars += len(sentence) + 1
        idx += 1
    return " ".join(lines)


def build_messages(user_prompt, context_tokens):
    messages = []
    if context_tokens > 0:
        padding = generate_padding_text(context_tokens)
        approx_tokens = len(padding) // CHARS_PER_TOKEN
        print(f"[INFO] Padding context: ~{approx_tokens:,} tokens ({len(padding):,} chars)")
        messages.append({
            "role": "user",
            "content": (
                "Below is a large reference document. Read it carefully, "
                "then answer the question that follows.\n\n"
                "--- BEGIN REFERENCE DOCUMENT ---\n"
                f"{padding}\n"
                "--- END REFERENCE DOCUMENT ---"
            ),
        })
        messages.append({
            "role": "assistant",
            "content": "I have read the entire reference document. Please ask your question.",
        })
    messages.append({"role": "user", "content": user_prompt})
    return messages


# --------------------------------------------------------------------------- #
# URL + request
# --------------------------------------------------------------------------- #

def build_url(host, port):
    """Turn ``host`` + ``port`` into a full /v1/chat/completions URL.

    ``host`` may be a bare hostname/IP (``localhost``, ``10.0.0.5``) or a
    full URL with scheme (``https://api.together.xyz``). For bare hosts the
    port is appended; for URL hosts the port argument is only appended if
    the URL has no explicit port.
    """
    if host.startswith(("http://", "https://")):
        base = host.rstrip("/")
        # Only append port if user URL has no explicit port.
        scheme_stripped = base.split("://", 1)[1]
        host_part = scheme_stripped.split("/", 1)[0]
        if ":" not in host_part and port:
            base = base.replace(host_part, f"{host_part}:{port}", 1)
        return f"{base}/v1/chat/completions"
    return f"http://{host}:{port}/v1/chat/completions"


def run_request(
    url,
    headers,
    payload,
    iteration,
    use_overlay,
    stop_on_chinese=False,
):
    """Run a single streaming request.

    Returns a stats dict on success or ``None`` on HTTP / network failure.
    When ``stop_on_chinese`` is ``True`` the stream is aborted as soon as
    the first CJK ideograph is seen.
    """
    quiet = payload.pop("__quiet", False)

    if not quiet:
        print(f"[INFO] === Iteration #{iteration} ===")
    t_start = time.time()
    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), stream=True, timeout=(10, None)
        )
    except requests.RequestException as e:
        if not quiet:
            print(f"{RED}[ERR] Request failed: {e}{RESET}")
        return None

    if not response.ok:
        if not quiet:
            print(f"{RED}[ERR] HTTP {response.status_code}: {response.text}{RESET}")
        response.close()
        return None

    t_first = None
    usage = None
    output_buffer = []
    reasoning_buffer = []
    content_buffer = []
    chunks = 0
    completion_tokens_live = 0   # precise, from server's per-chunk usage
    chinese_count = 0
    stopped_early = False
    t_last_overlay = 0.0
    finish_reason = None

    if not quiet:
        print("---- Response (streaming) ----")
    try:
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8", errors="replace")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Servers honoring stream_options.continuous_usage_stats publish
            # a live usage block in every chunk — use it for exact tok/s.
            if data.get("usage"):
                usage = data["usage"]
                if "completion_tokens" in usage:
                    completion_tokens_live = usage["completion_tokens"]

            if not data.get("choices"):
                continue

            choice = data["choices"][0]
            delta = choice.get("delta", {})
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
            reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
            content = delta.get("content") or ""
            chunk_text = reasoning + content
            if not chunk_text:
                continue

            if t_first is None:
                t_first = time.time()

            output_buffer.append(chunk_text)
            if reasoning:
                reasoning_buffer.append(reasoning)
            if content:
                content_buffer.append(content)
            chunks += 1
            if not quiet:
                print(chunk_text, end="", flush=True)

            chunk_chinese = count_chinese_chars(chunk_text)
            chinese_count += chunk_chinese

            if use_overlay:
                now = time.time()
                if now - t_last_overlay >= 0.1:
                    elapsed = now - t_first
                    if completion_tokens_live > 0:
                        rate = completion_tokens_live / elapsed if elapsed > 0 else 0.0
                        rate_part = f"{rate:6.1f} tok/s │ tok {completion_tokens_live:5d}"
                    else:
                        # Server did not honor continuous_usage_stats — fall
                        # back to chunk counts but label them clearly.
                        chunk_rate = chunks / elapsed if elapsed > 0 else 0.0
                        rate_part = f"{YELLOW}~{chunk_rate:5.1f} chunk/s │ chunks {chunks:5d}{RESET}{GREEN}"
                    status = (
                        f"{GREEN}{BOLD}▶ iter {iteration}{RESET}"
                        f"{GREEN} │ {rate_part}"
                        f" │ t {elapsed:5.1f}s{RESET}"
                    )
                    if chinese_count > 0:
                        status += f" {RED}{BOLD}│ CJK {chinese_count}{RESET}"
                    update_overlay_live(status)
                    t_last_overlay = now

            if stop_on_chinese and chunk_chinese > 0:
                stopped_early = True
                break
    finally:
        response.close()

    t_end = time.time()
    if not quiet:
        print()  # newline after streamed content

    if t_first:
        if not quiet:
            print(f"[INFO] TTFT: {t_first - t_start:.2f}s")
    else:
        if not quiet:
            print("[INFO] No tokens received.")
    if not quiet:
        print(f"[INFO] Iteration total time: {t_end - t_start:.2f}s")

    completion_tokens = usage.get("completion_tokens", 0) if usage else 0
    total_elapsed = t_end - t_start
    gen_elapsed = (t_end - t_first) if t_first else 0.0

    if completion_tokens > 0:
        total_tok_s = completion_tokens / total_elapsed if total_elapsed > 0 else 0.0
        if not quiet:
            print(f"{GREEN}{BOLD}[INFO] Completion tokens: {completion_tokens}{RESET}")
            print(f"{GREEN}[INFO] Throughput (incl. TTFT): {total_tok_s:.2f} tok/s{RESET}")
        if gen_elapsed > 0:
            if not quiet:
                print(f"{GREEN}[INFO] Generation-only:      {completion_tokens / gen_elapsed:.2f} tok/s{RESET}")
    else:
        if not quiet:
            print(f"{YELLOW}[INFO] Server did not return usage.completion_tokens — precise tok/s unavailable.{RESET}")

    if chinese_count > 0:
        if not quiet:
            print(f"{RED}{BOLD}[!!!] CJK CHARACTERS IN OUTPUT: {chinese_count}"
                  f"{' (stream aborted early)' if stopped_early else ''}{RESET}")
    else:
        if not quiet:
            print("[INFO] CJK characters in output: 0")

    return {
        "iteration": iteration,
        "completion_tokens": completion_tokens,
        "output_text": "".join(output_buffer),
        "reasoning_text": "".join(reasoning_buffer),
        "content_text": "".join(content_buffer),
        "chinese_count": chinese_count,
        "elapsed": total_elapsed,
        "gen_elapsed": gen_elapsed,
        "ttft": (t_first - t_start) if t_first else 0.0,
        "stopped_early": stopped_early,
        "finish_reason": finish_reason,
        "usage": usage or {},
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

EPILOG = """\
examples:
  # single shot against a local SGLang/vLLM on :5000
  python3 llm_cjk_watchdog.py

  # loop forever until the model leaks a Chinese character
  python3 llm_cjk_watchdog.py --loop

  # remote OpenAI-compatible endpoint with API key
  python3 llm_cjk_watchdog.py --host https://api.together.xyz \\
      --api-key $TOGETHER_API_KEY --model meta-llama/llama-3-70b

  # simulate a 40k-token input context
  python3 llm_cjk_watchdog.py --context-tokens 40000 --max-tokens 2000

exit codes:
  0  finished normally (no CJK characters found, or single-shot completed)
  2  CJK characters detected
  130  interrupted (Ctrl+C)
"""


def parse_args():
    parser = argparse.ArgumentParser(
        prog="llm_cjk_watchdog.py",
        description="Streaming chat watchdog that detects Chinese / CJK character leaks in LLM output.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Server hostname or full URL (default: localhost). Accepts a full URL with "
             "scheme for HTTPS endpoints, e.g. https://api.together.xyz.",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Server port (default: 5000). Ignored when --host already contains a port.",
    )
    parser.add_argument(
        "--api-key", default="",
        help="API key sent as 'Authorization: Bearer <key>'. Empty = no auth header.",
    )
    parser.add_argument(
        "--model", default="",
        help="Model name for the API request. Empty is fine for SGLang/vLLM servers that "
             "ignore the field; required by hosted APIs (OpenAI, OpenRouter, Together, ...).",
    )
    parser.add_argument(
        "-c", "--context-tokens", type=int, default=0, metavar="N",
        help="Size of synthetic padding context in tokens, e.g. 40000. Default: 0 (no padding).",
    )
    parser.add_argument(
        "-p", "--prompt", default=None,
        help=f"User prompt text. Default: {DEFAULT_PROMPT!r}",
    )
    parser.add_argument(
        "-f", "--prompt-file", default=None, metavar="PATH",
        help="Read user prompt from a file ('-' for stdin). Mutually exclusive with --prompt.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2000,
        help="Max output tokens per request (default: 2000).",
    )
    parser.add_argument(
        "-L", "--loop", action="store_true",
        help="Run in a loop, stopping only when a CJK character appears in the output.",
    )
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="Disable the bottom status overlay (useful when the terminal is misbehaving).",
    )
    parser.add_argument(
        "--version", action="version", version=f"llm_cjk_watchdog {VERSION}",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress streamed text and info lines; intended for scripted evaluation.",
    )
    parser.add_argument(
        "--json-summary", default=None, metavar="PATH",
        help="Write final result JSON to PATH or '-' for stdout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    url = build_url(args.host, args.port)

    if args.prompt is not None and args.prompt_file is not None:
        print(f"{RED}[ERR] --prompt and --prompt-file are mutually exclusive.{RESET}", file=sys.stderr)
        return 2
    if args.prompt_file is not None:
        try:
            if args.prompt_file == "-":
                user_prompt = sys.stdin.read()
            else:
                with open(args.prompt_file, "r", encoding="utf-8") as fh:
                    user_prompt = fh.read()
        except OSError as e:
            print(f"{RED}[ERR] Cannot read prompt file {args.prompt_file!r}: {e}{RESET}", file=sys.stderr)
            return 2
        user_prompt = user_prompt.rstrip("\n")
        if not user_prompt:
            print(f"{RED}[ERR] Prompt file {args.prompt_file!r} is empty.{RESET}", file=sys.stderr)
            return 2
        print(f"[INFO] Prompt loaded from {args.prompt_file!r} ({len(user_prompt):,} chars)")
    elif args.prompt is not None:
        user_prompt = args.prompt
    else:
        user_prompt = DEFAULT_PROMPT
    messages = build_messages(user_prompt, args.context_tokens)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    payload = {
        "model": args.model,
        "messages": messages,
        "stream": True,
        # continuous_usage_stats → SGLang / vLLM publish exact completion_tokens
        # in every chunk so the live tok/s readout is never estimated. Servers
        # that don't understand the field ignore it and we fall back gracefully.
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "max_tokens": args.max_tokens,
        "__quiet": args.quiet,
    }

    if not args.quiet:
        print(f"[INFO] llm_cjk_watchdog {VERSION} → {url}")
        print(f"[INFO] Model: {args.model!r} | max_tokens: {args.max_tokens} | messages: {len(messages)}")
    if args.loop:
        if not args.quiet:
            print(f"{BOLD}[INFO] LOOP mode — running until a CJK character appears (Ctrl+C to stop).{RESET}")

    use_overlay = sys.stdout.isatty() and not args.no_overlay

    iteration = 0
    total_tokens = 0
    total_elapsed = 0.0
    total_gen_elapsed = 0.0
    total_chinese = 0
    last_result = None
    interrupted = False

    def render_stats_row():
        if not use_overlay:
            return
        parts = []
        if last_result is not None:
            li = last_result
            gen = li["gen_elapsed"]
            iter_rate = li["completion_tokens"] / gen if gen > 0 else 0.0
            parts.append(
                f"{GREEN}{BOLD}● last #{li['iteration']}{RESET}"
                f"{GREEN}: {li['completion_tokens']} tok @ {iter_rate:.1f} tok/s"
                f" │ TTFT {li['ttft']:.2f}s{RESET}"
            )
        else:
            parts.append(f"{BOLD}● waiting for first iteration...{RESET}")
        if iteration > 0:
            avg = total_tokens / total_gen_elapsed if total_gen_elapsed > 0 else 0.0
            parts.append(
                f"{GREEN}Σ {iteration}× │ {total_tokens} tok │ {avg:.1f} tok/s avg{RESET}"
            )
        if total_chinese > 0:
            parts.append(f"{RED}{BOLD}CJK: {total_chinese}{RESET}")
        else:
            parts.append(f"{GREEN}CJK: 0{RESET}")
        update_overlay_stats("  ║  ".join(parts))

    try:
        if use_overlay:
            start_overlay()
            render_stats_row()

        if args.loop:
            try:
                while True:
                    iteration += 1
                    result = run_request(
                        url,
                        headers,
                        payload,
                        iteration,
                        use_overlay,
                        stop_on_chinese=True,
                    )
                    if result is None:
                        break
                    last_result = result
                    total_tokens += result["completion_tokens"]
                    total_elapsed += result["elapsed"]
                    total_gen_elapsed += result["gen_elapsed"]
                    total_chinese += result["chinese_count"]
                    render_stats_row()
                    if result["chinese_count"] > 0:
                        break
            except KeyboardInterrupt:
                interrupted = True
        else:
            iteration = 1
            last_result = run_request(
                url,
                headers,
                payload,
                1,
                use_overlay,
                stop_on_chinese=False,
            )
            if last_result:
                total_tokens = last_result["completion_tokens"]
                total_elapsed = last_result["elapsed"]
                total_gen_elapsed = last_result["gen_elapsed"]
                total_chinese = last_result["chinese_count"]
                render_stats_row()
    finally:
        if use_overlay:
            stop_overlay()

    if args.loop:
        if not args.quiet:
            print()
            print(f"{BOLD}=== LOOP SUMMARY ==={RESET}")
            print(f"Iterations completed: {iteration}")
            print(f"{GREEN}{BOLD}Total tokens: {total_tokens}{RESET}")
            if total_elapsed > 0:
                print(f"{GREEN}Average throughput (incl. TTFT): {total_tokens / total_elapsed:.2f} tok/s{RESET}")
            if total_gen_elapsed > 0:
                print(f"{GREEN}Average generation-only:         {total_tokens / total_gen_elapsed:.2f} tok/s{RESET}")
            if interrupted:
                print(f"{BOLD}Interrupted by user (Ctrl+C).{RESET}")
            elif last_result and last_result["chinese_count"] > 0:
                print(f"{RED}{BOLD}STOPPED: CJK characters found in iteration #{iteration} "
                      f"(count: {last_result['chinese_count']}).{RESET}")
            else:
                print(f"{BOLD}Run finished without finding any CJK characters.{RESET}")

    if args.json_summary is not None:
        summary = {
            "iteration_count": iteration,
            "interrupted": interrupted,
            "last_result": last_result,
            "total_tokens": total_tokens,
            "total_elapsed": total_elapsed,
            "total_gen_elapsed": total_gen_elapsed,
            "total_chinese": total_chinese,
        }
        if args.json_summary == "-":
            print(json.dumps(summary, ensure_ascii=False))
        else:
            with open(args.json_summary, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
                fh.write("\n")

    if interrupted:
        return 130
    if last_result and last_result["chinese_count"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        try:
            stop_overlay()
        except Exception:
            pass
        print("\nInterrupted.")
        sys.exit(130)
