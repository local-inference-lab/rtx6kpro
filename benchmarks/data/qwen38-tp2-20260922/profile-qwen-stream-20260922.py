"""Capture warmed decode CPU/GPU activity while preserving an inspectable stream."""

import json
import os
import threading
import time
from pathlib import Path

import requests

base = os.environ["PROFILE_BASE"]
engine = os.environ["PROFILE_ENGINE"]
client = requests.Session()
client.trust_env = False
output = Path(os.environ["PROFILE_OUTPUT"])
concurrency = int(os.getenv("PROFILE_CONCURRENCY", "1"))
payload = {
    "model": "Qwen3.8-Flash-Next",
    "messages": [{"role": "user", "content": "Write a detailed tutorial explaining how to build and test a database engine in Python. Include code, transaction handling, persistence, indexing, and concurrency. Explain every design choice carefully."}],
    "temperature": 1, "top_p": 0.95, "top_k": -1,
    "reasoning_effort": "medium", "max_tokens": 8192,
    "stream": True, "ignore_eos": True,
    "stream_options": {"include_usage": True},
}
first_tokens = [threading.Event() for _ in range(concurrency)]
stop_client = threading.Event()


def stream(index):
    request_client = requests.Session()
    request_client.trust_env = False
    path = output if concurrency == 1 else output.with_name(f"{output.stem}-{index}{output.suffix}")
    with request_client.post(base + "/v1/chat/completions", json=payload, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        with path.open("w") as file:
            for line in response.iter_lines():
                if line.startswith(b"data: {"):
                    event = json.loads(line[6:])
                    file.write(json.dumps(event) + "\n")
                    if any(choice.get("delta", {}).get("content") or choice.get("delta", {}).get("reasoning_content") or choice.get("delta", {}).get("reasoning") for choice in event.get("choices", [])):
                        first_tokens[index].set()
                if stop_client.is_set():
                    break


workers = [threading.Thread(target=stream, args=(index,)) for index in range(concurrency)]
for worker in workers:
    worker.start()
for ready in first_tokens:
    if not ready.wait(90):
        raise RuntimeError("No generated tokens before profiler trigger")
time.sleep(3)
body = {} if engine == "vllm" else {
    "output_dir": "/cache/profiles/qwen-sglang-tp2",
    "num_steps": 8, "activities": ["CPU", "GPU"],
    "with_stack": True, "record_shapes": False,
    "profile_id": f"tp2-c{concurrency}-decode",
}
response = client.post(base + "/start_profile", json=body, timeout=120)
print(json.dumps({"profile_status": response.status_code, "body": response.text}), flush=True)
response.raise_for_status()
time.sleep(10)
stop_client.set()
for worker in workers:
    worker.join(60)
    if worker.is_alive():
        raise RuntimeError("Streaming client did not finish")
print(json.dumps({"stream": str(output), "health": client.get(base + "/health", timeout=20).status_code}), flush=True)
