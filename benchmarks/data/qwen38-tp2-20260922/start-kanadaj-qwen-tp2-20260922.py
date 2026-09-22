"""Reproduce the published TP2/HiCache recipe on a fixed QAD snapshot."""

import json
import os
import subprocess
from pathlib import Path

source = Path("/tmp/kanadaj-sglang-feedback-20260921")
args = json.loads((source / "deploy/production/args.json").read_text())
environment = json.loads((source / "deploy/production/environment.json").read_text())


def set_arg(name, value):
    for index, argument in enumerate(args):
        if argument == name:
            args[index + 1] = str(value)
            return
        if argument.startswith(name + "="):
            args[index] = name + "=" + str(value)
            return
    args.extend([name, str(value)])


set_arg("--model-path", "/hf-qwen/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd")
set_arg("--served-model-name", "Qwen3.8-Flash-Next")
set_arg("--host", "0.0.0.0")
set_arg("--port", "5076")
set_arg("--mem-fraction-static", "0.92")
set_arg("--max-total-tokens", "4342208")
set_arg("--chunked-prefill-size", "6144")
set_arg("--prefill-batches-before-decode", "0.5")
set_arg("--max-running-requests", "64")
set_arg("--cuda-graph-max-bs-decode", "64")
args.extend([
    "--enable-hierarchical-cache", "--hicache-size=45",
    "--hicache-write-policy=write_back", "--hicache-io-backend=kernel",
    "--hicache-mem-layout=page_first", "--hicache-storage-backend=file",
    "--hicache-storage-prefetch-policy=timeout",
    '--hicache-storage-backend-extra-config={"max_size":500000000000,"enable_metadata_cache":true}',
])
environment.update({
    "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": "/cache/hicache-qad-7c4f1bc",
    "SGLANG_CACHE_DIR": "/cache/sglang",
    "XDG_CACHE_HOME": "/cache",
    "TRITON_CACHE_DIR": "/cache/triton",
    "CUDA_CACHE_PATH": "/cache/cuda",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "OMP_NUM_THREADS": "2",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
})
command = [
    "docker", "run", "-d", "--name", "feedback-sglang-qwen-tp2-20260922",
    "--gpus", '"device=5,6"', "--init", "--ipc=host", "--network=host",
    "--shm-size=32g", "--ulimit", "memlock=-1:-1",
    "--label", "lil.validation=qwen-tp2-feedback",
    "-v", "/root/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4:/hf-qwen:ro",
    "-v", "feedback-sglang-qwen-cache:/cache",
    "--entrypoint", "python3",
]
for key, value in sorted(environment.items()):
    command.extend(["-e", f"{key}={value}"])
command.extend(["local/kanadaj-qwen-hicache:524f248", "-m", "sglang.launch_server", *args])
print(json.dumps({"command": command}, indent=2), flush=True)
if os.getenv("PRINT_ONLY") != "1":
    subprocess.run(command, check=True)
