#!/usr/bin/env python3
"""Compare profile and expanded configuration inside an already pulled image.

Every container uses runc, no GPU device request, no network, no model/cache
mount, and --print-config. Nothing serves requests, downloads checkpoints or
executes CUDA. The selected image is never pulled implicitly.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ssh-target")
    parser.add_argument("--ssh-host-key-alias")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads((args.directory / "catalog.json").read_text())
    image = snapshot["image"]["reference"]
    if args.output.exists():
        raise FileExistsError(args.output)

    def run(command):
        if args.ssh_target:
            prefix = ["ssh"]
            if args.ssh_host_key_alias:
                prefix += ["-o", "HostKeyAlias=" + args.ssh_host_key_alias]
            command = [*prefix, args.ssh_target, shlex.join(command)]
        return subprocess.check_output(command, text=True, timeout=60)

    local_image = run(["docker", "image", "inspect", image, "--format", "{{.Id}}"]).strip()
    report = {"status": "qualified CPU configuration parity; not GPU serving validation",
              "image": image, "local_image_id": local_image, "runtime": "runc",
              "gpu_device_requests": [], "network": "none", "cases": {}}
    for name, export in snapshot["exports"].items():
        path = args.directory / (name + ".compose.yaml")
        # Compose config is a parser-only call on the documentation host.
        normalized = json.loads(subprocess.check_output(
            ["docker", "compose", "-f", str(path), "config", "--format", "json"], text=True))
        service = normalized["services"]["model"]
        data = yaml.safe_load(normalized["configs"]["lil-launch"]["content"])
        command = ["docker", "run", "--rm", "--pull", "never", "--runtime", "runc",
                   "--network", "none", "--memory", "1g", "--cpus", "1",
                   "--pids-limit", "128", "--label", "lil.docs-check=compose-defaults",
                   "--entrypoint", "/opt/venv/bin/python"]
        ordinary = yaml.safe_load(export["compose"])["services"]["model"]["command"]
        baseline = json.loads(run([*command, image, "-m", "runtime.launcher",
                                   "--print-config", *ordinary]))
        for key, value in service["environment"].items():
            # 'docker compose config' emits interpolation-safe serialization.
            command += ["--env", key + "=" + value.replace("$$", "$")]
        explicit = json.loads(run([*command, image, "-m", "runtime.explicit",
                                   "--config-json", json.dumps(data), "--print-config"]))
        assert explicit["argv"] == baseline["argv"] == export["plan"]["argv"], name
        assert explicit["cache_service"] == baseline["cache_service"] is None, name
        expected = dict(export["environment"])
        expected["NVIDIA_VISIBLE_DEVICES"] = ",".join(export["setup"]["gpus"])
        observed = {key: item["value"] for key, item in explicit["environment"].items()}
        assert observed == expected, name
        assert all(observed[key] == item["value"]
                   for key, item in baseline["environment"].items()), name
        report["cases"][name] = {"status": "passed", "environment_entries": len(observed),
                                  "argv_sha256": hashlib.sha256(
                                      json.dumps(explicit["argv"]).encode()).hexdigest(),
                                  "compose_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(name + ": matching native arguments, ENV and cache mode", flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
