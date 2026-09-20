#!/usr/bin/env python3
"""Render expandable Compose references from the image-aware Docker catalog.

The catalog verifies the release receipt, registry configuration and runtime
source identity. This renderer selects documented deployments; it does not
implement model, speculation or cache defaults. A saved catalog response allows
offline regeneration and checks without Docker, GPUs or downloaded checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = Path("docs/compose/karmic-kraken-beta")
MARKER = "LIL-COMPOSE-REFERENCE"
REGISTRY = "ghcr.io/local-inference-lab/vllm"
RAW = "https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/"

# These are user-facing selections, not a second copy of runtime defaults.
DEPLOYMENTS = (
    ("glm53-tp4-off", "GLM-5.3 Flash: TP4, no speculation", "glm-5.3-flash.md",
     "glm53-flash", None, {"mode": "off"}),
    ("glm53-tp4-mtp3", "GLM-5.3 Flash: TP4, MTP3", "glm-5.3-flash.md",
     "glm53-flash", None, {"mode": "mtp", "draft-tokens": 3}),
    ("glm53-tp4-dflash2", "GLM-5.3 Flash: TP4, DFlash2 K7", "glm-5.3-flash.md",
     "glm53-flash", None, {"mode": "dflash2", "draft-tokens": 7}),
    ("glm53-spark-tp2", "GLM Spark: TP2/DCP2, MTP3", "glm-5.3-flash-spark-tp2.md",
     "glm53-flash", "glm53-spark-tp2", {}),
    ("qwen38-tp1", "Qwen3.8 Flash Next: TP1, MTP3", "qwen38-flash-next.md",
     "qwen38-flash-next", None, {}),
    ("qwen38-tp2", "Qwen3.8 Flash Next: TP2, MTP3", "qwen38-flash-next.md",
     "qwen38-flash-next", "qwen38-tp2", {}),
    ("ds4-flash-tp2", "DeepSeek V4 text: TP2, DSpark K5", "deepseek-v4-flash.md",
     "ds4-flash", None, {}),
    ("ds4-vision-tp2", "DeepSeek V4 Vision: TP2, DSpark K3", "deepseek-v4-flash-vision.md",
     "ds4-vision", None, {}),
    ("ds41-flash-tp4", "DeepSeek V4.1: TP4, adaptive DSpark K7, disk Engram",
     "deepseek-v4.1-flash.md", "ds41-flash", None, {}),
)


class LiteralDumper(yaml.SafeDumper):
    pass


def represent_string(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:str", value,
                                   style="|" if "\n" in value else None)


LiteralDumper.add_representer(str, represent_string)


def dump(value):
    return yaml.dump(value, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True,
                     width=110)


def dollars(value):
    """Protect literal runtime values from host-side Compose interpolation."""
    if isinstance(value, str):
        return value.replace("$", "$$")
    if isinstance(value, list):
        return [dollars(item) for item in value]
    if isinstance(value, dict):
        return {key: dollars(item) for key, item in value.items()}
    return value


def api(base, path, data=None):
    payload = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(base.rstrip("/") + path, data=payload,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def capture(base, release_id):
    listing = api(base, "/api/docker/catalog")
    release = next(row for row in listing["images"] if row["id"] == release_id)
    image = api(base, "/api/docker/image?id=" + release_id)
    exports = {}
    for name, _title, _page, profile, preset, options in DEPLOYMENTS:
        request = {"image": release_id, "profile": profile, "options": options,
                   "name": name, "restart": "unless-stopped"}
        if preset:
            request["preset"] = preset
        exports[name] = api(base, "/api/docker/resolve", request)
    return {"schema": "lil-wiki-compose-reference/v1", "release": release,
            "image": {key: image[key] for key in
                      ("reference", "recipe", "runtime_identity", "cuda", "driver_requirement")},
            "exports": exports}


def materialize(export, release, identity):
    """Expose actual execution inputs without loading profile defaults twice."""
    if export["image"] != identity["reference"] or export["recipe"] != identity["recipe"]:
        raise ValueError("All exports must describe the selected image and recipe")
    base = yaml.safe_load(export["compose"])
    service = base["services"]["model"]
    data = copy.deepcopy(export["explicit"])
    if data["schema"] != "lil-explicit-launch/v1":
        raise ValueError("The image must support the explicit runtime document")
    environment = data.pop("environment")
    if environment != export["environment"]:
        raise ValueError("Explicit and preview environments differ")
    devices = service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"]
    if len(devices) != data["options"]["tensor-parallel-size"]:
        raise ValueError("GPU selection and tensor parallelism disagree")
    # Docker's GPU selection is a host-container boundary, not a model setting.
    # Never copy the image's broad NVIDIA_VISIBLE_DEVICES=all into this export.
    environment["NVIDIA_VISIBLE_DEVICES"] = ",".join(devices)
    # The explicit runner binds this supported placeholder to its own lock.
    for key, value in environment.items():
        if "/cache/jit/" + identity["runtime_identity"] in value:
            environment[key] = value.replace(identity["runtime_identity"], "UNBOUND-RUNTIME")
    data["environment_keys"] = sorted(environment)
    service["environment"] = dict(sorted(environment.items()))
    service["image"] = REGISTRY + ":" + release["tag"]
    service["entrypoint"] = ["/opt/venv/bin/python", "-m", "runtime.explicit"]
    service["command"] = ["--config", "/etc/lil-launch.yaml"]
    service["configs"] = [{"source": "lil-launch", "target": "/etc/lil-launch.yaml"}]
    service["volumes"] = [volume.replace("lil-hf:", "lil-huggingface:")
                          for volume in service["volumes"]]
    base["volumes"]["lil-huggingface"] = {"name": "lil-huggingface"}
    base["volumes"].pop("lil-hf")
    # The web extension fingerprints a different, profile-based service. Do not
    # retain it on a materialized service or pretend it can be imported as-is.
    base.pop("x-lil-configurator", None)
    base["configs"] = {"lil-launch": {"content": dump(data)}}
    return base


def command_text(argv):
    groups = [argv[:5]]
    for argument in argv[5:]:
        if argument.startswith("--"):
            groups.append([argument])
        else:
            groups[-1].append(argument)
    return (" " + chr(92) + "\n  ").join(shlex.join(group) for group in groups)


def block(name, title, compose_text, export):
    url = RAW + str(DESTINATION / (name + ".compose.yaml"))
    return f"""<details>
<summary>{title}: full Compose, ENV and vLLM command</summary>

[Download the complete Compose file]({url}). Save it as `{name}.compose.yaml`,
choose GPU IDs, then run:

```bash
docker compose -f {name}.compose.yaml up -d
```

Logs: `docker compose -f {name}.compose.yaml logs -f model`.
Stop: `docker compose -f {name}.compose.yaml down` (keeps model/cache volumes).

```yaml
{compose_text.rstrip()}
```

The explicit runner passes these native arguments to vLLM through the image's
CUDA/NCCL bootstrap. This command is shown for inspection; the Compose file
above also supplies its environment and persistent volumes.

```bash
{command_text(export['plan']['argv'])}
```

</details>
"""


def replace_section(text, rendered):
    start, end = f"<!-- BEGIN {MARKER} -->", f"<!-- END {MARKER} -->"
    region = start + "\n" + rendered.rstrip() + "\n" + end
    if start in text:
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError("Expected exactly one generated reference region")
        text = re.sub(re.escape(start) + r".*?" + re.escape(end) + r"\n*",
                      "", text, count=1, flags=re.S)
    # Keep the shared model selector and both Qwen quick starts ahead of long
    # reference material. Each disclosure remains closed on initial rendering.
    if "## Inspect or override the launch" in text:
        index = text.index("## Inspect or override the launch")
    elif "## Precision, model tables and cache" in text:
        index = text.index("## Precision, model tables and cache")
    else:
        headings = list(re.finditer(r"^## ", text, re.M))
        index = headings[1].start() if len(headings) > 1 else len(text)
    return text[:index].rstrip() + "\n\n" + region + "\n\n" + text[index:]


def render(root, snapshot):
    if snapshot.get("schema") != "lil-wiki-compose-reference/v1":
        raise ValueError("Unrecognized reference snapshot")
    if set(snapshot["exports"]) != {row[0] for row in DEPLOYMENTS}:
        raise ValueError("Every documented deployment must have one export")
    files = {}
    grouped = {}
    for name, title, page, _profile, _preset, _options in DEPLOYMENTS:
        export = snapshot["exports"][name]
        compose = materialize(export, snapshot["release"], snapshot["image"])
        text = "# Generated from the selected image's shared runtime profiles.\n"
        text += "# Requires Docker Compose 2.23.1 or newer. No HF credentials are embedded.\n"
        text += dump(dollars(compose))
        files[DESTINATION / (name + ".compose.yaml")] = text
        grouped.setdefault(page, []).append(block(name, title, text, export))
    introduction = """## Expand the complete default configurations

These are runnable, release-tagged snapshots of the shared profiles, including
all non-secret image/profile ENV values and resolved serving options. They use
GPU-only prefix caching. Inactive `cache-*` options do not start LMCache.
They are deployment defaults, not copies of benchmark-only overrides.

Requires [Docker Compose 2.23.1 or newer](https://docs.docker.com/reference/compose-file/configs/).
`services.model.environment` is applied to vLLM; `configs.lil-launch.content`
contains the resolved options. The image's explicit runner retains bootstrap,
validation and cache supervision without re-reading model defaults.

Change GPU IDs in both `device_ids` and `NVIDIA_VISIBLE_DEVICES`. Only start one
example at a time unless GPU IDs, API ports and container names do not overlap.
Named volumes keep checkpoints and compiled kernels across container restarts.

For ordinary changes to TP, speculation or LMCache, prefer the short launch
commands on this page: they recompute dependent settings. The expanded files
freeze those settings; do not change only a derived field or image tag. The
`UNBOUND-RUNTIME` cache-path marker is filled by the image at startup.
`vllm_defaults` lists options left to native vLLM, not hidden profile overrides.
Host-injected credentials, container IDs and driver-provided variables are not
predicted by this static snapshot.
The image's `*_VERSION` ENV entries are build metadata; changing those strings
does not install another CUDA, NCCL or Python package.

"""
    for page, blocks in grouped.items():
        path = Path("models") / page
        model_intro = """## Expand the complete default configurations

Each block contains a runnable release-tagged Compose file, all image/profile
ENV settings and the resolved vLLM command. Requires Docker Compose 2.23.1+.
These snapshots use GPU-only prefix caching; inactive cache options do not
start LMCache. They are deployment defaults, not benchmark-only overrides.

For ordinary TP, speculation or cache changes, use the short commands above
so dependent settings are recalculated. See the
[expanded-file editing guide](../docs/unified-vllm-docker.md#expand-the-complete-default-configurations)
before modifying a frozen configuration. No credentials are included.

"""
        files[path] = replace_section((root / path).read_text(),
                                      model_intro + "\n".join(blocks))
    shared = Path("docs/unified-vllm-docker.md")
    shared_intro = introduction + """The quick-start commands follow the moving beta channel. These expanded
files keep the release tag whose settings they display. Every model page has
the corresponding expandable block as well.

"""
    files[shared] = replace_section((root / shared).read_text(),
                                   shared_intro + "\n".join(b for rows in grouped.values() for b in rows))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki", type=Path, default=ROOT)
    parser.add_argument("--snapshot", type=Path, default=DESTINATION / "catalog.json")
    parser.add_argument("--catalog-url")
    parser.add_argument("--release-id")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = args.wiki / args.snapshot
    if bool(args.catalog_url) != bool(args.release_id):
        parser.error("--catalog-url and --release-id must be provided together")
    snapshot = (capture(args.catalog_url, args.release_id) if args.catalog_url
                else json.loads(path.read_text()))
    files = render(args.wiki, snapshot)
    if args.catalog_url:
        files[args.snapshot] = json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
    differences = []
    for relative, content in files.items():
        destination = args.wiki / relative
        if not destination.exists() or destination.read_text() != content:
            differences.append(str(relative))
            if args.write:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content)
    print(json.dumps({"mode": "write" if args.write else "check", "differences": differences}, indent=2))
    if differences and not args.write:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
