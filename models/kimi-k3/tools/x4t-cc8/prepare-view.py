"""Create serving metadata for an immutable Kimi-K3 X4T checkpoint."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
for source in (args.checkpoint / "metadata").iterdir():
    if source.name not in ("config.json", "model.safetensors.index.json"):
        (args.output / source.name).symlink_to(Path("/x4t/metadata") / source.name)
config = json.loads((args.checkpoint / "metadata/config.json").read_text())
quant = {"quant_method": "kimi_x4t", "format_version": 1, "checkpoint_root": "/x4t"}
config["text_config"]["quantization_config"] = quant
config["quantization_config"] = quant
(args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
