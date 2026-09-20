"""Check expanded wiki exports against the shared runtime without using GPUs."""

import copy
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "compose_reference", Path(__file__).with_name("generate-compose-reference.py"))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.fixture(scope="module")
def snapshot():
    return json.loads((reference.ROOT / reference.DESTINATION / "catalog.json").read_text())


@pytest.fixture(scope="module")
def runtime():
    source = os.environ.get("LIL_RUNTIME_SOURCE")
    if not source:
        pytest.skip("Set LIL_RUNTIME_SOURCE to validate native execution parity")
    sys.path.insert(0, source)
    from runtime.explicit import load_plan
    return load_plan


@pytest.mark.parametrize("name", [row[0] for row in reference.DEPLOYMENTS])
def test_complete_environment_and_native_arguments(snapshot, runtime, name):
    export = snapshot["exports"][name]
    before = copy.deepcopy(export)
    compose = reference.materialize(export, snapshot["release"], snapshot["image"])
    assert export == before
    service = compose["services"]["model"]
    data = yaml.safe_load(compose["configs"]["lil-launch"]["content"])
    plan = runtime(data, runtime_identity=snapshot["image"]["runtime_identity"],
                   incoming_environment=service["environment"])
    assert plan.argv == export["plan"]["argv"]
    assert plan.cache_service is None
    assert plan.values["cache-mode"] == "vram"
    expected = dict(export["environment"])
    expected["NVIDIA_VISIBLE_DEVICES"] = ",".join(export["setup"]["gpus"])
    assert plan.environment == expected
    assert service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == export["setup"]["gpus"]
    assert service["image"] == reference.REGISTRY + ":" + snapshot["release"]["tag"]
    assert service["restart"] == "unless-stopped"
    assert service["entrypoint"] == ["/opt/venv/bin/python", "-m", "runtime.explicit"]
    assert "x-lil-configurator" not in compose
    assert "lil-huggingface:/root/.cache/huggingface" in service["volumes"]


@pytest.mark.parametrize("name", [row[0] for row in reference.DEPLOYMENTS])
def test_compose_parser_and_shell_display(snapshot, name, tmp_path):
    export = snapshot["exports"][name]
    compose = reference.materialize(export, snapshot["release"], snapshot["image"])
    path = tmp_path / "compose.yaml"
    path.write_text(reference.dump(reference.dollars(compose)))
    # 'config' validates only; it never starts containers or accesses a GPU.
    result = subprocess.run(["docker", "compose", "-f", str(path), "config", "--format", "json"],
                            check=True, capture_output=True, text=True)
    actual = json.loads(result.stdout)
    assert yaml.safe_load(actual["configs"]["lil-launch"]["content"]) == yaml.safe_load(
        compose["configs"]["lil-launch"]["content"])
    assert actual["services"]["model"]["environment"] == compose["services"]["model"]["environment"]
    rendered = reference.command_text(export["plan"]["argv"])
    assert shlex.split(rendered.replace("\\\n", "")) == export["plan"]["argv"]


def test_dollar_interpolation_is_not_host_expansion(snapshot, tmp_path):
    export = copy.deepcopy(snapshot["exports"]["qwen38-tp1"])
    for environment in (export["environment"], export["explicit"]["environment"]):
        environment["LIL_LITERAL_TEST"] = "$HOME ${SHOULD_NOT_EXPAND}"
    compose = reference.materialize(export, snapshot["release"], snapshot["image"])
    path = tmp_path / "compose.yaml"
    path.write_text(reference.dump(reference.dollars(compose)))
    actual = json.loads(subprocess.check_output(
        ["docker", "compose", "-f", str(path), "config", "--format", "json"], text=True))
    # 'config' produces a reusable, escaped Compose serialization rather than
    # container getenv. CPU-only Compose execution independently checks that
    # the resulting container receives one literal dollar at each position.
    assert actual["services"]["model"]["environment"]["LIL_LITERAL_TEST"] == "$$HOME $${SHOULD_NOT_EXPAND}"


@pytest.mark.parametrize("field", ["image", "recipe"])
def test_mixed_image_exports_fail(snapshot, field):
    export = copy.deepcopy(snapshot["exports"]["qwen38-tp1"])
    export[field] = "another-identity"
    with pytest.raises(ValueError, match="selected image"):
        reference.materialize(export, snapshot["release"], snapshot["image"])


def test_generator_is_idempotent_and_covers_all_pages(snapshot):
    first = reference.render(reference.ROOT, snapshot)
    for path, text in first.items():
        assert text == (reference.ROOT / path).read_text()
        if path.suffix == ".md":
            assert text.count("<!-- BEGIN " + reference.MARKER) == 1
            assert text.count("<details>") == text.count("</details>")
            assert "<details open" not in text
    shared = first[Path("docs/unified-vllm-docker.md")]
    assert shared.index("## Start a server") < shared.index("## Expand the complete")
    qwen = first[Path("models/qwen38-flash-next.md")]
    assert qwen.index("## Start on two GPUs") < qwen.index("## Expand the complete")
