"""Contract tests for needle_gate.py.

Covers the review findings on PR #54: exact ordered two-line validation,
context-limit-only HTTP 400 skips, failure artifacts on probe exceptions,
and the --required-clean-k probing requirement.
"""

import io
import json
import sys
import urllib.error
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest import mock

import pytest

SCRIPT = Path(__file__).resolve().parent / "needle_gate.py"
SPEC = spec_from_file_location("needle_gate", SCRIPT)
assert SPEC and SPEC.loader
NG = module_from_spec(SPEC)
SPEC.loader.exec_module(NG)


def _chat_response(content: str):
    payload = json.dumps({
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 123},
    }).encode()

    class FakeResponse:
        def __enter__(self):
            return io.BytesIO(payload)

        def __exit__(self, *exc):
            return False

    return FakeResponse()


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://base/v1/chat/completions", code, "Bad Request",
        None, io.BytesIO(body.encode()),
    )


def _probe_with_urlopen(**kwargs):
    return mock.patch.object(NG.urllib.request, "urlopen", **kwargs)


def _run_main(tmp_path, sizes, required_clean_k, probe_side_effect):
    output = tmp_path / "needle-gate.json"
    argv = [
        "needle_gate.py", "--model", "test-model",
        "--sizes", sizes, "--required-clean-k", str(required_clean_k),
        "--output", str(output),
    ]
    with mock.patch.object(NG, "probe", side_effect=probe_side_effect), \
            mock.patch.object(sys, "argv", argv):
        rc = NG.main()
    return rc, json.loads(output.read_text())


def _run_main_with_urlopen(tmp_path, sizes, required_clean_k, urlopen_side_effect):
    """Drives the real probe(); only the HTTP transport is faked."""
    output = tmp_path / "needle-gate.json"
    argv = [
        "needle_gate.py", "--model", "test-model",
        "--sizes", sizes, "--required-clean-k", str(required_clean_k),
        "--output", str(output),
    ]
    with _probe_with_urlopen(side_effect=urlopen_side_effect), \
            mock.patch.object(sys, "argv", argv):
        rc = NG.main()
    return rc, json.loads(output.read_text())


def _clean_probe(base, model, prompt, max_tokens):
    return {"alpha": True, "omega": True, "prompt_tokens": 100,
            "finish": "stop", "wall_s": 0.1}


def test_exact_ordered_two_line_response_passes():
    with _probe_with_urlopen(return_value=_chat_response("84291\nFERN-72")):
        result = NG.probe("http://base", "m", "prompt", 8)
    assert result["alpha"] is True
    assert result["omega"] is True


def test_reversed_lines_fail():
    with _probe_with_urlopen(return_value=_chat_response("FERN-72\n84291")):
        result = NG.probe("http://base", "m", "prompt", 8)
    assert result["alpha"] is False
    assert result["omega"] is False


def test_extra_lines_fail():
    for content in (
        "84291\nFERN-72\nanything else",
        "The answers are:\n84291\nFERN-72",
        "vault code 84291\nexit password FERN-72",
        "84291",
        "",
    ):
        with _probe_with_urlopen(return_value=_chat_response(content)):
            result = NG.probe("http://base", "m", "prompt", 8)
        assert result["alpha"] is False, content
        assert result["omega"] is False, content


def test_whitespace_padded_two_line_response_passes():
    with _probe_with_urlopen(return_value=_chat_response("  84291 \n FERN-72  \n")):
        result = NG.probe("http://base", "m", "prompt", 8)
    assert result["alpha"] is True
    assert result["omega"] is True


def test_context_limit_http_400_skips():
    bodies = (
        '{"error": {"message": "This model\'s maximum context length is '
        '65536 tokens. However, you requested 70000 tokens."}}',
        '{"object": "error", "message": "max_tokens plus prompt exceeds '
        'max_model_len 65536", "type": "BadRequestError"}',
        '{"detail": "prompt exceeds the model context window"}',
    )
    for body in bodies:
        with _probe_with_urlopen(side_effect=_http_error(400, body)):
            result = NG.probe("http://base", "m", "prompt", 8)
        assert result.get("skipped_context_limit") is True, body


def test_non_context_http_400_is_not_a_skip():
    assert not NG.is_context_limit_rejection(
        '{"error": {"message": "model `does-not-exist` was not found"}}')
    assert not NG.is_context_limit_rejection(
        '{"detail": "malformed payload: messages[0].content missing"}')
    with _probe_with_urlopen(side_effect=_http_error(
            400, '{"error": {"message": "model `nope` was not found"}}')):
        with pytest.raises(urllib.error.HTTPError):
            NG.probe("http://base", "m", "prompt", 8)


def test_non_context_http_400_fails_gate_and_writes_artifact(tmp_path):
    rc, data = _run_main_with_urlopen(
        tmp_path, "8,64", 8,
        _http_error(400, '{"error": {"message": "model `nope` was not found"}}'),
    )
    assert rc == 1
    assert data["verdict"] == "NEEDLE_GATE_FAIL"
    assert len(data["results"]) == 2
    for entry in data["results"]:
        assert entry["ok"] is False
        assert "HTTPError" in entry["error"]
        assert "model `nope` was not found" in entry["error"]


def test_network_error_fails_gate_and_writes_artifact(tmp_path):
    rc, data = _run_main_with_urlopen(
        tmp_path, "8,64", 8,
        urllib.error.URLError("connection refused"),
    )
    assert rc == 1
    assert data["verdict"] == "NEEDLE_GATE_FAIL"
    assert len(data["results"]) == 2
    for entry in data["results"]:
        assert entry["ok"] is False
        assert "connection refused" in entry["error"]


def test_required_clean_k_outside_sizes_fails_gate(tmp_path):
    rc, data = _run_main(tmp_path, "8,64", 96, _clean_probe)
    assert rc == 1
    assert data["verdict"] == "NEEDLE_GATE_FAIL"
    config_errors = [r for r in data["results"] if "required-clean-k" in (r.get("error") or "")]
    assert config_errors, data["results"]
    assert config_errors[0]["target_k"] == 96
    assert config_errors[0]["ok"] is False


def test_required_clean_k_probe_failure_fails_gate(tmp_path):
    def flaky(base, model, prompt, max_tokens):
        if prompt == NG.build_prompt(96):
            raise urllib.error.URLError("connection reset")
        return _clean_probe(base, model, prompt, max_tokens)

    rc, data = _run_main(tmp_path, "8,64,96", 96, flaky)
    assert rc == 1
    assert data["verdict"] == "NEEDLE_GATE_FAIL"
    by_size = {r["target_k"]: r for r in data["results"]}
    assert by_size[96]["ok"] is False


def test_clean_run_passes_and_writes_artifact(tmp_path):
    rc, data = _run_main(tmp_path, "8,64,96", 96, _clean_probe)
    assert rc == 0
    assert data["verdict"] == "NEEDLE_GATE_PASS"
    assert [r["target_k"] for r in data["results"]] == [8, 64, 96]
    assert all(r["ok"] is True for r in data["results"])


def test_required_size_skip_fails_gate(tmp_path):
    def skip_96(base, model, prompt, max_tokens):
        if prompt == NG.build_prompt(96):
            return {"skipped_context_limit": True,
                    "reason": "prompt exceeds the server context window",
                    "wall_s": 0.1}
        return _clean_probe(base, model, prompt, max_tokens)

    rc, data = _run_main(tmp_path, "8,64,96", 96, skip_96)
    assert rc == 1
    assert data["verdict"] == "NEEDLE_GATE_FAIL"


def test_informational_probe_error_does_not_fail_gate(tmp_path):
    def fail_128(base, model, prompt, max_tokens):
        if prompt == NG.build_prompt(128):
            raise urllib.error.URLError("read timed out")
        return _clean_probe(base, model, prompt, max_tokens)

    rc, data = _run_main(tmp_path, "8,96,128", 96, fail_128)
    assert rc == 0
    assert data["verdict"] == "NEEDLE_GATE_PASS"
    by_size = {r["target_k"]: r for r in data["results"]}
    assert by_size[128]["ok"] is False
    assert by_size[128]["required"] is False


def test_malformed_sizes_fails_gate_and_writes_artifact(tmp_path):
    rc, data = _run_main(tmp_path, "8,,96", 96, _clean_probe)
    assert rc == 1
    assert data["verdict"] == "NEEDLE_GATE_FAIL"
    assert "invalid --sizes" in data["results"][0]["error"]
