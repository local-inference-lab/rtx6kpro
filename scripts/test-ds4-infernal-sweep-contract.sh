#!/usr/bin/env bash
set -euo pipefail

# Verify that the Infernal Invocation sweep assigns independent resource
# contracts to concurrent decode and long-context prefill measurements.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "${tmp}"' EXIT

cat > "${tmp}/generic-sweep" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s|%s|%s|%s|%s|%s|%s|%s\n' \
  "${QUALIFICATION_ROLE}" "${RUN_DECODE}" "${RUN_PREFILL}" \
  "${MAX_NUM_SEQS}" "${MAX_MODEL_LEN}" "${GPU_MEM}" \
  "${PREFILL_CONTEXTS}" "${OUT}" >> "${CALL_LOG}"
SH

cat > "${tmp}/renderer" <<'PY'
#!/usr/bin/env python3
import os
import pathlib
import sys

pathlib.Path(os.environ["RENDER_LOG"]).write_text(sys.argv[1] + "\n")
PY

PYTHONDONTWRITEBYTECODE=1 python3 - "${SCRIPT_DIR}" <<'PY'
import runpy
import sys

module = runpy.run_path(
    sys.argv[1] + "/validate-ds4-agent-concurrency.py",
    run_name="ds4_agent_concurrency_contract",
)
spec = module["RequestSpec"]("AGENT_ALPHA", 1000, "ALPHA-731", "BETA-731")
messages = module["build_messages"](spec, "AGENT_ALPHA record")
serialized_messages = repr(messages)
assert "ALPHA-731" in serialized_messages
assert "BETA-731" not in serialized_messages

empty = {
    "characters": 10,
    "replacement_characters": 0,
    "non_printable_characters": 0,
    "non_ascii_fraction": 0.0,
    "cjk_characters": 0,
    "cjk_fraction": 0.0,
    "max_cjk_run": 0,
    "forbidden_marker_count": 0,
    "raw_token_pattern_counts": {"token_id": 0},
}
valid = {
    "content_prefix": "ALPHA-731 REPORT",
    "tool_call_delta_count": 0,
    "content_indicators": empty,
    "reasoning_indicators": empty,
}
assert module["integrity_violations"](valid) == []

corrupt = {
    **valid,
    "content_indicators": {
        **empty,
        "forbidden_marker_count": 1,
        "raw_token_pattern_counts": {"token_id": 2},
    },
}
assert module["integrity_violations"](corrupt) == [
    "content.forbidden_marker_count=1",
    "content.raw_token[token_id]=2",
]

dsml = module["count_text_indicators"](
    "</｜DSML｜tool_calls><｜DSML｜tool_calls>", "BETA-731"
)
dsml_counts = dsml["raw_token_pattern_counts"]
assert dsml_counts[r"</?｜[^>\n]{1,120}>"] == 2
PY

chmod +x "${tmp}/generic-sweep" "${tmp}/renderer"

out="${tmp}/results"
CALL_LOG="${tmp}/calls" \
RENDER_LOG="${tmp}/render" \
GENERIC_SWEEP_SCRIPT="${tmp}/generic-sweep" \
RESULT_RENDERER="${tmp}/renderer" \
OUT="${out}" \
"${SCRIPT_DIR}/run-ds4-infernal-sweep.sh"

expected=$(cat <<EOF
decode|1|0|1|10240|0.975|8k,64k,131008|${out}
prefill|0|1|16|131072|0.98|8k,64k,131008|${out}
EOF
)

[[ "$(<"${tmp}/calls")" == "${expected}" ]]
[[ "$(<"${tmp}/render")" == "${out}" ]]

grep -Fq 'GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}' \
  "${SCRIPT_DIR}/run-ds4-infernal-server.sh"
grep -Fq 'NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}' \
  "${SCRIPT_DIR}/run-ds4-infernal-server.sh"
grep -Fq -- '-e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"' \
  "${SCRIPT_DIR}/run-ds4-infernal-server.sh"
grep -Fq -- '-e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"' \
  "${SCRIPT_DIR}/run-ds4-infernal-server.sh"
grep -Fq -- '--entrypoint /usr/local/bin/lmcache-mp-wrapper.sh' \
  "${SCRIPT_DIR}/run-ds4-infernal-server.sh"
grep -Fq 'KV_OFFLOADING_SIZE NATIVE_L2_GB NATIVE_L2_PATH' \
  "${SCRIPT_DIR}/run-ds4-infernal-server.sh"
grep -Fq 'LMCACHE_MODE LMCACHE_L1_GB LMCACHE_L1_INIT_GB LMCACHE_L2_GB' \
  "${SCRIPT_DIR}/run-ds4-infernal-server.sh"
grep -Fq 'VLLM_SERVER_DEV_MODE EXTRA_VLLM_ARGS DRY_RUN' \
  "${SCRIPT_DIR}/run-ds4-infernal-server.sh"
grep -Fq 'RUNTIME_WARMUP_SPARSE_CONTEXT:-4096' \
  "${SCRIPT_DIR}/run-ds4-infernal-sweep.sh"
grep -Fq 'DECODE_CONCURRENCY:-1}' \
  "${SCRIPT_DIR}/run-ds4-infernal-sweep.sh"
grep -Fq -- '--output "$case_dir/warmup-sparse-decode.json"' \
  "${SCRIPT_DIR}/run-ds4-v9-sweep.sh"
grep -Fq -- '--concurrency "$DECODE_CONCURRENCY"' \
  "${SCRIPT_DIR}/run-ds4-v9-sweep.sh"
grep -Fq -- '--max-total-tokens "$DECODE_TOKEN_BUDGET"' \
  "${SCRIPT_DIR}/run-ds4-v9-sweep.sh"
grep -Fq -- '--sparse-warmup-context "$RUNTIME_WARMUP_SPARSE_CONTEXT"' \
  "${SCRIPT_DIR}/run-ds4-v9-sweep.sh"
grep -Fq 'measurement-start-utc.txt' \
  "${SCRIPT_DIR}/run-ds4-v9-sweep.sh"
grep -Fq 'validate-ds4-runtime-log.py' \
  "${SCRIPT_DIR}/run-ds4-v9-sweep.sh"
grep -Fq -- '--shm-size "${SHM_SIZE}"' \
  "${SCRIPT_DIR}/run-ds4-infernal-server.sh"
grep -Fq '/usr/local/bin/serve-ds4-flash.sh' \
  "${SCRIPT_DIR}/run-ds4-infernal-server.sh"
grep -Fq 'infernal-invocation-vllm7ed814e-b12x5d648d9-fi1ac6942-cu133-torch213-20260813-r7' \
  "${SCRIPT_DIR}/run-ds4-infernal-server.sh"
grep -Fq 'ds4-infernal-invocation-r7-' \
  "${SCRIPT_DIR}/run-ds4-infernal-sweep.sh"

PYTHONDONTWRITEBYTECODE=1 python3 - "${SCRIPT_DIR}" <<'PY'
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) / "render-ds4-infernal-results.py"
spec = importlib.util.spec_from_file_location("ds4_results", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

actual, row = module.select_prefill_row(
    {"131008": {"tok_per_sec": 1.0}}, 131072, "test"
)
assert actual == 131008
assert row["tok_per_sec"] == 1.0
try:
    module.select_prefill_row({"130000": {}}, 131072, "test")
except SystemExit:
    pass
else:
    raise AssertionError("out-of-contract prefill target was accepted")
PY

PYTHONDONTWRITEBYTECODE=1 python3 - "${SCRIPT_DIR}" <<'PY'
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) / "validate-ds4-sweep-case.py"
spec = importlib.util.spec_from_file_location("ds4_validator", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

valid = {
    "prefill": {
        "131008": {
            "prompt_tokens": 127696,
            "tok_per_sec": 12000.0,
            "ttft_seconds": 10.0,
            "samples": 1,
        }
    }
}
module.validate_prefill(valid, [131008])

invalid = {
    "prefill": {
        "131008": {
            "prompt_tokens": 120000,
            "tok_per_sec": 12000.0,
            "ttft_seconds": 10.0,
            "samples": 1,
        }
    }
}
try:
    module.validate_prefill(invalid, [131008])
except module.ValidationError:
    pass
else:
    raise AssertionError("miscalibrated prefill prompt was accepted")
PY
