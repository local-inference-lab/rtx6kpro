# syntax=docker/dockerfile:1.7
#
# Overlay the shared-PLE-table patch onto the qualified Jovian Judgement R35
# community image.  The base is a source-overlay image: the venv imports vLLM
# from the git checkout at /opt/glm53-flash/vllm (VLLM_SOURCE_OVERLAY_ACTIVE=1),
# so replacing Python files in that tree is the sanctioned way to compose it.
#
# Build context: `git archive` of the fork branch build/qwen38-shared-ple-r35 at
# VLLM_OVERLAY_COMMIT restricted to the overlaid files, plus this Dockerfile.
# See README.md in this directory for the exact commands.

ARG BASE=docker.io/localinferencelab/vllm@sha256:7a425c6864b951bbc368111490a4b0ac69d8cd0dd4987075b2c7d40b753b1bf5

FROM ${BASE}

ARG VLLM_OVERLAY_REPOSITORY=https://github.com/renehonig/vllm.git
ARG VLLM_OVERLAY_BRANCH=build/qwen38-shared-ple-r35
ARG VLLM_OVERLAY_COMMIT
ARG VLLM_BASE_COMMIT=de982a50c6a3e4718e5cf9f00423a92192718da1
ARG ENVS_SHA256
ARG MODEL_SHA256
ARG PLE_LAYER_SHA256
ARG PLE_SHARED_TABLE_SHA256
ARG TEST_SHA256

COPY --chown=root:root vllm/envs.py /opt/glm53-flash/vllm/vllm/envs.py
COPY --chown=root:root \
  vllm/models/qwen3_8_flash_next/model.py \
  vllm/models/qwen3_8_flash_next/ple_layer.py \
  vllm/models/qwen3_8_flash_next/ple_shared_table.py \
  /opt/glm53-flash/vllm/vllm/models/qwen3_8_flash_next/
COPY --chown=root:root \
  tests/models/test_qwen3_8_flash_next_ple_shared_table.py \
  /opt/glm53-flash/vllm/tests/models/test_qwen3_8_flash_next_ple_shared_table.py

# The base tree must be the commit the overlay was written against, the
# overlaid files must be exactly the committed ones, and the working tree must
# differ from the base commit in those files only.
RUN set -eu \
 && cd /opt/glm53-flash/vllm \
 && test "$(git rev-parse HEAD)" = "${VLLM_BASE_COMMIT}" \
 && test "$(sha256sum vllm/envs.py | cut -d' ' -f1)" = "${ENVS_SHA256}" \
 && test "$(sha256sum vllm/models/qwen3_8_flash_next/model.py | cut -d' ' -f1)" = "${MODEL_SHA256}" \
 && test "$(sha256sum vllm/models/qwen3_8_flash_next/ple_layer.py | cut -d' ' -f1)" = "${PLE_LAYER_SHA256}" \
 && test "$(sha256sum vllm/models/qwen3_8_flash_next/ple_shared_table.py | cut -d' ' -f1)" = "${PLE_SHARED_TABLE_SHA256}" \
 && test "$(sha256sum tests/models/test_qwen3_8_flash_next_ple_shared_table.py | cut -d' ' -f1)" = "${TEST_SHA256}" \
 && test "$(git status --short | sort | tr -d ' ' | paste -sd,)" = \
    "Mvllm/envs.py,Mvllm/models/qwen3_8_flash_next/model.py,Mvllm/models/qwen3_8_flash_next/ple_layer.py,??tests/models/test_qwen3_8_flash_next_ple_shared_table.py,??vllm/models/qwen3_8_flash_next/ple_shared_table.py" \
 && rm -rf vllm/__pycache__ vllm/models/qwen3_8_flash_next/__pycache__ \
 && /opt/venv/bin/python -c "import vllm.models.qwen3_8_flash_next.ple_shared_table as m; print(m.__name__)" \
 && /opt/venv/bin/python -c "import vllm.envs as e; assert e.VLLM_PLE_SHARED_TABLE_ROLE == 'auto'; print(e.VLLM_PLE_SHARED_TABLE_DIR)" \
 && /opt/venv/bin/python -m vllm.models.qwen3_8_flash_next.ple_shared_table --dir /tmp/no-such-dir list \
 && printf 'overlay.repository=%s\noverlay.branch=%s\noverlay.commit=%s\noverlay.base.commit=%s\n' \
      "${VLLM_OVERLAY_REPOSITORY}" "${VLLM_OVERLAY_BRANCH}" "${VLLM_OVERLAY_COMMIT}" "${VLLM_BASE_COMMIT}" \
      > /opt/glm53-flash/overlay.identity

LABEL org.opencontainers.image.base.digest="sha256:7a425c6864b951bbc368111490a4b0ac69d8cd0dd4987075b2c7d40b753b1bf5" \
      org.opencontainers.image.base.name="docker.io/localinferencelab/vllm:jovian-judgement-community-20260911-r35" \
      org.opencontainers.image.source="${VLLM_OVERLAY_REPOSITORY}" \
      org.opencontainers.image.revision="${VLLM_OVERLAY_COMMIT}" \
      io.local-inference.overlay.name="qwen38-shared-ple-table" \
      io.local-inference.overlay.branch="${VLLM_OVERLAY_BRANCH}" \
      io.local-inference.overlay.base-commit="${VLLM_BASE_COMMIT}" \
      io.local-inference.overlay.files="vllm/envs.py,vllm/models/qwen3_8_flash_next/model.py,vllm/models/qwen3_8_flash_next/ple_layer.py,vllm/models/qwen3_8_flash_next/ple_shared_table.py,tests/models/test_qwen3_8_flash_next_ple_shared_table.py"
