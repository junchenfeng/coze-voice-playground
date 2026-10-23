#!/bin/bash
# 验收：pytest（没装 torch 的机器上，torch 相关用例会自动 skip）。
set -Eeuo pipefail

COZE_WORKSPACE_PATH="${COZE_WORKSPACE_PATH:-$(pwd)}"
cd "${COZE_WORKSPACE_PATH}"

PYTHON="${COZE_WORKSPACE_PATH}/.venv/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"

exec "${PYTHON}" -m pytest tests -q "$@"
