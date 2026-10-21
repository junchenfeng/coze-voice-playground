#!/bin/bash
# 生产态启动：不开 --reload，前台跑（进程由 coze 托管）。
set -Eeuo pipefail

COZE_WORKSPACE_PATH="${COZE_WORKSPACE_PATH:-$(pwd)}"
PORT=5000
DEPLOY_RUN_PORT="${DEPLOY_RUN_PORT:-${PORT}}"

cd "${COZE_WORKSPACE_PATH}"

PYTHON="${COZE_WORKSPACE_PATH}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo ".venv 不存在，先跑 bash scripts/prepare.sh" >&2
  exit 1
fi

echo "Starting FastAPI on port ${DEPLOY_RUN_PORT}..."
exec "${PYTHON}" -m uvicorn web.app:app --host 0.0.0.0 --port "${DEPLOY_RUN_PORT}" \
  --workers 1 --timeout-keep-alive 75
