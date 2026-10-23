#!/bin/bash
# 开发态启动：uvicorn --reload，detached 到独立进程组（照抄参考项目 dev.sh 的做法）。
#
# 为什么不能直接 `uvicorn ...`：coze-daemon 会在 runtime shell 退出时清理原进程组，
# 服务会被一起杀掉。所以借 node 的 detached spawn 把 wrapper 挪进独立 session，
# wrapper 再带一个到点自尽的 watchdog，避免端口/内存长期泄露。
set -Eeuo pipefail

PORT=5000
COZE_WORKSPACE_PATH="${COZE_WORKSPACE_PATH:-$(pwd)}"
DEPLOY_RUN_PORT="${DEPLOY_RUN_PORT:-${PORT}}"

cd "${COZE_WORKSPACE_PATH}"

kill_port_if_listening() {
    local pids=""
    if command -v ss >/dev/null 2>&1; then
      pids=$(ss -H -lntp 2>/dev/null | awk -v port="${DEPLOY_RUN_PORT}" '$4 ~ ":"port"$"' | grep -o 'pid=[0-9]*' | cut -d= -f2 | paste -sd' ' - || true)
    elif command -v lsof >/dev/null 2>&1; then
      # macOS / 无 ss 的环境：本地调试时也能清干净
      pids=$(lsof -ti "tcp:${DEPLOY_RUN_PORT}" 2>/dev/null | paste -sd' ' - || true)
    fi
    if [[ -z "${pids}" ]]; then
      echo "Port ${DEPLOY_RUN_PORT} is free."
      return
    fi
    echo "Port ${DEPLOY_RUN_PORT} in use by PIDs: ${pids} (SIGKILL)"
    echo "${pids}" | xargs -I {} kill -9 {}
    sleep 1
}

LOG_DIR="${COZE_LOG_DIR:-${COZE_WORKSPACE_PATH}/logs}"
LOG_FILE="${LOG_DIR}/server.log"
PID_FILE="${LOG_DIR}/server.pid"
MAX_RUNTIME_SECONDS="${NOVA_MAX_RUNTIME_SECONDS:-3600}"

PYTHON="${COZE_WORKSPACE_PATH}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo ".venv 不存在，先跑 bash scripts/prepare.sh" >&2
  exit 1
fi

timeout_watchdog_enabled() {
  [[ -z "${COZE_EVAL:-}" && -z "${COZE_PROJECT_TYPE:-}" ]]
}

RUN_WITH_TIMEOUT="$(declare -f timeout_watchdog_enabled)"'
timeout_seconds=$1
shift

"$@" &
child_pid=$!

if timeout_watchdog_enabled; then
( trap "" TERM
  sleep "${timeout_seconds}"
  echo "[dev] 后台进程运行超过 ${timeout_seconds}s，回收进程组 $$。"
  kill -TERM -- "-$$" 2>/dev/null || true
  sleep 5
  kill -KILL -- "-$$" 2>/dev/null || true
) &
fi

wait "${child_pid}"
kill -KILL -- "-$$" 2>/dev/null || true
'

spawn_detached() {
  local cwd="$1"
  local log_file="$2"
  shift 2

  node - "$cwd" "$log_file" \
    /bin/bash -c "${RUN_WITH_TIMEOUT}" detached-runner "${MAX_RUNTIME_SECONDS}" "$@" <<'NODE'
const fs = require('node:fs');
const { spawn } = require('node:child_process');

const [cwd, logFile, command, ...args] = process.argv.slice(2);
if (!cwd || !logFile || !command) {
  throw new Error('spawn_detached 缺少 cwd、log_file 或 command');
}

const logFd = fs.openSync(logFile, 'a');
try {
  const child = spawn(command, args, {
    cwd,
    detached: true,
    env: process.env,
    stdio: ['ignore', logFd, logFd],
  });
  child.unref();
  process.stdout.write(String(child.pid));
} finally {
  fs.closeSync(logFd);
}
NODE
}

stop_detached() {
  local pid="${1:-}"
  if [[ -z "${pid}" ]]; then
    return
  fi
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  sleep 1
  kill -KILL -- "-${pid}" 2>/dev/null || true
}

if [[ -f "${PID_FILE}" ]]; then
  stop_detached "$(cat "${PID_FILE}" 2>/dev/null || true)"
  rm -f "${PID_FILE}"
fi

echo "Clearing port ${DEPLOY_RUN_PORT} before start."
kill_port_if_listening
echo "Starting FastAPI (uvicorn --reload) on port ${DEPLOY_RUN_PORT}..."

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

export PORT="${DEPLOY_RUN_PORT}"
server_pid="$(spawn_detached "${COZE_WORKSPACE_PATH}" "${LOG_FILE}" \
  "${PYTHON}" -m uvicorn web.app:app --host 0.0.0.0 --port "${DEPLOY_RUN_PORT}" \
  --reload --reload-dir web --reload-dir trainer)"
echo "${server_pid}" > "${PID_FILE}"

sleep 2
if [[ -z "${server_pid}" ]] || ! kill -0 "${server_pid}" 2>/dev/null; then
  echo "Dev server failed to start. See ${LOG_FILE}." >&2
  tail -n 30 "${LOG_FILE}" >&2 || true
  rm -f "${PID_FILE}"
  exit 1
fi

echo "Dev server started (PID: ${server_pid})."
echo "Log file: ${LOG_FILE}"
echo "Health:  curl -s http://127.0.0.1:${DEPLOY_RUN_PORT}/healthz"
