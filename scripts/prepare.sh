#!/bin/bash
# 装依赖：Python 虚拟环境 + 训练依赖（torch CPU 单独一层）。
#
# 放在 Coze Drive 上时，.venv 会落在 /tmp（同步盘里塞 500 MB 的 site-packages
# 会拖慢整个盘），项目根只留一个 symlink——和参考项目的 node_modules 同思路。
set -Eeuo pipefail

ROOT="${COZE_WORKSPACE_PATH:-$(pwd)}"
cd "${ROOT}"

VENV="${ROOT}/.venv"
PIP_INDEX="${NOVA_PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"

pick_python() {
  for cand in "${PYTHON_BIN:-}" python3.12 python3; do
    if [[ -n "${cand}" ]] && command -v "${cand}" >/dev/null 2>&1; then
      if "${cand}" -c 'import sys;raise SystemExit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then
        echo "${cand}"; return 0
      fi
    fi
  done
  echo "找不到 Python ≥ 3.10。扣子编程的 .coze 里声明了 python-3.12；本地请装 3.12" >&2
  return 1
}

PY="$(pick_python)"
echo "[prepare] python: $($PY -V 2>&1)  ($(command -v "$PY"))"

case "${ROOT}" in
  /Coze/*)
    # 同步盘：venv 放 /tmp，项目里只留 symlink。已存在且不是 symlink 就拒绝覆盖，
    # 免得把别人手建的虚拟环境静默换掉。
    if [[ -e "${VENV}" && ! -L "${VENV}" ]]; then
      echo "[prepare] ${VENV} 是真实目录，不是本脚本管理的 symlink；请先自行处理" >&2
      exit 1
    fi
    key="$(printf '%s' "${ROOT}" | md5sum | cut -c1-8)"
    target="/tmp/nova-venv/$(basename "${ROOT}")-${key}"
    mkdir -p /tmp/nova-venv
    if [[ ! -x "${target}/bin/python" ]]; then
      echo "[prepare] 建虚拟环境（/tmp）: ${target}"
      "${PY}" -m venv "${target}"
    fi
    [[ -L "${VENV}" ]] || ln -s "${target}" "${VENV}"
    ;;
  *)
    if [[ ! -x "${VENV}/bin/python" ]]; then
      echo "[prepare] 建虚拟环境: ${VENV}"
      "${PY}" -m venv "${VENV}"
    fi
    ;;
esac

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -q --upgrade "pip>=24" -i "${PIP_INDEX}" || true

echo "[prepare] 装基础依赖…"
pip install -q -r requirements.txt -i "${PIP_INDEX}"

if [[ "${NOVA_SKIP_TORCH:-}" != "1" ]]; then
  if python -c "import torch" >/dev/null 2>&1; then
    echo "[prepare] torch 已装：$(python -c 'import torch;print(torch.__version__)')"
  else
    echo "[prepare] 装 torch CPU（约 200 MB，先试官方 CPU 源）…"
    # requirements-torch.txt 自带 --index-url，别再用 -i 覆盖它。
    # 装不上不阻塞：录音/合成/列表都能用，只是训练会失败（bootstrap.md 里有备线）。
    if ! pip install -q -r requirements-torch.txt; then
      pip install -q -r requirements-torch.txt \
        -i "${PIP_INDEX}" --extra-index-url https://download.pytorch.org/whl/cpu \
        || echo "[prepare] torch 没装上：训练会失败，其余功能可用" >&2
    fi
  fi
fi

echo "[prepare] 预训练底座自检…"
python scripts/fetch_seed.py --models-only --check || true

echo "[prepare] 完成。下一步：python scripts/fetch_projenv.py --token <PROJENV_TOKEN>"
