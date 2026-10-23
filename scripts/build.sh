#!/bin/bash
# 部署前构建：只做"能起得来"的检查——Python 语法、模板、预训练底座是否齐。
# 没有前端构建步骤（Jinja2 + 原生 JS），所以很快。
set -Eeuo pipefail

COZE_WORKSPACE_PATH="${COZE_WORKSPACE_PATH:-$(pwd)}"
cd "${COZE_WORKSPACE_PATH}"

PYTHON="${COZE_WORKSPACE_PATH}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "缺少 .venv：先跑 bash scripts/prepare.sh（或 bash scripts/bootstrap.sh）" >&2
  exit 1
fi

echo "[build] 语法检查…"
"${PYTHON}" -m compileall -q web trainer scripts

echo "[build] 模板与静态文件自检…"
"${PYTHON}" - <<'PY'
import pathlib, sys
root = pathlib.Path("web")
missing = [str(p) for name in ("base.html", "studio.html", "notes.html")
           for p in [root / "templates" / name] if not p.is_file()]
missing += [str(p) for p in [root / "static" / "app.js", root / "static" / "style.css"]
            if not p.is_file()]
if missing:
    print("缺少文件：" + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
print("templates/static OK")
PY

echo "[build] 预训练底座（缺失只警告：麦克风/训练 CNN 不受影响）…"
"${PYTHON}" scripts/fetch_seed.py --models-only || \
  echo "[build] 底座没拉齐：head 模型会不可用，稍后跑 python scripts/fetch_seed.py" >&2

echo "[build] 完成"
