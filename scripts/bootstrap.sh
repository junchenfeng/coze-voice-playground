#!/bin/bash
# 一键初始化：装依赖 → 拉预训练底座 → 提示领 .env → 拉种子语料。
#
# 幂等：任何一步重复跑都不会破坏已有环境。设计成"能跑完就说明能起服务"，
# 所以每一步失败都给出下一步该做什么，而不是一长串堆栈。
set -Eeuo pipefail

ROOT="${COZE_WORKSPACE_PATH:-$(pwd)}"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"

step() { printf '\n\033[1m[%s/5]\033[0m %s\n' "$1" "$2"; }

step 1 "Python 虚拟环境与依赖"
bash scripts/prepare.sh

step 2 "预训练底座（openWakeWord，约 2.4 MB）"
if "${PYTHON}" scripts/fetch_seed.py --models-only; then
  echo "底座就绪：models/backbone/openwakeword/"
else
  echo "底座没拉齐：麦克风录音、简单/复杂 CNN 训练都不受影响；head 模型会不可用。" >&2
  echo "稍后重试：${PYTHON} scripts/fetch_seed.py --models-only" >&2
fi

step 3 "配置文件 .env"
if [[ -f "${ROOT}/.env" ]]; then
  echo "已有 .env，跳过（要换一份就加 --force 语义的手动删除后重跑）"
else
  cat <<'EOF'
还缺 .env。向老师要 PROJENV_TOKEN，然后：

    .venv/bin/python scripts/fetch_projenv.py --token <PROJENV_TOKEN>

没有 .env 也能打开站点：录音、训练、测试全可用；只有「合成语料」和
「预置语料下载」会不可用（预置语料需要 OSS 凭证）。
EOF
fi

step 4 "种子语料（builtin + 历史人工语料）"
if "${PYTHON}" scripts/fetch_seed.py; then
  echo "种子语料就绪：.cache/seed/"
else
  echo "种子语料没拉齐。没有它也能用自己录的音训练，只是正负样本会少很多。" >&2
fi

step 5 "自检"
"${PYTHON}" - <<'PY'
import sys
sys.path.insert(0, ".")
from web import seed, settings
print(f"data 根     : {settings.DATA_ROOT}")
print(f"缓存根      : {settings.CACHE_ROOT}")
print(f"语料根      : {seed.counts()}")
from trainer.backbone import available
print(f"预训练底座  : {'就绪' if available() else '缺失（head 不可用）'}")
print(f"OSS 凭证    : {'已配置' if settings.OSS_ACCESS_KEY_ID else '未配置（预置语料需先下载）'}")
PY

cat <<'EOF'

下一步：
  bash scripts/dev.sh          # 起服务（默认 5000）
  curl -s localhost:5000/healthz
  bash scripts/validate.sh     # 跑测试
EOF
