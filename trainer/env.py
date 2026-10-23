"""最小 .env 读取器（不引 python-dotenv）。

容器里靠 `docker compose env_file` 注入环境变量；本地直接 `uvicorn web.app:app`
或 `python scripts/build_dataset.py` 时没人读 `.env`，所以入口处补一次。

规则：只在环境变量**缺失**时填，绝不覆盖真实 env（容器 / CI 注入的优先）；
支持 `KEY=VALUE`、`export KEY=VALUE`、值两侧的单双引号；其余行（注释、空行）跳过。
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path | None = None) -> None:
    target = Path(path) if path else BASE_DIR / ".env"
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
