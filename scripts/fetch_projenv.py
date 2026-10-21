#!/usr/bin/env python3
"""从线上 `GET https://voice.aibeaver.cn/projenv` 领取本项目的 `.env`。

    python scripts/fetch_projenv.py --token <PROJENV_TOKEN>
    python scripts/fetch_projenv.py --token ... --check     # 只体检，不写盘
    python scripts/fetch_projenv.py --token ... --url http://127.0.0.1:8000/projenv

为什么单独一个端点：`/credential` 是给桌面端的 ASR/TTS 下发口，只给两项；本项目
还需要 OSS 读写和 TTS 地址，所以线上另开了 `/projenv`（独立 token、独立模块）。
返回体就是 `.env` 文本，这里只做三件事：校验必需键、原子写、`chmod 600`。
"""
import argparse
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "https://voice.aibeaver.cn/projenv"

# 缺任何一项都跑不起来的功能 → 直接失败，别让人带着半个 .env 去 debug
REQUIRED = (
    "DOUBAO_APP_ID", "DOUBAO_ACCESS_TOKEN",      # 合成语料（缺了只是合成不可用）
    "TENCENT_SECRET_ID", "TENCENT_SECRET_KEY",   # 桌面端 ASR（线上校验用）
    "OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET",  # 种子语料下载 / 合成音频上传
    "OSS_BUCKET", "OSS_ENDPOINT",
)


def parse_env(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def fetch(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        if resp.status != 200:
            raise SystemExit(f"projenv 返回 {resp.status}")
        return resp.read().decode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="领取 .env（来自线上 /projenv）")
    ap.add_argument("--token", default=os.environ.get("PROJENV_TOKEN", ""),
                    help="projenv token（也可用环境变量 PROJENV_TOKEN）")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"接口地址，默认 {DEFAULT_URL}")
    ap.add_argument("--out", default=str(ROOT / ".env"), help="输出文件，默认 .env")
    ap.add_argument("--check", action="store_true", help="只校验，不写盘")
    args = ap.parse_args(argv)

    if not args.token:
        print("缺少 token：--token <PROJENV_TOKEN>，或设环境变量 PROJENV_TOKEN\n"
              "（token 找老师要；它和桌面端 /credential 的 token 不是同一个）", file=sys.stderr)
        return 2

    sep = "&" if "?" in args.url else "?"
    url = f"{args.url}{sep}token={urllib.parse.quote(args.token)}"
    try:
        text = fetch(url)
    except urllib.error.HTTPError as exc:
        hint = {401: "token 不对", 503: "线上还没配 PROJENV_TOKEN"}.get(exc.code, "")
        print(f"projenv 返回 {exc.code} {hint}".rstrip(), file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"连不上 {args.url}：{exc.reason}", file=sys.stderr)
        return 1

    env = parse_env(text)
    missing = [key for key in REQUIRED if not env.get(key)]
    if missing:
        print("projenv 返回的 .env 缺这些键：" + ", ".join(missing)
              + "\n（线上 .env 没配全，或 OSS 凭据还没接进 projenv）", file=sys.stderr)
        return 1
    if env.get("OSS_PREFIX") != "classroom/voice-playground":
        print(f"注意：OSS_PREFIX = {env.get('OSS_PREFIX')!r}，"
              "与清单里的 classroom/voice-playground 不一致时下载会 404", file=sys.stderr)

    if args.check:
        print(f"校验通过：{len(env)} 个键（未写盘）")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        backup = out.with_suffix(out.suffix + ".bak")
        backup.write_bytes(out.read_bytes())
        os.chmod(backup, 0o600)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, out)
    print(f"写入 {out}（{len(env)} 个键，权限 600）")
    if out.exists() and out.with_suffix(out.suffix + ".bak").exists():
        print(f"旧文件备份在 {out.with_suffix(out.suffix + '.bak')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
