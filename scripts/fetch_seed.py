#!/usr/bin/env python3
"""按 `data/seed-manifest.json` 把预置语料 / 预训练底座下载到本地缓存。

    python scripts/fetch_seed.py                 # 补齐 builtin + corpus + dataset
    python scripts/fetch_seed.py --all           # 连 dataset / online 这些归档根一起下
    python scripts/fetch_seed.py --models-only   # 只下 2 个 onnx 底座（prepare.sh 用）
    python scripts/fetch_seed.py --check         # 只体检，不下载（返回码 1 表示缺文件）

幂等：已存在且大小一致的文件直接跳过；`--force` 才重下。
下载走 CDN 优先、OSS 源站兜底（见 web/oss.py），逐条校验 sha256。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web import seed  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="按清单下载预置语料与预训练底座")
    ap.add_argument("--roots", default="builtin,corpus,dataset",
                    help="要下载的语料根（逗号分隔），默认 builtin,corpus,dataset")
    ap.add_argument("--all", action="store_true", help="下载清单里的全部根")
    ap.add_argument("--models-only", action="store_true", help="只下载预训练模型")
    ap.add_argument("--check", action="store_true", help="只检查，不下载")
    ap.add_argument("--force", action="store_true", help="忽略本地已有文件，全部重下")
    args = ap.parse_args(argv)

    manifest = seed.manifest()
    if not manifest.get("files") and not manifest.get("models"):
        print("没有清单（data/seed-manifest.json 为空或不存在）。\n"
              "先在有 OSS 钥匙的机器上跑：python scripts/oss_push.py --apply", file=sys.stderr)
        return 0

    if args.force:
        for entry in manifest.get("files") or []:
            seed.cache_path(entry).unlink(missing_ok=True)
        for entry in manifest.get("models") or []:
            seed.model_path(entry).unlink(missing_ok=True)

    download = not args.check
    stats = {"models": None, "files": None}
    if not args.models_only:
        roots = None if args.all else [r.strip() for r in args.roots.split(",") if r.strip()]
        stats["files"] = seed.ensure_local(roots, verify=True, download=download)
    stats["models"] = seed.ensure_models(verify=True, download=download)

    total = {"ok": 0, "downloaded": 0, "missing": 0, "failed": 0, "bytes": 0}
    for name, one in stats.items():
        if not one:
            continue
        for key in total:
            total[key] += one[key]
        if one["downloaded"]:
            print(f"[seed] {name}: 新下 {one['downloaded']} 个 / {one['bytes'] // 1024} KB")
    for name, one in stats.items():
        if one:
            for warning in one["warnings"][:5]:
                print(f"[seed] 警告 {name}: {warning}", file=sys.stderr)

    for root, count in sorted(seed.counts().items()):
        local = seed.local_root(root)
        have = sum(1 for label in ("pos", "neg") for p in (local / label).glob("*.wav")
                   if not p.name.startswith("._")) if local.is_dir() else 0
        print(f"[seed] {root:16s} 清单 {count['pos'] + count['neg']:4d}  本地 {have:4d}")

    if args.check:
        if total["missing"] or total["failed"]:
            print(f"[seed] 缺 {total['missing']} 个、失败 {total['failed']} 个", file=sys.stderr)
            return 1
        print("[seed] 全部就绪")
        return 0

    print(f"[seed] 完成：命中 {total['ok']}，新下 {total['downloaded']}，"
          f"失败 {total['failed']}")
    # 单个文件失败不致命（训练只用本地已有的），但要让调用方看见
    return 1 if total["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
