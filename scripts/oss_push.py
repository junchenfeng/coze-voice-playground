#!/usr/bin/env python3
"""把种子语料 / 预训练底座推上 OSS，并重写 `data/seed-manifest.json`。

    # 1) 先看一眼要传什么（默认 dry-run，不写盘、不上传）
    python scripts/oss_push.py --source ~/git/nova-voice-playground

    # 2) 确认后真传（要 OSS 写凭证：环境变量或 ai-gallery/config.json）
    python scripts/oss_push.py --source ~/git/nova-voice-playground --apply

    # 3) 补传运行时生成、当时没传上去的合成音频
    python scripts/oss_push.py --synth --apply

凭据只从这两处读（不复制、不硬编码）：
- `.env` 的 `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`；
- `--gallery-config`（默认 `~/git/ai-gallery/config.json`）的 `oss` 块。
日志只打印 key id 前 4 位。

去重：`online-20261020/` 是 `corpus/` 的历史镜像（同名同 sha256），只上传一份，
另一份记在清单的 `deduped` 里——省流量，也让训练不会把同一段声音数两遍。
"""
import argparse
import hashlib
import json
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web import oss, settings  # noqa: E402

DEFAULT_ROOTS = ("builtin", "corpus", "dataset", "online-20261020")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _wav_meta(path: Path) -> dict:
    try:
        with wave.open(str(path), "rb") as w:
            frames, rate = w.getnframes(), w.getframerate() or 0
    except (OSError, wave.Error, EOFError):
        return {"duration": 0.0, "sample_rate": 0}
    return {"duration": round(frames / rate, 3) if rate else 0.0, "sample_rate": rate}


def scan(source: Path, roots) -> tuple[list[dict], list[dict], list[dict]]:
    """扫源仓库的 data/<root>/{pos,neg}/*.wav，按 sha256 去重。"""
    files: list[dict] = []
    deduped: list[dict] = []
    seen: dict[str, str] = {}
    for root in roots:
        for label in ("pos", "neg"):
            folder = source / "data" / root / label
            if not folder.is_dir():
                continue
            for wav in sorted(folder.glob("*.wav")):
                if wav.name.startswith("._") or not wav.is_file():
                    continue
                digest = _sha256(wav)
                rel = f"seed/{root}/{label}/{wav.name}"
                if digest in seen:
                    deduped.append({"path": rel, "same_as": seen[digest],
                                    "sha256": digest, "size": wav.stat().st_size})
                    continue
                seen[digest] = rel
                meta = _wav_meta(wav)
                files.append({
                    "path": rel, "root": root, "label": label, "name": wav.name,
                    "key": oss.object_key(rel), "sha256": digest,
                    "size": wav.stat().st_size,
                    "synth": "-synth-" in wav.name,
                    **meta,
                })
    models = []
    for onnx in sorted((source / "models" / "backbone" / "openwakeword").glob("*.onnx")):
        if onnx.name.startswith("._"):
            continue
        rel = f"models/backbone/openwakeword/{onnx.name}"
        models.append({"path": rel, "key": oss.object_key(rel), "name": onnx.name,
                       "sha256": _sha256(onnx), "size": onnx.stat().st_size})
    return files, models, deduped


def push(items: list[tuple[dict, Path, str]], apply: bool,
         force: bool, gallery_config: str | None) -> dict:
    """items = [(清单条目, 本地文件, content-type), ...]。"""
    stats = {"uploaded": 0, "skipped": 0, "failed": 0, "bytes": 0}
    if not apply:
        stats["bytes"] = sum(entry["size"] for entry, _local, _ct in items)
        return stats
    if not oss.configured(gallery_config):
        raise SystemExit("没有 OSS 写凭证：配 OSS_ACCESS_KEY_ID/SECRET，"
                         "或让 --gallery-config 指向 ai-gallery/config.json")

    for entry, local, content_type in items:
        if not local.is_file():
            print(f"  跳过（源文件不在）：{local}", file=sys.stderr)
            stats["failed"] += 1
            continue
        try:
            if not force and oss.object_size(entry["key"], gallery_config) == entry["size"]:
                stats["skipped"] += 1
                continue
            oss.put_object(entry["key"], local.read_bytes(),
                           content_type=content_type, gallery_config=gallery_config)
        except oss.OSSError as exc:
            print(f"  失败 {entry['path']}: {exc}", file=sys.stderr)
            stats["failed"] += 1
            continue
        stats["uploaded"] += 1
        stats["bytes"] += entry["size"]
    return stats


def push_synth(apply: bool, gallery_config: str | None) -> dict:
    """补传 data/synth.json 里 oss_key 为空、但本地缓存还在的合成音频。"""
    index_path = Path(settings.SYNTH_INDEX)
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("没有 data/synth.json（还没合成过音频）")
        return {"uploaded": 0, "skipped": 0, "failed": 0}
    stats = {"uploaded": 0, "skipped": 0, "failed": 0}
    changed = False
    for clip in index.get("clips", []):
        name, label = clip.get("name"), clip.get("label")
        if not name or not label:
            continue
        local = settings.SYNTH_CACHE_ROOT / label / name
        key = oss.object_key(f"synth/{label}/{name}")
        if clip.get("oss_key"):
            stats["skipped"] += 1
            continue
        if not local.is_file():
            print(f"  跳过（本地缓存已清理）：{local}", file=sys.stderr)
            stats["skipped"] += 1
            continue
        if not apply:
            stats["uploaded"] += 1
            continue
        try:
            oss.put_object(key, local.read_bytes(), content_type="audio/wav",
                           gallery_config=gallery_config)
        except oss.OSSError as exc:
            print(f"  失败 {name}: {exc}", file=sys.stderr)
            stats["failed"] += 1
            continue
        clip["oss_key"] = key
        changed = True
        stats["uploaded"] += 1
    if changed:
        index["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = index_path.with_name(f".{index_path.name}.tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, index_path)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="把种子语料推到 OSS 并生成清单")
    ap.add_argument("--source", default="~/git/nova-voice-playground",
                    help="源仓库（含 data/ 与 models/），默认 ~/git/nova-voice-playground")
    ap.add_argument("--roots", default=",".join(DEFAULT_ROOTS),
                    help=f"要上传的语料根，默认 {','.join(DEFAULT_ROOTS)}")
    ap.add_argument("--gallery-config", default=None,
                    help="ai-gallery/config.json 路径（默认自动找）")
    ap.add_argument("--manifest", default=str(settings.SEED_MANIFEST),
                    help="清单输出路径，默认 data/seed-manifest.json")
    ap.add_argument("--apply", action="store_true", help="真的上传并写清单（默认 dry-run）")
    ap.add_argument("--force", action="store_true", help="忽略远端已存在，全部重传")
    ap.add_argument("--synth", action="store_true", help="只补传运行时合成的音频")
    args = ap.parse_args(argv)

    print("[oss]", oss.describe(args.gallery_config))

    if args.synth:
        stats = push_synth(args.apply, args.gallery_config)
        print(f"[oss] 合成音频：上传 {stats['uploaded']}，跳过 {stats['skipped']}，"
              f"失败 {stats['failed']}" + ("" if args.apply else "（dry-run）"))
        return 1 if stats["failed"] else 0

    source = Path(args.source).expanduser()
    if not (source / "data").is_dir():
        print(f"源仓库不对：{source}/data 不存在", file=sys.stderr)
        return 2
    roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    files, models, deduped = scan(source, roots)
    print(f"[oss] 扫描到 {len(files)} 个 wav / {len(models)} 个 onnx，"
          f"去重 {len(deduped)} 个；合计 {(sum(f['size'] for f in files) + sum(m['size'] for m in models)) // 1024} KB")
    for root in roots:
        rows = [f for f in files if f["root"] == root]
        if rows:
            print(f"       {root:18s} pos {sum(1 for f in rows if f['label'] == 'pos'):4d} "
                  f"neg {sum(1 for f in rows if f['label'] == 'neg'):4d}")

    # 清单里的 path 是**项目内缓存路径**（seed/<root>/<label>/<name>），
    # 源仓库的实际布局是 data/<root>/<label>/<name>，两者不能直接拼。
    items = [(entry, source / "data" / entry["root"] / entry["label"] / entry["name"],
              "audio/wav") for entry in files]
    items += [(entry, source / entry["path"], "application/octet-stream") for entry in models]
    stats = push(items, args.apply, args.force, args.gallery_config)
    if not args.apply:
        print(f"[oss] dry-run：可传 {len(files) + len(models)} 个 / "
              f"{stats['bytes'] // 1024} KB。加 --apply 真传")
        return 0

    manifest = {
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prefix": settings.OSS_PREFIX,
        "bucket": settings.OSS_BUCKET,
        "endpoint": settings.OSS_ENDPOINT,
        "cdn_base": settings.OSS_CDN_BASE,
        "roots": roots,
        "files": files,
        "models": models,
        "deduped": deduped,
    }
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    if stats["failed"]:
        # 有失败就别覆盖正式清单：半份清单会让 fetch_seed 到处 404
        out = out.with_name(out.stem + ".partial.json")
        print(f"[oss] 有 {stats['failed']} 个失败，清单先写 {out}（修好后重跑 --apply）",
              file=sys.stderr)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    print(f"[oss] 上传 {stats['uploaded']}，跳过 {stats['skipped']}，失败 {stats['failed']}；"
          f"清单写入 {out}")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
