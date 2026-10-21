"""预置语料 / 预训练底座的清单驱动缓存。

仓库里只有 `data/seed-manifest.json`（每条 wav / onnx 的 key、sha256、大小、标签、
时长），音频本体在 OSS。用到时按清单下载到 `.cache/`——那里随时可以删掉重建，
`data/` 永远不存种子音频。

命中判断刻意做成两段（性能考虑）：
1. 文件存在且大小一致 → 直接认为命中，不读 800 个文件的 sha256；
2. 大小不一致才做 sha256，对得上也认（不同工具的写入方式可能差几字节）。
`--check` 就是 `verify=True, download=False`。
"""
import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

from . import oss, settings

_lock = threading.Lock()
_cache: dict = {"mtime": None, "data": None}


class SeedError(Exception):
    """清单缺失 / 损坏。"""


def manifest_path() -> Path:
    return Path(settings.SEED_MANIFEST)


def manifest() -> dict:
    """读清单（按 mtime 失效的进程内缓存；清单很小）。"""
    path = manifest_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {"schema": 0, "files": [], "models": []}
    with _lock:
        if _cache["mtime"] == mtime and _cache["data"] is not None:
            return _cache["data"]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SeedError(f"清单读不出来：{exc}") from exc
    if not isinstance(data, dict):
        raise SeedError("清单格式不对")
    data.setdefault("files", [])
    data.setdefault("models", [])
    with _lock:
        _cache.update({"mtime": mtime, "data": data})
    return data


def files(roots=None) -> list[dict]:
    rows = manifest().get("files") or []
    if roots is None:
        return rows
    wanted = {str(r) for r in roots}
    return [row for row in rows if str(row.get("root")) in wanted]


def models() -> list[dict]:
    return manifest().get("models") or []


def roots() -> list[str]:
    return sorted({str(row.get("root")) for row in files()})


def counts() -> dict:
    """每个根的条数（页面/日志用）。"""
    out: dict[str, dict] = {}
    for row in files():
        root = str(row.get("root"))
        label = str(row.get("label"))
        out.setdefault(root, {"pos": 0, "neg": 0})
        if label in out[root]:
            out[root][label] += 1
    return out


# ---------------------------------------------------------------- 路径
def cache_path(entry: dict) -> Path:
    """`seed/<root>/<label>/<name>` → `.cache/seed/<root>/<label>/<name>`。"""
    rel = str(entry.get("path") or "").lstrip("/")
    if not rel:
        raise SeedError("清单条目缺少 path")
    return (settings.CACHE_ROOT / rel).resolve()


def model_path(entry: dict) -> Path:
    """模型条目是项目相对路径（`models/backbone/openwakeword/xxx.onnx`）。"""
    rel = str(entry.get("path") or "").lstrip("/")
    if not rel:
        raise SeedError("清单条目缺少 path")
    return (settings.BASE_DIR / rel).resolve()


def local_root(root: str) -> Path:
    return (settings.SEED_CACHE_ROOT / str(root)).resolve()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_ok(path: Path, entry: dict, verify: bool) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    size = entry.get("size")
    if isinstance(size, int) and size >= 0 and stat.st_size == size:
        return True
    if not verify:
        return False
    digest = entry.get("sha256")
    if not digest:
        return False
    return _sha256(path) == digest


def _download(entry: dict, dest: Path) -> int:
    key = str(entry.get("key") or "")
    if not key:
        raise SeedError("清单条目缺少 key")
    errors = []
    for url in oss.download_urls(key):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        digest = entry.get("sha256")
        if digest and hashlib.sha256(data).hexdigest() != digest:
            errors.append(f"{url}: sha256 不符")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        return len(data)
    raise SeedError("下载失败：" + (errors[0] if errors else "没有可用地址"))


def _walk(entries: list[dict], path_of, verify: bool, download: bool) -> dict:
    stats = {"total": len(entries), "ok": 0, "downloaded": 0, "missing": 0,
             "failed": 0, "bytes": 0, "warnings": []}
    for entry in entries:
        try:
            dest = path_of(entry)
        except SeedError as exc:
            stats["failed"] += 1
            stats["warnings"].append(str(exc))
            continue
        if _looks_ok(dest, entry, verify):
            stats["ok"] += 1
            continue
        if not download:
            stats["missing"] += 1
            continue
        try:
            size = _download(entry, dest)
        except SeedError as exc:
            stats["failed"] += 1
            stats["warnings"].append(f"{entry.get('path')}: {exc}")
            continue
        stats["downloaded"] += 1
        stats["bytes"] += size
    return stats


def ensure_local(roots_=None, verify: bool = True, download: bool = True) -> dict:
    """把指定根的种子音频准备好；网络异常只记 warning，不阻断训练。"""
    return _walk(files(roots_), cache_path, verify, download)


def ensure_models(verify: bool = True, download: bool = True) -> dict:
    return _walk(models(), model_path, verify, download)


def materialize(roots_=None) -> dict:
    """训练前拿到「根 → 本地目录」映射（只返回真的存在的目录）。"""
    out = {}
    for root in (roots_ or roots()):
        path = local_root(root)
        if path.is_dir() and any(path.glob("*/*.wav")):
            out[root] = path
    return out


_started = False


def prefetch_async(roots_=("builtin", "corpus")) -> bool:
    """服务启动后后台补齐预置语料（幂等，只起一次）。

    放后台的理由：coze 冷启动时 ~40 MB 种子音频要下几秒，把首屏/首个训练请求
    卡在这里体验很差。失败只写日志——训练时 `jobs._prefetch()` 还会再试一次。
    """
    global _started
    if _started:
        return False
    _started = True
    try:
        if not manifest().get("files"):
            return False
    except SeedError:
        return False

    def run() -> None:
        try:
            stats = ensure_local(list(roots_), verify=False, download=True)
        except Exception:                       # noqa: BLE001 - 后台任务不该冒泡
            return
        if stats["downloaded"]:
            print(f"[seed] 预置语料补齐 {stats['downloaded']} 条 / "
                  f"{stats['bytes'] // 1024} KB", flush=True)

    threading.Thread(target=run, daemon=True, name="seed-prefetch").start()
    return True
