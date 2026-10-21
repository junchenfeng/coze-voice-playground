"""单数据域存储层：语料 / 运行 / 自定义唤醒词 / active 指针。

和旧 `workspaces.py` 的区别：
- 没有昵称、没有 slug、没有「谁录的」：扣子编程项目本身就是一台独立机器，
  全站只有一份语料、一份运行记录。
- 语料分两层：
  * **麦克风录音**——真身留在 `data/corpus/{pos,neg}`（用户自己录的，唯一的本地音频）；
  * **合成音频**——wav 写 `.cache/synth/{pos,neg}`，元数据写 `data/synth.json`，
    同时尽力上传 OSS（见 `web/synth.py`）；
  * **预置语料**——在 OSS 上，按 `data/seed-manifest.json` 下载到 `.cache/seed/`，
    训练前由 `web/seed.py` 预取，列表页只给条数。
- 文件即状态：没有数据库，目录 + JSON 就是全部状态。

所有外部传入的名字（run_id / 文件名 / 词 id）先过白名单正则，再 resolve 校验，
必须仍然位于目标目录内；符号链接一律视为越权。
"""
import hashlib
import json
import os
import re
import secrets
import shutil
import time
import uuid
from pathlib import Path

from . import settings


class StoreError(Exception):
    """可预期的存储层错误（带 HTTP 状态码）。"""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


RUN_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
# 语料文件名：`pos-<时间>-<随机>.wav` / `pos-synth-<时间>-<随机>.wav` /
# 预置语料保留原名（`builtin@pos-synth-001.wav`），`@` 与 `_` 都在白名单里。
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,79}$")
TAG_RE = re.compile(r"^[a-z][a-z0-9]{0,7}$")
LABELS = ("pos", "neg")


# ---------------------------------------------------------------- 基础工具
def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_json(path: Path, obj) -> None:
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))


def _read_json(path: Path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _safe_child(base: Path, name: str, pattern: re.Pattern,
                root: Path | None = None) -> Path:
    if not isinstance(name, str) or not pattern.match(name):
        raise StoreError("非法名称", 400)
    root = (root or settings.DATA_ROOT).resolve()
    base = base.resolve()
    # 纵深防御：连 base 本身也必须在数据根下（防止目录被换成符号链接）
    if base != root and root not in base.parents:
        raise StoreError("非法路径", 400)
    path = (base / name).resolve()
    if path == base or base not in path.parents:
        raise StoreError("非法路径", 400)
    if path.is_symlink():
        raise StoreError("非法链接", 400)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 语料
def corpus_dir(label: str) -> Path:
    """麦克风录音的 pos / neg 目录。"""
    if label not in LABELS:
        raise StoreError("label 只能是 pos / neg")
    path = settings.CORPUS_ROOT / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def synth_dir(label: str) -> Path:
    """合成音频的本地缓存目录（可重建，不在 data/ 下）。"""
    if label not in LABELS:
        raise StoreError("label 只能是 pos / neg")
    path = settings.SYNTH_CACHE_ROOT / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_clip(folder: Path, head: str, data: bytes) -> str:
    """`<head><时间>-<随机>.wav`，随机后缀撞了就换一个（同一秒连录十几条会撞）。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for _ in range(8):
        name = f"{head}{stamp}-{secrets.token_hex(3)}.wav"
        path = folder / name
        if not path.exists():
            _atomic_write(path, data)
            return name
    raise StoreError("文件名一直冲突，请重试", 500)


def add_clip(label: str, data: bytes, *, tag: str = "") -> str:
    """写入麦克风语料目录。tag 进文件名（合成语料用 tag="synth"）。"""
    folder = corpus_dir(label)
    existing = sum(1 for p in folder.glob("*.wav") if NAME_RE.match(p.name))
    if existing >= settings.MAX_CLIPS_PER_LABEL:
        raise StoreError(
            f"语料每类最多 {settings.MAX_CLIPS_PER_LABEL} 条，先删几条再录", 400)
    if tag and not TAG_RE.match(tag):
        raise StoreError("非法标签", 400)
    return _write_clip(folder, f"{label}-" + (f"{tag}-" if tag else ""), data)


def clip_path(label: str, name: str) -> Path:
    """试听 / 删除：先找麦克风池，再找合成缓存。"""
    for folder, root in ((corpus_dir(label), settings.CORPUS_ROOT),
                         (synth_dir(label), settings.CACHE_ROOT)):
        try:
            path = _safe_child(folder, name, NAME_RE, root=root)
        except StoreError:
            continue
        if path.is_file():
            return path
    raise StoreError("文件不存在", 404)


def list_clips(label: str) -> list[dict]:
    from .audio import wav_info

    items = []
    roots = ((corpus_dir(label), "mic"), (synth_dir(label), "synth"))
    for folder, fallback_source in roots:
        for path in sorted(folder.glob("*.wav")):
            name = path.name
            if not NAME_RE.match(name):
                continue
            try:
                info = wav_info(path)
            except Exception:                       # 坏文件不该拖垮整个列表
                info = {"duration": 0, "sample_rate": 0}
            stat = path.stat()
            items.append({"name": name, "size": stat.st_size,
                          "source": "synth" if "-synth-" in name else fallback_source,
                          "mine": True,
                          "created_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                      time.localtime(stat.st_mtime)),
                          **info})
    items.sort(key=lambda it: it["created_at"], reverse=True)
    return items


def builtin_summary() -> dict:
    """预置语料（OSS 上的 builtin 根）在本地缓存里的条数。

    没下载过就是 0/available=False —— 页面据此把「带上预置语料」置灰，
    `scripts/fetch_seed.py` 跑完就有了。
    """
    out = {}
    for label in LABELS:
        d = settings.SEED_CACHE_ROOT / "builtin" / label
        # pathlib 的 glob 会匹配 ._xxx.wav（macOS 的 AppleDouble），手动排掉
        out[label] = (sum(1 for p in d.glob("*.wav") if not p.name.startswith("._"))
                      if d.is_dir() else 0)
    return {**out, "available": out["pos"] > 0 or out["neg"] > 0}


def corpus_summary() -> dict:
    pos, neg = list_clips("pos"), list_clips("neg")
    return {"pos": pos, "neg": neg,
            "counts": {"pos": len(pos), "neg": len(neg)},
            "ready": len(pos) >= settings.MIN_POS and len(neg) >= settings.MIN_NEG,
            "builtin": builtin_summary(),
            "min_pos": settings.MIN_POS, "min_neg": settings.MIN_NEG}


def delete_clip(label: str, name: str) -> None:
    """删本地这条（合成音频的 OSS 副本不跟着删：无凭证时也能正常工作）。"""
    clip_path(label, name).unlink()


# ---------------------------------------------------------------- 训练运行
def runs_dir() -> Path:
    settings.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return settings.RUNS_ROOT


def run_dir(run_id: str) -> Path:
    return _safe_child(runs_dir(), run_id, RUN_RE)


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def create_run(config: dict) -> str:
    for _ in range(3):
        run_id = new_run_id()
        path = run_dir(run_id)
        try:
            path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:              # 同一秒 + 随机后缀撞车，换一个
            continue
    else:
        raise StoreError("创建运行目录失败，请重试", 500)

    _write_json(path / "config.json", config)
    _write_json(path / "status.json", {
        "state": "queued", "pid": None, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return run_id


def read_config(run_id: str) -> dict:
    cfg = _read_json(run_dir(run_id) / "config.json")
    if cfg is None:
        raise StoreError("运行不存在", 404)
    return cfg


def read_status(run_id: str) -> dict:
    return _read_json(run_dir(run_id) / "status.json", {}) or {}


def write_status(run_id: str, status: dict) -> None:
    _write_json(run_dir(run_id) / "status.json", status)


def read_result(run_id: str) -> dict | None:
    return _read_json(run_dir(run_id) / "result.json")


def read_history(run_id: str) -> list[dict]:
    """progress.jsonl 是训练过程中唯一的增量来源；坏行直接跳过。"""
    path = run_dir(run_id) / "progress.jsonl"
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return rows
    return rows


def list_runs() -> list[dict]:
    active = get_active()
    rows = []
    for path in runs_dir().iterdir():
        if not path.is_dir() or path.is_symlink() or not RUN_RE.match(path.name):
            continue
        status = _read_json(path / "status.json", {}) or {}
        cfg = _read_json(path / "config.json", {}) or {}
        result = _read_json(path / "result.json", {}) or {}
        rows.append({
            "id": path.name,
            "config": cfg,
            "state": status.get("state", "unknown"),
            "created_at": status.get("created_at") or cfg.get("created_at"),
            "elapsed_s": result.get("elapsed_s"),
            "params_total": result.get("params_total"),
            "final": result.get("final"),
            "active": path.name == active,
            # 「这次运行算不算一个能用的模型」的唯一判据：真有权重文件
            "has_ckpt": (path / "model.pt").is_file(),
        })
    rows.sort(key=lambda r: r["id"], reverse=True)
    return rows


# ---------------------------------------------------------------- active 指针
def get_active() -> str | None:
    record = _read_json(settings.ACTIVE_PATH, {}) or {}
    run_id = record.get("run_id")
    return run_id if isinstance(run_id, str) and RUN_RE.match(run_id) else None


def set_active(run_id: str) -> None:
    run_dir(run_id)
    _write_json(settings.ACTIVE_PATH, {
        "run_id": run_id, "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


def clear_active() -> None:
    """取消「当前模型」指针（删掉正被使用的模型时必须走这里）。"""
    try:
        settings.ACTIVE_PATH.unlink()
    except FileNotFoundError:
        pass


def delete_run(run_id: str) -> None:
    """删掉一次运行：权重、曲线、日志整目录一起走，不可恢复。"""
    path = run_dir(run_id)
    if not (path / "config.json").is_file():
        raise StoreError("运行不存在", 404)
    try:
        shutil.rmtree(path)
    except OSError as exc:                        # 权限 / 占用：别假装删成功
        raise StoreError(f"删除失败：{exc}", 500) from exc
    if get_active() == run_id:                    # 删的正是在用的模型 → 指针一起清
        clear_active()
