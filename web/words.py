"""自定义唤醒词数据集。

为什么单独存：默认语料是「全班喊 buddy助手」的那一份，一个词一套正例没法塞进同一个
pos/ 里——同一段音频对 A 词是正例、对 B 词是负例。所以每个词一个目录，只放**这个词
的正例**；负例永远来自本地语料 + 预置语料（见 trainer/data.py 的 `pos_dir` 口径）。

    data/words/<word_id>/
    ├── word.json     # {"id", "text", "created_at"}
    └── pos/*.wav     # `pos-[synth-]<时间>-<随机>.wav`

word_id = `w-<sha1(NFKC(文本))[:8]>`：同一段文本永远落到同一个目录，改词就是换
数据集，重名不会打架。run 的 config.json 里记 `wake_word:{id,text}`，历史运行靠它
显示唤醒词前缀。
"""
import hashlib
import re
import time
import unicodedata
from pathlib import Path

from . import settings, store
from .store import StoreError

WORD_ID_RE = re.compile(r"^w-[0-9a-f]{8}$")
# 当前词指针：界面上只展示一个词，旧词目录留盘（不可见、按 id 仍可访问）。
CURRENT_FILE = "current.json"


# ---------------------------------------------------------------- 文本 / id
def normalize_text(text: str) -> str:
    """唤醒词原文：NFKC + 去首尾空白；控制字符、换行、路径分隔符一律拒绝。

    不做大小写折叠：`Hi Buddy` 和 `hi buddy` 发音一样但也可能被当成两个词，
    让他自己看着办；只有 NFKC（全角/半角）必须归一，否则「小福小福」会被当成
    两个不同的数据集。
    """
    if not isinstance(text, str):
        raise StoreError("唤醒词不能为空")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise StoreError("唤醒词里有非法字符")
    name = unicodedata.normalize("NFKC", text).strip()
    name = re.sub(r" +", " ", name)
    if not 1 <= len(name) <= settings.WORD_TEXT_MAX:
        raise StoreError(f"唤醒词长度需在 1~{settings.WORD_TEXT_MAX} 个字符之间")
    if any(ch in name for ch in "/\\"):
        raise StoreError("唤醒词里有非法字符")
    return name


def word_id(text: str) -> str:
    return "w-" + hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------- 目录
def words_root() -> Path:
    settings.WORDS_ROOT.mkdir(parents=True, exist_ok=True)
    return settings.WORDS_ROOT


def word_dir(wid: str) -> Path:
    """白名单 + resolve 校验，规则与 runs/ 一致（防路径穿越与符号链接）。"""
    if not WORD_ID_RE.match(wid or ""):
        raise StoreError("非法唤醒词", 404)
    base = words_root()
    path = (base / wid).resolve()
    root = settings.WORDS_ROOT.resolve()
    if root not in path.parents:
        raise StoreError("非法路径", 400)
    if path.parent != root:
        raise StoreError("非法唤醒词", 404)
    if path.is_symlink():
        raise StoreError("非法链接", 400)
    return path


# ---------------------------------------------------------------- 词的增删查
def counts(wid: str) -> dict:
    clips = list_clips(wid)
    mic = sum(1 for c in clips if c["source"] == "mic")
    synth = len(clips) - mic
    return {"pos": len(clips), "mic": mic, "synth": synth,
            "ready": mic >= settings.MIN_WORD_POS,
            "min_pos": settings.MIN_WORD_POS}


def describe(wid: str) -> dict:
    record = store._read_json(word_dir(wid) / "word.json")
    if not record:
        raise StoreError("没有这个唤醒词", 404)
    return {**record, "counts": counts(wid)}


def list_words() -> list[dict]:
    base = words_root()
    rows = []
    for path in sorted(base.iterdir()):
        if not path.is_dir() or path.is_symlink() or not WORD_ID_RE.match(path.name):
            continue
        try:
            rows.append(describe(path.name))
        except StoreError:
            continue                      # 半个目录（建到一半）不该拖垮列表
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows


def get_word(wid: str) -> dict:
    return describe(wid)


# ---------------------------------------------------------------- 当前词
def _current_path() -> Path:
    return words_root() / CURRENT_FILE


def set_current(wid: str) -> None:
    """把 wid 记成"当前词"。只接受合法 id，且目录必须真的存在。"""
    if not WORD_ID_RE.match(wid or ""):
        raise StoreError("非法唤醒词", 404)
    describe(wid)                             # 顺带校验 word.json 存在
    store._write_json(_current_path(), {"id": wid})


def current_word() -> dict | None:
    """当前展示的词；指针失效就读磁盘全部目录，取最新的一个并落盘。"""
    record = store._read_json(_current_path()) or {}
    wid = record.get("id")
    if wid:
        try:
            return describe(wid)
        except StoreError:
            pass                              # 指针指向的目录被删了：往下走重新挑
    rows = list_words()                       # 已按 created_at 倒序
    if not rows:
        return None
    set_current(rows[0]["id"])
    return rows[0]


def create_word(text: str) -> tuple[dict, bool]:
    """幂等：同一段文本重复提交返回同一个数据集（不丢已录的正例）。

    成功后把这个词设为「当前词」：界面上永远只展示一个词，换词就是再调一次这里。
    """
    name = normalize_text(text)
    wid = word_id(name)
    path = word_dir(wid)
    record = store._read_json(path / "word.json")
    if record:
        set_current(wid)
        return {**record, "counts": counts(wid)}, False
    (path / "pos").mkdir(parents=True, exist_ok=True)
    record = {"id": wid, "text": name,
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    store._write_json(path / "word.json", record)
    set_current(wid)
    return {**record, "counts": counts(wid)}, True


# ---------------------------------------------------------------- 正例
def pos_dir(wid: str) -> Path:
    path = word_dir(wid) / "pos"
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_clip(wid: str, data: bytes, *, tag: str = "") -> str:
    word_dir(wid)                            # 顺带校验 wid
    folder = pos_dir(wid)
    if sum(1 for p in folder.glob("*.wav")) >= settings.WORD_POS_MAX:
        raise StoreError(
            f"这个词最多存 {settings.WORD_POS_MAX} 条正例，先删几条再录", 400)
    if tag and not store.TAG_RE.match(tag):
        raise StoreError("非法标签", 400)
    return store._write_clip(folder, "pos-" + (f"{tag}-" if tag else ""), data)


def clip_path(wid: str, name: str) -> Path:
    folder = pos_dir(wid)
    path = store._safe_child(folder, name, store.NAME_RE, root=settings.WORDS_ROOT)
    if not path.is_file():
        raise StoreError("文件不存在", 404)
    return path


def list_clips(wid: str) -> list[dict]:
    from .audio import wav_info

    folder = pos_dir(wid)
    items = []
    for path in sorted(folder.glob("*.wav")):
        name = path.name
        if not store.NAME_RE.match(name):
            continue
        try:
            info = wav_info(path)
        except Exception:                       # 坏文件不该拖垮整个列表
            info = {"duration": 0, "sample_rate": 0}
        stat = path.stat()
        items.append({"name": name, "size": stat.st_size,
                      "source": "synth" if "-synth-" in name else "mic",
                      "mine": True,
                      "created_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                  time.localtime(stat.st_mtime)),
                      **info})
    items.sort(key=lambda it: it["created_at"], reverse=True)
    return items


def delete_clip(wid: str, name: str) -> None:
    clip_path(wid, name).unlink()


def require_ready(wid: str) -> dict:
    """开训门槛：真人正例 ≥ MIN_WORD_POS。合成语料再多样也替代不了真人喊一遍。"""
    record = describe(wid)
    c = record["counts"]
    if not c["ready"]:
        raise StoreError(
            f"「{record['text']}」还差 {c['min_pos'] - c['mic']} 条真人录音"
            f"（当前 {c['mic']}/{c['min_pos']}，合成语料 {c['synth']} 条不算）", 409)
    return record
