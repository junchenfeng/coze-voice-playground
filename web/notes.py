"""研究笔记：`/notes` 页与 `/note-log/{1..4}` 只读接口。

内容源是 `data/research-notes.json`（本轮先留空：4 条 `published:false` 的空壳）。
渲染规则照抄 machine-learning-interactive-demo：

- 正文只取 `## 摘要` 到 `## 附录` 之间的部分；「给 Agent 的指令」「附录」永不展示
  （渲染前兜底切掉，哪怕作者忘了删）；
- 第四章（`## 四、` 起）套「手写体便签块」样式，是给孩子抄的笔记。

`note-logs/note-{x}.json` 是 AI 学习伙伴写的原始日志，本站只读、不做二次加工；
接口契约见 `note-logs/README.md`。
"""
import json
import re
from pathlib import Path

from . import settings
from .store import StoreError

NOTE_IDS = ("1", "2", "3", "4")
NOTE_LOG_RE = re.compile(r"^note-[1-4]\.json$")
_FENCE_RE = re.compile(r"^```")
_AGENT_HEADING = re.compile(r"^##+\s*给\s*Agent", re.IGNORECASE)


def _path() -> Path:
    return Path(settings.RESEARCH_NOTES)


def load() -> list[dict]:
    """读笔记 JSON 并做最小结构校验；缺文件按「四条空壳」处理。"""
    try:
        with open(_path(), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            note_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        rows.append({
            "id": note_id,
            "title": str(item.get("title") or f"笔记 {note_id}"),
            "subtitle": str(item.get("subtitle") or ""),
            "level": str(item.get("level") or ""),
            "source": str(item.get("source") or ""),
            "published": bool(item.get("published")),
            "markdown": str(item.get("markdown") or ""),
        })
    rows.sort(key=lambda r: r["id"])
    return rows


def extract_body(markdown: str) -> str:
    """只留 `## 摘要` ~ `## 附录` 之间的正文，并剥掉「给 Agent 的指令」整节。"""
    text = (markdown or "").replace("\r\n", "\n")
    start = re.search(r"^##\s*摘要\s*$", text, re.MULTILINE)
    if start:
        text = text[start.end():]
    end = re.search(r"^##\s*附录\s*$", text, re.MULTILINE)
    if end:
        text = text[:end.start()]

    kept: list[str] = []
    skipping = False
    in_fence = False
    for line in text.split("\n"):
        if _FENCE_RE.match(line.strip()):
            in_fence = not in_fence
        if not in_fence and _AGENT_HEADING.match(line.strip()):
            skipping = True
            continue
        if skipping:
            # 遇到下一个同级或更高级标题才恢复
            if re.match(r"^#{1,2}\s", line.strip()):
                skipping = False
            else:
                continue
        kept.append(line)
    return "\n".join(kept).strip()


def render(markdown: str) -> str:
    """Markdown → HTML（服务端渲染，前端不引 marked）。"""
    body = extract_body(markdown)
    if not body:
        return ""
    try:
        import markdown as md
    except ImportError:                     # 没装 markdown 时退化成纯文本，别 500
        return f'<pre class="note-plain">{_escape(body)}</pre>'
    return md.markdown(body, extensions=["tables", "fenced_code", "sane_lists"])


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def views() -> list[dict]:
    """页面直接用：未发布的正文一律不渲染（前端只显示空态）。"""
    out = []
    for note in load():
        body = render(note["markdown"]) if note["published"] else ""
        out.append({**note, "html": body})
    return out


def note_log_path(x: str) -> Path:
    """`/note-log/{x}` 的白名单 + 目录锁定（只认 1~4，且必须在 note-logs/ 下）。"""
    if x not in NOTE_IDS:
        raise StoreError("笔记编号只能是 1 / 2 / 3 / 4", 400)
    base = Path(settings.NOTE_LOGS_DIR).resolve()
    path = (base / f"note-{x}.json").resolve()
    if base not in path.parents and path.parent != base:
        raise StoreError("非法路径", 400)
    return path


def note_log_text(x: str) -> str:
    """原样返回日志文件内容；没生成过 → 404，内容不是合法 JSON → 500。"""
    path = note_log_path(x)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise StoreError("这份日志还没生成", 404) from exc
    except OSError as exc:
        raise StoreError(f"日志读不出来：{exc}", 500) from exc
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise StoreError("日志不是合法 JSON", 500) from exc
    return text
