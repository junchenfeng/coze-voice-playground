"""研究笔记：正文裁剪、markdown 渲染、note-log 的三种错误码。"""
import json

import pytest

from web import notes, settings
from web.store import StoreError

MD = """# 标题

## 摘要

这一节会出现在页面上。

## 一、

正文 A。

## 四、

照着抄：**简单 CNN 只有一层卷积**。

## 给 Agent 的指令

不要展示这段。

## 附录

也不要展示这段。
"""


def test_extract_body_strips_agent_and_appendix():
    body = notes.extract_body(MD)
    assert "这一节会出现在页面上" in body
    assert "简单 CNN 只有一层卷积" in body
    assert "不要展示这段" not in body
    assert "也不要展示这段" not in body
    assert "## 摘要" not in body


def test_extract_body_without_headings_is_noop():
    assert notes.extract_body("就是一段话") == "就是一段话"
    assert notes.extract_body("") == ""


def test_render_markdown():
    html = notes.render(MD)
    assert "<h2>" in html
    assert "<strong>简单 CNN 只有一层卷积</strong>" in html
    assert "给 Agent" not in html


def test_render_empty_is_empty():
    assert notes.render("") == ""
    assert notes.render("## 给 Agent 的指令\n\n只有指令") == ""


def write_notes(rows):
    settings.RESEARCH_NOTES.parent.mkdir(parents=True, exist_ok=True)
    settings.RESEARCH_NOTES.write_text(json.dumps(rows, ensure_ascii=False),
                                       encoding="utf-8")


def test_load_filters_bad_rows():
    write_notes([
        {"id": 1, "title": "A", "published": True, "markdown": MD},
        {"id": "x", "title": "坏 id"},
        "不是对象",
        {"id": 2},
    ])
    rows = notes.load()
    assert [r["id"] for r in rows] == [1, 2]
    assert rows[1]["title"] == "笔记 2"
    assert rows[0]["published"] is True


def test_load_missing_file_is_empty():
    settings.RESEARCH_NOTES.unlink(missing_ok=True)
    assert notes.load() == []
    assert notes.views() == []


def test_views_only_render_published():
    write_notes([
        {"id": 1, "title": "已发布", "published": True, "markdown": MD},
        {"id": 2, "title": "草稿", "published": False, "markdown": MD},
    ])
    a, b = notes.views()
    assert "<h2>" in a["html"]
    assert b["html"] == ""              # 未发布：正文一个字都不给前端


def test_note_log_whitelist():
    for bad in ("5", "0", "1abc", "../note-1"):
        with pytest.raises(StoreError) as exc:
            notes.note_log_path(bad)
        assert exc.value.status == 400


def test_note_log_status_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "NOTE_LOGS_DIR", tmp_path)
    with pytest.raises(StoreError) as exc:
        notes.note_log_text("1")
    assert exc.value.status == 404

    (tmp_path / "note-2.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(StoreError) as exc:
        notes.note_log_text("2")
    assert exc.value.status == 500

    payload = '{"id": 3, "entries": []}'
    (tmp_path / "note-3.json").write_text(payload, encoding="utf-8")
    assert notes.note_log_text("3") == payload      # 原样返回，不做二次加工


def test_notes_route_renders_published(client):
    html = client.get("/notes").text
    assert "笔记 1" in html and "笔记 4" not in html    # 只有已发布的一篇
    assert "这份笔记还没发布" not in html
    assert "note-sheet" in html                         # 正文容器已渲染
