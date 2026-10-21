"""自定义唤醒词：目录口径、幂等、当前词指针、开训门槛。"""
import json

import pytest

from web import settings, store, synth, words
from web.store import StoreError


def test_normalize_text():
    assert words.normalize_text("  小福小福 ") == "小福小福"
    assert words.normalize_text("hi   buddy") == "hi buddy"
    assert words.normalize_text("ＡＢＣ") == "ABC"
    for bad in ("", "   ", "a" * (settings.WORD_TEXT_MAX + 1), "a/b", "a\\b", "换\n行"):
        with pytest.raises(StoreError):
            words.normalize_text(bad)


def test_word_id_depends_only_on_text():
    assert words.word_id("小福小福") == words.word_id(" 小福小福 ")
    assert words.word_id("小福小福") != words.word_id("小福小福2")
    assert words.word_id("小福小福").startswith("w-")


def test_create_word_is_idempotent(wav_bytes):
    record, created = words.create_word("小福小福")
    assert created is True
    assert (settings.WORDS_ROOT / record["id"] / "word.json").is_file()

    words.add_clip(record["id"], wav_bytes())
    again, created2 = words.create_word("小福小福")
    assert created2 is False
    assert again["counts"]["pos"] == 1          # 已录的正例没被清掉


def test_counts_split_mic_and_synth(wav_bytes):
    record, _ = words.create_word("hey nova")
    wid = record["id"]
    words.add_clip(wid, wav_bytes())
    words.add_clip(wid, wav_bytes(), tag="synth")
    counts = words.counts(wid)
    assert counts["pos"] == 2
    assert counts["mic"] == 1 and counts["synth"] == 1
    assert counts["ready"] is False              # 真人还不够


def test_require_ready_gate(wav_bytes):
    record, _ = words.create_word("小福小福")
    wid = record["id"]
    for _ in range(settings.MIN_WORD_POS):
        words.add_clip(wid, wav_bytes())
    assert words.require_ready(wid)["counts"]["ready"] is True

    other, _ = words.create_word("另一个词")
    with pytest.raises(StoreError) as exc:
        words.require_ready(other["id"])
    assert "真人录音" in str(exc.value)


def test_current_word_falls_back_when_pointer_stale(wav_bytes):
    first, _ = words.create_word("第一个")
    second, _ = words.create_word("第二个")
    assert words.current_word()["id"] == second["id"]

    # 指针指向的词目录被整个删掉：下次读就重新挑一个最新的，不 500
    import shutil

    shutil.rmtree(settings.WORDS_ROOT / second["id"])
    got = words.current_word()
    assert got["id"] == first["id"]


def test_current_word_none_when_empty():
    assert words.current_word() is None


def test_clip_path_rejects_traversal(wav_bytes):
    record, _ = words.create_word("hey nova")
    for bad in ("../word.json", "a/b.wav", "x" * 300):
        with pytest.raises(StoreError):
            words.clip_path(record["id"], bad)


def test_word_dir_rejects_illegal_id():
    for bad in ("w-zzzzzzzz", "../w-12345678", "w-123", ""):
        with pytest.raises(StoreError):
            words.word_dir(bad)


def test_delete_clip_and_list(wav_bytes):
    record, _ = words.create_word("hey nova")
    name = words.add_clip(record["id"], wav_bytes())
    assert [c["name"] for c in words.list_clips(record["id"])] == [name]
    words.delete_clip(record["id"], name)
    assert words.list_clips(record["id"]) == []


def test_word_json_survives_list_words(wav_bytes):
    record, _ = words.create_word("小福小福")
    rows = words.list_words()
    assert [r["id"] for r in rows] == [record["id"]]
    assert rows[0]["counts"]["pos"] == 0

    # 半个目录（没有 word.json）不该拖垮列表
    orphan = settings.WORDS_ROOT / "w-deadbeef"
    orphan.mkdir(parents=True)
    assert [r["id"] for r in words.list_words()] == [record["id"]]


def test_auto_word_tasks_cap_and_voices():
    tasks = synth.auto_word_tasks("小福小福")
    assert len(tasks) == min(len(synth.ZH_VOICES) * 2, settings.WORD_SYNTH_MAX)
    assert all(t[0] == "pos" for t in tasks)

    # 英文单词多出英文音色
    more = synth.auto_word_tasks("hey nova")
    assert len(more) > len(tasks) or settings.WORD_SYNTH_MAX <= len(tasks)


def test_word_synth_status_idle():
    assert synth.word_synth_status("w-12345678")["state"] == "idle"
