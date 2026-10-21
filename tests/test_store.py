"""存储层：路径围栏、运行的增删查、active 指针一致性。"""
import json
import os

import pytest

from web import settings, store
from web.store import StoreError


def test_safe_child_rejects_escape_and_symlink(tmp_path):
    base = tmp_path / "root"
    (base / "sub").mkdir(parents=True)
    os.symlink("/etc", base / "evil")
    for bad in ("../x", "a/b", "..", "", "evil"):
        with pytest.raises(StoreError):
            store._safe_child(base, bad, store.NAME_RE, root=base)
    assert store._safe_child(base, "ok.wav", store.NAME_RE, root=base).name == "ok.wav"


def test_safe_child_rejects_base_outside_root(tmp_path):
    with pytest.raises(StoreError):
        store._safe_child(tmp_path, "x.wav", store.NAME_RE, root=settings.DATA_ROOT)


def test_atomic_write_leaves_no_temp_files(tmp_path):
    path = tmp_path / "a.json"
    store._write_json(path, {"a": 1})
    store._write_json(path, {"a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_run_lifecycle():
    run_id = store.create_run({"model": "cnn", "epochs": 1})
    assert store.read_config(run_id)["model"] == "cnn"
    assert store.read_status(run_id)["state"] == "queued"

    store.write_status(run_id, {"state": "succeeded"})
    assert store.read_status(run_id)["state"] == "succeeded"

    store.set_active(run_id)
    assert store.get_active() == run_id
    rows = store.list_runs()
    assert rows[0]["id"] == run_id and rows[0]["active"] is True

    store.delete_run(run_id)
    assert store.list_runs() == []
    assert store.get_active() is None            # 删在用的模型：指针一起清


def test_run_id_whitelist():
    for bad in ("../x", "2026", "20260101-000000-ZZZZ"):
        with pytest.raises(StoreError):
            store.run_dir(bad)


def test_missing_run_is_404():
    with pytest.raises(StoreError) as exc:
        store.read_config("20260101-000000-dead")
    assert exc.value.status == 404


def test_history_skips_bad_lines():
    run_id = store.create_run({"model": "cnn"})
    path = store.run_dir(run_id) / "progress.jsonl"
    path.write_text('{"epoch":1}\nnot json\n\n{"epoch":2}\n', encoding="utf-8")
    assert [r["epoch"] for r in store.read_history(run_id)] == [1, 2]


def test_delete_rejects_half_made_dir(tmp_path):
    path = store.runs_dir() / "20260101-000000-aaaa"
    path.mkdir()
    with pytest.raises(StoreError) as exc:
        store.delete_run("20260101-000000-aaaa")
    assert exc.value.status == 404


def test_corpus_summary_shape():
    body = store.corpus_summary()
    assert body["counts"] == {"pos": 0, "neg": 0}
    assert body["ready"] is False
    assert body["builtin"]["available"] is False
    assert body["min_pos"] == settings.MIN_POS


def test_list_clips_sorted_and_tagged(wav_bytes):
    store.add_clip("pos", wav_bytes())
    store.add_clip("pos", wav_bytes(), tag="synth")
    clips = store.list_clips("pos")
    assert len(clips) == 2
    assert {c["source"] for c in clips} == {"mic", "synth"}
    assert all(c["mine"] is True for c in clips)


def test_synth_clip_lives_in_cache_not_data(wav_bytes):
    """合成音频是缓存：落在 .cache，不能污染 data/。"""
    name = store._write_clip(store.synth_dir("pos"), "pos-synth-", wav_bytes())
    assert (settings.SYNTH_CACHE_ROOT / "pos" / name).is_file()
    assert not (settings.CORPUS_ROOT / "pos" / name).exists()
