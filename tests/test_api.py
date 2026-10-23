"""页面与 JSON API 的行为。

单数据域后没有登录/工作区：`client` 一进来就能读写自己这台机器的语料。
训练子进程一律 stub（`jobs.start`），只验证"请求 → 配置 → 目录"这条链。
"""
import json

import pytest

from web import jobs, predictor, settings, store, words


def seed_corpus(client, wav_bytes, label, n=4):
    for _ in range(n):
        res = client.post(f"/api/corpus/{label}",
                          files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        assert res.status_code == 200, res.text


def make_run(*, model="cnn", status="succeeded", with_ckpt=True,
             wake=None, result=None) -> str:
    run_id = store.create_run({
        "model": model, "epochs": 2, "batch": 4, "lr": 0.001, "val_ratio": 0.2,
        "seed": 1, "max_pos": 30, "max_neg": 100, "use_builtin": True,
        "wake_word": wake, "created_at": "2026-01-01T00:00:00",
    })
    path = store.run_dir(run_id)
    if with_ckpt:
        (path / "model.pt").write_bytes(b"not-a-real-ckpt")
    rows = [{"epoch": 1, "train_loss": 0.9, "train_acc": 0.5,
             "val_loss": 0.8, "val_acc": 0.6},
            {"epoch": 2, "train_loss": 0.4, "train_acc": 1.0,
             "val_loss": 0.3, "val_acc": 1.0}]
    (path / "progress.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    if result is None:
        result = {"final": {"train_acc": 1.0, "val_acc": 1.0, "gap": 0.0},
                  "elapsed_s": 1.5, "params_total": 1234, "params_trainable": 1234}
    store._write_json(path / "result.json", result)
    store.write_status(run_id, {"state": status, "final": {"val_acc": 1.0},
                                "elapsed_s": 1.5})
    return run_id


def _boot_step(html: str) -> str:
    import re

    found = re.search(r'data-step="([a-z]+)"', html)
    assert found, "页面缺少 #boot 的 data-step"
    return found.group(1)


# ---------------------------------------------------------------- 页面
def test_root_redirects_to_studio(client):
    res = client.get("/", follow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/studio"


def test_studio_renders_three_model_cards(client):
    html = client.get("/studio").text
    assert "Nova Voice Lab" in html
    # 三张卡：简单 CNN / 复杂 CNN / 预训练底座；mlp 已下线
    for value in ('value="simple"', 'value="cnn"', 'value="head"'):
        assert value in html
    assert 'value="mlp"' not in html
    assert "研究笔记" in html              # 顶栏入口
    assert 'data-mode="guide"' not in html  # 没有引导态了


def test_studio_step_param_and_url(client):
    assert _boot_step(client.get("/studio?step=word").text) == "word"
    assert _boot_step(client.get("/studio?tab=test").text) == "check"
    assert _boot_step(client.get("/studio?step=nope").text) == "train"


def test_notes_page_published_note(client):
    html = client.get("/notes").text
    assert "研究笔记" in html
    assert "这份笔记还没发布" not in html      # 唯一一篇已发布：无空态
    assert "三个模型，同一场考试" in html      # 正文已渲染


def test_note_log_contract(client):
    assert client.get("/note-log/1").status_code == 404
    assert client.get("/note-log/5").status_code == 400
    assert client.get("/note-log/1abc").status_code == 400


def test_api_me_has_no_active_model(client):
    body = client.get("/api/me").json()
    assert body["active"] is None
    assert body["onboarded"] is True       # 保留字段，但已没有引导流程
    assert body["seed"] == {} or isinstance(body["seed"], dict)


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["ok"] is True
    assert body["running"] == 0


# ---------------------------------------------------------------- 语料
def test_corpus_upload_list_play_delete(client, wav_bytes):
    seed_corpus(client, wav_bytes, "pos", n=2)
    body = client.get("/api/corpus").json()
    assert body["counts"] == {"pos": 2, "neg": 0}
    assert body["ready"] is False
    name = body["pos"][0]["name"]
    assert name.startswith("pos-") and name.endswith(".wav")
    assert "@" not in name                  # 单数据域：不再带谁的昵称前缀
    assert body["pos"][0]["mine"] is True

    played = client.get(f"/api/corpus/pos/{name}")
    assert played.status_code == 200
    assert played.headers["content-type"].startswith("audio/wav")

    assert client.delete(f"/api/corpus/pos/{name}").json()["counts"]["pos"] == 1


def test_corpus_rejects_silence_and_bad_wav(client, wav_bytes):
    quiet = wav_bytes(amp=0.0005)
    res = client.post("/api/corpus/pos", files={"file": ("a.wav", quiet, "audio/wav")})
    assert res.status_code == 400
    assert "安静" in res.json()["detail"] or "声音" in res.json()["detail"]

    res = client.post("/api/corpus/pos", files={"file": ("a.wav", b"not a wav", "audio/wav")})
    assert res.status_code == 400
    assert client.get("/api/corpus").json()["counts"] == {"pos": 0, "neg": 0}


def test_corpus_rejects_out_of_range_duration(client, wav_bytes):
    res = client.post("/api/corpus/pos",
                      files={"file": ("a.wav", wav_bytes(seconds=0.1), "audio/wav")})
    assert res.status_code == 400


@pytest.mark.parametrize("name", ["a%2Fb.wav", ".hidden.wav", "x" * 300])
def test_clip_name_is_whitelisted(client, name):
    assert client.get(f"/api/corpus/pos/{name}").status_code in (400, 404)


def test_unknown_label_rejected(client, wav_bytes):
    res = client.post("/api/corpus/sideways",
                      files={"file": ("a.wav", wav_bytes(), "audio/wav")})
    assert res.status_code == 400


# ---------------------------------------------------------------- 自定义唤醒词
def test_word_create_is_idempotent_and_current(client, wav_bytes):
    first = client.post("/api/words", json={"text": " 小福小福 "}).json()
    assert first["text"] == "小福小福"
    assert first["created"] is True
    second = client.post("/api/words", json={"text": "小福小福"}).json()
    assert second["created"] is False and second["id"] == first["id"]

    # 录一条正例后再"新建"同一个词：目录不能清空
    client.post(f"/api/words/{first['id']}/clips",
                files={"file": ("a.wav", wav_bytes(), "audio/wav")})
    again = client.post("/api/words", json={"text": "小福小福"}).json()
    assert again["counts"]["pos"] == 1

    body = client.get("/api/words").json()
    assert body["word"]["id"] == first["id"]
    assert body["min_pos"] == settings.MIN_WORD_POS


def test_word_clip_crud_and_progress(client, wav_bytes):
    wid = client.post("/api/words", json={"text": "hey nova"}).json()["id"]
    res = client.post(f"/api/words/{wid}/clips",
                      files={"file": ("a.wav", wav_bytes(), "audio/wav")})
    assert res.status_code == 200, res.text
    name = res.json()["name"]
    assert res.json()["clips"][0]["source"] == "mic"
    assert client.get(f"/api/words/{wid}/clips/{name}").status_code == 200
    assert client.delete(f"/api/words/{wid}/clips/{name}").json()["counts"]["pos"] == 0
    # 没有后台任务时进度是 idle
    assert client.get(f"/api/words/{wid}/synth/progress").json()["state"] == "idle"


@pytest.mark.parametrize("text", ["", "   ", "a" * 40, "带/斜杠", "换\n行"])
def test_word_text_validation(client, text):
    assert client.post("/api/words", json={"text": text}).status_code == 400


def test_word_requires_min_mic_clips_before_training(client, wav_bytes, monkeypatch):
    seed_corpus(client, wav_bytes, "pos", n=4)
    seed_corpus(client, wav_bytes, "neg", n=4)
    wid = client.post("/api/words", json={"text": "小福小福"}).json()["id"]
    monkeypatch.setattr(jobs, "start", lambda config: pytest.fail("不该开训"))
    res = client.post("/api/runs", json={"model": "cnn", "wake_word_id": wid})
    assert res.status_code == 409
    assert "真人录音" in res.json()["detail"]


# ---------------------------------------------------------------- 训练
def test_run_validation(client, monkeypatch):
    monkeypatch.setattr(jobs, "start", lambda config: pytest.fail("不该开训"))
    for body in ({"model": "mlp"}, {"model": "cnn", "epochs": 0},
                 {"model": "cnn", "batch": 1}, {"model": "cnn", "lr": 9},
                 {"model": "cnn", "max_pos": 5}):
        res = client.post("/api/runs", json=body)
        assert res.status_code == 400, body


def test_run_requires_corpus(client, monkeypatch):
    monkeypatch.setattr(jobs, "start", lambda config: pytest.fail("不该开训"))
    res = client.post("/api/runs", json={"model": "cnn"})
    assert res.status_code == 400
    assert "语料不足" in res.json()["detail"]


def test_run_created_with_valid_config(client, wav_bytes, monkeypatch):
    seed_corpus(client, wav_bytes, "pos", n=4)
    seed_corpus(client, wav_bytes, "neg", n=4)
    captured = {}

    def fake_start(config):
        captured.update(config)
        return make_run(model=config["model"], status="running", with_ckpt=False)

    monkeypatch.setattr(jobs, "start", fake_start)
    res = client.post("/api/runs", json={"model": "simple", "epochs": 3,
                                         "use_builtin": False})
    assert res.status_code == 200, res.text
    assert captured["model"] == "simple"
    assert captured["split_by"] == "human"
    assert isinstance(captured["init_seed"], int)
    assert res.json()["id"]


def test_run_list_detail_and_stream(client, wav_bytes, monkeypatch):
    run_id = make_run()
    body = client.get("/api/runs").json()
    assert [r["id"] for r in body["runs"]] == [run_id]
    assert body["runs"][0]["has_ckpt"] is True

    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["result"]["final"]["val_acc"] == 1.0
    assert len(detail["history"]) == 2
    assert detail["has_ckpt"] is True

    # SSE：训练已结束 → 两条 epoch + 一条 status 后关流
    with client.stream("GET", f"/api/runs/{run_id}/stream") as res:
        assert res.status_code == 200
        text = "".join(res.iter_text())
    assert text.count("event: epoch") == 2
    assert "event: status" in text and '"state": "succeeded"' in text


def test_run_missing_is_404(client):
    assert client.get("/api/runs/20260101-000000-dead").status_code == 404


def test_apply_and_delete_flow(client, monkeypatch):
    run_id = make_run()
    assert client.post(f"/api/runs/{run_id}/apply").json()["active"] == run_id
    assert store.get_active() == run_id
    assert client.get("/api/me").json()["active"]["model"] == "cnn"

    monkeypatch.setattr(predictor, "forget", lambda _rid: None)
    body = client.delete(f"/api/runs/{run_id}").json()
    assert body["runs"] == [] and body["active"] is None
    assert store.get_active() is None


def test_apply_requires_succeeded(client):
    run_id = make_run(status="running", with_ckpt=False)
    res = client.post(f"/api/runs/{run_id}/apply")
    assert res.status_code == 409
    assert "训练成功" in res.json()["detail"]


def test_delete_rejects_running_job(client, monkeypatch):
    run_id = make_run(status="running")
    monkeypatch.setattr(jobs, "is_running", lambda _rid: True)
    assert client.delete(f"/api/runs/{run_id}").status_code == 409


def test_rerun_copies_config_but_new_init_seed(client, wav_bytes, monkeypatch):
    seed_corpus(client, wav_bytes, "pos", n=4)
    seed_corpus(client, wav_bytes, "neg", n=4)
    original = make_run()
    cfg_path = store.run_dir(original) / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.update({"init_seed": 7, "split_by": "human", "epochs": 5})
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    captured = {}

    def fake_start(config):
        captured.update(config)
        return make_run(model=config["model"], status="running", with_ckpt=False)

    monkeypatch.setattr(jobs, "start", fake_start)
    res = client.post(f"/api/runs/{original}/rerun", json={"model": "simple"})
    assert res.status_code == 200, res.text
    assert captured["model"] == "simple"      # 换模型卡
    assert captured["epochs"] == 5           # 其余照抄
    assert captured["split_by"] == "human"   # 考卷口径不变
    assert captured["init_seed"] != 7        # 换个起点，别训出逐位相同的模型


def test_rerun_rejects_unknown_model(client):
    run_id = make_run()
    assert client.post(f"/api/runs/{run_id}/rerun", json={"model": "x"}).status_code == 400


def test_stop_unknown_run(client):
    run_id = make_run()
    assert client.post(f"/api/runs/{run_id}/stop").status_code == 409


# ---------------------------------------------------------------- 推理
def test_predict_without_model(client, wav_bytes):
    res = client.post("/api/predict", files={"file": ("a.wav", wav_bytes(), "audio/wav")})
    assert res.status_code == 409
    assert "还没有应用" in res.json()["detail"]


def test_predict_returns_probability(client, wav_bytes, monkeypatch):
    run_id = make_run()
    store.set_active(run_id)
    monkeypatch.setattr(predictor, "predict",
                        lambda pcm: {"p_wake": 0.75, "run_id": run_id, "model": "cnn"})
    body = client.post("/api/predict",
                       files={"file": ("a.wav", wav_bytes(), "audio/wav")}).json()
    assert body["p_wake"] == 0.75
    assert body["duration"] == pytest.approx(1.0, abs=0.05)


def test_predict_rejects_tiny_clip(client, wav_bytes):
    run_id = make_run()
    store.set_active(run_id)
    res = client.post("/api/predict",
                      files={"file": ("a.wav", wav_bytes(seconds=0.05), "audio/wav")})
    assert res.status_code == 400


# ---------------------------------------------------------------- 存储边界
def test_safe_child_blocks_traversal():
    for bad in ("../x", "a/b", "..", ""):
        with pytest.raises(store.StoreError):
            store._safe_child(store.corpus_dir("pos"), bad, store.NAME_RE)
    assert store._safe_child(store.corpus_dir("pos"), "pos-1.wav", store.NAME_RE).name


def test_word_id_is_stable_and_nfkc():
    a = words.word_id("小福小福")
    assert a == words.word_id(" 小福小福 ") == words.word_id("小福小福".replace("小", "小"))
    assert a != words.word_id("小福小福2")
    assert words.word_id("ＡＢＣ") == words.word_id("ABC")     # 全角归一
