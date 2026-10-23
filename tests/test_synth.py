"""合成语料（豆包 TTS）的接口行为：不真连 TTS，只校验口径与落盘。"""
import pytest

from web import settings, synth


def fake_pcm(seconds=1.0):
    def _tts(text, speaker, resource_id=None, speech_rate=0, **_kw):
        return b"\x00\x00" * int(16000 * seconds)
    return _tts


def patch_tts(monkeypatch, seconds=1.0):
    monkeypatch.setattr(synth.tts_mod, "APP_ID", "test-app", raising=False)
    monkeypatch.setattr(synth.tts_mod, "TOKEN", "test-token", raising=False)
    monkeypatch.setattr(synth.tts_mod, "tts_pcm", fake_pcm(seconds))


ZH_VOICE = synth.ZH_VOICES[0][0]
EN_VOICE = synth.EN_VOICES[0][0]


def test_voices_needs_no_login(client):
    """单数据域：没有登录这一步，音色表直接可读。"""
    res = client.get("/api/voices")
    assert res.status_code == 200
    assert "voices" in res.json()


def test_voices_lists_both_languages(client, monkeypatch):
    monkeypatch.setattr(synth.tts_mod, "APP_ID", "", raising=False)
    monkeypatch.setattr(synth.tts_mod, "TOKEN", "", raising=False)
    body = client.get("/api/voices").json()
    assert body["configured"] is False
    assert body["max_pos"] >= 1 and body["max_neg"] >= 1
    assert {v["lang"] for v in body["voices"]} == {"zh", "en"}
    assert any(v["can_pos"] is False for v in body["voices"])
    assert any(v["can_pos"] is True for v in body["voices"])


def test_synth_needs_credentials(client, monkeypatch):
    monkeypatch.setattr(synth.tts_mod, "APP_ID", "", raising=False)
    monkeypatch.setattr(synth.tts_mod, "TOKEN", "", raising=False)
    res = client.post("/api/corpus/synth",
                           json={"voice": ZH_VOICE, "pos": 2, "neg": 0})
    assert res.status_code == 503
    assert "TTS" in res.json()["detail"]


def test_synth_creates_synthetic_clips(client, monkeypatch):
    patch_tts(monkeypatch)
    res = client.post("/api/corpus/synth",
                           json={"voice": ZH_VOICE, "pos": 2, "neg": 3})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] == {"pos": 2, "neg": 3}
    assert body["counts"] == {"pos": 2, "neg": 3}
    assert body["failed"] == 0
    assert body["voice"] == ZH_VOICE

    clips = body["pos"] + body["neg"]
    assert all(c["source"] == "synth" for c in clips)
    assert all("-synth-" in c["name"] for c in clips)

    played = client.get(f"/api/corpus/pos/{body['pos'][0]['name']}")
    assert played.status_code == 200
    assert played.headers["content-type"].startswith("audio/wav")


def test_synth_english_voice_cannot_make_positives(client, monkeypatch):
    patch_tts(monkeypatch)
    res = client.post("/api/corpus/synth",
                           json={"voice": EN_VOICE, "pos": 1, "neg": 0})
    assert res.status_code == 400
    assert "英文音色" in res.json()["detail"]

    # 只造负样本可以
    ok = client.post("/api/corpus/synth",
                          json={"voice": EN_VOICE, "pos": 0, "neg": 2})
    assert ok.status_code == 200
    assert ok.json()["created"] == {"pos": 0, "neg": 2}


def test_synth_rejects_unknown_voice_and_empty_request(client, monkeypatch):
    patch_tts(monkeypatch)
    assert client.post("/api/corpus/synth",
                            json={"voice": "no-such-voice", "pos": 1}).status_code == 400
    assert client.post("/api/corpus/synth",
                            json={"voice": ZH_VOICE, "pos": 0, "neg": 0}).status_code == 400
    assert client.post("/api/corpus/synth",
                            json={"voice": ZH_VOICE, "pos": -1}).status_code == 400


def test_synth_clamps_counts(client, monkeypatch):
    patch_tts(monkeypatch)
    monkeypatch.setattr(settings, "SYNTH_MAX_POS", 2)
    monkeypatch.setattr(settings, "SYNTH_MAX_NEG", 2)
    body = client.post("/api/corpus/synth",
                            json={"voice": ZH_VOICE, "pos": 99, "neg": 99}).json()
    assert body["created"] == {"pos": 2, "neg": 2}
    assert body["requested"] == {"pos": 2, "neg": 2}


def test_synth_drops_clips_out_of_duration_range(client, monkeypatch):
    patch_tts(monkeypatch, seconds=6.0)          # 服务端上限 5 s
    res = client.post("/api/corpus/synth",
                           json={"voice": ZH_VOICE, "pos": 1, "neg": 0})
    assert res.status_code == 502
    assert client.get("/api/corpus").json()["counts"] == {"pos": 0, "neg": 0}


def test_synth_reports_tts_failure(client, monkeypatch):
    monkeypatch.setattr(synth.tts_mod, "APP_ID", "test-app", raising=False)
    monkeypatch.setattr(synth.tts_mod, "TOKEN", "test-token", raising=False)

    def boom(*_a, **_kw):
        raise RuntimeError("TTS 失败: 鉴权不对")

    monkeypatch.setattr(synth.tts_mod, "tts_pcm", boom)
    res = client.post("/api/corpus/synth",
                           json={"voice": ZH_VOICE, "pos": 1, "neg": 0})
    assert res.status_code == 502
    assert "鉴权" in res.json()["detail"]


@pytest.mark.parametrize("tag,ok", [("synth", True), ("mic", True),
                                    ("SYNTH", False), ("a-b", False)])
def test_add_clip_tag_whitelist(tag, ok):
    from web import store

    if ok:
        name = store.add_clip("pos", b"RIFF", tag=tag)
        assert tag in name
    else:
        with pytest.raises(store.StoreError):
            store.add_clip("pos", b"RIFF", tag=tag)
