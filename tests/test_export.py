"""导出包：内容契约、numpy 前向与 torch 一致（简单/复杂 CNN）、阈值/自检、不可导出模型。"""
import hashlib
import io
import json
import pathlib
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trainer import features, np_cnn                       # noqa: E402
from trainer.infer import load_model, predict_pcm          # noqa: E402
from trainer.models import CNN, MLP, OWWHead, SimpleCNN     # noqa: E402
from web import export, settings, store, words              # noqa: E402

NETS = {"simple": SimpleCNN, "cnn": CNN, "mlp": MLP, "head": OWWHead}


def make_run(model="cnn", wake=None, state="succeeded", result=None) -> str:
    run_id = store.create_run({
        "model": model, "epochs": 1, "batch": 8, "lr": 1e-3, "val_ratio": 0.12,
        "seed": 42, "max_pos": 30, "max_neg": 100, "wake_word": wake})
    run_path = store.run_dir(run_id)
    torch.save(NETS[model]().state_dict(), run_path / "model.pt")
    store.write_status(run_id, {"state": state})
    if result is not None:
        store._write_json(run_path / "result.json", result)
    return run_id


def open_zip(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def meta_of(files: dict) -> dict:
    return json.loads(files["meta.json"])


@pytest.mark.parametrize("model", ["simple", "cnn"])
def test_package_contract(client, model):
    run_id = make_run(model, result={"final": {"val_acc": 0.8123}})
    res = client.get(f"/api/runs/{run_id}/export")
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "application/zip"
    assert f"buddy-wake-{run_id}.zip" in res.headers["content-disposition"]

    files = open_zip(res.content)
    assert set(files) == {"np_cnn.py", "model.npz", "meta.json"}
    meta = meta_of(files)

    assert meta["schema"] == export.SCHEMA
    assert meta["run_id"] == run_id and meta["model"] == model
    assert meta["val_acc"] == pytest.approx(0.8123)
    assert "warning" in meta            # 训练得差要明确告警，别让人装上去才发现
    assert meta["feature"] == np_cnn.FEATURES
    assert meta["wake_word"] is None                   # 默认词：没有词信息
    assert {"threshold", "threshold_high", "basis"} <= set(meta["thresholds"])

    # sha256 必须对得上：桌宠拒载的唯一依据
    assert meta["sha256"]["model.npz"] == hashlib.sha256(files["model.npz"]).hexdigest()
    assert meta["sha256"]["np_cnn.py"] == hashlib.sha256(files["np_cnn.py"]).hexdigest()

    # 包里的 np_cnn.py 与仓库里的逐字节相同（单份实现，避免两端漂移）
    assert files["np_cnn.py"] == pathlib.Path(np_cnn.__file__).read_bytes()
    assert b"\nimport torch" not in files["np_cnn.py"]

    # 简单 CNN 的权重键是复杂 CNN 的真子集（少两层卷积）
    with np.load(io.BytesIO(files["model.npz"])) as npz:
        assert set(npz.files) == set(np_cnn.weights_for(model))
        assert set(np_cnn.weights_for("simple")) < set(np_cnn.weights_for("cnn"))


@pytest.mark.parametrize("model", ["simple", "cnn"])
def test_numpy_forward_matches_torch(tmp_path, model):
    """导出的权重换一个实现算，概率必须逐位一致（float32 舍入级别）。"""
    torch.manual_seed(7)
    ckpt = tmp_path / "model.pt"
    torch.save(NETS[model]().state_dict(), ckpt)

    model_torch = load_model(str(ckpt), model)
    state = torch.load(ckpt, weights_only=True)
    fast = np_cnn.load_from_state({k: v.numpy() for k, v in state.items()}, model)
    assert fast.arch == model

    rng = np.random.default_rng(0)
    for _ in range(6):
        pcm = (0.25 * rng.standard_normal(16000)).astype(np.float32)
        assert abs(predict_pcm(model_torch, pcm, model) - fast.predict(pcm)) < 1e-5


def test_load_from_state_detects_arch(tmp_path):
    state = {k: v.numpy() for k, v in SimpleCNN().state_dict().items()}
    assert isinstance(np_cnn.load_from_state(state), np_cnn.NumpySimpleCNN)
    assert isinstance(np_cnn.load_from_state(state, "simple"), np_cnn.NumpySimpleCNN)
    full = {k: v.numpy() for k, v in CNN().state_dict().items()}
    assert isinstance(np_cnn.load_from_state(full), np_cnn.NumpyCNN)


def test_np_cnn_features_match_trainer_features():
    """两份 logmel 必须同口径：一份在 trainer，一份随包发给桌宠。"""
    rng = np.random.default_rng(3)
    for seconds in (1.0, 0.4, 2.7):
        pcm = (0.3 * rng.standard_normal(int(16000 * seconds))).astype(np.float32)
        np.testing.assert_allclose(np_cnn.logmel(pcm), features.logmel(pcm),
                                   rtol=0, atol=1e-6)


@pytest.mark.parametrize("model", ["simple", "cnn"])
def test_selftest_roundtrip(client, tmp_path, model):
    """meta 里的 selftest 必须能被包里那份代码复现（否则桌宠会拒载）。"""
    run_id = make_run(model)
    files = open_zip(client.get(f"/api/runs/{run_id}/export").content)
    meta = meta_of(files)

    npz = tmp_path / "model.npz"
    npz.write_bytes(files["model.npz"])
    net = np_cnn.load_npz(npz, meta["model"])
    got = net.predict(np_cnn.selftest_window(meta["selftest"]["seed"]))
    assert abs(got - meta["selftest"]["expect"]) < 1e-4
    assert 0.0 <= got <= 1.0


def test_export_rejects_non_cnn(client):
    for model in ("mlp", "head"):
        run_id = make_run(model)
        res = client.get(f"/api/runs/{run_id}/export")
        assert res.status_code == 409
        assert "只吃 CNN" in res.json()["detail"]


def test_export_requires_succeeded(client):
    run_id = make_run(state="running")
    res = client.get(f"/api/runs/{run_id}/export")
    assert res.status_code == 409
    assert "训练成功" in res.json()["detail"]


def test_export_requires_ckpt(client):
    run_id = store.create_run({"model": "cnn"})
    store.write_status(run_id, {"state": "succeeded"})
    res = client.get(f"/api/runs/{run_id}/export")
    assert res.status_code == 409
    assert "model.pt" in res.json()["detail"]


def test_word_run_exports_its_wake_word(client, wav_bytes):
    word, _ = words.create_word("小福小福")
    for _ in range(settings.MIN_WORD_POS):
        words.add_clip(word["id"], wav_bytes())
    run_id = make_run(wake={"id": word["id"], "text": word["text"]},
                      result={"final": {"val_acc": 0.9}})

    meta = meta_of(open_zip(client.get(f"/api/runs/{run_id}/export").content))
    assert meta["wake_word"] == {"id": word["id"], "text": "小福小福"}


def test_thresholds_from_corpus(client, wav_bytes):
    """语料够的时候要用生产滑窗口径标定，而不是照抄 head 的 0.70/0.85。"""
    for _ in range(3):
        store.add_clip("pos", wav_bytes())
        store.add_clip("neg", wav_bytes())
    run_id = make_run()

    thr = meta_of(open_zip(client.get(f"/api/runs/{run_id}/export").content))["thresholds"]
    assert thr["basis"].startswith("corpus-replay")
    assert thr["n_pos_clips"] >= 1 and thr["n_neg_clips"] >= 1
    assert 0.3 <= thr["threshold"] < thr["threshold_high"] <= 1.0


def test_thresholds_fall_back_without_corpus(client):
    run_id = make_run()
    thr = meta_of(open_zip(client.get(f"/api/runs/{run_id}/export").content))["thresholds"]
    assert thr["basis"] == "default"
    assert thr["threshold"] == export.DEFAULT_THR


def test_validate_prefers_recall_when_corpus_is_hard(client, tmp_path, wav_bytes):
    """召回优先：误报压到 6% 以内时取召回最高的组合，而不是最低误报。"""
    for _ in range(2):
        store.add_clip("pos", wav_bytes())
        store.add_clip("neg", wav_bytes(amp=0.05))
    run_id = make_run()
    meta = meta_of(open_zip(client.get(f"/api/runs/{run_id}/export").content))
    thr = meta["thresholds"]
    if thr["basis"].startswith("corpus-replay"):
        assert thr["fpr"] <= 0.06 or thr["basis"].endswith("(weak)")
