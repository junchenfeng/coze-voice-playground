"""推理冒烟：随机权重存盘再加载，只验证口径与形状，不依赖真实训练。"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trainer import backbone  # noqa: E402
from trainer.infer import load_model, predict_pcm  # noqa: E402
from trainer.models import CNN, MLP, OWWHead  # noqa: E402


def _model(name):
    if name == "mlp":
        return MLP()
    if name == "cnn":
        return CNN()
    return OWWHead()


@pytest.mark.parametrize("name", ["mlp", "cnn", "head"])
def test_load_and_predict(tmp_path, name):
    pytest.importorskip("onnxruntime")
    if name == "head" and not backbone.available():
        pytest.skip("没装 openWakeWord 底座")
    model = _model(name)
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)

    loaded = load_model(str(ckpt), name)
    pcm = np.zeros(16000, dtype=np.float32)
    prob = predict_pcm(loaded, pcm, name)
    assert 0.0 <= prob <= 1.0


def test_predict_accepts_short_clip(tmp_path):
    model = CNN()
    ckpt = tmp_path / "cnn.pt"
    torch.save(model.state_dict(), ckpt)
    loaded = load_model(str(ckpt), "cnn")
    assert 0.0 <= predict_pcm(loaded, np.zeros(3200, dtype=np.float32), "cnn") <= 1.0


def test_incompatible_ckpt_has_clear_error(tmp_path):
    ckpt = tmp_path / "old.pt"
    torch.save(MLP().state_dict(), ckpt)          # 旧结构存成 cnn 的权重
    with pytest.raises(RuntimeError, match="重新训练"):
        load_model(str(ckpt), "cnn")
