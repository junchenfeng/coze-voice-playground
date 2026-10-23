"""推理：加载 ckpt → 特征 → 唤醒词概率。

web 层的 predict 与离线自测共用这一份，保证和训练口径完全一致：
- mlp / simple / cnn 吃 40×100 对数梅尔谱（1 s 窗口）
- head 吃 openWakeWord 底座的 192 维池化向量
"""
import numpy as np
import torch

from .backbone import get_features
from .features import logmel
from .models import CNN, MLP, OWWHead, SimpleCNN
from .train import load_state


def load_model(ckpt, model_name):
    if model_name == "mlp":
        model = MLP()
    elif model_name == "simple":
        model = SimpleCNN()
    elif model_name == "cnn":
        model = CNN()
    elif model_name == "head":
        model = OWWHead()
    else:
        raise ValueError(f"未知模型: {model_name}")
    try:
        model.load_state_dict(load_state(ckpt))
    except RuntimeError as exc:
        raise RuntimeError(
            "权重与当前模型结构不兼容（多半是旧版本训练出来的），请重新训练一次") from exc
    model.eval()
    return model


@torch.no_grad()
def predict_pcm(model, pcm, model_name="cnn"):
    """float32 单声道 16k 波形 → 正类（唤醒词）概率。"""
    pcm = np.asarray(pcm, dtype=np.float32)
    if model_name == "head":
        x = torch.from_numpy(get_features().embed(pcm)).unsqueeze(0)      # (1,192)
    else:
        x = torch.from_numpy(logmel(pcm)).unsqueeze(0).unsqueeze(0)       # (1,1,40,100)
    prob = torch.softmax(model(x), dim=1)[0]
    return float(prob[1])
