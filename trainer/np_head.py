"""Standalone numpy/ONNX inference for exported openWakeWord HEAD packages.

This file is shipped verbatim with the small HEAD weights. The two ONNX files
are supplied separately and verified by the desktop installer.
"""
from pathlib import Path

import numpy as np

WEIGHT_KEYS = ("net.1.weight", "net.1.bias", "net.4.weight", "net.4.bias")
FEATURES = {"type": "openwakeword-pooled", "sample_rate": 16000,
            "dimension": 192, "mel_window": 76, "mel_hop": 8}
BACKBONE = [
    {"name": "embedding_model.onnx", "url": "https://cdn.aibeaver.cn/classroom/voice-playground/models/backbone/openwakeword/embedding_model.onnx",
     "sha256": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f"},
    {"name": "melspectrogram.onnx", "url": "https://cdn.aibeaver.cn/classroom/voice-playground/models/backbone/openwakeword/melspectrogram.onnx",
     "sha256": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f"},
]


def selftest_window(seed):
    return (np.random.default_rng(seed).standard_normal(16000) * 0.1).astype(np.float32)


class NumpyHead:
    def __init__(self, state, backbone_dir):
        import onnxruntime as ort

        self.state = {k: np.asarray(state[k], dtype=np.float32) for k in WEIGHT_KEYS}
        expected = {"net.1.weight": (64, 192), "net.1.bias": (64,),
                    "net.4.weight": (2, 64), "net.4.bias": (2,)}
        for key, shape in expected.items():
            if self.state[key].shape != shape:
                raise ValueError(f"{key}: expected {shape}, got {self.state[key].shape}")
        root = Path(backbone_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.mel = ort.InferenceSession(str(root / "melspectrogram.onnx"),
                                        sess_options=opts, providers=["CPUExecutionProvider"])
        self.emb = ort.InferenceSession(str(root / "embedding_model.onnx"),
                                        sess_options=opts, providers=["CPUExecutionProvider"])

    def embed(self, pcm):
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if len(pcm) > 16000:
            start = (len(pcm) - 16000) // 2
            pcm = pcm[start:start + 16000]
        if len(pcm) < 12160:
            pcm = np.pad(pcm, (0, 12160 - len(pcm)))
        spec = np.squeeze(self.mel.run(None, {"input": pcm[None, :]})[0]) / 10.0 + 2.0
        wins = [spec[i:i + 76] for i in range(0, spec.shape[0], 8)
                if spec[i:i + 76].shape[0] == 76]
        if not wins:
            wins = [np.ones((76, 32), dtype=np.float32)]
        batch = np.stack(wins)[..., None].astype(np.float32)
        emb = np.squeeze(self.emb.run(None, {"input_1": batch})[0])
        if emb.ndim == 1:
            emb = emb[None, :]
        return np.concatenate([emb.mean(0), emb.std(0)]).astype(np.float32)

    def predict(self, pcm):
        x = self.embed(pcm)
        s = self.state
        hidden = np.maximum(0, s["net.1.weight"] @ x + s["net.1.bias"])
        logits = s["net.4.weight"] @ hidden + s["net.4.bias"]
        logits -= logits.max()
        probs = np.exp(logits)
        return float(probs[1] / probs.sum())


def load_npz(path, backbone_dir):
    with np.load(path, allow_pickle=False) as data:
        return NumpyHead({k: data[k] for k in WEIGHT_KEYS}, backbone_dir)
