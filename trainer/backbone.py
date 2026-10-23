"""openWakeWord 预训练底座：16k PCM → mel(32) → Google speech embedding(96)。

只做特征，不做分类：输出和训练/推理共用的 192 维池化向量（mean + std）。
权重是 openWakeWord 官方发布的两个 ONNX 文件（共 ~2.4 MB），随镜像走，
学生端不需要下载任何东西。

    pcm (float32, 16k) → embed() → (192,)

底座缺失时抛 RuntimeError，由 web 层翻译成 409，不影响 mlp / cnn 训练。
"""
import os
import threading

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.environ.get("NOVA_OWW_DIR") or os.path.join(
    _HERE, "models", "backbone", "openwakeword")

MEL_WINDOW = 76          # 76 帧 mel ≈ 0.78 s，官方口径
MEL_HOP = 8              # 80 ms
EMB_DIM = 96
POOLED_DIM = EMB_DIM * 2  # mean + std
WIN = 16000              # 和 logmel 一样：长音频取中心 1 s，保证训练/推理同口径


class BackboneError(RuntimeError):
    pass


class OWWFeatures:
    def __init__(self, root=DEFAULT_DIR):
        self.root = root
        mel_path = os.path.join(root, "melspectrogram.onnx")
        emb_path = os.path.join(root, "embedding_model.onnx")
        for p in (mel_path, emb_path):
            if not os.path.isfile(p):
                raise BackboneError(
                    f"预训练底座文件缺失：{p}（应随镜像放在 models/backbone/openwakeword/）")
        try:
            import onnxruntime as ort
        except ImportError as exc:                     # pragma: no cover
            raise BackboneError("缺少 onnxruntime，无法加载预训练底座") from exc

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1                    # 测试台每 300 ms 一次，别抢线程
        so.inter_op_num_threads = 1
        self.mel = ort.InferenceSession(
            mel_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.emb = ort.InferenceSession(
            emb_path, sess_options=so, providers=["CPUExecutionProvider"])

    # --- 单条 -----------------------------------------------------------
    def melspectrogram(self, pcm):
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if len(pcm) > WIN:                             # 与 logmel 的中心裁剪一致
            st = (len(pcm) - WIN) // 2
            pcm = pcm[st:st + WIN]
        if len(pcm) < MEL_WINDOW * 160:
            pcm = np.pad(pcm, (0, MEL_WINDOW * 160 - len(pcm)))
        spec = np.squeeze(self.mel.run(None, {"input": pcm[None, :]})[0])
        return spec / 10.0 + 2.0                       # 官方变换，别改

    def embed(self, pcm):
        spec = self.melspectrogram(pcm)
        wins = [spec[i:i + MEL_WINDOW] for i in range(0, spec.shape[0], MEL_HOP)
                if spec[i:i + MEL_WINDOW].shape[0] == MEL_WINDOW]
        if not wins:
            wins = [np.ones((MEL_WINDOW, 32), dtype=np.float32)]
        batch = np.stack(wins)[..., None].astype(np.float32)
        e = np.squeeze(self.emb.run(None, {"input_1": batch})[0])
        if e.ndim == 1:
            e = e[None, :]
        return np.concatenate([e.mean(0), e.std(0)]).astype(np.float32)

    # --- 批量 -----------------------------------------------------------
    def embed_many(self, pcms, progress=None):
        out = np.zeros((len(pcms), POOLED_DIM), dtype=np.float32)
        for i, pcm in enumerate(pcms):
            out[i] = self.embed(pcm)
            if progress and (i + 1) % 200 == 0:
                progress(i + 1, len(pcms))
        return out


_cache: dict[str, OWWFeatures] = {}
_lock = threading.Lock()


def get_features(root=DEFAULT_DIR) -> OWWFeatures:
    """进程内复用（ONNX 会话加载一次约几十毫秒，别在每个窗口里重开）。"""
    key = os.path.abspath(root)
    with _lock:
        if key not in _cache:
            _cache.clear()
            _cache[key] = OWWFeatures(key)
        return _cache[key]


def available(root=DEFAULT_DIR) -> bool:
    return all(os.path.isfile(os.path.join(root, f))
               for f in ("melspectrogram.onnx", "embedding_model.onnx"))


def embed(pcm) -> np.ndarray:
    return get_features().embed(pcm)
