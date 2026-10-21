"""CNN 的纯 numpy 前向：模型导出到桌宠后不再需要 torch。

一个文件同时装**简单 CNN**（一层卷积 + 一层池化）和**复杂 CNN**（三层卷积）：
桌宠侧只 `load_npz()` 拿到对应实现，导出包仍是这一份源码。

为什么单独一个文件：它会**原样**打进部署包（`buddy-wake-<run_id>.zip`），由桌宠
从包目录 import。所以这里：
- 只依赖 numpy，不许 import torch / 不许用包内相对导入；
- 特征口径与 `trainer/features.py` 完全一致（同 40×100 对数梅尔、25 ms 窗 /
  10 ms 跳、逐条均值方差归一化）。`tests/test_export.py` 会逐条对比两者；
- 与 torch 的 `models.CNN` / `models.SimpleCNN` 逐位一致（float32 舍入级别）。

桌宠侧只调 `load_npz()` + `predict()`；导出侧还会用 `FEATURES` 写进 meta.json，
`selftest_window()` 生成确定性噪声让两端各算一次概率（防实现漂移）。
"""
import numpy as np

# ---- 特征参数（与 features.py 同口径，改这里必须同时改那边 + 重新训练）--------
SR = 16000
N_FFT = 512
WIN = 400          # 25 ms
HOP = 160          # 10 ms
N_MELS = 40
FMIN, FMAX = 20.0, 7600.0
T_FRAMES = 100     # 1 秒 → 100 帧

FEATURES = {"sr": SR, "n_fft": N_FFT, "win": WIN, "hop": HOP, "n_mels": N_MELS,
            "fmin": FMIN, "fmax": FMAX, "t_frames": T_FRAMES}


def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    bins = np.floor((n_fft + 1) * _mel2hz(
        np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_mels + 2)) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, fb.shape[1])):
            fb[i, k] = (k - left) / (center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[i, k] = (right - k) / (right - center)
    return fb


_FB = _mel_filterbank()
_HANN = np.hanning(WIN).astype(np.float32)


def logmel(wav):
    """float32 单声道波形 → (40,100)，逐条按均值方差归一化。"""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < WIN:
        wav = np.pad(wav, (0, WIN - len(wav)))

    n = 1 + (len(wav) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    frames = wav[idx] * _HANN

    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    mel = np.log(spec @ _FB.T + 1e-6).T          # (N_MELS, n)

    if mel.shape[1] < T_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, T_FRAMES - mel.shape[1])),
                     mode="constant", constant_values=float(mel.min()))
    else:
        start = (mel.shape[1] - T_FRAMES) // 2
        mel = mel[:, start:start + T_FRAMES]

    return ((mel - mel.mean()) / (mel.std() + 1e-5)).astype(np.float32)


# ---- 前向需要的四个算子 -------------------------------------------------------
def _im2col(x, kh, kw, pad, stride=1):
    c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    ho = (h + 2 * pad - kh) // stride + 1
    wo = (w + 2 * pad - kw) // stride + 1
    shape = (c, kh, kw, ho, wo)
    strides = (xp.strides[0], xp.strides[1], xp.strides[2],
               xp.strides[1] * stride, xp.strides[2] * stride)
    col = np.lib.stride_tricks.as_strided(xp, shape, strides)
    return col.reshape(c * kh * kw, ho * wo)


def _conv2d(x, w, b, pad=1):
    cout, _cin, kh, kw = w.shape
    ho = x.shape[1] + 2 * pad - kh + 1
    wo = x.shape[2] + 2 * pad - kw + 1
    y = w.reshape(cout, -1) @ _im2col(x, kh, kw, pad)
    return y.reshape(cout, ho, wo) + b[:, None, None]


def _bn(x, w, b, mean, var, eps=1e-5):
    scale = w / np.sqrt(var + eps)
    return x * scale[:, None, None] + (b - mean * scale)[:, None, None]


def _maxpool2(x):
    c, h, w = x.shape
    return x[:, : h // 2 * 2, : w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


class _NumpyWaveNet:
    """两个架构共用的推理骨架：卷积块由子类给，池化 / 头在这里。"""

    keys: tuple[str, ...] = ()
    arch = ""

    def __init__(self, state: dict):
        self.s = {k: np.asarray(v, dtype=np.float32)
                  for k, v in state.items() if k in self.keys}
        missing = [k for k in self.keys if k not in self.s]
        if missing:
            raise ValueError(f"权重缺键：{', '.join(missing[:3])}")

    # 子类实现：特征卷积 + 池化前的形状
    def _blocks(self, x):                     # pragma: no cover - 抽象
        raise NotImplementedError

    def _pool(self, x):
        s = self.s
        mean = x.mean(axis=(1, 2))
        std = x.std(axis=(1, 2))                       # 与 torch 一致：unbiased=False
        att = _conv2d(x, s["pool.att.weight"], s["pool.att.bias"], pad=0).reshape(-1)
        w = np.exp(att - att.max())
        w = (w / w.sum()).reshape(x.shape[1], x.shape[2])
        pooled = (x * w[None]).sum(axis=(1, 2))
        return np.concatenate([mean, std, pooled])

    def prob(self, mel):
        mel = np.asarray(mel, np.float32)
        if mel.ndim == 2:
            mel = mel[None]
        feats = self._pool(self._blocks(mel))
        logits = self.s["head.weight"] @ feats + self.s["head.bias"]
        e = np.exp(logits - logits.max())
        return float((e / e.sum())[1])

    def predict(self, pcm):
        """16k float32 波形（任意长度，内部按 1 s 居中裁剪/补齐）→ 正类概率。"""
        return self.prob(logmel(pcm))


class NumpySimpleCNN(_NumpyWaveNet):
    """`models.SimpleCNN` 的推理版：一层卷积 + 一层池化 + 统计池化 + 线性头。"""

    arch = "simple"
    keys = (
        "features.0.weight", "features.0.bias",
        "features.1.weight", "features.1.bias",
        "features.1.running_mean", "features.1.running_var",
        "pool.att.weight", "pool.att.bias",
        "head.weight", "head.bias",
    )

    def _blocks(self, x):
        s = self.s
        x = _conv2d(x, s["features.0.weight"], s["features.0.bias"])
        x = np.maximum(_bn(x, s["features.1.weight"], s["features.1.bias"],
                           s["features.1.running_mean"], s["features.1.running_var"]), 0)
        return _maxpool2(x)


class NumpyCNN(_NumpyWaveNet):
    """`models.CNN` 的推理版：三层卷积 + 统计/注意力池化 + 线性头。"""

    arch = "cnn"
    keys = (
        "features.0.weight", "features.0.bias",
        "features.1.weight", "features.1.bias",
        "features.1.running_mean", "features.1.running_var",
        "features.4.weight", "features.4.bias",
        "features.5.weight", "features.5.bias",
        "features.5.running_mean", "features.5.running_var",
        "features.8.weight", "features.8.bias",
        "features.9.weight", "features.9.bias",
        "features.9.running_mean", "features.9.running_var",
        "pool.att.weight", "pool.att.bias",
        "head.weight", "head.bias",
    )

    def _blocks(self, x):
        s = self.s
        x = _conv2d(x, s["features.0.weight"], s["features.0.bias"])
        x = np.maximum(_bn(x, s["features.1.weight"], s["features.1.bias"],
                           s["features.1.running_mean"], s["features.1.running_var"]), 0)
        x = _maxpool2(x)

        x = _conv2d(x, s["features.4.weight"], s["features.4.bias"])
        x = np.maximum(_bn(x, s["features.5.weight"], s["features.5.bias"],
                           s["features.5.running_mean"], s["features.5.running_var"]), 0)
        x = _maxpool2(x)

        x = _conv2d(x, s["features.8.weight"], s["features.8.bias"])
        x = np.maximum(_bn(x, s["features.9.weight"], s["features.9.bias"],
                           s["features.9.running_mean"], s["features.9.running_var"]), 0)
        return x


# 导出时按这份清单过滤（BatchNorm 的 num_batches_tracked 是训练用的，别带进包里）
WEIGHT_KEYS_SIMPLE = NumpySimpleCNN.keys
WEIGHT_KEYS = NumpyCNN.keys
ARCH = {"simple": NumpySimpleCNN, "cnn": NumpyCNN}


def weights_for(model: str) -> tuple[str, ...]:
    """导出侧按模型取权重清单（simple 是 cnn 的真子集）。"""
    try:
        return tuple(ARCH[model].keys)
    except KeyError as exc:
        raise ValueError(f"没有这个可导出的模型: {model}") from exc


def load_from_state(state: dict, model: str | None = None):
    """认架构：显式给 model 就用它，否则按权重键判断（老导出包只有 CNN）。"""
    if model in ARCH:
        return ARCH[model](state)
    return NumpySimpleCNN(state) if "features.4.weight" not in state else NumpyCNN(state)


def load_npz(path, model: str | None = None):
    with np.load(path) as data:
        return load_from_state({k: data[k] for k in data.files}, model)


def selftest_window(seed: int = 1234):
    """确定性噪声窗口：两端各算一次概率，实现漂移就当场发现（零额外体积）。"""
    return (np.random.default_rng(seed).standard_normal(SR).astype(np.float32) * 0.1)
