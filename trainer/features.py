"""波形 → 对数梅尔谱。

只依赖 numpy：不引入 librosa / torchaudio，镜像小、构建快，
且参数（40 × 100，25 ms 窗 / 10 ms 跳）在这里集中定义，
以后浏览器端要复刻同一套，改这一个文件即可。
"""
import numpy as np

SR = 16000
N_FFT = 512
WIN = 400          # 25 ms
HOP = 160          # 10 ms
N_MELS = 40
FMIN, FMAX = 20.0, 7600.0
T_FRAMES = 100     # 1 秒 → 100 帧


def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
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


_FB = mel_filterbank()
_WIN = np.hanning(WIN).astype(np.float32)


def logmel(wav):
    """float32 单声道波形 → (N_MELS, T_FRAMES)，逐条按均值方差归一化。"""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < WIN:
        wav = np.pad(wav, (0, WIN - len(wav)))

    n = 1 + (len(wav) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    frames = wav[idx] * _WIN

    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    mel = np.log(spec @ _FB.T + 1e-6).T          # (N_MELS, n)

    if mel.shape[1] < T_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, T_FRAMES - mel.shape[1])),
                     mode="constant", constant_values=float(mel.min()))
    else:
        start = (mel.shape[1] - T_FRAMES) // 2
        mel = mel[:, start:start + T_FRAMES]

    return ((mel - mel.mean()) / (mel.std() + 1e-5)).astype(np.float32)
