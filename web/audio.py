"""语料上传的音频校验：只接受 16kHz / 16bit / 单声道 wav（与训练口径一致）。"""
import io
import wave

import numpy as np

from trainer.data import speech_stats

from . import settings


class AudioError(ValueError):
    """音频不合规（返回 400 给前端）。"""


def decode_wav(data: bytes, *, min_s: float, max_s: float) -> tuple[np.ndarray, float]:
    if not data:
        raise AudioError("空文件")
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise AudioError(f"文件超过 {settings.MAX_UPLOAD_BYTES // 1024} KB")

    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            channels, width = w.getnchannels(), w.getsampwidth()
            rate, nframes = w.getframerate(), w.getnframes()
            raw = w.readframes(nframes)
    except (wave.Error, EOFError) as exc:
        raise AudioError(f"不是合法的 wav：{exc}") from exc

    if channels != 1:
        raise AudioError(f"需要单声道，收到 {channels} 声道")
    if width != 2:
        raise AudioError("需要 16bit PCM")
    if rate != 16000:
        raise AudioError(f"需要 16000 Hz，收到 {rate} Hz")

    duration = nframes / float(rate or 1)
    if duration < min_s:
        raise AudioError(f"太短了（{duration:.2f}s，至少 {min_s}s）")
    if duration > max_s:
        raise AudioError(f"太长了（{duration:.2f}s，最多 {max_s}s）")

    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return pcm, duration


def clip_pct(pcm) -> float:
    """贴到满刻度的样本占比（%）：>1% 基本就是"贴着麦克风喊"的爆音。"""
    arr = np.asarray(pcm)
    if arr.size == 0:
        return 0.0
    return 100.0 * float(np.mean(np.abs(arr) >= 0.999))


def check_speech(pcm) -> dict:
    """录音上传门限：太安静 / 只听到很短一点 / 爆音，都拒收（400 回前端）。

    只校验时长能拦住"没按"，拦不住"按了但没开麦"——线上 25 条空录音就是后者。
    消息要能直接念给孩子听，所以用短句 + 具体动作。
    """
    rms_db, voiced = speech_stats(pcm)
    if rms_db < settings.MIN_CLIP_RMS_DB:
        raise AudioError("没听到声音：看看麦克风开没开，靠近一点再来一次")
    if voiced < settings.MIN_VOICED_S:
        raise AudioError("只听到一点点音，说完一整句再松手")
    clipped = clip_pct(pcm)
    if clipped > settings.MAX_CLIP_PCT:
        raise AudioError("声音爆掉了：离麦克风太近，退后半臂再录一遍")
    return {"rms_db": round(rms_db, 1), "voiced_s": round(voiced, 2),
            "clip_pct": round(clipped, 2)}


def wav_info(path) -> dict:
    """列表页只读头，不解码整段。"""
    with wave.open(str(path), "rb") as w:
        return {"duration": round(w.getnframes() / float(w.getframerate() or 1), 2),
                "sample_rate": w.getframerate()}
