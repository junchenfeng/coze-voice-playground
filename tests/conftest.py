"""测试夹具：所有用例都跑在临时数据目录里，不会碰 ./data。

单数据域后没有账户夹具了：`client` 就是全部，"谁在用"这件事不存在。
"""
import io
import math
import os
import struct
import tempfile
import wave

_TMP = tempfile.mkdtemp(prefix="nova-web-tests-")
os.environ["NOVA_DATA_ROOT"] = _TMP
# 缓存也放进临时目录：种子语料 / 合成音频的下载不该污染工作区
os.environ.setdefault("NOVA_CACHE_ROOT", os.path.join(_TMP, ".cache"))
os.environ["NOVA_MAX_RUNNING_TRAININGS"] = "1"
# 测试里不发真实网络请求：把 OSS/CDN 都指到 127.0.0.1:9（立刻 connection refused）
os.environ.setdefault("OSS_CDN_BASE", "http://127.0.0.1:9/cdn")
os.environ.setdefault("OSS_ENDPOINT", "http://127.0.0.1:9")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from web.app import app  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_store():
    """单数据域是全局状态：每个用例开始前清空语料 / 运行 / 词库 / 指针。"""
    import shutil

    from web import settings

    for path in (settings.CORPUS_ROOT, settings.RUNS_ROOT, settings.WORDS_ROOT,
                 settings.SEED_CACHE_ROOT, settings.SYNTH_CACHE_ROOT):
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
    settings.ACTIVE_PATH.unlink(missing_ok=True)
    settings.SYNTH_INDEX.unlink(missing_ok=True)
    # 笔记是仓库里的静态内容：临时 data 根里放一份，/notes 才有东西可渲染
    notes_src = settings.BASE_DIR / "data" / "research-notes.json"
    if notes_src.is_file():
        settings.RESEARCH_NOTES.parent.mkdir(parents=True, exist_ok=True)
        settings.RESEARCH_NOTES.write_bytes(notes_src.read_bytes())
    yield


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def build_wav(seconds=1.0, sample_rate=16000, channels=1, freq=440.0, amp=0.3,
              speech_s=None):
    """speech_s: 只有前这么多秒有声，其余补静音（用来造"只说了半句"的录音）。"""
    frames = int(sample_rate * seconds)
    data = bytearray()
    for i in range(frames):
        quiet = speech_s is not None and i > speech_s * sample_rate
        value = 0 if quiet else int(amp * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        for _ in range(channels):
            data += struct.pack("<h", value)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(data))
    return buf.getvalue()


@pytest.fixture()
def wav_bytes():
    return build_wav
