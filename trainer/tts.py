"""音色合成：豆包（火山）TTS 为主，edge-tts 为备。

豆包走 v3 单向流式 SSE，直接要 16k PCM，落盘不需要任何重采样或解码器。
凭证只从环境变量读（仓库根的 `.env` 会被自动加载，但不覆盖已注入的 env），不进仓库。
"""
import base64
import json
import os
import urllib.request
import uuid

from .env import load_dotenv

# 本地裸跑时也认仓库根的 .env（容器里 compose 已经注入，这里不会覆盖）
load_dotenv()

URL = os.environ.get("DOUBAO_TTS_URL",
                     "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse")
APP_ID = os.environ.get("DOUBAO_APP_ID", "")
TOKEN = os.environ.get("DOUBAO_ACCESS_TOKEN", "")

RID_2 = "seed-tts-2.0"
RID_1 = "seed-tts-1.0"
# 只在调用方没指定 resource_id 时兜底；音色与 resource_id 必须匹配，
# 所以 VOICES 表里的 rid 优先于这个默认值。
DEFAULT_RESOURCE_ID = os.environ.get("DOUBAO_RESOURCE_ID", RID_2)

# 实测可用的中文音色（都能读中英混合文本）。resource_id 必须与音色匹配，否则报
# "resource ID is mismatched with speaker related resource"。
ZH_VOICES = [
    ("zh_female_qinqienv_uranus_bigtts", RID_2),
    ("zh_female_vv_uranus_bigtts", RID_2),
    ("zh_male_beijingxiaoye_emo_v2_mars_bigtts", RID_1),
    ("zh_female_wanwanxiaohe_moon_bigtts", RID_1),
    ("zh_female_linjianvhai_moon_bigtts", RID_1),
    ("zh_male_yuanboxiaoshu_moon_bigtts", RID_1),
    ("zh_male_shaonianzixin_moon_bigtts", RID_1),
    ("zh_female_zhixingnvsheng_mars_bigtts", RID_1),
    ("zh_male_ahu_conversation_wvae_bigtts", RID_1),
    ("zh_male_yangguangqingnian_moon_bigtts", RID_1),
    ("zh_female_meilinvyou_moon_bigtts", RID_1),
    ("zh_male_lengkugege_emo_v2_mars_bigtts", RID_1),
    ("zh_female_gaolengyujie_emo_v2_mars_bigtts", RID_1),
    ("zh_female_sajiaonvyou_moon_bigtts", RID_1),
    ("zh_male_jingqiangkanye_moon_bigtts", RID_1),
]

# 英文音色只能读纯英文：实测拿它读 "buddy助手" 会拖成 6.2 秒（把中文念坏了）。
EN_VOICES = [
    ("en_male_tim_uranus_bigtts", RID_2),
    ("en_male_corey_emo_v2_mars_bigtts", RID_1),
]


def tts_pcm(text, speaker, resource_id=None, speech_rate=0, sr=16000, timeout=40):
    """返回 16k/16bit/单声道 PCM 字节。resource_id 省略时用 DOUBAO_RESOURCE_ID。"""
    if not APP_ID or not TOKEN:
        raise RuntimeError("缺 DOUBAO_APP_ID / DOUBAO_ACCESS_TOKEN（放 .env，不进 git）")
    resource_id = resource_id or DEFAULT_RESOURCE_ID
    body = {
        "user": {"uid": "nova-voice-playground"},
        "req_params": {
            "text": text,
            "speaker": speaker,
            "audio_params": {"format": "pcm", "sample_rate": sr,
                             "speech_rate": int(speech_rate)},
        },
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Api-App-Id": APP_ID,
            "X-Api-Access-Key": TOKEN,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
        })
    buf = bytearray()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            s = line.decode("utf-8", "ignore").strip()
            if not s.startswith("data:"):
                continue
            payload = s[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            code = obj.get("code", 0)
            if code == 20000000:          # 火山约定：会话正常结束
                break
            if code not in (0, None):
                raise RuntimeError(f"TTS 失败: {obj.get('message') or obj}")
            if obj.get("data"):
                buf += base64.b64decode(obj["data"])
    if not buf:
        raise RuntimeError("TTS 没返回音频")
    return bytes(buf)


def edge_voices(locale_prefix="en-"):
    """edge-tts 可用音色（需要外网；返回 ShortName 列表）。"""
    token = "6A5AA1D4EAFF4E9FB37E23D68491D6F4"
    url = ("https://speech.platform.bing.com/consumer/speech/synthesize/"
           f"readaloud/voices/list?trustedclienttoken={token}")
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = json.loads(resp.read())
    return [v["ShortName"] for v in data if v.get("Locale", "").startswith(locale_prefix)]
