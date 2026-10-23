"""合成语料：用豆包 TTS 造正 / 负样本。

为什么放服务端：容器里没有 ffmpeg，豆包能直出 16k/16bit/单声道 PCM，
落盘就是训练口径；浏览器只需要选音色和条数，不用开麦克风。

口径与录音完全一致：生成的 wav 再走一遍 `audio.decode_wav`（16k / 单声道 /
0.3~5 s），过不了的丢弃并计入 failed，绝不写进语料目录。

**音频不进 data/**：wav 写 `.cache/synth/{pos,neg}`（可重建的缓存），元数据写
`data/synth.json`（含 oss_key / sha256 / 时长）；配了 OSS 凭证就顺手传上去，
上传失败只记 `oss_key: null`，由 `scripts/oss_push.py --synth` 补传——功能不降级。
"""
import hashlib
import io
import random
import re
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed

from trainer import tts as tts_mod
from trainer.tts import EN_VOICES, ZH_VOICES

from . import audio, oss, settings, store, words

WAKE_WORD = "buddy助手"
# 中文音色念中英混排没问题（"buddy助手" 就是这么念的），英文音色念中文会拖成
# 6 秒多。所以「能不能造正样本」取决于**这个词里有没有中文**，不是音色单方面的事。
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


# 音色 id → 下拉里显示的名字（学生不认 id）
VOICE_LABELS = {
    "zh_female_qinqienv_uranus_bigtts": "亲切女声",
    "zh_female_vv_uranus_bigtts": "VV 女声",
    "zh_male_beijingxiaoye_emo_v2_mars_bigtts": "北京小爷",
    "zh_female_wanwanxiaohe_moon_bigtts": "湾湾小何",
    "zh_female_linjianvhai_moon_bigtts": "邻家女孩",
    "zh_male_yuanboxiaoshu_moon_bigtts": "渊博小叔",
    "zh_male_shaonianzixin_moon_bigtts": "少年自信",
    "zh_female_zhixingnvsheng_mars_bigtts": "知性女生",
    "zh_male_ahu_conversation_wvae_bigtts": "阿虎",
    "zh_male_yangguangqingnian_moon_bigtts": "阳光青年",
    "zh_female_meilinvyou_moon_bigtts": "甜美女友",
    "zh_male_lengkugege_emo_v2_mars_bigtts": "冷酷哥哥",
    "zh_female_gaolengyujie_emo_v2_mars_bigtts": "高冷御姐",
    "zh_female_sajiaonvyou_moon_bigtts": "撒娇女友",
    "zh_male_jingqiangkanye_moon_bigtts": "京腔侃爷",
    "en_male_tim_uranus_bigtts": "Tim（英文）",
    "en_male_corey_emo_v2_mars_bigtts": "Corey（英文）",
}

# 同一条正样本换语速，避免 3 条几乎一模一样的波形
POS_RATES = (-20, -10, 0, 10, 20)

NEG_ZH = [
    "今天天气不错", "打开台灯", "我想吃冰淇淋", "帮我找一下书包", "这道题我不会",
    "老师再见", "把音量调小一点", "明天要下雨吗", "给我讲个故事", "我的铅笔不见了",
    "我们一起去操场", "妈妈在做饭", "这个字怎么读", "我想喝水", "现在几点了",
]
NEG_EN = [
    "turn on the light", "what time is it", "i am hungry", "play some music",
    "how are you today", "open the window", "where is my bag", "good morning",
]

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_index_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def configured() -> bool:
    """凭证在进程启动时从 env 读入（.env → env.load_dotenv），这里只做只读判断。"""
    return bool(tts_mod.APP_ID and tts_mod.TOKEN)


def voice_options(text: str | None = None) -> list[dict]:
    """text 给了就按这个词算「能不能造正样本」（自定义唤醒词可能是纯英文）。"""
    cjk = text is None or has_cjk(text)
    rows = []
    for voice, _rid in ZH_VOICES:
        rows.append({"id": voice, "label": VOICE_LABELS.get(voice, voice),
                     "lang": "zh", "can_pos": True})
    for voice, _rid in EN_VOICES:
        rows.append({"id": voice, "label": VOICE_LABELS.get(voice, voice),
                     "lang": "en", "can_pos": not cjk})
    return rows


def limits() -> dict:
    return {"max_pos": settings.SYNTH_MAX_POS, "max_neg": settings.SYNTH_MAX_NEG,
            "min_clip_s": settings.MIN_CLIP_S, "max_clip_s": settings.MAX_CLIP_S}


def _find_voice(voice: str) -> tuple[str | None, bool]:
    """返回 (resource_id, 是否中文音色)；找不到返回 (None, False)。"""
    for vid, rid in ZH_VOICES:
        if vid == voice:
            return rid, True
    for vid, rid in EN_VOICES:
        if vid == voice:
            return rid, False
    return None, False


def _wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue()


def _plan(n_pos: int, n_neg: int, zh: bool,
          text: str = WAKE_WORD) -> list[tuple[str, str, int]]:
    """(label, text, speech_rate) 列表。负样本句子按需循环，保证条数够。"""
    rng = random.Random()
    tasks: list[tuple[str, str, int]] = []
    for i in range(n_pos):
        tasks.append(("pos", text, POS_RATES[i % len(POS_RATES)]))

    pool = NEG_ZH if zh else NEG_EN
    texts = rng.sample(pool, k=min(n_neg, len(pool)))
    while len(texts) < n_neg:                       # 要的比句子多，就再来一轮
        texts.append(rng.choice(pool))
    for line in texts[:n_neg]:
        tasks.append(("neg", line, rng.choice((-10, 0, 10))))
    return tasks


def _prepare(voice: str, n_pos: int, n_neg: int, text: str):
    """凭证 / 音色 / 条数校验，返回 (rid, zh, n_pos, n_neg)（条数已夹紧）。"""
    if not configured():
        raise store.StoreError(
            "服务器还没配豆包 TTS 凭证（DOUBAO_APP_ID / DOUBAO_ACCESS_TOKEN），"
            "先用麦克风录制吧", 503)

    rid, zh = _find_voice(voice)
    if not rid:
        raise store.StoreError("没有这个音色", 400)
    if n_pos < 0 or n_neg < 0:
        raise store.StoreError("条数不能是负数", 400)
    if not zh and n_pos > 0 and has_cjk(text):
        raise store.StoreError(
            f"英文音色读不了「{text}」（中文会被拖长），换中文音色，或只造负样本", 400)

    n_pos = min(int(n_pos), settings.SYNTH_MAX_POS)
    n_neg = min(int(n_neg), settings.SYNTH_MAX_NEG)
    if n_pos + n_neg == 0:
        raise store.StoreError("至少选一条要合成", 400)
    return rid, zh, n_pos, n_neg


# ---------------------------------------------------------------- 元数据索引
def index() -> dict:
    data = store._read_json(settings.SYNTH_INDEX, {}) or {}
    data.setdefault("clips", [])
    return data


def _record(name: str, label: str, voice: str, text: str, data: bytes,
            oss_key: str | None) -> None:
    with _index_guard:
        data_idx = index()
        digest = hashlib.sha256(data).hexdigest()
        data_idx["clips"] = [c for c in data_idx["clips"] if c.get("name") != name]
        data_idx["clips"].append({
            "name": name, "label": label, "voice": voice, "text": text,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "size": len(data), "sha256": digest, "oss_key": oss_key,
        })
        data_idx["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        store._write_json(settings.SYNTH_INDEX, data_idx)


def store_synth(label: str, data: bytes, voice: str = "", text: str = "") -> str:
    """落盘到 .cache/synth，上传 OSS（尽力而为），写元数据。返回文件名。"""
    folder = store.synth_dir(label)
    name = store._write_clip(folder, f"{label}-synth-", data)
    key = oss.object_key(f"synth/{label}/{name}")
    uploaded = None
    if oss.configured():
        try:
            oss.put_object(key, data, content_type="audio/wav")
            uploaded = key
        except oss.OSSError:
            uploaded = None          # 失败降级：本地仍可用，oss_push --synth 会补传
    _record(name, label, voice, text, data, uploaded)
    return name


# ---------------------------------------------------------------- 合成执行
def _speak(key: str, voice: str, rid, zh: bool, n_pos: int, n_neg: int,
           text: str, sink) -> dict:
    """公共合成核心：`sink(label, data)` 决定写共享语料还是词目录。"""
    tasks = [(label, line, rate, voice, rid)
             for label, line, rate in _plan(n_pos, n_neg, zh, text)]
    out = _execute(key, tasks, sink)
    return {**out, "requested": {"pos": n_pos, "neg": n_neg}, "voice": voice}


def _execute(key: str, tasks: list[tuple[str, str, int, str, str]], sink,
             on_progress=None) -> dict:
    """跑一批任务：`(label, text, rate, voice, rid)`。

    `on_progress(done, total, created, failed)` 按**真实完成顺序**回调（as_completed），
    合成进度条不必等第一个慢请求。`sink(label, data)` 决定落盘位置。
    """
    lock = _lock_for(key)
    if not lock.acquire(blocking=False):
        raise store.StoreError("这个数据集正在合成，等这批跑完再点", 409)
    try:
        created = {"pos": 0, "neg": 0}
        errors: list[str] = []
        total, done = len(tasks), 0

        def run(task: tuple[str, str, int, str, str]) -> tuple[str, bytes]:
            label, line, rate, voice, rid = task
            return label, tts_mod.tts_pcm(line, voice, rid, rate)

        workers = max(1, min(settings.SYNTH_WORKERS, total or 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                label, line, _rate, _voice, _rid = task
                try:
                    got_label, pcm = future.result()
                except Exception as exc:                    # noqa: BLE001
                    errors.append(f"{label}/{line}: {str(exc)[:120]}")
                else:
                    data = _wav_bytes(pcm)
                    try:
                        audio.decode_wav(data, min_s=settings.MIN_CLIP_S,
                                         max_s=settings.MAX_CLIP_S)
                    except audio.AudioError as exc:
                        errors.append(f"{label}/{line}: {exc}")
                    else:
                        try:
                            sink(got_label, data)
                        except store.StoreError as exc:
                            errors.append(f"{label}/{line}: {exc}")
                        else:
                            created[got_label] += 1
                done += 1
                if on_progress:
                    on_progress(done, total, created["pos"] + created["neg"], len(errors))

        if not created["pos"] and not created["neg"]:
            detail = errors[0] if errors else "TTS 没有返回音频"
            raise store.StoreError(f"合成失败：{detail}", 502)
        return {"created": created, "failed": len(errors), "errors": errors[:5]}
    finally:
        lock.release()


def synthesize(voice: str, n_pos: int, n_neg: int, text: str = WAKE_WORD) -> dict:
    """同步合成一批默认语料；返回新建条数 + 最新语料清单。"""
    rid, zh, n_pos, n_neg = _prepare(voice, n_pos, n_neg, text)
    out = _speak("corpus", voice, rid, zh, n_pos, n_neg, text,
                 lambda label, data: store_synth(label, data, voice, text))
    return {**out, **store.corpus_summary(), "synth_configured": oss.configured()}


def synthesize_word(wid: str, voice: str, n_pos: int) -> dict:
    """给自定义唤醒词造正例（只造 pos，负例永远来自默认语料）。"""
    record = words.get_word(wid)
    rid, zh, n_pos, _ = _prepare(voice, n_pos, 0, record["text"])
    out = _speak(wid, voice, rid, zh, n_pos, 0, record["text"],
                 lambda label, data: words.add_clip(wid, data, tag="synth"))
    return {**out, "word": record, "counts": words.counts(wid),
            "clips": words.list_clips(wid)}


# ---------------------------------------------------------------- 自动多音色（带进度）
def auto_pos_voices(text: str) -> list[tuple[str, str]]:
    """能念这个词正样本的音色：中文音色都行；英文音色只给纯英文词。"""
    rows = [(voice, rid) for voice, rid in ZH_VOICES]
    if not has_cjk(text):
        rows += [(voice, rid) for voice, rid in EN_VOICES]
    return rows


def auto_word_tasks(text: str) -> list[tuple[str, str, int, str, str]]:
    """每个音色 2 条、语速不同：中文词 30 条，英文词 34 条（受 WORD_SYNTH_MAX 兜底）。"""
    rows = auto_pos_voices(text)
    tasks: list[tuple[str, str, int, str, str]] = []
    for i, (voice, rid) in enumerate(rows):
        rates = (POS_RATES[i % len(POS_RATES)], POS_RATES[(i + 2) % len(POS_RATES)])
        for rate in rates:
            tasks.append(("pos", text, rate, voice, rid))
    return tasks[: settings.WORD_SYNTH_MAX]


_word_jobs: dict[str, dict] = {}
_word_jobs_guard = threading.Lock()


def _job_snapshot(job: dict | None) -> dict:
    base = {"state": "idle", "done": 0, "total": 0, "created": 0, "failed": 0,
            "voices": [], "error": None}
    if not job:
        return base
    return {**base, **job}


def word_synth_status(wid: str) -> dict:
    """单进程内存态：没任务就是 idle（服务重启后前端把它当"已中断，可重试"）。"""
    with _word_jobs_guard:
        job = dict(_word_jobs.get(wid) or {})
    return _job_snapshot(job)


def _patch_job(wid: str, **fields) -> None:
    with _word_jobs_guard:
        job = _word_jobs.get(wid)
        if job is not None:
            job.update(fields)


def start_word_synth(wid: str) -> dict:
    """起一个后台合成任务，立刻返回进度快照；重复点击由任务表 + 非阻塞锁双拦。"""
    if not configured():
        raise store.StoreError(
            "服务器还没配豆包 TTS 凭证（DOUBAO_APP_ID / DOUBAO_ACCESS_TOKEN），"
            "先用麦克风录制吧", 503)
    record = words.get_word(wid)
    tasks = auto_word_tasks(record["text"])
    if not tasks:
        raise store.StoreError("没有能念这个词的音色", 400)
    voices = list(dict.fromkeys(task[3] for task in tasks))

    with _word_jobs_guard:
        job = _word_jobs.get(wid)
        if job and job.get("state") == "running":
            raise store.StoreError("这个词正在合成，等这批跑完再点", 409)
        job = {"state": "running", "done": 0, "total": len(tasks), "created": 0,
               "failed": 0, "voices": voices, "error": None,
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _word_jobs[wid] = job

    threading.Thread(target=_word_synth_worker, args=(wid, tasks),
                     daemon=True, name=f"word-synth-{wid}").start()
    return _job_snapshot(job)


def _word_synth_worker(wid: str, tasks: list[tuple[str, str, int, str, str]]) -> None:
    def on_progress(done: int, total: int, created: int, failed: int) -> None:
        _patch_job(wid, done=done, total=total, created=created, failed=failed)

    try:
        out = _execute(wid, tasks,
                       lambda label, data: words.add_clip(wid, data, tag="synth"),
                       on_progress)
    except store.StoreError as exc:
        _patch_job(wid, state="failed", error=str(exc))
        return
    except Exception as exc:                            # noqa: BLE001
        _patch_job(wid, state="failed", error=f"合成失败：{str(exc)[:160]}")
        return
    _patch_job(wid, state="done", done=len(tasks), created=out["created"]["pos"],
               failed=out["failed"], counts=words.counts(wid),
               finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
