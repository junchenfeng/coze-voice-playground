"""FastAPI 应用：页面 + JSON API + SSE。

约定：
- **没有账户、没有工作区**：每个扣子编程项目就是一台独立机器，全站一个数据域；
  前端不需要（也不允许）自己传路径。
- 可预期错误统一走 `StoreError` / `AudioError`，返回 `{"detail": "..."}`。
- 只读的外部接口：`/note-log/{1..4}`（研究笔记日志，见 web/notes.py）。
"""
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, RedirectResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import jobs, notes, predictor, seed, settings, store, synth, words
from .audio import AudioError, check_speech, decode_wav
from .store import StoreError

templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))
NO_STORE = {"Cache-Control": "no-store"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    settings.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for label in store.LABELS:                     # 麦克风语料 pos/neg + 合成缓存
        store.corpus_dir(label)
        store.synth_dir(label)
    settings.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    settings.WORDS_ROOT.mkdir(parents=True, exist_ok=True)
    jobs.reconcile()
    # 预置语料：后台补齐（第一个训练请求不必等 800 个文件下完），失败只记日志
    seed.prefetch_async()
    yield
    jobs.stop_all()


app = FastAPI(title="Nova Voice Lab", docs_url=None, redoc_url=None,
              openapi_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")


@app.middleware("http")
async def guard_body_size(request: Request, call_next):
    """在解析 multipart 之前先按 Content-Length 拦掉超大请求。"""
    if request.method in ("POST", "PUT", "PATCH"):
        raw = request.headers.get("content-length")
        if raw and raw.isdigit() and int(raw) > settings.MAX_UPLOAD_BYTES + 256 * 1024:
            return JSONResponse({"detail": "上传内容太大"}, status_code=413)
    return await call_next(request)


@app.exception_handler(StoreError)
async def _store_error(_request: Request, exc: StoreError):
    return JSONResponse({"detail": str(exc)}, status_code=exc.status)


@app.exception_handler(AudioError)
async def _audio_error(_request: Request, exc: AudioError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_request: Request, exc: StarletteHTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


# ---------------------------------------------------------------- 页面
@app.get("/")
def index():
    return RedirectResponse("/studio", status_code=303)


STEPS = ("record", "train", "check", "word", "models")


@app.get("/studio", response_class=HTMLResponse)
def studio(request: Request, step: str = "", tab: str = ""):
    """五个房间：record / train / check / word / models（没有 login、没有引导态）。"""
    active_id = store.get_active()
    active = None
    if active_id:
        try:
            cfg = store.read_config(active_id)
            active = {"run_id": active_id, "model": cfg.get("model", "?"),
                      "wake_word": cfg.get("wake_word")}
        except StoreError:
            active = None

    want = step or ("check" if tab == "test" else "")
    if want not in STEPS:
        want = "train"

    # 「喊一声试试」测的是当前应用的模型：副标题必须跟它认的词走
    check_word = ((active or {}).get("wake_word") or {}).get("text") or "buddy助手"
    return templates.TemplateResponse(
        request=request, name="studio.html",
        context={"active": active, "step": want, "wake_word": "buddy助手",
                 "check_word": check_word,
                 "min_pos": settings.MIN_POS, "min_neg": settings.MIN_NEG,
                 "min_word_pos": settings.MIN_WORD_POS},
    )


@app.get("/notes", response_class=HTMLResponse)
def notes_page(request: Request):
    rows = notes.views()
    first = rows[0]["id"] if rows else 0
    return templates.TemplateResponse(
        request=request, name="notes.html",
        context={"notes": rows, "first": first},
    )


@app.get("/note-log/{x}")
def note_log(x: str):
    """只读：原样返回 `note-logs/note-<x>.json`（AI 学习伙伴写的原始日志）。"""
    return PlainTextResponse(notes.note_log_text(x), media_type="application/json",
                             headers=NO_STORE)


# ---------------------------------------------------------------- 当前状态
@app.get("/api/me")
def me():
    """单数据域状态：当前应用的模型（前端启动时拉一次）。"""
    active_id = store.get_active()
    active = None
    if active_id:
        try:
            cfg = store.read_config(active_id)
            active = {"run_id": active_id, "model": cfg.get("model"),
                      "wake_word": cfg.get("wake_word")}
        except StoreError:
            active = None
    return {"active": active, "wake_word": "buddy助手",
            "onboarded": True, "seed": seed.counts(), "oss": bool(settings.OSS_ACCESS_KEY_ID)}


# ---------------------------------------------------------------- 语料
@app.get("/api/corpus")
def get_corpus():
    return store.corpus_summary()


@app.get("/api/voices")
def get_voices(text: str | None = None):
    """合成语料能选的音色；没配 TTS 凭证时前端把入口置灰。"""
    return {"configured": synth.configured(), "voices": synth.voice_options(text),
            **synth.limits()}


class SynthIn(BaseModel):
    voice: str
    pos: int = 3
    neg: int = 3


@app.post("/api/corpus/synth")
def synthesize_clips(body: SynthIn):
    """用选中的 TTS 音色造一批正 / 负样本，返回最新语料清单。"""
    return synth.synthesize(body.voice, body.pos, body.neg)


@app.post("/api/corpus/{label}")
async def upload_clip(label: str, file: UploadFile = File(...)):
    data = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise AudioError(f"文件超过 {settings.MAX_UPLOAD_BYTES // 1024} KB")
    pcm, _dur = decode_wav(data, min_s=settings.MIN_CLIP_S, max_s=settings.MAX_CLIP_S)
    check_speech(pcm)                       # 太安静 / 只有一点点声音：拒收
    name = store.add_clip(label, data)
    return {"name": name, **store.corpus_summary()}


@app.get("/api/corpus/{label}/{name}")
def play_clip(label: str, name: str):
    path = store.clip_path(label, name)
    return FileResponse(path, media_type="audio/wav", headers=NO_STORE)


@app.delete("/api/corpus/{label}/{name}")
def remove_clip(label: str, name: str):
    store.delete_clip(label, name)
    return store.corpus_summary()


# ---------------------------------------------------------------- 自定义唤醒词
class WordIn(BaseModel):
    text: str


class WordSynthIn(BaseModel):
    voice: str
    pos: int = 6


@app.get("/api/words")
def get_words():
    """当前唤醒词（界面只展示一个；旧词目录留盘，按 id 仍可访问）。"""
    word = words.current_word()
    return {"word": word, "words": [word] if word else [],
            "min_pos": settings.MIN_WORD_POS, "text_max": settings.WORD_TEXT_MAX}


@app.post("/api/words")
def create_word(body: WordIn):
    """幂等：同一个词重复提交返回同一个数据集，已录的正例不会被清掉。"""
    record, created = words.create_word(body.text)
    return {**record, "created": created}


@app.get("/api/words/{wid}")
def get_word(wid: str):
    return {**words.get_word(wid), "clips": words.list_clips(wid)}


@app.post("/api/words/{wid}/clips")
async def upload_word_clip(wid: str, file: UploadFile = File(...)):
    data = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise AudioError(f"文件超过 {settings.MAX_UPLOAD_BYTES // 1024} KB")
    pcm, _dur = decode_wav(data, min_s=settings.MIN_CLIP_S, max_s=settings.MAX_CLIP_S)
    check_speech(pcm)
    name = words.add_clip(wid, data)
    return {"name": name, "counts": words.counts(wid), "clips": words.list_clips(wid)}


@app.get("/api/words/{wid}/clips/{name}")
def play_word_clip(wid: str, name: str):
    path = words.clip_path(wid, name)
    return FileResponse(path, media_type="audio/wav", headers=NO_STORE)


@app.delete("/api/words/{wid}/clips/{name}")
def remove_word_clip(wid: str, name: str):
    words.delete_clip(wid, name)
    return {"counts": words.counts(wid), "clips": words.list_clips(wid)}


@app.post("/api/words/{wid}/synth")
def synthesize_word_clips(wid: str, body: WordSynthIn):
    """给这个词造合成正例（只造 pos；负例永远来自默认语料）。"""
    return synth.synthesize_word(wid, body.voice, body.pos)


@app.post("/api/words/{wid}/synth/auto")
def synthesize_word_auto(wid: str):
    """一键多音色合成：后台线程跑，立刻返回进度快照（前端轮询 progress）。"""
    words.get_word(wid)                        # 存在性校验
    return synth.start_word_synth(wid)


@app.get("/api/words/{wid}/synth/progress")
def word_synth_progress(wid: str):
    """合成进度：idle / running / done / failed（无任务 = idle，可重新点）。"""
    words.get_word(wid)
    return synth.word_synth_status(wid)


# ---------------------------------------------------------------- 训练
class RunIn(BaseModel):
    model: str = "cnn"
    base_run: str | None = None
    epochs: int = 30
    batch: int = 16
    lr: float = 1e-3
    val_ratio: float = 0.12           # 页面上已不暴露，保留以兼容老 run
    seed: int = 42                    # 同上：考卷种子，页面固定用默认值
    max_pos: int = settings.TRAIN_MAX_POS   # 「训练数据集大小」正例上限（滑杆）
    max_neg: int = settings.TRAIN_MAX_NEG   # 负例上限（滑杆）
    init_seed: int | None = None      # 初始化/打乱顺序的种子；不传 = 跟 seed 一致
    use_builtin: bool = True          # 预置语料默认参与训练，可取消勾选
    wake_word_id: str | None = None   # 自定义唤醒词数据集；不传 = 默认词 buddy助手


def _validate_run(body: RunIn) -> dict:
    if body.model not in settings.MODELS:
        raise StoreError("模型只能是 simple / cnn / head")
    if not 1 <= body.epochs <= 300:
        raise StoreError("epochs 需在 1~300 之间")
    if not 2 <= body.batch <= 128:
        raise StoreError("batch 需在 2~128 之间（CNN 的 BatchNorm 不接受 batch=1）")
    if not 1e-5 <= float(body.lr) <= 1.0:
        raise StoreError("学习率需在 1e-5 ~ 1 之间")
    if not 0.05 <= float(body.val_ratio) <= 0.5:
        raise StoreError("考卷比例需在 0.05 ~ 0.5 之间")
    if not 0 <= body.seed <= 2 ** 31 - 1:
        raise StoreError("seed 需是非负整数")
    pos_lo, pos_hi = settings.TRAIN_MAX_POS_RANGE
    neg_lo, neg_hi = settings.TRAIN_MAX_NEG_RANGE
    if not pos_lo <= int(body.max_pos) <= pos_hi:
        raise StoreError(f"正例上限需在 {pos_lo}~{pos_hi} 之间")
    if not neg_lo <= int(body.max_neg) <= neg_hi:
        raise StoreError(f"负例上限需在 {neg_lo}~{neg_hi} 之间")
    if body.init_seed is not None and not 0 <= body.init_seed <= 2 ** 31 - 1:
        raise StoreError("init_seed 需是非负整数")

    # 语料门槛分两种口径：
    #   默认词 —— 正负都从本地语料数（预置语料可补足差额）；
    #   自定义词 —— 正例只认词目录的真人录音（≥ MIN_WORD_POS），负例由本地 +
    #     预置语料的**全部**片段充当（老词的正例对新词是难负例，见 trainer/data.py）。
    counts = store.corpus_summary()["counts"]
    builtin = store.builtin_summary() if body.use_builtin else {"available": False}
    wake = None
    if body.wake_word_id:
        record = words.require_ready(body.wake_word_id)
        wake = {"id": record["id"], "text": record["text"]}
        neg = counts["pos"] + counts["neg"]
        if builtin.get("available"):
            neg += builtin["pos"] + builtin["neg"]
        if neg < settings.MIN_NEG:
            raise StoreError(
                f"负样本不够：新唤醒词需要 {settings.MIN_NEG} 条「不是这个唤醒词」的录音，"
                f"现在本地 + 预置只有 {neg} 条")
    else:
        pos, neg = counts["pos"], counts["neg"]
        if builtin.get("available"):
            pos += builtin["pos"]
            neg += builtin["neg"]
        if pos < settings.MIN_POS or neg < settings.MIN_NEG:
            hint = "，可先用预置语料开训（scripts/fetch_seed.py）" if builtin.get("available") else ""
            raise StoreError(
                f"语料不足：至少 {settings.MIN_POS} 条正样本 + {settings.MIN_NEG} 条负样本"
                f"（当前 {counts['pos']} / {counts['neg']}{hint}）")

    # head 的底座是 openWakeWord 预训练模型，不再借用本机的 cnn
    if body.model == "head":
        from trainer.backbone import available as backbone_available
        if not backbone_available():
            raise StoreError(
                "预训练底座文件缺失（models/backbone/openwakeword/），"
                "先跑 python scripts/fetch_seed.py 再训 head", 409)

    return {
        "model": body.model,
        "epochs": body.epochs,
        "batch": body.batch,
        "lr": float(body.lr),
        "val_ratio": float(body.val_ratio),
        "seed": body.seed,          # 只决定考卷（页面固定默认值）
        "max_pos": int(body.max_pos),
        "max_neg": int(body.max_neg),
        # 考卷口径：真人只做考卷（老 run 的 config 没有这个字段，rerun 时会补）
        "split_by": "human",
        # 初始化起点默认随机：同一份声音 + 同一个 seed 训两遍会逐位相同，
        # 孩子会以为"根本没重新训练"。random 起点让每次都能看到曲线不一样。
        "init_seed": body.init_seed if body.init_seed is not None else secrets.randbelow(2 ** 31),
        "use_builtin": bool(body.use_builtin),
        # 自定义唤醒词：老 run 没有这个字段 = 默认词，前端据此决定模型名前缀
        "wake_word": wake,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


@app.post("/api/runs")
def create_run(body: RunIn):
    config = _validate_run(body)
    run_id = jobs.start(config)
    return {"id": run_id, "config": config}


@app.get("/api/runs")
def get_runs():
    return {"runs": store.list_runs(), "active": store.get_active()}


def _run_detail(run_id: str) -> dict:
    cfg = store.read_config(run_id)
    status = store.read_status(run_id)
    run_path = store.run_dir(run_id)
    log = jobs._tail(run_path / "train.log", 8000)
    return {
        "id": run_id,
        "config": cfg,
        "status": status,
        "result": store.read_result(run_id),
        "history": store.read_history(run_id),
        "log": log,
        "active": store.get_active() == run_id,
        "has_ckpt": (run_path / "model.pt").is_file(),
    }


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return _run_detail(run_id)


@app.get("/api/runs/{run_id}/stream")
def stream_run(run_id: str):
    store.read_config(run_id)                     # 存在性校验 → 404
    return StreamingResponse(
        jobs.stream(run_id), media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    store.read_config(run_id)
    jobs.stop(run_id)
    return {"ok": True}


@app.post("/api/runs/{run_id}/apply")
def apply_run(run_id: str):
    store.read_config(run_id)
    status = store.read_status(run_id)
    if status.get("state") != "succeeded":
        raise StoreError("只有训练成功的运行才能应用", 409)
    if not (store.run_dir(run_id) / "model.pt").is_file():
        raise StoreError("这次运行没有保存权重（model.pt 不存在）", 409)
    store.set_active(run_id)
    return {"ok": True, "active": run_id}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    """删掉一次运行 = 删掉一个模型（权重、曲线、日志整目录）。

    执行中的任务先拦：子进程还在往目录里写，删了它会立刻把目录重建出来。
    删的若是在用的模型，active 指针一并清掉，再喊一声试试会提示"还没有模型"。
    """
    store.read_config(run_id)
    if jobs.is_running(run_id):
        raise StoreError("这次训练还在跑，先停下再删", 409)
    store.delete_run(run_id)
    predictor.forget(run_id)                      # 内存里那份权重也别留着
    return {"runs": store.list_runs(), "active": store.get_active()}


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str):
    """导出部署包（桌宠 `./run.sh pull` 或浏览器直接下载都吃这个 zip）。"""
    from . import export as export_mod

    data, name = export_mod.build_package(run_id)
    return Response(
        content=data, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"',
                 "Cache-Control": "no-store"},
    )


class RerunIn(BaseModel):
    """重跑时允许覆盖的设置：模型卡、两个滑杆、以及当前数据集（唤醒词）。

    其余（epochs / batch / lr）仍照抄原 run，保证曲线可比。
    `default_word` 用来显式表示"这次用默认词"——否则 `wake_word_id=None`
    和"没传"分不开，学生切回默认数据集后重跑还会训旧的自定义词。
    """
    model: str | None = None
    max_pos: int | None = None
    max_neg: int | None = None
    wake_word_id: str | None = None
    default_word: bool = False


@app.post("/api/runs/{run_id}/rerun")
def rerun(run_id: str, body: RerunIn | None = None):
    """"再学一遍"：配置照抄，但换一个随机起点。

    考卷（`seed` + `val_ratio` + `split_by`）必须留着——同一张卷才比得出"这次是不是更好了"；
    初始化种子另给一个随机的，否则同一份数据、同一个 seed 训出来逐位相同。
    """
    over = body or RerunIn()
    if over.model is not None and over.model not in settings.MODELS:
        raise StoreError("模型只能是 simple / cnn / head")
    cfg = store.read_config(run_id)
    wake = cfg.get("wake_word") or {}
    if over.wake_word_id:
        wake_id = over.wake_word_id          # 学生切到了另一个词的数据集
    elif over.default_word:
        wake_id = None                       # 学生切回了默认数据集
    else:
        wake_id = wake.get("id")             # 没动数据集：沿用原 run 的词
    run_body = RunIn(model=over.model or cfg.get("model", "cnn"),
                     epochs=int(cfg.get("epochs", 30)), batch=int(cfg.get("batch", 16)),
                     lr=float(cfg.get("lr", 1e-3)),
                     val_ratio=float(cfg.get("val_ratio", 0.12)),
                     seed=int(cfg.get("seed", 42)),
                     max_pos=int(over.max_pos if over.max_pos is not None
                                 else cfg.get("max_pos", settings.TRAIN_MAX_POS)),
                     max_neg=int(over.max_neg if over.max_neg is not None
                                 else cfg.get("max_neg", settings.TRAIN_MAX_NEG)),
                     init_seed=secrets.randbelow(2 ** 31),
                     use_builtin=bool(cfg.get("use_builtin", True)),
                     wake_word_id=wake_id)
    new_cfg = _validate_run(run_body)
    # 老 run（上线前）用的是 clip 随机考卷：重跑要沿用同一张卷的口径，否则 val_acc 不可比
    new_cfg["split_by"] = cfg.get("split_by", "clip")
    new_id = jobs.start(new_cfg)
    return {"id": new_id, "config": new_cfg}


# ---------------------------------------------------------------- 推理
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise AudioError(f"文件超过 {settings.MAX_UPLOAD_BYTES // 1024} KB")
    pcm, duration = decode_wav(data, min_s=settings.MIN_PREDICT_S,
                               max_s=settings.MAX_CLIP_S)
    result = predictor.predict(pcm)
    result["duration"] = round(duration, 2)
    return result


@app.get("/healthz")
def healthz():
    return {"ok": True, "running": len(jobs.running_jobs()),
            "runs": len(store.list_runs()),
            "seed_roots": seed.roots()}
