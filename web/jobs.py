"""训练任务：子进程跑 `python -m trainer.train`，progress.jsonl 尾随给 SSE。

为什么用子进程而不是线程里的 torch：
- 训练崩了不会拖垮 web 进程；
- 「暂停」就是 SIGTERM，干净利落；
- 和 `docker compose run trainer` 共用同一份入口。

数据分三层（见 web/seed.py）：
- `--data data/corpus`            麦克风录音（用户自己的，唯一留在 data/ 的音频）
- `--builtin-dir .cache/seed/builtin`  预置语料（OSS 清单驱动的本地缓存）
- `--extra-dir .cache/seed/corpus`     历史上的人工语料（同上，按「额外根」接入）
"""
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time

from . import seed, settings, store, words
from .store import StoreError

_registry: dict[str, dict] = {}
_lock = threading.Lock()

# 训练要用的预置语料根：builtin（内置合成）+ corpus（历史人工语料）
SEED_ROOTS = ("builtin", "corpus")


# ---------------------------------------------------------------- 启动 / 停止
def running_jobs() -> list[str]:
    with _lock:
        return [key for key, job in _registry.items() if job["proc"].poll() is None]


def is_running(run_id: str) -> bool:
    """这个 run 的训练子进程还活着吗（删模型前要拦）。"""
    with _lock:
        job = _registry.get(run_id)
        return bool(job and job["proc"].poll() is None)


def _build_cmd(run_id: str, config: dict) -> list[str]:
    run_path = store.run_dir(run_id)
    cmd = [
        sys.executable, "-m", "trainer.train",
        "--data", str(settings.CORPUS_ROOT),
        "--model", config["model"],
        "--epochs", str(config["epochs"]),
        "--batch", str(config["batch"]),
        "--lr", repr(float(config["lr"])),
        "--val-ratio", str(config["val_ratio"]),
        "--seed", str(config["seed"]),
        # 考卷口径：human = 训练时留出的那批真人录音（某类没真人时 trainer 回落 clip）。
        # 老 config 没有这个字段时才按 clip 走；页面上的「再学一遍」会显式写 human。
        "--split-by", str(config.get("split_by", "clip")),
        "--max-pos", str(int(config.get("max_pos", settings.TRAIN_MAX_POS))),
        "--max-neg", str(int(config.get("max_neg", settings.TRAIN_MAX_NEG))),
        # 真人进训练集的配额（其余当固定考卷）；老 config 没有就兜默认值
        "--human-train-pos",
        str(int(config.get("human_train_pos", settings.HUMAN_TRAIN_POS))),
        "--human-train-neg",
        str(int(config.get("human_train_neg", settings.HUMAN_TRAIN_NEG))),
        "--threads", str(settings.TRAIN_THREADS),
        "--out", str(run_path / "result.json"),
        "--save-ckpt", str(run_path / "model.pt"),
        "--progress-jsonl", str(run_path / "progress.jsonl"),
        "--split-json", str(run_path / "split.json"),
    ]
    if config.get("init_seed") is not None:
        cmd += ["--init-seed", str(config["init_seed"])]

    # 预置语料：缓存里没有就当场不传（页面另行提示「先跑 fetch_seed.py」），
    # 绝不能因为一次网络抖动让训练直接失败。
    builtin_dir = seed.local_root("builtin")
    if config.get("use_builtin", True) and builtin_dir.is_dir():
        cmd += ["--builtin-dir", str(builtin_dir)]
    elif not config.get("use_builtin", True):
        cmd += ["--no-builtin"]
    corpus_seed = seed.local_root("corpus")
    if corpus_seed.is_dir():
        cmd += ["--extra-dir", str(corpus_seed)]

    # 自定义唤醒词：正例只从词目录来，上面的 pos 全部改标负例
    wake = config.get("wake_word") or {}
    if wake.get("id"):
        cmd += ["--pos-dir", str(words.word_dir(wake["id"]))]
        if wake.get("text"):
            cmd += ["--wake-word", str(wake["text"])]
    return cmd


def _prefetch() -> list[str]:
    """训练前把清单里的种子音频补齐（已在本地就只做 stat，不读 sha256）。

    补下载有总预算（settings.SEED_PREFETCH_BUDGET_S）：本沙箱会掐 Python socket
    的下载，缺一条就是一次满超时，绝不能让「点一下教它」等上几十分钟。
    超预算就停手——训练继续用本地已有的，缺什么在 warnings 里说清楚。
    """
    warnings = []
    for root in SEED_ROOTS:
        if not seed.files([root]):
            continue
        try:
            stats = seed.ensure_local([root], verify=False, download=True,
                                      budget_s=settings.SEED_PREFETCH_BUDGET_S)
        except seed.SeedError as exc:
            warnings.append(f"清单读不出来：{exc}")
            continue
        if stats["skipped"]:
            warnings.append(f"种子语料 {root}: {stats['skipped']} 条补下载超预算，"
                            "训练先用本地已有的（可稍后跑 scripts/fetch_seed.py 补齐）")
        warnings += [f"种子语料 {root}: {w}" for w in stats["warnings"][:3]]
        if stats["failed"]:
            warnings.append(f"种子语料 {root}: {stats['failed']} 条下载失败，训练只用本地已有的")
    return warnings


def start(config: dict) -> str:
    """创建 run 目录并拉起子进程；返回 run_id。

    容量检查 + 建目录 + Popen + 登记在同一个锁内完成，避免两个请求都以为自己
    抢到了唯一的训练槽位。
    """
    with _lock:
        if any(job["proc"].poll() is None for job in _registry.values()):
            raise StoreError("服务器同一时间只跑一个训练，请等当前任务结束", 409)

        # 清掉一小时前就结束的句柄，长期运行不攒内存
        for key, job in list(_registry.items()):
            if job["proc"].poll() is not None and time.time() - job["started_at"] > 3600:
                _registry.pop(key, None)

        run_id = store.create_run(config)
        run_path = store.run_dir(run_id)
        (run_path / "progress.jsonl").write_text("", encoding="utf-8")

    # 下载放到锁外：可能几秒，不能把「查询运行状态」的请求一起卡住
    warnings = _prefetch()
    status = store.read_status(run_id)
    status.update({"state": "preparing", "warnings": warnings})
    store.write_status(run_id, status)

    log_path = run_path / "train.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = str(settings.TRAIN_THREADS)
    env["MKL_NUM_THREADS"] = str(settings.TRAIN_THREADS)
    # FaaS 形态下 BASE_DIR 只读：trainer 侧的特征缓存默认写 BASE_DIR/.cache/features
    # 会 Errno 30，这里统一指到可写缓存根；底座 onnx 同步对齐 settings 的口径
    #（trainer/backbone.py 读的是 NOVA_OWW_DIR，变量名与 web 侧的 NOVA_BACKBONE_DIR 不同）。
    env.setdefault("NOVA_FEATURE_CACHE", str(settings.CACHE_ROOT / "features"))
    env.setdefault("NOVA_OWW_DIR", str(settings.BACKBONE_DIR))

    try:
        with open(log_path, "wb") as log_fp:
            proc = subprocess.Popen(
                _build_cmd(run_id, config),
                cwd=str(settings.BASE_DIR),
                stdout=log_fp, stderr=subprocess.STDOUT,
                env=env, start_new_session=True,
            )
    except OSError as exc:
        status.update({"state": "failed", "error_tail": f"启动训练进程失败：{exc}",
                       "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        store.write_status(run_id, status)
        raise StoreError(f"启动训练进程失败：{exc}", 500) from exc

    with _lock:
        _registry[run_id] = {"proc": proc, "stop": False,
                             "started_at": time.time(), "log": str(log_path)}

    status.update({"state": "running", "pid": proc.pid,
                   "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    store.write_status(run_id, status)
    threading.Thread(target=_supervise, args=(run_id, proc),
                     daemon=True, name=f"train-{run_id}").start()
    return run_id


def _supervise(run_id: str, proc: subprocess.Popen) -> None:
    deadline = time.time() + settings.TRAIN_TIMEOUT_S
    while proc.poll() is None:
        if time.time() > deadline:
            _terminate(proc)
            break
        time.sleep(1.0)
    code = proc.poll()
    if code is None:
        code = -9

    with _lock:
        job = _registry.get(run_id) or {}
        stopped = bool(job.get("stop"))
    if stopped:
        state = "stopped"
    elif code == 0:
        state = "succeeded"
    else:
        state = "failed"

    status = store.read_status(run_id) or {}
    status.update({
        "state": state,
        "exit_code": code,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "error_tail": _tail(store.run_dir(run_id) / "train.log"),
    })
    result = store.read_result(run_id)
    if result:
        status["final"] = result.get("final")
        status["elapsed_s"] = result.get("elapsed_s")
        status["params_total"] = result.get("params_total")
        status["params_trainable"] = result.get("params_trainable")
    store.write_status(run_id, status)


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), sig)
        else:
            proc.send_signal(sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except OSError:
            pass


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _signal_group(proc, signal.SIGKILL)


def stop(run_id: str) -> None:
    with _lock:
        job = _registry.get(run_id)
        if not job:
            raise StoreError("这个训练不在运行中", 409)
        job["stop"] = True
    _terminate(job["proc"])


def stop_all() -> None:
    with _lock:
        jobs = list(_registry.values())
    for job in jobs:
        job["stop"] = True
        _terminate(job["proc"])


# ---------------------------------------------------------------- 重启自愈
def reconcile() -> None:
    """web 重启后把残留的 running / preparing 状态收敛掉，避免 UI 永久转圈。"""
    try:
        runs = store.list_runs()
    except StoreError:
        return
    for row in runs:
        if row["state"] not in ("running", "queued", "preparing"):
            continue
        try:
            status = store.read_status(row["id"])
        except StoreError:
            continue
        pid = status.get("pid")
        # web 重启后进程句柄丢了，无法再暂停/收结果：如果是本项目的训练子进程就收掉；
        # 其它情况只收敛状态，避免误杀复用了同一 PID 的无关进程。
        if isinstance(pid, int) and _pid_alive(pid) and _looks_like_training(pid):
            _terminate_pid(pid)
        status.update({"state": "interrupted",
                       "error_tail": status.get("error_tail") or "服务重启，训练已中断",
                       "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        store.write_status(row["id"], status)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OverflowError):
        return False


def _looks_like_training(pid: int) -> bool:
    """只在 Linux 上校验 /proc/<pid>/cmdline，其它平台一律不杀。"""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"trainer.train" in f.read()
    except OSError:
        return False


def _terminate_pid(pid: int) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


# ---------------------------------------------------------------- SSE
def _tail(path, limit: int | None = None) -> str:
    limit = limit or settings.ERROR_TAIL_BYTES
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _new_lines(path, offset: int) -> tuple[list[dict], int]:
    """从 offset 读到文件尾，只消费完整行（最后半行留到下次）。"""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
    except OSError:
        return [], offset
    if not chunk:
        return [], offset
    end = chunk.rfind(b"\n")
    if end < 0:
        return [], offset
    rows = []
    for line in chunk[:end].split(b"\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows, offset + end + 1


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream(run_id: str):
    """逐 epoch 推送；结束/失败后补一条 status 事件再关闭。"""
    run_path = store.run_dir(run_id)
    progress = run_path / "progress.jsonl"
    offset = 0
    last_beat = time.time()

    while True:
        rows, offset = _new_lines(progress, offset)
        for row in rows:
            yield _sse("epoch", row)

        status = store.read_status(run_id) or {}
        state = status.get("state", "unknown")
        if state not in ("running", "queued", "preparing"):
            rows, offset = _new_lines(progress, offset)
            for row in rows:
                yield _sse("epoch", row)
            payload = {k: status.get(k) for k in
                       ("state", "exit_code", "error_tail", "final", "elapsed_s",
                        "params_total", "params_trainable", "finished_at", "warnings")}
            yield _sse("status", payload)
            return

        now = time.time()
        if now - last_beat > settings.SSE_HEARTBEAT_S:
            last_beat = now
            yield ": ping\n\n"
        await asyncio.sleep(settings.SSE_POLL_S)
