"""active 模型的加载与推理。

测试页每 300ms 打一次请求，绝不能每次都 torch.load：按 (run_id, mtime) 缓存，
权重换文件才失效。推理串行化，避免高频请求叠在一起抢线程。
单数据域下缓存键就是 run_id——不存在"哪个工作区的权重"。
"""
import threading

from . import store
from .store import StoreError

_cache: dict[str, dict] = {}
_lock = threading.Lock()


def active_info() -> dict:
    run_id = store.get_active()
    if not run_id:
        raise StoreError("还没有应用任何模型，先去训练台点「应用」", 409)
    run_path = store.run_dir(run_id)
    ckpt = run_path / "model.pt"
    if not ckpt.is_file():
        raise StoreError("当前模型的权重文件不见了，请重新训练或换一个运行", 409)
    cfg = store.read_config(run_id)
    return {"run_id": run_id, "model": cfg.get("model", "cnn"),
            "ckpt": ckpt, "config": cfg}


def _load(info: dict):
    key = info["run_id"]
    mtime = info["ckpt"].stat().st_mtime_ns
    entry = _cache.get(key)
    if entry and entry["mtime"] == mtime:
        return entry["model"]

    # 延迟 import：没装 torch 的环境（例如只跑接口测试）也能 import web.app
    import torch

    from trainer.backbone import BackboneError
    from trainer.infer import load_model

    torch.set_num_threads(1)
    try:
        model = load_model(str(info["ckpt"]), info["model"])
    except BackboneError as exc:
        raise StoreError(str(exc), 409) from exc
    except RuntimeError as exc:
        raise StoreError(str(exc), 409) from exc
    # 只留最近几次应用过的模型，避免来回切换时内存一路涨
    if len(_cache) >= 8:
        for old in [k for k in _cache if k != key][:len(_cache) - 7]:
            _cache.pop(old, None)
    _cache[key] = {"mtime": mtime, "model": model}
    return model


def forget(run_id: str) -> None:
    """运行被删掉后把缓存条目摘掉：别让内存里还留着一份能推理的权重。"""
    with _lock:
        _cache.pop(run_id, None)


def predict(pcm) -> dict:
    info = active_info()
    with _lock:
        model = _load(info)
        import torch

        from trainer.infer import predict_pcm

        with torch.no_grad():
            p_wake = predict_pcm(model, pcm, info["model"])
    return {"p_wake": round(float(p_wake), 4), "run_id": info["run_id"],
            "model": info["model"]}
