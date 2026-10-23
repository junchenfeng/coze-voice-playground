"""部署包：把 CNN 或 HEAD 训练结果打包给桌宠。

包只包含小型分类器权重；HEAD 的 ONNX 底座通过 OSS 清单下载：
    buddy-wake-<run_id>.zip
    ├── np_cnn.py / np_head.py    纯 numpy 前向（简单 CNN / 复杂 CNN 共用一份，与 trainer/np_cnn.py 逐字节相同）
    ├── model.npz    float32 权重
    └── meta.json    特征参数 / 标定好的阈值 / sha256 / 自检向量 / 唤醒词

两件容易翻车的事，这里都处理了：
1. **阈值不能沿用 0.70/0.85**。那是旧 head 模型调出来的；同一份语料上 CNN 用
   0.70/0.85 的误报率约为 head 的 3 倍。导出时按生产滑窗口径（1 s 窗 / 300 ms
   跳 / 能量门控）重新扫一遍，按「误报 ≤6% 且召回最大」给建议值。语料就是训练
   语料，所以这个数字是**上界**，meta.basis 里写明口径。
2. **实现漂移不能靠人盯**。meta.selftest 给一个确定性噪声种子和期望概率，桌宠
   加载时自己算一遍，对不上就拒载（logmel 与网络两处都覆盖）。
"""
import hashlib
import io
import json
import math
import pathlib
import time
import zipfile

import numpy as np

from trainer import np_cnn, np_head
from trainer.data import read_wav

from . import seed, settings, store, words
from .store import StoreError

SCHEMA = 1
# 生产滑窗口径，与 desktop-buddy/tools/replay_eval.py 完全一致
EVAL_WIN, EVAL_HOP, EVAL_PAD = 16000, 4800, 8000
RMS_GATE, PEAK_GATE = 0.008, 0.02
DEFAULT_THR, DEFAULT_HIGH = 0.70, 0.85
MAX_CALIB_CLIPS = 240
MAX_HEAD_CALIB_CLIPS = 24          # 两侧各最多取这么多条：一次导出 ~1 s
SELFTEST_SEED = 1234
EXPORTABLE = ("simple", "cnn", "head")


# ---------------------------------------------------------------- 标定
def _scan(model, pcm):
    """按生产口径滑窗，返回每个窗口的概率；整条都被门控丢掉时返回 None。

    全被门控的片段（空录音/极弱）在生产里根本不可能触发，不该进召回分母——
    这与 `desktop-buddy/tools/replay_eval.py` 的"有效集"口径一致。
    """
    canvas = np.concatenate([np.zeros(EVAL_PAD, np.float32), pcm,
                             np.zeros(EVAL_PAD, np.float32)])
    out, alive = [], False
    for st in range(0, len(canvas) - EVAL_WIN + 1, EVAL_HOP):
        win = canvas[st:st + EVAL_WIN]
        rms = float(np.sqrt((win ** 2).mean()))
        peak = float(np.abs(win).max())
        if rms < RMS_GATE or peak < PEAK_GATE:
            out.append(0.0)
            continue
        alive = True
        out.append(model.predict(win))
    return out if alive else None


def _fired(probs, thr, high):
    """与桌宠 WakeDetector 同一套双阈值迟滞。"""
    prev = 0.0
    for p in probs:
        if p >= high:
            return True
        if p >= thr and prev >= thr:
            return True
        prev = p
    return False


def _load_clips(folders, limit, skip_synth=False):
    rows = []
    for folder in folders:
        path = pathlib.Path(folder)
        if not path.is_dir():
            continue
        for wav in sorted(path.glob("*.wav")):
            if len(rows) >= limit:
                return rows
            if wav.name.startswith("._"):
                continue
            if skip_synth and "-synth-" in wav.name:
                continue
            try:
                pcm = read_wav(str(wav))
            except (OSError, ValueError):
                continue
            if len(pcm) > 0:
                rows.append(pcm)
    return rows


def _corpus_dirs(label: str) -> list[pathlib.Path]:
    """默认语料的目录：麦克风录音 + 预置语料（builtin / 历史人工语料缓存）。"""
    dirs = [settings.CORPUS_ROOT / label]
    for root in ("corpus", "builtin"):
        dirs.append(seed.local_root(root) / label)
    return [d for d in dirs if d.is_dir()]


def _bases(wake, run_id=None):
    """返回 (正例目录列表, 负例目录列表)。

    自定义词：正例只认词目录（词目录没了就回落到 result.json 记的 pos_dir）；
    负例 = 默认语料的全部（别的词的正例正是最像的难负例）。
    """
    if not wake:
        return _corpus_dirs("pos"), _corpus_dirs("neg")
    wid = wake.get("id")
    pos_dirs = []
    if wid:
        try:
            pos_dirs.append(words.word_dir(wid) / "pos")
        except StoreError:
            pos_dirs = []
    if (not pos_dirs or not any(p.glob("*.wav") for p in pos_dirs)) and run_id:
        recorded = (store.read_result(run_id) or {}).get("pos_dir")
        if recorded and pathlib.Path(recorded, "pos").is_dir():
            pos_dirs = [pathlib.Path(recorded, "pos")]
    return pos_dirs, _corpus_dirs("pos") + _corpus_dirs("neg")


def thresholds(model, wake=None, run_id=None):
    """在语料上滑窗扫阈值；语料不够就退回默认值并标明 basis。"""
    pos_dirs, neg_dirs = _bases(wake, run_id)
    half = (MAX_HEAD_CALIB_CLIPS if isinstance(model, np_head.NumpyHead) else MAX_CALIB_CLIPS) // 2
    pos = _load_clips(pos_dirs, half, skip_synth=bool(wake))
    basis_extra = ""
    if wake and not pos:
        # 还没录真人正例（只合成过）：合成音也一样能标定，只是数字更乐观，标出来
        pos = _load_clips(pos_dirs, half)
        basis_extra = "(synth)"
    neg = _load_clips(neg_dirs, half)
    if not pos or not neg:
        return {"threshold": DEFAULT_THR, "threshold_high": DEFAULT_HIGH,
                "basis": "default", "n_pos_clips": len(pos), "n_neg_clips": len(neg)}

    pos_raw = [_scan(model, pcm) for pcm in pos]
    neg_raw = [_scan(model, pcm) for pcm in neg]
    pos_p = [p for p in pos_raw if p is not None]
    neg_p = [p for p in neg_raw if p is not None]
    dirty = (len(pos_raw) - len(pos_p)) + (len(neg_raw) - len(neg_p))
    if not pos_p or not neg_p:
        return {"threshold": DEFAULT_THR, "threshold_high": DEFAULT_HIGH,
                "basis": "default", "n_pos_clips": len(pos_p), "n_neg_clips": len(neg_p)}

    # 目标：**先保住召回**，误报压在 ≤6% 的片段比例内。这个权衡沿用桌宠上一轮
    # 实测结论（wake.py 注释）：把误报从 2.9% 收紧到 1.4%，主人自己的召回从 92%
    # 掉到 69% —— 喊了没反应比偶尔被误唤醒更破坏体验。都不满足时退化成
    # "召回 − 误报"最大。
    best = None
    loose = None
    for thr in [round(0.35 + 0.05 * i, 2) for i in range(10)]:        # 0.35 ~ 0.80
        for high in [round(min(thr + 0.05 * i, 0.99), 2) for i in (1, 2, 3, 4)]:
            rec = sum(1 for p in pos_p if _fired(p, thr, high)) / len(pos_p)
            fpr = sum(1 for p in neg_p if _fired(p, thr, high)) / len(neg_p)
            if fpr <= 0.06 and (best is None or rec > best[2]):
                best = (thr, high, rec, fpr)
            score = rec - fpr
            if loose is None or score > loose[4]:
                loose = (thr, high, rec, fpr, score)
    basis = "corpus-replay" + basis_extra
    if best is None:
        basis = "corpus-replay(weak)" + basis_extra
        best = loose[:4]
    thr, high, rec, fpr = best
    return {"threshold": thr, "threshold_high": high,
            "basis": basis, "recall": round(rec, 3), "fpr": round(fpr, 3),
            "n_pos_clips": len(pos_p), "n_neg_clips": len(neg_p), "n_dirty_clips": dirty}


# ---------------------------------------------------------------- 打包
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_package(run_id: str) -> tuple[bytes, str]:
    """生成 zip 字节流与下载文件名；不可导出的情况抛 409。"""
    cfg = store.read_config(run_id)
    status = store.read_status(run_id)
    if status.get("state") != "succeeded":
        raise StoreError("只有训练成功的运行才能导出", 409)
    model_name = str(cfg.get("model") or "")
    if model_name not in EXPORTABLE:
        raise StoreError("桌宠只能导出简单 CNN、复杂 CNN 或 HEAD 模型", 409)
    run_path = store.run_dir(run_id)
    ckpt = run_path / "model.pt"
    if not ckpt.is_file():
        raise StoreError("这次运行没有保存权重（model.pt 不存在）", 409)

    import torch

    from trainer.train import load_state

    from trainer import backbone

    runtime = np_head if model_name == "head" else np_cnn
    keys = np_head.WEIGHT_KEYS if model_name == "head" else np_cnn.weights_for(model_name)
    sd = load_state(str(ckpt))
    state = {k: v.detach().cpu().numpy().astype(np.float32)
             for k, v in sd.items() if k in keys}
    model = (np_head.NumpyHead(state, backbone.DEFAULT_DIR) if model_name == "head"
             else np_cnn.load_from_state(state, model_name))

    wake = cfg.get("wake_word") or None
    thr = thresholds(model, wake, run_id)

    buf = io.BytesIO()
    np.savez(buf, **state)
    npz = buf.getvalue()
    src = pathlib.Path(runtime.__file__).read_bytes()
    expect = round(model.predict(runtime.selftest_window(SELFTEST_SEED)), 6)
    runtime_name = "np_head.py" if model_name == "head" else "np_cnn.py"

    result = store.read_result(run_id) or {}
    try:
        val_acc = float((result.get("final") or {}).get("val_acc"))
    except (TypeError, ValueError):
        val_acc = None
    if val_acc is not None and not math.isfinite(val_acc):
        val_acc = None                      # NaN/Inf 会让 JSON 变成非法格式
    meta = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run_id,
        "model": model_name,
        "wake_word": wake,
        "val_acc": val_acc,
        # 标签很差时别让人装上去才发现"怎么喊都不醒"：标定召回就是最好的自检
        "warning": (f"标定召回只有 {thr['recall']:.0%}：这版模型对唤醒词很不敏感，"
                    f"建议多录几条正例（尤其真人）后重训" if thr.get("recall", 1) < 0.6 else None),
        "params": int(sum(v.size for v in state.values())),
        "feature": dict(runtime.FEATURES),
        "backbone": np_head.BACKBONE if model_name == "head" else None,
        "thresholds": thr,
        "selftest": {"seed": SELFTEST_SEED, "expect": expect},
        "sha256": {"model.npz": _sha256(npz), runtime_name: _sha256(src)},
    }

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(runtime_name, src)
        zf.writestr("model.npz", npz)
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=1))
    return out.getvalue(), f"buddy-wake-{run_id}.zip"
