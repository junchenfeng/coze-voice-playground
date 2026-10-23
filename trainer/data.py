"""wav 读写（只用标准库 wave）+ 数据集装载 + 滑窗切分。

训练样本的切法刻意和测试台一致：1 s 窗口、按与唤醒词的重叠比例给软标签
（≥0.9 正、0.3~0.9 半正、<0.3 负）。这样模型见过"只有半个词"的图，
而不是只在整段居中裁 1 s 的干净样本上训练。
"""
import glob
import json
import os
import wave

import numpy as np

from .features import SR, logmel

WIN = SR                    # 1 s 窗口，与测试台一致
PAD = int(SR * 0.75)        # 两侧留白，够窗口滑到唤醒词外面
POS_OFFSETS = (-600, -450, -300, -150, 0, 150, 300, 450, 600)   # ms
NEG_OFFSETS = (0, -300, 300)
# 自定义唤醒词的正例文件名前缀：和共享池的 `pos/`、内置的 `builtin/` 区分开，
# 切考卷时 groups 必须唯一，否则两个目录里的同名 wav 会被当成同一条录音。
WORD_PREFIX = "word/"
# 合成片段的名字里一律带 `-synth-`（共享池 / 词目录 / 内置语料同一套规则），
# 训练侧据此区分"真人录音"与"TTS 合成"：真人只做考卷，训练集只用合成。
SYNTH_MARK = "-synth-"
# 训练装载上限：合成数据可能很多，超过就按固定种子抽样（真人优先占名额）。
MAX_POS_CLIPS = 50
MAX_NEG_CLIPS = 300


def is_synth(name: str) -> bool:
    return SYNTH_MARK in os.path.basename(name or "")


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: 需要 16bit 单声道 wav")
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sr != SR:
        raise ValueError(f"{path}: 采样率 {sr} != {SR}（合成时就要指定 16000）")
    return pcm


def write_wav(path, pcm_bytes, sr=SR):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm_bytes)


# ---------------------------------------------------------------- 滑窗

def speech_span(pcm, rel=0.08, pad_ms=50):
    """能量法找说话段，返回 (start, end) 样本下标。"""
    if len(pcm) < 400:
        return 0, len(pcm)
    n = len(pcm) // 160
    frames = pcm[:n * 160].reshape(n, 160)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    idx = np.flatnonzero(rms > rel * rms.max())
    if len(idx) == 0:
        return 0, len(pcm)
    pad = int(SR * pad_ms / 1000)
    return max(0, idx[0] * 160 - pad), min(len(pcm), (idx[-1] + 1) * 160 + pad)


# 音量兜底：和 web 上传门限（settings.MIN_CLIP_RMS_DB / MIN_VOICED_S）保持一致。
# 上传门限只能挡住以后的录音，历史上已经落盘的空录音要在这里被丢掉——
# 否则 speech_span 的相对阈值照样能在底噪里"找到说话段"，切出一堆假正样本。
MIN_SPEECH_DB = -45.0
MIN_VOICED_S = 0.25
VOICE_GATE_DB = -38.0


def speech_stats(pcm, sr=SR):
    """(全片 RMS dBFS, 有效语音秒数)。25 ms 窗 / 10 ms 跳，超过 -38 dBFS 算说话。"""
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.size == 0:
        return -120.0, 0.0
    rms_db = 20.0 * np.log10(max(float(np.sqrt(np.mean(pcm ** 2) + 1e-12)), 1e-9))
    win, hop = int(sr * 0.025), int(sr * 0.01)
    if len(pcm) < win:
        return rms_db, 0.0
    idx = np.arange(win)[None, :] + hop * np.arange(1 + (len(pcm) - win) // hop)[:, None]
    fr = np.sqrt((pcm[idx] ** 2).mean(axis=1) + 1e-12)
    voiced = float(np.mean(fr > 10 ** (VOICE_GATE_DB / 20.0))) * (hop / sr) * len(fr)
    return rms_db, voiced


def has_speech(pcm):
    """这条录音值不值得进训练（太安静 / 只有很短一点声音都不值得）。"""
    rms_db, voiced = speech_stats(pcm)
    return rms_db >= MIN_SPEECH_DB and voiced >= MIN_VOICED_S


def _word_region(span, hard_cap=1.3):
    """长录音里唤醒词的位置无法精确定位：只取说话段正中的 1.3 s 当"词"。

    短录音（≤1.3 s）直接用整段说话段，这正是教案里学生按一下说一句的场景。
    """
    a, b = span
    if (b - a) <= hard_cap * SR:
        return a, b
    mid = (a + b) // 2
    half = int(hard_cap * SR / 2)
    return max(a, mid - half), min(b, mid + half)


def make_windows(pcm, label, span=None):
    """一条录音 → [(1 s 窗口, 软标签)]。

    软标签 = 窗口覆盖"词区"的比例：≥0.9 记 1，0.3~0.9 记 0.5，其余记 0。
    """
    span = span or speech_span(pcm)
    region = _word_region(span)
    canvas = np.concatenate([np.zeros(PAD, dtype=np.float32), pcm,
                             np.zeros(PAD, dtype=np.float32)])
    a, b = region[0] + PAD, region[1] + PAD
    center = (a + b) // 2
    base = center - WIN // 2
    denom = float(max(1, min(b - a, WIN)))
    offsets = POS_OFFSETS if label else NEG_OFFSETS

    out = []
    for ms in offsets:
        st = base + int(SR * ms / 1000)
        st = max(0, min(len(canvas) - WIN, st))
        win = canvas[st:st + WIN]
        cov = max(0, min(b, st + WIN) - max(a, st)) / denom
        target = 1.0 if (label and cov >= 0.9) else (0.5 if (label and cov >= 0.3) else 0.0)
        out.append((win, target))
    return out


def _featurize_for(model):
    if model == "head":
        from .backbone import get_features
        return get_features().embed
    return logmel


def load_dataset(root, extra_roots=(), model="mlp", expand=True, pos_dir=None,
                 max_pos=MAX_POS_CLIPS, max_neg=MAX_NEG_CLIPS, sample_seed=0,
                 stats=None):
    """root/{pos,neg}/*.wav + 可选 meta.json → (X, y, t, files, groups, meta)

    - y 是硬标签（0/1），只用来分层切考卷；
    - t 是软标签（0 / 0.5 / 1），训练损失用；
    - files 是窗口级唯一名（`<原文件>#k`），groups 是它来自哪条录音——
      切分必须按 groups 走，否则同一条录音的 9 个窗口会横跨训练和考卷。

    extra_roots 是内置合成语料，文件名带 `<basename>/` 前缀，避免与主池重名。

    pos_dir 是自定义唤醒词的正例目录（读 `<pos_dir>/pos/*.wav`）。给了它就走
    「新词口径」：**正例只来自这个目录**，root 与 extra_roots 下的 pos/neg 一律
    按负例切窗——老唤醒词的正例对新词来说不是正例，而是最像的难负例（不给它
    标负，新模型照样会被老唤醒词叫醒）。缺省 None 时行为与以前完全一致。

    max_pos / max_neg 是每个类别的**条数上限**（不是窗口数）：合成语料多起来以后
    训练会越跑越慢，超过上限就在读 wav 之前按条抽样。真人录音优先占名额（全留），
    名额不够时才从合成片段里按 sample_seed 固定随机抽——同一份语料、同一个种子
    每次都抽到同一批，rerun 与 split 缓存都稳定。

    stats 传一个 dict 进来，会写入抽样明细（before/after/limit/human/synth）与
    `human_groups`（真人录音的文件名集合，给 make_split(split_by="human") 用）。
    """
    if stats is None:
        stats = {}
    roots = [(root, "")]
    for extra in extra_roots:
        if not extra:
            continue
        extra = str(extra)
        if not os.path.isdir(extra) or os.path.abspath(extra) == os.path.abspath(root):
            continue
        roots.append((extra, os.path.basename(os.path.normpath(extra)) + "/"))

    featurize = _featurize_for(model)
    meta, X, y, t, files, groups = {}, [], [], [], [], []
    skipped = []
    local_cache: dict[str, dict] = {}

    def _local(base: str) -> dict:
        """每个 root 的 meta.json 只读一次（speaker 字段给 split_by=speaker 用）。"""
        if base not in local_cache:
            rows = {}
            meta_path = os.path.join(base, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    for row in json.load(f):
                        rows[row["file"]] = row
            local_cache[base] = rows
        return local_cache[base]

    if pos_dir:
        # 顺序与老口径对齐：每个 root 先 pos 再 neg，保证老 run 的窗口顺序不变
        sources = [(base, prefix, sub, 0) for base, prefix in roots for sub in ("pos", "neg")]
        sources.append((str(pos_dir), WORD_PREFIX, "pos", 1))
    else:
        sources = [(base, prefix, sub, lab) for base, prefix in roots
                   for lab, sub in ((1, "pos"), (0, "neg"))]

    # 先把「条」收集起来（不读音频），抽样完再读：30 条上限时没必要解码 300 条。
    entries = []       # (序号, label, 绝对路径, 窗口级唯一名, base, rel)
    for base, prefix, sub, label in sources:
        paths = sorted(glob.glob(os.path.join(base, sub, "*.wav")))
        if not paths:
            # 内置语料允许缺一类；主池也允许为空——只要带了内置语料就能开训
            # （网站"自由态"允许不录音直接练，此时 data/corpus 是空的）
            # 新词口径下所有来源都允许为空：学生可能一条老正例都没有。
            if pos_dir or prefix or extra_roots:
                continue
            raise SystemExit(f"{base}/{sub} 下没有 wav")
        for p in paths:
            rel = os.path.relpath(p, base)
            entries.append((len(entries), label, p, prefix + rel, base, rel))

    def _sample(rows, cap, key):
        """每类按条抽样：真人全留，名额不够才抽合成；真人自己超上限才截断。"""
        cap = max(1, int(cap))            # 0 / 负数没有意义，至少留一条
        if not rows:
            stats[key] = {"before": 0, "after": 0, "limit": cap, "human": 0, "synth": 0}
            return rows
        human = [e for e in rows if not is_synth(e[3])]
        synth = sorted((e for e in rows if is_synth(e[3])), key=lambda e: e[3])
        if len(rows) <= cap:
            kept = rows
        else:
            if len(human) >= cap:
                rng = np.random.default_rng(sample_seed)
                idx = set(rng.choice(len(human), size=cap, replace=False).tolist())
                kept_human = [e for i, e in enumerate(human) if i in idx]
            else:
                kept_human = human
            room = cap - len(kept_human)
            if room > 0 and synth:
                rng = np.random.default_rng(sample_seed)
                idx = set(rng.choice(len(synth), size=min(room, len(synth)),
                                     replace=False).tolist())
                kept = kept_human + [e for i, e in enumerate(synth) if i in idx]
            else:
                kept = kept_human
            kept.sort(key=lambda e: e[0])          # 保持原来的语料顺序，窗口顺序稳定
        stats[key] = {"before": len(rows), "after": len(kept), "limit": cap,
                      "human": sum(1 for e in kept if not is_synth(e[3])),
                      "synth": sum(1 for e in kept if is_synth(e[3]))}
        return kept

    pos_rows = _sample([e for e in entries if e[1] == 1], int(max_pos), "pos")
    neg_rows = _sample([e for e in entries if e[1] == 0], int(max_neg), "neg")
    kept_entries = sorted(pos_rows + neg_rows, key=lambda e: e[0])
    stats["human_groups"] = sorted({e[3] for e in kept_entries if not is_synth(e[3])})
    for key, zh in (("pos", "正例"), ("neg", "负例")):
        row = stats[key]
        if row["before"] > row["after"]:
            print(f"[data] 抽样：{zh} {row['before']}→{row['after']}"
                  f"（真人 {row['human']} 条保留、合成保留 {row['synth']} 条，"
                  f"上限 {row['limit']}）")

    for _idx, label, p, name, base, rel in kept_entries:
        local = _local(base)
        pcm = read_wav(p)
        if not has_speech(pcm):
            skipped.append(name)          # 空录音 / 底噪：不进训练，也不占考卷
            continue
        wins = make_windows(pcm, label) if expand else [(pcm[:WIN], float(label))]
        for k, (win, target) in enumerate(wins):
            X.append(featurize(win))
            y.append(label)
            t.append(target)
            files.append(f"{name}#{k}")
            groups.append(name)
        if rel in local:
            meta[name] = {**local[rel], "file": name}
    if skipped:
        head = "、".join(skipped[:4])
        more = "" if len(skipped) <= 4 else f" 等 {len(skipped)} 条"
        print(f"[data] 跳过 {len(skipped)} 条几乎没有声音的录音：{head}{more}")
    quiet = f"（{len(skipped)} 条都因为太安静被跳过）" if skipped else ""
    # 新词口径下负例永远有（共享池/内置），光看 X 是否为空会放过"一条正例都没有"：
    # 那样训出来的是"永远不醒"的模型，必须当错误报出来。
    if pos_dir and 1 not in y:
        raise SystemExit(f"新唤醒词还没有可用正例：{pos_dir}/pos 下没有 wav{quiet}")
    if not X:
        raise SystemExit(f"{root} 下没有可用的 wav{quiet}")
    return (np.stack(X), np.array(y, dtype=np.int64), np.array(t, dtype=np.float32),
            files, groups, meta)


def make_split(labels, files, meta=None, val_ratio=0.12, seed=42, out=None,
               split_by="clip", groups=None, human_groups=None):
    """切考卷；结果落盘，保证三个模型考同一张卷。

    split_by="clip"    随机抽条（同一个人的其它录音仍可能在训练里）
    split_by="speaker" 留出若干"人"，他们一条都不参与训练（真·没见过的题）
    split_by="human"   真人录音全部当考卷、训练集只用合成片段；某类没有可用真人
                       或这样会让该类训练侧空掉时，该类别回落 clip 随机 12%

    groups 是窗口所属的原始录音：同一段录音的窗口必须整组进同一侧。
    """
    groups = list(groups) if groups is not None else list(files)
    human = set(human_groups or ())
    fingerprint = {"split_by": split_by, "val_ratio": val_ratio, "seed": seed,
                   "human_n": len(human)}
    if out and os.path.exists(out) and os.path.exists(out + ".by"):
        try:
            cached = json.loads(open(out + ".by", encoding="utf-8").read())
        except json.JSONDecodeError:
            cached = None
        if cached == fingerprint:
            with open(out, encoding="utf-8") as f:
                saved = json.load(f)
            idx = {f: i for i, f in enumerate(files)}
            if all(f in idx for f in saved["val"] + saved["train"]):
                return np.array([idx[f] for f in saved["val"]], dtype=np.int64)

    rng = np.random.default_rng(seed)
    g_label = {}
    for g, lab in zip(groups, labels):
        g_label[g] = int(lab)
    val_groups: set[str] = set()
    if split_by == "speaker" and meta:
        speakers = sorted({meta[g]["speaker"] for g in g_label if g in meta})
        n_hold = max(1, int(round(len(speakers) * val_ratio)))
        held = set(rng.choice(speakers, size=n_hold, replace=False).tolist())
        val_groups = {g for g in g_label if meta.get(g, {}).get("speaker") in held}
    elif split_by == "human":
        for lab in (0, 1):
            pool = sorted(g for g, v in g_label.items() if v == lab)
            if not pool:
                continue
            mine = [g for g in pool if g in human]
            if mine and len(mine) < len(pool):
                val_groups.update(mine)          # 真人全进考卷，合成全留训练
                continue
            # 该类没有真人（或全是真人）：回落随机抽，保证训练侧不空
            k = max(1, int(round(len(pool) * val_ratio)))
            val_groups.update(rng.choice(pool, size=min(k, len(pool)),
                                         replace=False).tolist())
            print(f"[data] split: 类别 {lab} 没有可用的真人考卷，回落随机 {val_ratio:.0%}")
    else:
        for lab in (0, 1):
            pool = sorted(g for g, v in g_label.items() if v == lab)
            k = max(1, int(round(len(pool) * val_ratio)))
            val_groups.update(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist())

    val = np.array([i for i, g in enumerate(groups) if g in val_groups], dtype=np.int64)

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        val_set = set(val_groups)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"val": [files[i] for i in val],
                       "train": [f for i, f in enumerate(files) if groups[i] not in val_set],
                       "val_groups": sorted(val_set)},
                      f, ensure_ascii=False, indent=1)
        with open(out + ".by", "w", encoding="utf-8") as f:
            json.dump(fingerprint, f)
    return val
