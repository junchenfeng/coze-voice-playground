"""load_dataset / make_windows：滑窗口径与内置语料合并。"""
import json
import os

import numpy as np

from trainer.data import load_dataset, make_split, make_windows, write_wav
from trainer.features import SR


def _wav(path, seconds=1.0, freq=440.0):
    n = int(SR * seconds)
    x = 0.2 * np.sin(2 * np.pi * freq * np.arange(n) / SR)
    write_wav(path, (x * 32767).astype("<i2").tobytes())


def _make(root, pos=2, neg=2, meta=None):
    for label, n in (("pos", pos), ("neg", neg)):
        os.makedirs(os.path.join(root, label), exist_ok=True)
        for i in range(n):
            _wav(os.path.join(root, label, f"{label}{i}.wav"), freq=440 + i)
    if meta is not None:
        with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)


def test_extra_root_is_prefixed_and_kept_out_of_primary_names(tmp_path):
    main, builtin = tmp_path / "corpus", tmp_path / "builtin"
    _make(main, 2, 2)
    _make(builtin, 2, 2, meta=[{"file": "pos/pos0.wav", "speaker": "voiceA"},
                               {"file": "neg/neg0.wav", "speaker": "voiceA"}])

    *_, files0, groups0, _ = load_dataset(str(main))
    *_, files1, groups1, _ = load_dataset(str(builtin))

    X, y, t, files, groups, meta = load_dataset(str(main), [str(builtin)])

    assert len(files) == len(files0) + len(files1)
    assert all(g in groups0 for g in groups[:len(groups0)])       # 主池名字保持原样
    assert all(g.startswith("builtin/") for g in groups[len(groups0):])
    assert "builtin/pos/pos0.wav" in meta
    assert meta["builtin/pos/pos0.wav"]["speaker"] == "voiceA"
    assert X.shape[0] == y.shape[0] == t.shape[0] == len(files) == len(groups)
    assert set(np.unique(t)) <= {0.0, 0.5, 1.0}


def test_empty_primary_root_falls_back_to_builtin(tmp_path):
    """自由态允许一条自己的都不录：主池为空时靠内置语料也能开训。"""
    import pytest

    main, builtin = tmp_path / "corpus", tmp_path / "builtin"
    os.makedirs(os.path.join(main, "pos"), exist_ok=True)
    os.makedirs(os.path.join(main, "neg"), exist_ok=True)
    _make(builtin, 2, 2)

    X, y, t, files, groups, _ = load_dataset(str(main), [str(builtin)])
    assert len(files) > 0
    assert all(g.startswith("builtin/") for g in groups)
    assert X.shape[0] == y.shape[0] == len(files) == len(groups)
    assert set(np.unique(y)) == {0, 1}

    # 没有内置语料补位时，空主池仍然是明确错误（别静默训一个空模型）
    with pytest.raises(SystemExit):
        load_dataset(str(main))


def test_missing_extra_root_is_ignored(tmp_path):
    main = tmp_path / "corpus"
    _make(main, 1, 1)

    *_, groups, _ = load_dataset(str(main), [str(tmp_path / "nope")])
    assert len(set(groups)) == 2                    # 1 正 1 负 → 9 + 3 个窗口


def test_silent_recording_is_dropped_from_training(tmp_path, capsys):
    """底噪录音不能进训练。

    `speech_span` 用的是"比自己的最大值低 8%"这种相对阈值，空录音里照样能找到
    "说话段"，切出 9 个假正样本。所以要在读文件时用绝对电平挡掉。
    """
    root = tmp_path / "corpus"
    _make(root, 2, 2)
    n = int(SR * 2.0)
    rng = np.random.default_rng(0)
    quiet = (0.004 * rng.standard_normal(n) * 32767).astype("<i2").tobytes()
    write_wav(str(root / "pos" / "quiet.wav"), quiet)

    *_, files, groups, _ = load_dataset(str(root))

    assert not any("quiet" in g for g in groups)       # 一条窗口都不该来自它
    assert "跳过 1 条" in capsys.readouterr().out      # 但要留一行日志，别静默消失
    assert len(set(groups)) == 4                       # 剩下 pos0/1、neg0/1


def test_pos_dir_relabels_the_whole_pool_as_negatives(tmp_path):
    """新唤醒词口径：正例只来自词目录，共享池 + 内置的 pos 一律变负例。

    老唤醒词的正例对新词不是"无关样本"，而是最像的难负例——不改标就会训出
    "喊 buddy助手 也能叫醒新词"的模型。
    """
    main, builtin, word = tmp_path / "corpus", tmp_path / "builtin", tmp_path / "w-abc"
    _make(main, 2, 2)
    _make(builtin, 1, 1)
    _make(word, 3, 0)                                # 词目录只有 pos/

    X, y, t, files, groups, _ = load_dataset(
        str(main), [str(builtin)], pos_dir=str(word))

    # 老池子的正例还在（数据不丢），但标签一律是 0
    assert "pos/pos0.wav" in groups and "builtin/pos/pos0.wav" in groups
    by_group = {}
    for g, lab in zip(groups, y):
        by_group.setdefault(g, set()).add(int(lab))
    assert by_group["pos/pos0.wav"] == {0}
    assert by_group["builtin/pos/pos0.wav"] == {0}
    assert by_group["neg/neg0.wav"] == {0}

    # 正例只有词目录来的，且带 word/ 前缀（不许和共享池同名撞组）
    positives = {g for g, lab in zip(groups, y) if lab == 1}
    assert positives == {"word/pos/pos0.wav", "word/pos/pos1.wav", "word/pos/pos2.wav"}
    assert all(g.startswith("word/") for g in positives)
    assert X.shape[0] == y.shape[0] == len(files) == len(groups)


def test_pos_dir_works_with_empty_pool(tmp_path):
    """一条老正例都没有的新工作区：靠内置语料当负例也能开训。"""
    main, builtin, word = tmp_path / "corpus", tmp_path / "builtin", tmp_path / "w-abc"
    os.makedirs(main / "pos", exist_ok=True)
    os.makedirs(main / "neg", exist_ok=True)
    _make(builtin, 1, 1)
    _make(word, 2, 0)

    *_, y, _t, _f, groups, _m = load_dataset(
        str(main), [str(builtin)], pos_dir=str(word))
    assert set(np.unique(y)) == {0, 1}
    assert {g for g, lab in zip(groups, y) if lab == 1} == {
        "word/pos/pos0.wav", "word/pos/pos1.wav"}
    assert {g for g, lab in zip(groups, y) if lab == 0} == {
        "builtin/pos/pos0.wav", "builtin/neg/neg0.wav"}


def test_pos_dir_without_recordings_is_a_clear_error(tmp_path):
    """词目录还没录音（也没合成）时不能静默训一个"全负例"的模型。"""
    import pytest

    main, word = tmp_path / "corpus", tmp_path / "w-empty"
    _make(main, 1, 1)
    os.makedirs(word / "pos", exist_ok=True)

    with pytest.raises(SystemExit, match="新唤醒词还没有可用正例"):
        load_dataset(str(main), pos_dir=str(word))


def test_default_call_keeps_positives_positive(tmp_path):
    """缺省 pos_dir 时口径必须完全不变（老 run 重训不受影响）。"""
    main, builtin = tmp_path / "corpus", tmp_path / "builtin"
    _make(main, 2, 2)
    _make(builtin, 1, 1)

    *_, y, _t, _f, groups, _m = load_dataset(str(main), [str(builtin)])
    assert {g for g, lab in zip(groups, y) if lab == 1} == {
        "pos/pos0.wav", "pos/pos1.wav", "builtin/pos/pos0.wav"}


def _many(root, label, n, synth):
    """造 n 条语料：synth=True 时文件名带 -synth-（训练侧据此区分真人/合成）。"""
    d = os.path.join(root, label)
    os.makedirs(d, exist_ok=True)
    tag = "synth" if synth else "mic"
    for i in range(n):
        _wav(os.path.join(d, f"{label}-{tag}-{i:03d}.wav"), freq=440 + (i % 40))


def test_training_clips_are_capped_and_humans_are_kept(tmp_path):
    """合成语料多起来以后按「条」抽样：正例≤50、负例≤300，真人全留。"""
    root = tmp_path / "corpus"
    _many(root, "pos", 60, synth=True)
    _many(root, "pos", 5, synth=False)
    _many(root, "neg", 320, synth=True)
    _many(root, "neg", 4, synth=False)

    stats = {}
    *_, y, _t, _f, groups, _m = load_dataset(str(root), max_pos=50, max_neg=300, stats=stats)

    kept_pos = {g for g, lab in zip(groups, y) if lab == 1}
    kept_neg = {g for g, lab in zip(groups, y) if lab == 0}
    assert len(kept_pos) == 50
    assert len(kept_neg) == 300

    # 真人一条不少（5 条正例 + 4 条负例），其余名额才轮到合成
    assert sum(1 for g in kept_pos if "-synth-" not in g) == 5
    assert sum(1 for g in kept_neg if "-synth-" not in g) == 4

    assert stats["pos"]["before"] == 65 and stats["pos"]["after"] == 50
    assert stats["neg"]["before"] == 324 and stats["neg"]["after"] == 300
    assert stats["pos"]["human"] == 5 and stats["neg"]["human"] == 4
    assert len(stats["human_groups"]) == 9


def test_sampling_is_deterministic(tmp_path):
    """同一份语料 + 同一种子 → 抽到同一批（rerun 才能复用考卷）。"""
    root = tmp_path / "corpus"
    _many(root, "pos", 70, synth=True)
    _many(root, "neg", 20, synth=False)

    first = load_dataset(str(root), max_pos=50, max_neg=300)[4]
    second = load_dataset(str(root), max_pos=50, max_neg=300)[4]
    assert list(first) == list(second)


def test_human_split_sends_all_human_to_validation():
    """真人录音只做考卷，训练集只剩合成片段。"""
    labels = [1] * 20 + [0] * 20
    files = [f"f{i}" for i in range(40)]
    groups = list(files)
    human = [f"f{i}" for i in range(5)] + [f"f{i}" for i in range(20, 25)]

    val_idx = make_split(labels, files, None, val_ratio=0.12, seed=42,
                         split_by="human", groups=groups, human_groups=human)
    val = {files[i] for i in val_idx}

    assert val == set(human)
    assert all("-synth-" not in g for g in val)          # 满足"考卷全是真人"
    assert set(files) - val                              # 训练侧不空


def test_human_split_falls_back_when_a_class_has_no_human():
    """某类没有真人（比如只勾了内置合成语料）：该类别回落随机 12%。"""
    labels = [1] * 10 + [0] * 10
    files = [f"f{i}" for i in range(20)]
    human = [f"f{i}" for i in range(3)]                  # 只有正例有真人

    val_idx = make_split(labels, files, None, val_ratio=0.12, seed=42,
                         split_by="human", groups=list(files), human_groups=human)
    val = {files[i] for i in val_idx}

    assert set(human) <= val                             # 正类真人仍然全进考卷
    assert any(f in val for f in files[10:])             # 负类回落后也有考卷


def test_windows_cover_full_and_partial_word():
    pcm = np.zeros(SR, dtype=np.float32)
    pcm[SR // 2: SR // 2 + int(0.9 * SR)] = 0.2    # 中间 0.9 s 有声（≈唤醒词长度）

    wins = make_windows(pcm, label=1)
    labels = [t for _, t in wins]
    assert labels[len(labels) // 2] == 1.0          # 正中那一窗完整覆盖
    assert 0.5 in labels                            # 偏 600 ms 的窗口只剩半个词
    assert min(labels) >= 0.5                       # ±600 ms 内还不会完全滑出去
    assert len(wins) == 9

    neg = make_windows(pcm, label=0)
    assert [t for _, t in neg] == [0.0, 0.0, 0.0]


# ---- 特征缓存：第一次提取后落盘，后续装载不再解码 / 提特征 -----------------

def test_feature_cache_second_load_skips_featurize(tmp_path, monkeypatch):
    from trainer import data as data_mod

    root = tmp_path / "corpus"
    _make(str(root), 2, 2)
    calls = {"n": 0}
    real = data_mod.logmel

    def counting(win):
        calls["n"] += 1
        return real(win)

    monkeypatch.setattr(data_mod, "logmel", counting)
    a = load_dataset(str(root), model="simple")
    n_first = calls["n"]
    assert n_first > 0
    b = load_dataset(str(root), model="simple")
    assert calls["n"] == n_first                       # 第二次完全没提特征
    assert np.array_equal(a[0], b[0])                  # 特征逐位一致
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert a[3] == b[3] and a[4] == b[4] and a[5] == b[5]


def test_feature_cache_label_key_and_silence(tmp_path, monkeypatch):
    from trainer import data as data_mod

    root = tmp_path / "corpus"
    os.makedirs(root / "pos")
    os.makedirs(root / "neg")
    w = (0.2 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR) * 32767).astype("<i2").tobytes()
    write_wav(str(root / "pos" / "a.wav"), w)
    write_wav(str(root / "neg" / "b.wav"), w)          # 同内容：sha 相同
    silent = np.zeros(SR, dtype=np.float32)
    write_wav(str(root / "neg" / "quiet.wav"), silent.tobytes())

    reads = {"n": 0}
    real_read = data_mod.read_wav

    def counting_read(path):
        reads["n"] += 1
        return real_read(path)

    monkeypatch.setattr(data_mod, "read_wav", counting_read)
    X, y, t, files, groups, _ = load_dataset(str(root), model="simple",
                                             max_pos=10, max_neg=10)
    n_first = reads["n"]
    assert not any(g.endswith("quiet.wav") for g in groups)        # 无声被跳过
    assert sum(1 for f in files if f.startswith("pos/a.wav#")) == 9   # 正例 9 窗
    assert sum(1 for f in files if f.startswith("neg/b.wav#")) == 3   # 负例 3 窗

    X2, y2, t2, files2, groups2, _ = load_dataset(str(root), model="simple",
                                                  max_pos=10, max_neg=10)
    assert reads["n"] == n_first                       # 无声条目也走缓存（ok=0）
    assert np.array_equal(X, X2) and np.array_equal(t, t2)
    assert files == files2 and groups == groups2
