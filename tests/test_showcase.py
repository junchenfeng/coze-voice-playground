"""Fixed dataset and exam plans for the projectless Coze variant."""
import json
from pathlib import Path

from trainer.showcase import build_plan, select_sets


def _seed_groups():
    path = Path(__file__).resolve().parents[1] / "data" / "seed-manifest.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["files"]
    rows = [r for r in rows if r["root"] in ("builtin", "corpus", "dataset")]
    return ([f"{r['root']}/{r['label']}/{r['name']}" for r in rows],
            [int(r["label"] == "pos") for r in rows])


def test_seed_plan_has_fixed_sizes_and_separate_exams():
    groups, labels = _seed_groups()
    plan = build_plan(groups, labels)
    by_label = dict(zip(groups, labels))
    assert len(plan["val"]) == 40
    assert len(plan["val_synthetic"]) == 36
    assert sum(by_label[g] == 1 for g in plan["val"]) == 10
    assert sum(by_label[g] == 1 for g in plan["val_synthetic"]) == 9
    for name in ("synthetic", "few_humans", "balanced_humans"):
        for exam in ("synthetic", "standard"):
            train, val, synth = select_sets(plan, groups, labels, name, exam)
            assert len(train) == 200
            assert sum(by_label[g] == 1 for g in train) == 50
            assert not set(train) & (set(val) | set(synth))


def test_local_recordings_join_exam_and_never_train():
    groups, labels = _seed_groups()
    own = [f"pos/own-{i}.wav" for i in range(4)] + [f"neg/own-{i}.wav" for i in range(5)]
    all_groups = groups + own
    all_labels = labels + [1] * 4 + [0] * 5
    plan = build_plan(all_groups, all_labels)
    train, val, synth = select_sets(plan, all_groups, all_labels,
                                    "balanced_humans", "standard_plus_recordings")
    assert set(own) <= set(val)
    assert not set(train) & (set(val) | set(synth))
    assert len(train) == 200
