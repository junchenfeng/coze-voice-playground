"""清单驱动的种子缓存：路径映射、命中判断、下载失败不阻断。"""
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from web import oss, seed, settings


def write_manifest(files=None, models=None):
    manifest = {"schema": 1, "prefix": settings.OSS_PREFIX,
                "files": files or [], "models": models or []}
    settings.SEED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    settings.SEED_MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")
    seed._cache.update({"mtime": None, "data": None})   # 同秒重写也要失效
    return manifest


def entry(name="pos-1.wav", label="pos", root="builtin", data=b"RIFF-data"):
    rel = f"seed/{root}/{label}/{name}"
    return {"path": rel, "root": root, "label": label, "name": name,
            "key": oss.object_key(rel), "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data), "duration": 1.0, "sample_rate": 16000,
            "synth": "-synth-" in name}, data


def test_manifest_missing_is_empty():
    settings.SEED_MANIFEST.unlink(missing_ok=True)
    assert seed.manifest()["files"] == []
    assert seed.counts() == {}
    assert seed.ensure_local(download=False)["total"] == 0


def test_counts_and_roots():
    a, _ = entry("a.wav", "pos", "builtin")
    b, _ = entry("b.wav", "neg", "builtin")
    c, _ = entry("c.wav", "pos", "corpus")
    write_manifest([a, b, c])
    assert seed.counts() == {"builtin": {"pos": 1, "neg": 1},
                             "corpus": {"pos": 1, "neg": 0}}
    assert seed.roots() == ["builtin", "corpus"]
    assert [f["name"] for f in seed.files(["corpus"])] == ["c.wav"]


def test_paths():
    row, _ = entry("x.wav", "pos", "builtin")
    assert seed.cache_path(row) == (settings.CACHE_ROOT / row["path"]).resolve()
    model = {"path": "models/backbone/openwakeword/melspectrogram.onnx"}
    assert seed.model_path(model) == (settings.BASE_DIR / model["path"]).resolve()
    with pytest.raises(seed.SeedError):
        seed.cache_path({"path": ""})


def test_check_reports_missing_without_network():
    row, _ = entry()
    write_manifest([row])
    stats = seed.ensure_local(["builtin"], download=False)
    assert stats["missing"] == 1 and stats["failed"] == 0
    assert stats["total"] == 1


def test_size_match_hit_skips_hashing(monkeypatch):
    """命中判断先看大小：一致就不再算 sha256（800 个文件的性能护栏）。"""
    row, data = entry()
    write_manifest([row])
    path = seed.cache_path(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

    calls = []
    monkeypatch.setattr(seed, "_sha256", lambda p: calls.append(p) or "x")
    stats = seed.ensure_local(["builtin"], verify=True, download=False)
    assert stats["ok"] == 1 and calls == []


def test_wrong_size_triggers_hash_check(monkeypatch):
    row, _ = entry()
    write_manifest([row])
    path = seed.cache_path(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"short")
    monkeypatch.setattr(seed, "_sha256", lambda p: "nope")
    assert seed.ensure_local(["builtin"], verify=True, download=False)["missing"] == 1


def test_download_from_local_http(monkeypatch):
    row, data = entry()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                    # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(settings, "OSS_CDN_BASE", base)
    monkeypatch.setattr(settings, "OSS_ENDPOINT", base)
    write_manifest([row])
    try:
        stats = seed.ensure_local(["builtin"])
    finally:
        server.shutdown()
    assert stats["downloaded"] == 1
    assert seed.cache_path(row).read_bytes() == data


def test_download_failure_is_warning_not_raise(monkeypatch):
    row, _ = entry()
    write_manifest([row])
    monkeypatch.setattr(settings, "OSS_CDN_BASE", "http://127.0.0.1:9/cdn")
    monkeypatch.setattr(settings, "OSS_ENDPOINT", "http://127.0.0.1:9")
    stats = seed.ensure_local(["builtin"])
    assert stats["failed"] == 1
    assert stats["warnings"]


def test_materialize_only_returns_existing_roots():
    row, data = entry()
    write_manifest([row])
    path = seed.cache_path(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    assert seed.materialize() == {"builtin": seed.local_root("builtin")}
