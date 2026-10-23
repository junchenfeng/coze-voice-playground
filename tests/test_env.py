"""`.env` 读取器的行为：只填缺失、不覆盖已有环境变量。"""
import os

from trainer import env


def test_load_dotenv_fills_missing_only(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        'KEEP=from-file\n'
        'export NEW="two words"\n'
        "QUOTED='single'\n"
        "# 注释行\n"
        "\n"
        "EMPTY=\n"
        "WITH_EQ=a=b\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KEEP", "from-env")
    for key in ("NEW", "QUOTED", "EMPTY", "WITH_EQ"):
        monkeypatch.delenv(key, raising=False)

    env.load_dotenv(path)

    assert os.environ["KEEP"] == "from-env"          # 不覆盖
    assert os.environ["NEW"] == "two words"
    assert os.environ["QUOTED"] == "single"
    assert os.environ["EMPTY"] == ""
    assert os.environ["WITH_EQ"] == "a=b"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    env.load_dotenv(tmp_path / "nope.env")           # 不抛异常即可


def test_default_resource_id_comes_from_env():
    from trainer import tts

    assert tts.DEFAULT_RESOURCE_ID == os.environ.get("DOUBAO_RESOURCE_ID", tts.RID_2)
