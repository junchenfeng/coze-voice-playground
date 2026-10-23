"""运行时配置。

两件事和旧站不同：
1. **没有账户/工作区**：每个扣子编程项目就是一台独立机器，全站只有一个数据域，
   所有路径都是常量（`data/corpus`、`data/runs` …），不再是 `data/workspaces/<slug>`。
2. **音频分层**：git 里没有 wav。预置语料与预训练模型放在 OSS，仓库只提交
   `data/seed-manifest.json`；用到时按清单下载到 `.cache/`（可随时删掉重建）。
   只有用户自己用麦克风录的音频才落在 `data/` 下。

全部路径与上限都走环境变量，测试时只需把 NOVA_DATA_ROOT / NOVA_CACHE_ROOT
指到临时目录，无需 mock 任何模块级常量（读取方一律 `settings.XXX`）。
"""
import os
from pathlib import Path

try:                                   # 本地裸跑时读仓库根的 .env；coze 里已由脚本注入
    from trainer.env import load_dotenv

    load_dotenv()
except ImportError:                    # web 被独立 import（sys.path 里没有项目根）时不致命
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("NOVA_DATA_ROOT", BASE_DIR / "data")).expanduser().resolve()
# 用户录音（麦克风）——唯一留在 data/ 里的音频
CORPUS_ROOT = DATA_ROOT / "corpus"
# 训练产物：data/runs/<run_id>/（权重、曲线、日志）
RUNS_ROOT = DATA_ROOT / "runs"
# 自定义唤醒词数据集：data/words/<word_id>/pos/*.wav
WORDS_ROOT = DATA_ROOT / "words"
ACTIVE_PATH = DATA_ROOT / "active.json"
# 合成语料的元数据索引（wav 本体在 .cache/synth 与 OSS）
SYNTH_INDEX = DATA_ROOT / "synth.json"
# 预置语料 / 预训练模型清单（进 git）
SEED_MANIFEST = DATA_ROOT / "seed-manifest.json"
# 研究笔记
RESEARCH_NOTES = DATA_ROOT / "research-notes.json"

# 可丢弃的缓存（种子音频、合成音频），删掉后按清单重建
CACHE_ROOT = Path(os.environ.get("NOVA_CACHE_ROOT", BASE_DIR / ".cache")).expanduser().resolve()
SEED_CACHE_ROOT = CACHE_ROOT / "seed"
SYNTH_CACHE_ROOT = CACHE_ROOT / "synth"

# openWakeWord 预训练底座（由 bootstrap / fetch_seed 下载，不进 git）
BACKBONE_DIR = Path(os.environ.get(
    "NOVA_BACKBONE_DIR", BASE_DIR / "models" / "backbone" / "openwakeword")).expanduser().resolve()

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
NOTE_LOGS_DIR = BASE_DIR / "note-logs"

# 上传与训练约束
MAX_UPLOAD_BYTES = int(os.environ.get("NOVA_MAX_UPLOAD_BYTES", 2 * 1024 * 1024))
MIN_CLIP_S = 0.3
MAX_CLIP_S = 5.0
# 上传的音量门限。只校验时长拦不住"按下了但没开麦"：线上真发生过一个人 15 条
# 空录音（RMS ≈ −50 dBFS）混进语料池，切出 45 个"正样本"全是底噪。
# 门限放在服务端，前端不传也逃不掉。
MIN_CLIP_RMS_DB = -45.0     # 全片 RMS 低于这个 = 什么都没录到
MIN_VOICED_S = 0.25         # -38 dBFS 以上的有效语音总秒数
# 削顶围栏：贴到满刻度的样本占比（%）。家长投诉过"贴着麦克风喊"的爆音，
# 听感已经裂了、训练也只学到失真；1% 以上一律拒收。
# 前端 recorder.js 的 GATES.maxClipPct 必须与这里同步。
MAX_CLIP_PCT = 1.0
MIN_PREDICT_S = 0.2

# 训练并发：coze 项目一般只有 1~2 核
MAX_RUNNING_TRAININGS = int(os.environ.get("NOVA_MAX_RUNNING_TRAININGS", 1))
TRAIN_THREADS = int(os.environ.get("NOVA_TRAIN_THREADS", 2))
TRAIN_TIMEOUT_S = int(os.environ.get("NOVA_TRAIN_TIMEOUT_S", 3 * 3600))
ERROR_TAIL_BYTES = 20 * 1024

# 训练前补种子语料的总预算（秒）。本沙箱会掐 Python socket 的大文件下载
# （TLS 握手正常、发完请求收不到数据，urllib 只能等满超时），缓存一旦缺失，
# 逐条下载能把 POST /api/runs 拖上几个小时——页面看起来就是永远"正在准备"。
# 超过预算就停：训练继续用本地已有的，warnings 里记一笔。
SEED_PREFETCH_BUDGET_S = int(os.environ.get("NOVA_SEED_PREFETCH_BUDGET_S", 30))

# SSE 轮询间隔
SSE_POLL_S = 0.3
SSE_HEARTBEAT_S = 15.0

# 语料下限：再少就没法分层切考卷（make_split 每类至少 1 条验证集，训练集不能空）
MIN_POS = 4
MIN_NEG = 4
# 单类上限（单数据域，所以是全局上限）
MAX_CLIPS_PER_LABEL = int(os.environ.get("NOVA_MAX_CLIPS_PER_LABEL", 1000))

# 自定义唤醒词：正例只数真人录音（合成语料用来补多样性，不能替代"自己喊一遍"）
MIN_WORD_POS = int(os.environ.get("NOVA_MIN_WORD_POS", 15))
WORD_TEXT_MAX = 24
# 单个词的 pos 目录上限：纯属防呆（误点合成按钮不能把磁盘塞满）
WORD_POS_MAX = int(os.environ.get("NOVA_WORD_POS_MAX", 400))

# 合成语料（豆包 TTS）：一次请求最多造几条，以及并发合成的线程数。
# 一次 TTS 约 0.5~2 s，条数放太大请求会挂太久（前端是同步等的）。
SYNTH_MAX_POS = int(os.environ.get("NOVA_SYNTH_MAX_POS", 6))
SYNTH_MAX_NEG = int(os.environ.get("NOVA_SYNTH_MAX_NEG", 6))
SYNTH_WORKERS = int(os.environ.get("NOVA_SYNTH_WORKERS", 3))

# 自定义唤醒词的「一键多音色合成」：每个音色 2 条、语速不同。
WORD_SYNTH_PER_VOICE = 2
WORD_SYNTH_MAX = int(os.environ.get("NOVA_WORD_SYNTH_MAX", 40))

# 训练装载上限（页面上「训练数据集大小」两个滑杆的边界与默认值）。
# 上限 = 训练集总条数（真人 + 合成），默认 50/150 = 真人 30/50 + 合成 20/100；
# 人声不够（少于真人配额）时由合成音补满到上限。
TRAIN_MAX_POS = int(os.environ.get("NOVA_TRAIN_MAX_POS", 50))
TRAIN_MAX_NEG = int(os.environ.get("NOVA_TRAIN_MAX_NEG", 150))
TRAIN_MAX_POS_RANGE = (30, 80)
TRAIN_MAX_NEG_RANGE = (100, 350)
# 真人录音的分配：每类固定抽这么多条进训练集，其余全部当考卷。
# 滑杆缩小时**先减合成音**，真人配额不缩——真人样本太少才是过拟合到合成音色的主因。
HUMAN_TRAIN_POS = int(os.environ.get("NOVA_HUMAN_TRAIN_POS", 30))
HUMAN_TRAIN_NEG = int(os.environ.get("NOVA_HUMAN_TRAIN_NEG", 50))

# 页面上可选的三张模型卡：简单 CNN / 复杂 CNN / 预训练底座。
# mlp 只留给 CLI（trainer.train --model mlp）兼容老实验，前端不再暴露。
MODELS = ("simple", "cnn", "head")
CLI_MODELS = ("mlp", "simple", "cnn", "head")

# OSS：预置语料与合成音频的落地处。读写凭证只从 .env 读（bootstrap 用
# scripts/fetch_projenv.py 从 voice.aibeaver.cn/projenv 取），绝不写进代码。
OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT", "https://oss-cn-heyuan.aliyuncs.com").strip()
OSS_BUCKET = os.environ.get("OSS_BUCKET", "ai-gallery-aibeaver-hy").strip()
OSS_PREFIX = os.environ.get("OSS_PREFIX", "classroom/voice-playground").strip().strip("/")
OSS_CDN_BASE = os.environ.get("OSS_CDN_BASE", "https://cdn.aibeaver.cn/classroom").strip().rstrip("/")
OSS_ACCESS_KEY_ID = os.environ.get("OSS_ACCESS_KEY_ID", "").strip()
OSS_ACCESS_KEY_SECRET = os.environ.get("OSS_ACCESS_KEY_SECRET", "").strip()

# projenv 是**另一个仓库**（线上 voice.aibeaver.cn）的端点，新项目只消费它。
# 旧站的 /credential 留给桌面端，两者 token 互不通用。
PROJENV_URL = os.environ.get("PROJENV_URL", "https://voice.aibeaver.cn/projenv").strip()
