"""阿里云 OSS 的最小客户端（只用标准库，不引 oss2）。

为什么手写签名：coze 环境的依赖越少越好，而这里只需要三个动作——
PUT / HEAD / DELETE。用 OSS 的 V1 签名（HMAC-SHA1）就够，且不依赖 SDK 版本。

凭据来源（按优先级）：
1. 环境变量 `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`（bootstrap 从 projenv 取的 .env）；
2. `--gallery-config` 指向的 ai-gallery `config.json` 的 `oss` 块（本机开发时唯一事实源）。

日志与错误里**绝不**出现 access key secret；`describe()` 只回显 key id 的前 4 位。
"""
import base64
import hashlib
import hmac
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from email.utils import formatdate
from pathlib import Path

from . import settings

DEFAULT_GALLERY_CONFIG = Path("~/git/ai-gallery/config.json").expanduser()
# 记下"出厂默认值"：只有仍是默认值的字段才允许被 ai-gallery 的配置覆盖
_DEFAULTS = {
    "endpoint": settings.OSS_ENDPOINT,
    "bucket": settings.OSS_BUCKET,
    "cdn_base": settings.OSS_CDN_BASE,
}


class OSSError(Exception):
    """OSS 调用失败（网络 / 签名 / 权限）。"""


# ---------------------------------------------------------------- 凭据
def read_gallery_config(path: str | Path | None = None) -> dict:
    """读 ai-gallery 的 config.json；读不到就返回空 dict（不抛错）。"""
    candidates = []
    if path:
        candidates.append(Path(path).expanduser())
    else:
        env = os.environ.get("GALLERY_CONFIG_PATH", "").strip()
        if env:
            candidates.append(Path(env).expanduser())
        candidates += [Path("/opt/ai-gallery/config.json"), DEFAULT_GALLERY_CONFIG]
    for candidate in candidates:
        try:
            with open(candidate, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def credentials(gallery_config: str | Path | None = None) -> dict:
    """返回 {endpoint,bucket,prefix,cdn_base,key_id,key_secret,source}。"""
    out = {
        "endpoint": settings.OSS_ENDPOINT,
        "bucket": settings.OSS_BUCKET,
        "prefix": settings.OSS_PREFIX,
        "cdn_base": settings.OSS_CDN_BASE,
        "key_id": settings.OSS_ACCESS_KEY_ID,
        "key_secret": settings.OSS_ACCESS_KEY_SECRET,
        "source": "env" if settings.OSS_ACCESS_KEY_ID else "",
    }
    if out["key_id"] and out["key_secret"]:
        return out

    conf = read_gallery_config(gallery_config)
    block = conf.get("oss") if isinstance(conf.get("oss"), dict) else {}
    cdn = conf.get("cdn") if isinstance(conf.get("cdn"), dict) else {}
    if block:
        # 只有"没被显式配置"的项才让 gallery 覆盖：环境变量 / .env 的优先级更高，
        # 否则本地把 base URL 指到测试服务器也会被 config.json 改回生产地址。
        if out["endpoint"] == _DEFAULTS["endpoint"]:
            out["endpoint"] = str(block.get("endpoint") or block.get("publicEndpoint")
                                  or out["endpoint"]).rstrip("/")
        if out["bucket"] == _DEFAULTS["bucket"]:
            out["bucket"] = str(block.get("bucket") or out["bucket"])
        if out["cdn_base"] == _DEFAULTS["cdn_base"]:
            if block.get("cdnBase"):
                out["cdn_base"] = str(block["cdnBase"]).rstrip("/")
            elif cdn.get("baseUrl"):
                # gallery 的 cdn.baseUrl 是 https://cdn.aibeaver.cn，音频挂在 /classroom 下
                out["cdn_base"] = str(cdn["baseUrl"]).rstrip("/") + "/classroom"
        out["key_id"] = str(block.get("accessKeyId") or "").strip()
        out["key_secret"] = str(block.get("accessKeySecret") or "").strip()
        if out["key_id"]:
            out["source"] = "gallery-config"
    return out


def configured(gallery_config: str | Path | None = None) -> bool:
    cred = credentials(gallery_config)
    return bool(cred["key_id"] and cred["key_secret"])


def describe(gallery_config: str | Path | None = None) -> dict:
    """给日志 / projenv 自检用的脱敏描述。"""
    cred = credentials(gallery_config)
    return {"endpoint": cred["endpoint"], "bucket": cred["bucket"],
            "prefix": cred["prefix"], "cdn_base": cred["cdn_base"],
            "key_id": (cred["key_id"][:4] + "***") if cred["key_id"] else "",
            "source": cred["source"] or "none"}


# ---------------------------------------------------------------- URL
def object_key(rel: str) -> str:
    rel = str(rel).lstrip("/")
    prefix = settings.OSS_PREFIX
    return f"{prefix}/{rel}" if prefix else rel


def _cdn_origin(cdn_base: str) -> str:
    """CDN 的路径前缀对应桶内同名目录：https://cdn.aibeaver.cn/classroom + key。

    key 本身就以 `classroom/` 开头，所以把 CDN base 的尾巴去掉再拼完整 key，
    换 bucket 前缀时只要保证这里是同一个目录名即可。
    """
    base = cdn_base.rstrip("/")
    tail = base.rsplit("/", 1)[-1]
    if tail and tail == settings.OSS_PREFIX.split("/")[0]:
        return base[: -len(tail)]
    return base


def public_url(key: str) -> str | None:
    """CDN 地址（公开读）。CDN base 缺失时返回 None。"""
    cred = credentials()
    if not cred["cdn_base"]:
        return None
    origin = _cdn_origin(cred["cdn_base"])
    sep = "" if origin.endswith("/") else "/"
    return f"{origin}{sep}{urllib.parse.quote(key)}"


def _object_url(endpoint: str, bucket: str, key: str) -> str:
    """对象 URL：必须用三级域名（`<bucket>.<endpoint-host>/<key>`）。

    阿里云新版地域（含 oss-cn-heyuan）拒绝 path-style 访问，会回
    `SecondLevelDomainForbidden`，所以这里统一拼虚拟主机风格。
    V1 签名的 CanonicalizedResource 仍是 `/<bucket>/<key>`，与 URL 风格无关。
    """
    endpoint = endpoint.rstrip("/")
    parts = urllib.parse.urlsplit(endpoint)
    if not parts.scheme:
        endpoint = "https://" + endpoint
        parts = urllib.parse.urlsplit(endpoint)
    host = parts.netloc
    if host.startswith(f"{bucket}."):
        base = f"{parts.scheme}://{host}"
    else:
        base = f"{parts.scheme}://{bucket}.{host}"
    return f"{base}/{urllib.parse.quote(key)}"


def origin_url(key: str) -> str:
    cred = credentials()
    return _object_url(cred["endpoint"], cred["bucket"], key)


def download_urls(key: str) -> list[str]:
    """先 CDN 再源站：CDN 命中率高、也不消耗 OSS 流量。"""
    urls = []
    cdn = public_url(key)
    if cdn:
        urls.append(cdn)
    urls.append(origin_url(key))
    return urls


# ---------------------------------------------------------------- 签名
def _sign(verb: str, content_type: str, date: str, key: str,
          key_id: str, key_secret: str) -> str:
    resource = f"/{settings.OSS_BUCKET}/{key}"
    to_sign = f"{verb}\n\n{content_type}\n{date}\n{resource}"
    digest = hmac.new(key_secret.encode("utf-8"), to_sign.encode("utf-8"),
                      hashlib.sha1).digest()
    return f"OSS {key_id}:{base64.b64encode(digest).decode('ascii')}"


def put_object(key: str, data: bytes, content_type: str = "application/octet-stream",
               gallery_config: str | Path | None = None) -> dict:
    """上传一个对象（幂等覆盖）。返回 {url, size}。"""
    cred = credentials(gallery_config)
    if not (cred["key_id"] and cred["key_secret"]):
        raise OSSError("没有 OSS 凭证（没配 OSS_ACCESS_KEY_ID 也读不到 ai-gallery config.json）")
    date = formatdate(usegmt=True)
    url = _object_url(cred["endpoint"], cred["bucket"], key)
    req = urllib.request.Request(url, data=data, method="PUT")
    req.add_header("Date", date)
    req.add_header("Content-Type", content_type)
    req.add_header("Content-Length", str(len(data)))
    req.add_header("Authorization",
                   _sign("PUT", content_type, date, key,
                         cred["key_id"], cred["key_secret"]))
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status not in (200, 201):
                raise OSSError(f"OSS 返回 {resp.status}")
    except urllib.error.HTTPError as exc:
        body = exc.read()[:200].decode("utf-8", "replace")
        raise OSSError(f"OSS 上传失败 {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise OSSError(f"OSS 连不上：{exc.reason}") from exc
    return {"url": public_url(key), "size": len(data)}


def delete_object(key: str, gallery_config: str | Path | None = None) -> None:
    cred = credentials(gallery_config)
    if not (cred["key_id"] and cred["key_secret"]):
        raise OSSError("没有 OSS 凭证")
    date = formatdate(usegmt=True)
    url = _object_url(cred["endpoint"], cred["bucket"], key)
    req = urllib.request.Request(url, method="DELETE")
    req.add_header("Date", date)
    req.add_header("Authorization", _sign("DELETE", "", date, key,
                                          cred["key_id"], cred["key_secret"]))
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status not in (200, 204):
                raise OSSError(f"OSS 返回 {resp.status}")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise OSSError(f"OSS 删除失败 {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise OSSError(f"OSS 连不上：{exc.reason}") from exc


def object_size(key: str, gallery_config: str | Path | None = None) -> int | None:
    cred = credentials(gallery_config)
    if not (cred["key_id"] and cred["key_secret"]):
        return None
    date = formatdate(usegmt=True)
    url = _object_url(cred["endpoint"], cred["bucket"], key)
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("Date", date)
    req.add_header("Authorization", _sign("HEAD", "", date, key,
                                          cred["key_id"], cred["key_secret"]))
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return int(resp.headers.get("Content-Length") or 0)
    except urllib.error.HTTPError:
        return None
    except urllib.error.URLError:
        return None
