"""Shared provider helpers for baseline generation and judge calls."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any

import requests


def first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def gemini_key() -> str:
    return first_env("GEMINI_API_KEY", "gemini_api_key")


def gemini_rest_base_url() -> str:
    explicit = os.environ.get("GEMINI_REST_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    openai_base = os.environ.get("OPENAI_BASE_URL", "").lower()
    if "yunwu.ai" in openai_base and os.environ.get("YUNWU_GEMINI_NATIVE", "1") != "0":
        return "https://yunwu.ai/v1beta"
    return "https://generativelanguage.googleapis.com/v1beta"


def gemini_rest_auth() -> tuple[str, dict[str, str], str]:
    base_url = gemini_rest_base_url()
    headers = {"content-type": "application/json"}
    if "yunwu.ai" in base_url.lower():
        key = first_env("OPENAI_API_KEY", "OPEN_API_KEY", "GPT_API_KEY", "GPT4O_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY/OPEN_API_KEY/GPT_API_KEY/GPT4O_API_KEY is required for Yunwu Gemini native API.")
        headers["Authorization"] = f"Bearer {key}"
        app_code = first_env("OPENAI_APP_CODE")
        if app_code or "yunwu.ai" in base_url.lower():
            headers["APP-Code"] = app_code or "APP-10012fcb"
        return base_url, headers, key
    key = gemini_key()
    if not key:
        raise EnvironmentError("GEMINI_API_KEY/gemini_api_key must be set for Google Gemini native API.")
    return base_url, headers, key


def _append_openai_image_parts(parts: list[dict[str, Any]], image_parts: list[dict[str, Any]] | None) -> None:
    for image_part in image_parts or []:
        if not isinstance(image_part, dict):
            continue
        image_url = image_part.get("image_url") or {}
        url = image_url.get("url") if isinstance(image_url, dict) else ""
        if not isinstance(url, str) or not url.startswith("data:"):
            continue
        header, _, data = url.partition(",")
        if not data:
            continue
        mime = "image/png"
        if header.startswith("data:"):
            mime = header[5:].split(";", 1)[0] or mime
        try:
            base64.b64decode(data, validate=True)
        except Exception:
            continue
        parts.append({"inlineData": {"mimeType": mime, "data": data}})


def generation_openai_base_url() -> str:
    if gemini_key():
        return os.environ.get(
            "GEMINI_OPENAI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    if first_env("DEEPSEEK_API_KEY", "deepseek_api_key"):
        return os.environ.get("DEEPSEEK_OPENAI_BASE_URL", "https://api.deepseek.com")
    if first_env("GLM_API_KEY", "glm_api_key"):
        return os.environ.get("GLM_OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    if first_env("DASHSCOPE_API_KEY"):
        return os.environ.get("DASHSCOPE_OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


def generation_model(
    *,
    deepseek_default: str = "deepseek-v4-pro",
    gemini_default: str = "gemini-2.5-pro",
    glm_default: str = "glm-4.6v-flashx",
    dashscope_default: str = "qwen3-max",
    openai_default: str = "gpt-4o-mini",
) -> str:
    if gemini_key():
        return os.environ.get("GEMINI_MODEL", gemini_default)
    if first_env("DEEPSEEK_API_KEY", "deepseek_api_key"):
        return first_env("DEEPSEEK_MODEL", "deepseek_model") or deepseek_default
    if first_env("GLM_API_KEY", "glm_api_key"):
        return os.environ.get("GLM_MODEL", glm_default)
    if first_env("DASHSCOPE_API_KEY"):
        return os.environ.get("DASHSCOPE_MODEL", dashscope_default)
    return os.environ.get("OPENAI_MODEL", openai_default)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    text = strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _safe_exception_text(exc: Exception) -> str:
    text = str(exc)
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            body = getattr(response, "text", "")
            if body:
                text = f"{text} | {body}"
        except Exception:
            pass
    for key in [
        gemini_key(),
        first_env("OPENAI_API_KEY", "OPEN_API_KEY", "GPT_API_KEY", "GPT4O_API_KEY"),
    ]:
        if not key:
            continue
        text = text.replace(key, "<redacted>")
    return text


def _exception_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def gemini_text_completion(
    prompt: str,
    model: str,
    max_tokens: int,
    *,
    system_prompt: str = "",
    temperature: float = 0.2,
    response_mime_type: str | None = None,
    image_parts: list[dict[str, Any]] | None = None,
    retries: int = 8,
    timeout: int = 180,
) -> str:
    base_url, headers, key = gemini_rest_auth()

    parts: list[dict[str, Any]] = [{"text": prompt}]
    _append_openai_image_parts(parts, image_parts)
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    if response_mime_type:
        payload["generationConfig"]["responseMimeType"] = response_mime_type

    url = f"{base_url}/models/{model}:generateContent"
    if "yunwu.ai" not in base_url.lower():
        url = f"{url}?key={key}"
    last_error = ""
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            candidates = body.get("candidates") or []
            if not candidates:
                return ""
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            texts = [str(part.get("text", "")) for part in parts if isinstance(part, dict) and part.get("text")]
            return "\n".join(texts).strip()
        except Exception as exc:
            last_error = _safe_exception_text(exc)
            status_code = _exception_status_code(exc)
            if status_code == 503 and ("distributor" in last_error or "无可用渠道" in last_error):
                break
            if status_code in {429, 503}:
                if attempt >= retries:
                    break
                time.sleep(min(300, 20 * attempt))
                continue
            if attempt >= retries:
                break
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Gemini request failed after retries: {last_error}") from None


def gemini_json_completion(
    prompt: str,
    model: str,
    max_tokens: int,
    *,
    system_prompt: str = "",
    temperature: float = 0.2,
    image_parts: list[dict[str, Any]] | None = None,
    retries: int = 4,
    timeout: int = 180,
) -> dict[str, Any]:
    text = gemini_text_completion(
        prompt,
        model,
        max_tokens,
        system_prompt=system_prompt,
        temperature=temperature,
        response_mime_type="application/json",
        image_parts=image_parts,
        retries=retries,
        timeout=timeout,
    )
    parsed = extract_first_json_object(text)
    if parsed is not None:
        return parsed
    return {
        "final_answer_markdown": strip_code_fences(text),
        "deliverables": {},
        "notes": "gemini returned non-json",
    }


def legacy_openai_base_url() -> str:
    if first_env("DEEPSEEK_API_KEY", "deepseek_api_key"):
        return os.environ.get("DEEPSEEK_OPENAI_BASE_URL", "https://api.deepseek.com")
    if first_env("GLM_API_KEY", "glm_api_key"):
        return os.environ.get("GLM_OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    if first_env("DASHSCOPE_API_KEY"):
        return os.environ.get("DASHSCOPE_OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


def legacy_model(
    *,
    deepseek_default: str = "deepseek-v4-pro",
    glm_default: str = "glm-4.6v-flashx",
    dashscope_default: str = "qwen3-max",
    openai_default: str = "gpt-4o-mini",
) -> str:
    if first_env("DEEPSEEK_API_KEY", "deepseek_api_key"):
        return first_env("DEEPSEEK_MODEL", "deepseek_model") or deepseek_default
    if first_env("GLM_API_KEY", "glm_api_key"):
        return os.environ.get("GLM_MODEL", glm_default)
    if first_env("DASHSCOPE_API_KEY"):
        return os.environ.get("DASHSCOPE_MODEL", dashscope_default)
    return os.environ.get("OPENAI_MODEL", openai_default)
