from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Cấu hình provider dùng chung cho các agent.

    Các provider bắt buộc của bài lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Chuẩn hóa tên provider và ánh xạ alias, ví dụ `anthorpic` -> `anthropic`."""

    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "google-generative-ai": "gemini",
        "open-ai": "openai",
        "open-router": "openrouter",
        "openai-compatible": "custom",
        "local": "ollama",
    }
    provider = aliases.get(normalized, normalized)
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if provider not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"Provider không được hỗ trợ: {value!r}. Các giá trị hợp lệ: {choices}.")
    return provider


def build_chat_model(config: ProviderConfig):
    """Khởi tạo chat model thật cho provider đã chọn.

    Mã giả:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """

    provider = normalize_provider(config.provider)
    common: dict[str, Any] = {
        "model": config.model_name,
        "temperature": config.temperature,
    }

    try:
        if provider in {"openai", "custom"}:
            from langchain_openai import ChatOpenAI

            if config.api_key:
                common["api_key"] = config.api_key
            if provider == "custom":
                if not config.base_url:
                    raise ValueError("CUSTOM_BASE_URL là bắt buộc khi provider là 'custom'.")
                common["base_url"] = config.base_url
            return ChatOpenAI(**common)

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            if config.api_key:
                common["google_api_key"] = config.api_key
            return ChatGoogleGenerativeAI(**common)

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            if config.api_key:
                common["api_key"] = config.api_key
            return ChatAnthropic(**common)

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            if config.base_url:
                common["base_url"] = config.base_url
            return ChatOllama(**common)

        from langchain_openrouter import ChatOpenRouter

        if config.api_key:
            common["api_key"] = config.api_key
        return ChatOpenRouter(**common)
    except ImportError as exc:
        package_by_provider = {
            "openai": "langchain-openai",
            "custom": "langchain-openai",
            "gemini": "langchain-google-genai",
            "anthropic": "langchain-anthropic",
            "ollama": "langchain-ollama",
            "openrouter": "langchain-openrouter",
        }
        package = package_by_provider[provider]
        raise ImportError(
            f"Thiếu package cho provider '{provider}'. Hãy cài bằng: pip install {package}"
        ) from exc
