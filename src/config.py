from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Cấu hình dùng chung cho bài lab.

    Gợi ý:
    - Lưu đường dẫn đến repo root, thư mục dataset và thư mục state.
    - Thêm cấu hình compact memory như ngưỡng kích hoạt và số message cần giữ.
    - Thêm cấu hình provider cho `openai`, `custom`, `gemini`, `anthropic`, `ollama` và `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Đọc các biến môi trường và trả về một `LabConfig`.

    Mã giả:
    1. Xác định repo root; mặc định suy ra từ thư mục chứa file hiện tại.
    2. Có thể đọc thêm các giá trị từ `.env`.
    3. Tạo `state/` nếu thư mục chưa tồn tại.
    4. Trả về một instance `LabConfig` đã có đầy đủ giá trị.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    load_dotenv(root / ".env")

    def env_int(name: str, default: int, minimum: int = 1) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} phải là số nguyên, nhận được {raw!r}.") from exc
        if value < minimum:
            raise ValueError(f"{name} phải >= {minimum}, nhận được {value}.")
        return value

    def env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} phải là số, nhận được {raw!r}.") from exc

    default_models = {
        "openai": "gpt-4o-mini",
        "custom": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash",
        "anthropic": "claude-sonnet-4-5",
        "ollama": "llama3.2",
        "openrouter": "openai/gpt-4o-mini",
    }

    def make_provider(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
        provider = normalize_provider(
            os.getenv(f"{prefix}_PROVIDER", fallback.provider if fallback else "openai")
        )
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "custom": os.getenv("CUSTOM_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "ollama": None,
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
        }
        base_urls = {
            "custom": os.getenv("CUSTOM_BASE_URL"),
            "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "openrouter": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        }
        fallback_model = (
            fallback.model_name
            if fallback is not None and fallback.provider == provider
            else default_models[provider]
        )
        return ProviderConfig(
            provider=provider,
            model_name=os.getenv(f"{prefix}_MODEL", fallback_model),
            temperature=env_float(
                f"{prefix}_TEMPERATURE", fallback.temperature if fallback else 0.0
            ),
            api_key=api_keys[provider],
            base_url=base_urls.get(provider),
        )

    model = make_provider("LLM")
    judge_model = make_provider("JUDGE", fallback=model)
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=env_int("COMPACT_THRESHOLD_TOKENS", 1_200),
        compact_keep_messages=env_int("COMPACT_KEEP_MESSAGES", 4),
        model=model,
        judge_model=judge_model,
    )
