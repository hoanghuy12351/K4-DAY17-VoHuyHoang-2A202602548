from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A chỉ có memory trong phạm vi thread hiện tại.

    Yêu cầu:
    - Chỉ có memory trong phạm vi một session
    - Không có `User.md` bền vững
    - Phải quên các fact dài hạn khi chuyển sang thread mới
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Trả về phản hồi của agent và số liệu token.

        Mã giả:
        - Nếu có live agent, gọi luồng live.
        - Nếu không, dùng luồng offline có kết quả xác định.
        """

        if not user_id.strip():
            raise ValueError("user_id không được rỗng.")
        if not thread_id.strip():
            raise ValueError("thread_id không được rỗng.")
        if not message.strip():
            raise ValueError("message không được rỗng.")
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        session.facts.update(extract_profile_updates(message))
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        result = self.langchain_agent.invoke(session.messages)
        response = str(getattr(result, "content", result))
        session.messages.append({"role": "assistant", "content": response})
        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
            "mode": "live",
        }

    def token_usage(self, thread_id: str) -> int:
        # Trả về tổng token tích lũy của agent trong một thread.
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        # Trả về lượng prompt context mà baseline đã xử lý tích lũy.
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline không có compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Triển khai hành vi offline xác định để benchmark lặp lại được.

        Hành vi gợi ý:
        - Lưu message mới của người dùng vào session
        - Sinh phản hồi ngắn và có kết quả xác định
        - Cập nhật số liệu token
        - Không nhớ fact giữa các thread id khác nhau
        """

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        session.facts.update(extract_profile_updates(message))

        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        response = self._answer_from_facts(session.facts, message)
        session.messages.append({"role": "assistant", "content": response})

        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
            "mode": "offline",
        }

    @staticmethod
    def _answer_from_facts(facts: dict[str, str], message: str) -> str:
        """Trả lời từ fact trong đúng thread hiện tại, không dùng dữ liệu bền vững."""

        lowered = message.casefold()
        is_recall = any(
            marker in lowered
            for marker in (
                "tên gì",
                "tên mình",
                "là ai",
                "ở đâu",
                "nơi ở",
                "nghề",
                "làm gì",
                "style",
                "kiểu trả lời",
                "đồ uống",
                "món ăn",
                "nuôi con gì",
                "mối quan tâm",
                "tóm tắt",
                "nhắc lại",
            )
        )
        if not is_recall:
            return "Mình đã ghi nhận thông tin trong thread hiện tại."
        if not facts:
            return "Mình chưa có đủ thông tin trong thread hiện tại để trả lời."

        requested: list[str] = []
        if any(marker in lowered for marker in ("tên", "là ai", "tóm tắt")):
            requested.append("name")
        if any(marker in lowered for marker in ("nghề", "làm gì", "là ai", "tóm tắt")):
            requested.append("profession")
        if any(marker in lowered for marker in ("ở đâu", "nơi ở", "còn ở")):
            requested.append("location")
        if any(marker in lowered for marker in ("style", "kiểu trả lời")):
            requested.append("response_style")
        if "đồ uống" in lowered:
            requested.append("favorite_drink")
        if "món ăn" in lowered:
            requested.append("favorite_food")
        if "nuôi con gì" in lowered:
            requested.append("pet")
        if any(marker in lowered for marker in ("mối quan tâm", "tóm tắt", "là ai")):
            requested.append("interests")

        labels = {
            "name": "Tên",
            "profession": "Nghề nghiệp",
            "location": "Nơi ở hiện tại",
            "response_style": "Style trả lời",
            "favorite_drink": "Đồ uống yêu thích",
            "favorite_food": "Món ăn yêu thích",
            "pet": "Thú cưng",
            "interests": "Mối quan tâm",
        }
        parts = [f"{labels[key]}: {facts[key]}" for key in requested if key in facts]
        return "; ".join(parts) if parts else "Mình chưa có đủ thông tin trong thread hiện tại để trả lời."

    def _maybe_build_langchain_agent(self):
        """Khởi tạo chat model tùy chọn; session memory được wrapper này quản lý.

        Dùng `build_chat_model(self.config.model)` để baseline chạy được với mọi provider được hỗ trợ.
        """

        provider = self.config.model.provider
        if provider != "ollama" and not self.config.model.api_key:
            return None
        return build_chat_model(self.config.model)
