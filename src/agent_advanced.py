from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B kết hợp short-term, persistent và compact memory.

    Các lớp memory bắt buộc:
    1. memory trong phạm vi session
    2. `User.md` bền vững
    3. compact memory cho thread dài
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.thread_owners: dict[str, str] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Điều phối giữa offline mode và live mode."""

        if not user_id.strip():
            raise ValueError("user_id không được rỗng.")
        if not thread_id.strip():
            raise ValueError("thread_id không được rỗng.")
        if not message.strip():
            raise ValueError("message không được rỗng.")
        owner = self.thread_owners.setdefault(thread_id, user_id)
        if owner != user_id:
            raise ValueError(
                f"thread_id {thread_id!r} đã thuộc user khác; từ chối dùng chung memory."
            )
        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)

        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        context = self.compact_memory.context(thread_id)
        profile = self.profile_store.read_text(user_id)
        system = "Hồ sơ bền vững của người dùng:\n" + profile
        if context["summary"]:
            system += "\nTóm tắt hội thoại cũ:\n" + str(context["summary"])
        messages = [{"role": "system", "content": system}, *context["messages"]]
        result = self.langchain_agent.invoke(messages)
        response = str(getattr(result, "content", result))
        self.compact_memory.append(thread_id, "assistant", response)
        response_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        return {
            "response": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
            "memory_path": str(self.profile_store.path_for(user_id)),
            "mode": "live",
        }

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Triển khai luồng advanced có kết quả xác định.

        Mã giả:
        1. Trích xuất các profile fact ổn định từ message đầu vào.
        2. Lưu bền vững các fact đó vào `User.md`.
        3. Thêm message vào compact memory.
        4. Ước lượng prompt context từ `User.md` + summary + các message gần nhất.
        5. Sinh phản hồi có thể trả lời các câu hỏi recall dài hạn.
        6. Thêm phản hồi của assistant và cập nhật bộ đếm token.
        """

        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)

        response_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        return {
            "response": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
            "memory_path": str(self.profile_store.path_for(user_id)),
            "mode": "offline",
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Ước lượng context được đưa vào một lượt xử lý.

        Gợi ý:
        - Bao gồm `User.md`
        - Bao gồm nội dung compact summary
        - Bao gồm các message gần nhất được giữ lại
        """

        context = self.compact_memory.context(thread_id)
        total = estimate_tokens(self.profile_store.read_text(user_id))
        total += estimate_tokens(str(context["summary"]))
        messages = context["messages"]
        assert isinstance(messages, list)
        total += sum(estimate_tokens(str(item.get("content", ""))) for item in messages)
        return total

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Trả về câu trả lời xác định bằng persisted memory.

        Đảm bảo advanced agent có thể trả lời các câu hỏi như:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        facts = self.profile_store.facts(user_id)
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
                "đâu mới là",
            )
        )
        if not is_recall:
            return "Mình đã cập nhật memory phù hợp cho thông tin trong lượt này."
        if not facts:
            return "Mình chưa có đủ thông tin bền vững để trả lời."

        requested: list[str] = []
        if any(marker in lowered for marker in ("tên", "là ai", "tóm tắt")):
            requested.append("name")
        if any(
            marker in lowered
            for marker in ("nghề", "làm gì", "là ai", "tóm tắt", "product manager")
        ):
            requested.append("profession")
        if any(
            marker in lowered
            for marker in ("ở đâu", "nơi ở", "còn ở", "huế", "hà nội", "đà nẵng")
        ):
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
            "profession": "Nghề nghiệp hiện tại",
            "location": "Nơi ở hiện tại",
            "response_style": "Style trả lời",
            "favorite_drink": "Đồ uống yêu thích",
            "favorite_food": "Món ăn yêu thích",
            "pet": "Thú cưng",
            "interests": "Mối quan tâm",
        }
        parts = [f"{labels[key]}: {facts[key]}" for key in requested if key in facts]
        return "; ".join(parts) if parts else "Mình chưa có đủ thông tin bền vững để trả lời."

    def _maybe_build_langchain_agent(self):
        """Khởi tạo chat model tùy chọn; memory được wrapper này quản lý.

        Thiết kế cấp cao:
        - Dùng `build_chat_model(self.config.model)` cho provider đã chọn
        - Dùng `InMemorySaver` cho short-term state của thread
        - Tool đọc `User.md`
        - Tool ghi/chỉnh sửa `User.md`
        - Dynamic prompt đưa profile memory vào context
        - Summarization middleware cho thread dài
        """

        provider = self.config.model.provider
        if provider != "ollama" and not self.config.model.api_key:
            return None
        return build_chat_model(self.config.model)
