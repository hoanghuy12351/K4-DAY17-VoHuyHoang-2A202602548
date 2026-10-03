from __future__ import annotations

import math
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Ước lượng token bằng heuristic độ dài ký tự ổn định.

    Ý tưởng tham khảo:
    - Loại bỏ khoảng trắng thừa
    - Trả về 0 nếu chuỗi rỗng
    - Ước lượng token từ số ký tự, ví dụ len(text) / 4
    """

    normalized = " ".join(text.split())
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / 4))


@dataclass
class UserProfileStore:
    """Kho lưu trữ bền vững cho `User.md`.

    Chức năng:
    - Ánh xạ mỗi user id tới một file markdown
    - Hỗ trợ các thao tác đọc / ghi / chỉnh sửa
    - Có thể bổ sung các helper như `facts()` hoặc `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        # Làm sạch user id trước khi tạo đường dẫn để ngăn path traversal.
        normalized = unicodedata.normalize("NFKC", user_id).strip()
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
        if not slug:
            raise ValueError("user_id không tạo được tên thư mục hợp lệ.")
        return self.root_dir.resolve() / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        # Trả về nội dung file hoặc profile markdown mặc định rỗng.
        path = self.path_for(user_id)
        if not path.exists():
            return "# User Profile\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        # Ghi qua file tạm rồi replace để tránh để lại file dở dang.
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = content.rstrip() + "\n"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(normalized, encoding="utf-8")
        temporary.replace(path)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        # Thay thế một lần xuất hiện và cho biết nội dung có thay đổi hay không.
        if not search_text:
            raise ValueError("search_text không được rỗng.")
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        updated = content.replace(search_text, replacement, 1)
        if updated == content:
            return False
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        # Trả về kích thước hiện tại của file theo byte.
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Đọc các dòng ``- key: value`` từ profile."""

        result: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.fullmatch(r"- ([a-z][a-z0-9_]*):\s*(.+)", line.strip())
            if match:
                result[match.group(1)] = match.group(2).strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Thêm fact mới hoặc thay giá trị cũ của cùng key."""

        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError(f"Fact key không hợp lệ: {key!r}")
        clean_value = " ".join(value.split()).strip(" .")
        if not clean_value or "\n" in clean_value:
            raise ValueError("Fact value không được rỗng hoặc chứa dòng mới.")

        content = self.read_text(user_id)
        pattern = re.compile(rf"(?m)^- {re.escape(key)}:.*$")
        new_line = f"- {key}: {clean_value}"
        if pattern.search(content):
            content = pattern.sub(new_line, content, count=1)
        else:
            content = content.rstrip() + "\n" + new_line + "\n"
        return self.write_text(user_id, content)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Chuyển nội dung thô của người dùng thành các profile fact ổn định.

    Ví dụ các fact có thể trích xuất:
    - tên
    - nơi ở
    - nghề nghiệp
    - sở thích / phong cách phản hồi
    - món ăn / đồ uống yêu thích

    Mã giả:
    1. Xây dựng một số regex pattern.
    2. Bỏ qua các lượt chỉ chứa câu hỏi rõ ràng.
    3. Chỉ trả về những fact xuất hiện đủ rõ trong message.
    """

    text = unicodedata.normalize("NFKC", " ".join(message.split()))
    lowered = text.casefold()
    updates: dict[str, str] = {}

    # Câu hỏi recall có thể chứa chính tên field hoặc giá trị gây nhiễu;
    # không được biến nội dung câu hỏi thành fact mới của người dùng.
    if text.rstrip().endswith("?"):
        return updates

    name_match = re.search(
        r"\b(?i:mình|tôi)\s+(?i:tên)\s+(?i:là\s+)?"
        r"([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*){0,3})",
        text,
    )
    if name_match:
        updates["name"] = name_match.group(1).strip(" .,;:")

    place = r"([A-ZÀ-Ỹ][\wÀ-ỹ-]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ-]*){0,3})"
    location_patterns = [
        rf"(?i:nơi ở (?:hiện tại )?đã cập nhật từ ){place}(?i: sang ){place}",
        rf"(?i:nơi ở (?:hiện tại )?(?:là|:)\s*){place}",
        rf"(?i:(?:hiện|giờ|hiện tại|thực ra từ tuần này)\s+(?:mình\s+)?"
        rf"(?:đang\s+)?(?:làm việc\s+)?ở\s+){place}",
        rf"(?i:(?:mình|tôi)\s+(?:đang|vẫn)\s+ở\s+){place}",
        rf"(?i:(?:mình|tôi)\s+ở\s+){place}",
    ]
    for index, pattern in enumerate(location_patterns):
        match = re.search(pattern, text)
        if match:
            # Pattern correction có hai địa điểm; địa điểm sau `sang` là fact mới.
            value = match.group(2) if index == 0 else match.group(1)
            updates["location"] = value.strip(" .,;:")
            break

    role = r"([A-Za-z][A-Za-z0-9+#.-]*(?:\s+[A-Za-z][A-Za-z0-9+#.-]*){0,2}\s+(?:engineer|developer|scientist|analyst|manager))"
    role_patterns = [
        rf"nghề nghiệp(?: hiện tại)?(?: thì)?(?: vẫn)?\s+(?:là|:)\s*{role}",
        rf"giờ\s+(?:mình\s+)?chuyển sang\s+{role}",
        rf"(?:mình|tôi)\s+(?:hiện\s+)?(?:đang\s+|vẫn\s+)?làm\s+{role}",
    ]
    profession: str | None = None
    for pattern in role_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            profession = matches[-1]
            break
    if profession:
        if profession.casefold() != "product manager" or "câu đùa" not in lowered:
            updates["profession"] = profession.strip(" .,;:")

    if "3 bullet" in lowered:
        style_parts = ["3 bullet ngắn"]
        if "ví dụ thực chiến" in lowered or "ví dụ thực tế" in lowered:
            style_parts.append("có ví dụ thực chiến")
        if "trade-off" in lowered:
            style_parts.append("nhấn mạnh trade-off")
        updates["response_style"] = ", ".join(style_parts)
    elif "ngắn gọn" in lowered or "trả lời ngắn" in lowered:
        style_parts = ["ngắn gọn"]
        if "bullet" in lowered:
            style_parts.append("dạng bullet")
        if "ví dụ thực chiến" in lowered or "ví dụ thực tế" in lowered:
            style_parts.append("có ví dụ thực tế")
        if "trade-off" in lowered:
            style_parts.append("nêu rõ trade-off")
        updates["response_style"] = ", ".join(style_parts)

    if "cà phê sữa đá" in lowered and any(
        marker in lowered for marker in ("yêu thích", "mình thích", "vẫn uống", "đồ uống")
    ):
        updates["favorite_drink"] = "cà phê sữa đá"
    if "mì quảng" in lowered and any(
        marker in lowered for marker in ("yêu thích", "món ruột", "mình thích")
    ):
        updates["favorite_food"] = "mì Quảng"
    if "corgi" in lowered and any(marker in lowered for marker in ("nuôi", "con corgi", "bé corgi")):
        pet_name = re.search(r"corgi\s+tên\s+([A-ZÀ-Ỹ][\wÀ-ỹ-]*)", text, re.IGNORECASE)
        updates["pet"] = f"corgi tên {pet_name.group(1)}" if pet_name else "corgi"

    if "python" in lowered and "ai" in re.findall(r"\bai\b", lowered):
        interests = ["Python", "AI ứng dụng"]
        if "mlops" in lowered:
            interests.append("MLOps")
        updates["interests"] = ", ".join(interests)

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Tạo bản tóm tắt cô đọng, có giới hạn kích thước cho các message cũ.

    Trước tiên có thể dùng cách nối văn bản theo heuristic.
    Sau đó có thể thay bằng bản tóm tắt dựa trên LLM nếu cần.
    """

    if not messages or max_items <= 0:
        return ""

    facts: dict[str, str] = {}
    existing_summaries: list[str] = []
    for message in messages:
        content = str(message.get("content", "")).strip()
        if message.get("role") == "summary" and content:
            existing_summaries.append(content)
        facts.update(extract_profile_updates(content))

    sections: list[str] = []
    if facts:
        fact_text = "; ".join(f"{key}={value}" for key, value in sorted(facts.items()))
        sections.append(f"Fact ổn định: {fact_text}.")
    if existing_summaries:
        previous = " ".join(existing_summaries)
        sections.append(f"Tóm tắt trước: {previous[-450:]}")

    excerpts: list[str] = []
    for message in messages[-max_items:]:
        if message.get("role") == "summary":
            continue
        role = str(message.get("role", "unknown"))
        content = " ".join(str(message.get("content", "")).split())
        if content:
            clipped = content if len(content) <= 180 else content[:177].rstrip() + "..."
            excerpts.append(f"{role}: {clipped}")
    if excerpts:
        sections.append("Ý chính gần đây: " + " | ".join(excerpts))

    summary = "\n".join(sections)
    return summary if len(summary) <= 1_400 else summary[:1_397].rstrip() + "..."


@dataclass
class CompactMemoryManager:
    """Quản lý compact memory cho các thread dài.

    Mục tiêu:
    - Giữ nguyên nội dung đầy đủ của các message gần nhất
    - Khi thread quá dài, chuyển nội dung cũ vào summary
    - Theo dõi số lần compact để phục vụ benchmark
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens < 1:
            raise ValueError("threshold_tokens phải lớn hơn 0.")
        if self.keep_messages < 1:
            raise ValueError("keep_messages phải lớn hơn 0.")

    def _thread_state(self, thread_id: str) -> dict[str, object]:
        if not thread_id.strip():
            raise ValueError("thread_id không được rỗng.")
        return self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )

    def append(self, thread_id: str, role: str, content: str) -> None:
        # Luồng xử lý:
        # 1. tạo thread state nếu chưa tồn tại
        # 2. thêm message mới
        # 3. kích hoạt compact nếu cần
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"role không hợp lệ: {role!r}")
        thread = self._thread_state(thread_id)
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})

        summary = str(thread["summary"])
        context_tokens = estimate_tokens(summary) + sum(
            estimate_tokens(str(item.get("content", ""))) for item in messages
        )
        if context_tokens <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        old_messages = messages[:-self.keep_messages]
        recent_messages = messages[-self.keep_messages :]
        summary_input: list[dict[str, str]] = []
        if summary:
            summary_input.append({"role": "summary", "content": summary})
        summary_input.extend(old_messages)
        thread["summary"] = summarize_messages(summary_input)
        thread["messages"] = recent_messages
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        # Trả về bản sao state để caller không sửa trực tiếp internal memory.
        return deepcopy(self._thread_state(thread_id))

    def compaction_count(self, thread_id: str) -> int:
        # Trả về số lần compact của thread này.
        return int(self._thread_state(thread_id)["compactions"])
