from __future__ import annotations

import json
import tempfile
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Đọc và kiểm tra schema cơ bản của conversation JSON."""

    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset không phải JSON hợp lệ: {path}") from exc
    if not isinstance(payload, list):
        raise ValueError("Dataset phải là một JSON array.")

    required = {"id", "user_id", "turns", "recall_questions"}
    for index, conversation in enumerate(payload):
        if not isinstance(conversation, dict):
            raise ValueError(f"Conversation #{index} phải là object.")
        missing = required - conversation.keys()
        if missing:
            raise ValueError(f"Conversation #{index} thiếu field: {sorted(missing)}")
        if not isinstance(conversation["turns"], list) or not all(
            isinstance(turn, str) and turn.strip() for turn in conversation["turns"]
        ):
            raise ValueError(f"Conversation {conversation['id']!r} có turns không hợp lệ.")
        if not isinstance(conversation["recall_questions"], list):
            raise ValueError(
                f"Conversation {conversation['id']!r} có recall_questions không hợp lệ."
            )
        for question in conversation["recall_questions"]:
            if not isinstance(question, dict) or not isinstance(question.get("question"), str):
                raise ValueError(f"Recall question không hợp lệ trong {conversation['id']!r}.")
            expected = question.get("expected_contains")
            if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
                raise ValueError(f"expected_contains không hợp lệ trong {conversation['id']!r}.")
    return payload


def recall_points(answer: str, expected: list[str]) -> float:
    """Trả về 0 / 0.5 / 1 tùy số expected fact xuất hiện."""

    if not expected:
        return 1.0

    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    normalized_answer = normalize(answer)
    matched = sum(normalize(item) in normalized_answer for item in expected)
    if matched == 0:
        return 0.0
    if matched == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Tính điểm chất lượng heuristic đơn giản cho offline mode."""

    stripped = answer.strip()
    if not stripped:
        return 0.0
    coverage = sum(
        unicodedata.normalize("NFKC", item).casefold()
        in unicodedata.normalize("NFKC", stripped).casefold()
        for item in expected
    ) / max(1, len(expected))
    # Câu trả lời offline nên đúng fact, đủ ngắn và không chỉ là một token rời rạc.
    clarity = 1.0 if 20 <= len(stripped) <= 800 else 0.5
    return round(min(1.0, coverage * 0.8 + clarity * 0.2), 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Đánh giá một agent trên nhiều conversation.

    Mã giả:
    1. Đưa toàn bộ các lượt hội thoại vào agent.
    2. Theo dõi `agent tokens only`.
    3. Theo dõi `prompt tokens processed`.
    4. Đặt câu hỏi recall trong một thread mới.
    5. Tính điểm recall và quality trung bình.
    6. Ghi nhận mức tăng của memory file và số lần compact.
    """

    del config  # Config đã được đóng gói trong agent; giữ tham số để ổn định API benchmark.
    users = {str(conversation["user_id"]) for conversation in conversations}
    before_sizes = {
        user_id: agent.memory_file_size(user_id) if hasattr(agent, "memory_file_size") else 0
        for user_id in users
    }
    used_threads: set[str] = set()
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    prefix = agent_name.casefold().replace(" ", "-")
    for conversation in conversations:
        user_id = str(conversation["user_id"])
        conversation_id = str(conversation["id"])
        thread_id = f"{prefix}:{conversation_id}"
        used_threads.add(thread_id)
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, turn)

        for question_index, question in enumerate(conversation["recall_questions"], start=1):
            recall_thread = f"{prefix}:{conversation_id}:recall:{question_index}"
            used_threads.add(recall_thread)
            result = agent.reply(user_id, recall_thread, question["question"])
            answer = str(result["response"])
            expected = list(question["expected_contains"])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    after_sizes = {
        user_id: agent.memory_file_size(user_id) if hasattr(agent, "memory_file_size") else 0
        for user_id in users
    }
    memory_growth = sum(max(0, after_sizes[user] - before_sizes[user]) for user in users)
    tokens = sum(agent.token_usage(thread_id) for thread_id in used_threads)
    prompt_tokens = sum(agent.prompt_token_usage(thread_id) for thread_id in used_threads)
    compactions = sum(agent.compaction_count(thread_id) for thread_id in used_threads)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recall_scores) / max(1, len(recall_scores)), 3),
        response_quality=round(sum(quality_scores) / max(1, len(quality_scores)), 3),
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """In bảng markdown bằng tabulate hoặc fallback nội bộ."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    values = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.3f}",
            f"{row.response_quality:.3f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(values, headers=headers, tablefmt="github")
    except ImportError:
        all_rows = [headers, *[[str(cell) for cell in row] for row in values]]
        widths = [max(len(row[index]) for row in all_rows) for index in range(len(headers))]

        def render(row: list[str]) -> str:
            return "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |"

        separator = "| " + " | ".join("-" * width for width in widths) + " |"
        return "\n".join([render(all_rows[0]), separator, *[render(row) for row in all_rows[1:]]])


def main() -> None:
    """Chạy cả Standard Benchmark và Long-Context Stress Benchmark.

    Các phần benchmark bắt buộc:
    - Standard benchmark từ `data/conversations.json`
    - Long-context stress benchmark từ `data/advanced_long_context.json`

    So sánh:
    - Baseline
    - Advanced

    Giữ nguyên các cột output như bản lab đã hoàn thiện:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    suites = [
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    ]

    with tempfile.TemporaryDirectory(prefix="day17-memory-benchmark-") as temporary:
        temporary_root = Path(temporary)
        for suite_index, (title, dataset_path) in enumerate(suites, start=1):
            conversations = load_conversations(dataset_path)
            baseline_config = replace(
                config,
                state_dir=temporary_root / f"suite-{suite_index}" / "baseline",
            )
            advanced_config = replace(
                config,
                state_dir=temporary_root / f"suite-{suite_index}" / "advanced",
            )
            baseline_config.state_dir.mkdir(parents=True, exist_ok=True)
            advanced_config.state_dir.mkdir(parents=True, exist_ok=True)
            agents = [
                ("Baseline", BaselineAgent(baseline_config, force_offline=True)),
                ("Advanced", AdvancedAgent(advanced_config, force_offline=True)),
            ]
            rows = [
                run_agent_benchmark(name, agent, conversations, agent.config)
                for name, agent in agents
            ]
            print(f"\n## {title}\n")
            print(format_rows(rows))


if __name__ == "__main__":
    main()
