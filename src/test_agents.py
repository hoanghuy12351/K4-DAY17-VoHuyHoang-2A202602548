from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import load_conversations, recall_points
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates
from model_provider import normalize_provider


@pytest.fixture
def workspace_tmp_path() -> Path:
    """Tạo test directory mới trong `state/`, không phụ thuộc Windows TEMP."""

    root = Path(__file__).resolve().parent.parent
    path = root / "state" / "test-runs" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_config(tmp_path: Path):
    """Tạo cấu hình độc lập cho các test."""

    config = load_config(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return replace(
        config,
        state_dir=state_dir,
        compact_threshold_tokens=80,
        compact_keep_messages=2,
    )


def test_user_markdown_read_write_edit(workspace_tmp_path: Path) -> None:
    """Kiểm tra `User.md` có thể được tạo, cập nhật và chỉnh sửa."""

    store = UserProfileStore(workspace_tmp_path / "profiles")
    path = store.write_text("dungct", "# User Profile\n- name: DũngCT")
    assert path.exists()
    assert "DũngCT" in store.read_text("dungct")
    assert store.edit_text("dungct", "DũngCT", "Dũng CT") is True
    assert store.edit_text("dungct", "không tồn tại", "x") is False
    store.upsert_fact("dungct", "location", "Huế")
    store.upsert_fact("dungct", "location", "Đà Nẵng")
    assert store.facts("dungct") == {"name": "Dũng CT", "location": "Đà Nẵng"}
    assert store.file_size("dungct") == len(path.read_bytes())


def test_compact_trigger() -> None:
    """Kiểm tra thread dài có kích hoạt compact."""

    memory = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    for index in range(8):
        memory.append("thread-1", "user", f"Lượt {index}: " + "ngữ cảnh dài " * 20)
    context = memory.context("thread-1")
    assert context["compactions"] > 0
    assert context["summary"]
    assert len(context["messages"]) <= 2


def test_cross_session_recall(workspace_tmp_path: Path) -> None:
    """Kiểm tra advanced nhớ qua session còn baseline thì không."""

    config = make_config(workspace_tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    fact = "Chào bạn, mình tên là DũngCT. Hiện tại mình đang ở Huế."
    question = "Sang thread mới, nhắc lại giúp mình tên và nơi ở hiện tại."

    baseline.reply("dungct", "baseline-old", fact)
    advanced.reply("dungct", "advanced-old", fact)
    baseline_answer = baseline.reply("dungct", "baseline-new", question)["response"]
    advanced_answer = advanced.reply("dungct", "advanced-new", question)["response"]

    assert "DũngCT" not in baseline_answer
    assert "Huế" not in baseline_answer
    assert "DũngCT" in advanced_answer
    assert "Huế" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(workspace_tmp_path: Path) -> None:
    """So sánh prompt load của baseline và advanced trên thread dài."""

    config = make_config(workspace_tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for index in range(20):
        message = (
            f"Lượt {index}: mình đang phân tích memory cho AI agent. "
            + "Đây là đoạn ngữ cảnh dài dùng để đo prompt load. " * 12
        )
        baseline.reply("user", "long-baseline", message)
        advanced.reply("user", "long-advanced", message)

    assert advanced.compaction_count("long-advanced") > 0
    assert advanced.prompt_token_usage("long-advanced") < baseline.prompt_token_usage(
        "long-baseline"
    )


def test_correction_and_noise_keep_latest_stable_fact(workspace_tmp_path: Path) -> None:
    config = make_config(workspace_tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    advanced.reply(
        "stress-user",
        "facts-1",
        "Mình tên là DũngCT Stress, hiện ở Huế và đang làm MLOps engineer.",
    )
    advanced.reply(
        "stress-user",
        "facts-1",
        "Thực ra từ tuần này mình đang làm việc ở Đà Nẵng vài tháng. Nghề nghiệp thì vẫn là MLOps engineer.",
    )
    advanced.reply(
        "stress-user",
        "facts-1",
        "Product manager chỉ là câu đùa. Hà Nội chỉ là nơi họp, không phải nơi ở hiện tại.",
    )
    answer = advanced.reply(
        "stress-user",
        "facts-2",
        "Huế, Hà Nội hay product manager đều đã được nhắc; đâu mới là nghề nghiệp và nơi ở hiện tại?",
    )["response"]
    assert "MLOps engineer" in answer
    assert "Đà Nẵng" in answer
    assert "product manager" not in answer.casefold()


def test_profile_path_cannot_escape_root(workspace_tmp_path: Path) -> None:
    store = UserProfileStore(workspace_tmp_path / "profiles")
    path = store.path_for("../outside-user")
    assert path.is_relative_to((workspace_tmp_path / "profiles").resolve())
    assert path.name == "User.md"


def test_question_does_not_create_profile_fact() -> None:
    assert extract_profile_updates("Mình tên gì và hiện tại đang ở đâu?") == {}


def test_provider_normalization_and_validation() -> None:
    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider("Google_GenAI") == "gemini"
    assert normalize_provider("open-router") == "openrouter"
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        normalize_provider("unknown-provider")


def test_dataset_contract_and_recall_metric() -> None:
    root = Path(__file__).resolve().parent.parent
    conversations = load_conversations(root / "data" / "conversations.json")
    assert len(conversations) == 10
    assert recall_points("Tên DũngCT, ở Huế", ["DũngCT", "Huế"]) == 1.0
    assert recall_points("Tên DũngCT", ["DũngCT", "Huế"]) == 0.5
    assert recall_points("Chưa biết", ["DũngCT", "Huế"]) == 0.0
