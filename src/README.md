# Phiên bản bài lab đã hoàn thiện

Thư mục `src/` chứa implementation offline xác định và tích hợp live provider tùy chọn.

- Thư mục giữ nguyên cấu trúc cấp cao của bài lab
- Các file Python đã hoàn thiện phần bắt buộc của rubric
- Cấu trúc benchmark cần có: standard benchmark + long-context stress benchmark
- Runtime cần hỗ trợ các provider: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`

Luồng triển khai đã áp dụng:

1. Bắt đầu với `config.py`
2. Triển khai `memory_store.py`
3. Hoàn thiện `agent_baseline.py`
4. Hoàn thiện `agent_advanced.py`
5. Triển khai `benchmark.py`
6. Kiểm chứng hành vi trong `test_agents.py`

Các dataset nằm trong thư mục `data/` ở repo root.
