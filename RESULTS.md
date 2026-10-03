# Kết quả Day 17: Memory Systems for AI Agent

Kết quả dưới đây được đo ngày 2026-10-03 bằng Python 3.14.5 ở offline mode. Vì output offline có tính xác định, benchmark có thể chạy lặp lại mà không cần API key.

## Automated tests

```text
9 passed
```

Các test tạo directory riêng dưới `state/test-runs/` để tránh lỗi quyền truy cập của `tmp_path` trên một số môi trường Windows.

Test bao phủ:

- thao tác read/write/edit/upsert với `User.md`
- compact trigger và giới hạn recent messages
- cross-session recall của Advanced so với Baseline
- giảm prompt load trên thread dài
- correction và loại nhiễu
- chống path traversal ở profile path
- không biến câu hỏi recall thành profile fact
- chuẩn hóa provider
- schema dataset và recall metric

## Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1501 | 14393 | 0.000 | 0.200 | 0 | 0 |
| Advanced | 1735 | 21113 | 1.000 | 0.993 | 267 | 0 |

Ở hội thoại ngắn, Advanced tốn thêm prompt token vì phải đưa `User.md` vào context. Đây là overhead thật, không phải compact luôn thắng.

## Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 285 | 22171 | 0.000 | 0.200 | 0 | 0 |
| Advanced | 362 | 13639 | 1.000 | 1.000 | 229 | 3 |

Trong stress benchmark, compact giảm prompt load của Advanced khoảng 38.5% so với Baseline trong khi vẫn giữ full recall trên bộ câu hỏi đã cho.

## Ranh giới bằng chứng

- `estimate_tokens()` là heuristic theo số ký tự, không phải tokenizer chính thức của provider.
- `Response quality` là heuristic dựa trên coverage và độ dài, không phải đánh giá bởi con người hoặc LLM judge.
- Test và benchmark xác nhận offline implementation, không chứng minh độ tin cậy của live model, API latency, chi phí thực tế hoặc production deployment.
- Regex extraction được thiết kế cho contract và dataset của lab. Production cần structured extraction, provenance, confidence, policy về dữ liệu cá nhân và test đa dạng hơn.

