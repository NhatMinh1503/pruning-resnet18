# CLAUDE.md — bối cảnh dự án cho Claude

Đọc file này trước khi giúp bất kỳ thành viên nào. Trả lời bằng tiếng Việt, giải thích dễ hiểu.

## Đề tài
AI ゼミ 2026後期: nén mô hình bằng **Magnitude Pruning** (cắt tỉa trọng số theo độ lớn) trên **ResNet-18 / CIFAR-10**.
Mục tiêu: tìm trade-off "cắt bao nhiêu mà vẫn giữ được độ chính xác".

Câu hỏi nghiên cứu:
1. Độ thưa 10% → 90% ảnh hưởng accuracy thế nào (pruning phi cấu trúc).
2. Local pruning (từng lớp) vs Global pruning (toàn mạng).
3. Hiệu quả phục hồi của Train → Prune → Fine-tune.
4. FLOPs / kích thước / độ trễ: giảm tính toán lý thuyết có thật sự nhanh hơn trên phần cứng không ("bẫy ma trận thưa": torch.nn.utils.prune chỉ nhân mask, tensor vẫn dense nên latency và kích thước file gần như không đổi).
5. (Nếu dư thời gian) so sánh với Structured Pruning theo channel. Chủ đề quan tâm thêm: Quantization.

## Thành viên và phân công
- **Hoa** (GitHub: NhatMinh1503, chủ repo): thuật toán pruning → `pruning/`, nhánh `pruning`
- **Khang** (GitHub: vonhukhang2404): train baseline, fine-tune, đánh giá accuracy → `training/`, nhánh `training`
- **Canh** (GitHub: TRANQUANGCANH068): đo FLOPs, MB, latency → `benchmark/`, nhánh `benchmark`

## Quy trình làm việc
- Viết code `.py` trong VS Code → push lên GitHub (nhánh riêng) → Pull Request, có người review → merge vào `main`.
- Chạy thật trên Google Colab (GPU): clone/pull repo, `pip install -r requirements.txt`, notebook chỉ import và gọi hàm trong repo. Không viết logic chính trong notebook, không sửa code trực tiếp trên Colab.
- File nặng (checkpoint `.pt`, dataset) để ở Google Drive chung `pruning-project/checkpoints/` và `pruning-project/results/`, không commit lên Git (.gitignore đã chặn `.pt`).
- Luôn `git pull` trước khi làm. Mỗi người chỉ sửa thư mục của mình.
- Xong một bước, nhắn chat nhóm: đã xong gì, file nằm ở đâu.

## Thứ tự bàn giao
1. Khang train baseline → `resnet18_baseline.pt` lên Drive.
2. Hoa viết `pruning/prune.py` (đã có).
3. Khang chạy Prune → Fine-tune cho mọi mức sparsity → checkpoint lên Drive, ghi `results/accuracy.csv`.
4. Canh đo trên từng checkpoint → `results/efficiency.csv`.
5. Cả nhóm gộp CSV, vẽ biểu đồ, viết báo cáo.

## Quy ước
- Tên checkpoint: `resnet18_{local|global}_{sparsity}_{raw|finetuned}.pt`, ví dụ `resnet18_global_0.5_finetuned.pt`.
- `seed = 42` cho cả nhóm. Đo latency trên cùng một loại máy, warm-up trước khi đo, lấy trung bình nhiều lần.
- `results/accuracy.csv`: method, sparsity, acc_before_finetune, acc_after_finetune, seed
- `results/efficiency.csv`: method, sparsity, flops, size_mb, latency_ms

## API của pruning/prune.py
- `local_prune(model, sparsity)` / `global_prune(model, sparsity)`: trả về **bản sao** đã cắt (model gốc không đổi), chỉ cắt `weight` của Conv2d và Linear (không cắt bias, BatchNorm).
- `count_sparsity(model)`, `layer_sparsity(model)`: đo độ thưa tổng và từng lớp.
- `make_permanent(model)`: gỡ mask, chốt số 0 vào trọng số.
- **Quan trọng:** giữ nguyên mask trong lúc fine-tune (nếu gỡ sớm, trọng số đã cắt sẽ "mọc lại"). Chỉ gọi `make_permanent` sau khi fine-tune xong, trước khi lưu checkpoint cuối.
- FLOPs đo bằng thop/fvcore không tính số 0 → cần tự tính "FLOPs hiệu dụng" theo độ thưa.

## Lưu ý khác
- Repo public: không ghi mã số sinh viên hay thông tin cá nhân vào repo.
- So sánh công bằng: cố định seed, số epoch, learning rate; nên chạy vài seed để có sai số.
