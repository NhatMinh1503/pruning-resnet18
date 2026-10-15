import argparse
import copy
import csv
import os
import random
import re
import sys
import tempfile
import time
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from thop import profile

# Thêm thư mục gốc của repo vào sys.path để import được module training/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from training.common import build_model, load_weights
except ImportError:
    # Fallback an toàn nếu nhánh benchmark chưa kịp merge training/common.py từ nhánh main
    import torchvision.models as models

    def build_model(num_classes: int = 10) -> nn.Module:
        m = models.resnet18(weights=None, num_classes=num_classes)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m

    def load_weights(model: nn.Module, checkpoint_path: str) -> nn.Module:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)
        return model


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Cố định seed = 42 theo quy ước chung của nhóm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_checkpoint_info(checkpoint_path: str) -> Dict[str, str]:
    """Phân tích tên file checkpoint theo quy ước nhóm:
    - resnet18_baseline.pt -> method='baseline' (thống nhất với accuracy.csv)
    - resnet18_{local|global}_{sparsity}_{raw|finetuned}.pt
    """
    filename = os.path.basename(checkpoint_path)

    if "baseline" in filename:
        return {
            "checkpoint_name": filename,
            "method": "baseline",
            "sparsity_target": "0.0",
            "status": "baseline",
        }

    if "dummy" in filename:
        return {
            "checkpoint_name": filename,
            "method": "dummy",
            "sparsity_target": "0.0",
            "status": "dummy",
        }

    pattern = r"resnet18_(local|global)_([\d\.]+)_([a-zA-Z0-9_-]+)\.pt"
    match = re.match(pattern, filename)
    if match:
        return {
            "checkpoint_name": filename,
            "method": match.group(1),
            "sparsity_target": match.group(2),
            "status": match.group(3),
        }

    return {
        "checkpoint_name": filename,
        "method": "custom",
        "sparsity_target": "unknown",
        "status": "custom",
    }


def calculate_actual_sparsity(model: nn.Module) -> float:
    """Đếm tổng trọng số và tính độ thưa (sparsity) thực tế trên các lớp Conv2d & Linear."""
    total_weights = 0
    zero_weights = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight
                total_weights += w.numel()
                zero_weights += int((w == 0).sum().item())

    return (zero_weights / total_weights) if total_weights > 0 else 0.0


def measure_zip_size_mb(checkpoint_path: str) -> float:
    """Đo dung lượng file .pt sau khi nén ZIP (MB) để kiểm chứng khả năng nén ma trận thưa."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "model.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(checkpoint_path, arcname=os.path.basename(checkpoint_path))
        return os.path.getsize(zip_path) / (1024 * 1024)


def measure_flops(
    model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32)
) -> float:
    """Đo Dense FLOPs (MFLOPs) qua thop (1 MAC = 2 FLOPs).
    Dùng deepcopy để tránh việc thop gắn thuộc tính phụ vào model gốc.
    """
    model_eval = copy.deepcopy(model).to("cpu").eval()
    set_seed(SEED)
    dummy_input = torch.randn(*input_size, device="cpu")

    macs, _ = profile(model_eval, inputs=(dummy_input,), verbose=False)
    flops = macs * 2
    return flops / 1e6


def measure_flops_sparse(
    model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32)
) -> Tuple[float, float]:
    """FLOPs lý thuyết khi bỏ qua trọng số = 0 (Conv2d/Linear).
    Trả về (dense, sparse) theo MFLOPs, 1 MAC = 2 FLOPs.
    """
    model = model.to("cpu").eval()
    total = {"dense": 0, "sparse": 0}

    def hook(m: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        macs_per_weight = out.shape[2] * out.shape[3] if isinstance(m, nn.Conv2d) else 1
        n = out.shape[0]
        total["dense"] += m.weight.numel() * macs_per_weight * n
        total["sparse"] += int((m.weight != 0).sum().item()) * macs_per_weight * n

    hs = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    set_seed(SEED)
    with torch.no_grad():
        model(torch.randn(*input_size))
    for h in hs:
        h.remove()
    return total["dense"] * 2 / 1e6, total["sparse"] * 2 / 1e6


def measure_latency(
    model: nn.Module,
    device_name: str,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    warmup: int = 25,
    runs: int = 100,
) -> float:
    """Đo độ trễ suy luận trung bình (ms) sau khi chạy ấm (warm-up)."""
    device = torch.device(device_name)
    model = model.to(device).eval()
    set_seed(SEED)
    dummy_input = torch.randn(*input_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings: List[float] = []

        with torch.no_grad():
            for _ in range(runs):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        return sum(timings) / len(timings)
    else:
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(dummy_input)
        total_time = time.perf_counter() - start_time
        return (total_time / runs) * 1000.0


def upsert_to_csv(csv_path: str, row_data: Dict[str, str]) -> None:
    """Ghi hoặc cập nhật dòng vào results/efficiency.csv theo checkpoint_name (tránh trùng dòng)."""
    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    fieldnames = [
        "checkpoint_name",
        "method",
        "sparsity_target",
        "actual_sparsity",
        "status",
        "file_size_mb",
        "zip_size_mb",
        "flops_m",
        "flops_sparse_m",
        "latency_cpu_ms",
        "latency_gpu_ms",
    ]

    rows: List[Dict[str, str]] = []
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("checkpoint_name") != row_data["checkpoint_name"]:
                    # Đảm bảo tương thích nếu file CSV cũ chưa có đủ cột mới
                    clean_row = {k: r.get(k, "") for k in fieldnames}
                    rows.append(clean_row)

    rows.append(row_data)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_model(
    checkpoint_path: str, output_csv: str = "results/efficiency.csv"
) -> Dict[str, str]:
    """Đọc checkpoint và thực hiện đo toàn diện các chỉ số."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {checkpoint_path}")

    set_seed(SEED)
    print(f"\n=======================================================")
    print(f"Đang benchmark: {checkpoint_path}")
    print(f"=======================================================")

    # 1. Đo dung lượng file gốc & file nén zip
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    zip_size_mb = measure_zip_size_mb(checkpoint_path)

    # 2. Tải mô hình theo chuẩn ResNet-18 CIFAR-10 của nhóm
    model = load_weights(build_model(), checkpoint_path)
    model.eval()

    # 3. Tính độ thưa thực tế & FLOPs (Dense + Sparse)
    actual_sparsity = calculate_actual_sparsity(model)
    flops_m = measure_flops(model)
    _, flops_sparse_m = measure_flops_sparse(model)

    # 4. Đo độ trễ CPU
    print("-> Đang đo độ trễ trên CPU...")
    latency_cpu = measure_latency(model, device_name="cpu")

    # 5. Đo độ trễ GPU (nếu có)
    latency_gpu = -1.0
    if torch.cuda.is_available():
        print("-> Đang đo độ trễ trên GPU...")
        latency_gpu = measure_latency(model, device_name="cuda")
    else:
        print("-> Không phát hiện GPU, bỏ qua đo độ trễ GPU.")

    # 6. Gom thông tin và cập nhật vào CSV
    meta = parse_checkpoint_info(checkpoint_path)
    result_row = {
        "checkpoint_name": meta["checkpoint_name"],
        "method": meta["method"],
        "sparsity_target": meta["sparsity_target"],
        "actual_sparsity": f"{actual_sparsity:.4f}",
        "status": meta["status"],
        "file_size_mb": f"{file_size_mb:.2f}",
        "zip_size_mb": f"{zip_size_mb:.2f}",
        "flops_m": f"{flops_m:.2f}",
        "flops_sparse_m": f"{flops_sparse_m:.2f}",
        "latency_cpu_ms": f"{latency_cpu:.3f}",
        "latency_gpu_ms": f"{latency_gpu:.3f}" if latency_gpu >= 0 else "N/A",
    }

    upsert_to_csv(output_csv, result_row)

    print("\n--- KẾT QUẢ ĐO ĐẠC ---")
    for k, v in result_row.items():
        print(f"  {k:<18}: {v}")
    print(f"\n[OK] Đã cập nhật kết quả thành công vào: {output_csv}\n")
    return result_row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đo hiệu năng mô hình ResNet-18 (CIFAR-10)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Đường dẫn tới file .pt cần benchmark",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--scan-dir",
        dest="checkpoint_dir",
        type=str,
        default=None,
        help="Đo mọi file resnet18_*.pt trong thư mục (bỏ qua thư mục con như _resume/)",
    )
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="Tạo một file ResNet-18 mẫu (resnet18_dummy.pt) để test script ngay",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/efficiency.csv",
        help="Đường dẫn file CSV lưu kết quả",
    )
    args = parser.parse_args()

    set_seed(SEED)

    if args.create_dummy:
        os.makedirs("checkpoints", exist_ok=True)
        dummy_path = "checkpoints/resnet18_dummy.pt"
        print(f"Đang khởi tạo mô hình ResNet-18 CIFAR mẫu tại {dummy_path}...")
        m = build_model()
        torch.save(m.state_dict(), dummy_path)
        print("Đã tạo xong model mẫu (không ghi đè baseline thật)!")
        benchmark_model(dummy_path, output_csv=args.output_csv)
    elif args.checkpoint_dir:
        if not os.path.exists(args.checkpoint_dir):
            print(f"[Lỗi] Không tìm thấy thư mục: {args.checkpoint_dir}")
        else:
            files = sorted(
                f
                for f in os.listdir(args.checkpoint_dir)
                if f.startswith("resnet18_") and f.endswith(".pt") and f != "resnet18_dummy.pt"
            )
            print(f"Tìm thấy {len(files)} checkpoint hợp lệ trong '{args.checkpoint_dir}'")
            for f in files:
                benchmark_model(
                    os.path.join(args.checkpoint_dir, f), output_csv=args.output_csv
                )
    elif args.checkpoint:
        benchmark_model(args.checkpoint, output_csv=args.output_csv)
    else:
        print("Vui lòng truyền đối số:")
        print("  python benchmark/measure.py --create-dummy                 (để test ngay)")
        print("  python benchmark/measure.py --checkpoint <path>            (đo 1 file)")
        print("  python benchmark/measure.py --checkpoint-dir <folder>      (đo cả thư mục)")