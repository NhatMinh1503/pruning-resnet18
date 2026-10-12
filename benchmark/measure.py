import argparse
import csv
import os
import re
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from thop import profile
import torchvision.models as models


def parse_checkpoint_info(checkpoint_path: str) -> Dict[str, str]:
    """Phân tích tên file checkpoint theo quy ước nhóm:

    - resnet18_baseline.pt
    - resnet18_{local|global}_{sparsity}_{raw|finetuned}.pt
    """
    filename = os.path.basename(checkpoint_path)

    if "baseline" in filename:
        return {
            "checkpoint_name": filename,
            "method": "none",
            "sparsity_target": "0.0",
            "status": "baseline",
        }

    # Regex bắt định dạng: resnet18_{method}_{sparsity}_{status}.pt
    pattern = r"resnet18_(local|global)_([\d\.]+)_([a-zA-Z]+)\.pt"
    match = re.match(pattern, filename)
    if match:
        return {
            "checkpoint_name": filename,
            "method": match.group(1),
            "sparsity_target": match.group(2),
            "status": match.group(3),
        }

    # Trường hợp tên file thử nghiệm khác
    return {
        "checkpoint_name": filename,
        "method": "custom",
        "sparsity_target": "unknown",
        "status": "custom",
    }


def calculate_actual_sparsity(model: nn.Module) -> float:
    """Đếm tổng trọng số và tính độ thưa (sparsity) thực tế trên các lớp Conv & Linear."""
    total_weights = 0
    zero_weights = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight
                total_weights += w.numel()
                zero_weights += int((w == 0).sum().item())

    return (zero_weights / total_weights) if total_weights > 0 else 0.0


def measure_flops(
    model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32)
) -> float:
    """Đo FLOPs lý thuyết trên kích thước ảnh CIFAR-10 (1, 3, 32, 32)."""
    model_eval = model.to("cpu").eval()
    dummy_input = torch.randn(*input_size, device="cpu")

    # thop tính MACs (Multiply-Accumulate operations)
    macs, _ = profile(model_eval, inputs=(dummy_input,), verbose=False)
    flops = macs * 2  # 1 MAC ~ 2 FLOPs
    return flops / 1e6  # Trả về MFLOPs


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
    dummy_input = torch.randn(*input_size, device=device)

    # Khởi động (Warm-up)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Đo thời gian trên GPU
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = []

        with torch.no_grad():
            for _ in range(runs):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        return sum(timings) / len(timings)

    # Đo thời gian trên CPU
    else:
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(dummy_input)
        total_time = time.perf_counter() - start_time
        return (total_time / runs) * 1000.0


def append_to_csv(csv_path: str, row_data: Dict[str, str]):
    """Ghi nối dòng vào results/efficiency.csv, tự tạo header nếu file chưa tồn tại."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "checkpoint_name",
        "method",
        "sparsity_target",
        "actual_sparsity",
        "status",
        "file_size_mb",
        "flops_m",
        "latency_cpu_ms",
        "latency_gpu_ms",
    ]

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def benchmark_model(checkpoint_path: str, output_csv: str = "results/efficiency.csv"):
    """Đọc checkpoint và thực hiện đo toàn diện các chỉ số."""
    print(f"\n=======================================================")
    print(f"Đang benchmark: {checkpoint_path}")
    print(f"=======================================================")

    # 1. Đo dung lượng file
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    # 2. Tải mô hình
    # ResNet-18 tiêu chuẩn cho bài toán 10 class (CIFAR-10)
    model = models.resnet18(num_classes=10)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # 3. Tính độ thưa thực tế & FLOPs
    actual_sparsity = calculate_actual_sparsity(model)
    flops_m = measure_flops(model)

    # 4. Đo độ trễ CPU
    print("-> Đang đo độ trễ trên CPU...")
    latency_cpu = measure_latency(model, device_name="cpu")

    # 5. Đo độ trễ GPU (nếu máy có GPU / chạy trên Colab)
    latency_gpu = -1.0
    if torch.cuda.is_available():
        print("-> Đang đo độ trễ trên GPU...")
        latency_gpu = measure_latency(model, device_name="cuda")
    else:
        print("-> Không phát hiện GPU, bỏ qua đo độ trễ GPU.")

    # 6. Gom thông tin và ghi vào CSV
    meta = parse_checkpoint_info(checkpoint_path)
    result_row = {
        "checkpoint_name": meta["checkpoint_name"],
        "method": meta["method"],
        "sparsity_target": meta["sparsity_target"],
        "actual_sparsity": f"{actual_sparsity:.4f}",
        "status": meta["status"],
        "file_size_mb": f"{file_size_mb:.2f}",
        "flops_m": f"{flops_m:.2f}",
        "latency_cpu_ms": f"{latency_cpu:.3f}",
        "latency_gpu_ms": f"{latency_gpu:.3f}" if latency_gpu >= 0 else "N/A",
    }

    append_to_csv(output_csv, result_row)

    print("\n--- KẾT QUẢ ĐO ĐẠC ---")
    for k, v in result_row.items():
        print(f"  {k:<18}: {v}")
    print(f"\n[OK] Đã ghi kết quả thành công vào: {output_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đo hiệu năng mô hình ResNet-18")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Đường dẫn tới file .pt cần benchmark",
    )
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="Tạo một file ResNet-18 gốc mẫu để test script ngay",
    )
    args = parser.parse_args()

    # Chế độ tạo model mẫu để thử nghiệm độc lập
    if args.create_dummy:
        os.makedirs("checkpoints", exist_ok=True)
        dummy_path = "checkpoints/resnet18_baseline.pt"
        print(f"Đang khởi tạo mô hình ResNet-18 mẫu tại {dummy_path}...")
        m = models.resnet18(num_classes=10)
        torch.save(m.state_dict(), dummy_path)
        print("Đã tạo xong model mẫu!")
        benchmark_model(dummy_path)
    elif args.checkpoint:
        benchmark_model(args.checkpoint)
    else:
        print("Vui lòng truyền đối số:")
        print("  python benchmark/measure.py --create-dummy       (để test ngay)")
        print(
            "  python benchmark/measure.py --checkpoint <path>   (đo file chỉ định)"
        )