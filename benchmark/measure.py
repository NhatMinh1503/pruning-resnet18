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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from thop import profile

# Thêm thư mục gốc của repo vào sys.path để import được training/ và pruning/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.common import build_model, load_weights

SEED = 42

STANDARD_FIELDNAMES = [
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

_BASELINE_CONV_LINEAR_WEIGHTS: Optional[int] = None


def set_seed(seed: int = SEED) -> None:
    """Cố định seed = 42 theo quy ước chung của nhóm."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_checkpoint_info(checkpoint_path: str) -> Dict[str, str]:
    """Phân tích tên file checkpoint theo quy ước nhóm:
    - resnet18_baseline.pt -> method='baseline' (thống nhất với accuracy.csv)
    - resnet18_{local|global|structured}_{sparsity}_{raw|finetuned|dummy}.pt
    - resnet18_structured_{sparsity}.pt -> status='raw'
    """
    filename = os.path.basename(checkpoint_path)

    if "baseline" in filename:
        return {
            "checkpoint_name": filename,
            "method": "baseline",
            "sparsity_target": "0.0",
            "status": "baseline",
        }

    # Pattern chính: resnet18_{local|global|structured}_{sparsity}_{status}.pt
    pattern = r"resnet18_(local|global|structured)_([\d\.]+)_([a-zA-Z0-9_-]+)\.pt"
    match = re.match(pattern, filename)
    if match:
        return {
            "checkpoint_name": filename,
            "method": match.group(1),
            "sparsity_target": match.group(2),
            "status": match.group(3),
        }

    # Pattern dự phòng: resnet18_structured_{sparsity}.pt (không có hậu tố status -> mặc định 'raw')
    pattern_nostatus = r"resnet18_(structured)_([\d\.]+)\.pt"
    match_nostatus = re.match(pattern_nostatus, filename)
    if match_nostatus:
        return {
            "checkpoint_name": filename,
            "method": match_nostatus.group(1),
            "sparsity_target": match_nostatus.group(2),
            "status": "raw",
        }

    if "dummy" in filename:
        return {
            "checkpoint_name": filename,
            "method": "dummy",
            "sparsity_target": "0.0",
            "status": "dummy",
        }

    return {
        "checkpoint_name": filename,
        "method": "custom",
        "sparsity_target": "unknown",
        "status": "custom",
    }


def load_any_model(checkpoint_path: str, device: str = "cpu") -> Tuple[nn.Module, str]:
    """Tự động phát hiện checkpoint Unstructured hay Structured và load đúng cách.
    Trả về: (model, model_type) với model_type in ['unstructured', 'structured'].
    """
    obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(obj, dict) and "widths" in obj and "model" in obj:
        from pruning.structured import load_structured

        model = load_structured(checkpoint_path, device=device)
        return model, "structured"
    else:
        model = load_weights(build_model(), checkpoint_path, device=device)
        return model, "unstructured"


def count_conv_linear_weights(model: nn.Module) -> int:
    """Tổng số phần tử trọng số weight của Conv2d và Linear."""
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, "weight") and module.weight is not None:
                total += module.weight.numel()
    return total


def get_baseline_conv_linear_weights() -> int:
    """Lấy số lượng trọng số Conv2d/Linear của mô hình gốc build_model() (có cache)."""
    global _BASELINE_CONV_LINEAR_WEIGHTS
    if _BASELINE_CONV_LINEAR_WEIGHTS is None:
        base = build_model()
        _BASELINE_CONV_LINEAR_WEIGHTS = count_conv_linear_weights(base)
    return _BASELINE_CONV_LINEAR_WEIGHTS


def count_model_params(model: nn.Module) -> Tuple[float, float]:
    """Đếm tổng số tham số và số tham số khác 0 (đơn vị: triệu tham số, M) trên model.parameters()."""
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum(int((p != 0).sum().item()) for p in model.parameters())
    return total_params / 1e6, nonzero_params / 1e6


def calculate_actual_sparsity(model: nn.Module) -> float:
    """Đếm tổng trọng số và tính độ thưa (sparsity) thực tế trên các lớp Conv2d & Linear (dành cho unstructured)."""
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


def measure_latency_multibatch(
    model: nn.Module,
    device_name: str,
    batch_sizes: Tuple[int, ...] = (1, 16, 64),
    warmup: int = 25,
    runs: int = 100,
) -> Dict[int, float]:
    """Đo độ trễ trên nhiều kích thước batch với warm-up và đồng bộ CUDA.
    Trước mỗi vòng lặp bs, nếu là CUDA thì giải phóng bộ nhớ đệm (empty_cache).
    """
    results: Dict[int, float] = {}
    for bs in batch_sizes:
        if device_name == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        lat = measure_latency(
            model,
            device_name=device_name,
            input_size=(bs, 3, 32, 32),
            warmup=warmup,
            runs=runs,
        )
        results[bs] = lat
    return results


def upsert_to_csv(
    csv_path: str,
    row_data: Dict[str, str],
    dynamic_fieldnames: bool = False,
) -> None:
    """Ghi hoặc cập nhật dòng vào file CSV theo checkpoint_name (tránh trùng dòng).
    Nếu dynamic_fieldnames=True (hoặc file đích khác results/efficiency.csv),
    tự động mở rộng danh sách cột theo row_data.keys().
    """
    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    if dynamic_fieldnames or os.path.normpath(csv_path) != os.path.normpath("results/efficiency.csv"):
        existing_keys: List[str] = []
        if os.path.exists(csv_path):
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    existing_keys = list(reader.fieldnames)
        fieldnames = existing_keys[:]
        for k in row_data.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    else:
        fieldnames = STANDARD_FIELDNAMES

    rows: List[Dict[str, str]] = []
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("checkpoint_name") != row_data["checkpoint_name"]:
                    clean_row = {k: r.get(k, "") for k in fieldnames}
                    rows.append(clean_row)

    clean_row_data = {k: row_data.get(k, "") for k in fieldnames}
    rows.append(clean_row_data)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_model(
    checkpoint_path: str,
    output_csv: str = "results/efficiency.csv",
    batch_sizes: Optional[List[int]] = None,
) -> Dict[str, str]:
    """Đọc checkpoint (tự động nhận diện Unstructured / Structured) và thực hiện đo toàn diện các chỉ số."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {checkpoint_path}")

    set_seed(SEED)
    print(f"\n=======================================================")
    print(f"Đang benchmark: {checkpoint_path}")
    print(f"=======================================================")

    # 1. Đo dung lượng file gốc & file nén zip
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    zip_size_mb = measure_zip_size_mb(checkpoint_path)

    # 2. Tải mô hình (tự động nhận diện Structured hoặc Unstructured)
    model, model_type = load_any_model(checkpoint_path)
    model.eval()
    print(f"-> Đã phát hiện và nạp mô hình: [{model_type.upper()}]")

    # 3. Tính số lượng tham số & độ thưa thực tế
    params_m, nonzero_params_m = count_model_params(model)
    if model_type == "structured":
        base_w = get_baseline_conv_linear_weights()
        curr_w = count_conv_linear_weights(model)
        actual_sparsity = 1.0 - (curr_w / base_w) if base_w > 0 else 0.0
    else:
        actual_sparsity = calculate_actual_sparsity(model)

    flops_m = measure_flops(model)
    _, flops_sparse_m = measure_flops_sparse(model)

    # 4 & 5. Đo độ trễ suy luận (hỗ trợ đơn hoặc đa batch size)
    use_multibatch = batch_sizes is not None and len(batch_sizes) > 0
    bs_list = batch_sizes if use_multibatch else [1]

    print(f"-> Đang đo độ trễ trên CPU (batch sizes: {bs_list})...")
    lats_cpu = measure_latency_multibatch(model, device_name="cpu", batch_sizes=tuple(bs_list))
    latency_cpu = lats_cpu.get(1, lats_cpu[bs_list[0]])

    latency_gpu = -1.0
    lats_gpu: Dict[int, float] = {}
    if torch.cuda.is_available():
        print(f"-> Đang đo độ trễ trên GPU (batch sizes: {bs_list})...")
        lats_gpu = measure_latency_multibatch(model, device_name="cuda", batch_sizes=tuple(bs_list))
        latency_gpu = lats_gpu.get(1, lats_gpu[bs_list[0]])
    else:
        print("-> Không phát hiện GPU, bỏ qua đo độ trễ GPU.")

    # 6. Gom thông tin và cập nhật vào CSV
    meta = parse_checkpoint_info(checkpoint_path)
    if meta["method"] in ("custom", "unknown") and model_type == "structured":
        meta["method"] = "structured"

    result_row: Dict[str, str] = {
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

    if use_multibatch:
        result_row["params_m"] = f"{params_m:.3f}"
        result_row["nonzero_params_m"] = f"{nonzero_params_m:.3f}"
        for bs in bs_list:
            result_row[f"latency_cpu_bs{bs}_ms"] = f"{lats_cpu[bs]:.3f}"
            result_row[f"latency_gpu_bs{bs}_ms"] = (
                f"{lats_gpu[bs]:.3f}" if (torch.cuda.is_available() and bs in lats_gpu) else "N/A"
            )

    dynamic_mode = use_multibatch or (os.path.normpath(output_csv) != os.path.normpath("results/efficiency.csv"))
    upsert_to_csv(output_csv, result_row, dynamic_fieldnames=dynamic_mode)

    print("\n--- KẾT QUẢ ĐO ĐẠC ---")
    for k, v in result_row.items():
        print(f"  {k:<22}: {v}")
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
        "--create-dummy-structured",
        action="store_true",
        help="Tạo một file ResNet-18 structured mẫu (resnet18_structured_0.5_dummy.pt) để test script ngay",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Danh sách batch size cần đo, vd: --batch-sizes 1 16 64",
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
        benchmark_model(dummy_path, output_csv=args.output_csv, batch_sizes=args.batch_sizes)
    elif args.create_dummy_structured:
        os.makedirs("checkpoints", exist_ok=True)
        dummy_path = "checkpoints/resnet18_structured_0.5_dummy.pt"
        print(f"Đang khởi tạo mô hình ResNet-18 Structured mẫu (ratio=0.5) tại {dummy_path}...")
        from pruning.structured import channel_prune, save_structured

        m = build_model()
        m_struct = channel_prune(m, 0.5)
        save_structured(m_struct, dummy_path)
        print("Đã tạo xong model structured mẫu!")
        benchmark_model(dummy_path, output_csv=args.output_csv, batch_sizes=args.batch_sizes)
    elif args.checkpoint_dir:
        if not os.path.exists(args.checkpoint_dir):
            print(f"[Lỗi] Không tìm thấy thư mục: {args.checkpoint_dir}")
        else:
            files = sorted(
                f
                for f in os.listdir(args.checkpoint_dir)
                if f.startswith("resnet18_") and f.endswith(".pt") and not f.endswith("_dummy.pt") and f != "resnet18_dummy.pt"
            )
            print(f"Tìm thấy {len(files)} checkpoint hợp lệ trong '{args.checkpoint_dir}'")
            for f in files:
                benchmark_model(
                    os.path.join(args.checkpoint_dir, f),
                    output_csv=args.output_csv,
                    batch_sizes=args.batch_sizes,
                )
    elif args.checkpoint:
        benchmark_model(args.checkpoint, output_csv=args.output_csv, batch_sizes=args.batch_sizes)
    else:
        print("Vui lòng truyền đối số:")
        print("  python benchmark/measure.py --create-dummy                 (test unstructured mẫu)")
        print("  python benchmark/measure.py --create-dummy-structured      (test structured mẫu)")
        print("  python benchmark/measure.py --checkpoint <path>            (đo 1 file)")
        print("  python benchmark/measure.py --checkpoint-dir <folder>      (đo cả thư mục)")
        print("  Tùy chọn: --batch-sizes 1 16 64                            (đo nhiều batch size)")
