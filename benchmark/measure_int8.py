"""Module đo lường và so sánh hiệu năng FP32 vs INT8 (Post-Training Quantization - PTQ)
cho các mô hình ResNet-18 (CIFAR-10) Baseline, Unstructured Pruning và Structured Pruning.

Sử dụng PyTorch FX Graph Mode Quantization (torch.ao.quantization.quantize_fx).
"""
import argparse
import copy
import os
import sys
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

# Lọc các cảnh báo deprecation từ torch.ao để log hiển thị sạch sẽ
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao")

# Thêm thư mục gốc của repo vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.measure import (
    SEED,
    build_model,
    load_any_model,
    measure_latency_multibatch,
    measure_zip_size_mb,
    parse_checkpoint_info,
    set_seed,
    upsert_to_csv,
)
from training.common import evaluate, get_loaders


def resolve_quant_engine(engine: Optional[str] = None) -> str:
    """Xác định và kiểm tra backend quantized engine hỗ trợ."""
    supported = torch.backends.quantized.supported_engines
    if engine is not None:
        if engine not in supported:
            raise ValueError(
                f"Quantized engine '{engine}' không được hỗ trợ trên môi trường này. "
                f"Các engine khả dụng: {supported}"
            )
        return engine

    # Nếu không chỉ định, ưu tiên 'onednn' hoặc 'fbgemm'
    for candidate in ("onednn", "fbgemm", "qnnpack"):
        if candidate in supported:
            return candidate

    valid_engines = [e for e in supported if e != "none"]
    if valid_engines:
        return valid_engines[0]
    return "onednn"


def measure_int8_sizes_mb(quantized_model: nn.Module) -> Tuple[float, float]:
    """Lưu state_dict của mô hình INT8 vào thư mục tạm và đo kích thước gốc + nén ZIP (MB)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "model_int8.pt")
        torch.save(quantized_model.state_dict(), tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        zip_size_mb = measure_zip_size_mb(tmp_path)
        return size_mb, zip_size_mb


def quantize_fx_int8(
    model: nn.Module,
    calib_loader: Any,
    num_batches: int = 10,
    engine: Optional[str] = None,
) -> Tuple[nn.Module, str]:
    """Lượng tử hóa mô hình FP32 sang INT8 bằng PyTorch FX Graph Mode Quantization."""
    selected_engine = resolve_quant_engine(engine)
    torch.backends.quantized.engine = selected_engine
    qconfig_mapping = get_default_qconfig_mapping(selected_engine)

    model_to_quant = copy.deepcopy(model).cpu().eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    # 1. Prepare FX (gắn Observers)
    prepared_model = prepare_fx(
        model_to_quant,
        qconfig_mapping,
        example_inputs=(dummy_input,),
    )

    # 2. Calibration
    set_seed(SEED)
    prepared_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if i >= num_batches:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to("cpu")
            prepared_model(images)

    # 3. Convert FX
    quantized_model = convert_fx(prepared_model)
    quantized_model.eval()
    return quantized_model, selected_engine


def benchmark_int8_model(
    checkpoint_path: str,
    train_loader: Any,
    test_loader: Optional[Any] = None,
    batch_sizes: Optional[List[int]] = None,
    calib_batches: int = 10,
    calib_mode: str = "real",
    engine: Optional[str] = None,
    eval_acc: bool = False,
    output_csv: str = "results/int8_efficiency.csv",
) -> Dict[str, str]:
    """Thực hiện đo đạc so sánh toàn diện FP32 vs INT8 trên CPU."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {checkpoint_path}")

    if batch_sizes is None:
        batch_sizes = [1, 16, 64]

    set_seed(SEED)
    print(f"\n=======================================================")
    print(f"Đang benchmark INT8: {checkpoint_path}")
    print(f"=======================================================")

    # 1. Đo dung lượng FP32
    fp32_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    fp32_zip_mb = measure_zip_size_mb(checkpoint_path)

    # 2. Nạp mô hình FP32
    model_fp32, model_type = load_any_model(checkpoint_path, device="cpu")
    model_fp32.eval()
    print(f"-> Đã phát hiện và nạp mô hình: [{model_type.upper()}]")

    # 3. Lượng tử hóa sang INT8
    print(f"-> Đang thực hiện FX Quantization (calib_batches={calib_batches}, mode={calib_mode})...")
    quantized_model, quant_engine = quantize_fx_int8(
        model_fp32, calib_loader=train_loader, num_batches=calib_batches, engine=engine
    )
    print(f"-> Hoàn tất lượng tử hóa với Quantized Engine: [{quant_engine}]")

    # 4. Đo dung lượng INT8
    int8_size_mb, int8_zip_mb = measure_int8_sizes_mb(quantized_model)
    size_reduction_pct = (1.0 - int8_size_mb / fp32_size_mb) * 100.0 if fp32_size_mb > 0 else 0.0
    zip_reduction_pct = (1.0 - int8_zip_mb / fp32_zip_mb) * 100.0 if fp32_zip_mb > 0 else 0.0

    # 5. Đo độ trễ suy luận trên CPU
    print(f"-> Đang đo độ trễ CPU (FP32 vs INT8, batch sizes: {batch_sizes})...")
    fp32_lats = measure_latency_multibatch(model_fp32, device_name="cpu", batch_sizes=tuple(batch_sizes))
    int8_lats = measure_latency_multibatch(quantized_model, device_name="cpu", batch_sizes=tuple(batch_sizes))

    # 6. Đánh giá độ chính xác (nếu bật --eval-acc)
    fp32_acc_str = "N/A"
    int8_acc_str = "N/A"
    acc_drop_str = "N/A"
    if eval_acc and test_loader is not None:
        print("-> Đang đánh giá Test Accuracy (FP32 vs INT8)...")
        device_fp32 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_fp32.type == "cuda":
            model_fp32 = model_fp32.to(device_fp32)
        fp32_acc = evaluate(model_fp32, test_loader, device_fp32)
        int8_acc = evaluate(quantized_model, test_loader, torch.device("cpu"))
        acc_drop = fp32_acc - int8_acc
        fp32_acc_str = f"{fp32_acc:.2f}"
        int8_acc_str = f"{int8_acc:.2f}"
        acc_drop_str = f"{acc_drop:.2f}"
        print(f"   Accuracy: FP32 = {fp32_acc_str}% ({device_fp32.type.upper()}) | INT8 = {int8_acc_str}% (CPU) | Acc Drop = {acc_drop_str}%")

    # 7. Gom dữ liệu theo đúng thứ tự cột
    meta = parse_checkpoint_info(checkpoint_path)
    if meta["method"] in ("custom", "unknown") and model_type == "structured":
        meta["method"] = "structured"

    row_data: Dict[str, str] = {
        "checkpoint_name": meta["checkpoint_name"],
        "method": meta["method"],
        "sparsity_target": meta["sparsity_target"],
        "status": meta["status"],
        "quant_engine": quant_engine,
        "calib_batches": str(calib_batches),
        "calib_mode": calib_mode,
        "fp32_size_mb": f"{fp32_size_mb:.2f}",
        "int8_size_mb": f"{int8_size_mb:.2f}",
        "size_reduction_pct": f"{size_reduction_pct:.2f}",
        "fp32_zip_mb": f"{fp32_zip_mb:.2f}",
        "int8_zip_mb": f"{int8_zip_mb:.2f}",
        "zip_reduction_pct": f"{zip_reduction_pct:.2f}",
    }

    for bs in batch_sizes:
        lat_fp32 = fp32_lats[bs]
        lat_int8 = int8_lats[bs]
        speedup = (lat_fp32 / lat_int8) if lat_int8 > 0 else 0.0
        row_data[f"fp32_lat_bs{bs}_ms"] = f"{lat_fp32:.3f}"
        row_data[f"int8_lat_bs{bs}_ms"] = f"{lat_int8:.3f}"
        row_data[f"speedup_bs{bs}"] = f"{speedup:.2f}"

    row_data["fp32_acc"] = fp32_acc_str
    row_data["int8_acc"] = int8_acc_str
    row_data["acc_drop"] = acc_drop_str

    upsert_to_csv(output_csv, row_data, dynamic_fieldnames=True)

    print("\n--- KẾT QUẢ ĐO ĐẠC SO SÁNH FP32 vs INT8 ---")
    for k, v in row_data.items():
        print(f"  {k:<24}: {v}")
    print(f"\n[OK] Đã cập nhật kết quả thành công vào: {output_csv}\n")
    return row_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đo đạc và so sánh hiệu năng FP32 vs INT8 (ResNet-18 CIFAR-10)")
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
        help="Quét và đo toàn bộ checkpoint trong thư mục",
    )
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="Tạo file ResNet-18 mẫu (resnet18_dummy.pt) để test ngay",
    )
    parser.add_argument(
        "--create-dummy-structured",
        action="store_true",
        help="Tạo file ResNet-18 structured mẫu (resnet18_structured_0.5_dummy.pt) để test ngay",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 16, 64],
        help="Danh sách batch size cần đo trên CPU, vd: --batch-sizes 1 16 64",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=10,
        help="Số lượng batch dùng để calibration (mặc định 10)",
    )
    parser.add_argument(
        "--dummy-calib",
        action="store_true",
        help="Hiệu chuẩn nhanh bằng FakeData (không cần tải CIFAR-10)",
    )
    parser.add_argument(
        "--eval-acc",
        action="store_true",
        help="Đánh giá Top-1 Test Accuracy của cả FP32 và INT8 trên tập test",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Chỉ định quantized engine (vd: onednn, fbgemm, qnnpack)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Thư mục chứa dataset CIFAR-10",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/int8_efficiency.csv",
        help="Đường dẫn file CSV lưu kết quả",
    )
    args = parser.parse_args()

    set_seed(SEED)

    # 1. Kiểm tra có ít nhất 1 hành động hợp lệ trước khi khởi tạo DataLoader
    has_action = (
        args.create_dummy
        or args.create_dummy_structured
        or (args.checkpoint and os.path.exists(args.checkpoint))
        or (args.checkpoint_dir and os.path.exists(args.checkpoint_dir))
    )
    if not has_action:
        if args.checkpoint and not os.path.exists(args.checkpoint):
            print(f"[Lỗi] Không tìm thấy file checkpoint: {args.checkpoint}")
            sys.exit(1)
        if args.checkpoint_dir and not os.path.exists(args.checkpoint_dir):
            print(f"[Lỗi] Không tìm thấy thư mục: {args.checkpoint_dir}")
            sys.exit(1)
        print("Vui lòng truyền đối số:")
        print("  python benchmark/measure_int8.py --create-dummy --dummy-calib          (test FP32 vs INT8 mẫu)")
        print("  python benchmark/measure_int8.py --create-dummy-structured --dummy-calib (test Structured INT8 mẫu)")
        print("  python benchmark/measure_int8.py --checkpoint <path>                   (đo 1 file)")
        print("  python benchmark/measure_int8.py --checkpoint-dir <folder>             (đo cả thư mục)")
        print("  Tùy chọn: --dummy-calib --eval-acc --batch-sizes 1 16 64")
        sys.exit(0)

    # 2. Cảnh báo nếu chạy dummy-calib cùng eval-acc
    if args.dummy_calib and args.eval_acc:
        print("[WARNING] Đang chạy --dummy-calib (FakeData): kết quả accuracy chỉ dùng để test pipeline, KHÔNG dùng cho báo cáo!")

    # 3. Khởi tạo DataLoader đúng 1 lần duy nhất
    print(f"-> Khởi tạo DataLoader (dummy_calib={args.dummy_calib}, data_dir='{args.data_dir}')...")
    train_loader, test_loader = get_loaders(
        data_dir=args.data_dir,
        batch_size=128,
        num_workers=0,
        smoke=args.dummy_calib,
        seed=SEED,
    )

    # 4. Điều phối theo tham số dòng lệnh
    if args.create_dummy:
        os.makedirs("checkpoints", exist_ok=True)
        dummy_path = "checkpoints/resnet18_dummy.pt"
        print(f"Đang khởi tạo mô hình ResNet-18 CIFAR mẫu tại {dummy_path}...")
        m = build_model()
        torch.save(m.state_dict(), dummy_path)
        print("Đã tạo xong model mẫu (không ghi đè baseline thật)!")
        benchmark_int8_model(
            dummy_path,
            train_loader=train_loader,
            test_loader=test_loader if args.eval_acc else None,
            batch_sizes=args.batch_sizes,
            calib_batches=args.calib_batches,
            calib_mode="dummy" if args.dummy_calib else "real",
            engine=args.engine,
            eval_acc=args.eval_acc,
            output_csv=args.output_csv,
        )
    elif args.create_dummy_structured:
        os.makedirs("checkpoints", exist_ok=True)
        dummy_path = "checkpoints/resnet18_structured_0.5_dummy.pt"
        print(f"Đang khởi tạo mô hình ResNet-18 Structured mẫu (ratio=0.5) tại {dummy_path}...")
        from pruning.structured import channel_prune, save_structured

        m = build_model()
        m_struct = channel_prune(m, 0.5)
        save_structured(m_struct, dummy_path)
        print("Đã tạo xong model structured mẫu!")
        benchmark_int8_model(
            dummy_path,
            train_loader=train_loader,
            test_loader=test_loader if args.eval_acc else None,
            batch_sizes=args.batch_sizes,
            calib_batches=args.calib_batches,
            calib_mode="dummy" if args.dummy_calib else "real",
            engine=args.engine,
            eval_acc=args.eval_acc,
            output_csv=args.output_csv,
        )
    elif args.checkpoint_dir:
        files = sorted(
            f
            for f in os.listdir(args.checkpoint_dir)
            if f.startswith("resnet18_")
            and f.endswith(".pt")
            and not f.endswith("_dummy.pt")
            and f != "resnet18_dummy.pt"
        )
        print(f"Tìm thấy {len(files)} checkpoint hợp lệ trong '{args.checkpoint_dir}'")
        for f in files:
            benchmark_int8_model(
                os.path.join(args.checkpoint_dir, f),
                train_loader=train_loader,
                test_loader=test_loader if args.eval_acc else None,
                batch_sizes=args.batch_sizes,
                calib_batches=args.calib_batches,
                calib_mode="dummy" if args.dummy_calib else "real",
                engine=args.engine,
                eval_acc=args.eval_acc,
                output_csv=args.output_csv,
            )
    elif args.checkpoint:
        benchmark_int8_model(
            args.checkpoint,
            train_loader=train_loader,
            test_loader=test_loader if args.eval_acc else None,
            batch_sizes=args.batch_sizes,
            calib_batches=args.calib_batches,
            calib_mode="dummy" if args.dummy_calib else "real",
            engine=args.engine,
            eval_acc=args.eval_acc,
            output_csv=args.output_csv,
        )
