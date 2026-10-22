import os
import matplotlib.pyplot as plt
import pandas as pd

# Đường dẫn file và thư mục chuẩn theo quy ước nhóm
eff_path = "results/efficiency.csv"
output_dir = "results/figures"
os.makedirs(output_dir, exist_ok=True)

df_eff = pd.read_csv(eff_path)
base = df_eff[df_eff["status"] == "baseline"].iloc[0]

# Lọc mô hình finetuned
g_ft = df_eff[(df_eff["method"] == "global") & (df_eff["status"] == "finetuned")].sort_values("sparsity_target")
l_ft = df_eff[(df_eff["method"] == "local") & (df_eff["status"] == "finetuned")].sort_values("sparsity_target")

g_all = pd.concat([pd.DataFrame([base]), g_ft])
l_all = pd.concat([pd.DataFrame([base]), l_ft])

x = g_all["sparsity_target"] * 100

# Vẽ biểu đồ dung lượng
plt.figure(figsize=(8.5, 4.8), dpi=300)
plt.plot(x, g_all["file_size_mb"], "k--", linewidth=2.2, label="Uncompressed .pt (Dense tensor: 42.70 MB)")
plt.plot(x, g_all["zip_size_mb"], "s-", color="#2ca02c", linewidth=2.2, markersize=6, label="Global Pruning (ZIP compressed)")
plt.plot(x, l_all["zip_size_mb"], "o-", color="#17becf", linewidth=2.2, markersize=6, label="Local Pruning (ZIP compressed)")

plt.title("Model Storage on Disk: Dense .pt vs. ZIP Compression", fontsize=12, fontweight="bold", pad=10)
plt.xlabel("Pruning Sparsity Target (%)", fontsize=10, fontweight="bold")
plt.ylabel("File Size on Disk (MB)", fontsize=10, fontweight="bold")
plt.ylim(0, 50)
plt.xticks(range(0, 100, 10))
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="lower left", framealpha=0.9)
plt.tight_layout()

output_path = os.path.join(output_dir, "storage_comparison.png")
plt.savefig(output_path)
plt.close()

print(f"[OK] Đã xuất hình dung lượng chuẩn tại: {output_path}")