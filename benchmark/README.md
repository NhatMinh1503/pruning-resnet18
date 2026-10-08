# benchmark/

**効率測定モジュール：FLOPs・モデルサイズ（MB）・推論レイテンシ（ms）　担当（Owner）：TRAN QUANG CANH**
チェックポイントを Google Drive から読み込み、測定結果を `results/efficiency.csv` へ書き出します。

## ベンチマークモジュール — 技術スケジュールと実装計画

本フォルダは、モデルの性能（FLOPs・保存サイズ・推論レイテンシ）の測定・分析を担当し、実験結果を `results/efficiency.csv` に記録します。

### 1. 目的とタスク範囲

- **メインソースコード：** `benchmark/measure.py`
- **測定指標（Metrics）：**
  - **Model Size：** ディスク上の `.pt` ファイルの実サイズ（MB）。
  - **FLOPs：** CIFAR-10 の標準入力テンソル `(1, 3, 32, 32)` に対して `thop` で算出（理論 Sparse FLOPs の計算も併せて行う）。
  - **Inference Latency：** ウォームアップ後の平均推論レイテンシ（ms）。CPU および GPU（Google Colab / CUDA Events）の両環境で検証。
- **標準出力：** 測定データを行単位で `results/efficiency.csv` に自動記録・追記する。

### 2. 詳細スケジュール（AIゼミの進捗に準拠）

| 段階 | 期間 | 技術タスク | ゼミのマイルストーン |
| --- | --- | --- | --- |
| P1: コアセットアップ | 10/08 – 10/18 | - `benchmark/measure.py` の作成<br>- `results/efficiency.csv` の構造作成<br>- 模擬 ResNet-18（Dummy Model）での単体テスト | `benchmark` ブランチの単体コード完成 |
| P2: ベースライン＆初期枝刈り | 10/19 – 11/04 | - training ブランチから `resnet18_baseline.pt` を受け取り、ベースライン数値を測定<br>- 初期実験の prune チェックポイント（Local / Global）を受領<br>- Sparsity vs. FLOPs / Latency グラフの予備描画 | 報告会1用のデータ準備 |
| 報告会1 | 11/05 または 11/09 | 進捗報告1（報告会1・第7週）：測定パイプラインと初期実験データを発表。 | 報告会1 |
| P3: バッチベンチマーク | 11/12 – 12/02 | - チェックポイントを一括スキャン・測定するスクリプトの作成（Raw / Finetuned の両方で Sparsity 10% – 90%）<br>- CPU（2502実習室）と GPU（Colab T4）の対照測定<br>- 「FLOPs は減少しても Latency は減少しない」現象（疎行列のハードウェア的ボトルネック）の分析 | ライトニングトーク用スライド準備 |
| 報告会2 | 12/03（予定） | 進捗報告2（報告会2 LT・第11週）：ハードウェア性能に特化したショートレポート（Lightning Talk）。 | 報告会2 LT |
| P4: 2026年 年間まとめ | 12/04 – 12/14 | - 測定データの100%を `results/efficiency.csv` へ記録完了<br>- 全分析グラフを `results/` へエクスポート<br>- 実験の再現手順を本 README に追記 | 年内の実験完了（年内目標） |
| P5: 期末レポート | 2027/01/07・01/18・01/21 | ゼミ内での最終プロジェクト発表（ゼミ内最終発表）。 | ゼミ内最終 |
| 選抜 | 2027/01/28 | ゼミ間選抜発表会（選出された場合）。 | ゼミ間選抜 |

### 3. 標準結果テーブルの構造（`results/efficiency.csv`）

データ列は以下の形式で統一して保存します：

```csv
checkpoint_name,method,sparsity_target,actual_sparsity,status,file_size_mb,flops_m,latency_cpu_ms,latency_gpu_ms
resnet18_baseline.pt,none,0.0,0.0,baseline,44.7,556.0,12.5,1.8
resnet18_local_0.5_raw.pt,local,0.5,0.50,raw,44.7,278.0,12.4,1.8
resnet18_local_0.5_finetuned.pt,local,0.5,0.50,finetuned,44.7,278.0,12.4,1.8
```

### 4. 実験結果の再現手順（段階 P4 で追記予定）

<!-- TODO: P4 期間中に、環境構築手順・コマンド・シード値などの再現手順をここに記載する -->