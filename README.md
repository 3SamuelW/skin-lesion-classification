<details open>
<summary><strong>English</strong> | <a href="#chinese">中文</a></summary>

# Skin Lesion Classification

A fully reproducible codebase for 3-class skin lesion classification (**mel** / **nv** / **vasc**), providing two independent training and inference pipelines.

| Method | Entry point | Best for |
|---|---|---|
| **SVM** (traditional ML) | `src/train_svm.py` | Small datasets, fast experiments, interpretable features |
| **timm deep-learning baseline** | `src/train_timm.py` / `src/train_timm_suite.py` | Large datasets, end-to-end fine-tuning, architecture comparison |

---

## Background

Skin lesion diagnosis is a clinically important computer vision problem. This project targets a **3-class dermoscopy classification** task:

| Class | Description |
|---|---|
| `mel` | Melanoma — a malignant skin cancer requiring early detection |
| `nv` | Melanocytic nevus — a benign mole, the most common class |
| `vasc` | Vascular lesion — rare, including cherry angiomas and angiokeratomas |

The dataset follows the [ISIC (International Skin Imaging Collaboration)](https://www.isic-archive.com/) data format. Each image is paired with a binary segmentation mask that isolates the lesion region. All models are evaluated with **Macro F1** as the primary metric to account for class imbalance.

Two complementary approaches are implemented:

- **SVM pipeline**: hand-crafted features (colour, texture, shape, clinical, melanin) extracted from the masked lesion region → StandardScaler → PCA → SVC with RBF kernel and grid search. Interpretable and competitive on small datasets.
- **timm DL baseline**: fine-tune any architecture from the [timm](https://github.com/huggingface/pytorch-image-models) library (EfficientNet-B0, ResNet-18, ViT-B/16, SAM-ViT-B by default). Mask-crop preprocessing, ImageNet normalisation, Adam + ReduceLROnPlateau + early stopping.

---

## Repository Layout

```
skin-lesion-classification/
├── config/
│   ├── svm.yaml              SVM config (V14b best-feature set)
│   └── timm_baseline.yaml    timm DL baseline config
├── src/
│   ├── dataloader/
│   │   ├── dataset.py        Metadata construction and validation
│   │   ├── features.py       Feature extraction (colour/texture/shape/clinical/melanin)
│   │   ├── preprocessing.py  Gamma + CLAHE + Skin-only SoG preprocessing
│   │   └── split.py          base_id-grouped split (prevents augmentation leakage)
│   ├── model/
│   │   └── svm.py            SVM pipeline (StandardScaler + PCA + SVC)
│   ├── timm_baseline/
│   │   ├── data.py           PyTorch Dataset / DataLoader
│   │   ├── engine.py         Training loop, evaluation, checkpoint utilities
│   │   └── model.py          timm model creation and checkpoint serialisation
│   ├── utils/
│   │   ├── config.py         YAML config load/save, apply_data_root helper
│   │   ├── evaluation.py     Metrics, confusion matrix, augmentation robustness
│   │   └── io.py             Directory creation, JSON save
│   ├── train_svm.py          SVM training entry point
│   ├── predict_svm.py        SVM single-image prediction
│   ├── train_timm.py         timm single-model training entry point
│   ├── train_timm_suite.py   timm multi-model batch training suite
│   └── predict_timm.py       timm single-image inference
├── outputs/                  Experiment outputs (auto-created by scripts)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

Recommended: use an isolated conda environment.

```bash
conda create -n skin-cls python=3.10
conda activate skin-cls
pip install -r requirements.txt
```

For GPU support, install the CUDA-compatible torch/torchvision first following the [PyTorch official guide](https://pytorch.org/get-started/locally/), then run `pip install timm`.

---

## Data Format

Prepare your dataset directory with the following layout:

```
data/
├── image/
│   ├── 1.jpg
│   ├── 1_flip_h.jpg
│   └── ...
├── mask/
│   ├── mask_1.jpg
│   ├── mask_1_flip_h.jpg
│   └── ...
└── label.csv          must contain columns: image_id, dx
```

`label.csv` example:

```
image_id,dx
1,nv
1_flip_h,nv
2,mel
2_flip_h,mel
```

Supported augmentation suffixes (used for automatic `base_id` extraction):
`_aug1` `_aug2` `_flip_h` `_flip_v` `_rot90` `_rot180` `_rot270` `_bright` `_gamma` `_clahe` `_blur`

Pass `--data_root <your_dataset_dir>` at runtime to point scripts at any dataset directory. This overrides `image_dir`, `mask_dir`, and `label_csv` in the YAML config without editing the file.

---

## SVM — Training & Prediction

### Train

```bash
python src/train_svm.py \
    --config config/svm.yaml \
    --experiment_id svm_exp \
    --data_root data

# Re-use cached features.csv to skip re-extraction on repeated runs
python src/train_svm.py \
    --config config/svm.yaml \
    --experiment_id svm_exp \
    --data_root data \
    --reuse_features
```

Output directory `outputs/svm_exp/`:

```
config.yaml           Full config snapshot
features.csv          Extracted feature matrix
split.csv             train/val/test split assignments
model.joblib          Trained SVM pipeline
metrics.json          train/val/test metrics
predictions.csv       Per-sample predictions
robustness_detail.csv Augmentation consistency detail
confusion_matrix.png  Test-set confusion matrix
```

### Single-image prediction

```bash
python src/predict_svm.py \
    --config config/svm.yaml \
    --experiment_id svm_exp \
    --image_path data/image/1.jpg \
    --mask_path  data/mask/mask_1.jpg
```

---

## timm DL Baseline — Training & Inference

### Single-model training

```bash
python src/train_timm.py \
    --config config/timm_baseline.yaml \
    --experiment_id timm_exp \
    --data_root data
```

Default architecture: `efficientnet_b0`. Change `timm.model_name` in `config/timm_baseline.yaml` to switch.

### Batch multi-model training

```bash
# Run all 4 default models
python src/train_timm_suite.py \
    --config config/timm_baseline.yaml \
    --experiment_id suite_exp \
    --data_root data

# Run a specific subset
python src/train_timm_suite.py \
    --config config/timm_baseline.yaml \
    --experiment_id suite_exp \
    --models efficientnet_b0 vit_base_patch16_224 \
    --data_root data
```

Default model list: `efficientnet_b0` / `resnet18` / `vit_base_patch16_224` / `samvit_base_patch16`

Output directory `outputs/timm_suite_exp/`:

```
suite_summary.csv          Cross-model comparison (accuracy / balanced_acc / macro_f1)
efficientnet_b0/
  config.yaml
  metadata.csv
  split.csv
  history.csv              Per-epoch train/val curves
  model_best.pth           Best checkpoint
  metrics.json
  predictions.csv
  robustness_detail.csv
  classification_report.txt
  confusion_matrix.png
resnet18/
vit_base_patch16_224/
samvit_base_patch16/
```

### Single-image inference

```bash
# From a single-model experiment
python src/predict_timm.py \
    --config config/timm_baseline.yaml \
    --experiment_id timm_exp \
    --image_path data/image/1.jpg \
    --mask_path  data/mask/mask_1.jpg

# From a suite experiment (specify --model_name)
python src/predict_timm.py \
    --config config/timm_baseline.yaml \
    --experiment_id suite_exp \
    --model_name vit_base_patch16_224 \
    --image_path data/image/1.jpg \
    --mask_path  data/mask/mask_1.jpg
```

---

## SVM Feature Groups (V14b)

| Group | Config flag | Content |
|---|---|---|
| Colour | `use_color` | RGB/HSV/Lab per-channel mean/std/skew, HSV histograms, dark-pixel ratio |
| Advanced colour | `use_advanced_color` | Blue-White Veil, variegation, colour entropy, centre-periphery diff, colour ratios |
| Texture | `use_texture` | LBP histogram, GLCM (4-level quantisation, multi-distance, multi-angle) |
| Advanced texture | `use_advanced_texture` | Local entropy, multi-scale LBP (radii=[1,3,5,7]), border GLCM |
| Shape | `use_shape` | Area ratio, perimeter, circularity, eccentricity, solidity, extent, bbox aspect ratio, border irregularity, asymmetry |
| Advanced shape | `use_advanced_shape` | Fractal dimension, colour asymmetry, radial distance CV, border jaggedness |
| Clinical | `use_clinical` | ABCD colour count, quadrant distribution, dark blob analysis, border gradient |
| Melanin | `use_melanin_features` | Melanin index, haemoglobin index, centre-periphery melanin ratio |
| HOG | `use_hog` | 64x64 HOG (off by default) |

Preprocessing pipeline (V14b):
1. Gamma correction (gamma=1.06)
2. CLAHE on Lab L-channel (kernel=7, clip=0.03)
3. Skin-only Shades-of-Gray colour normalisation (illuminant estimated from healthy skin pixels only)

---

## Reproducibility Notes

- All random seeds are fixed at `random_state: 42` (SVM splits, cross-validation, timm training).
- Data splits are grouped by `base_id` so original images and all their augmented variants always land in the same split — preventing data leakage.
- Re-running with the same `experiment_id` overwrites `outputs/{experiment_id}/`. Use a different `experiment_id` to preserve prior results.
- SVM feature extraction is cached in `features.csv`. If you modify the feature code, delete the old cache or omit `--reuse_features`.
- The timm pipeline runs on CPU but GPU is strongly recommended. Set `timm.device: cpu` in the YAML to force CPU.

---

## Key Findings

- ViT-B/16 is the best model across all dataset sizes.
- SVM outperforms EfficientNet-B0 on small datasets and is competitive with ResNet-18.
- SVM achieves near-perfect `vasc` recall and the highest augmentation consistency on small data.
- Deep models pull ahead significantly when more data is added.

</details>

---

<a name="chinese"></a>

<details>
<summary><strong>中文</strong> | <a href="#top">English</a></summary>

# 皮肤病变分类

皮肤病变三分类（**mel** / **nv** / **vasc**）完整可复现代码包，提供两套独立的训练和推理流程。

| 方法 | 入口 | 适用场景 |
|---|---|---|
| **SVM**（传统机器学习） | `src/train_svm.py` | 小数据集，快速实验，可解释性强 |
| **timm 深度学习基线** | `src/train_timm.py` / `src/train_timm_suite.py` | 大数据集，端到端微调，对比实验 |

---

## 背景

皮肤病变诊断是临床上重要的计算机视觉问题。本项目针对**3类皮肤镜图像分类**任务：

| 类别 | 说明 |
|---|---|
| `mel` | 黑色素瘤 — 恶性皮肤癌，需要早期检测 |
| `nv` | 黑色素细胞痣 — 良性痣，最常见类别 |
| `vasc` | 血管病变 — 较少见，包括樱桃状血管瘤和角化血管瘤 |

数据集采用 [ISIC（国际皮肤成像协作）](https://www.isic-archive.com/) 数据格式。每张图像配有二值分割掩膜以定位病灶区域。所有模型以 **Macro F1** 为主要评估指标以处理类别不平衡问题。

实现了两种互补方法：

- **SVM 流程**：从掩膜病灶区域提取手工特征（颜色、纹理、形状、临床、黑素）→ StandardScaler → PCA → RBF 核 SVC + 网格搜索。在小数据集上可解释且具竞争力。
- **timm 深度学习基线**：微调 [timm](https://github.com/huggingface/pytorch-image-models) 库中的任意架构（默认：EfficientNet-B0、ResNet-18、ViT-B/16、SAM-ViT-B）。使用掩膜裁剪预处理、ImageNet 归一化、Adam + ReduceLROnPlateau + 早停策略。

---

## 目录结构

```
skin-lesion-classification/
├── config/
│   ├── svm.yaml              SVM 配置（V14b 最佳特征版）
│   └── timm_baseline.yaml    timm 深度学习基线配置
├── src/
│   ├── dataloader/
│   │   ├── dataset.py        元数据构建与验证
│   │   ├── features.py       特征提取（颜色/纹理/形状/临床/黑素）
│   │   ├── preprocessing.py  预处理（Gamma + CLAHE + Skin-only SoG）
│   │   └── split.py          按 base_id 分组划分，避免增强泄漏
│   ├── model/
│   │   └── svm.py            SVM pipeline（StandardScaler + PCA + SVC）
│   ├── timm_baseline/
│   │   ├── data.py           PyTorch Dataset / DataLoader
│   │   ├── engine.py         训练循环、评估、checkpoint 工具
│   │   └── model.py          timm 模型创建与 checkpoint 序列化
│   ├── utils/
│   │   ├── config.py         YAML 配置读写，apply_data_root
│   │   ├── evaluation.py     指标计算、混淆矩阵、增强鲁棒性
│   │   └── io.py             目录创建、JSON 保存
│   ├── train_svm.py          SVM 训练入口
│   ├── predict_svm.py        SVM 单图预测
│   ├── train_timm.py         timm 单模型训练入口
│   ├── train_timm_suite.py   timm 批量多模型训练套件
│   └── predict_timm.py       timm 单图推理
├── outputs/                  实验输出（由脚本自动创建）
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 环境配置

建议使用独立 conda 环境：

```bash
conda create -n skin-cls python=3.10
conda activate skin-cls
pip install -r requirements.txt
```

如需 GPU 支持，按 [PyTorch 官方](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本的 torch/torchvision，再 `pip install timm`。

---

## 数据集格式

每个数据集目录结构：

```
data/
├── image/
│   ├── 1.jpg
│   ├── 1_flip_h.jpg
│   └── ...
├── mask/
│   ├── mask_1.jpg
│   ├── mask_1_flip_h.jpg
│   └── ...
└── label.csv          必须包含 image_id 和 dx 两列
```

`label.csv` 示例：

```
image_id,dx
1,nv
1_flip_h,nv
2,mel
2_flip_h,mel
```

支持的增强命名后缀（用于自动识别 base_id）：
`_aug1` `_aug2` `_flip_h` `_flip_v` `_rot90` `_rot180` `_rot270` `_bright` `_gamma` `_clahe` `_blur`

运行时通过 `--data_root <数据集目录>` 指定数据，无需修改 YAML 配置文件。该参数会自动覆盖 config 中的 `image_dir`、`mask_dir`、`label_csv`。

---

## SVM 训练与预测

### 训练

```bash
python src/train_svm.py \
    --config config/svm.yaml \
    --experiment_id svm_exp \
    --data_root data

# 复用已有 features.csv（加速重复实验）
python src/train_svm.py \
    --config config/svm.yaml \
    --experiment_id svm_exp \
    --data_root data \
    --reuse_features
```

输出目录 `outputs/svm_exp/`：

```
config.yaml           本次使用的完整配置快照
features.csv          提取的特征矩阵
split.csv             train/val/test 划分结果
model.joblib          训练好的 SVM pipeline
metrics.json          train/val/test 全部指标
predictions.csv       每样本逐行预测
robustness_detail.csv 增强一致性明细
confusion_matrix.png  测试集混淆矩阵
```

### 单图预测

```bash
python src/predict_svm.py \
    --config config/svm.yaml \
    --experiment_id svm_exp \
    --image_path data/image/1.jpg \
    --mask_path  data/mask/mask_1.jpg
```

---

## timm 深度学习基线

### 单模型训练

```bash
python src/train_timm.py \
    --config config/timm_baseline.yaml \
    --experiment_id timm_exp \
    --data_root data
```

默认使用 `efficientnet_b0`。修改 `config/timm_baseline.yaml` 中的 `timm.model_name` 可切换架构。

### 批量多模型训练

```bash
# 跑全部 4 个默认模型
python src/train_timm_suite.py \
    --config config/timm_baseline.yaml \
    --experiment_id suite_exp \
    --data_root data

# 只跑指定子集
python src/train_timm_suite.py \
    --config config/timm_baseline.yaml \
    --experiment_id suite_exp \
    --models efficientnet_b0 vit_base_patch16_224 \
    --data_root data
```

默认模型列表：`efficientnet_b0` / `resnet18` / `vit_base_patch16_224` / `samvit_base_patch16`

输出目录 `outputs/timm_suite_exp/`：

```
suite_summary.csv          所有模型汇总对比表（accuracy / balanced_acc / macro_f1）
efficientnet_b0/           每个模型的完整输出目录
  config.yaml
  metadata.csv
  split.csv
  history.csv              每 epoch 训练/验证曲线
  model_best.pth           最优 checkpoint
  metrics.json
  predictions.csv
  robustness_detail.csv
  classification_report.txt
  confusion_matrix.png
resnet18/
vit_base_patch16_224/
samvit_base_patch16/
```

### 单图推理

```bash
# 单模型训练的 checkpoint
python src/predict_timm.py \
    --config config/timm_baseline.yaml \
    --experiment_id timm_exp \
    --image_path data/image/1.jpg \
    --mask_path  data/mask/mask_1.jpg

# suite 训练的 checkpoint（需指定 --model_name）
python src/predict_timm.py \
    --config config/timm_baseline.yaml \
    --experiment_id suite_exp \
    --model_name vit_base_patch16_224 \
    --image_path data/image/1.jpg \
    --mask_path  data/mask/mask_1.jpg
```

---

## SVM 特征说明（V14b）

| 特征组 | 开关 | 内容 |
|---|---|---|
| 颜色 | `use_color` | RGB/HSV/Lab 各通道 mean/std/skew，HSV 直方图，暗像素比例 |
| 进阶颜色 | `use_advanced_color` | Blue-White Veil、颜色变异度、颜色熵、中心外围差异、颜色比值 |
| 纹理 | `use_texture` | LBP 直方图，GLCM（4 级量化，多距离多角度） |
| 进阶纹理 | `use_advanced_texture` | 局部熵、多尺度 LBP（radii=[1,3,5,7]）、边界 GLCM |
| 形状 | `use_shape` | 面积比、周长、圆度、偏心率、solidity、extent、bbox 宽高比、边界不规则度、对称性 |
| 进阶形状 | `use_advanced_shape` | 分形维度、颜色不对称、径向距离变异系数、边界锯齿度 |
| 临床 | `use_clinical` | ABCD 颜色计数、四象限分布、暗斑分析、边界梯度 |
| 黑素 | `use_melanin_features` | 黑素指数、血红蛋白指数、中心外围黑素比 |
| HOG | `use_hog` | 64×64 HOG（默认关闭） |

预处理流程（V14b）：
1. Gamma 校正（gamma=1.06）
2. CLAHE（Lab L 通道，kernel=7，clip=0.03）
3. Skin-only Shades-of-Gray 颜色归一化（仅用健康皮肤像素估计光源）

---

## 复现说明

- 所有随机数种子统一设为 `random_state: 42`（SVM 划分、交叉验证、timm 训练）。
- 数据划分基于 `base_id` 分组，保证原图与所有增强变体在同一 split，避免数据泄漏。
- 同一 `experiment_id` 重复运行会覆盖 `outputs/{experiment_id}/` 下的文件。若要保留历史结果，请使用不同的 `experiment_id`。
- SVM 特征提取结果缓存在 `features.csv`；如特征代码有改动，需删除旧缓存或去掉 `--reuse_features` 参数。
- timm 训练在 CPU 上可正常运行，但建议使用 GPU。可在 YAML 中显式设置 `timm.device: cpu`。

---

## 主要结论

- ViT 在所有数据集上均为最优。
- SVM 在小数据集上优于 EfficientNet-B0，与 ResNet-18 相当。
- SVM 在 vasc 类别 recall 极高（或达 1.0），增强一致性最稳定（或达 1.00）。
- 数据量增大后，深度模型优势明显拉开。

</details>
