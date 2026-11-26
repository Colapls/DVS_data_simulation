# DVS+RGB 事件流目标检测 / 识别

按《DVS检测识别跟踪算法技术方案》（`../docs/DVS检测识别跟踪算法技术方案.md` v1.1）
实现的数据层与检测+识别模块。**识别 = 检测头的三分类分支，与检测同训同出**
（方案 §4.1：事件域做独立识别器没有信息增益）。

## 环境

```bash
# 已就绪：E:\ProgramData\miniconda3\envs\dvs
conda activate dvs
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1
pip install numba pycocotools tqdm scipy matplotlib filterpy lap omegaconf hydra-core tensorboard
```

## 快速开始

```bash
# 1) 数据层自检（先跑这个，确认数据没问题）
python scripts/00_check_data.py --samples 200

# 2) 切片：采集产物 → 体素网格训练窗
python scripts/01_slice_events.py --config configs/data/slice_carla.yaml

# 3) 划分 train/val/test（按序列切 + 按「地图+天气」成组，防泄漏）
python scripts/02_make_splits.py --ratios 70,15,15 --sequences all --seed 0

# 4) 训练
python scripts/10_train.py --config configs/train/light_detector.yaml

# 5) 评估（COCO mAP + 每类召回）
python scripts/30_eval_detection.py --ckpt runs/light_detector_v1/best.pth \
    --config configs/train/light_detector.yaml --split val

# 6) 序列级推理 + 可视化
python scripts/20_infer_sequence.py --sequence sequence_000011 \
    --ckpt runs/light_detector_v1/best.pth --config configs/train/light_detector.yaml

# 7) 延迟基准（DoD：batch=1 端到端 < 15 ms）
python scripts/21_benchmark_latency.py --ckpt runs/light_detector_v1/best.pth \
    --config configs/train/light_detector.yaml

# 自检（改完代码先跑这两个）
python scripts/99_selftest.py
python scripts/98_test_targets.py
```

## 目录

```text
Object_Detection/
├── configs/
│   ├── data/slice_carla.yaml       # 切片参数（窗/网格/时间通道）
│   └── train/light_detector.yaml   # 训练超参（一实验一配置）
├── dvsdet/
│   ├── slicing/                    # 事件切片 + 体素化（numba）
│   ├── datasets/                   # 数据集 + dataloader + 类别映射
│   ├── models/                     # 自建轻量检测器 + 损失
│   ├── augment/                    # 事件特有增强（4 条）
│   └── eval/                       # COCO mAP + 每类召回
├── scripts/
│   ├── 00_check_data.py            # 数据层自检（DoD 自动化）
│   ├── 01_slice_events.py
│   ├── 02_make_splits.py           # 固定模式 / 比例模式（网页端用）
│   ├── 10_train.py
│   ├── 20_infer_sequence.py
│   ├── 21_benchmark_latency.py     # 端到端延迟基准
│   ├── 30_eval_detection.py
│   ├── 96_diagnose.py              # 真值 vs 预测诊断（loss 降但 mAP≈0 时先跑它）
│   ├── 98_test_targets.py          # 目标分配几何单测
│   └── 99_selftest.py              # 全链路自检（5 项）
├── data/{slices,splits}/           # 切片产物与划分（可再生）
├── runs/                           # checkpoint / 训练历史 / 推理结果
└── reports/                        # 评估结果、实验记录表
```

## 与技术方案的差异（按真实数据与真实环境修正）

### 数据层

| 方案假设 | 实际数据 | 本实现的处理 |
|---|---|---|
| `anns.json`，字段 `frames[].bboxes/labels/track_ids` | `annotations.json`，字段 `frames[].annotations[]`（含 bbox/category/actor_id/ignore/quality_flags） | `slicer.py` 按真实 schema 解析；忽略框单独在 `frames[].ignored_annotations[]` |
| `events/t` 是 µs | **ns**（`meta/event_timestamp_unit`） | 体素化内部统一 ns，窗边界由帧时间戳 ×1000 得到 |
| 640×360 采集分辨率 | 采集是 **1280×720** | 体素化时直接降采样到 640×360（方案 §4.4 允许，记录在案） |
| 需自行按时间窗切事件 | `frames/event_offsets` 已给出**逐帧事件边界** | 用二分在事件数组上定位窗区间，不全量扫时间轴 |
| 标注在 `bboxes.json` | 见上 | 已适配 |
| split 按"地图+天气"三段 | 本数据集 2 张图、16 条序列 | 按「地图+天气」成组 + 0/1 背包 DP 精确配比（`--ratios 70,15,15`），train/val/test 之间地图+天气零重叠 |
| 行人:汽车 ≥ 1:1.5 | 16 条里有 6 条是修复行人 bug 之前采的（`walkers=0`） | 默认排除无行人序列（`--keep-zero-pedestrian` 可保留），当前划分 train/val/test 的行人:汽车 = 1.18 / 1.05 / 1.02 |

### 模型

方案 §7.1 的主基线是 RVT-B + GEN1 预训练迁移。**本次改为自建轻量检测器**
（用户决策），理由与实现要点：

- **输入/输出契约与 RVT 一致**：`(B, 18, H, W)` 体素网格 → `{box, score, cls ∈ {car, motorcycle, pedestrian}}`，
  后续换 RVT 后端时数据层与训练脚本不用改。
- **anchor-free（FCOS 风格）而非 YOLOX anchor-based**：标注框来自 3D 框投影，
  远处小车只有几像素宽、尺寸分布杂，anchor 尺寸难先验；FCOS 逐位置回归对
  尺度更鲁棒，且没有 anchor 匹配造成的正负样本失衡。
- **模型刻意做小**（~1.5M 参数）：本数据集只有千级样本，大模型必过拟合。
- 损失照方案 §4.4：focal(γ=2, α=0.25) + GIoU + centerness BCE，
  类别权重 car/moto/ped = 1.0/1.5/2.0。

### 增强

方案 §4.5 的四条全部实现（时间轴抖动 / 事件丢弃 / 背景活动噪声注入 / 几何增强），
**禁用亮度色彩类**。几何增强同时作用于标注框，保证框与体素不错位。

### 未实现（本次范围外）

- **目标跟踪**（方案第 5 章）：用户本次只要求"检测和识别"。数据层已保留
  `track_ids`（采集器的 `actor_id`），后续接 ByteTrack 无需改数据层。
- **RVT 后端**、**Route C（YOLO 头）**、**消融 1~4**、**TrackEval 封装**。

## 踩过的坑（改代码前先读）

- **切片 npz 里的框是 xywh**（采集器 `annotations.json` 的 bbox 格式）。
  评估侧曾按 xyxy 拆（`w = x2 - x1`），得到负宽高、COCO 永远匹配不上，
  mAP 恒为 0——而模型其实已经定位准了。`scripts/99_selftest.py` 的第 5 项
  检查专门盯这个契约。
- **训练与推理解码必须同一套回归缩放**（`criterion.REG_SCALE` 与
  `detector.reg_scale`）。曾经训练用 ×4、推理用 ×[4,8,16]，粗尺度层的框
  系统性放大 2~4 倍。
- **回归目标的内存布局**：`ltrb_target` 是 (B,H,W,4)，取通道维必须先
  `permute(0,3,1,2)` 再 reshape，否则 GIoU 恒为 0、回归学不动。
- **损失里的目标分配要放在 CPU**：逐 GT 的小张量操作在 GPU 上是 kernel
  launch 风暴（644 ms/batch），torch-CPU 也要 375 ms，改 numpy 后 <10 ms。

## 数据现状提醒

当前 `dvs_roadside_multicam` 有 16 条序列，其中 **6 条（000012~000016、
000018）是修复行人 bug 之前采的，`walkers=0`**——划分脚本默认排除它们
（`--keep-zero-pedestrian` 可保留）。补采后重跑切片 + 划分即可纳入训练。

## 数据层 DoD 对照

`scripts/00_check_data.py` 自动输出：

- [x] 每窗事件数分布与空窗比例
- [x] 框-事件覆盖率（IoU 抽查的自动化替代：框内事件像素占比 < 1% 的比例）
- [x] 三类别计数表与行人:汽车配比
- [x] 标注框短边分布（对照 DVS 可见性下限 16 px）
- [x] 体素化耗时（ms/窗，需 numba）
- [ ] 人工目视：体素投回像素 vs RGB 对照帧（需人眼，脚本给不出）
