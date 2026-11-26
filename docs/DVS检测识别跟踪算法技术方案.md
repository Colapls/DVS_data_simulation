# DVS+RGB 事件流目标检测 / 识别 / 跟踪算法技术方案（工程实施版）

> 版本：v1.1（2026-11-11）　适用数据：CARLA 同步模式采集的 DVS+RGB 事件流数据集（汽车 / 摩托车 / 行人）
> v1.1 变更：① 识别收敛为一级（检测头三分类），移除二级细粒度识别及 ReID 分支；② 全文改为工程实施口径——每模块给出目录结构、脚本骨架、配置示例、命令与验收标准，研发人员可直接照做。
> 定位：算法链路验证 + 基线模型留档（2026 Q4 考核），并为 DVSync 真实数据微调预留全部接口。

---

## 1. 任务定义与总体架构

三个任务的边界（避免"检测/识别"混用）：

| 任务 | 英文 | 输入 | 输出 | 实现位置 |
| --- | --- | --- | --- | --- |
| 目标检测 | Object Detection | 事件窗（50ms） | 目标 2D 框 + 置信度 | RVT 检测器 |
| 目标识别 | Object Recognition | 检测框 + 事件体素 | 类别 ∈ {car, motorcycle, pedestrian} | 检测头的分类分支（与检测同训同出，零额外成本） |
| 目标跟踪 | Multi-Object Tracking | 连续检测流 | 带 track_id 的轨迹 | ByteTrack 式关联层（检测后处理，独立模块） |

架构总览（识别已并入检测头，不再单列模块）：

```javascript
CARLA episode (events.h5 + bboxes.json + track_id)
        │
        ▼
[数据层] 50ms 因果窗切片 → 体素网格 (B,2C,H,W)   质检：事件密度 / 框-轮廓 IoU / 时间对齐
        │
        ▼
[检测+识别层] RVT-B（体素网格 → Transformer 编码 → YOLOX 头）
              输出: {box, score, cls ∈ {car, motorcycle, pedestrian}}
        │
        ▼
[跟踪层] ByteTrack 关联（双阈值 + 卡尔曼 + IoU + 事件密度先验）
         输出: {track_id, box, cls, 轨迹状态}
        │
        ▼
[评估层] mAP@0.5（留档）/ COCO mAP / 每类召回率 / HOTA / MOTA / IDF1 / 端到端延迟
```

---

## 2. 研发环境与工程目录

### 2.1 环境（Ubuntu 4090 工作站）

```bash
# CUDA 12.x 环境，Python 3.10
conda create -n dvsdet python=3.10 -y
conda activate dvsdet

# 深度学习主栈
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install hydra-core omegaconf timm  # RVT 依赖
pip install pycocotools                  # mAP 评估
pip install h5py numba opencv-python pyyaml tqdm tensorboard
pip install filterpy lap                 # 跟踪：卡尔曼 + 匈牙利匹配
pip install ultralytics                  # 路线 C（YOLO 头对照线）
```

外部仓库（clone 到 `external/`，不改动上游代码，改造通过配置文件覆盖）：

| 仓库 | 用途 | 版本 |
| --- | --- | --- |
| `https://github.com/uzh-rpg/rvt` | 主基线检测器 | master（2026-11 快照） |
| `https://github.com/JonathonLuiten/TrackEval` | MOT 评估 | master |

### 2.2 工程目录

```text
dvsdet/
├── configs/                  # 所有实验配置（一实验一文件，见 2.3）
│   ├── data/
│   ├── train/
│   └── track/
├── data/
│   ├── carla_episodes/       # 【只读】原始 CARLA 采集产物
│   ├── slices/               # 切片后的训练样本（脚本生成，可再生）
│   ├── splits/               # train.txt / val.txt / test.txt
│   └── mot/                  # MOT 格式真值与结果（TrackEval 输入）
├── external/                 # rvt / TrackEval（git submodule 管理）
├── dvsdet/
│   ├── slicing/              # 事件窗切片 + 体素化
│   ├── datasets/             # RVT 格式数据集注册与 dataloader
│   ├── augment/              # 事件特有增强
│   ├── tracking/             # ByteTrack 关联器 + MOT 格式转换
│   ├── infer/                # 在线推理管线 + 延迟基准
│   └── eval/                 # mAP / 每类召回 / TrackEval 封装
├── scripts/
│   ├── 01_slice_events.py
│   ├── 02_gen_coco_anns.py
│   ├── 03_gen_mot_gt.py
│   ├── 10_train_rvt.py
│   ├── 20_infer_sequence.py
│   ├── 21_benchmark_latency.py
│   ├── 30_eval_detection.py
│   └── 31_eval_tracking.py
├── runs/                     # 训练输出（按实验名归档，不入 git）
└── reports/                  # 选型报告 / 消融报告 / 基线留档
```

**工程纪律**：① 任何"改原始数据"的操作只写新文件，`carla_episodes/` 只读；② 每次实验的 git commit hash 写进实验记录表；③ `runs/` 只存 checkpoint 和日志，数据与代码不进 runs。

### 2.3 配置即实验（一实验一配置）

所有超参集中在一个 YAML，实验间 diff 即差异。模板 `configs/train/rvt_baseline.yaml`：

```yaml
# ============ 数据 ============
data:
  root: data/slices
  split_file: data/splits/train.txt
  val_split_file: data/splits/val.txt
  sensor_hw: [640, 360]          # 与采集分辨率一致
  window_ms: 50                  # 事件累积窗
  overlap_ms: 25                 # 相邻窗重叠
  num_time_channels: 9           # 体素网格时间通道数（2*9=18 通道）
  min_events_per_window: 10      # 低于则跳过该窗
  class_names: [car, motorcycle, pedestrian]
  class_weights: [1.0, 1.5, 2.0] # 新类摩托车 / 难点行人加权

# ============ 模型 ============
model:
  backbone: rvt_b                # 预训练权重: rvt_b / rvt_s / rvt_t
  pretrained: runs/pretrained/rvt_b_gen1.pth
  reset_head: true               # 3 分类头重初始化（GEN1 预训练是 2 类）
  temporal_memory: true          # ConvLSTM 时序记忆（防静止目标假阴性）

# ============ 训练 ============
train:
  optimizer: adamw
  lr: 2.0e-4                     # backbone；head 用 5 倍 lr（见训练脚本）
  weight_decay: 1.0e-4
  batch_size: 16
  epochs: 50
  lr_scheduler: cosine
  warmup_epochs: 3
  loss:
    cls: focal                   # 抗背景事件爆炸
    reg: giou
  augment:                       # 事件特有增强，禁用亮度/色彩类
    temporal_jitter_ratio: 0.1   # 时间轴 ±10%
    event_drop_ratio: [0.1, 0.3] # 随机丢事件，模拟低对比度
    geometry: [hflip, translate, scale]  # 几何增强全窗一致施加

# ============ 评估 ============
eval:
  iou_thresholds: [0.5, 0.75]    # mAP@0.5 为留档口径
  report_per_class_recall: true
  latency_batch1: true           # 4090 实测端到端延迟

# ============ 复现 ============
seed: 42
experiment_name: rvt_baseline_v1
```

---

## 3. 数据层（实施细节）

输入：CARLA 采集产物，每 episode 一个目录——`events.h5`（字段 `/events/t` µs 级 int64、`/events/x,y` u16、`/events/p` u8）+ `anns.json`（逐窗标注：bbox[x,y,w,h]、label、track_id、时间戳）+ `meta/`（相机内外参、地图、天气、对比度阈值）。时间戳统一 µs、坐标原点左上 y 向下（与 1Mpx 风格一致）。

### 3.1 切片与体素化（scripts/01_slice_events.py）

```python
"""把 episode 切成 50ms 训练窗并体素化。用法：
python scripts/01_slice_events.py --config configs/data/slice_carla.yaml
输出: data/slices/<seq_id>/<win_idx>.npz （体素网格 + 标注 + meta），另存 raw 事件备查
"""
import h5py, numpy as np, yaml
from pathlib import Path
from numba import njit

@njit(cache=True)
def voxelize(x, y, t, p, H, W, C, t0, t1):
    """事件按时间线性插值到 C 个时间通道，返回 (2C, H, W) float32。
    极性分离：前 C 通道 ON，后 C 通道 OFF。"""
    out = np.zeros((2 * C, H, W), np.float32)
    denom = max(t1 - t0, 1)
    for i in range(len(t)):
        c = int((t[i] - t0) / denom * (C - 1))
        ch = c if p[i] > 0 else C + c
        out[ch, y[i], x[i]] += 1.0
    return out

def main(cfg):
    for ep_dir in Path(cfg["episodes_root"]).iterdir():
        with h5py.File(ep_dir / "events.h5", "r") as f:
            t = f["/events/t"][:] ; x = f["/events/x"][:]
            y = f["/events/y"][:] ; p = f["/events/p"][:]
        anns = json.load(open(ep_dir / "anns.json"))   # frames: [{ts_us, bboxes, labels, track_ids}]
        meta = json.load(open(ep_dir / "meta/meta.json"))

        win, hop = cfg["window_us"], cfg["window_us"] - cfg["overlap_us"]
        t_start, t_end = t[0], t[-1]
        win_idx = 0
        for t0 in range(t_start, t_end - win, hop):
            t1 = t0 + win
            m = (t >= t0) & (t < t1)
            if m.sum() < cfg["min_events"]:        # <10 事件的窗跳过
                continue
            vox = voxelize(x[m], y[m], t[m], p[m],
                           cfg["H"], cfg["W"], cfg["C"], t0, t1)
            # 取窗末时刻的标注（因果：只用窗内及之前信息）
            frame = nearest_frame_le(anns["frames"], t1)
            np.savez(out_dir / f"{win_idx:06d}.npz",
                     voxel=vox, bboxes=frame["bboxes"], labels=frame["labels"],
                     track_ids=frame["track_ids"], t0=t0, t1=t1,
                     contrast_threshold=meta["dvs"]["pos_threshold"],  # 域随机化用
                     map_name=meta["map"], weather=meta["weather"])
            win_idx += 1
```

**切片口径的坑（写进 README）**：① 标注取"窗末时刻"是因果对齐，训练与在线推理一致，禁止用窗中时刻；② `static` 近静止样本（标注协议已标记）单独导出到 `data/slices_static/`，不进训练主集；③ 体素网格必须 numba 加速，纯 numpy 版慢 20 倍以上；④ 每个 npz 保留 `contrast_threshold / map / weather`，噪声注入与域随机化训练直接读。

### 3.2 split 划分（防泄漏红线）

按**地图 + 天气**划分，不按帧随机切：

| split | 地图 | 天气 | 用途 |
| --- | --- | --- | --- |
| train | Town01 / 03 / 04 / 05 | 全 6 种（夜晚/雨天正常参与训练） | 训练 |
| val | Town07 / 10 | 白天 + 夜晚各半 | 训练中超参选择 |
| test（冻结） | Town07 / 10 | 雨天 + 夜晚为主 | **仅最终评估，mAP@0.5 留档**；冻结后任何人不得再用它调参 |

生成脚本逻辑：遍历 `data/slices/` 读 npz 的 meta 字段 → 写 `train.txt / val.txt / test.txt`（每行一个样本路径）。test 集冻结后，任何"在 test 上 rerun 看结果再改模型"的行为视为违规，消融全部在 val 上做。

### 3.3 检测标注格式（RVT/1Mpx 风格 COCO json）

`scripts/02_gen_coco_anns.py` 把 npz 标注汇总为 COCO json（RVT dataloader 原生支持）：

```json
{
  "info": {"dataset": "carla_dvs_3cls", "version": "v1.1"},
  "categories": [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "motorcycle"},
    {"id": 3, "name": "pedestrian"}],
  "images": [{"id": 10001, "file_name": "sequence_000042/000013.npz",
              "width": 640, "height": 360}],
  "annotations": [{"id": 1, "image_id": 10001, "category_id": 2,
                   "bbox": [412.0, 180.0, 46.0, 62.0], "area": 2852.0,
                   "iscrowd": 0, "attributes": {"track_id": 17,
                   "contrast_threshold": 0.18, "map": "Town07", "weather": "WetNight"}}]
}
```

bbox 为 `[x, y, w, h]`（COCO 约定）。attributes 里的 `track_id` 供 MOT 真值生成（见 5.3），不得丢弃。

### 3.4 数据层验收标准（DoD）

- [ ] 随机抽 20 个 npz，体素网格投回像素平面与 RGB 对照帧目视：行人腿部、车轮轮廓有清晰事件；
- [ ] 标注框与事件轮廓 IoU 抽查 100 框，IoU < 0.1 的框占比 < 2%（超出则查投影 bug）；
- [ ] 三类别样本计数表输出（car / motorcycle / pedestrian 数量与比例），行人:汽车 ≥ 1:1.5 配比确认；
- [ ] train/val/test 无同地图同天气片段泄漏（脚本断言）；
- [ ] 体素化 50ms 窗平均耗时 < 2ms/窗（计时打印）。

---

## 4. 检测 + 识别模块（实施细节）

### 4.1 为什么识别不做独立模块

三分类（car / motorcycle / pedestrian）所需的信息（目标尺寸、事件密度、运动模式）在检测头分类分支里全部可得，事件域做独立识别器没有信息增益，反而引入框-裁剪对齐的工程复杂度。识别 = 检测头分类分支，训练一次完成，识别指标随检测评估同出（每类 AP / 每类召回）。

### 4.2 RVT 安装与三分类头改造

```bash
cd external && git clone https://github.com/uzh-rpg/rvt.git
pip install -e external/rvt        # 装 rvt 包本体
mkdir -p runs/pretrained
# GEN1 预训练权重（官方发布）:
wget -O runs/pretrained/rvt_b_gen1.pth <官方权重链接，见 rvt 仓库 README>
```

改造点只有两处，通过配置覆盖，不改上游源码：

1. **检测头分类维度**：2 → 3。在 `configs/train/rvt_baseline.yaml` 设 `model.reset_head: true`，训练脚本加载预训练权重时按 `class_names` 重建 YOLOX 分类分支（新分支用 5 倍 lr，见 4.4）；
2. **类别名映射**：dataloader 把 CARLA 标签（vehicle / motorcycle / walker）映射为 [1, 2, 3]，转换表固化在 `dvsdet/datasets/label_map.py`，GEN1 风格的公开数据（car / pedestrian）映射为 [1, 3]，保住日后混训接口。

### 4.3 训练脚本（scripts/10_train_rvt.py 骨架）

```python
"""训练入口。用法：
python scripts/10_train_rvt.py --config configs/train/rvt_baseline.yaml
"""
import hydra, torch
from rvt.models import RVTDetector            # 上游模型类
from dvsdet.datasets import build_dataloader
from dvsdet.augment import EventAugment

@hydra.main(config_path="../configs/train", config_name="rvt_baseline")
def main(cfg):
    set_seed(cfg.seed)
    model = RVTDetector(**cfg.model.arch, num_classes=len(cfg.data.class_names))
    ckpt = torch.load(cfg.model.pretrained, map_location="cpu")
    load_backbone_only(model, ckpt)             # 只迁移 backbone，head 重初始化

    opt = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": cfg.train.lr},
         {"params": model.head.parameters(),     "lr": cfg.train.lr * 5}],  # 新头学快
        weight_decay=cfg.train.weight_decay)

    for epoch in range(cfg.train.epochs):
        for vox, targets in train_loader:
            vox = EventAugment(vox, cfg.train.augment)      # 见 4.5
            loss = model(vox, targets,
                         cls_weight=cfg.data.class_weights) # focal + giou
            loss.backward(); opt.step(); opt.zero_grad()
        mAP = evaluate(model, val_loader)                   # 每 epoch 出数
        log(epoch, mAP)                                     # tensorboard + 实验记录表
        save_ckpt(f"runs/{cfg.experiment_name}/last.pth")
```

训练纪律：① 实验记录表（`reports/experiment_log.csv`）每行一个实验：experiment_name、git hash、配置 diff、mAP@0.5(val)、每类召回、最终 epoch、备注；② val 上早停 patience=8；③ test 集只在实验完全冻结后跑一次。

### 4.4 关键超参与理由

| 项 | 值 | 理由 |
| --- | --- | --- |
| backbone / head lr | 2e-4 / 1e-3 | 预训练 backbone 微调、新三分类头快速学习 |
| batch / epoch | 16 / 50（早停 8） | 4090 单卡显存与 Q4 时间盒的折中 |
| 类别权重 car/moto/ped | 1.0 / 1.5 / 2.0 | 摩托为新类保护、行人为难点类（运动不规则、遮挡多、夜间事件稀疏） |
| focal loss | γ=2, α=0.25 | 背景事件远多于前景，抗类别不平衡 |
| 体素时间通道 C | 9（18 通道） | 1Mpx 官方口径，预训练权重迁移无分布偏移 |
| 输入分辨率 | 640×360 | 与采集一致；若显存吃紧降采样 2 倍（1Mpx 官方做法，须记录在案） |

### 4.5 事件特有增强（dvsdet/augment/EventAugment.py）

四条，全部保持窗内时空一致性，**禁用任何亮度/色彩类增强**（事件数据没有亮度概念）：

```python
class EventAugment:
    def __call__(self, voxel, cfg):
        # 1) 时间轴抖动：窗内事件时间整体 ±10% 抖动 → 等价于对体素按时间维弹性扰动
        if rand() < 0.5: voxel = temporal_jitter(voxel, cfg.temporal_jitter_ratio)
        # 2) 事件密度缩放：随机置零 10~30% 体素计数，模拟低对比度/远距小目标
        if rand() < 0.5: voxel = event_dropout(voxel, uniform(*cfg.event_drop_ratio))
        # 3) 背景活动噪声注入：按 npz 里 contrast_threshold 反推噪声强度，
        #    撒泊松噪声点（模拟真实器件背景活动，sim-to-real 对冲手段，见第 8 章）
        if rand() < 0.5: voxel = inject_background_noise(voxel, strength=cfg.noise_scale)
        # 4) 几何增强：hflip / translate / scale，对所有时间通道一致施加（勿逐通道独立）
        voxel = geometry_augment(voxel, cfg.geometry)
        return voxel
```

注意第 3 条：CARLA 事件是均匀采样、无真实器件噪声，不注入噪声训出的模型上真实相机会退化——该增强同时是消融第 3 组（噪声注入 on/off 对比）。

### 4.6 检测模块验收标准（DoD）

- [ ] val mAP@0.5 达到留档线（首训合理预期 ≥ 0.55，随数据量浮动，不达标先查数据配比与标注质量，不先调模型）；
- [ ] 每类召回率表输出，pedestrian 召回与 car 召回差 > 15 个点时须分析并记录（不强制拉平，但要解释）；
- [ ] motorcycle AP 非零且 ≥ car AP 的 40%（低于则触发专项补采）；
- [ ] 端到端延迟（含体素化）batch=1 实测并记录：目标 <15ms；
- [ ] test 集跑一次，mAP@0.5 写入 `reports/baseline_lock.md`（冻结留档）。

---

## 5. 目标跟踪模块（实施细节）

### 5.1 模块边界

输入：连续事件窗的检测列表 `{box, score, cls}`（来自第 4 章模型推理）。输出：带 `track_id` 的轨迹流。跟踪是纯后处理模块，不训练任何网络——这保证了检测模型升级（换 RVT 后继模型）时跟踪层零改动。依赖只有 `filterpy`（卡尔曼）与 `lap`（匈牙利匹配）。

### 5.2 关联器实现（dvsdet/tracking/byte_tracker.py 骨架）

```python
class ByteTracker:
    """ByteTrack 骨架 + 事件密度先验。每个事件窗调用一次 update(dets, voxel)。"""
    def __init__(self, cfg):
        self.t_high, self.t_low = cfg.match_score_high, cfg.match_score_low   # 0.6 / 0.1
        self.max_lost = cfg.max_lost_frames        # 轨迹消失保留窗数: 5
        self.min_hits = cfg.min_hits_to_confirm    # 确认所需命中数: 2
        self.w_iou, self.w_motion, self.w_density = cfg.association_weights    # 1.0/0.5/0.3

    def update(self, dets, voxel):
        # 1) 卡尔曼预测所有活动轨迹
        for trk in self.tracks: trk.predict()
        # 2) 高分匹配：score>t_high 的检测 vs 活动轨迹，代价 = IoU + 运动马氏距离
        cost = self.association_cost(dets_high, self.tracks, voxel)
        matches, u_dets, u_trks = hungarian_match(cost, gate=0.6)
        # 3) 低分二次匹配：0.1<t_low<score<0.6 的检测补偿 u_trks（行人遮挡/事件稀疏场景降 ID switch 的关键）
        cost = self.association_cost(dets_low, u_trks, voxel)
        matches2, _, u_trks = hungarian_match(cost, gate=0.5)
        # 4) 生命周期：u_dets 建新轨迹（tentative）；u_trks 连续丢失>max_lost 删除
        # 5) 输出确认轨迹（含 track_id, box, cls, age）
```

**事件密度先验**（本项目相对原版 ByteTrack 的改造点，也是消融与论文创新储备）：

```python
def density_prior(det, voxel):
    """框内事件体素计数 / 框面积 → 归一化密度。刚体目标（car/motorcycle）密度
    显著高于行人；作为弱特征加入关联代价，帮助'同位置新出现目标'与
    '旧目标重现'的判别。对遮挡后的重匹配尤其有效。"""
    x, y, w, h = det["box"]
    region = voxel[:, int(y):int(y+h), int(x):int(x+w)]
    return region.sum() / max(w * h, 1.0)
```

配置 `configs/track/bytetrack_default.yaml`：

```yaml
match_score_high: 0.6
match_score_low: 0.1
max_lost_frames: 5
min_hits_to_confirm: 2
association_weights: {iou: 1.0, motion: 0.5, density: 0.3}
use_density_prior: true      # 消融开关：false 即原版 ByteTrack 对照
```

### 5.3 MOT 真值生成（scripts/03_gen_mot_gt.py）

从检测 COCO json 的 `attributes.track_id` 直接生成 TrackEval 的 MOT 格式真值，零额外标注：

```text
data/mot/gt/train/sequence_000042.txt      # TrackEval MOTChallenge 格式:
# <frame>, <id>, <x>, <y>, <w>, <h>, <conf=1>, <class>, <visibility>
data/mot/gt/train/sequence_000042.ini      # TrackEval 序列元数据（fps, imWidth, imHeight）
data/mot/trackers/<exp_name>/sequence_000042.txt   # 跟踪器输出（同格式，第 7 章推理产出）
```

frame 编号 = 事件窗序号（50ms 窗 → fps=20 写入 ini）。CARLA track_id 跨 episode 会重复，生成时拼接 `episode_id * 100000 + track_id` 保证全局唯一。

### 5.4 跟踪评估（scripts/31_eval_tracking.py）

```bash
python scripts/31_eval_tracking.py \
  --gt data/mot/gt/test --tracker data/mot/trackers/rvt_baseline_v1 \
  --metrics HOTA MOTA IDF1 --split test
```

评估纪律：① 指标看 HOTA（分解 DetA / AssA，分别归因检测质量与关联质量）+ MOTA + IDF1；② **每类单独报**（MOTChallenge 格式 class 列区分），行人 IDF1 必须单独成行——行人是 ID switch 重灾区；③ 对比基线 = `use_density_prior: false` 的原版 ByteTrack，报消融差值。

### 5.5 跟踪模块验收标准（DoD）

- [ ] test 集 HOTA / MOTA / IDF1 出数，DetA 与第 4 章检测 mAP 的差距在合理范围（检测差 → DetA 先掉，可解释）；
- [ ] 密度先验开关的 IDF1 差值记录进消融报告（预期正增益 ≥1 个点；若为负，分析后决定保留与否并记录原因）；
- [ ] 抽查 5 段可视化视频：行人遮挡重现后 ID 保持率 ≥ 80%（人工数）；
- [ ] 跟踪单窗耗时 <1ms（1000 轨迹量级计时）。

---

## 6. 在线推理管线（实施细节）

### 6.1 序列级推理（scripts/20_infer_sequence.py）

```python
"""对一个 episode 跑 检测+识别+跟踪 全流程并落盘。用法：
python scripts/20_infer_sequence.py --ep data/carla_episodes/sequence_000042 \
       --ckpt runs/rvt_baseline_v1/best.pth --config configs/track/bytetrack_default.yaml
"""
def main():
    tracker = ByteTracker(cfg)
    for win in slice_stream(ep, window_ms=50, overlap_ms=25):   # 与训练同口径
        voxel = voxelize(win)                                   # numba, <2ms
        dets = detector(voxel)                                  # RVT 前向 ~10ms
        tracks = tracker.update(dets, voxel)                    # <1ms
        dump_mot_row(tracks); render_overlay(voxel, tracks)     # 可视化 mp4 + MOT txt
```

### 6.2 延迟基准（scripts/21_benchmark_latency.py）

测量口径写死，方便各实验横向对比：batch=1、含体素化、warmup 100 窗后测 1000 窗取均值/P95。目标：**均值 <15ms，P95 <20ms**（等效 >50Hz 有效吞吐）。结果写入实验记录表 `latency_ms_mean / latency_ms_p95` 两列。

### 6.3 推理模块验收标准（DoD）

- [ ] 同一 episode 推理输出的 MOT txt 与第 5 章评估链路一致（TrackEval 直接可读）；
- [ ] 延迟基准达标并记录；不达标时定位到体素化/前向/关联哪一段（脚本分段时间戳）；
- [ ] 可视化 mp4 抽查 3 段：框随目标稳定贴合、ID 不闪跳、类别标注正确。

---

## 7. 实验计划与考核对齐（2026 Q4）

### 7.1 三路线对比（考核要求 ≥3 路线 → 选型报告素材）

| 路线 | 表征 | 模型 | 配置名 | 回答的问题 |
| --- | --- | --- | --- | --- |
| A | 事件帧（ON/OFF 计数图，2 通道） | ResNet18 + YOLOX 头 | `train/eventframe_cnn.yaml` | 时序信息值多少 mAP？（消融基线） |
| B | 体素网格（18 通道） | RVT-B（预训练迁移） | `train/rvt_baseline.yaml` | **主基线，mAP@0.5 留档来源** |
| C | 体素网格（18 通道） | YOLO26n 头（Ultralytics） | `train/yolo26_head.yaml` | 部署延迟线，mAP-延迟曲线 |
| D（二期） | 体素网格 | SpikeYOLO（SNN） | —— | 低功耗对照，Q4 后立项 |

路线 A/C 与 B 共用第 3 章数据层，只换表征函数与模型类，dataloader 接口不变。

### 7.2 消融计划（考核要求 ≥3 组，全部在 val 上做）

| # | 变量 | 对照 | 指标 | 配置 diff |
| --- | --- | --- | --- | --- |
| 1 | 表征 | 事件帧 vs 体素网格（同骨架） | mAP@0.5 | 路线 A vs B |
| 2 | 累积窗长 | 25ms / 50ms / 100ms | mAP-延迟联合 | `data.window_ms` |
| 3 | 噪声注入 | 增强第 3 条 on / off | mAP + 真实数据子集退化率 | `train.augment` 删除一项 |
| 4（加分） | 密度先验 | `use_density_prior` t/f | IDF1 / AssA | 第 5.2 节开关 |

### 7.3 四周排期与交付物

| 周 | 动作 | 交付物 |
| --- | --- | --- |
| W1 | 数据层 DoD 全过 + 路线 B 首训出数 | 切片/体素化脚本、split 文件、实验记录表建立 |
| W2 | 路线 A/C 训完 + 消融 1/2 | 选型报告初稿（三路线对比表） |
| W3 | 跟踪接入 + 消融 3/4 + 路线 B test 冻结评估 | mAP@0.5 写入 `baseline_lock.md`（留档） |
| W4 | 报告定稿 | 选型报告 / 消融报告 / 基线留档三份 md 入 `reports/` |

---

## 8. sim-to-real 对接（为 DVSync 预留）

- DVSync 真实数据经已交付的 `dvs_rgb_pipeline.py`（raw+mp4+json 拆帧，50ms 因果窗、ON/OFF 双通道积累、窗末配对 RGB、manifest.json）产出后，**复用第 3 章同套切片脚本**（换 `--config configs/data/slice_dvsync.yaml`，仅改分辨率与窗参数），格式对齐零改造；
- 混合训练：CARLA 数据预训练 → DVSync 数据微调。真实数据整体进 fine-tune 集 + 独立 test 集，split 原则不变（按天/场景划分，不按帧切）；
- 域随机化三件套在训练侧已就位：episode 级对比度阈值随机化（0.10~0.30，npz 里 `contrast_threshold` 字段即增强强度依据）、背景活动噪声注入（增强第 3 条）、时间轴抖动（增强第 1 条）；
- 每轮迭代用 DVSync 子集测 CARLA 训练模型的退化幅度，按失败模式回调仿真采集参数（阈值分布、噪声强度、天气配比）。

## 9. 风险与对策

| 风险 | 触发信号 | 对策 |
| --- | --- | --- |
| motorcycle 样本不足 | moto AP < car AP 的 40% | 专项补采路口/换道场景 + loss 权重已 ×1.5 |
| 静止目标假阴性 | static 子集召回接近 0 | 已隔离不训练（防假目标）；上线场景靠 RVT 时序记忆缓解，报告注明边界 |
| 行人 ID switch 高 | 行人 IDF1 显著低于 MOTA | 低分二次匹配 + 密度先验 + 消失保留窗数调 5→8 |
| 体素化瓶颈 | 延迟分段时间戳显示 >2ms | numba 已就位；仍超则降采样 2 倍（记录口径变更） |
| 仿真→真实退化超预期 | DVSync 子集 mAP 跌幅 > 15 点 | 加大噪声注入强度、回调 CARLA 阈值随机化区间、尽早排真实数据补采 |

---

## 附：关键外部资源

- RVT 官方实现与 GEN1/1Mpx 预训练权重：https://github.com/uzh-rpg/rvt
- 检测 SOTA 演进参考：SMamba（AAAI'25）、SATE（NeurIPS'25）、EvRT-DETR（ICCV'25）、Ev-DTAD（arXiv 2605.08825）——均列为 RVT 后继升级候选
- 跟踪基准口径：DSEC-MOT（IEEE Access 2025）、FEMOT / FEMOTR（arXiv 2606.14094）
- 评估工具：TrackEval（HOTA/MOTA/IDF1）、pycocotools（mAP）
- ByteTrack 原始论文：arXiv 2110.06864（双阈值关联策略的依据）