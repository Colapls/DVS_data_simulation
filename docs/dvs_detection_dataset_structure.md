# DVS 事件流目标检测数据集 · 数据结构规范

> 目标检测任务：车辆（Vehicle）、行人（Pedestrian）
> 采集设备：RGB + DVS 融合相机（事件流与 RGB 帧硬件级同步采集，如 Prophesee EVK4 HD + RGB / iniVation DVXplorer S Duo 类设备）
> 数据模态：DVS 事件流（必需）+ 同步 RGB 帧（必需），无 IMU
> 版本：v1.1 ｜ 最后更新：2026-10-19

---

## 1. 数据集根目录结构

```text
dvs_vehicle_pedestrian_detection/
│
├── README.md                  # 数据集总说明（必读）
├── LICENSE.txt                # 使用许可
├── CHANGELOG.md               # 版本变更记录
│
├── events/                    # 【核心】事件流数据
│   ├── train/
│   │   ├── sequence_000001.h5
│   │   ├── sequence_000002.h5
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
│
├── annotations/               # 检测标注（与事件序列一一对应）
│   ├── train/
│   │   ├── sequence_000001.json
│   │   └── ...
│   ├── val/
│   └── test/                  # 测试集可只提供 image_id 占位，不含框
│
├── rgb_frames/                # 【同步】RGB 帧（与事件流硬件同步）
│   └── <split>/<sequence_id>/
│       ├── 000000.png
│       ├── 000001.png
│       └── timestamps.txt
│
├── calibration/               # 相机标定参数（DVS 与 RGB 双目标定）
│   ├── intrinsics_dvs.yaml    # DVS 内参 + 畸变系数
│   ├── intrinsics_rgb.yaml    # RGB 相机内参 + 畸变系数
│   ├── extrinsics_dvs_rgb.yaml  # DVS 与 RGB 之间的外参（联合标定）
│   ├── resolution.yaml        # 传感器分辨率等硬件参数
│   └── time_offset.yaml       # 事件流与 RGB 帧的时间偏移
│
├── splits/                    # 数据划分索引
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
│
├── event_representations/     # 【可选】预生成的事件体素网格 / 事件帧
│   └── <split>/<sequence_id>/
│       ├── event_frame_000000.npy
│       └── voxel_000000.npy
│
└── tools/                     # 官方工具脚本
    ├── visualize.py           # 事件流 + 标注可视化
    ├── convert_aedat4_to_h5.py
    ├── evaluate.py            # 检测评估（AP / mAP）
    └── requirements.txt
```

---

## 2. 文件命名与编号规则

| 项目 | 规则 | 示例 |
|---|---|---|
| 序列 ID | `sequence_` + 6 位数字，全局唯一 | `sequence_000123` |
| 事件文件 | `<sequence_id>.h5` | `sequence_000123.h5` |
| 标注文件 | `<sequence_id>.json` | `sequence_000123.json` |
| 数据划分 | 同一序列只出现在一个 split 中 | train / val / test 互斥 |

---

## 3. 事件数据格式（`events/*.h5`）

每个 HDF5 文件对应一个采集序列，包含以下数据集（dataset）：

| HDF5 路径 | 形状 | 数据类型 | 说明 |
|---|---|---|---|
| `/events/t` | (N,) | uint64 | 时间戳，单位：**微秒（µs）**，序列内从 0 开始 |
| `/events/x` | (N,) | uint16 | 像素横坐标，范围 [0, width-1] |
| `/events/y` | (N,) | uint16 | 像素纵坐标，范围 [0, height-1]，**y 轴向下** |
| `/events/p` | (N,) | int8 | 极性：1 = 亮度增加（ON），-1 = 亮度减小（OFF） |
| `/meta/sensor_width` | 标量 | uint16 | DVS 传感器宽度（如 1280） |
| `/meta/sensor_height` | 标量 | uint16 | DVS 传感器高度（如 720） |
| `/meta/sequence_id` | 字符串 | — | 与文件名一致的序列 ID |
| `/meta/synced_rgb` | 字符串 | — | 对应 RGB 帧目录路径（如 `rgb_frames/train/sequence_000123/`） |

> 约定：
> - 事件按时间戳严格递增排序；
> - 坐标原点为 DVS 像素平面左上角（与 RGB 帧坐标系可能不一致，见第 5 节双目标定）；
> - 若原始采集为 `.aedat4` / `.raw`，转换后必须保留原始文件路径于 `/meta/source_file`。

---

## 4. 标注格式（`annotations/*.json`）

采用序列级 JSON，结构如下：

```json
{
  "sequence_id": "sequence_000123",
  "sensor_width": 1280,
  "sensor_height": 720,
  "frames": [
    {
      "frame_id": "sequence_000123_000010",
      "timestamp_us": 10234567,
      "integration_time_us": 50000,
      "annotations": [
        {
          "id": 1,
          "category": "vehicle",
          "bbox": [x_min, y_min, width, height],
          "truncated": false,
          "occluded": false,
          "difficult": false
        },
        {
          "id": 2,
          "category": "pedestrian",
          "bbox": [x_min, y_min, width, height],
          "truncated": false,
          "occluded": true,
          "difficult": false
        }
      ]
    }
  ]
}
```

### 4.1 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `frame_id` | string | 唯一 ID：`<sequence_id>_<6 位帧序号>` |
| `timestamp_us` | int | 该标注对应的中心时刻（µs） |
| `integration_time_us` | int | 累积事件的窗口长度（µs），常用 50000（50 ms） |
| `category` | string | 仅限 `vehicle` / `pedestrian` 两类 |
| `bbox` | [int, int, int, int] | `[x_min, y_min, w, h]`，**RGB 帧图像坐标系**（标注基于 RGB 帧；若需映射到 DVS 坐标，用 `calibration/extrinsics_dvs_rgb.yaml` 转换，见第 5 节） |
| `truncated` / `occluded` / `difficult` | bool | 沿用 PASCAL VOC 语义 |

### 4.2 类别定义

| 类别 | 英文 | 包含范围 |
|---|---|---|
| 车辆 | `vehicle` | 轿车、卡车、公交车、摩托车、自行车等 |
| 行人 | `pedestrian` | 步行的人（含骑行者归为 vehicle 时需在 README 说明） |

---

## 5. 标定文件（`calibration/`）

融合相机的 DVS 与 RGB 为两个物理传感器，存在视差与各自畸变，使用时必须做双目标定与外参变换。

### 5.1 `intrinsics_dvs.yaml`

```yaml
camera_model: pinhole
width: 1280          # DVS 分辨率
height: 720
fx: 812.5
fy: 811.9
cx: 639.2
cy: 359.4
distortion_model: radtan
distortion_coeffs: [-0.18, 0.06, 0.0002, -0.0001, 0.0]
```

### 5.2 `intrinsics_rgb.yaml`

```yaml
camera_model: pinhole
width: 1920          # RGB 相机分辨率
height: 1080
fx: 1218.3
fy: 1217.5
cx: 958.6
cy: 539.1
distortion_model: radtan
distortion_coeffs: [-0.22, 0.09, 0.0001, 0.0003, 0.0]
```

### 5.3 `extrinsics_dvs_rgb.yaml`

> 描述 DVS 坐标系到 RGB 相机坐标系的刚体变换：`p_rgb = R · p_dvs + t`

```yaml
reference: "transform maps DVS pixel points into the RGB camera frame (after undistort+rectify)"
rotation_matrix:        # 3×3，R
  [[1.0, 0.0008, 0.0005],
   [-0.0008, 1.0, -0.0003],
   [-0.0005, 0.0003, 1.0]]
translation_m: [0.058, 0.0012, 0.0031]   # 平移向量 t（单位：米），基线约 5.8 cm
rectified: true                          # true 表示已提供去畸变+极线校正后的映射
remap_dvs_to_rgb: remap_dvs2rgb.npz      # 可选：预计算的像素映射表（H×W×2，存于 calibration/）
```

### 5.4 `resolution.yaml`

```yaml
device: Prophesee EVK4 HD + RGB / DVXplorer S Duo 类融合相机
dvs:
  native_width: 1280
  native_height: 720
rgb:
  native_width: 1920
  native_height: 1080
event_timestamps: us
polarity_encoding: {1: ON, -1: OFF}
sync_method: hardware_trigger            # 事件流与 RGB 帧的同步方式（硬件触发/软件对齐）
```

### 5.5 `time_offset.yaml`

```yaml
# 时间戳映射到事件时间基准所需的偏移量（µs）
events_to_rgb_offset_us: 0               # 硬件同步时通常为 0，以实测标定值为准
```

---

## 6. RGB 帧与事件表征

### 6.1 RGB 帧（`rgb_frames/`，同步数据）

- 图像格式：PNG（无损）或 JPG（质量 ≥ 95），三通道 BGR/RGB 需在 README 中固定声明（本数据集统一为 RGB）；
- `timestamps.txt`：每帧一个曝光中心时刻（µs），与事件时间基准一致（偏移见 `calibration/time_offset.yaml`）；
- 帧序号与 `annotations/*.json` 中的 `frame_id` 尾号一一对应；
- 典型用法：以 RGB 帧时刻为锚点，取其前后 `integration_time_us/2` 内的事件累积成事件帧，与 RGB 帧配对训练融合检测模型。

### 6.2 预生成事件表征（`event_representations/`，可选）

- `event_frame`: 固定窗口内事件的 2D 计数图（CHW 或 HWC，np.int16），分辨率默认与 DVS 一致（1280×720）；
- `voxel`: 时间维度分桶的体素网格（C×T×H×W，np.float32），供 SNN / 体素网络直接使用；
- 文件名尾号与 `frame_id` 尾号一致，保证与 RGB 帧、标注三者可索引对齐。

---

## 7. 数据划分（`splits/*.txt`)

每行一个序列 ID：

```text
sequence_000001
sequence_000002
sequence_000015
```

- `train.txt`：训练集序列列表
- `val.txt`：验证集序列列表（含完整标注，用于调参）
- `test.txt`：测试集序列列表（标注文件可为空 `annotations` 列表，仅保留 frame_id）

建议比例：train : val : test ≈ 7 : 1.5 : 1.5，且按**采集场景（地点/天气/时段）**划分，避免同一场景泄漏到不同 split。

---

## 8. 评估约定

- 指标：AP@0.5、AP@0.75、mAP@[0.5:0.95]（车辆、行人类别分别报告）
- 匹配方式：按标注窗口 `timestamp_us ± integration_time_us/2` 对齐预测
- 时间预算：可附加 latency 指标（事件到达至输出结果的时间差）
- 官方评估脚本：`tools/evaluate.py`

---

## 9. 使用注意事项（易踩坑清单）

1. **时间戳单位**：全数据集统一为微秒（µs），RGB 帧与事件共用同一时间基准（`time_offset.yaml` 修正）；
2. **双坐标系**：标注 bbox 在 **RGB 图像坐标系**，事件坐标在 **DVS 像素平面**，两者分辨率与畸变均不同，混用前必须先做去畸变 + 外参映射（`extrinsics_dvs_rgb.yaml` 或 `remap_dvs2rgb.npz`）；
3. **极性编码**：ON=1 / OFF=-1，加载后如需可映射为 0/1；
4. **标注窗口重叠**：相邻标注窗口可能重叠，训练时自行决定去重或滑动采样策略；
5. **类别平衡**：夜间/远距离小目标行人稀疏，建议报告各类别 AP 而非仅 mAP；
6. **同步验证**：RGB 帧与事件虽为硬件同步，发布前仍应抽查高速运动场景（如行驶车辆）验证对齐误差，并将实测偏移写入 `time_offset.yaml`；
7. **原始数据留存**：发布版为 HDF5 + PNG，采集原始 `.aedat4` / `.raw` 文件应归档备份并在 `/meta/source_file` 中登记。
