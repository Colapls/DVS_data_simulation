# Changelog

## 2026-11-13 检测+识别落地：训练链路、评估口径修复、网页端划分/训练/测试

### 评估口径修复（mAP 恒为 0 的根因）

症状：训练 loss 一路下降，但 val mAP@0.5 一直是 0.001 量级；而把预测框
和真值框手工算 IoU，同一个样本里能看到 0.69 / 0.93 的重叠——模型其实
已经定位准了，是评估没匹配上。

根因：**切片 npz 里的框是 xywh**（采集器 `annotations.json` 的 bbox 格式），
`build_coco_ground_truth()` 却按 xyxy 拆（`w = x2 - x1`），得到负宽高、
负面积的框，COCO 里永远匹配不上 → AP=0。

修复：`dvsdet/eval/detection_eval.py` 按 xywh 直接写 COCO bbox（COCO 的
bbox 本来就是 [x, y, w, h]，与切片格式一致）。修复后同一个权重
（只训了 2 轮）的 val mAP@0.5 从 0.001 → 0.062。

同时给 `scripts/99_selftest.py` 加了第 5 项检查「框格式契约」：切片 npz
→ dataloader → 归一化 cxcywh 的往返必须一致，评估侧与训练侧口径必须相同。
这一类"坐标系/格式不一致"的错误正是自检要挡住的。

### 训练链路

- `scripts/10_train.py`：新增 `--batch-size` / `--lr` / `--experiment` 覆盖、
  显存不足时跳过该 batch 连续 20 次才中止、每轮打印 val mAP 与每类召回。
- `scripts/02_make_splits.py` 新增**比例模式**（网页端用）：`--sequences`
  （逗号分隔或 all）+ `--ratios 70,15,15` + `--seed`。
  - 划分粒度是**整条序列**，且按「地图+天气」成组分配——同场景同天气不跨
    split，杜绝泄漏（方案 §3.2 红线）。
  - 组内子集用 0/1 背包 DP 精确求解，让窗数比例最接近设定值（此前的贪心
    会把一条 3000 窗的序列整个塞进 test，比例从 15% 冲到 35%）。
  - 产出 `data/splits/assignment.json`（分配存档，网页端与复现实验都读它）。
- `dvsdet/models/criterion.py`：目标分配（center sampling + ltrb + centerness）
  改用 numpy 实现。GPU 上每 GT 十几次小 kernel launch → 实测 644 ms/batch；
  CPU + torch 仍要 375 ms；numpy 后 <10 ms。前向只有 38 ms，损失一度是
  整个 batch 的 90%。

### 划分结果（seed=0，70/15/15）

| split | 序列 | 窗数 | 框数（car/moto/ped） |
| --- | --- | --- | --- |
| train | 000011, 000021, 000025, 000026 | 7800 | 9045 / 14710 / 10634 |
| val | 000022, 000023, 000024 | 1600 | 5445 / 5425 / 5711 |
| test | 000017, 000019, 000020 | 1600 | 6380 / 4590 / 6476 |

train/val/test 之间「地图+天气」零重叠；行人:汽车 = 1.18 / 1.05 / 1.02
（DoD 要求 ≥ 1:1.5）。默认排除 6 条无行人序列（行人 bug 修复前采的）。

### 网页端新增「③ 数据集划分与模型训练」

- `webui.py`：新增子进程链路（切片 / 划分 / 训练 / 评估）与 4 个只读接口
  （`/api/detect/sequences|splits|train/history|eval/result`）。
  训练与评估用 `dvs` conda 环境（页面顶部显示解释器路径），与网页端进程隔离。
- `webui/index.html`：新增面板——刷新切片序列清单（可勾选参与划分的序列）、
  自定义 train/val/test 比例与随机种子、生成划分、训练（轮数/批大小/
  学习率/实验名/续训）、评估（val 或 test），页面直接显示当前划分、
  每轮 val mAP@0.5 与每类召回、评估结果。

### 其他

- `scripts/96_diagnose.py`：诊断脚本——打印真值框 vs 预测框、最佳 IoU、
  分数分布。"loss 在降但 mAP≈0" 这类问题先跑它。
- `scripts/21_benchmark_latency.py`：端到端延迟基准（体素化 + 前向 + NMS）。
- `scripts/10_train.py` / `30_eval_detection.py` / `02_make_splits.py`：
  Windows 控制台默认 GBK，打印中文/符号会炸，统一改成 UTF-8 容错。

## 2026-11-11 行人采集修复 + 天气均衡 + 索引自愈

### 修复：Town10HD_Opt 序列 0 行人

症状：网页端采集时 CARLA 里几乎不生成行人，Town10HD_Opt 的 6 次采集
`spawned_traffic.walkers` 全部为 0（Town03 正常，17~26 个）。

根因（三层，逐层实测确认）：

1. **落地高度偏差**：Town10HD_Opt 的实际碰撞面比 `waypoint.transform.location.z`
   高 5~20 cm。行人在 z=0 生成一律报 `Spawn failed because of collision at
   spawn position`（实测 12/12 全失败），抬到 z+0.05/+0.20 才能成功（10/12、
   11/12）；同位置车辆生成 5/5 成功，排除蓝图问题。Town03 偏差在容差内，
   所以同一份代码在两张地图上表现不同。
2. **人行横道稀疏**：Town10HD_Opt 全图只有 16 条人行横道（Town03 有 73 条）。
   路口周边 60 m 内仅 4 条，其中 3 条离相机不足 10 m 被滤掉，剩 1 条还正好
   落在有静态碰撞体的位置。
3. **卡住检测失效**：CARLA 客户端在 actor 生成后需要一次 tick 才有真实位姿，
   此前 `get_location()` 一律返回 `(0,0,0)`。`_prune_stuck_walkers` 在这个
   时机取"起点"，位移恒为"真实位置到原点的距离"，卡住的行人也永远检测不出来。

改动：

- `dvs_collector/traffic.py`
  - `_spawn_walker`：落地高度逐级自适应（0 / +0.05 / +0.12 / +0.25 / +0.45 m），
    碰撞时同步扩大横向抖动（±1 m → ±4 m）。
  - 新增 `_roadside_pairs`：行人沿路肩步行，横向偏移到车道边缘，作为横道
    稀疏地图的行人来源。
  - `_prune_stuck_walkers`：先空跑一帧同步位姿再取起点，卡住检测恢复有效。
  - `update_walkers`：位姿未同步时不下发控制（否则方向指向原点）。
  - 目标点分配改为横道 / 人行道 / 路侧三路轮转。
  - 行人目标点为 0、或生成后存活为 0 时直接报错，不再静默产出无行人序列。
- `tools/test_traffic.py`：新增 6 个回归测试（落地高度自适应、路侧目标点、
  卡住检测用同步后的位姿、未同步位姿不下发控制、横道兜底排序）。

实测结果（真实 CARLA，5 个 Town10HD_Opt 路口 × 40 行人）：生成成功率
0% → 100%，存活 198/200，平均位移 1.46~1.57 m / 1.5 s（0.97~1.05 m/s，
真实步行速度）；Town03 从 40→17→14 存活改善到 40→33 存活。

### 天气多样化

- `dvs_collector/collector.py`：新增 `pick_balanced_weather()`，按数据集里
  各天气的使用次数选最少的那个（并列时按种子取舍）。此前是纯随机抽样，
  6 条序列里 WetSunset/CloudyNoon/SoftRainSunset 各重复 2 次，一次夜间都没有。
- `webui.py` / `webui/index.html`：采集面板新增天气下拉框，可手动指定。
- `rigs/**/*.yaml`：`weather_cycle` 补入 `HardRainNight`。
- `tools/test_weather_balance.py`：新增 4 个自检。

### 索引与数据集自愈

- `dvs_collector/dataset.py`：`register_sequence()` 改为按磁盘重建 `index_all.txt`
  （此前只追加不清理，索引 13 条、磁盘 8 条，含一条唯一的夜间序列）。
- `tools/check_dataset.py`：按磁盘上的序列目录校验，索引与磁盘不一致时单独
  报一条错，不再对已删除的目录刷一堆"缺少文件"。
- `dvs_collector/dataset.py`：把 `tools/annotate_offline.py` 一并拷进数据集
  （README 让用户用它重投影，但文件此前没有跟过去）。
- `tools/rebuild_index.py`：立即重建存量数据集的索引。

### 工具

- `tools/probe_walker_pipeline.py`：行人生成→行走→存活的闭环验收脚本。
- `tools/verify_junction_centers.py`：比对 rig 的路口中心与 CARLA 运行时报告值。

### 机位组从 5 个补到 9 个

Town10HD_Opt 地图上有 **9 个路口**，但原先只生成了 5 个机位组：
`plan_rig.py` 只为**带信号灯的路口**规划（相机装在信号灯杆上），4 个没有
信号灯的路口（j134、j664、j841、j895）被整个跳过——而这 4 个路口周边
同样有横道和可走地面，完全能采到数据。

改动：

- `tools/plan_rig.py`
  - 新增 `virtual_pole_groups()`：无信号灯路口改用**路侧虚拟杆位**——从路口
    周边 14~26 m 的可走地面 waypoint 里按方位角分桶取点（6 桶），保证多台
    相机视野互不重叠、全部朝向路口中心，安装高度用默认 4 m。
  - `build_poles()` 对无信号灯路口走虚拟杆位分支。
  - 新增 `--include-unlit`（连同无信号灯路口一起规划）和 `--only-unlit`
    （只补无信号灯的路口，不动既有机位组）；`--only-unlit` 的杆位清单写到
    `poles_unlit.json` / `poles_unlit.md`，不覆盖既有的 `poles.json`。
  - 默认行为不变（仍只处理有信号灯的路口），并在输出里提示有多少路口被跳过。

新增 4 个机位组（20 台相机）：`Town10HD_Opt_j134/j664/j841/j895.yaml`。
实测 9 个路口 × 40 行人：360 生成 → 351 存活（97.5%），平均位移
1.42~1.57 m / 1.5 s。
