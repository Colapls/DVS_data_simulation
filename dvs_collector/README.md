# CARLA 路侧固定多机位 DVS/RGB 采集器

按[路侧多机位采集技术方案](../docs/路侧多机位DVS+RGB事件流数据集采集技术方案.md)实现。相机直接固定生成在 CARLA 世界坐标中，不挂载到自车；每台 DVS 相机与同机位、同分辨率和视场角的 RGB 相机配对。一个序列包含该机位组的全部视角，不使用 train/val/test 划分。

采集的同时自动产出**车辆 / 行人 / 摩托车三类**目标的逐帧 2D 标注（含可见度、遮挡、截断、困难样本标记），实现见[DVS 自动标注方案_精简可执行版](../docs/DVS自动标注方案_精简可执行版.md)。

> 本文件位于 Python 包目录中。请从项目根目录运行命令；不要在 `dvs_collector` 目录内直接执行采集。

## 代码结构

整个项目只有一个采集包，`dvs_collector/`：

```text
dvs_collector/
├── collector.py   采集主流程：场景初始化、同步采集、标注、状态恢复
├── config.py      机位组 YAML 解析、命令行参数、参数校验、依赖加载
├── dataset.py     数据集目录布局、元数据、标定文件、索引、快照写入
├── sensors.py     固定 DVS+RGB(+实例分割) 相机组的创建与标定读取
├── traffic.py     按种子生成背景车流与行人
├── classes.py     三类别蓝图映射表（car / motorcycle / pedestrian）
├── labels.py      投影、可见度、遮挡/截断/困难判定（在线与离线同一份代码）
└── storage.py     HDF5 写入器与后台落盘线程
```

`labels.py` 的 `frame_labels()` 位置参数签名保持不变，但返回的标签多了 `category_id` / `blueprint_id` / `visibility` / `quality_flags` 等字段，旧代码按 `label["category"]` 取值的仍然可用（取值从 `vehicle`/`walker` 变为 `car`/`motorcycle`/`pedestrian`）。

## 网页控制台（推荐）

`webui.py` 是一个本地网页控制台，在浏览器里点按钮就能完成采集、标注后处理、质检和结果预览：

```powershell
python .\webui.py          # 自动打开 http://127.0.0.1:8787
```

只用 Python 标准库，无需额外安装依赖，仅监听本机。页面分三块：

- **① 数据采集**：选机位组、帧数、车辆/行人数后点「开始采集」，采集时自动生成三类标注；进度条按采集器的 `采集进度 n/N 帧` 实时刷新，可随时「终止任务」（走 CTRL_BREAK 优雅停止，采集器会恢复 CARLA 状态）。还能一键启动 CARLA 并显示当前地图。
- **② 标注后处理与质检**：按钮触发 `annotate_offline.py stats`（事件统计）、`check_dataset.py`（结构质检）和离线重投影，任务输出在下方日志区。
- **③ 数据集划分与模型训练**（检测 + 识别三分类）：「事件切片」把采集产物切成体素网格训练窗；「生成划分」支持自选序列范围（勾选清单）+ train/val/test 比例，按**序列**（且按「地图+天气」成组）分配，绝不按帧随机切，杜绝泄漏；「开始训练」跑轻量检测器，「评估」给 mAP@0.5/@0.75 与每类召回。训练/评估在 `dvs` conda 环境里跑子进程（页面顶部显示解释器路径），页面上能直接看到当前划分、每轮 val 指标和评估结果。
- **④ 数据浏览与标注预览**：选序列/机位/帧号，左边画同步 RGB + 自动标注框（车=橙、摩托=品红、行人=绿，灰框=忽略/困难），右边画该帧窗口内的事件累积（ON=红、OFF=蓝），支持左右方向键翻帧。

**右上角「实时日志」按钮**开关右侧的日志面板（再次点击或点击面板外空白处收起），汇总所有后台程序的输出：采集/统计/质检子进程逐行日志、启动 CARLA 的全过程（启动命令、PID、端口等待每 10 s 报告、就绪后报告地图名）。注意 0.9.16 打包版 CARLA 的 UE4 文件日志是编译掉的，引擎内部日志拿不到；面板会兜底尾随标准日志路径，换成会写日志的 CARLA 版本后日志自动进面板。开始新任务时面板会自动展开。

CARLA 运行状态由后台线程每 2 秒自动探测（依次扫 2000–2002 端口，在线时显示端口号和地图名），页面状态灯每 3 秒刷新，「开始采集」会自动连到探测到的端口。CARLA 已在线时「启动 CARLA」按钮不会重复拉起新实例。如果状态灯显示「控制台连接断开」，是 webui.py 进程本身停了，和 CARLA 无关。

## 一键采集（双击运行）

项目根目录的 `一键采集.bat` 是交互式向导的入口：双击后会**自动启动 CARLA 仿真平台**
（默认安装位置 `D:\CARLA_0.9.16`，端口 `PORT=2000`），然后按提示输入参数并自动完成
布点 → 批量采集 → 质检 → 位置时间索引摘要。

向导会依次询问：

| 步骤 | 参数 | 默认值 |
|---|---|---|
| 1 | CARLA 主机 / 端口 / Traffic Manager 端口 | 127.0.0.1 / 2000 / 8000 |
| 2 | 是否按当前地图重新自动布点、目标地图、选择机位组 | 用当前地图 / 全部 |
| 3 | **每个机位组启用的相机个数**、每个相机采集时长、帧率 | 全部 / 90 秒 / 40 Hz |
| 4 | 车辆数、行人数、固定天气 | 80 / 40 / 按种子随机 |
| 5 | 输出目录、采集日志目录、每个机位组重复遍数 | `dvs_roadside_multicam` / `logs` / 1 |
| 6 | 参数确认（回车开始采集） | 确认 |

端口没有响应时，向导会自己在 `CARLA_HOME` 和常见安装位置里找 `CarlaUE4.exe`
并启动 CARLA；用 `--no-start` 可以禁止这一步，用 `--offscreen` 以无窗口模式启动。

要改 CARLA 安装位置、Python 环境或端口，直接编辑 `一键采集.bat` 顶部的
`CARLA_HOME`、`PY`、`PORT`（也可以设成同名环境变量）；`OFFSCREEN=1` 让批处理
以无窗口模式启动 CARLA。

```powershell
# 也可以在终端里直接跑向导
python .\tools\auto_collect.py --port 2000
python .\tools\auto_collect.py --self-test     # 不连 CARLA 的自检
```

## 环境和 CARLA

需要 CARLA Python API、Python 3.12、NumPy、h5py 和 PyYAML：

```powershell
conda activate dvs
Set-Location E:\Projects\DVS_data
python --version
python -m pip install -r requirements.txt
```

如果 CARLA Python API 尚未安装：

```powershell
python -m pip install "D:\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl"
```

默认使用 [Town03 示例机位组](../rigs/town03_intersection_a.yaml)。该示例只有方案附录明确给出世界坐标的 `cam_01` 和 `cam_02`，不代表完整八机位布设。新增机位前，请填写目标 Town 中实际测量或规划的坐标、朝向和机位参数。采集器会校验 CARLA 当前地图与机位组中的 Town 是否匹配（`Town10` 与 `Town10HD_Opt` 等别名视为同一张地图）。

## 运行

启动 CARLA 服务端后，从项目根目录执行：

```powershell
python .\collect_dvs.py --rig .\rigs\town03_intersection_a.yaml --load-town --output .\dvs_roadside_multicam --frames 200
```

`--load-town` 会调用 CARLA `load_world` 切换到 YAML 指定的地图，并重载当前 world；这会影响连接到该 CARLA 服务端的其他客户端。若已手动加载正确地图且不想重载，请省略该参数。

省略 `--frames` 时，默认按机位组配置采集 90 秒、40 Hz（3600 帧）。40 Hz 是自动标注方案的硬性要求：20 Hz 时 ΔT=50 ms 的事件窗口没法做 25/50/100 ms 的对比实验，40 Hz（ΔT=25 ms）下 25 ms 取单帧、50 ms 合并 2 帧、100 ms 合并 4 帧，三种窗口都有真值。可用 `--duration`、`--fps` 覆盖时长和帧率。完整参数：

| 参数 | 说明 |
|---|---|
| `--host` | CARLA 服务端地址，默认 `127.0.0.1` |
| `--port` | CARLA RPC 端口，默认 `2000` |
| `--tm-port` | Traffic Manager 端口，默认 `8000` |
| `--rig` | 机位组 YAML，默认 `rigs/town03_intersection_a.yaml` |
| `--output` | 数据集根目录，默认 `dvs_roadside_multicam` |
| `--load-town` | 切换到机位组声明的地图 |
| `--frames` | 指定采集帧数；不可与 `--duration` 同时使用 |
| `--duration` / `--fps` | 覆盖序列时长或帧率 |
| `--vehicles` / `--walkers` | 覆盖 YAML 中配置的交通流数量 |
| `--cameras` | 只启用机位组里的前 N 台相机（按 `camera_id` 排序） |
| `--weather` | 固定天气；不指定时从 YAML 的天气列表按种子选择 |
| `--seed` | 随机种子；默认使用全局序列编号 |
| `--min-speed` | 静止目标判定阈值，默认 `0.1 m/s` |
| `--instance-seg` / `--no-instance-seg` | 实例分割传感器（算可见度/遮挡），默认开；关掉后 `occluded`/`difficult` 恒为 `False` |
| `--instance-seg-scale` | 实例分割图相对 DVS 分辨率的降采样倍数，默认 `2`（半分辨率） |
| `--rgb-format` | RGB 帧格式 `png`（默认）或 `jpeg` |
| `--rgb-quality` | JPEG 质量，默认 `90` |
| `--min-detect-px` | 可探测的最小投影短边（像素），默认 `16`；更小的目标标 `ignore` |
| `--max-detect-distance` | 最大标注距离（米），默认 `120`；更远的目标标 `ignore` |
| `--flush-frames` | HDF5 攒批刷盘的帧数，默认 `10` |
| `--sensor-timeout` | 等待传感器帧的超时秒数，默认 `15` |

运行 `python .\collect_dvs.py --help` 查看全部参数。

### 落盘与卡顿

每台相机有一个**后台落盘线程**：PNG 编码、HDF5 追加和时间戳写入都不占用采集
主循环，多相机同时写盘的磁盘卡顿由每相机 30 帧的有界队列吸收，不会传导成传感器
回调超时（实测 3 相机 1.34 → 0.40 s/帧）。磁盘长期跟不上时，某一帧入队超过
60 秒仍未消化就显式报错；`--flush-frames`（默认 10）控制 HDF5 攒批刷盘的帧数。

RGB 默认存 PNG（无损，1280×720 约 2.6 GB/相机/序列）。规模化前按方案 §8 换成
JPEG：`--rgb-format jpeg --rgb-quality 90`，体积约为 PNG 的 1/10，文件名变成
`rgb/000000.jpg`。质检和质检视频都同时认这两种格式。

### DVS 是事件流，不是一帧一包

CARLA 的 DVS 传感器**不是每帧都发数据**：这一帧没有任何事件（画面完全静止）时
它就不发这一帧。旧代码像 RGB 那样按 frame 号死等，画面一安静下来就永远等不到，
整条序列以"等待 cam_XX_dvs 的 CARLA frame=N 超时"退出（实测 cam_02 停在
frame=1934，cam_01 的同一时刻还在正常出帧）。

现在 DVS 按**流**处理：先等一小会儿（0.1 s）拿这一帧的事件，拿不到就把队列里
现有的合并成这一帧的内容，再没有就写**空帧**（0 个事件，但 RGB、时间戳和标注
照常落盘）。事件自带绝对时间戳，空帧不影响事件本身的时间精度。RGB 和实例分割
仍然按 frame 号严格配对——多机位之间必须帧号对齐。

连续 60 帧没有事件会打印提示，用来区分"画面静止"和"传感器彻底停了"。

## 机位组 YAML

每组配置 1–8 台相机，包含 Town 名称、相机世界坐标、pitch/yaw/roll、分辨率、FOV、DVS 参数、交通流和天气列表，以及 `sequence_duration_s`、`fps`、`seed` 和可选的 `location` 块（路口编号、路口中心、地理坐标）。DVS 预设为 `road_day`、`road_lowlight` 和 `road_near`，单个相机也可覆盖预设值。采集器不会根据道路出生点随机放置相机。机位组可由 `tools/plan_rig.py` 依据地图中的信号灯杆自动生成。

`traffic` 块（默认值即方案 §6.1/§6.2 的要求）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `vehicles` / `walkers` | 80 / 40 | 车辆总数与行人数 |
| `motorcycle_ratio` | `0.3` | 车辆里摩托车的比例，其余为汽车（7:3） |
| `pedestrian_spawn_radius_m` | `[30, 60]` | 行人出生点距路口中心的半径区间 |
| `pedestrian_walk_distance_m` | `[15, 40]` | 行人沿人行道行走的目标点距离区间 |
| `crosswalk_ratio` | `0.4` | 行人里走人行横道（往返横穿）的比例 |
| `ignore_lights_pct` / `speed_difference_pct` / `following_distance_m` | — | 交通管理器参数 |

行人在路口附近的人行道上生成并沿人行道行走（或在人行横道两端往返），生成后空跑 60 tick，位移 < 0.5 m 的会被销毁重生——不再出现旧版「39 个行人叠在一点、速度 0」的情况。阈值按行人的实际步速定：60 tick（约 1.5 s）内正常步速能走 1.2~2.7 m，原地不动的是 0 m；早先用「20 tick / 1 m」，行人走不到 1 m 就被判成卡住，实测 36 个行人只剩 1 个。自行车蓝图（`vehicle.bh.crossbike` 等）不在生成列表里。

行人出生点必须落在机位附近：拿不到路口坐标时用第一台相机的位置当中心。**这里不会退回 `world.get_random_location_from_navigation()` 的全城随机撒点**——那个原生导航调用在 Windows 客户端上会栈溢出（exit `0xC00000FD`）把整个采集带崩，而且撒出来的行人在相机视野外根本拍不到。

行人**不用 `controller.ai.walker`**：它要走导航网格（`Nav/TownXX.bin`），Town03 的导航数据不完整，服务端的路径搜索会一路递归到爆栈，同样是 exit `0xC00000FD`。改用 `carla.WalkerControl` 直接推人走，方向自己算，每个 tick 前调一次 `update_walkers()`，完全不碰导航网格。

注意 `update_walkers()` **方向没变就不重复下发控制**：`apply_control` 会让服务端重做一次导航寻路，40 个行人 × 40 tick/s 就是每秒近千次寻路，服务端帧时间被拖到十几秒，传感器直接等不到数据（报"等待 cam_XX_dvs 的 frame=N 超时"）。只在掉头或偏了 10° 以上时补一次即可。

站位来源按可用性依次降级：人行道 waypoint → 人行横道 → 地图出生点。Town03 没有车道级 OpenDRIVE 数据，人行道 waypoint 取不到，只剩人行横道可用：把 `Map.get_crosswalks()` 的顶点按文档给的"闭合多边形、首点重复"还原成一条条横道，取每条横道的**长边**（横跨马路的那一段，8~14 m）作为往复通道——取对角线会斜穿马路，取短边则是沿路走。还原不出来时按 10 m 紧半径重新聚类，旧的 12 m 半径会把整个路口连成一坨，取出的"对侧点"能差出 78 m。

横道**两端都要离相机至少 10 m**：行人会沿整条横道走，站到镜头底下会瞬间占满
整幅画面，事件缓冲区被灌爆后那个机位的 DVS 就此停摆。所以横道不再受
`pedestrian_spawn_radius_m` 下界（30 m，按人行道 waypoint 定的，横道全挤在路口上，
用下界会把它们全滤光）的约束，只受上界和这条离镜头的距离约束。同一批行人再沿
过街方向抖动（±35% 通道长）分开站位，避免挤在一点生成失败。

## 自动标注：车辆 / 行人 / 摩托车三类

### 类别映射

映射表在 `dvs_collector/classes.py`，按蓝图 ID **显式列出**，不做子串猜测（`vehicle.kawasaki.ninja` 既不含 "motorcycle" 也不含 "bicycle"，靠子串必然把摩托车标成汽车）：

| 类别 | category_id | 来源 |
|---|---|---|
| `car` | 0 | 其余 `vehicle.*`（兜底规则，见下） |
| `motorcycle` | 1 | `vehicle.harley-davidson.low_rider` / `vehicle.kawasaki.ninja` / `vehicle.yamaha.yzf` / `vehicle.vespa.zx125` |
| `pedestrian` | 2 | `walker.pedestrian.*` |

自行车和 `vehicle.carlamotors.carlacola` 映射为 `ignore`（不生成，出现也不标）。子类（`sedan`/`suv`/`van`/`bus`/`truck`/`emergency`，认不出来的记 `other`）写在标注的 `subtype` 字段里。

**表外的蓝图不会被静默归类**：兜底规则把它算作 `car`，但同时

- 采集时打印提示，列出本序列出现过的兜底蓝图；
- 写进数据集根目录的 `unknown_blueprints.csv`；
- 写进 `meta/episode_manifest.json` 的 `blueprints_seen` 与 `blueprints_by_fallback`。

重采之前先跑一次下面的核对命令，把清单导出人工过一遍：

```powershell
python .\tools\export_blueprints.py --host 127.0.0.1 --port 2000 --out-dir .
```

它生成 `blueprints.csv`（全部蓝图及其类别/子类）和 `unknown_blueprints.csv`（靠兜底规则归类的那些）。

### 可见度与遮挡判定

每台 DVS 相机会额外生成一台同位、同 FOV 的**实例分割相机**（默认半分辨率，`--instance-seg-scale` 可调），用它统计每个目标在 3D 框里的实际可见像素：

```text
occupancy  = 可见像素数 / 3D 框投影面积
visibility = min(1, occupancy / FILL_RATIO[类别])     # FILL_RATIO: car 0.55 / motorcycle 0.45 / pedestrian 0.30
```

按方案 §5.3 的顺序定级，**先命中先定级**：

| 顺序 | 条件 | 结果 |
|---|---|---|
| 1 | 投影后无有效面积 / 完全在图像外 | 丢弃，不写 |
| 2 | 映射为 `ignore`（自行车、未知资产） | 写进 `ignored_annotations[]`，记 reason |
| 3 | 超出有效探测范围（短边 < 16 px 或距离 > 120 m） | `ignore=true`，标 `beyond_detection_range` |
| 4 | `visibility < 0.05` | `ignore=true`，标 `ignored_low_visibility` |
| 5 | `visibility < 0.20` | `difficult=true`、`occluded=true` |
| 6 | 框被图像边界截断 | `truncated=true`（**只有**被边界裁切才算截断） |
| 7 | 框面积 < 400 px² | `difficult=true`，标 `tiny` |
| 8 | `stationary=true` 且框内事件 < 阈值 | 标 `low_event` / `static` |
| 9 | 其余 | 正常样本 |

`ignore=true` 与 `low_event` 的样本**保留在文件里**，由加载器按配置过滤——采集时删掉会丢掉真实的背景分布。

### 有效探测范围

太远的目标在画面上只有几个像素，相机采不到足以定位的事件，标注出来是噪声。
判据参照 DVSense DVSync 的 DVS 面板参数（1 MP、像元 1.45 μm、C-MOUNT FOV
70°~130°，角分辨率约 0.055°~0.10°/px）定出：

| 判据 | 默认值 | 说明 |
|---|---|---|
| 投影短边像素数 | ≥ 16 px | 主判据。实测本项目 events.h5：运动目标短边 <10 px 时仅 10%~39% 的帧有事件，10~18 px 为 19%~40%，18~24 px 升到 67%~74%，≥24 px 稳定在 75%~100% |
| 推算距离 | ≤ 120 m | 150 m 外的车只有几个像素高，DVS 不会产生足以定位的事件簇 |

用**投影短边**而不是纯距离做主判据：同样距离下正面朝相机的车短边大、
侧向的车短边小，只按距离过滤会误伤近距离的大目标。距离由框高反推
（`d ≈ focal_px × 目标实际高度 / 框高像素`），与目标朝向无关。

阈值用 `--min-detect-px` / `--max-detect-distance` 调整；离线重投影
（`tools/annotate_offline.py project`）要用同样的参数，保证在线/离线一致。

标注字段（`annotations.json` 里每个目标）：

| 字段 | 含义 |
|---|---|
| `category` / `category_id` / `subtype` | 三类别、类别编号、子类 |
| `blueprint_id` | CARLA 蓝图 ID（可复现、可审计） |
| `bbox` | `[x, y, w, h]`，像素，左上角原点 |
| `truncated` | 框是否被图像边界裁切 |
| `occluded` / `difficult` | 遮挡 / 困难样本标记（由可见度决定，不再是写死的 `false`） |
| `visibility` / `visible_px` / `visible_bbox` / `occupancy` | 可见度指标与可见像素的紧框 |
| `ignore` / `ignore_reason` | 是否忽略及原因（`low_visibility` / `beyond_detection_range` / 非目标类别） |
| `actor_id` / `speed_mps` / `stationary` | CARLA actor 编号（跨机位关联用）、瞬时速度、静止标记 |
| `distance_m` | 由投影框高度反推的目标距离（米），与朝向无关 |
| `quality_flags` | `tiny` / `low_event` / `static` / `ignored_low_visibility` / `severe_occlusion` / `beyond_detection_range` 等标记 |
| `event_count` / `event_density` / `active_pixel_ratio` / `on_off_ratio` | 框内事件统计（`tools/annotate_offline.py stats` 写回） |

### 在线算 + 离线可重算

采集时除了落盘事件和 RGB，还会写一份 `snapshot.json`：**每帧每个 actor 的 4×4 世界矩阵、蓝图 ID、速度，以及每个机位的可见像素统计**。分割图本身不保存。

投影代码改了以后不需要重采，用快照重投影即可：

```powershell
python .\tools\annotate_offline.py project --sequence .\dvs_roadside_multicam\sequences\sequence_000001
```

离线走的是**和在线完全相同**的 `project_records()`，重算结果与采集时逐字段一致（有自检覆盖）。

> ⚠️ **2026-11-09 之前采集的旧序列**（`snapshot.json` 里相机 `world_matrix` 为单位阵的
> 那些）无法离线重投影：它们的相机位姿没被正确记录，重投影结果和在线标注对不上。
> 修复后新采的数据在线/离线逐字段一致。旧序列只能用在线那份 `annotations.json`。

### 事件统计与质量标记

```powershell
python .\tools\annotate_offline.py stats --dataset .\dvs_roadside_multicam
```

逐帧统计每个框内的事件数、密度、活跃像素比例和 ON/OFF 极性比，把 `low_event` / `static` 写回 `quality_flags`，并在数据集根目录写 `annotation_report.json`（含整体零事件比例）。

### 质检与划分

```powershell
python .\tools\qc_video.py --dataset .\dvs_roadside_multicam --sequence sequence_000001 --camera cam_01 --out-dir .\qc
python .\tools\split_dataset.py --dataset .\dvs_roadside_multicam
```

- `qc_video.py` 输出三栏帧（RGB ｜ 事件帧 ｜ 带框叠加），车=橙、摩托=品红、行人=绿，忽略/困难样本画灰框，用来肉眼确认骑手没有被重复标成行人、树后没有框、三类别计数正确。默认输出 PNG 帧序列；装了 `imageio-ffmpeg` 会另外编码 mp4。
- `split_dataset.py` 按**路口**而不是按帧随机划分（同一路口不同序列外观高度相似，按帧分会让同一批目标同时进训练集和测试集），比例 70/15/15，测试集保证至少含一条夜间序列，结果写 `splits/{train,val,test}.txt` 和 `split_report.json`（各类实例数、轨迹数、零事件比例、框尺寸分布）。

## 自动布点：tools/plan_rig.py

该工具扫描 CARLA 地图，把每个路口的每一根信号灯杆变成一台固定 DVS+RGB 相机，输出与上面完全同构的机位组 YAML，采集器代码无需改动：

```powershell
python .\tools\plan_rig.py --town Town03 --dry-run          # 只看清单，不写文件
python .\tools\plan_rig.py --town Town03 --out-dir .\rigs\Town03
python .\tools\plan_rig.py --self-test                      # 不连接 CARLA 的自检
```

- **杆位**：`map.get_all_landmarks_of_type('1000001')` → `world.get_traffic_light()`；actor 的水平坐标即杆位，安装高度取杆上灯头包围盒的顶面（CARLA 中约 4 m），可用 `--mount-height` 覆盖。
- **分组**：`world.get_traffic_lights_in_junction()` 把信号灯归到所属路口，一个路口一个机位组（相机数 1–8，超过 8 台按 `--max-cameras` 拆成多组）。
- **朝向**：默认瞄准路口中心（`--aim junction`）；`--aim approach` 改为瞄准该信号灯控制的进口道。yaw/pitch 由杆位与目标点算出（CARLA 约定：yaw 0 = +X，负 pitch = 俯视）。
- **其它参数**：`--fov`、`--preset`、`--duration`、`--fps` 控制生成机位组的成像与采集参数，`--warmup-ticks` 控制空跑帧数。
- **输出**：`rigs/<town>/` 下每个机位组一个 YAML，外加 `poles.json` / `poles.md`（杆位与机位对照表，含灯头 actor id、目标点、距车道距离）和 `commands.txt`（逐机位组的采集命令）。
- **校验**：生成的 YAML 会用 `dvs_collector.config.load_rig()` 复核一遍，保证采集器可直接读取。

> **必须在 CARLA 异步模式下运行**：同步模式下 CARLA 不会实例化地图的信号灯 actor，
> 工具会直接报错退出。采集进行中（同步模式）也不要运行它。

```powershell
python .\collect_dvs.py --rig .\rigs\Town03\Town03_j012.yaml --load-town --output .\dvs_roadside_multicam
```

## 批量采集：tools/collect_batch.py

按机位组目录批量跑序列，一个机位组一条序列，`--repeat` 控制重复遍数：

```powershell
python .\tools\collect_batch.py --rig-dir .\rigs\Town03 --repeat 2 --output .\dvs_roadside_multicam
python .\tools\collect_batch.py --rig-dir .\rigs\Town03 --dry-run      # 只列清单
python .\tools\collect_batch.py --self-test                          # 不连接 CARLA 的自检
```

- **必须串行**：同步模式下同一个 world 只能由一个客户端驱动 tick，多采集器并行会互抢 tick。
- 每条序列消耗一个全局序列号 → 种子按序列号派生 → 同一机位不同遍数的交通流/天气互不相同且可复现。
- 单条失败自动重试（`--retries`，默认 1 次），失败序列不写入 `index_all.txt`。
- 结束后在输出目录写 `batch_report.json`（每条序列的机位组、遍数、重试次数、耗时、失败日志尾部）。
- `--frames/--vehicles/--walkers/--weather` 可覆盖机位组中的默认值；`--load-town` 允许按机位组的 town 切换地图。
- `--cameras N` 让每个机位组只启用前 N 台相机；`--rigs a.yaml,b.yaml` 只采集指定机位组。
- `--sensor-timeout` 是传感器回调超时秒数（不传则用采集器的默认 15）。仿真远慢于实时时若仍报“等待 … 超时”，把它调大。
- `--log-dir` 指定采集日志目录（默认 `logs/`），见下。

## 采集日志

长时间采集时不会再静默：

```text
[1/5] Town10HD_Opt_j023.yaml 开始采集 …
    | 采集进度 10/1800 帧（已用 9 s，0.88 s/帧），各机位累计事件数 {'cam_01': 217956, ...}
    | 采集进度 20/1800 帧（已用 18 s，0.90 s/帧），各机位累计事件数 {...}
  成功（1560.2 s）
  [OK] sequence_000001 | 路口 j23 | 3 相机 | 1800 帧 | 12,345,678 事件 | ClearNoon | 采集于 2026-10-26T11:37:52+08:00 | 用时 1560.2 s
=== 进度：已完成 10 条，累计 98,765,432 事件，总用时 260.1 分钟，平均每条 1560 s ===
```

- 采集器每 **10 帧**打印一次进度（帧数、已用时间、每帧耗时、各机位累计事件）；
- 批量器每采集完 **一条序列**打印一行结果摘要（位置、时间、规模、事件量）；
- 每 **10 条序列**打印一次累计汇总。

### 每次采集的日志目录

控制台输出之外，每次批量采集还会在 `--log-dir`（默认项目根目录 `logs/`）下建一个
`<日期-时间>_<地图>/` 子目录，记录这次采集的时间、机位、数量和完整过程：

```text
logs/
├── index.csv                              采集历史总表，每次采集追加一行
└── 20261026-160638_Town10HD_Opt/
    ├── run.json                           运行摘要：起止时间、端口、机位组、参数、成功失败、事件总量
    ├── run.log                            完整控制台输出（含逐帧进度）
    └── sequences.csv                      每条序列一行：机位、路口、帧数、事件数、天气、耗时
```

## 输出结构

所有数据平铺在同一个数据集目录里，**采集器不做 train/val/test 划分**（用 `tools/split_dataset.py` 按路口划分）。

```text
<output>/
├── sequences/sequence_XXXXXX/
│   ├── cameras/cam_XX/
│   │   ├── events.h5
│   │   ├── rgb/000000.png
│   │   ├── timestamps.txt
│   │   └── annotations.json    三类别标注 + 可见度 + 质量标记
│   ├── snapshot.json           采集时的 actor 快照（位姿/蓝图/可见像素），投影可离线重算
│   ├── calibration/
│   ├── event_representations/
│   └── meta/
│       ├── episode_manifest.json
│       ├── dvs_params.json
│       └── rig_geometry.json
├── calibration/rigs/          # 本次用到的机位组 YAML + poles.json
├── tools/                     # 质检与时间戳修复工具副本（可独立运行）
├── index_all.txt              # 全量索引（所有完成序列一行）
├── locations.csv              # 位置时间索引：每条序列一行
├── unknown_blueprints.csv     # 靠兜底规则归类的蓝图（需人工确认）
├── batch_report.json
├── annotation_report.json     # 事件统计汇总（stats 步骤写）
├── splits/{train,val,test}.txt + split_report.json
├── README.md / CHANGELOG.md / LICENSE.txt
```

采集过程本身的日志在项目根目录的 `logs/`（每次采集一个子目录），见“采集日志”一节。

## 位置与时间标注

采集时自动为每条序列写入位置和时间，不需要人工登记：

- `meta/episode_manifest.json` 的 `location` 块：地图、路口编号、路口中心与地理坐标，
  以及每台相机的世界坐标、朝向和地理坐标（`map.transform_to_geolocation()` 换算，
  注意官方地图的地理参考是地图内一致的相对定位，不等于真实城市经纬度）；
- `meta/episode_manifest.json` 的 `capture_time` 块：真实采集时刻（UTC + 本地时区）、
  墙钟耗时、仿真时间起止；
- `locations.csv`：数据集级的位置时间索引，一行一条序列，字段扁平，
  可直接用 pandas 按路口或时间段筛选；下游需要划分时自行按该表过滤。

```text
sequence_id,capture_started_at_utc,capture_started_at_local,timezone,duration_s,town,map_name,junction_id,...
sequence_000001,2026-10-26T06:45:12+00:00,2026-10-26T14:45:12+08:00,中国标准时间,90.0,Town10HD_Opt,Town10HD_Opt,23,...
```

`locations.csv` 的全部列：

```text
sequence_id, capture_started_at_utc, capture_started_at_local, timezone, duration_s,
town, map_name, junction_id, junction_center_x_m, junction_center_y_m,
geo_latitude, geo_longitude, geo_altitude_m, camera_count, camera_poses,
frame_count, fps, weather, seed, total_events, vehicles, walkers
```

机位组 YAML 里的 `location` 块由 `tools/plan_rig.py` 写入，采集器据此记录路口级位置。

每帧标注包含 `actor_id`、`speed_mps` 和 `stationary`，可用于跨机位目标关联。HDF5 保留规范事件数据字段，并记录相机、Town、天气与 DVS 参数。标定文件中的 DVS/RGB 外参与零时间偏移是基于 CARLA 同位、同 tick 的仿真假设，不是实测硬件标定。

`meta/episode_manifest.json` 里还有几个用于验收的块：`class_map_version` / `classes`（类别映射表版本）、`instance_segmentation`（实例分割是否开启及降采样倍数）、`annotation_validation`（每机位各类别的轨迹数、运动轨迹数、静止轨迹数、平均位移，方案 §6.1.5 的验收项）、`blueprints_seen` / `blueprints_by_fallback`（蓝图审计）、`spawned_traffic`（实际生成的汽车/摩托车/行人数）。

## 事件时间戳的单位

`events.h5` 里：

- `/events/t` 是**纳秒**（`/meta/event_timestamp_unit` = `ns`），序列内从 0 开始；
- `/frames/timestamp_us` 是**微秒**，与 `timestamps.txt`、`annotations.json` 同一基准；
- 换算：`events/t ÷ 1000` 就是与帧时间同一时间轴的微秒值。

原因是 CARLA 0.9.16 的 DVS 传感器给出的原始时间戳本身就是纳秒，而 20 Hz 下每帧
约 4 万个事件，微秒分辨率会让同一微秒挤进多个事件、无法保持严格递增。
写入器按帧绝对时间自动识别 CARLA 给的单位（`/meta/event_timestamp_source_unit`）。

时间原点取自**相机生成之前**的世界快照，采集前另有一次预热 tick（避免新建的 DVS
传感器漏掉首个回调），因此 `timestamps.txt` 的第一个值是 `100000` µs 而不是 0。

2026-10-26 之前采集的数据是按微秒基准写的（时间轴整体错位），用下面的命令修复
（只平移 `events/t`，不动其它数据）：

```powershell
python .\tools\fix_event_timestamps.py --dataset .\dvs_roadside_multicam --dry-run
python .\tools\fix_event_timestamps.py --dataset .\dvs_roadside_multicam
```

质检工具会检查“每帧最后一个事件是否落在该帧曝光窗口内”，单位不对会直接报错。

采集完成后运行结构和同步检查：

```powershell
python .\tools\check_dataset.py --dataset .\dvs_roadside_multicam
```

校验会检查相机数、HDF5 数据类型与事件时间戳、帧事件偏移、RGB/标注帧数、时间戳及跨机位同步情况，并检查 `locations.csv` 与 `index_all.txt` 是否一一对应。标注本身也会被检查（方案 §7.3 的六项自动检查）：

| 检查 | 级别 |
|---|---|
| 事件时间戳严格递增、坐标不越界、极性合法 | 错误 |
| 帧事件区间与 `event_offsets` 一致、事件落在曝光窗口内 | 错误 |
| bbox 是有限值、宽高 > 0、不越界 | 错误 |
| 同一 actor 同一帧不重复标注 | 错误 |
| `ignore` 的样本必须有原因（`ignore_reason` 或 `quality_flags`） | 错误 |
| 相邻帧同一 actor 的框面积/尺寸突变（面积 > 6× 或边长 > 2.5×） | 报警（WARNING） |

检查工具和时间戳修复工具都会复制到每个数据集的 `tools` 目录中。

## 自检（不连接 CARLA）

改动采集器后可以先跑这些脚本，它们都不需要 CARLA 服务端：

```powershell
python .\tools\plan_rig.py --self-test           # 朝向数学、机位切分、杆位合并、机位组格式
python .\tools\collect_batch.py --self-test      # 机位组发现、town 解析、日志落盘
python .\tools\auto_collect.py --self-test        # 向导输入解析与机位组选择
python .\tools\test_disk_writer.py               # 后台落盘线程：帧序、时间戳、写盘失败回传
python .\tools\test_event_timestamps.py          # 事件时间戳单位与历史数据修复
python .\tools\test_location_metadata.py         # 位置时间标注与 locations.csv
python .\tools\test_annotation.py                # 三类别映射、投影、可见度、离线重投影一致性
python .\tools\test_traffic.py                   # 行人生成/清理配对（控制器配对回归）
python .\tools\test_sensors.py                   # DVS 事件流读取（传感器不发数据时不能卡死）
```

## 运行注意事项

- 同一个 CARLA world 在采集期间必须由采集器独占同步 tick。
- `--load-town` 前请确认不会影响其他连接该 CARLA 服务端的客户端。
- `tools/plan_rig.py` 只能在 CARLA 异步模式下运行，不能与采集同时进行。
- 失败采集不会写入 `index_all.txt`；失败目录保留用于诊断并避免复用序列编号。
- **旧数据不能回填**：`blueprint_id`、摩托车类别、可见度都依赖采集时才拿到的信息，
  2026-11 之前采集的序列只能当二分类基线用，三类实验需要重新采集。
