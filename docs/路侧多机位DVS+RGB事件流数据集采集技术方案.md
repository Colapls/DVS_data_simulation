# 基于 CARLA 的路侧多机位 DVS+RGB 事件流数据集采集技术方案（v3.0）

> **标注与帧率部分已被 [DVS 自动标注方案（精简可执行版）](./DVS自动标注方案_精简可执行版.md) 取代**：
> 类别从 `vehicle`/`walker` 二分类改为 `car`/`motorcycle`/`pedestrian` 三类，采集帧率从 20 Hz 改为
> 40 Hz，可见度由实例分割相机计算。本文其余部分（机位布设、批量采集、日志、数据结构）仍然有效。
>
> v3.0 修订说明：机位布设由"人工填写坐标"改为**按信号灯杆自动布点**（`tools/plan_rig.py`），
> 采集改为**按机位组批量执行**（`tools/collect_batch.py`），并同步修正了 CARLA 0.9.16 的 API 差异。
> 整合来源：《基于 CARLA 的多机位 DVS 事件流数据集采集技术方案》+《DVS 事件流目标检测数据集 · 数据结构规范 v1.1》
> 版本：v3.0 ｜ 更新：2026-10-26 ｜ 目标：事件相机目标检测 / DVS+RGB 融合检测 / 多机位目标关联

---

## 1. 项目概述

### 1.1 目标

构建一个**路侧固定视角**的多机位 DVS+RGB 融合事件流数据集：**相机安装到路口的信号灯杆上**，车辆与行人持续运动穿过视野，相机本身不动。数据同时包含：

- **DVS 事件流**（CARLA 原生 `sensor.camera.dvs`）：稀疏异步事件 (t, x, y, pol)；
- **同步 RGB 帧**：与每台 DVS 同机位、同朝向、同步曝光的融合相机帧；
- **跨机位标注**：每帧目标框带 CARLA `actor_id`，支持跨视角关联与 3D 重建。

支持任务：事件相机目标检测、DVS+RGB 融合检测、多机位目标关联/重识别、路侧交通感知。

### 1.2 版本演进与关键差异

| 决策项 | v1.0（自车环视） | v2.0（人工路侧机位） | **v3.0（本方案，信号灯杆自动布点）** |
|---|---|---|---|
| 相机载体 | 自车 + 少量路侧 | 路口固定机位，坐标手填 | **路口每一根信号灯杆一台相机，坐标自动生成** |
| 机位来源 | 出生点随机 | 人工测量并写入 YAML | **`map.get_all_landmarks_of_type('1000001')` → `world.get_traffic_light()`** |
| 路口分组 | 无 | 人工分组 | **`world.get_traffic_lights_in_junction()` 自动分组，一路口一机位组** |
| 朝向 | 朝前 | 人工指定 yaw/pitch | **按目标点自动解算 yaw/pitch（默认瞄路口中心）** |
| 采集执行 | 单次 | 单次 | **按机位组批量串行采集（`tools/collect_batch.py`）** |
| 事件产生前提 | 自车必须持续运动 | 相机静止、目标运动 | 不变；静止目标标记 `stationary` |
| 数据划分 | train/val/test | 不划分 | 不划分，全部序列入同一目录 |
| 目标类别 | 车/卡车/自行车/摩托/行人 | vehicle / pedestrian | 不变（对齐数据规范 v1.1） |

### 1.3 事件产生的前提约束（仍然成立）

DVS 只在亮度变化时产生事件：
- **静止相机 + 运动目标** → 目标边缘与背景交界处产生事件（本项目常态）；
- **完全静止的目标**（等红灯的车辆、停留的行人）→ 不产生事件，标注流水线按速度过滤（speed < 0.1 m/s 标 `stationary=true`）；
- 夜晚/逆光下事件密度下降 → 采用低阈值预设并保证路灯/车灯照明。

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│  CARLA Server 0.9.16 (GPU ≥ 8GB)                             │
│                                                              │
│   ┌────────── 路口信号灯杆上的固定融合相机阵列 ─────────────┐  │
│   │  杆A: DVS + RGB ┐                                      │  │
│   │  杆B: DVS + RGB ├ 每个路口 2–4 台，杆位/朝向自动解算   │  │
│   │  杆C: DVS + RGB ┘ 视野指向路口中心，FOV 有重叠        │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                              │
│   交通流：车辆 80 辆 autopilot + 行人 40 个（经斑马线穿行）    │
│   无自车（ego 不参与采集）                                    │
└───────────────────────────┬──────────────────────────────────┘
                            │ synchronous tick (20 Hz)
┌───────────────────────────▼──────────────────────────────────┐
│  ① tools/plan_rig.py   自动布点                                │
│     枚举路口 + 信号灯杆 → 机位组 YAML + 杆位报告              │
│  ② collect_dvs.py      单序列采集（事件/RGB/标注/标定）          │
│  ③ tools/collect_batch.py  批量调度（串行、失败重试、报告）     │
│  ④ tools/check_dataset.py  结构与同步质检                      │
└───────────────────────────┬──────────────────────────────────┘
                            │
                   ┌────────▼────────┐
                   │  Storage (NAS)  │  sequence_XXXXXX/（单一目录，无 split）
                   └─────────────────┘
```

**角色划分**：CARLA Server 承载静态场景与全部传感器；采集器进程只做取数、攒批与落盘，回调内仅 `queue.put()`，避免阻塞传感器回调线程。

---

## 3. 机位布设设计（核心章节）

### 3.1 布设原则

| 原则 | 说明 |
|---|---|
| 装在信号灯杆上 | 杆位取 CARLA 地图的真实灯杆位置，安装高度取杆上灯头顶面（CARLA 中约 4.0–4.1 m） |
| 俯视路口 | 相机瞄准路口中心，俯角典型 −10°~−17°（4 m 杆高、15–20 m 水平距离），FOV 60° |
| 一路口一机位组 | 同一路口的各杆组成一个机位组（1–8 台），一个机位组 = 一个序列 = 一个路口的多视角 |
| FOV 重叠 | 同一路口的相机都看向路口中心，天然重叠 ≥ 40%，支持跨视角关联/多视角 3D 标注 |
| 行人与车辆兼顾 | 路口中心视角同时覆盖进口道车流与斑马线行人；需要更强行人覆盖时用 `--aim approach` 改瞄进口道 |

### 3.2 CARLA 中如何拿到"灯杆"（实现要点）

链路：

```python
landmarks = world.get_map().get_all_landmarks_of_type('1000001')   # OpenDRIVE 信号灯地标
light = world.get_traffic_light(landmark)                          # 信号灯 actor
transform = light.get_transform()                                  # actor 的水平坐标 = 杆位
boxes = light.get_light_boxes()                                    # 灯杆/横臂上的灯头包围盒
```

**四个必须知道的坑**（v3.0 实测，CARLA 0.9.16）：

1. **actor 的 `transform.z` 不是物理高度**（值约 0.3 m，只是虚拟锚点）。杆上灯头高度要从 `get_light_boxes()` 里取：杆心附近（水平 ≤ 1.5 m）的包围盒最高点即杆上信号灯高度，实测约 **4.05 m**；更高的 5.2 m 那一档是横臂上的灯头。
2. **一根物理杆 = 一个 actor，多个灯头**。地图里 58 个 OpenDRIVE 地标只对应 **15 个 actor**（一根杆上每个进口方向各一个灯头，共用 actor）。按 actor 去重后才是杆位数量。
3. **同步模式下 CARLA 不提供信号灯 actor**：`world.get_traffic_light()` 恒返回 `None`。自动布点必须在**异步模式**下空跑若干帧（工具的 `--warmup-ticks`，默认 10）让 TrafficLightManager 把灯实例化出来。
4. **`get_stop_waypoints()` 的 waypoint 在路口之外**（`junction_id == -1`），不能用来判路口。路口归属要用 `get_affected_lane_waypoints()` 的 `junction_id`，或直接 `world.get_traffic_lights_in_junction(jid)`。

### 3.3 路口与机位组的划分

- **路口发现用两条路径取并集**：
  1. `map.generate_waypoints(2.0)` 中 `is_junction == True` 的 waypoint → `waypoint.get_junction()`；
  2. 每个信号灯 actor 的 `get_affected_lane_waypoints()` 中的 `junction_id`（补齐路径 1 漏掉的路口）。
- **路口中心** = 该路口所有车道两端点位置的均值（`junction.get_waypoints(LaneType.Driving)`）。
- **杆位归属**：同一根杆可能同时服务两个路口，只归属到路口中心更近的那一个。
- **一个路口 = 一个机位组**（1–8 台相机）；超过 8 台按 `--max-cameras` 拆成多组（`*_p2.yaml`）。
- 工具会输出"地图信号灯 actor 数 / 已归属路口数"，未归属数应恒为 0，否则说明路口发现漏了。

### 3.4 朝向解算

CARLA 角度约定：**yaw 0 指向 +X、90 指向 +Y，正角度向上旋转；pitch 为负表示俯视**。

```python
dx = target.x - mount.x
dy = target.y - mount.y
yaw   = degrees(atan2(dy, dx)) % 360            # 指向目标点
pitch = -degrees(atan2(mount.z - target.z, hypot(dx, dy)))   # 俯视目标点
```

- 目标点默认取**路口中心**（`--aim junction`）；`--aim approach` 改为取该信号灯所控制车道的中心。
- 安装高度默认取杆上灯头实际高度，`--mount-height` 可统一覆盖（如统一到 6 m 获得更陡的俯角）。

### 3.5 DVS 参数预设

```python
DVS_PRESETS = {
    'road_day':      dict(res=(1280, 720), fov=60, pos_th=0.25, neg_th=0.25, refractory_ms=0.5),
    'road_lowlight': dict(res=(1280, 720), fov=60, pos_th=0.10, neg_th=0.10, refractory_ms=0.3),
    'road_near':     dict(res=(1280, 720), fov=90, pos_th=0.20, neg_th=0.20, refractory_ms=0.5),
}
```

- 阈值：白天 0.25（CARLA 默认）；黄昏/夜晚 0.10 模拟高灵敏度档；
- 不应期：0.3–0.5 ms，贴近真实芯片（DAVIS346 / EVK4 级别）；
- 每个序列记录实际使用的预设到 `meta/dvs_params.json`。

---

## 4. 场景与交通生成

### 4.1 地图与机位库

- 使用带信号灯的官方 Town：**Town03、Town04、Town05、Town10HD** 为主（Town01/02 补充简单路口）。
- 每个 Town 单独跑一次自动布点，得到该 Town 的一组机位组 YAML（`rigs/<town>/*.yaml`）。
- 注意地图后缀：`_Opt` 地图为优化版，信号灯 actor 可能明显少于完整版（例如 Town10HD_Opt 仅 15 根杆 / 5 个信号化路口），选图时应比较各图的杆位数。

### 4.2 交通流生成

```python
# 车辆：autopilot，强制途经路口
traffic_manager.set_synchronous_mode(True)
for v in vehicles:
    tm.vehicle_percentage_speed_difference(v, random.uniform(-10, 20))  # 速度扰动
    tm.ignore_lights_percentage(v, random.uniform(0, 30))               # 部分闯红灯，增加事件丰富度
    tm.distance_to_leading_vehicle(v, random.uniform(2.0, 6.0))

# 行人：经路口人行道穿行（不用 AI 控制器）
# 不使用 controller.ai.walker / go_to_location：那条路径要走导航网格
# （Nav/TownXX.bin），Town03 的导航数据不完整，服务端路径搜索会一路递归到
# 爆栈（exit 0xC00000FD），把整个采集带崩。改为直接用 WalkerControl
# 推人走，方向自己算，彻底不碰导航网格。
for plan in walkers:
    target = plan["goal"] if plan["to_goal"] else plan["start"]
    dx, dy = target.x - plan["actor"].get_location().x, target.y - plan["actor"].get_location().y
    d = math.hypot(dx, dy)
    plan["actor"].apply_control(
        carla.WalkerControl(direction=carla.Vector3D(dx / d, dy / d, 0.0),
                            speed=plan["speed"])
    )
```

- 车辆 80 辆、行人 40 个为默认值（可用 `--vehicles/--walkers` 覆盖）。密度按目标事件量（每帧每相机 1k–200k 事件）动态调参；
- 行人站位来源依次降级：人行道 waypoint → 人行横道 → 地图出生点。**Town03 没有车道级 OpenDRIVE 数据，取不到人行道 waypoint，只剩人行横道可用**：把 `Map.get_crosswalks()` 的顶点按文档给的"闭合多边形、首点重复"还原成一条条横道，取每条横道的**长边**（横跨马路的 8~14 m）作为往复通道；还原不出来时按 10 m 紧半径重新聚类。行人沿直线往返，到头掉头，不需要导航网格。
- **红绿灯保持开启**：车辆启停产生丰富事件边缘（刹车灯、启动），行人等待/过街形成典型路侧场景；
- 每个序列 60–120 秒（仿真时长），序列间重洗交通流种子。

### 4.3 天气与光照随机化

| 类别 | 天气预设 | 说明 |
|---|---|---|
| 白天 | ClearNoon / CloudyNoon | 基准场景 |
| 黄昏 | SoftRainSunset / WetSunset | 逆光 + 湿地反光，事件丰富 |
| 夜晚 | ClearNight / HardRainNight | 低阈值预设 + 依赖路灯/车灯 |

天气默认由序列种子在 `weather_cycle` 中选择（同一 seed 可复现）；`--weather` 可固定。

---

## 5. 采集流程

### 5.1 单序列主循环（`dvs_collector`）

```python
FPS = 20
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1.0 / FPS
world.apply_settings(settings)

for frame_index in range(frame_count):
    carla_frame = world.tick()                    # 所有传感器在同一 tick 出帧
    timestamp_us = round(snapshot.timestamp.elapsed_seconds * 1e6) - time_origin_us
    labels_future = label_worker.submit(project_bboxes, world, rig, cameras)   # bbox 投影并行
    for camera in cameras:
        rgb = frame_reader.get(f'{camera}_rgb', carla_frame)      # 按 frame id 配对
        dvs = frame_reader.get_dvs(f'{camera}_dvs', carla_frame)  # DVS 按事件流取，可为空
        disk_writers[camera].submit(frame_index, frame_id, timestamp_us, dvs, rgb)  # 只入队
    annotations[camera_id].append(frame_record)
```

要点：
- 所有相机共用同一时间基准 `time_origin_us`，跨机位事件/帧可直接按时间戳关联；
- 传感器数据按 **CARLA frame id** 配对，而不是"按队列顺序"，避免多机位下错帧；
- **DVS 是例外**：CARLA 的 DVS 传感器只在"这一帧有事件"时才发数据，画面静止时
  一帧都不发，死等 frame 号会永远等不到（实测 cam_02 停在 frame=1934）。所以 DVS
  按流取：等一小会儿拿这一帧的事件，拿不到就合并队列里现有的，再没有就写空帧。
  事件自带绝对时间戳，空帧不影响时间精度；RGB/分割仍严格按 frame id 配对；
- bbox 投影在独立 worker 线程执行，主循环只做取数与入队；
- **落盘也是每相机一个后台线程**（`CameraDiskWriter`）：PNG 编码、HDF5 追加和时间戳
  写入全部移出主循环，磁盘卡顿被每相机 30 帧的有界队列吸收，不再传导成传感器回调
  超时；磁盘长期跟不上时队列写满会显式报错，而不是无限占用内存。实测该改动把
  3 相机的帧耗时从 1.34 s 降到 0.40 s（瓶颈回到服务端渲染）；
- 事件按帧攒批写 HDF5（`--flush-frames`，默认 10），避免长序列内存膨胀和每帧刷盘；
- 写盘线程一旦失败（磁盘满等），主循环下一帧立即中止并带出真实原因；
- 结束时销毁本次 actor 并恢复 world 设置（见 §5.3）。

### 5.2 批量采集（`tools/collect_batch.py`）

```powershell
# 1) 自动布点（会加载该地图）
python .\tools\plan_rig.py --town Town03 --out-dir .\rigs\Town03

# 2) 批量采集（每个机位组一条序列，--repeat 控制重复遍数）
python .\tools\collect_batch.py --rig-dir .\rigs\Town03 --repeat 2 --output .\dvs_roadside_multicam
```

- **必须串行**：同步模式下同一个 world 只能由一个客户端驱动 tick，多个采集器并行会互相抢 tick；
- 每次运行消耗一个全局序列号 → 种子自动按序列号派生 → 同机位不同遍数的交通流/天气互不相同且可复现；
- 单条失败自动重试（`--retries`，默认 1 次），失败序列保留现场并跳过序列号登记；
- `--cameras N` 让每个机位组只启用前 N 台相机，`--rigs a.yaml,b.yaml` 只采集指定机位组；
- **实时日志**：采集器每 10 帧打印一次进度（帧数、已用时间、每帧耗时、各机位累计事件），
  批量器实时透传子进程输出，每条序列打印一行结果摘要，每 10 条打印累计汇总；
- 传感器回调超时默认 **15 秒**（`--sensor-timeout`）：仿真通常远慢于实时（0.5–1.5 s/帧），
  超时太紧会在长序列末尾误判失败；
- 结束后写出 `batch_report.json`（每条序列的机位组、遍数、重试次数、耗时、失败日志尾部）。
- **每次运行落一个采集日志目录**（`--log-dir`，默认 `logs/`），见 §6.6。

### 5.2.1 一键采集向导（`tools/auto_collect.py` + `一键采集.bat`）

面向日常使用的交互式入口：双击项目根目录的 `一键采集.bat`，脚本会**自动启动 CARLA
仿真平台**（`D:\CARLA_0.9.16`，端口 2000），随后按提示输入核心参数并自动跑完
布点 → 采集 → 质检 → 索引摘要。

| 步骤 | 参数 | 默认值 |
|---|---|---|
| 1 | CARLA 主机 / 端口 / Traffic Manager 端口 | 127.0.0.1 / 2000 / 8000 |
| 2 | 重新布点、目标地图、选择机位组 | 用当前地图 / 全部 |
| 3 | **每个机位组启用的相机个数**、每个相机采集时长、帧率 | 全部 / 90 秒 / 20 Hz |
| 4 | 车辆数、行人数、固定天气 | 80 / 40 / 按种子随机 |
| 5 | 输出目录、采集日志目录、重复遍数 | `dvs_roadside_multicam` / `logs` / 1 |

向导是 `plan_rig.py` + `collect_batch.py` + `check_dataset.py` 的编排层，不含任何
新的采集逻辑；`CARLA_HOME`、`PY`、`PORT` 可在 `.bat` 顶部或同名环境变量中修改。

### 5.3 清理与 CARLA 0.9.16 差异

- `client.apply_batch()` 在 **0.9.16 返回 None**（无逐条结果），逐条错误信息要用 `client.apply_batch_sync(commands, do_tick=True)`。v2.0 之前按 `apply_batch()` 的返回值遍历，导致每次采集都在清理阶段抛 `TypeError`，清单停留在 `in_progress`、序列不进索引——v3.0 已修正。
- 采集结束必须恢复 `world` 设置与 Traffic Manager 同步模式，否则该 world 会一直处于同步模式，后续自动布点会找不到信号灯。
- `--load-town` 会调用 `client.load_world()` 重载当前 world，影响连接同一服务端的其他客户端；无窗口模式（`-RenderOffScreen`）下切图可能长时间无响应。

---

## 6. 数据结构与格式（对齐《数据结构规范 v1.1》，多机位扩展）

### 6.1 目录结构（无训练/测试划分）

```text
dvs_roadside_multicam/
├── README.md / LICENSE.txt / CHANGELOG.md
├── sequences/                        # 所有采集序列，单一目录，不分 train/val/test
│   └── sequence_000001/
│       ├── cameras/
│       │   ├── cam_01/
│       │   │   ├── events.h5          # /events/t,x,y,p + /meta/*
│       │   │   ├── rgb/               # 000000.png ...
│       │   │   ├── timestamps.txt     # 每帧中心时刻（µs，与事件同基准）
│       │   │   └── annotations.json   # 该相机视角的检测标注
│       │   ├── cam_02/ ... cam_08/
│       ├── calibration/
│       │   ├── extrinsics_world.yaml  # 每台相机的外参（世界坐标系）
│       │   ├── intrinsics_dvs.yaml    # 每台相机的 DVS 内参（CARLA 理想针孔，畸变=0）
│       │   ├── intrinsics_rgb.yaml
│       │   ├── dvs_rgb_extrinsic.yaml # 同机位 DVS↔RGB 外参（理论恒等，实测记录）
│       │   └── time_offset.yaml       # DVS↔RGB 帧时间偏移（标定值）
│       ├── event_representations/     # 可选：预生成事件帧/体素
│       └── meta/
│           ├── episode_manifest.json  # 位置/时间/地图/天气/机位组/种子/交通密度
│           ├── dvs_params.json        # 每台相机阈值/不应期/曝光
│           └── rig_geometry.json      # 各相机相对位姿（多机位关联用）
├── calibration/rigs/                 # 全局：本次用到的机位组 YAML + poles.json
├── index_all.txt                     # 全量索引（所有完成序列一行，不做数据集划分）
├── locations.csv                     # 位置时间索引：每条序列一行
├── batch_report.json                 # 批量采集执行报告
└── tools/                            # 质检与可视化脚本
```

> **不做 train/val/test 划分**：所有序列平铺在 `sequences/` 下，`index_all.txt` 是全量索引，
> `locations.csv` 记录每条序列的采集位置与时间；需要划分时由下游按 `locations.csv` 的
> 路口或时间段自行过滤。

**布点侧产物**（在 `rigs/<town>/` 下，不进数据集目录）：

```text
rigs/Town03/
├── Town03_j012.yaml          # 一个路口 = 一个机位组
├── Town03_j045.yaml
├── poles.json                # 杆位 ↔ 机位对照（杆位坐标、灯头 actor id、目标点、距车道距离）
├── poles.md                  # 同上的表格版，便于人工复核与手改 pitch/yaw
└── commands.txt              # 逐机位组的采集命令
```

### 6.2 事件文件（`cam_XX/events.h5`）

| HDF5 路径 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `/events/t` | (N,) | uint64 | **ns**，**全序列统一基准**（序列内从 0 开始） |
| `/events/x` | (N,) | uint16 | [0, width-1] |
| `/events/y` | (N,) | uint16 | [0, height-1]，y 向下 |
| `/events/p` | (N,) | int8 | ON=1 / OFF=−1 |
| `/frames/event_offsets` | (F+1,) | uint64 | 第 i 帧事件区间 `[offsets[i], offsets[i+1])` |
| `/frames/event_count` | (F,) | uint64 | 每帧事件数 |
| `/frames/frame_id` | (F,) | string | `{sequence_id}_{camera_id}_{帧号:06d}` |
| `/frames/timestamp_us` | (F,) | uint64 | 每帧时刻（µs，与事件同基准） |
| `/meta/sensor_width` / `sensor_height` | 标量 | uint16 | 1280×720 |
| `/meta/camera_id` | string | — | 如 `cam_01` |
| `/meta/town` / `/meta/weather` | string | — | 场景标识 |
| `/meta/dvs_params` | JSON 字符串 | — | 阈值/不应期 |
| `/meta/time_origin_us` / `/meta/time_origin_ns` | 标量 | uint64 | 序列起点（仿真时刻） |
| `/meta/event_timestamp_unit` | string | — | 恒为 `ns` |
| `/meta/event_timestamp_source_unit` | string | — | CARLA 原始单位（`ns`/`us`，运行时自动识别） |

> **时间戳单位：纳秒。** CARLA 0.9.16 的 DVS 传感器给出的原始事件时间戳是**纳秒**，
> 而 20 Hz 下每帧约 4 万个事件（实测 3.2–7.7 万），微秒分辨率会让同一微秒挤进
> 多个事件、无法保持严格递增，因此事件时间轴统一用纳秒保存；
> `/frames/timestamp_us` 仍是微秒，换算关系是 `events/t ÷ 1000`。
> 写入器按帧绝对时间自动识别 CARLA 给的单位（`event_timestamp_source_unit`），
> 避免把微秒当纳秒（或反之）写入；`tools/test_event_timestamps.py` 覆盖这条链路。
> 2026-10-26 之前采集的数据是按微秒基准写的（时间轴整体错位），
> 用 `tools/fix_event_timestamps.py` 平移修复，质检工具会检查这一项。

### 6.3 标注格式（`cam_XX/annotations.json`）

沿用规范 v1.1 的序列级 JSON，**frame 为 RGB 帧（20 Hz）锚点**：

```json
{
  "sequence_id": "sequence_000001",
  "camera_id": "cam_01",
  "sensor_width": 1280,
  "sensor_height": 720,
  "frames": [
    {
      "frame_id": "sequence_000001_cam_01_000010",
      "timestamp_us": 500500,
      "integration_time_us": 50000,
      "annotations": [
        {
          "id": 1,
          "category": "vehicle",
          "bbox": [312, 288, 96, 54],
          "truncated": false, "occluded": false, "difficult": false,
          "speed_mps": 8.2,
          "actor_id": 1042,
          "stationary": false
        }
      ]
    }
  ]
}
```

扩展字段（相对 v1.1）：

| 字段 | 说明 |
|---|---|
| `actor_id` | CARLA actor 全局 ID，**跨机位关联同一目标的唯一键** |
| `speed_mps` / `stationary` | 静止目标标记（等灯车辆），静止期间事件稀疏，使用者可自行取舍 |
| `occluded` | 除遮挡外，跨机位可见性也可由 `actor_id` 推得 |

类别映射：CARLA `vehicle.*`（含 bicycle/motorcycle）→ `vehicle`；`walker.*` → `pedestrian`。

### 6.4 标定文件

- **DVS 与 RGB 同位同参**：CARLA 中二者理想针孔、无畸变，`dvs_rgb_extrinsic.yaml` 记录理论恒等变换 + 采集程序实测偏移；
- **内参**由分辨率与 FOV 推导：`fx = fy = (width/2) / tan(fov/2)`；
- **跨机位外参**（`extrinsics_world.yaml`）取自 CARLA 实际传感器位姿，支持多视角 3D 重建/关联评测；
- `time_offset.yaml`：DVS 事件与 RGB 帧在同一 tick 出帧，理论偏移 0；仍按规范要求实测抽检并记录。

### 6.5 位置与时间标注（自动生成）

每条序列在采集时自动写入位置和时间，无需人工登记。

**序列清单 `meta/episode_manifest.json`** 新增两个块：

```json
{
  "location": {
    "town": "Town10HD_Opt",
    "map_name": "Carla/Maps/Town10HD_Opt",
    "junction_id": 23,
    "junction_center_m": [-64.264, 7.063, 0.0],
    "geo": {"latitude": 41.8712, "longitude": -87.6298, "altitude": 181.0},
    "aim": "junction_center",
    "cameras": {
      "cam_01": {
        "location_m": [-64.264, 7.063, 4.054],
        "rotation_degrees": {"pitch": -10.88, "yaw": 36.402, "roll": 0.0},
        "geo": {"latitude": 41.8712, "longitude": -87.6298, "altitude": 181.0}
      }
    }
  },
  "capture_time": {
    "started_at_utc": "2026-10-26T06:45:12.123456+00:00",
    "started_at_local": "2026-10-26T14:45:12.123456+08:00",
    "timezone": "中国标准时间",
    "ended_at_utc": "2026-10-26T06:45:47.456789+00:00",
    "elapsed_wall_s": 35.33,
    "simulation_start_s": 12.34,
    "simulation_end_s": 102.34,
    "frame_count": 1800,
    "fps": 20.0
  }
}
```

- **位置**分两级：路口级（`junction_id` + 路口中心 + 地理坐标，来自机位组 YAML）和机位级
  （每台相机的世界坐标、朝向、地理坐标，运行时按实际位姿换算）；
- **地理坐标**由 `map.transform_to_geolocation()` 从 CARLA 地图自带的地理参考换算。
  注意 CARLA 官方地图的地理参考是任意的（Town10HD_Opt 的原点附近约 ±0.0005°），
  它是**地图内一致**的相对定位，不等于真实城市经纬度；需要真实坐标时按地图统一偏移即可；
- **时间**分两种：真实墙钟时间（UTC + 本地时区，用于数据集管理与溯源）与仿真时间
  （`simulation_start_s/end_s`，与事件时间戳同一基准）。

**数据集级 `locations.csv`**：每条序列一行，扁平字段，方便 pandas 直接筛选。

| 列 | 含义 |
|---|---|
| `sequence_id` | 序列号 |
| `capture_started_at_utc` / `capture_started_at_local` / `timezone` | 采集真实时刻 |
| `duration_s` | 序列时长 |
| `town` / `map_name` | 地图 |
| `junction_id` | 路口编号 |
| `junction_center_x_m` / `junction_center_y_m` | 路口中心（CARLA 世界坐标） |
| `geo_latitude` / `geo_longitude` / `geo_altitude_m` | 路口地理坐标 |
| `camera_count` / `camera_poses` | 机位数与各机位的位置朝向摘要 |
| `frame_count` / `fps` / `weather` / `seed` | 采集参数 |
| `total_events` | 该序列事件总数 |
| `vehicles` / `walkers` | 实际生成的车/人数量 |

同一序列重复写入时按 `sequence_id` 覆盖，不会产生重复行；`tools/check_dataset.py` 会校验
索引中的每条序列在 `locations.csv` 都有位置与时间记录。

### 6.6 采集日志目录（`logs/`）

数据集本身记录的是「数据是什么」，采集过程本身（跑了多久、哪个机位、采了多少、
中途报了什么错）单独记在项目根目录的 `logs/` 下，**每次采集一个子目录**：

```text
logs/
├── index.csv                              采集历史总表：每次采集追加一行
└── 20261026-160638_Town10HD_Opt/          <日期-时间>_<地图>
    ├── run.json                           结构化运行摘要
    ├── run.log                            完整控制台输出（含逐帧进度）
    └── sequences.csv                      每条序列一行
```

| 文件 | 内容 |
|---|---|
| `run.json` | 起止时间（UTC + 本地）、耗时、主机/端口/TM 端口、机位组目录与文件列表、地图、输出目录、重复遍数、相机数/帧数/帧率/车流/天气、成功失败数、事件总量、本次命令行、`sequences_detail`（逐条序列明细） |
| `run.log` | 控制台原样落盘：采集器每 10 帧的进度、每条序列的摘要、每 10 条序列的累计汇总、失败时的日志尾部 |
| `sequences.csv` | `sequence_id, rig, repeat, attempt, status, junction_id, camera_count, frame_count, total_events, weather, started_at_local, elapsed_s` |
| `index.csv` | `run_id, started_at_local, ended_at_local, towns, rig_dir, rig_count, repeat, sequences, succeeded, failed, total_events, elapsed_s, log_dir` |

写入时机：`sequences.csv` 每采完一条序列就追加一行（中途崩溃也不丢已完成的部分），
`run.json` 与 `index.csv` 行在整轮结束时写。
`--log-dir` 可改目录位置，向导里也会询问。

---

## 7. 规模化与吞吐

| 项目 | 实测/估算 |
|---|---|
| 单机位组 20 帧短序列 | 21–37 s（含交通生成与销毁），5 个路口 5 条序列共 151 s |
| 单序列（默认 90 s / 1800 帧 / 3–4 机位） | 约 5–10 min（受交通密度与事件量影响） |
| 落盘并行化后帧耗时 | 3 机位 0.40 s/帧、4 机位 0.55 s/帧（改前 3 机位 1.34 s/帧；瓶颈已是服务端渲染而非客户端写盘） |
| 事件量 | 4 机位 × 30 帧 ≈ 5.1 M 事件（约 40k–80k 事件/机位/帧） |
| 磁盘写入速率 | 8 机位 × 20 Hz 峰值约 20 MB/s（PNG + HDF5），SSD 均可承受；卡顿由每相机 30 帧队列吸收 |
| 并行 | **不可对同一 CARLA world 并行**（同步 tick 独占）；并行策略是"多个 CARLA 实例 × 不同 (town, rig) 组合" |
| 目标规模 | 每 Town 约 5–15 个信号化路口 × 每路口 2–4 机位；4 周生产 ≥ 400 序列 ≈ 72 万帧·机位 |

种子管理：种子由全局序列号派生，任意序列可复现。

---

## 8. 质量控制与验收

| 检查项 | 标准 | 工具 |
|---|---|---|
| 数据结构与同步 | 相机数 1–8、HDF5 字段类型、事件时间戳单调、帧事件偏移、RGB/标注帧数与时间戳一致、跨机位时间戳一致 | `tools/check_dataset.py` |
| 布点完整性 | 地图信号灯 actor 数 = 已归属路口的杆位数（未归属 = 0） | `tools/plan_rig.py` 输出 |
| 时间戳完整性 | DVS↔RGB 偏差 < 1 仿真步长（50 ms @20 Hz 内帧对齐，事件 µs 级自由） | 质检工具 |
| 跨机位同步 | 同一 `actor_id` 在各相机标注时间窗内同时出现 | 抽检 |
| 事件密度 | 每帧每相机 1k–200k 事件；整段 <1k/帧 的场景判为交通流不足，重采 | 质检工具 + 统计 |
| 类别覆盖 | vehicle / pedestrian 帧占比各 ≥ 5% | 标注统计 |
| 标注精度 | 抽检 200 帧，投影 bbox 与事件簇边缘偏差 < 15 px | 可视化抽检 |
| 小目标覆盖 | 远端机位行人目标 ≥ 20 px 高的帧占比 ≥ 10% | 标注统计 |
| 可复现性 | seed + rig + 配置可完整重放 | `batch_report.json` + manifest |
| 位置时间标注 | 索引中每条序列在 `locations.csv` 都有位置、地理坐标和采集时间 | `tools/check_dataset.py` |
| 事件时间单位 | 每帧最后一个事件落在该帧曝光窗口内（帧时刻 ± 积分时间），即 `events/t` 与 `frames/timestamp_us` 同一时间轴 | `tools/check_dataset.py`（`tools/fix_event_timestamps.py` 修复历史数据） |

---

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| 静止相机导致静止目标无事件（等灯车辆） | 标注流水线标记 `stationary` 字段；红绿灯场景保留（事件重启时产生强边缘），README 中说明 |
| 单路口交通流模式单一 | 每序列随机 `ignore_lights_percentage`、速度扰动；多 Town + 多机位组轮换 |
| DVS 回调阻塞丢帧 | 回调仅 `put()`；落盘/投影在 worker |
| **多相机同时写盘造成卡顿 → 回调超时** | 每相机一个后台落盘线程，磁盘 I/O 完全移出采集主循环；有界队列（30 帧）吸收卡顿，写满报错；`--flush-frames` 默认 10 批量刷 HDF5（v3.1 已修，实测 3 相机 1.34→0.40 s/帧） |
| 事件量过大撑爆内存 | 攒批阈值（每帧或 10k 事件即写 h5） |
| 多实例 GPU 争抢 | 限 `fixed_delta_seconds`、错峰、`-quality-level Low` |
| 合成数据域差 | 保留 RGB 原帧；可用 v2e/ESIM 对 RGB 帧再仿真真实噪声（热噪声、泄漏、低频截止）做增强 |
| 夜晚事件稀疏 | 低阈值预设（0.10）+ 提高 ClearNight/HardRainNight 序列占比 + 车灯照明 |
| **同步模式下自动布点找不到信号灯** | 布点工具检测到同步模式直接报错退出；采集前先跑布点 |
| **`apply_batch()` 在 0.9.16 返回 None** | 清理改用 `apply_batch_sync()`（v3.0 已修） |
| **切图/无窗口模式卡死** | `--load-town` 只在必要时使用；无窗口模式（`-RenderOffScreen`）下建议切图前重启 CARLA |
| **杆位数量少于预期** | 优化版地图（`*_Opt`）信号灯较少；布点时比较各图的 actor 数，必要时换完整版地图 |
| **DVS 原始时间戳是纳秒** | 事件时间轴统一按纳秒落盘，写入器自动识别 CARLA 给的单位；质检工具检查事件与帧时间是否同轴；历史数据用 `tools/fix_event_timestamps.py` 平移修复 |

---

## 10. 实施计划

| 阶段 | 状态 | 交付物 |
|---|---|---|
| 1. 环境搭建与单机验证 | ✅ 完成 | CARLA 0.9.16 + conda `dvs` 环境；DVS+RGB 同步验证 |
| 2. 自动布点工具 | ✅ 完成 | `tools/plan_rig.py`：路口/杆位枚举、朝向解算、机位组 YAML + 杆位报告；`--self-test` |
| 3. 多机位采集与标注 | ✅ 完成 | `dvs_collector`：固定机位采集、bbox 投影 + `actor_id`、标定、HDF5 落盘 |
| 4. 批量调度与质检 | ✅ 完成 | `tools/collect_batch.py`（串行、重试、报告）、`tools/check_dataset.py` |
| 4.1 一键采集向导 | ✅ 完成 | `一键采集.bat` + `tools/auto_collect.py`：自动启动 CARLA、交互式参数、端到端采集与质检 |
| 5. 规模化生产 | ⏳ 进行中 | 多 Town 批量采集；每 Town 杆位覆盖率核对；数据集发布 v1.0 |
| 6. 训练侧准备 | ⏳ 待办 | 标注质检（静止目标过滤、行人占比、跨机位一致性）、训练/测试划分、基线检测模型 |

---

## 附录 A：机位组 YAML 与自动布点用法

```yaml
# rigs/Town03/Town03_j012.yaml —— 一个路口一个机位组
town: Town03
sequence_duration_s: 90.0
fps: 20.0
seed: null
cameras:
  cam_01:
    location: [-64.264, 7.063, 4.054]   # 杆位 x, y, 安装高度（杆上灯头顶面）
    pitch: -10.88                        # 负值为俯视
    yaw: 36.402                          # 0 = +X，90 = +Y
    fov: 60.0
    preset: road_day
traffic:
  vehicles: 80
  walkers: 40
  ignore_lights_pct: [0, 30]
  speed_difference_pct: [-10, 20]
  following_distance_m: [2, 6]
weather_cycle: [ClearNoon, CloudyNoon, SoftRainSunset, WetSunset, ClearNight]
```

```powershell
# 自动布点（--dry-run 只看不写；--self-test 不连 CARLA）
python .\tools\plan_rig.py --town Town03 --out-dir .\rigs\Town03 --dry-run
python .\tools\plan_rig.py --town Town03 --out-dir .\rigs\Town03
python .\tools\plan_rig.py --town Town03 --out-dir .\rigs\Town03 --aim approach --mount-height 6

# 批量采集 + 质检
python .\tools\collect_batch.py --rig-dir .\rigs\Town03 --repeat 2 --output .\dvs_roadside_multicam
python .\tools\check_dataset.py --dataset .\dvs_roadside_multicam
```

## 附录 B：环境与已知限制

- 环境：CARLA **0.9.16**、Python **3.12**、conda 环境 `dvs`、依赖 NumPy / h5py / PyYAML；
- CARLA 安装目录 `D:\CARLA_0.9.16`，Python API wheel 在 `PythonAPI/carla/dist`；
- 采集期间同一 world 必须由采集器独占同步 tick，其他客户端不要同时 `world.tick()`；
- 失败采集不会写入 `index_all.txt`，失败目录保留用于诊断并占用序列号；
- 路灯杆在 CARLA 中没有枚举 API（静态几何体），本方案只覆盖信号灯杆；
- 无窗口模式（`-RenderOffScreen`）下 `client.load_world()` 可能长时间无响应，需在窗口模式切换地图；
- **DVS 原始事件时间戳的单位是纳秒**（0.9.16 实测），不是微秒；本项目事件时间轴统一用纳秒，
  详见 §6.2；
- 传感器回调偶发丢帧会让某台 DVS 的时间戳在一段时间内塌缩成同一个值（一次失败采集里出现过），
  这种文件无法用常量平移修复，只能丢弃重采——质检工具的帧窗口检查能把它挑出来。
