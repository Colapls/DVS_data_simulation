from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .labels import MAX_DETECT_DISTANCE_M, MIN_DETECT_SHORT_SIDE_PX


# 默认配置只包含技术方案附录里给出明确坐标的示例机位。
DEFAULT_RIG = Path(__file__).resolve().parent.parent / "rigs" / "town03_intersection_a.yaml"
WEATHER_PRESETS = (
    "ClearNoon",
    "CloudyNoon",
    "SoftRainSunset",
    "WetSunset",
    "ClearNight",
    "HardRainNight",
)

#: 天气预设的中文名（界面显示用；CARLA API 只认英文名）。
#: 覆盖 CARLA 0.9.x 的全部预设，手动 --weather 传了表外的名字也不会显示成英文。
WEATHER_LABELS = {
    "Default": "默认",
    "ClearNoon": "晴天正午",
    "CloudyNoon": "多云正午",
    "WetNoon": "湿地正午",
    "WetCloudyNoon": "湿地多云正午",
    "MidRainyNoon": "中雨正午",
    "HardRainNoon": "大雨正午",
    "SoftRainNoon": "小雨正午",
    "ClearSunset": "晴天日落",
    "CloudySunset": "多云日落",
    "WetSunset": "湿地日落",
    "WetCloudySunset": "湿地多云日落",
    "MidRainSunset": "中雨日落",
    "HardRainSunset": "大雨日落",
    "SoftRainSunset": "小雨日落",
    "ClearNight": "晴夜",
    "CloudyNight": "多云夜",
    "WetNight": "湿地夜",
    "WetCloudyNight": "湿地多云夜",
    "MidRainNight": "中雨夜",
    "HardRainNight": "大雨夜",
    "SoftRainNight": "小雨夜",
    "DustStorm": "沙尘暴",
}


def weather_label(name: str) -> str:
    """天气预设的中文名；未知名字原样返回。"""
    return WEATHER_LABELS.get(name, name)
CAMERA_ID_PATTERN = re.compile(r"^cam_\d{2}$")

#: CARLA DVS 回调的原始字节布局：x、y、t、极性。
DVS_EVENT_DTYPE = [
    ("x", "u2"),
    ("y", "u2"),
    ("t", "i8"),
    ("pol", "?"),
]

DVS_PRESETS: dict[str, dict[str, Any]] = {
    # 曝光必须锁成 manual。
    #
    # 自动曝光（histogram）每帧都在按直方图调亮度，画面整体一变，
    # 满屏像素同时越过阈值 —— 那不是事件流，是曝光抖动。实测 Town03
    # ClearNight 静止场景：histogram 56.8 万事件/帧、16.4% 的像素在触发；
    # 同一个场景 manual 只有 902 事件/帧、0.1% 的像素。
    # 手动曝光下动目标仍有 3357 事件/帧，信号没有被丢掉。
    # 不要再加 exposure_compensation：+2 反而让事件掉到个位数，
    # +4 直接连帧都不发（画面被推到过曝/欠曝的极端）。
    #
    # 低照度降低事件阈值，以提高暗光下的事件响应。
    "road_day": {
        "width": 1280,
        "height": 720,
        "fov": 60.0,
        "positive_threshold": 0.25,
        "negative_threshold": 0.25,
        "refractory_ms": 0.5,
        "exposure_mode": "manual",
    },
    "road_lowlight": {
        "width": 1280,
        "height": 720,
        "fov": 60.0,
        "positive_threshold": 0.10,
        "negative_threshold": 0.10,
        "refractory_ms": 0.3,
        "exposure_mode": "manual",
    },
    "road_near": {
        "width": 1280,
        "height": 720,
        "fov": 90.0,
        "positive_threshold": 0.20,
        "negative_threshold": 0.20,
        "refractory_ms": 0.5,
        "exposure_mode": "manual",
    },
}


@dataclass(frozen=True)
class CameraConfig:
    """单台固定融合相机的安装位姿、成像参数和 DVS 参数。"""

    camera_id: str
    location: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    width: int
    height: int
    fov: float
    positive_threshold: float
    negative_threshold: float
    refractory_ms: float
    exposure_mode: str | None

    def dvs_parameters(self) -> dict[str, Any]:
        """返回可写入清单/HDF5 的 DVS 参数快照。"""
        return {
            "image_size_x": self.width,
            "image_size_y": self.height,
            "fov": self.fov,
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "refractory_period_ms": self.refractory_ms,
            "exposure_mode": self.exposure_mode,
        }

    def mount(self) -> dict[str, Any]:
        """使用世界固定标记，明确该机位不跟随任何车辆移动。"""
        return {
            "attachment": "world_fixed",
            "location_m": list(self.location),
            "rotation_degrees": {
                "pitch": self.pitch,
                "yaw": self.yaw,
                "roll": self.roll,
            },
        }


@dataclass(frozen=True)
class RigConfig:
    """从机位组 YAML 解析出的整套路侧采集配置。"""

    source: Path
    town: str
    cameras: tuple[CameraConfig, ...]
    traffic: dict[str, Any]
    weather_cycle: tuple[str, ...]
    sequence_duration_s: float
    fps: float
    seed: int | None
    location: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="采集 CARLA 路侧固定多机位 DVS 事件流、同步 RGB 和检测标注。"
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA 服务端地址")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC 端口")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager 端口")
    parser.add_argument("--rig", type=Path, default=DEFAULT_RIG, help="机位组 YAML")
    parser.add_argument(
        "--load-town",
        action="store_true",
        help="按 rig YAML 切换 CARLA 地图（会重载当前 world）",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dvs_roadside_multicam")
    )
    parser.add_argument("--duration", type=float, help="覆盖 YAML 中的序列时长（秒）")
    parser.add_argument("--frames", type=int, help="覆盖时长，直接指定采集帧数")
    parser.add_argument("--fps", type=float, help="覆盖 YAML 帧率")
    parser.add_argument("--seed", type=int, help="随机种子；默认按序列编号派生")
    parser.add_argument("--weather", choices=WEATHER_PRESETS, help="覆盖随机天气")
    parser.add_argument("--vehicles", type=int, help="覆盖机位 YAML 中的车辆数")
    parser.add_argument("--walkers", type=int, help="覆盖机位 YAML 中的行人数")
    parser.add_argument(
        "--cameras",
        type=int,
        help="只启用机位组里的前 N 台相机（按 camera_id 排序）",
    )
    parser.add_argument("--min-speed", type=float, default=0.1, help="静止标记速度阈值 m/s")
    parser.add_argument(
        "--flush-frames",
        type=int,
        default=10,
        help="HDF5 批量写入帧数；越大磁盘写入越平滑",
    )
    parser.add_argument(
        "--sensor-timeout",
        type=float,
        default=15.0,
        help="传感器回调超时秒数；仿真慢于实时时需要放宽",
    )
    parser.add_argument(
        "--instance-seg",
        action="store_true",
        default=True,
        help="开启实例分割传感器（算可见度/遮挡，方案 D4，默认开）",
    )
    parser.add_argument(
        "--no-instance-seg",
        dest="instance_seg",
        action="store_false",
        help="关闭实例分割传感器：没有可见度，occluded/difficult 恒为 False",
    )
    parser.add_argument(
        "--instance-seg-scale",
        type=int,
        default=2,
        help="实例分割图相对 DVS 分辨率的降采样倍数（2 = 半分辨率）",
    )
    parser.add_argument(
        "--rgb-format",
        choices=("png", "jpeg"),
        default="png",
        help="RGB 帧格式；jpeg 体积约为 png 的 1/10（方案 §8 存储瘦身）",
    )
    parser.add_argument(
        "--rgb-quality",
        type=int,
        default=90,
        help="--rgb-format jpeg 时的 JPEG 质量",
    )
    parser.add_argument(
        "--min-detect-px",
        type=int,
        default=MIN_DETECT_SHORT_SIDE_PX,
        help=(
            "可探测的最小投影短边（像素）：目标在画面上比这更小就标为 ignore"
            f"（默认 {MIN_DETECT_SHORT_SIDE_PX}，按 DVSync 的角分辨率与实测"
            "事件产出率标定）"
        ),
    )
    parser.add_argument(
        "--max-detect-distance",
        type=float,
        default=MAX_DETECT_DISTANCE_M,
        help=(
            "最大标注距离（米）：超过这个距离的目标标为 ignore"
            f"（默认 {MAX_DETECT_DISTANCE_M}）"
        ),
    )
    return parser.parse_args()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """在读取 YAML 字段时明确检查映射类型，尽早报告配置错误。"""
    if not isinstance(value, dict):
        raise ValueError(f"机位 YAML 的 {name} 必须是映射")
    return value


def load_dependencies() -> tuple[Any, Any, Any]:
    """导入 CARLA / h5py / numpy，缺依赖时给出可执行的中文提示。"""
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(
            "找不到 CARLA Python API。请激活 dvs Conda 环境并安装 "
            r"D:\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl。"
        ) from exc

    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "缺少采集依赖，请在 dvs 环境执行：python -m pip install -r requirements.txt"
        ) from exc
    return carla, h5py, np


def load_rig(path: Path) -> RigConfig:
    """加载并校验机位组；单机参数覆盖内置预设，最终值写入 CameraConfig。"""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "机位组需要 PyYAML；请在 dvs 环境执行 "
            "python -m pip install -r requirements.txt"
        ) from exc

    rig_path = path.expanduser().resolve()
    if not rig_path.is_file():
        raise ValueError(f"找不到机位组 YAML：{rig_path}")
    try:
        raw = yaml.safe_load(rig_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取机位组 YAML {rig_path}: {exc}") from exc
    root = _mapping(raw, "根节点")

    town = root.get("town")
    if not isinstance(town, str) or not town.strip():
        raise ValueError("机位组必须指定 town")
    raw_cameras = _mapping(root.get("cameras"), "cameras")
    if not 1 <= len(raw_cameras) <= 8:
        raise ValueError("机位组必须配置 1–8 台相机")

    # YAML 可扩展或覆盖预设；具体相机字段再覆盖预设默认值。
    presets = {**DVS_PRESETS}
    for name, values in _mapping(root.get("dvs_presets", {}), "dvs_presets").items():
        presets[name] = {**presets.get(name, {}), **_mapping(values, f"dvs_presets.{name}")}

    cameras: list[CameraConfig] = []
    for camera_id, raw_camera in raw_cameras.items():
        if not isinstance(camera_id, str) or not CAMERA_ID_PATTERN.fullmatch(camera_id):
            raise ValueError(f"相机 ID 必须符合 cam_XX 格式：{camera_id!r}")
        camera = _mapping(raw_camera, f"cameras.{camera_id}")
        preset_name = camera.get("preset", "road_day")
        if preset_name not in presets:
            raise ValueError(f"{camera_id} 引用了未知 DVS preset：{preset_name}")
        values = {**presets[preset_name], **camera}
        try:
            location = tuple(float(v) for v in values["location"])
            if len(location) != 3:
                raise ValueError
            config = CameraConfig(
                camera_id=camera_id,
                location=location,
                pitch=float(values.get("pitch", 0.0)),
                yaw=float(values.get("yaw", 0.0)),
                roll=float(values.get("roll", 0.0)),
                width=int(values["width"]),
                height=int(values["height"]),
                fov=float(values["fov"]),
                positive_threshold=float(values["positive_threshold"]),
                negative_threshold=float(values["negative_threshold"]),
                refractory_ms=float(values["refractory_ms"]),
                exposure_mode=values.get("exposure_mode"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{camera_id} 的位置或相机参数无效") from exc
        if not 1 <= config.width <= 65535 or not 1 <= config.height <= 65535:
            raise ValueError(f"{camera_id} 的分辨率必须在 uint16 范围内")
        if not 0 < config.fov < 180:
            raise ValueError(f"{camera_id} 的 fov 必须在 (0, 180) 范围内")
        if config.positive_threshold <= 0 or config.negative_threshold <= 0:
            raise ValueError(f"{camera_id} 的 DVS 正负阈值必须大于 0")
        if config.refractory_ms < 0:
            raise ValueError(f"{camera_id} 的 refractory_ms 不能小于 0")
        cameras.append(config)

    raw_traffic = _mapping(root.get("traffic", {}), "traffic")
    traffic = {
        "vehicles": int(raw_traffic.get("vehicles", 80)),
        "walkers": int(raw_traffic.get("walkers", 40)),
        "ignore_lights_pct": raw_traffic.get("ignore_lights_pct", [0, 30]),
        "speed_difference_pct": raw_traffic.get("speed_difference_pct", [-10, 20]),
        "following_distance_m": raw_traffic.get("following_distance_m", [2, 6]),
        # 摩托车占车辆的比例（方案 §6.2 建议 7:3）。
        "motorcycle_ratio": float(raw_traffic.get("motorcycle_ratio", 0.3)),
        # 行人在路口周边的出生半径和行走距离（方案 §6.1）。
        "pedestrian_spawn_radius_m": raw_traffic.get(
            "pedestrian_spawn_radius_m", [30, 60]
        ),
        "pedestrian_walk_distance_m": raw_traffic.get(
            "pedestrian_walk_distance_m", [15, 40]
        ),
        # 横穿人行横道的行人占比。
        "crosswalk_ratio": float(raw_traffic.get("crosswalk_ratio", 0.4)),
        # 车辆出生点优先落在相机视野内（相机高 4 m、俯角 14°~17° 时，
        # 地面可见范围约 14~39 m）。这两个半径决定"看得见"的判定范围。
        "spawn_far_camera_m": float(raw_traffic.get("spawn_far_camera_m", 60.0)),
        "spawn_near_camera_m": float(raw_traffic.get("spawn_near_camera_m", 45.0)),
    }
    if traffic["vehicles"] < 0 or traffic["walkers"] < 0:
        raise ValueError("车辆数和行人数不能小于 0")
    if not 0.0 <= traffic["motorcycle_ratio"] <= 1.0:
        raise ValueError("traffic.motorcycle_ratio 必须在 [0, 1] 之间")
    if not 0.0 <= traffic["crosswalk_ratio"] <= 1.0:
        raise ValueError("traffic.crosswalk_ratio 必须在 [0, 1] 之间")
    if traffic["spawn_far_camera_m"] <= 0.0:
        raise ValueError("traffic.spawn_far_camera_m 必须大于 0")
    if not 0.0 < traffic["spawn_near_camera_m"] <= traffic["spawn_far_camera_m"]:
        raise ValueError(
            "traffic.spawn_near_camera_m 必须大于 0 且不超过 spawn_far_camera_m"
        )
    for key in ("ignore_lights_pct", "speed_difference_pct", "following_distance_m"):
        limits = traffic[key]
        if (
            not isinstance(limits, (list, tuple))
            or len(limits) != 2
            or float(limits[0]) > float(limits[1])
        ):
            raise ValueError(f"traffic.{key} 必须是 [最小值, 最大值]")
        traffic[key] = [float(limits[0]), float(limits[1])]
    for key in ("pedestrian_spawn_radius_m", "pedestrian_walk_distance_m"):
        limits = traffic[key]
        if (
            not isinstance(limits, (list, tuple))
            or len(limits) != 2
            or float(limits[0]) < 0
            or float(limits[0]) > float(limits[1])
        ):
            raise ValueError(f"traffic.{key} 必须是 [最小值, 最大值]")
        traffic[key] = [float(limits[0]), float(limits[1])]

    # 对机位排序，保证传感器创建顺序和输出顺序可重复。
    weather_cycle = root.get("weather_cycle", ["ClearNoon", "CloudyNoon"])
    if (
        not isinstance(weather_cycle, list)
        or not weather_cycle
        or any(weather not in WEATHER_PRESETS for weather in weather_cycle)
    ):
        raise ValueError("weather_cycle 必须包含一个或多个受支持的天气预设")
    duration = float(root.get("sequence_duration_s", 90))
    # 40 Hz 是方案 §5.2 的建议值：标签落在每个帧边界上，25/50/100 ms 三种
    # 事件窗口都能用真实标签支撑，不用伪造插值。
    fps = float(root.get("fps", 40))
    seed = root.get("seed")
    if duration <= 0 or fps <= 0:
        raise ValueError("sequence_duration_s 和 fps 必须大于 0")
    if seed is not None:
        seed = int(seed)
    # 采集位置：由 tools/plan_rig.py 写入路口编号、路口中心和地理坐标。
    location = _mapping(root.get("location", {}), "location")
    return RigConfig(
        source=rig_path,
        town=town,
        cameras=tuple(sorted(cameras, key=lambda item: item.camera_id)),
        traffic=traffic,
        weather_cycle=tuple(weather_cycle),
        sequence_duration_s=duration,
        fps=fps,
        seed=seed,
        location=location,
    )


def select_cameras(rig: RigConfig, count: int) -> RigConfig:
    """只启用机位组里的前 count 台相机（按 camera_id 排序）。"""
    if count < 1:
        raise ValueError("--cameras 必须大于 0")
    if count >= len(rig.cameras):
        return rig
    return replace(rig, cameras=rig.cameras[:count])


def validate_args(args: argparse.Namespace) -> None:
    """检查命令行组合和数值范围，避免采集开始后才发现配置冲突。"""
    if args.duration is not None and args.duration <= 0:
        raise ValueError("--duration 必须大于 0")
    if args.frames is not None and args.frames <= 0:
        raise ValueError("--frames 必须大于 0")
    if args.duration is not None and args.frames is not None:
        raise ValueError("--duration 和 --frames 不能同时指定")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps 必须大于 0")
    if args.vehicles is not None and args.vehicles < 0:
        raise ValueError("--vehicles 不能小于 0")
    if args.walkers is not None and args.walkers < 0:
        raise ValueError("--walkers 不能小于 0")
    if args.cameras is not None and args.cameras < 1:
        raise ValueError("--cameras 必须大于 0")
    if args.min_speed < 0:
        raise ValueError("--min-speed 不能小于 0")
    if args.flush_frames <= 0 or args.sensor_timeout <= 0:
        raise ValueError("--flush-frames 和 --sensor-timeout 必须大于 0")
    if args.instance_seg_scale < 1:
        raise ValueError("--instance-seg-scale 必须大于等于 1")
    if not 1 <= args.rgb_quality <= 100:
        raise ValueError("--rgb-quality 必须在 [1, 100] 之间")
