#!/usr/bin/env python3
"""量一量 CARLA 0.9.16 的 DVS 在不同天气/曝光设置下每帧到底出多少事件。

要回答三个问题：

1. 0.9.16 的 DVS 蓝图到底有哪些参数可用（曝光模式、手动曝光偏置…）；
2. 默认的 ``exposure_mode="histogram"``（自动曝光）是不是每帧都在改亮度，
   把整幅画面变成"全像素事件"；
3. 换成手动曝光后，夜里还采不采得到事件（画面会不会黑成一片）。

做法：挂一台 DVS，按 天气 × 曝光设置 组合各跑 N 帧，输出

- 每帧事件数；
- 有事件的像素占全画面的比例（区分"全局曝光抖动"和"局部运动"）；
- 再挂一个移动目标，确认换完曝光后事件还在。

    E:/ProgramData/miniconda3/envs/dvs/python.exe tools/probe_dvs_exposure.py --port 2000

退出码 0 = 全部用例跑完。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 和 dvs_collector.config.DVS_EVENT_DTYPE 一致：x(u2) y(u2) t(i8) pol(bool)。
DTYPE = [("x", "u2"), ("y", "u2"), ("t", "i8"), ("pol", "?")]
ITEMSIZE = 13

# 机位参数照抄 rigs/town03_intersection_a.yaml（cam_01 的 road_day 预设）。
CAM_X, CAM_Y, CAM_Z = 88.2, -12.5, 6.0
CAM_YAW = 180.0
WIDTH, HEIGHT, FOV = 1280, 720, 60.0
POS_THRESH, NEG_THRESH, REFRACTORY_NS = 0.25, 0.25, 500_000

WEATHERS = ("ClearNoon", "ClearNight")
SETTINGS: list[tuple[str, float | None]] = [
    ("histogram", None),
    ("manual", None),
    ("manual", 2.0),
    ("manual", 4.0),
]
WARMUP_FRAMES = 5


def dump_attributes(world: object) -> None:
    """打印 DVS / RGB 蓝图的全部属性，看 0.9.16 到底给了哪些旋钮。"""
    library = world.get_blueprint_library()
    for kind in ("dvs", "rgb"):
        blueprint = library.find(f"sensor.camera.{kind}")
        print(f"--- sensor.camera.{kind} 可用属性 ---")
        for index, attribute in enumerate(blueprint):
            if index == 0:
                # 0.9.16 的 ActorAttribute 取值接口和文档对不上，
                # 直接把能用的属性名列出来，省得再来一轮。
                names = [
                    name
                    for name in dir(attribute)
                    if not name.startswith("_")
                ]
                print(f"  [ActorAttribute 可用属性] {names}")
            value = ""
            for name in ("value", "default_value"):
                if hasattr(attribute, name):
                    value = getattr(attribute, name)
                    break
            extra = ""
            if hasattr(attribute, "recommended_values"):
                try:
                    extra = f" 推荐值={list(attribute.recommended_values)}"
                except (RuntimeError, TypeError):
                    extra = ""
            print(
                "  %-28s type=%-8s 值=%r%s"
                % (attribute.id, attribute.type, value, extra)
            )


def _blueprint(world: object, mode: str, compensation: float | None) -> object:
    blueprint = world.get_blueprint_library().find("sensor.camera.dvs")
    blueprint.set_attribute("image_size_x", str(WIDTH))
    blueprint.set_attribute("image_size_y", str(HEIGHT))
    blueprint.set_attribute("fov", str(FOV))
    blueprint.set_attribute("positive_threshold", str(POS_THRESH))
    blueprint.set_attribute("negative_threshold", str(NEG_THRESH))
    blueprint.set_attribute("refractory_period_ns", str(REFRACTORY_NS))
    if blueprint.has_attribute("exposure_mode"):
        blueprint.set_attribute("exposure_mode", mode)
    elif mode != "histogram":
        print(f"  提示：0.9.16 的 DVS 没有 exposure_mode 属性，忽略 {mode}")
    if compensation is not None:
        if blueprint.has_attribute("exposure_compensation"):
            blueprint.set_attribute("exposure_compensation", str(compensation))
        else:
            print("  提示：DVS 没有 exposure_compensation 属性，忽略手动曝光偏置")
    return blueprint


def _stats(raw: bytes) -> tuple[int, float]:
    """返回 (事件数, 触发了事件的像素占全画面的比例)。"""
    import numpy as np

    count = len(raw) // ITEMSIZE
    if not count:
        return 0, 0.0
    array = np.frombuffer(raw, dtype=DTYPE, count=count)
    keys = array["x"].astype(np.int64) * HEIGHT + array["y"].astype(np.int64)
    return count, len(np.unique(keys)) / float(WIDTH * HEIGHT)


def measure(
    world: object,
    carla: object,
    mode: str,
    compensation: float | None,
    ticks: int,
    moving: bool = False,
) -> tuple[list[int], list[float]]:
    """按给定设置跑 ticks 帧，返回 (每帧事件数, 每帧触发的像素比例)。

    前 WARMUP_FRAMES 帧是预热（自动曝光要几帧才收敛），不计入统计。
    """
    blueprint = _blueprint(world, mode, compensation)
    sensor = world.spawn_actor(
        blueprint,
        carla.Transform(
            carla.Location(x=CAM_X, y=CAM_Y, z=CAM_Z),
            carla.Rotation(pitch=-25, yaw=CAM_YAW),
        ),
    )
    received: list[object] = []
    sensor.listen(lambda data: received.append(data))

    mover = None
    if moving:
        # 拿相机的前向向量当行进方向，免得猜 CARLA 的 yaw 约定。
        forward = carla.Rotation(yaw=CAM_YAW).get_forward_vector()
        vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        start = carla.Location(
            x=CAM_X + forward.x * 20.0, y=CAM_Y + forward.y * 20.0, z=0.5
        )
        mover = world.try_spawn_actor(vehicle_bp, carla.Transform(start))

    for index in range(ticks + WARMUP_FRAMES):
        if mover is not None:
            forward = carla.Rotation(yaw=CAM_YAW).get_forward_vector()
            distance = 20.0 + 8.0 * index * 0.05
            mover.set_transform(
                carla.Transform(
                    carla.Location(
                        x=CAM_X + forward.x * distance,
                        y=CAM_Y + forward.y * distance,
                        z=0.5,
                    )
                )
            )
        world.tick()
        time.sleep(0.02)

    if mover is not None:
        mover.destroy()
    sensor.destroy()

    counts: list[int] = []
    ratios: list[float] = []
    for data in received[WARMUP_FRAMES:]:
        count, ratio = _stats(data.raw_data)
        counts.append(count)
        ratios.append(ratio)
    return counts, ratios


def _summarise(label: str, counts: list[int], ratios: list[float]) -> None:
    if not counts:
        print(f"  {label}: 一帧数据都没收到", flush=True)
        return
    ordered = sorted(counts)
    median = ordered[len(ordered) // 2]
    tail = ratios[-10:] if ratios else [0.0]
    print(
        "  %-26s 帧数=%-3d 事件数 中位=%-9d 最大=%-9d 末=%-9d 触发像素比例=%.1f%%"
        % (
            label,
            len(counts),
            median,
            max(counts),
            counts[-1],
            100.0 * sum(tail) / len(tail),
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CARLA DVS 曝光模式与事件量探针")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--ticks", type=int, default=30)
    args = parser.parse_args()

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    settings = world.get_settings()
    original = (settings.synchronous_mode, settings.fixed_delta_seconds)
    previous_weather = world.get_weather()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    try:
        print("=== 蓝图属性 ===", flush=True)
        dump_attributes(world)

        for weather_name in WEATHERS:
            weather = getattr(carla.WeatherParameters, weather_name, None)
            if weather is None:
                print(f"\n跳过不认识的天气 {weather_name}")
                continue
            world.set_weather(weather)
            for _ in range(3):
                world.tick()
                time.sleep(0.02)
            print(f"\n=== {weather_name}：静止场景 ===", flush=True)
            for mode, compensation in SETTINGS:
                label = mode if compensation is None else f"{mode}+补偿{compensation}"
                counts, ratios = measure(
                    world, carla, mode, compensation, args.ticks
                )
                _summarise(label, counts, ratios)

            print(f"=== {weather_name}：有目标在动 ===", flush=True)
            for mode, compensation in SETTINGS:
                label = mode if compensation is None else f"{mode}+补偿{compensation}"
                counts, ratios = measure(
                    world, carla, mode, compensation, args.ticks, moving=True
                )
                _summarise(label, counts, ratios)
    finally:
        world.set_weather(previous_weather)
        settings.synchronous_mode = original[0]
        settings.fixed_delta_seconds = original[1]
        world.apply_settings(settings)

    print("\n判读：")
    print("  · 静止场景下 histogram 事件数很大 → 自动曝光每帧在改亮度，")
    print("    整幅画面同时突变，数据里大部分像素是噪声。")
    print("  · 静止场景下 manual 事件数接近 0、动目标仍有事件 → 改用手动曝光。")
    print("  · manual 仍然接近 0 → 画面太黑，需要更大的曝光偏置（看补偿那几行）。")
    print("  · 触发像素比例 >50% 的设置一定不能用：那不是事件流，是曝光抖动。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
