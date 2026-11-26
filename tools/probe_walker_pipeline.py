#!/usr/bin/env python3
"""完整闭环验证：生成 → 行走 → 存活（模拟采集器的行人流程）。

复现采集器的真实流程：生成 40 个行人 → 空跑 60 tick → 按位移判定
卡住销毁 → 清理离相机太近/悬空的行人 → 报告存活数与位移。

    python tools/probe_walker_pipeline.py --port 2000 --town Town10HD_Opt
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import carla  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvs_collector.traffic import (  # noqa: E402
    _crosswalk_pairs,
    _prune_stuck_walkers,
    _roadside_pairs,
    _sanitize_walkers,
    _spawn_walker,
    _spawn_point_pairs,
    update_walkers,
)

RIG_DIR = Path(__file__).resolve().parent.parent / "rigs"


def load_rigs() -> tuple[dict[int, tuple[float, float]], dict[int, list]]:
    """从机位组 YAML 读路口中心与相机位置（比硬编码可靠）。"""
    import yaml

    centers: dict[int, tuple[float, float]] = {}
    cameras: dict[int, list] = {}
    for path in sorted(RIG_DIR.rglob("*_j*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        location = data.get("location") or {}
        junction = location.get("junction_id")
        center = location.get("junction_center_m")
        if junction is None or not center:
            continue
        centers[int(junction)] = (float(center[0]), float(center[1]))
        cameras[int(junction)] = [
            tuple(float(v) for v in camera["location"])
            for camera in data["cameras"].values()
        ]
    return centers, cameras


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--walkers", type=int, default=40)
    parser.add_argument(
        "--only", type=int, default=0, help="只测某一个路口编号（0=全部）"
    )
    args = parser.parse_args()

    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    current = world.get_map().name.rstrip("/\\").rsplit("/", 1)[-1]
    if current != args.town:
        print(f"切换地图 {current} → {args.town} …", flush=True)
        world = client.load_world(args.town)
    game_map = world.get_map()

    settings = world.get_settings()
    was_sync = settings.synchronous_mode
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.025
    world.apply_settings(settings)
    world.tick()

    blueprints = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
    blueprint = blueprints[0]
    if blueprint.has_attribute("speed"):
        blueprint.set_attribute("speed", "2.0")

    total_alive = 0
    rig_centers, rig_cameras = load_rigs()
    for jid, center in sorted(rig_centers.items()):
        if args.only and jid != args.only:
            continue
        cameras = [
            {"camera_id": f"cam_{i}", "location": loc, "yaw": 0.0}
            for i, loc in enumerate(rig_cameras[jid], 1)
        ]
        rng = random.Random(jid)
        crosswalk = _crosswalk_pairs(
            world, carla, center, [30.0, 60.0], 40, cameras=cameras
        )
        roadside = _roadside_pairs(
            world, carla, rng, center, [30.0, 60.0], [15.0, 40.0], 40,
            cameras=cameras,
        )
        pools = []
        if crosswalk:
            pools.append(crosswalk)
        if roadside:
            pools.append(roadside)
        if not pools:
            print(f"junction {jid}: 无目标点")
            continue
        spawned: list = []
        plans = []
        for index in range(args.walkers):
            pool = pools[index % len(pools)]
            start, goal = pool[rng.randrange(len(pool))]
            plan = _spawn_walker(
                world, carla, blueprint, start, goal, rng, spawned
            )
            if plan is not None:
                plans.append(plan)
        # 生成后客户端要等一次 tick 才有真实位姿，否则测到的位移是
        # "真实位置到原点"的距离（几十米），毫无意义。
        world.tick()
        before = {
            p["actor"].id: p["actor"].get_location() for p in plans
        }
        alive, stuck = _prune_stuck_walkers(world, carla, plans, spawned)
        alive, unsafe = _sanitize_walkers(world, carla, alive, spawned, cameras)
        moves = [
            math.hypot(
                before[p["actor"].id].x - p["actor"].get_location().x,
                before[p["actor"].id].y - p["actor"].get_location().y,
            )
            for p in alive
        ]
        avg = sum(moves) / len(moves) if moves else 0.0
        total_alive += len(alive)
        print(
            f"junction {jid}: 生成 {len(plans)} → 存活 {len(alive)} "
            f"（卡住销毁 {stuck}，离镜头过近 {unsafe}）平均位移 {avg:.2f} m"
        )
        for p in alive:
            p["actor"].destroy()

    world.apply_settings(settings)
    print(f"\n合计存活行人：{total_alive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
