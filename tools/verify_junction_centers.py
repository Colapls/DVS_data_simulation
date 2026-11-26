#!/usr/bin/env python3
"""实测 CARLA 报告的 junction 中心，验证 rig 的 Y 翻转修复是否正确。

只依赖 CARLA Python API（不需要 h5py/yaml），可以在只装了 carla 的
conda 环境里跑： CARLA 运行时报告的路口坐标是权威事实来源，.xodr
只是它的静态副本。

    python tools/verify_junction_centers.py --port 2000
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import carla  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RIG_DIR = Path(__file__).resolve().parent.parent / "rigs" / "Town10HD_Opt"


def junction_center(junction) -> tuple[float, float]:
    points = []
    for in_wp, out_wp in junction.get_waypoints(carla.LaneType.Driving):
        for loc in (in_wp.transform.location, out_wp.transform.location):
            points.append((loc.x, loc.y))
    if not points:
        return (float("nan"), float("nan"))
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    args = parser.parse_args()

    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    current = world.get_map().name.rstrip("/\\").rsplit("/", 1)[-1]
    if current != args.town:
        print(f"切换地图到 {args.town}（当前 {current}）…", flush=True)
        world = client.load_world(args.town)
    game_map = world.get_map()

    # rig 里的中心（修复后）。手写正则解析，避免依赖 PyYAML——
    # 只装了 carla 的环境跑不了 yaml。
    import re

    rigs = {}
    for path in sorted(RIG_DIR.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        junction = re.search(r"junction_id:\s*(\d+)", text)
        center = re.search(
            r"junction_center_m:\s*\n\s*-\s*(-?[\d.]+)\s*\n\s*-\s*(-?[\d.]+)",
            text,
        )
        if junction and center:
            rigs[int(junction.group(1))] = (
                path.name,
                (float(center.group(1)), float(center.group(2))),
            )

    failures = 0
    print(f"\n{'junction':>9} {'CARLA 真实中心':>24} {'rig 中心':>24} {'偏距':>8}")
    for junction_id, (name, rig_center) in sorted(rigs.items()):
        found = None
        for wp in game_map.generate_waypoints(2.0):
            if wp.is_junction and int(wp.junction_id) == junction_id:
                found = wp.get_junction()
                break
        if found is None:
            print(f"{junction_id:>9}  CARLA 里找不到该 junction")
            failures += 1
            continue
        cx, cy = junction_center(found)
        dist = math.hypot(rig_center[0] - cx, rig_center[1] - cy)
        mark = "✓" if dist <= 30.0 else "✗"
        failures += 0 if dist <= 30.0 else 1
        print(
            f"{junction_id:>9} ({cx:9.2f},{cy:9.2f}) "
            f"({rig_center[0]:9.2f},{rig_center[1]:9.2f}) {dist:7.1f} m {mark}"
        )

    # 目标路口周边到底有没有行人可走的资源
    print("\n=== 每个 junction 的行人资源（60 m 内）===")
    for junction_id, (name, rig_center) in sorted(rigs.items()):
        cx, cy = rig_center
        crosswalks = [
            wp for wp in game_map.generate_waypoints(2.0)
            if wp.is_junction and int(wp.junction_id) == junction_id
        ]
        sidewalk = [
            wp for wp in game_map.generate_waypoints(2.0)
            if wp.lane_type == carla.LaneType.Sidewalk
            and math.hypot(wp.transform.location.x - cx,
                           wp.transform.location.y - cy) <= 60.0
        ]
        # 人行横道多边形：取该路口中心附近的横道顶点
        points = game_map.get_crosswalks()
        near_cw = []
        for p in points:
            if math.hypot(p.x - cx, p.y - cy) <= 60.0:
                near_cw.append(p)
        print(
            f"  junction {junction_id:>3}: 60 m 内人行道 waypoint {len(sidewalk):>3} 个，"
            f"横道顶点 {len(near_cw):>3} 个（{len(near_cw)//5} 条横道）"
        )
        if not sidewalk and len(near_cw) < 5:
            failures += 1
            print(f"    ✗ 该路口周边没有行人资源")
    print()
    print("结论：" + ("rig 中心与 CARLA 一致 ✓" if failures == 0
                     else f"{failures} 处不一致 ✗"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
