#!/usr/bin/env python3
"""天气均衡选择的自检（不依赖 CARLA）。

    python tools/test_weather_balance.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvs_collector.collector import pick_balanced_weather  # noqa: E402


def test_least_used_weather_wins() -> None:
    cycle = ("ClearNoon", "ClearNight", "WetSunset")
    used = {"ClearNoon": 3, "ClearNight": 1, "WetSunset": 2}
    assert pick_balanced_weather(cycle, used, random.Random(0)) == "ClearNight"


def test_unused_weather_preferred() -> None:
    cycle = ("ClearNoon", "HardRainNight")
    used = {"ClearNoon": 5}  # HardRainNight 从没采过
    assert pick_balanced_weather(cycle, used, random.Random(0)) == "HardRainNight"


def test_tie_break_is_seeded() -> None:
    cycle = ("ClearNoon", "ClearNight")
    first = pick_balanced_weather(cycle, {}, random.Random(42))
    second = pick_balanced_weather(cycle, {}, random.Random(42))
    assert first == second
    assert first in cycle


def test_counts_outside_cycle_are_ignored() -> None:
    """locations.csv 里可能有别的 rig 用过的天气，不该影响本 rig 的选择。"""
    cycle = ("ClearNoon", "ClearNight")
    used = {"SoftRainSunset": 10, "ClearNoon": 1, "ClearNight": 2}
    assert pick_balanced_weather(cycle, used, random.Random(0)) == "ClearNoon"


def main() -> int:
    tests = [
        test_least_used_weather_wins,
        test_unused_weather_preferred,
        test_tie_break_is_seeded,
        test_counts_outside_cycle_are_ignored,
    ]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print("天气均衡选择自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
