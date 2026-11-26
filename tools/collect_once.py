#!/usr/bin/env python3
"""跑一次采集，输出同时落到日志文件；失败时自动补跑地图探测。

    python tools/collect_once.py
    python tools/collect_once.py --frames 400 --vehicles 50 --walkers 30

日志默认写到 logs/collect_once.log，采集失败时再把 tools/probe_map_ops.py
的输出追加进去——这样一次运行就能拿到"卡在哪、地图接口哪些能用"的全部证据。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "logs" / "collect_once.log"


def _run_streaming(command: list[str], log) -> int:
    """跑子进程，stdout/stderr 实时转发到终端并写入日志。"""
    print(f"$ {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        # 子进程走管道时默认用系统 GBK 码页，这里强制 UTF-8 以免中文乱码。
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="跑一次采集并留日志")
    parser.add_argument("--rig", default="rigs/town03_intersection_a.yaml")
    parser.add_argument("--output", default=r"E:\Projects\DVS_data\dvs_roadside_multicam")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--vehicles", type=int, default=80)
    parser.add_argument("--walkers", type=int, default=40)
    parser.add_argument("--min-detect-px", type=int, default=16)
    parser.add_argument("--max-detect-distance", type=float, default=120.0)
    parser.add_argument("--log", default=str(LOG_PATH))
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="采集失败时不跑地图探测",
    )
    args = parser.parse_args()

    # Windows 控制台默认 GBK，这里统一成 UTF-8 以免中文输出报错。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "collect_dvs.py",
        "--rig", args.rig,
        "--output", args.output,
        "--frames", str(args.frames),
        "--port", str(args.port),
        "--load-town",
        "--vehicles", str(args.vehicles),
        "--walkers", str(args.walkers),
        "--min-detect-px", str(args.min_detect_px),
        "--max-detect-distance", str(args.max_detect_distance),
    ]

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"\n===== 采集 {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log.flush()
        code = _run_streaming(command, log)
        log.write(f"\n===== 采集退出码 {code} =====\n")
        log.flush()

        if code != 0 and not args.no_probe:
            # 采集失败：把地图接口逐项探测一遍，定位是地图能力问题还是
            # 传感器/性能问题（每个用例单独子进程，原生崩溃也拦得住）。
            log.write("\n===== 失败后自动探测 =====\n")
            log.flush()
            _run_streaming(
                [
                    sys.executable,
                    "tools/probe_map_ops.py",
                    "--port", str(args.port),
                ],
                log,
            )

    print(
        f"\n日志：{log_path}（耗时 {time.perf_counter() - started:.0f} s）",
        flush=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
