#!/usr/bin/env python3
"""按机位组批量采集序列（一个机位组一个序列，多次重复即多条序列）。

    python tools/plan_rig.py --town Town03 --out-dir rigs/Town03
    python tools/collect_batch.py --rig-dir rigs/Town03 --repeat 2 --output ./dvs_roadside_multicam

所有机位组在同一个 CARLA world 上串行采集：同步模式下同一 world 只能由一个
客户端驱动 tick，因此不支持并行。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 每次采集的日志目录里，逐条序列的记录列。
SEQUENCE_LOG_COLUMNS = (
    "sequence_id",
    "rig",
    "repeat",
    "attempt",
    "status",
    "junction_id",
    "camera_count",
    "frame_count",
    "total_events",
    "weather",
    "started_at_local",
    "elapsed_s",
)

# 采集历史总表（logs/index.csv）每一行代表一次批量采集。
RUN_LOG_COLUMNS = (
    "run_id",
    "started_at_local",
    "ended_at_local",
    "towns",
    "rig_dir",
    "rig_count",
    "repeat",
    "sequences",
    "succeeded",
    "failed",
    "total_events",
    "elapsed_s",
    "log_dir",
)


class _Tee:
    """把控制台输出同时写进运行日志文件。"""

    def __init__(self, stream: Any, handle: Any) -> None:
        self._stream = stream
        self._handle = handle

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._handle.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


@contextlib.contextmanager
def _tee_to(handle: Any) -> Iterator[None]:
    """在上下文内把标准输出同时写入文件。"""
    original = sys.stdout
    sys.stdout = _Tee(original, handle)
    try:
        yield
    finally:
        sys.stdout = original


def _append_run_index(
    log_root: Path,
    summary: dict[str, Any],
    run_dir: Path,
) -> Path:
    """把本次采集追加到 logs/index.csv 历史总表。"""
    log_root.mkdir(parents=True, exist_ok=True)
    index = log_root / "index.csv"
    new_file = not index.exists()
    with index.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": summary["run_id"],
                "started_at_local": summary["started_at_local"],
                "ended_at_local": summary["ended_at_local"],
                "towns": "|".join(summary["towns"]),
                "rig_dir": summary["rig_dir"],
                "rig_count": summary["rig_count"],
                "repeat": summary["repeat"],
                "sequences": summary["sequences"],
                "succeeded": summary["succeeded"],
                "failed": summary["failed"],
                "total_events": summary["total_events"],
                "elapsed_s": summary["elapsed_s"],
                "log_dir": str(run_dir),
            }
        )
    return index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量采集 CARLA 路侧固定机位组序列。")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA 服务端地址")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC 端口")
    parser.add_argument(
        "--tm-port",
        type=int,
        default=8000,
        help="Traffic Manager 端口；多实例并行时必须与目标 CARLA 实例的 TM 端口一致",
    )
    parser.add_argument(
        "--rig-dir",
        type=Path,
        help="机位组目录（其中所有 .yaml 都会采集）",
    )
    parser.add_argument("--output", type=Path, default=Path("dvs_roadside_multicam"))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="采集日志目录；每次采集建一个子目录存放 run.json/run.log/sequences.csv",
    )
    parser.add_argument("--repeat", type=int, default=1, help="每个机位组重复采集次数")
    parser.add_argument("--frames", type=int, help="每条序列帧数；不给则用机位组里的时长")
    parser.add_argument("--fps", type=float, help="覆盖机位组中的采集帧率")
    parser.add_argument("--vehicles", type=int, help="覆盖机位组中的车辆数")
    parser.add_argument("--walkers", type=int, help="覆盖机位组中的行人数")
    parser.add_argument("--weather", help="固定天气；不给则按种子随机")
    parser.add_argument(
        "--cameras",
        type=int,
        help="每个机位组只启用前 N 台相机",
    )
    parser.add_argument(
        "--rigs",
        help="只采集指定机位组（逗号分隔的文件名）；不给则采集目录内全部",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="单条序列失败后的重试次数",
    )
    parser.add_argument(
        "--sensor-timeout",
        type=float,
        help="传感器回调超时秒数；仿真慢于实时时需要放宽",
    )
    parser.add_argument(
        "--load-town",
        action="store_true",
        help="按机位组的 town 切换地图（需要 CARLA 支持热切换）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将要采集的机位组，不启动 CARLA",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="不连接 CARLA 的自检：机位组发现与 town 解析",
    )
    return parser.parse_args(argv)


def discover_rigs(rig_dir: Path, only: str | None = None) -> list[Path]:
    """按文件名排序返回机位组 YAML（目录内的其它文件忽略）。"""
    rigs = sorted(rig_dir.glob("*.yaml"))
    if only:
        wanted = {name.strip() for name in only.split(",") if name.strip()}
        rigs = [path for path in rigs if path.name in wanted]
    return rigs


def rig_meta(path: Path) -> dict[str, Any]:
    """读取机位组 YAML 的内容。"""
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def rig_town(path: Path) -> str:
    """读取机位组声明的地图名。"""
    town = rig_meta(path).get("town")
    if not isinstance(town, str) or not town.strip():
        raise ValueError(f"机位组 {path} 没有声明 town")
    return town


def rig_location(path: Path) -> dict[str, Any]:
    """读取机位组声明的采集位置（路口编号、路口中心、地理坐标）。"""
    location = rig_meta(path).get("location")
    return location if isinstance(location, dict) else {}


def ensure_town(host: str, port: int, town: str, load_town: bool) -> None:
    """确认服务端已加载目标地图；必要时切换。"""
    import carla

    client = carla.Client(host, port)
    client.set_timeout(180.0)
    current = client.get_world().get_map().name.rstrip("/\\").replace("\\", "/")
    current = current.rsplit("/", 1)[-1]
    if current.lower() == town.lower():
        return
    if not load_town:
        raise RuntimeError(
            f"机位组属于 {town}，当前 CARLA 地图是 {current}；"
            "请加载匹配地图，或加 --load-town 让本工具切换。"
        )
    client.load_world(town)


def latest_manifest(output: Path) -> dict[str, Any] | None:
    """读取输出目录里最新序列的清单，用于打印采集结果摘要。"""
    sequences = output / "sequences"
    if not sequences.is_dir():
        return None
    for path in sorted(sequences.iterdir(), key=lambda item: item.name, reverse=True):
        manifest = path / "meta" / "episode_manifest.json"
        if path.is_dir() and manifest.is_file():
            try:
                return json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    return None


def run_sequence(
    args: argparse.Namespace,
    rig_path: Path,
    attempt: int,
) -> dict[str, Any]:
    """采集一条序列，返回本次尝试的结果记录。"""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "collect_dvs.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tm-port",
        str(args.tm_port),
        "--rig",
        str(rig_path),
        "--output",
        str(args.output),
    ]
    if args.frames is not None:
        command += ["--frames", str(args.frames)]
    if args.fps is not None:
        command += ["--fps", str(args.fps)]
    if args.vehicles is not None:
        command += ["--vehicles", str(args.vehicles)]
    if args.walkers is not None:
        command += ["--walkers", str(args.walkers)]
    if args.weather is not None:
        command += ["--weather", args.weather]
    if args.cameras is not None:
        command += ["--cameras", str(args.cameras)]
    if args.sensor_timeout is not None:
        command += ["--sensor-timeout", str(args.sensor_timeout)]
    started = time.time()
    # 实时透传子进程输出：长时间采集时也能看到进度，而不是几十秒没有动静。
    # 子进程走管道时默认用系统 GBK 码页，这里强制 UTF-8 以免中文乱码。
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        print(f"    | {line}", flush=True)
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
    process.wait()
    elapsed = time.time() - started
    return {
        "rig": rig_path.name,
        "attempt": attempt,
        "returncode": process.returncode,
        "elapsed_s": round(elapsed, 1),
        "location": rig_location(rig_path),
        "log_tail": tail[-12:],
    }


def self_test() -> int:
    """不依赖 CARLA 的自检：机位组发现与 town 解析。"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rig_dir = Path(tmp)
        (rig_dir / "a_j001.yaml").write_text("town: Town03\n", encoding="utf-8")
        (rig_dir / "b_j002.yaml").write_text("town: Town04\n", encoding="utf-8")
        (rig_dir / "notes.md").write_text("忽略我", encoding="utf-8")
        found = discover_rigs(rig_dir)
        assert [path.name for path in found] == ["a_j001.yaml", "b_j002.yaml"], found
        assert rig_town(found[0]) == "Town03"
        assert rig_town(found[1]) == "Town04"

        # 缺 town 时必须报错，而不是静默采错地图。
        (rig_dir / "c_bad.yaml").write_text("cameras: {}\n", encoding="utf-8")
        try:
            rig_town(rig_dir / "c_bad.yaml")
        except ValueError:
            pass
        else:
            raise AssertionError("缺少 town 的机位组应当报错")

        # 采集位置随机位组一起读取。
        (rig_dir / "a_j001.yaml").write_text(
            "town: Town03\nlocation: {junction_id: 12, geo: {latitude: 1.0}}\n",
            encoding="utf-8",
        )
        assert rig_location(rig_dir / "a_j001.yaml")["junction_id"] == 12
        assert rig_location(rig_dir / "c_bad.yaml") == {}

        # 采集日志：控制台输出同时落盘，历史总表能追加。
        log_root = Path(tmp) / "logs"
        run_dir = log_root / "20260101-120000_Town03"
        run_dir.mkdir(parents=True)
        with (run_dir / "run.log").open("w", encoding="utf-8") as handle:
            with _tee_to(handle):
                print("采集进度 1/10 帧")
        assert "采集进度 1/10 帧" in (run_dir / "run.log").read_text(encoding="utf-8")
        summary = {
            "run_id": "20260101-120000",
            "started_at_local": "2026-01-01T12:00:00+08:00",
            "ended_at_local": "2026-01-01T12:10:00+08:00",
            "towns": ["Town03"],
            "rig_dir": str(rig_dir),
            "rig_count": 2,
            "repeat": 1,
            "sequences": 1,
            "succeeded": 1,
            "failed": 0,
            "total_events": 100,
            "elapsed_s": 600.0,
        }
        index = _append_run_index(log_root, summary, run_dir)
        _append_run_index(log_root, summary, run_dir)
        rows = list(csv.DictReader(index.open("r", encoding="utf-8", newline="")))
        assert len(rows) == 2, rows
        assert rows[0]["run_id"] == "20260101-120000"
        assert rows[0]["total_events"] == "100"
        assert len(set(SEQUENCE_LOG_COLUMNS)) == len(SEQUENCE_LOG_COLUMNS)
        assert len(set(RUN_LOG_COLUMNS)) == len(RUN_LOG_COLUMNS)
    print("collect_batch 自检通过")
    return 0


def main(argv: list[str] | None = None) -> int:
    # 控制台可能是 GBK 码页，遇到无法编码的字符用替代符而不是崩溃。
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.repeat < 1 or args.retries < 0:
        print("--repeat 必须大于 0，--retries 不能小于 0", file=sys.stderr)
        return 1
    if args.rig_dir is None:
        print("必须指定 --rig-dir", file=sys.stderr)
        return 1

    rig_dir = args.rig_dir.expanduser().resolve()
    rigs = discover_rigs(rig_dir, args.rigs)
    if not rigs:
        print(f"{rig_dir} 中没有机位组 YAML", file=sys.stderr)
        return 1
    towns = sorted({rig_town(path) for path in rigs})
    if args.dry_run:
        print(f"{rig_dir}：{len(rigs)} 个机位组，地图 {towns}，每个重复 {args.repeat} 次")
        for path in rigs:
            print(f"  {path.name}")
        return 0

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # 每次采集一个日志目录：时间 + 地图，便于按次检索。
    started_at = datetime.now().astimezone()
    run_id = started_at.strftime("%Y%m%d-%H%M%S")
    run_dir = args.log_dir.expanduser().resolve() / (
        f"{run_id}_{towns[0] if len(towns) == 1 else 'multi'}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    sequence_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    started = time.time()
    current_town = ""
    done = 0
    total_events = 0
    with (run_dir / "run.log").open("w", encoding="utf-8", newline="\n") as log_handle:
        with _tee_to(log_handle):
            sequence_file = (run_dir / "sequences.csv").open(
                "w", encoding="utf-8", newline=""
            )
            writer = csv.DictWriter(sequence_file, fieldnames=SEQUENCE_LOG_COLUMNS)
            writer.writeheader()
            sequence_file.flush()
            print(f"{rig_dir}：{len(rigs)} 个机位组，地图 {towns}，每个重复 {args.repeat} 次")
            print(f"本次采集日志目录：{run_dir}")
            for index, rig_path in enumerate(rigs, start=1):
                town = rig_town(rig_path)
                if town != current_town:
                    ensure_town(args.host, args.port, town, args.load_town)
                    current_town = town
                for repeat in range(args.repeat):
                    label = f"[{index}/{len(rigs)}] {rig_path.name}" + (
                        f" 第 {repeat + 1} 遍" if args.repeat > 1 else ""
                    )
                    print(f"{label} 开始采集 …", flush=True)
                    attempt = 0
                    while True:
                        attempt += 1
                        record = run_sequence(args, rig_path, attempt)
                        record["repeat"] = repeat + 1
                        records.append(record)
                        if record["returncode"] == 0 or attempt > args.retries:
                            break
                        print(f"  第 {attempt} 次失败，重试中 …", flush=True)
                    status = "成功" if record["returncode"] == 0 else "失败"
                    print(f"  {status}（{record['elapsed_s']} s）", flush=True)
                    if status == "失败":
                        for line in record["log_tail"]:
                            print(f"    {line}", file=sys.stderr)
                        row = {
                            "sequence_id": "",
                            "rig": rig_path.name,
                            "repeat": repeat + 1,
                            "attempt": attempt,
                            "status": status,
                            "junction_id": (rig_location(rig_path) or {}).get(
                                "junction_id"
                            ),
                            "camera_count": "",
                            "frame_count": "",
                            "total_events": "",
                            "weather": "",
                            "started_at_local": datetime.now().astimezone().isoformat(),
                            "elapsed_s": record["elapsed_s"],
                        }
                        writer.writerow(row)
                        sequence_file.flush()
                        sequence_rows.append(row)
                        continue

                    # 每条序列打印一行结果摘要：位置、时间、规模、事件量。
                    done += 1
                    manifest = latest_manifest(output) or {}
                    counts = manifest.get("event_counts") or {}
                    events = sum(counts.values())
                    total_events += events
                    location = manifest.get("location") or {}
                    junction = location.get("junction_id")
                    print(
                        f"  [OK] {manifest.get('sequence_id', '?')}"
                        f" | 路口 j{junction if junction is not None else '?'}"
                        f" | {len(counts)} 相机"
                        f" | {manifest.get('frame_count', '?')} 帧"
                        f" | {events:,} 事件"
                        f" | {manifest.get('weather', '?')}"
                        f" | 采集于 {manifest.get('capture_time', {}).get('started_at_local', '?')}"
                        f" | 用时 {record['elapsed_s']} s",
                        flush=True,
                    )
                    row = {
                        "sequence_id": manifest.get("sequence_id", ""),
                        "rig": rig_path.name,
                        "repeat": repeat + 1,
                        "attempt": attempt,
                        "status": status,
                        "junction_id": junction if junction is not None else "",
                        "camera_count": len(counts),
                        "frame_count": manifest.get("frame_count", ""),
                        "total_events": events,
                        "weather": manifest.get("weather", ""),
                        "started_at_local": manifest.get("capture_time", {}).get(
                            "started_at_local", ""
                        ),
                        "elapsed_s": record["elapsed_s"],
                    }
                    writer.writerow(row)
                    sequence_file.flush()
                    sequence_rows.append(row)
                    if done % 10 == 0:
                        elapsed = time.time() - started
                        print(
                            f"=== 进度：已完成 {done} 条，"
                            f"累计 {total_events:,} 事件，"
                            f"总用时 {elapsed / 60:.1f} 分钟，"
                            f"平均每条 {elapsed / done:.0f} s ===",
                            flush=True,
                        )
            sequence_file.close()

            succeeded = sum(1 for record in records if record["returncode"] == 0)
            ended_at = datetime.now().astimezone()
            report = {
                "rig_dir": str(rig_dir),
                "output": str(output),
                "towns": towns,
                "repeat": args.repeat,
                "total": len(records),
                "succeeded": succeeded,
                "failed": len(records) - succeeded,
                "elapsed_s": round(time.time() - started, 1),
                "runs": records,
            }
            report_path = output / "batch_report.json"
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # 运行摘要：机器可读的结构化记录 + 采集历史总表。
            run_summary = {
                "run_id": run_id,
                "started_at_utc": started_at.astimezone(timezone.utc).isoformat(),
                "started_at_local": started_at.isoformat(),
                "ended_at_local": ended_at.isoformat(),
                "elapsed_s": round((ended_at - started_at).total_seconds(), 1),
                "host": args.host,
                "port": args.port,
                "tm_port": args.tm_port,
                "rig_dir": str(rig_dir),
                "rig_count": len(rigs),
                "rigs": [path.name for path in rigs],
                "towns": towns,
                "output": str(output),
                "repeat": args.repeat,
                "cameras": args.cameras,
                "frames": args.frames,
                "fps": args.fps,
                "vehicles": args.vehicles,
                "walkers": args.walkers,
                "weather": args.weather,
                "sequences": len(records),
                "succeeded": succeeded,
                "failed": len(records) - succeeded,
                "total_events": total_events,
                "command": sys.argv,
                "sequences_detail": sequence_rows,
            }
            (run_dir / "run.json").write_text(
                json.dumps(run_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            _append_run_index(args.log_dir.expanduser().resolve(), run_summary, run_dir)
            print(
                f"批量采集结束：{succeeded}/{len(records)} 条成功，"
                f"耗时 {report['elapsed_s']} s，报告见 {report_path}"
            )
            print(f"本次采集日志：{run_dir}")
    return 0 if succeeded == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
