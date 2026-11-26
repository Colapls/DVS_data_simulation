#!/usr/bin/env python3
"""交互式采集向导：按提示输入参数后自动完成布点、批量采集和质检。

    python tools/auto_collect.py

Windows 下双击项目根目录的 `一键采集.bat` 即可运行本向导：脚本会按需自动
启动 CARLA 仿真平台（默认安装位置 D:\\CARLA_0.9.16），采集完成后给出
位置时间索引摘要。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TOOLS = PROJECT_ROOT / "tools"
CARLA_HOME = Path(r"D:\CARLA_0.9.16")
WEATHERS = (
    "ClearNoon",
    "CloudyNoon",
    "SoftRainSunset",
    "WetSunset",
    "ClearNight",
    "HardRainNight",
)


# --------------------------------------------------------------------------
# 交互输入。
# --------------------------------------------------------------------------


def ask(prompt: str, default: str = "") -> str:
    """读取一行输入，回车使用默认值。"""
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        raise SystemExit("\n输入已结束，采集取消。")
    return value or default


def ask_int(prompt: str, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    """读取整数输入并检查范围。"""
    while True:
        raw = ask(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("  请输入整数。")
            continue
        if value < minimum or (maximum is not None and value > maximum):
            print(f"  取值范围是 {minimum} 到 {maximum}。")
            continue
        return value


def ask_yes(prompt: str, default: bool = True) -> bool:
    """读取是/否输入。"""
    hint = "Y/n" if default else "y/N"
    while True:
        raw = ask(f"{prompt} ({hint})", "y" if default else "n").lower()
        if raw.startswith("y"):
            return True
        if raw.startswith("n"):
            return False
        print("  请输入 y 或 n。")


# --------------------------------------------------------------------------
# CARLA 连接。
# --------------------------------------------------------------------------


def carla_alive(host: str, port: int, timeout: float = 5.0) -> bool:
    """检查 CARLA 服务端是否可连接。"""
    try:
        import carla
    except ImportError:
        return False
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        client.get_world()
        return True
    except Exception:
        return False


def current_town(host: str, port: int) -> str:
    """读取当前已加载的地图名。"""
    import carla

    client = carla.Client(host, port)
    client.set_timeout(15.0)
    name = client.get_world().get_map().name
    return name.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]


def find_carla_exe() -> Path | None:
    """在常见安装位置和 CARLA_HOME 环境变量中查找 CARLA 可执行文件。"""
    candidates = []
    home = os.environ.get("CARLA_HOME")
    if home:
        candidates.append(Path(home) / "CarlaUE4.exe")
    candidates += [
        CARLA_HOME / "CarlaUE4.exe",
        Path(r"C:\CARLA_0.9.16\CarlaUE4.exe"),
        Path(r"C:\Program Files\CARLA_0.9.16\CarlaUE4.exe"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def wait_for_carla(host: str, port: int, seconds: int) -> bool:
    """等待 CARLA 端口就绪（启动脚本可能正在加载地图）。"""
    for _ in range(max(1, seconds // 2)):
        if carla_alive(host, port, timeout=3.0):
            return True
        time.sleep(2.0)
    return False


def start_carla(exe: Path, port: int, offscreen: bool) -> bool:
    """启动 CARLA 仿真平台并等待端口就绪。"""
    command = [str(exe), f"-port={port}"]
    if offscreen:
        command.append("-RenderOffScreen")
    print(f"  启动命令：{' '.join(command)}")
    print("  首次启动需要加载地图，通常要 30–60 秒，请耐心等待 …")
    subprocess.Popen(command, cwd=str(exe.parent))
    for attempt in range(90):
        time.sleep(2.0)
        if carla_alive("127.0.0.1", port, timeout=3.0):
            print(f"  CARLA 已就绪（等待 {attempt * 2 + 2} 秒）。")
            return True
    return False


# --------------------------------------------------------------------------
# 机位组选择。
# --------------------------------------------------------------------------


def list_rigs(rig_dir: Path) -> list[Path]:
    """列出机位组 YAML。"""
    return sorted(rig_dir.glob("*.yaml")) if rig_dir.is_dir() else []


def camera_count(rig_path: Path) -> int:
    """机位组里配置的相机数量。"""
    import yaml

    payload = yaml.safe_load(rig_path.read_text(encoding="utf-8")) or {}
    cameras = payload.get("cameras") or {}
    return len(cameras) if isinstance(cameras, dict) else 0


def choose_rigs(rigs: list[Path]) -> list[Path]:
    """让用户选择要采集的机位组。"""
    if len(rigs) == 1:
        print(f"  只有 1 个机位组：{rigs[0].name}")
        return rigs
    print("  可用机位组：")
    for index, path in enumerate(rigs, start=1):
        print(f"    [{index}] {path.name}")
    print("    [A] 全部")
    raw = ask("选择机位组（序号，逗号分隔，A=全部）", "A")
    if raw.strip().upper() == "A":
        return rigs
    picked: list[Path] = []
    for token in raw.replace("，", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit() or not 1 <= int(token) <= len(rigs):
            print(f"  忽略无效序号：{token}")
            continue
        picked.append(rigs[int(token) - 1])
    return picked or rigs


# --------------------------------------------------------------------------
# 步骤执行。
# --------------------------------------------------------------------------


def run(command: list[str], title: str) -> int:
    """执行一个子步骤并返回退出码。"""
    print(f"\n>>> {title}")
    print("    " + " ".join(str(part) for part in command[1:]))
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
    return completed.returncode


def latest_run_dir(log_dir: Path) -> Path | None:
    """找最近一次采集的日志目录（目录名按时间排序）。"""
    if not log_dir.is_dir():
        return None
    candidates = [path for path in log_dir.iterdir() if path.is_dir()]
    return max(candidates, key=lambda path: path.name) if candidates else None


def summarize(output: Path) -> None:
    """打印本次采集的位置时间索引摘要。"""
    path = output / "locations.csv"
    if not path.is_file():
        print("\n没有生成 locations.csv。")
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("\nlocations.csv 为空。")
        return
    print(f"\n共 {len(rows)} 条序列（位置与时间索引见 {path}）：")
    for row in rows:
        print(
            f"  {row['sequence_id']}  路口 j{row['junction_id']}  "
            f"机位 {row['camera_count']}  事件 {row['total_events']}  "
            f"{row['weather']}  {row['capture_started_at_local']}"
        )


# --------------------------------------------------------------------------
# 向导主体。
# --------------------------------------------------------------------------


def self_test() -> int:
    """不连接 CARLA 的自检：输入解析、机位组选择和相机计数。"""
    import tempfile

    original_ask = ask
    try:
        # 空输入走默认值，非数字给出提示后重试。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.yaml").write_text(
                "town: Town03\ncameras: {cam_01: {}, cam_02: {}}\n", encoding="utf-8"
            )
            (root / "b.yaml").write_text(
                "town: Town03\ncameras: {cam_01: {}}\n", encoding="utf-8"
            )
            assert camera_count(root / "a.yaml") == 2
            assert camera_count(root / "b.yaml") == 1

            answers = iter(["1", "1,2", "A"])
            globals()["ask"] = lambda prompt, default="": next(answers)
            rigs = list_rigs(root)
            with contextlib.redirect_stdout(io.StringIO()):
                assert [path.name for path in choose_rigs(rigs)] == ["a.yaml"], rigs
                assert [path.name for path in choose_rigs(rigs)] == ["a.yaml", "b.yaml"]
                assert [path.name for path in choose_rigs(rigs)] == ["a.yaml", "b.yaml"]

                # 非法序号被忽略；全空则回退到全部。
                globals()["ask"] = lambda prompt, default="": "9"
                assert [path.name for path in choose_rigs(rigs)] == ["a.yaml", "b.yaml"]
    finally:
        globals()["ask"] = original_ask

    # input() 在 EOF 时应给出明确退出，而不是抛栈。
    import builtins

    original_input = builtins.input
    try:
        builtins.input = lambda prompt="": (_ for _ in ()).throw(EOFError())
        try:
            ask("测试")
        except SystemExit as exc:
            assert "取消" in str(exc), exc
        else:
            raise AssertionError("EOF 时应当退出")
    finally:
        builtins.input = original_input
    print("采集向导自检通过")
    return 0


def main() -> int:
    # 行缓冲：向导提示和子进程输出按顺序显示，不会被缓冲打乱。
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="交互式路侧 DVS+RGB 批量采集向导。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="采集日志目录；每次采集建一个子目录存放 run.json/run.log/sequences.csv",
    )
    parser.add_argument("--no-start", action="store_true", help="不自动启动 CARLA")
    parser.add_argument("--offscreen", action="store_true", help="用无窗口模式启动 CARLA")
    parser.add_argument("--self-test", action="store_true", help="只跑向导逻辑自检")
    options = parser.parse_args()
    if options.self_test:
        return self_test()

    print("=" * 62)
    print("  CARLA 路侧固定多机位 DVS+RGB 批量采集向导")
    print("=" * 62)

    print("\n[1/6] CARLA 服务端")
    host = ask("主机", options.host)
    port = ask_int("端口", options.port, minimum=1, maximum=65535)
    tm_port = ask_int(
        "Traffic Manager 端口（多实例时需与目标实例一致）", options.tm_port, minimum=1
    )
    started_carla = False
    if not carla_alive(host, port):
        print(f"  端口 {port} 暂无响应，等待 CARLA 就绪 …")
        if wait_for_carla(host, port, 60):
            print("  CARLA 已就绪。")
        else:
            if options.no_start:
                print("  已指定 --no-start，请手动启动 CARLA 后重试。")
                return 1
            exe = find_carla_exe()
            if exe is None:
                print(
                    f"  找不到 CarlaUE4.exe（尝试过 {CARLA_HOME} 和 CARLA_HOME 环境变量），"
                    "请手动启动 CARLA 后重试。"
                )
                return 1
            offscreen = options.offscreen or ask_yes(
                "使用无窗口模式启动？（无窗口模式下切换地图可能卡住）", default=False
            )
            if not start_carla(exe, port, offscreen):
                print(
                    f"  CARLA 启动超时。端口 {port} 可能已被挂死的进程占用，"
                    "请结束 CarlaUE4.exe 后重试，或换一个端口。"
                )
                return 1
            started_carla = True
    else:
        print("  CARLA 已在运行。")
    town = current_town(host, port)
    print(f"  当前地图：{town}")

    print("\n[2/6] 机位组")
    replan = ask_yes(f"是否按 {town} 重新自动布点？（会重写 rigs/{town}）", default=False)
    target_town = ask("目标地图（直接回车 = 用当前地图）", town)
    rig_dir = Path("rigs") / target_town
    if replan:
        command = [
            sys.executable,
            str(TOOLS / "plan_rig.py"),
            "--host", host,
            "--port", str(port),
            "--out-dir", str(rig_dir),
        ]
        if target_town != town:
            command += ["--town", target_town]
        if run(command, "自动布点") != 0:
            print("布点失败，已中止。")
            return 1
    rigs = list_rigs(rig_dir)
    if not rigs:
        print(f"{rig_dir} 中没有机位组，请先运行 plan_rig.py。")
        return 1
    rigs = choose_rigs(rigs)

    print("\n[3/6] 相机与采集量")
    max_cameras = max(camera_count(path) for path in rigs)
    camera_choice = ask(
        f"每个机位组启用的相机个数（1-{max_cameras}，留空 = 全部）", ""
    )
    cameras = None
    if camera_choice:
        try:
            cameras = int(camera_choice)
        except ValueError:
            print("  输入无效，按全部机位处理。")
        else:
            if not 1 <= cameras <= max_cameras:
                print(f"  超出范围 1-{max_cameras}，按全部机位处理。")
                cameras = None
    duration = ask_int("每个相机采集时长（秒）", 90, minimum=1)
    fps = ask_int("采集帧率", 40, minimum=1, maximum=60)

    print("\n[4/6] 交通与天气")
    vehicles = ask_int("车辆数", 80, minimum=0)
    walkers = ask_int("行人数", 40, minimum=0)
    print("  可选天气：" + "、".join(WEATHERS))
    weather = ask("固定天气（留空 = 按种子随机）", "")
    if weather and weather not in WEATHERS:
        print(f"  未知天气 {weather}，改为随机。")
        weather = ""

    print("\n[5/6] 输出与遍数")
    output = Path(ask("输出目录", "dvs_roadside_multicam"))
    log_dir = Path(ask("采集日志目录", "logs"))
    repeat = ask_int("每个机位组重复遍数", 1, minimum=1)

    frames = max(1, round(duration * fps))
    print("\n[6/6] 参数确认")
    print(f"  地图        : {target_town}")
    print(f"  机位组      : {len(rigs)} 个（{'、'.join(p.name for p in rigs)}）")
    print(f"  相机个数    : {cameras if cameras else '全部'}")
    print(f"  采集量      : {duration} 秒 = {frames} 帧 @ {fps} Hz")
    print(f"  交通        : {vehicles} 车 / {walkers} 人")
    print(f"  天气        : {weather or '按种子随机'}")
    print(f"  重复遍数    : {repeat}")
    print(f"  输出目录    : {output.resolve()}")
    print(f"  日志目录    : {log_dir.resolve()}")
    if not ask_yes("确认开始采集？", default=True):
        print("已取消。")
        return 0

    command = [
        sys.executable,
        str(TOOLS / "collect_batch.py"),
        "--host", host,
        "--port", str(port),
        "--tm-port", str(tm_port),
        "--rig-dir", str(rig_dir),
        "--rigs", ",".join(path.name for path in rigs),
        "--output", str(output),
        "--log-dir", str(log_dir),
        "--repeat", str(repeat),
        "--frames", str(frames),
        "--fps", str(fps),
        "--vehicles", str(vehicles),
        "--walkers", str(walkers),
    ]
    if cameras:
        command += ["--cameras", str(cameras)]
    if weather:
        command += ["--weather", weather]
    code = run(command, "批量采集")

    print("\n>>> 数据集质检")
    subprocess.run(
        [sys.executable, str(TOOLS / "check_dataset.py"), "--dataset", str(output)],
        cwd=str(PROJECT_ROOT),
    )
    summarize(output.resolve())
    run_dir = latest_run_dir(log_dir.resolve())
    if run_dir is not None:
        print(f"\n本次采集日志：{run_dir}")
        print("  run.json      运行摘要（时间、机位、数量、事件量）")
        print("  run.log       完整控制台输出")
        print("  sequences.csv 每条序列一行")
        print(f"  历史总表      {log_dir.resolve() / 'index.csv'}")
    print("\n完成。数据目录：" + str(output.resolve()))
    if started_carla and ask_yes("关闭本次启动的 CARLA 仿真平台？", default=False):
        print("  请手动关闭 CARLA 窗口（或结束 CarlaUE4.exe 进程）。")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
