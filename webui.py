#!/usr/bin/env python3
"""本地网页控制台：点按钮完成 CARLA 采集（自动标注）、事件统计、质检与结果预览。

    python webui.py                    # 打开 http://127.0.0.1:8787
    python webui.py --port 9000 --no-browser

只用标准库，不需要额外安装依赖。仅监听 127.0.0.1；采集/统计/质检共用一个
串行任务槽——同步模式下同一个 CARLA world 只能由一个客户端驱动 tick。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "webui"
CARLA_HOME = os.environ.get("CARLA_HOME", r"D:\CARLA_0.9.16")
DEFAULT_DATASET = "dvs_roadside_multicam"

#: 检测/训练脚本用的 Python（dvs conda 环境——装了 torch/numba/pycocotools；
#: 网页控制台本身可能在别的解释器下跑）。可用 DVS_PYTHON 覆盖。
DETECT_ROOT = ROOT / "Object_Detection"
DETECT_PYTHON = os.environ.get(
    "DVS_PYTHON", r"E:\ProgramData\miniconda3\envs\dvs\python.exe"
)
if not Path(DETECT_PYTHON).is_file():
    DETECT_PYTHON = sys.executable

sys.path.insert(0, str(ROOT))
from dvs_collector.config import weather_label  # noqa: E402

SAFE_NAME = re.compile(r"^[\w.-]+$")      # 序列/相机名白名单，防路径穿越
SEQ_ID = re.compile(r"^sequence_\d{6}$")
DETACHED_PROCESS = 0x00000008


# ---------------------------------------------------------------- 任务管理
class JobManager:
    """串行子进程任务：stdout 逐行进环形缓冲，SSE 增量推给浏览器。"""

    def __init__(self, capacity: int = 4000) -> None:
        # RLock：_stream_events 持锁取增量时还要调 snapshot()，同线程可重入。
        self.lock = threading.RLock()
        self.lines: deque[dict] = deque(maxlen=capacity)
        self.seq = 0
        self.proc: subprocess.Popen | None = None
        self.action = ""
        self.command: list[str] = []
        self.started_at = 0.0
        self.status = "idle"          # idle / running / done / failed / stopped
        self.exit_code: int | None = None
        self.stop_requested = False

    def append(self, text: str, kind: str = "log") -> None:
        """追加一行日志（job 子进程输出、CARLA 尾随、控制台自身事件共用）。"""
        with self.lock:
            self.seq += 1
            self.lines.append({"seq": self.seq, "text": text, "kind": kind})

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "action": self.action,
                "command": " ".join(self.command),
                "exit_code": self.exit_code,
                "started_at": self.started_at,
                "runtime_s": round(time.time() - self.started_at, 1)
                if self.status == "running"
                else None,
            }

    def _pump(self) -> None:
        proc = self.proc
        assert proc is not None and proc.stdout is not None
        for raw in proc.stdout:
            self.append(raw.rstrip("\r\n"))
        code = proc.wait()
        with self.lock:
            self.exit_code = code
            if code == 0:
                self.status = "done"
            elif self.stop_requested:
                self.status = "stopped"
            else:
                self.status = "failed"
        self.append(
            f"—— 进程退出，exit code = {code} ——",
            "ok" if code == 0 else "error",
        )

    def start(self, action: str, command: list[str]) -> tuple[bool, str]:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return False, "已有任务在运行，请等它结束或先点「终止任务」"
            command = [str(item) for item in command]
            # 子进程输出统一按 UTF-8 解码：采集器里的中文才不会变成乱码。
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            self.proc = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            self.action, self.command = action, command
            self.started_at = time.time()
            self.status, self.exit_code = "running", None
            self.stop_requested = False
            text = "$ " + " ".join(command)
        self.append(text, "cmd")
        threading.Thread(target=self._pump, daemon=True).start()
        return True, ""

    def stop(self) -> None:
        with self.lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return
            self.stop_requested = True
        if os.name == "nt":
            # CTRL_BREAK 让子进程里的 SIGBREAK 处理器抛 KeyboardInterrupt，
            # 采集器的 finally 会销毁 actor 并恢复 world 设置；等不到再强杀。
            try:
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
            except OSError:
                proc.terminate()
                return
            threading.Thread(target=self._escalate, args=(proc,), daemon=True).start()
        else:
            proc.terminate()

    def _escalate(self, proc: subprocess.Popen) -> None:
        # 采集主循环可能正阻塞在 world.tick()（客户端超时默认 15 s），给足时间。
        for _ in range(40):
            if proc.poll() is not None:
                return
            time.sleep(0.5)
        proc.terminate()


jobs = JobManager()


class CarlaMonitor:
    """后台线程探测 CARLA 运行状态，HTTP 端点只读缓存。

    之前在请求线程里 import carla + 建 Client 轮询，boost-python 在工作线程
    首次导入容易把整个进程带崩（表现：页面永远「状态未知」）。这里改成：
    主线程启动时导入一次 carla，探测线程复用每端口一个 Client，端点读缓存
    微秒级返回，探测慢/失败也不影响页面。
    """

    #: 依次探测的常见 RPC 端口（多实例并行时命中先连上的那个）。
    PORTS = (2000, 2001, 2002)

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = {"up": False, "port": None, "map": None}
        self._carla: Any = None
        self._clients: dict[int, Any] = {}

    def start(self) -> None:
        try:  # 主线程导入一次；没装 carla 也不影响端口探测
            import carla

            self._carla = carla
        except Exception:
            jobs.append(
                "当前 Python 未安装 CARLA API：地图名检测停用，端口检测不受影响",
                "app",
            )
        threading.Thread(target=self._run, daemon=True, name="carla-monitor").start()

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def _run(self) -> None:
        while True:
            try:
                state = self._probe()
            except Exception:  # 探测线程绝不能死
                state = {"up": False, "port": None, "map": None}
            with self.lock:
                # 地图名拿不到时保留上一个，采集（同步模式）期间 RPC 常超时。
                if state["up"] and not state["map"] and self.state.get("map"):
                    state["map"] = self.state["map"]
                self.state = state
            time.sleep(2.0)

    def _probe(self) -> dict:
        for port in self.PORTS:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    pass
            except OSError:
                continue
            return {"up": True, "port": port, "map": self._probe_map(port)}
        return {"up": False, "port": None, "map": None}

    def _probe_map(self, port: int) -> str | None:
        if self._carla is None:
            return None
        try:
            client = self._clients.get(port)
            if client is None:
                client = self._carla.Client("127.0.0.1", port)
                client.set_timeout(2.0)
                self._clients[port] = client
            return str(client.get_world().get_map().name)
        except Exception:
            self._clients.pop(port, None)  # 连接失效，下次重建
            return None


carla_monitor = CarlaMonitor()


# ---------------------------------------------------------------- 检测训练链路
def detect_sequence_overview() -> dict:
    """切片目录里各序列的窗数与类别分布（划分界面的数据源）。

    统计信息从划分脚本留的 assignment.json / 逐窗 npz 聚合太慢，直接扫
    slices/<seq>/<cam>/ 目录数文件（快）；类别框数只在生成划分时统计。
    """
    slices_root = DETECT_ROOT / "data" / "slices"
    sequences = []
    if slices_root.is_dir():
        for seq_dir in sorted(slices_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            windows = sum(1 for _ in seq_dir.rglob("*.npz"))
            if windows == 0:
                continue
            sequences.append({"id": seq_dir.name, "windows": windows})
    return {"sequences": sequences, "python": DETECT_PYTHON}


def detect_split_state() -> dict:
    """当前划分：各 split 的序列分配与窗数。"""
    splits_dir = DETECT_ROOT / "data" / "splits"
    result = {"exists": False, "mode": None, "ratios": None, "splits": {}}
    assignment = splits_dir / "assignment.json"
    if assignment.is_file():
        try:
            result.update(json.loads(assignment.read_text(encoding="utf-8")))
            result["exists"] = True
        except (OSError, ValueError):
            pass
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.txt"
        result["splits"][name] = {
            "windows": sum(1 for _ in path.open(encoding="utf-8"))
            if path.is_file()
            else 0
        }
    return result


def detect_train_history(experiment: str) -> dict:
    """训练进度：读 runs/<experiment>/history.json（每 epoch 一条）。"""
    safe = re.fullmatch(r"[\w.-]+", experiment or "")
    name = experiment if safe and experiment else "light_detector_v1"
    path = DETECT_ROOT / "runs" / name / "history.json"
    if not path.is_file():
        return {"experiment": name, "epochs": []}
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        history = []
    return {"experiment": name, "epochs": history}


def detect_eval_result(split: str) -> dict:
    """最近一次评估结果（30_eval_detection.py 写入 reports/eval_<split>.json）。"""
    safe = re.fullmatch(r"\w+", split or "")
    path = DETECT_ROOT / "reports" / f"eval_{split if safe else 'test'}.json"
    if not path.is_file():
        return {"exists": False}
    try:
        return {"exists": True, **json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return {"exists": False}


# ---------------------------------------------------------------- 数据读取
_annot_cache: dict[Path, tuple[float, dict]] = {}


def _load_annotations(path: Path) -> dict:
    """带 mtime 缓存地读 annotations.json（每相机可达几 MB，滑条要反复读）。"""
    mtime = path.stat().st_mtime
    cached = _annot_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(_annot_cache) >= 8:
        _annot_cache.clear()
    _annot_cache[path] = (mtime, data)
    return data


def _dataset_root(raw: str) -> Path:
    path = Path(raw or DEFAULT_DATASET)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def list_sequences(dataset: Path) -> list[dict]:
    rows: dict[str, dict] = {}
    csv_path = dataset / "locations.csv"
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                if row.get("sequence_id"):
                    rows[row["sequence_id"]] = row
    result: list[dict] = []
    seq_root = dataset / "sequences"
    if seq_root.is_dir():
        for entry in sorted(seq_root.iterdir()):
            if entry.is_dir() and SEQ_ID.fullmatch(entry.name):
                row = rows.get(entry.name, {})
                result.append(
                    {
                        "id": entry.name,
                        "town": row.get("town") or "",
                        "junction": row.get("junction_id") or "",
                        "weather": row.get("weather") or "",
                        # 界面上显示中文，weather 字段仍保留英文预设名。
                        "weather_label": weather_label(row.get("weather") or ""),
                        "frames": int(row["frame_count"])
                        if (row.get("frame_count") or "").isdigit()
                        else None,
                        "total_events": row.get("total_events") or "",
                        "capture_local": row.get("capture_started_at_local") or "",
                    }
                )
    return result


def list_cameras(dataset: Path, sequence_id: str) -> dict:
    seq_dir = dataset / "sequences" / sequence_id
    cameras = []
    cameras_root = seq_dir / "cameras"
    if cameras_root.is_dir():
        for entry in sorted(cameras_root.iterdir()):
            if not (entry.is_dir() and SAFE_NAME.fullmatch(entry.name)):
                continue
            timestamps = entry / "timestamps.txt"
            frame_count = (
                sum(1 for _ in timestamps.open("r", encoding="utf-8"))
                if timestamps.is_file()
                else 0
            )
            suffix = ".png"
            rgb_dir = entry / "rgb"
            if rgb_dir.is_dir():
                first = next(iter(sorted(rgb_dir.iterdir())), None)
                if first is not None:
                    suffix = first.suffix or ".png"
            cameras.append(
                {"camera_id": entry.name, "frames": frame_count, "suffix": suffix}
            )
    return {
        "cameras": cameras,
        "has_snapshot": (seq_dir / "snapshot.json").is_file(),
    }


def render_event_frame(events_path: Path, frame_index: int) -> bytes:
    """把第 frame_index 帧窗口内的事件画成 JPEG：ON=红、OFF=蓝、黑底。"""
    import h5py
    import numpy as np
    from PIL import Image

    with h5py.File(events_path, "r") as handle:
        width = int(handle["meta/sensor_width"][()])
        height = int(handle["meta/sensor_height"][()])
        offsets = handle["frames/event_offsets"]
        start, end = int(offsets[frame_index]), int(offsets[frame_index + 1])
        xs = handle["events/x"][start:end]
        ys = handle["events/y"][start:end]
        ps = handle["events/p"][start:end]

    flat = ys.astype(np.int64) * width + xs
    on = np.bincount(flat[ps == 1], minlength=width * height)
    off = np.bincount(flat[ps == -1], minlength=width * height)
    peak = max(1.0, float(on.max() if on.size else 0), float(off.max() if off.size else 0))
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = (on.reshape(height, width) / peak * 255).astype(np.uint8)
    image[..., 2] = (off.reshape(height, width) / peak * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image, "RGB").save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


def list_rigs() -> list[str]:
    relative = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for pattern in ("rigs/*.yaml", "rigs/*/*.yaml")
        for path in ROOT.glob(pattern)
    ]
    # town03 示例机位组放最前面当默认值，其余按名称排。
    return sorted(
        relative,
        key=lambda name: (0 if "town03" in name else 1, name),
    )


def carla_status(port: int | None = None) -> dict:
    """读监控缓存（即时返回）；port 参数保留兼容，探测由后台线程做多端口扫描。"""
    return carla_monitor.snapshot()


def carla_start(port: int, offscreen: bool) -> tuple[bool, str]:
    exe = Path(CARLA_HOME) / "CarlaUE4.exe"
    if not exe.is_file():
        jobs.append(f"启动 CARLA 失败：找不到 {exe}", "app")
        return False, f"找不到 {exe}；请设置 CARLA_HOME 环境变量或手动启动 CARLA"
    (ROOT / "logs").mkdir(exist_ok=True)
    args = [str(exe), f"-port={port}"]
    if offscreen:
        args.append("-RenderOffScreen")
    # 打包版 CARLA 的 UE4 文件日志默认是编译掉的（实测 -ABSLOG 也不落盘），
    # 这里仍带上参数并把该路径加入尾随清单：换成会写日志的版本时日志自动进面板。
    forced_log = ROOT / "logs" / "carla_ue4.log"
    args.append(f"-ABSLOG={forced_log}")
    jobs.append("启动 CARLA：" + " ".join(args), "app")
    proc = subprocess.Popen(
        args,
        cwd=str(exe.parent),
        creationflags=DETACHED_PROCESS
        | subprocess.CREATE_NEW_PROCESS_GROUP,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    jobs.append(
        f"CARLA 进程已创建（PID {proc.pid}），等待端口 {port} 响应…", "app"
    )
    threading.Thread(
        target=_await_carla_ready, args=(proc, port), daemon=True
    ).start()
    return True, "CARLA 启动中，进度见右上角「实时日志」面板"


def _await_carla_ready(proc: subprocess.Popen, port: int, timeout_s: int = 180) -> None:
    """轮询 CARLA 端口直到就绪，把过程写进日志面板；进程退出/超时也如实记录。"""
    start = time.time()
    last_note = 0.0
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            jobs.append(
                f"CARLA 进程已退出（exit code {proc.returncode}），端口始终未响应", "error"
            )
            return
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.8):
                pass
        except OSError:
            elapsed = time.time() - start
            if elapsed - last_note >= 10:
                jobs.append(
                    f"仍在等待 CARLA 端口 {port} 响应…（{elapsed:.0f} s）", "app"
                )
                last_note = elapsed
            time.sleep(1.0)
            continue
        map_name = None
        try:
            import carla

            client = carla.Client("127.0.0.1", port)
            client.set_timeout(5.0)
            map_name = str(client.get_world().get_map().name)
        except Exception:
            pass
        jobs.append(
            f"CARLA 已就绪（用时 {time.time() - start:.0f} s）· 地图 {map_name or '未知'}",
            "ok",
        )
        return
    jobs.append(
        f"等待 CARLA 就绪超时（{timeout_s} s）；进程可能还在加载，稍后看状态灯", "error"
    )


def carla_log_candidates() -> list[Path]:
    """UE4 打包版的日志位置，候选都盯着，谁先出现跟谁。

    当前 0.9.16 打包版实测不写文件日志，这里主要给会写日志的版本兜底；
    日常「启动 CARLA」的过程日志由 _await_carla_ready 的端口探测产生。
    """
    return [
        ROOT / "logs" / "carla_ue4.log",
        Path(CARLA_HOME) / "CarlaUE4" / "Saved" / "Logs" / "CarlaUE4.log",
        Path(os.environ.get("LOCALAPPDATA", "")) / "CarlaUE4" / "Saved" / "Logs" / "CarlaUE4.log",
    ]


def tail_carla_log() -> None:
    """尾随 CARLA 的 UE4 日志文件，把新增行推进日志缓冲（kind=carla）。

    UE4 窗口程序不往 stdout 写日志，唯一的出口是 Saved/Logs/CarlaUE4.log；
    文件在 CARLA 首次运行时才出现，重启时会被轮转成 *-backup-*.log，
    所以这里轮询探测、按“新文件比已读位置小”识别轮转后重接。
    """
    position = 0
    handle = None
    watching: Path | None = None
    while True:
        try:
            if handle is None:
                found = next((p for p in carla_log_candidates() if p.is_file()), None)
                if found is None:
                    time.sleep(2.0)
                    continue
                watching = found
                handle = found.open("r", encoding="utf-8", errors="replace")
                # 从文件末尾附近接上（只回看最后一小段，历史不刷屏），
                # 先丢弃对齐前的半行。
                handle.seek(max(0, found.stat().st_size - 8192))
                handle.readline()
                position = handle.tell()
                jobs.append(f"已接上 CARLA 日志：{found}", "app")
            line = handle.readline()
            if line:
                jobs.append(line.rstrip("\r\n"), "carla")
                position = handle.tell()
                continue
            if watching is not None and watching.stat().st_size < position:
                handle.close()
                handle = None
                jobs.append("检测到 CARLA 日志轮转（CARLA 重启），重新接上…", "app")
                continue
            time.sleep(0.5)
        except OSError:
            if handle is not None:
                handle.close()
            handle = None
            time.sleep(2.0)


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "DVSWebUI/1.0"

    def log_message(self, *args: object) -> None:  # 静默访问日志
        pass

    # ---- 基础工具 ----
    def _json(self, obj: object, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query(self) -> dict[str, str]:
        parsed = parse_qs(urlparse(self.path).query)
        return {key: values[0] for key, values in parsed.items()}

    @staticmethod
    def _safe(component: str) -> str:
        if not SAFE_NAME.fullmatch(component):
            raise ValueError(f"非法路径分量：{component!r}")
        return component

    # ---- GET ----
    def do_GET(self) -> None:  # noqa: N802
        try:
            self._route_get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # 页面上直接给出可读错误
            try:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)
            except OSError:
                pass

    def _route_get(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._bytes(
                (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8"
            )
            return

        if path == "/api/status":
            self._json(jobs.snapshot())
            return

        if path == "/api/events":
            self._stream_events(int(self._query().get("since", 0) or 0))
            return

        if path == "/api/rigs":
            self._json({"rigs": list_rigs()})
            return

        if path == "/api/sequences":
            dataset = _dataset_root(self._query().get("dataset", ""))
            self._json({"sequences": list_sequences(dataset)})
            return

        match = re.fullmatch(r"/api/seq/([\w.-]+)/cameras", path)
        if match:
            dataset = _dataset_root(self._query().get("dataset", ""))
            self._json(list_cameras(dataset, self._safe(match.group(1))))
            return

        match = re.fullmatch(r"/api/seq/([\w.-]+)/([\w.-]+)/frame/(\d+)", path)
        if match:
            dataset = _dataset_root(self._query().get("dataset", ""))
            self._serve_frame(dataset, match.group(1), match.group(2), int(match.group(3)))
            return

        match = re.fullmatch(r"/api/seq/([\w.-]+)/([\w.-]+)/labels/(\d+)", path)
        if match:
            dataset = _dataset_root(self._query().get("dataset", ""))
            self._serve_labels(dataset, match.group(1), match.group(2), int(match.group(3)))
            return

        match = re.fullmatch(r"/api/seq/([\w.-]+)/([\w.-]+)/events/(\d+)", path)
        if match:
            dataset = _dataset_root(self._query().get("dataset", ""))
            self._serve_events(dataset, match.group(1), match.group(2), int(match.group(3)))
            return

        if path == "/api/carla/status":
            self._json(carla_status())
            return

        if path == "/api/detect/sequences":
            self._json(detect_sequence_overview())
            return

        if path == "/api/detect/splits":
            self._json(detect_split_state())
            return

        if path == "/api/detect/train/history":
            self._json(detect_train_history(self._query().get("experiment", "")))
            return

        if path == "/api/detect/eval/result":
            self._json(detect_eval_result(self._query().get("split", "test")))
            return

        self._json({"error": f"未知路径：{path}"}, 404)

    def _serve_frame(self, dataset: Path, seq_id: str, cam_id: str, index: int) -> None:
        rgb_dir = dataset / "sequences" / seq_id / "cameras" / cam_id / "rgb"
        for suffix in (".png", ".jpg", ".jpeg"):
            path = rgb_dir / f"{index:06d}{suffix}"
            if path.is_file():
                ctype = "image/jpeg" if suffix.startswith(".j") else "image/png"
                self._bytes(path.read_bytes(), ctype)
                return
        raise FileNotFoundError(f"没有第 {index} 帧：{rgb_dir}")

    def _serve_labels(self, dataset: Path, seq_id: str, cam_id: str, index: int) -> None:
        path = (
            dataset / "sequences" / seq_id / "cameras" / cam_id / "annotations.json"
        )
        data = _load_annotations(path)
        frames = data.get("frames", [])
        if not 0 <= index < len(frames):
            raise IndexError(f"帧号 {index} 超出范围（共 {len(frames)} 帧）")
        self._json(
            {
                "frame": index,
                "timestamp_us": frames[index].get("timestamp_us"),
                "sensor_width": data.get("sensor_width"),
                "sensor_height": data.get("sensor_height"),
                "annotations": frames[index].get("annotations", []),
                "ignored_annotations": frames[index].get("ignored_annotations", []),
            }
        )

    def _serve_events(self, dataset: Path, seq_id: str, cam_id: str, index: int) -> None:
        path = dataset / "sequences" / seq_id / "cameras" / cam_id / "events.h5"
        self._bytes(render_event_frame(path, index), "image/jpeg")

    def _stream_events(self, since: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last_seq = since
        last_status = 0.0
        while True:
            messages: list[dict] = []
            with jobs.lock:
                for item in jobs.lines:
                    if item["seq"] > last_seq:
                        messages.append(item)
                status = jobs.snapshot()
            for message in messages:
                last_seq = message["seq"]
            payload = [{"kind": "log", **message} for message in messages]
            if time.monotonic() - last_status >= 1.0:
                payload.append({"kind": "status", **status})
                last_status = time.monotonic()
            for obj in payload:
                chunk = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
                self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.25)

    # ---- POST ----
    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_body()
            path = urlparse(self.path).path
            if path == "/api/collect":
                self._start_collect(body)
            elif path == "/api/stats":
                self._start_job(
                    "stats",
                    [
                        sys.executable,
                        "tools/annotate_offline.py",
                        "stats",
                        "--dataset",
                        _dataset_root(body.get("dataset", "")),
                    ],
                )
            elif path == "/api/check":
                self._start_job(
                    "check",
                    [
                        sys.executable,
                        "tools/check_dataset.py",
                        "--dataset",
                        _dataset_root(body.get("dataset", "")),
                    ],
                )
            elif path == "/api/project":
                dataset = _dataset_root(body.get("dataset", ""))
                seq_dir = dataset / "sequences" / self._safe(str(body.get("sequence", "")))
                if not (seq_dir / "snapshot.json").is_file():
                    self._json({"error": "该序列没有 snapshot.json（旧数据），无法离线重算"}, 409)
                    return
                project_command = [
                    sys.executable,
                    "tools/annotate_offline.py",
                    "project",
                    "--sequence",
                    seq_dir,
                ]
                # 重投影要用与采集时相同的探测范围阈值，否则结果对不上。
                if body.get("min_detect_px"):
                    project_command += ["--min-detect-px", int(body["min_detect_px"])]
                if body.get("max_detect_distance"):
                    project_command += [
                        "--max-detect-distance",
                        float(body["max_detect_distance"]),
                    ]
                self._start_job("project", project_command)
            elif path == "/api/stop":
                jobs.stop()
                self._json({"ok": True})
            elif path == "/api/slice":
                self._start_slice(body)
            elif path == "/api/split":
                self._start_split(body)
            elif path == "/api/train":
                self._start_train(body)
            elif path == "/api/evaluate":
                self._start_evaluate(body)
            elif path == "/api/carla/start":
                ok, message = carla_start(
                    int(body.get("port", 2000) or 2000),
                    bool(body.get("offscreen")),
                )
                self._json({"ok": ok, "message": message}, 200 if ok else 400)
            else:
                self._json({"error": f"未知路径：{path}"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)
            except OSError:
                pass

    def _start_collect(self, body: dict) -> None:
        frames = int(body.get("frames") or 200)
        if frames <= 0:
            raise ValueError("帧数必须大于 0")
        command = [
            sys.executable,
            "collect_dvs.py",
            "--rig",
            str(body.get("rig") or list_rigs()[0]),
            "--output",
            _dataset_root(body.get("dataset", "")),
            "--frames",
            frames,
        ]
        # CARLA 在跑就自动用它探测到的端口；body.port 可显式覆盖。
        port = int(body.get("port") or 0) or carla_monitor.snapshot().get("port") or 2000
        command += ["--port", str(port)]
        if body.get("load_town", True):
            command.append("--load-town")
        if body.get("vehicles") is not None:
            command += ["--vehicles", int(body["vehicles"])]
        if body.get("walkers") is not None:
            command += ["--walkers", int(body["walkers"])]
        if body.get("weather"):
            command += ["--weather", str(body["weather"])]
        if body.get("cameras"):
            command += ["--cameras", int(body["cameras"])]
        # 有效探测范围：页面可覆盖，留空则用采集器默认值。
        if body.get("min_detect_px"):
            command += ["--min-detect-px", int(body["min_detect_px"])]
        if body.get("max_detect_distance"):
            command += ["--max-detect-distance", float(body["max_detect_distance"])]
        self._start_job("collect", command)

    def _start_job(self, action: str, command: list[str]) -> None:
        ok, message = jobs.start(action, command)
        self._json({"ok": ok, "error": None if ok else message}, 200 if ok else 409)

    # ---- 检测链路：切片 / 划分 / 训练 / 评估（都走 dvs 环境的子进程）----
    def _start_slice(self, body: dict) -> None:
        command = [
            DETECT_PYTHON,
            str(DETECT_ROOT / "scripts" / "01_slice_events.py"),
            "--config",
            str(DETECT_ROOT / "configs" / "data" / "slice_carla.yaml"),
        ]
        if body.get("overwrite"):
            command.append("--overwrite")
        self._start_job("slice", command)

    def _start_split(self, body: dict) -> None:
        command = [
            DETECT_PYTHON,
            str(DETECT_ROOT / "scripts" / "02_make_splits.py"),
        ]
        ratios = body.get("ratios")
        sequences = body.get("sequences")
        if ratios:
            parts = [float(v) for v in str(ratios).split(",")]
            if len(parts) != 3 or any(v < 0 for v in parts) or sum(parts) <= 0:
                raise ValueError(f"划分比例格式不对：{ratios}（应为 70,15,15）")
            command += ["--ratios", ",".join(str(v) for v in parts)]
            command += [
                "--sequences",
                ",".join(sequences) if isinstance(sequences, list) and sequences else "all",
            ]
            if body.get("seed") is not None:
                command += ["--seed", int(body["seed"])]
        else:
            if body.get("val_sequence"):
                command += ["--val-sequence", str(body["val_sequence"])]
            if body.get("test_sequence"):
                command += ["--test-sequence", str(body["test_sequence"])]
        if body.get("keep_zero_pedestrian"):
            command.append("--keep-zero-pedestrian")
        self._start_job("split", command)

    def _start_train(self, body: dict) -> None:
        command = [
            DETECT_PYTHON,
            str(DETECT_ROOT / "scripts" / "10_train.py"),
            "--config",
            str(DETECT_ROOT / "configs" / "train" / "light_detector.yaml"),
        ]
        if body.get("epochs"):
            command += ["--epochs", int(body["epochs"])]
        if body.get("batch_size"):
            command += ["--batch-size", int(body["batch_size"])]
        if body.get("lr"):
            command += ["--lr", float(body["lr"])]
        if body.get("experiment"):
            command += ["--experiment", str(body["experiment"])]
        if body.get("resume"):
            resume = Path(str(body["resume"]))
            if not resume.is_absolute():
                resume = DETECT_ROOT / resume  # 子进程 cwd 是仓库根，相对路径要拼上
            command += ["--resume", str(resume)]
        self._start_job("train", command)

    def _start_evaluate(self, body: dict) -> None:
        experiment = str(body.get("experiment") or "light_detector_v1")
        ckpt = DETECT_ROOT / "runs" / experiment / (
            "best.pth" if body.get("use_best", True) else "last.pth"
        )
        if not ckpt.is_file():
            self._json(
                {"ok": False, "error": f"找不到权重：{ckpt}（先完成训练）"}, 400
            )
            return
        command = [
            DETECT_PYTHON,
            str(DETECT_ROOT / "scripts" / "30_eval_detection.py"),
            "--ckpt",
            str(ckpt),
            "--config",
            str(DETECT_ROOT / "configs" / "train" / "light_detector.yaml"),
            "--split",
            str(body.get("split") or "test"),
        ]
        self._start_job("evaluate", command)


def main() -> int:
    parser = argparse.ArgumentParser(description="DVS 采集/标注 本地网页控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    url = f"http://{args.host}:{args.port}"
    print(f"DVS 控制台已启动：{url}")
    print(f"CARLA 安装目录：{CARLA_HOME}（可用 CARLA_HOME 环境变量覆盖）")
    jobs.append(f"控制台已启动：{url}", "app")
    carla_monitor.start()
    threading.Thread(target=tail_carla_log, daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n控制台已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
