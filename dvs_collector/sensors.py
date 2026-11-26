from __future__ import annotations

import math
import queue
import time
from dataclasses import dataclass, field
from typing import Any

from .config import CameraConfig
from .dataset import SequencePaths
from .storage import CameraWriter


@dataclass
class CameraRig:
    """运行期间管理传感器 actor、回调队列、HDF5 写入器和标定值。"""

    camera_names: list[str]
    sensors: list[Any] = field(default_factory=list)
    dvs_sensors: dict[str, Any] = field(default_factory=dict)
    segmentation_sensors: dict[str, Any] = field(default_factory=dict)
    segmentation_scales: dict[str, tuple[float, float]] = field(default_factory=dict)
    writers: dict[str, CameraWriter] = field(default_factory=dict)
    queues: dict[str, queue.Queue[Any]] = field(default_factory=dict)
    future_data: dict[str, dict[int, Any]] = field(default_factory=dict)
    calibration: dict[str, dict[str, Any]] = field(default_factory=dict)
    applied_exposure_modes: dict[str, str | None] = field(default_factory=dict)


class SensorFrameReader:
    """按 CARLA frame 号取传感器数据；晚到的帧先缓存，晚于目标的帧留给下一轮。"""

    def __init__(self, rig: CameraRig, timeout: float, dvs_grace_s: float = 0.1) -> None:
        self.rig = rig
        self.timeout = timeout
        # DVS 等这一帧的宽限期：太短会丢掉慢一拍的数据，太长则每帧都要
        # 白等（DVS 静止时每一帧都要等满）。
        self.dvs_grace_s = dvs_grace_s
        self.last_received_frame: dict[str, int | None] = {
            name: None for name in rig.queues
        }

    def _state(self) -> str:
        """各传感器最近收到的帧号和队列长度，超时时一并打出来定位。"""
        return "，".join(
            f"{name}={self.last_received_frame[name]}"
            f"(队列{self.rig.queues[name].qsize()})"
            for name in sorted(self.rig.queues)
        )

    def get_dvs(self, queue_name: str, target_frame: int) -> bytes:
        """取这一帧的 DVS 事件数据（原始字节流，可能为空）。

        DVS 不能像 RGB 那样死等某个 frame 号：CARLA 的 DVS 传感器只在
        "这一帧产生了事件"时才发数据，画面完全静止时它一帧都不发。
        实测 cam_02 在路口安静下来后停在 frame=1934，后面每一帧都等不到，
        采集直接以"等待 cam_02_dvs 的 CARLA frame=1935 超时"退出。

        这里的做法：先按 frame 号等一小会儿（对齐最准，慢一拍的数据也能
        等到）；等不到就把队列里现有的都取走；还没有就写空帧——空帧是
        合法数据，事件本身的时间戳在 raw_data 里，不依赖帧号。

        跑到目标帧前面去的（DVS 比 RGB 快）要存起来留给下一轮，
        不能顺手并进当前帧：那样这一帧的事件数会算到下一帧头上。
        """
        sensor_queue = self.rig.queues[queue_name]
        # 上一轮跑到前面去的数据：帧号正好对上，直接用。
        cached = self.rig.future_data[queue_name].pop(target_frame, None)
        if cached is not None:
            return cached.raw_data
        chunks: list[bytes] = []
        deadline = time.monotonic() + self.dvs_grace_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                data = sensor_queue.get(timeout=remaining)
            except queue.Empty:
                break
            self.last_received_frame[queue_name] = data.frame
            if data.frame > target_frame:
                # 跑到前面去了：留给下一轮，别把下一帧的事件算到这一帧头上。
                self.rig.future_data[queue_name][data.frame] = data
                break
            chunks.append(data.raw_data)
            if data.frame == target_frame:
                break
        return b"".join(chunks)

    def get(self, queue_name: str, target_frame: int) -> Any:
        cached = self.rig.future_data[queue_name].pop(target_frame, None)
        if cached is not None:
            return cached
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"等待 {queue_name} 的 CARLA frame={target_frame} 超时，"
                    f"最近收到 frame={self.last_received_frame[queue_name]}；"
                    f"各传感器状态：{self._state()}；"
                    "请确认没有其他客户端调用 world.tick()。"
                )
            try:
                data = self.rig.queues[queue_name].get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(
                    f"等待 {queue_name} 的 CARLA frame={target_frame} 超时，"
                    f"最近收到 frame={self.last_received_frame[queue_name]}；"
                    f"各传感器状态：{self._state()}"
                ) from exc
            self.last_received_frame[queue_name] = data.frame
            if data.frame == target_frame:
                return data
            if data.frame > target_frame:
                self.rig.future_data[queue_name][data.frame] = data


def configure_dvs_blueprint(
    blueprint: Any,
    width: int,
    height: int,
    fov: float,
    positive_threshold: float,
    negative_threshold: float,
    refractory_ms: float,
) -> None:
    attributes = {
        "image_size_x": str(width),
        "image_size_y": str(height),
        "fov": str(fov),
        "positive_threshold": str(positive_threshold),
        "negative_threshold": str(negative_threshold),
        "refractory_period_ns": str(round(refractory_ms * 1_000_000)),
    }
    for name, value in attributes.items():
        if not blueprint.has_attribute(name):
            raise RuntimeError(f"CARLA DVS blueprint 缺少必需属性：{name}")
        blueprint.set_attribute(name, value)


def _sensor_blueprint(
    world: Any,
    camera: CameraConfig,
    kind: str,
) -> Any:
    """按相机配置建立 RGB/DVS 蓝图，并返回实际启用的曝光模式。"""
    blueprint = world.get_blueprint_library().find(f"sensor.camera.{kind}")
    if kind == "dvs":
        configure_dvs_blueprint(
            blueprint,
            camera.width,
            camera.height,
            camera.fov,
            camera.positive_threshold,
            camera.negative_threshold,
            camera.refractory_ms,
        )
        exposure_mode = camera.exposure_mode
        if exposure_mode and blueprint.has_attribute("exposure_mode"):
            blueprint.set_attribute("exposure_mode", exposure_mode)
        else:
            exposure_mode = None
    else:
        exposure_mode = None
        for name, value in (
            ("image_size_x", camera.width),
            ("image_size_y", camera.height),
            ("fov", camera.fov),
        ):
            if not blueprint.has_attribute(name):
                raise RuntimeError(f"CARLA RGB blueprint 缺少必需属性：{name}")
            blueprint.set_attribute(name, str(value))
    return blueprint, exposure_mode


def _segmentation_blueprint(
    world: Any,
    camera: CameraConfig,
    scale: int,
) -> Any:
    """实例分割蓝图：与 DVS 同位同 FOV，分辨率可低采样。

    可见度只需要知道"目标在画面里占了多少像素"，640×360 足够；
    分割图本身不落盘，只在采集时取每个 actor 的可见像素统计。
    """
    blueprint = world.get_blueprint_library().find(
        "sensor.camera.instance_segmentation"
    )
    width = max(1, camera.width // max(1, scale))
    height = max(1, camera.height // max(1, scale))
    for name, value in (
        ("image_size_x", width),
        ("image_size_y", height),
        ("fov", camera.fov),
    ):
        if not blueprint.has_attribute(name):
            raise RuntimeError(f"CARLA 实例分割蓝图缺少必需属性：{name}")
        blueprint.set_attribute(name, str(value))
    return blueprint, width, height


def spawn_static_camera_rig(
    world: Any,
    carla: Any,
    h5py: Any,
    np: Any,
    cameras: tuple[CameraConfig, ...],
    sequence: SequencePaths,
    time_origin_us: int,
    integration_time_us: int,
    flush_frames: int,
    town: str,
    weather: str,
    instance_seg: bool = True,
    instance_seg_scale: int = 2,
) -> CameraRig:
    """在世界坐标直接生成未附着 actor 的固定 DVS/RGB 相机对。"""
    rig = CameraRig(camera_names=[camera.camera_id for camera in cameras])
    for camera in cameras:
        transform = carla.Transform(
            carla.Location(
                x=camera.location[0],
                y=camera.location[1],
                z=camera.location[2],
            ),
            carla.Rotation(
                pitch=camera.pitch,
                yaw=camera.yaw,
                roll=camera.roll,
            ),
        )
        dvs_blueprint, exposure_mode = _sensor_blueprint(
            world, camera, "dvs"
        )
        # 不传 attach_to 父 actor，传感器因此固定在 CARLA 世界坐标中。
        dvs_sensor = world.spawn_actor(dvs_blueprint, transform)
        rig.sensors.append(dvs_sensor)
        rig.dvs_sensors[camera.camera_id] = dvs_sensor
        rig.applied_exposure_modes[camera.camera_id] = exposure_mode

        rgb_blueprint, _ = _sensor_blueprint(world, camera, "rgb")
        rgb_sensor = world.spawn_actor(rgb_blueprint, transform)
        rig.sensors.append(rgb_sensor)

        for sensor, kind in ((dvs_sensor, "dvs"), (rgb_sensor, "rgb")):
            queue_name = f"{camera.camera_id}_{kind}"
            sensor_queue: queue.Queue[Any] = queue.Queue()
            rig.queues[queue_name] = sensor_queue
            rig.future_data[queue_name] = {}
            # 回调只入队，不做磁盘 I/O，避免阻塞 CARLA 的传感器回调线程。
            sensor.listen(lambda data, q=sensor_queue: q.put(data))

        if instance_seg:
            seg_blueprint, seg_width, seg_height = _segmentation_blueprint(
                world, camera, instance_seg_scale
            )
            seg_sensor = world.spawn_actor(seg_blueprint, transform)
            rig.sensors.append(seg_sensor)
            rig.segmentation_sensors[camera.camera_id] = seg_sensor
            rig.segmentation_scales[camera.camera_id] = (
                seg_width / camera.width,
                seg_height / camera.height,
            )
            queue_name = f"{camera.camera_id}_seg"
            sensor_queue = queue.Queue()
            rig.queues[queue_name] = sensor_queue
            rig.future_data[queue_name] = {}
            seg_sensor.listen(lambda data, q=sensor_queue: q.put(data))

        dvs_params = camera.dvs_parameters()
        rig.writers[camera.camera_id] = CameraWriter(
            sequence.cameras[camera.camera_id].events_file,
            h5py,
            np,
            sequence.sequence_id,
            sequence.sequence_id,
            camera.camera_id,
            camera.mount(),
            sequence.cameras[camera.camera_id].synced_rgb_relative,
            camera.width,
            camera.height,
            time_origin_us,
            integration_time_us,
            flush_frames,
            extra_metadata={
                "town": town,
                "weather": weather,
                "dvs_params": dvs_params,
            },
        )
        # 注意：传感器位姿不能在 spawn_actor 之后立刻读取。CARLA 异步下发
        # 生成指令，同步模式下只有 world.tick() 之后 get_transform() 才是
        # 真实位姿；在此之前它返回原点单位阵。标定因此改由
        # capture_calibration() 在预热 tick 之后采集。
    return rig


def capture_calibration(
    rig: CameraRig, cameras: tuple[CameraConfig, ...]
) -> None:
    """采集各相机��世界位姿，用于保存外参（须在一次 world.tick() 之后调用）。

    CARLA 的生成指令是异步下发的：``spawn_actor`` 刚返回时传感器还停在
    原点，``get_transform()`` 返回单位阵。同步模式下只有 tick 推进后位姿
    才生效，所以这里由采集流程在预热 tick 之后调用，否则写出的
    ``world_matrix`` 是单位阵，离线重投影会投到错误位置。
    """
    for camera in cameras:
        sensor = rig.dvs_sensors[camera.camera_id]
        transform_matrix = sensor.get_transform().get_matrix()
        rig.calibration[camera.camera_id] = {
            "world_matrix": [
                [float(value) for value in row] for row in transform_matrix
            ],
            "focal_length_px": camera.width
            / (2.0 * math.tan(math.radians(camera.fov) / 2.0)),
        }


__all__ = [
    "CameraRig",
    "SensorFrameReader",
    "capture_calibration",
    "spawn_static_camera_rig",
]
