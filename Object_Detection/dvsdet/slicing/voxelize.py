"""事件体素化：把一个事件窗压成 (2C, H, W) 的体素网格。

方案 §3.1 的实现有两个必须修正的地方：

1. **时间单位**：采集器落盘的 `events/t` 是**纳秒**（`meta/event_timestamp_unit`），
   方案里写的 µs 与实际不符。体素化内部统一按 ns 计算时间通道，调用方
   传入的窗边界也必须是 ns。
2. **分辨率**：实际采集是 1280×720，方案按 640×360 写。体素化直接按目标
   网格尺寸重采样（x/y 线性映射 + 边界裁剪），不必先建全分辨率网格。

极性分离：前 C 个通道是 ON，后 C 个通道是 OFF。
"""

from __future__ import annotations

import numpy as np

try:  # numba 加速；缺失时回退纯 numpy（慢约 20 倍）
    from numba import njit

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - 环境未装 numba 时走回退分支
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        if args and callable(args[0]):
            return args[0]

        def _wrap(func):
            return func

        return _wrap


@njit(cache=True)
def _voxelize_kernel(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    out: np.ndarray,
    width: int,
    height: int,
    channels: int,
    t0: int,
    t1: int,
    source_width: int,
    source_height: int,
) -> None:
    """事件按时间线性分配到 C 个时间通道；极性分离为前后两段。

    输出数组由调用方分配并传入——**不能在 kernel 里 np.zeros**：numba 的
    NRT 分配 16 MB 要 4.6 ms（numpy 自己的内存池只要 0.02 ms），实测
    这一项就是体素化耗时的全部（事件循环本身仅 0.03 ms）。
    """
    span = t1 - t0
    if span <= 0:
        span = 1
    scale_x = width / source_width
    scale_y = height / source_height
    plane = height * width
    for i in range(len(t)):
        gx = int(x[i] * scale_x)
        if gx < 0 or gx >= width:
            continue
        gy = int(y[i] * scale_y)
        if gy < 0 or gy >= height:
            continue
        slot = int((t[i] - t0) * channels / span)
        if slot < 0:
            slot = 0
        elif slot >= channels:
            slot = channels - 1
        if p[i] > 0:
            out[slot * plane + gy * width + gx] += 1.0
        else:
            out[(channels + slot) * plane + gy * width + gx] += 1.0


def voxelize(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    width: int,
    height: int,
    channels: int,
    t0: int,
    t1: int,
    source_width: int = 1280,
    source_height: int = 720,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """把窗内事件体素化成 (2C, H, W) float32。

    坐标按采集分辨率线性映射到目标网格；t0/t1 与 t 同单位（ns）。

    **传入 out 复用缓冲区可省掉 4~6 ms/窗**：每次新建 16.6 MB 数组时，
    kernel 写入会触发整块内存的首次触碰页错误（Windows 上实测 ~5 ms）；
    复用已驻留的缓冲区后同一操作只要 0.09 ms。方案 §3.4 的 DoD 要求
    体素化 < 2 ms/窗，训练与推理都应复用缓冲区。
    """
    if out is None:
        out = np.zeros((2 * channels, height, width), dtype=np.float32)
    elif out.shape != (2 * channels, height, width):
        raise ValueError(
            f"缓冲区形状 {out.shape} 与目标 {(2*channels, height, width)} 不符"
        )
    else:
        out.fill(0.0)
    if len(t) == 0:
        return out
    _voxelize_kernel(
        np.ascontiguousarray(x, dtype=np.uint32),
        np.ascontiguousarray(y, dtype=np.uint32),
        np.ascontiguousarray(t, dtype=np.int64),
        np.ascontiguousarray(p, dtype=np.int8),
        out.reshape(-1),  # 扁平视图：kernel 里用一维索引写更快
        width,
        height,
        channels,
        int(t0),
        int(t1),
        int(source_width),
        int(source_height),
    )
    return out


def numba_available() -> bool:
    """是否启用了 numba 加速（体素化耗时的验收口径依赖它）。"""
    return _HAVE_NUMBA
