"""测 numba 的调用开销与 kernel 各段耗时（诊断用，不属于交付代码）。"""

import time

import numpy as np
from numba import njit


@njit(cache=True)
def empty(a, b, c, d, w, h, ch, t0, t1, sw, sh):
    return 0.0


@njit(cache=True)
def alloc(a, b, c, d, w, h, ch, t0, t1, sw, sh):
    out = np.zeros(2 * ch * h * w, np.float32)
    return out.reshape(2 * ch, h, w)


@njit(cache=True)
def touch(a, b, c, d, w, h, ch, t0, t1, sw, sh):
    """只做坐标计算，不写内存。"""
    acc = 0.0
    scale_x = w / sw
    scale_y = h / sh
    span = t1 - t0
    for i in range(len(c)):
        gx = int(a[i] * scale_x)
        gy = int(b[i] * scale_y)
        slot = int((c[i] - t0) * ch / span)
        acc += gx + gy + slot
    return acc


@njit(cache=True)
def write(a, b, c, d, w, h, ch, t0, t1, sw, sh):
    out = np.zeros(2 * ch * h * w, np.float32)
    plane = h * w
    scale_x = w / sw
    scale_y = h / sh
    span = t1 - t0
    for i in range(len(c)):
        gx = int(a[i] * scale_x)
        gy = int(b[i] * scale_y)
        slot = int((c[i] - t0) * ch / span)
        if d[i] > 0:
            out[slot * plane + gy * w + gx] += 1.0
        else:
            out[(ch + slot) * plane + gy * w + gx] += 1.0
    return out.reshape(2 * ch, h, w)


def main() -> None:
    n = 13000
    a = np.zeros(n, np.uint32)
    b = np.zeros(n, np.uint32)
    c = np.sort(np.random.default_rng(0).integers(0, 50_000_000, n)).astype(np.int64)
    d = np.random.default_rng(1).choice([-1, 1], n).astype(np.int8)
    args = (a, b, c, d, 640, 360, 9, 0, 50_000_000, 1280, 720)
    for fn in (empty, alloc, touch, write):
        fn(*args)
        t0 = time.perf_counter()
        for _ in range(300):
            fn(*args)
        print(f"{fn.py_func.__name__:8s}: {(time.perf_counter()-t0)/300*1000:.3f} ms")
    import numba

    print("numba", numba.__version__)


if __name__ == "__main__":
    main()
