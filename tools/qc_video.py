#!/usr/bin/env python3
"""三栏质检视频：RGB | 事件帧 | 带框的 RGB。

按方案 §7.3 生成质检材料，用来肉眼确认：骑手没有被重复标成行人、树后没有
框、三类别计数正确、框和事件簇对得上。

输出默认是一串 PNG 帧（不依赖 ffmpeg）；装了 imageio-ffmpeg 时另外编码一个
mp4。

用法::

    python tools/qc_video.py --dataset dvs_roadside_multicam \
        --sequence sequence_000001 --camera cam_01 --out-dir qc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



#: 类别 → 画框颜色（RGB）。
COLORS = {
    "car": (255, 96, 0),
    "motorcycle": (255, 0, 255),
    "pedestrian": (0, 220, 0),
}


def _load_font(size: int = 14) -> Any:
    """尽量用系统字体；没有就用 PIL 自带的默认字体。"""
    from PIL import ImageFont

    for candidate in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _event_frame(
    xs: Any, ys: Any, polarity: Any, width: int, height: int, np: Any
) -> Any:
    """把一帧的事件画成 RGB 图：ON 红、OFF 蓝。"""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if xs.size:
        on = polarity > 0
        canvas[ys[on], xs[on]] = (255, 64, 64)
        canvas[ys[~on], xs[~on]] = (64, 64, 255)
    return canvas


def _draw_labels(image: Any, annotations: list[dict[str, Any]], font: Any) -> Any:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        color = COLORS.get(annotation.get("category", ""), (255, 255, 0))
        if annotation.get("ignore") or annotation.get("difficult"):
            color = (128, 128, 128)
        draw.rectangle(
            [x, y, x + width, y + height], outline=color, width=2
        )
        flags = ",".join(annotation.get("quality_flags", []))
        label = annotation.get("category", "?")
        if flags:
            label = f"{label}[{flags}]"
        draw.text((x, max(0, y - 14)), label, fill=color, font=font)
    return image


def render_sequence(
    dataset_root: Path,
    sequence_id: str,
    camera_id: str,
    out_dir: Path,
    limit: int,
    np: Any,
) -> list[Path]:
    """渲染一个机位的三栏帧序列，返回写出的图片路径。"""
    import h5py
    from PIL import Image

    camera_dir = dataset_root / "sequences" / sequence_id / "cameras" / camera_id
    annotations = json.loads(
        (camera_dir / "annotations.json").read_text(encoding="utf-8")
    )
    frames = annotations.get("frames", [])
    font = _load_font()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with h5py.File(camera_dir / "events.h5", "r") as handle:
        offsets = handle["frames/event_offsets"][:]
        width = int(handle["meta/sensor_width"][()])
        height = int(handle["meta/sensor_height"][()])
        event_x = handle["events/x"]
        event_y = handle["events/y"]
        event_p = handle["events/p"]
        for index, frame in enumerate(frames):
            if limit and index >= limit:
                break
            start = int(offsets[index]) if index + 1 < len(offsets) else 0
            end = int(offsets[index + 1]) if index + 1 < len(offsets) else start
            if end > start:
                xs = event_x[start:end].astype(np.int64)
                ys = event_y[start:end].astype(np.int64)
                polarity = event_p[start:end]
            else:
                xs = np.zeros(0, dtype=np.int64)
                ys = np.zeros(0, dtype=np.int64)
                polarity = np.zeros(0, dtype=np.int64)
            events = Image.fromarray(
                _event_frame(xs, ys, polarity, width, height, np)
            )
            rgb_path = camera_dir / "rgb" / f"{index:06d}.png"
            if not rgb_path.is_file():
                rgb_path = camera_dir / "rgb" / f"{index:06d}.jpg"
            rgb = Image.open(rgb_path).convert("RGB") if rgb_path.is_file() else None
            overlay = rgb.copy() if rgb is not None else Image.new(
                "RGB", (width, height)
            )
            _draw_labels(overlay, frame.get("annotations", []), font)
            if rgb is None:
                rgb = Image.new("RGB", (width, height))
            canvas = Image.new("RGB", (width * 3, height))
            canvas.paste(rgb, (0, 0))
            canvas.paste(events, (width, 0))
            canvas.paste(overlay, (width * 2, 0))
            path = out_dir / f"{index:06d}.png"
            canvas.save(path)
            written.append(path)
    return written


def encode_video(frames: list[Path], output: Path) -> bool:
    """有 imageio-ffmpeg 就编码 mp4；没有就只留 PNG 帧。"""
    try:
        import imageio_ffmpeg
    except ImportError:
        return False
    import numpy as np
    from PIL import Image

    if not frames:
        return False
    with Image.open(frames[0]) as first:
        size = first.size
    writer = imageio_ffmpeg.write_frames(
        str(output),
        size=size,
        fps=20,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        pixelformat="yuv420p",
    )
    writer.send(None)
    try:
        for path in frames:
            with Image.open(path) as image:
                writer.send(np.asarray(image.convert("RGB")))
    finally:
        writer.close()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="生成三栏质检视频")
    parser.add_argument(
        "--dataset", type=Path, default=Path("dvs_roadside_multicam")
    )
    parser.add_argument("--sequence", required=True, help="序列 ID，如 sequence_000001")
    parser.add_argument("--camera", required=True, help="机位 ID，如 cam_01")
    parser.add_argument("--out-dir", type=Path, default=Path("qc"))
    parser.add_argument("--limit", type=int, default=0, help="只渲染前 N 帧（0 = 全部）")
    args = parser.parse_args()
    import numpy as np

    dataset_root = args.dataset.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() / f"{args.sequence}_{args.camera}"
    frames = render_sequence(
        dataset_root, args.sequence, args.camera, out_dir, args.limit, np
    )
    print(f"已渲染 {len(frames)} 帧 → {out_dir}")
    video = out_dir / f"{args.sequence}_{args.camera}.mp4"
    if encode_video(frames, video):
        print(f"视频：{video}")
    else:
        print(
            "未找到 imageio-ffmpeg，只输出了 PNG 帧；"
            "需要 mp4 可执行：ffmpeg -framerate 20 -i %06d.png -c:v libx264 out.mp4"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
