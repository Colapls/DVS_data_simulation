"""体素网格数据集与 dataloader。

样本单位 = 一个事件窗（npz）。collate 时把每帧的框数补齐到最大值，
目标张量格式：

    boxes  (B, N, 4)  cxcywh，归一化到 [0,1]
    labels (B, N)     0 起类别索引
    track_ids (B, N)  采集器 actor_id
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .label_map import CLASS_NAMES
from .visibility import (
    DEFAULT_MIN_EVENT_RATIO,
    box_event_counts,
    visible_mask,
)


def _sparse_to_dense(
    sparse: tuple[np.ndarray, np.ndarray, tuple[int, ...]]
) -> np.ndarray:
    """稀疏（索引+计数）→ dense 体素网格。"""
    indices, counts, shape = sparse
    flat = np.zeros(int(np.prod(shape)), dtype=np.float32)
    flat[indices] = counts.astype(np.float32)
    return flat.reshape(shape)


#: 稀疏化值得做的阈值：非零格子占比低于此值才转稀疏。
#: 夜间/暴雨样本的体素本就接近稠密（实测 17%~67% 非零），稀疏化没有
#: 收益、还多一次转换。
SPARSE_WORTH_RATIO = 0.2


def _is_sparse_worthwhile(dense: np.ndarray) -> bool:
    """这个体素稀疏到值得走稀疏传输吗？"""
    flat = dense.reshape(-1)
    nonzero = int(np.count_nonzero(flat))
    return nonzero < SPARSE_WORTH_RATIO * flat.size


def _dense_to_sparse(
    dense: np.ndarray, min_count: float = 0.5
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """dense 体素 → 稀疏（索引+计数）。计数小于 min_count 的丢弃。"""
    flat = dense.reshape(-1)
    nonzero = np.nonzero(flat >= min_count)[0]
    return (
        nonzero.astype(np.int32),
        np.minimum(flat[nonzero], 255.0).astype(np.uint8),
        tuple(int(v) for v in dense.shape),
    )


def _load_voxel(data) -> np.ndarray | None:
    """从 npz 重建 dense 体素网格。

    切片按稀疏格式存（非零格子的索引 + 计数）——事件体素里只有约 0.2%
    的格子非零，dense float16 每窗 8.3 MB、总量 ~28 GB，训练时 IO 会
    成为瓶颈。重建 dense 是一次索引赋值，比读 8 MB 快两个数量级。

    没有体素字段时返回 None（合成的测试数据）。
    """
    if "voxel_indices" in data:
        shape = tuple(int(v) for v in data["voxel_shape"])
        flat = np.zeros(int(np.prod(shape)), dtype=np.float32)
        flat[data["voxel_indices"]] = data["voxel_counts"].astype(np.float32)
        return flat.reshape(shape)
    if "voxel" in data:
        return data["voxel"].astype(np.float32)
    return None


def _load_voxel_sparse(data) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]] | None:
    """取稀疏体素（索引 + 计数 + 形状），不重建 dense。

    训练时用它做跨进程传输：稀疏格式每样本约 50 KB，dense 是 16.6 MB，
    差 300 倍。dense 在 collate 里一次性重建（主进程，省掉 IPC 开销）。
    """
    if "voxel_indices" in data:
        shape = tuple(int(v) for v in data["voxel_shape"])
        return data["voxel_indices"], data["voxel_counts"], shape
    if "voxel" in data:
        dense = data["voxel"].astype(np.float32)
        flat = dense.reshape(-1)
        nonzero = np.nonzero(flat)[0]
        return nonzero.astype(np.int32), flat[nonzero].astype(np.uint8), dense.shape
    return None


class CarlaDVSDataset(Dataset):
    """按 npz 路径列表读取切片样本。"""

    def __init__(
        self,
        samples: Sequence[str | Path],
        grid_size: tuple[int, int] = (360, 640),
        training: bool = False,
        augment=None,
        min_event_ratio: float = DEFAULT_MIN_EVENT_RATIO,
        filter_invisible: bool = True,
    ) -> None:
        self.samples = [Path(item) for item in samples]
        self.grid_size = grid_size
        self.training = training
        self.augment = augment
        # 训练时丢弃 DVS 看不见的框（框内几乎没有事件）：这些框是
        # "静止目标假阴性"标签的来源，会教模型去预测不存在的事件。
        # 评估侧把同样的框按 iscrowd 处理，两端口径一致。
        self.min_event_ratio = min_event_ratio
        self.filter_invisible = filter_invisible

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with np.load(self.samples[index], allow_pickle=True) as data:
            sparse = _load_voxel_sparse(data)
            boxes = data["boxes"].astype(np.float32)
            labels = data["labels"].astype(np.int64)
            track_ids = data["track_ids"].astype(np.int64)

        # 裁掉超出画面的框（标注是投影框，可能越界）
        if boxes.size:
            height, width = self.grid_size
            x1 = np.clip(boxes[:, 0], 0, width)
            y1 = np.clip(boxes[:, 1], 0, height)
            x2 = np.clip(boxes[:, 0] + boxes[:, 2], 0, width)
            y2 = np.clip(boxes[:, 1] + boxes[:, 3], 0, height)
            keep = (x2 > x1) & (y2 > y1)
            boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)[keep]
            labels = labels[keep]
            track_ids = track_ids[keep]
            # 丢弃 DVS 看不见的框（静止/极远目标，事件流里没有痕迹）
            if self.filter_invisible and len(boxes) and sparse is not None:
                voxel = _sparse_to_dense(sparse)
                counts = box_event_counts(voxel, boxes)
                visible = visible_mask(counts, boxes, self.min_event_ratio)
                boxes = boxes[visible]
                labels = labels[visible]
                track_ids = track_ids[visible]

        # 切片 npz 的框是 xywh 像素 → 归一化 cxcywh
        if boxes.size:
            height, width = self.grid_size
            cx = (boxes[:, 0] + boxes[:, 2] / 2) / width
            cy = (boxes[:, 1] + boxes[:, 3] / 2) / height
            bw = boxes[:, 2] / width
            bh = boxes[:, 3] / height
            boxes = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)

        sample = {
            "sparse": sparse,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "track_ids": torch.from_numpy(track_ids),
        }
        if self.training and self.augment is not None:
            # 增强要在 dense 上做（几何变换），但跨进程传输要尽量小：
            # 在 worker 进程里 dense→增强→(稀疏或 dense)，主进程收到的
            # 是处理好的数据。
            dense = _sparse_to_dense(sparse) if sparse is not None else None
            sample = self.augment(
                {**sample, "voxel": torch.from_numpy(dense) if dense is not None else None}
            )
            voxel_out = sample.pop("voxel", None)
            if voxel_out is not None:
                array = voxel_out.numpy()
                # 夜间/暴雨样本的体素本就接近稠密（实测 17%~67% 非零），
                # 稀疏化没有收益、还多一次转换——这类样本直接传 dense。
                if _is_sparse_worthwhile(array):
                    sample["sparse"] = _dense_to_sparse(array)
                else:
                    sample["dense"] = voxel_out
        return sample


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """把批内的框补齐到最大数量；稀疏体素合并后重建 dense。

    稀疏数据在 worker 进程里合成 dense 会让 IPC 传 133 MB/batch（dense
    每样本 16.6 MB），是训练慢的主因（GPU 利用率只有 14%）。这里在主
    进程里合并：先在稀疏域把各样本的索引平移到 batch 偏移，再一次性
    分配 dense 并写入。
    """
    max_boxes = max(int(item["boxes"].shape[0]) for item in batch)
    max_boxes = max(max_boxes, 1)
    batch_size = len(batch)
    shape = None
    for item in batch:
        if item.get("sparse") is not None:
            shape = item["sparse"][2]
            break
        if item.get("dense") is not None:
            shape = tuple(item["dense"].shape)
            break
        if item.get("voxel") is not None:  # 合成样本直接给 dense
            shape = tuple(item["voxel"].shape)
            break
    if shape is None:
        voxels = torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32)
    else:
        flat_size = int(np.prod(shape))
        flat = np.zeros(batch_size * flat_size, dtype=np.float32)
        for index, item in enumerate(batch):
            dense = item.get("dense")
            if dense is None:
                dense = item.get("voxel")
            if dense is not None:
                flat[index * flat_size : (index + 1) * flat_size] = (
                    dense.reshape(-1).numpy()
                )
                continue
            sparse = item.get("sparse")
            if sparse is None:
                continue
            indices, counts, _ = sparse
            flat[index * flat_size + indices] = counts.astype(np.float32)
        voxels = torch.from_numpy(
            flat.reshape((batch_size,) + tuple(shape))
        )
    boxes = torch.zeros(batch_size, max_boxes, 4, dtype=torch.float32)
    labels = torch.full((batch_size, max_boxes), -1, dtype=torch.long)
    track_ids = torch.zeros(batch_size, max_boxes, dtype=torch.long)
    for index, item in enumerate(batch):
        count = int(item["boxes"].shape[0])
        if count:
            boxes[index, :count] = item["boxes"]
            labels[index, :count] = item["labels"]
            track_ids[index, :count] = item["track_ids"]
    return {
        "voxel": voxels,
        "boxes": boxes,
        "labels": labels,
        "track_ids": track_ids,
    }


def build_dataloader(
    samples: Sequence[str | Path],
    batch_size: int = 16,
    grid_size: tuple[int, int] = (360, 640),
    training: bool = False,
    augment=None,
    workers: int = 4,
    min_event_ratio: float = DEFAULT_MIN_EVENT_RATIO,
    filter_invisible: bool = True,
) -> torch.utils.data.DataLoader:
    """构造 dataloader。"""
    dataset = CarlaDVSDataset(
        samples,
        grid_size=grid_size,
        training=training,
        augment=augment,
        min_event_ratio=min_event_ratio,
        filter_invisible=filter_invisible,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=training,
        persistent_workers=workers > 0,
    )


def class_names() -> tuple[str, ...]:
    return CLASS_NAMES
