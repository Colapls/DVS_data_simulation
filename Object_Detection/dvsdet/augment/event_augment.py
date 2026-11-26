"""事件特有增强（方案 §4.5）。

四条增强，全部保持窗内时空一致性，**禁用任何亮度/色彩类增强**——事件
数据没有亮度概念：

1. 时间轴抖动：把窗内事件时间整体 ±10% 抖动，等价于对体素按时间维弹性
   扰动。实现上等价于重排时间通道的权重。
2. 事件密度缩放：随机置零 10~30% 体素计数，模拟低对比度/远距小目标。
3. 背景活动噪声注入：按 npz 里的 `contrast_threshold` 反推噪声强度，撒
   泊松噪声点（真实器件背景活动，sim-to-real 对冲手段；也是消融第 3 组）。
4. 几何增强：hflip / translate / scale，对所有时间通道一致施加（**不能
   逐通道独立**，那会破坏事件的时间-空间一致性）。

注意：几何增强必须同时作用到标注框，否则框与体素错位。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class EventAugment:
    """对单个样本（dict）做事件增强，就地返回新样本。"""

    def __init__(
        self,
        temporal_jitter_ratio: float = 0.1,
        event_drop_ratio: tuple[float, float] = (0.1, 0.3),
        noise_scale: float = 1.0,
        geometry: tuple[str, ...] = ("hflip", "translate", "scale"),
        prob: float = 0.5,
        seed: int | None = None,
    ):
        self.temporal_jitter_ratio = temporal_jitter_ratio
        self.event_drop_ratio = event_drop_ratio
        self.noise_scale = noise_scale
        self.geometry = tuple(geometry)
        self.prob = prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        voxel = sample["voxel"]
        boxes = sample["boxes"]
        if self.rng.random() < self.prob:
            voxel = self._temporal_jitter(voxel)
        if self.rng.random() < self.prob:
            voxel = self._event_dropout(voxel)
        if self.rng.random() < self.prob:
            voxel = self._inject_noise(voxel, sample.get("contrast_threshold", 0.25))
        voxel, boxes = self._geometry(voxel, boxes)
        return {**sample, "voxel": voxel, "boxes": boxes}

    # ---- 1) 时间轴抖动 ----
    def _temporal_jitter(self, voxel: torch.Tensor) -> torch.Tensor:
        """按时间维做随机线性拉伸/压缩（模拟窗内事件时间分布抖动）。

        实现：时间通道按 (1±ratio) 重采样——输出第 i 个通道取输入的
        i*(1+ratio) 处（线性插值），越界端点用 clamp。等价于把窗内事件的
        时间轴弹性拉伸，对体素做 gather 即可，不需要 interpolate。
        """
        channels = voxel.shape[0] // 2
        if channels < 2:
            return voxel
        ratio = self.rng.uniform(
            -self.temporal_jitter_ratio, self.temporal_jitter_ratio
        )
        if abs(ratio) < 1e-3:
            return voxel
        positions = torch.arange(channels, dtype=torch.float32) * (1.0 + ratio)
        positions = positions.clamp(0, channels - 1)
        lower = positions.floor().long()
        upper = (lower + 1).clamp(max=channels - 1)
        weight = (positions - lower.float()).view(-1, 1, 1)

        def resample(block: torch.Tensor) -> torch.Tensor:
            return block[lower] * (1 - weight) + block[upper] * weight

        return torch.cat(
            [resample(voxel[:channels]), resample(voxel[channels:])], dim=0
        )

    # ---- 2) 事件密度缩放 ----
    def _event_dropout(self, voxel: torch.Tensor) -> torch.Tensor:
        """随机置零部分体素计数，模拟低对比度。"""
        ratio = self.rng.uniform(*self.event_drop_ratio)
        mask = torch.from_numpy(
            (self.rng.random(voxel.shape[-2:]) >= ratio).astype(np.float32)
        )
        return voxel * mask

    # ---- 3) 背景活动噪声注入 ----
    def _inject_noise(
        self, voxel: torch.Tensor, contrast_threshold: float = 0.25
    ) -> torch.Tensor:
        """按对比度阈值反推噪声强度，撒泊松噪声点。

        对比度阈值越低（暗光/低对比度场景）真实器件的背景活动越强，
        所以噪声强度与 threshold 成反比。
        """
        strength = float(self.noise_scale) * max(0.02, 0.25 / max(contrast_threshold, 0.05))
        if strength <= 0:
            return voxel
        noise = self.rng.poisson(strength, size=voxel.shape[-2:]).astype(np.float32)
        out = voxel.clone()
        # 噪声主要落在空白时间通道上（真实背景活动散布在时间轴各处）
        out += torch.from_numpy(noise).unsqueeze(0)
        return out

    # ---- 4) 几何增强（对所有通道一致）----
    def _geometry(
        self, voxel: torch.Tensor, boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """几何增强。boxes 是归一化 cxcywh（dataloader 的输出格式），
        hflip 只需 cx → 1−cx；scale 不改归一化框，但体素必须缩回原尺寸，
        否则批内样本尺寸不一致，collate 直接崩。"""
        for name in self.geometry:
            if self.rng.random() >= self.prob:
                continue
            if name == "hflip":
                voxel = torch.flip(voxel, dims=[-1])
                if boxes.numel():
                    boxes = boxes.clone()
                    boxes[:, 0] = 1.0 - boxes[:, 0]
            elif name == "translate":
                dx = float(self.rng.uniform(-0.08, 0.08))
                dy = float(self.rng.uniform(-0.08, 0.08))
                voxel = torch.roll(
                    voxel,
                    shifts=(int(dy * voxel.shape[-2]), int(dx * voxel.shape[-1])),
                    dims=(-2, -1),
                )
                if boxes.numel():
                    boxes = boxes.clone()
                    boxes[:, 0] += dx
                    boxes[:, 1] += dy
            elif name == "scale":
                factor = float(self.rng.uniform(0.85, 1.15))
                height, width = voxel.shape[-2:]
                zoom_h = max(int(height * factor), 32)
                zoom_w = max(int(width * factor), 32)
                voxel = torch.nn.functional.interpolate(
                    voxel.unsqueeze(0),
                    size=(zoom_h, zoom_w),
                    mode="bilinear",
                    align_corners=False,
                )
                # 缩放后必须回到原尺寸（zoom-in/out 效果），归一化框不变
                voxel = torch.nn.functional.interpolate(
                    voxel,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        return voxel, boxes
