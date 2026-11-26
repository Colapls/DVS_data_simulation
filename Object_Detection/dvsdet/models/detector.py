"""自建轻量检测器：体素网格 → 目标框 + 三分类（识别即检测头分类分支）。

方案 §7.1 路线 A/B 的自建实现，与 RVT 保持同样的输入输出契约：

    输入 (B, 2C, H, W) 体素网格
    输出 {box(归一化 cxcywh), score, cls ∈ {car, motorcycle, pedestrian}}

结构（anchor-free，FCOS 风格）：

    Stem(18→32, s2) → Stage(32, s4) → Stage(64, s8) → Stage(96, s16)
                                              ↓ FPN 上采样
    P3(s8) P4(s16) P5(s32) → 共享的解耦检测头 → 分类 + 回归 + centerness

选 anchor-free 而非 YOLOX anchor-based 的理由：标注框来自 3D 框投影，
尺寸分布比自然图像杂（远处小车只有几个像素宽），anchor 尺寸难先验；
FCOS 逐位置回归对尺度更鲁棒，且没有 anchor 匹配导致的正负样本失衡。

数据量小（千级样本），所以 backbone 刻意做小（~1.5M 参数）。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_act(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm2d(channels),
        nn.SiLU(inplace=True),
    )


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            _norm_act(out_channels),
        )


class ResidualBlock(nn.Module):
    """倒残差块（移动端风格），小数据下比 ResNet 更好训。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        hidden = max(out_channels // 2, 16)
        self.conv1 = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(
            hidden, hidden, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.bn1(self.conv1(x)), inplace=True)
        out = F.silu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        return F.silu(out + self.shortcut(x), inplace=True)


class TinyFPN(nn.Module):
    """轻量 FPN：自底向上 + 自顶向下路径，输出 3 个尺度。"""

    def __init__(self, channels: tuple[int, int, int], out_channels: int = 96):
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(channels, out_channels, 1, bias=False)
            for channels in channels
        )
        self.lateral_bn = nn.ModuleList(
            nn.BatchNorm2d(out_channels) for _ in channels
        )
        self.smooth = nn.ModuleList(
            ConvBNAct(out_channels, out_channels) for _ in channels
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = [
            bn(conv(feature))
            for conv, bn, feature in zip(
                self.lateral, self.lateral_bn, features
            )
        ]
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                size=laterals[index - 1].shape[-2:],
                mode="nearest",
            )
        return [smooth(lateral) for smooth, lateral in zip(self.smooth, laterals)]


class DetectionHead(nn.Module):
    """解耦检测头：分类 / 回归 / centerness 三条分支。"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_convs: int = 2,
        prior_prob: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in range(num_convs):
            self.cls_convs.append(
                nn.Sequential(
                    ConvBNAct(in_channels, in_channels),
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    ConvBNAct(in_channels, in_channels),
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                )
            )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.ctr_pred = nn.Conv2d(in_channels, 1, 3, padding=1)
        # 分类分支的偏置初始化成 prior_prob，让初始预测概率很低（focal loss 友好）
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.ctr_pred.bias, -math.log((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.reg_pred.bias, 0.0)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        cls_feat = feature
        reg_feat = feature
        for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
            cls_feat = cls_conv(cls_feat)
            reg_feat = reg_conv(reg_feat)
        return {
            "cls_logits": self.cls_pred(cls_feat),
            "reg_preds": self.reg_pred(reg_feat),
            "ctr_logits": self.ctr_pred(reg_feat),
        }


class LightVoxelDetector(nn.Module):
    """体素网格上的轻量 anchor-free 检测器（三分类，识别同出）。"""

    def __init__(
        self,
        in_channels: int = 18,
        num_classes: int = 3,
        width: int = 64,
        strides: tuple[int, ...] = (8, 16, 32),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, width // 2, stride=2),
            ConvBNAct(width // 2, width // 2, stride=1),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(width // 2, width, stride=2),  # s4
            ResidualBlock(width, width, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(width, width * 2, stride=2),  # s8
            ResidualBlock(width * 2, width * 2, stride=1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(width * 2, width * 3, stride=2),  # s16
            ResidualBlock(width * 3, width * 3, stride=1),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(width * 3, width * 3, stride=2),  # s32
        )
        self.fpn = TinyFPN(
            (width * 2, width * 3, width * 3), out_channels=width + width // 2
        )
        head_channels = width + width // 2
        self.head = DetectionHead(head_channels, num_classes)
        # 回归量的缩放与 criterion.REG_SCALE 严格一致：训练与推理解码
        # 必须同参，否则粗尺度层的框会系统性放大。
        self.reg_scale = 4.0

    def forward(self, voxel: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = self.stem(voxel)
        x = self.stage1(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        features = self.fpn([c3, c4, c5])
        outputs = [self.head(feature) for feature in features]
        return {
            "cls_logits": [out["cls_logits"] for out in outputs],
            "reg_preds": [out["reg_preds"] for out in outputs],
            "ctr_logits": [out["ctr_logits"] for out in outputs],
        }

    @torch.no_grad()
    def predict(
        self,
        voxel: torch.Tensor,
        score_threshold: float = 0.05,
        topk: int = 100,
        nms_threshold: float = 0.5,
        image_width: int = 640,
        image_height: int = 360,
    ) -> list[dict[str, torch.Tensor]]:
        """推理：逐位置打分 → topk → NMS，返回每张图的检测结果。

        返回的框是**图像像素坐标** xyxy（已裁剪到画面内）。
        """
        self.eval()
        outputs = self(voxel)
        results = []
        for batch_index in range(voxel.shape[0]):
            boxes_all: list[torch.Tensor] = []
            scores_all: list[torch.Tensor] = []
            labels_all: list[torch.Tensor] = []
            for level, stride in enumerate(self.strides):
                cls_logits = outputs["cls_logits"][level][batch_index]
                reg = outputs["reg_preds"][level][batch_index]
                ctr = outputs["ctr_logits"][level][batch_index]
                scores = (
                    cls_logits.sigmoid().amax(dim=0) * ctr.sigmoid().squeeze(0)
                )
                keep = scores > score_threshold
                if keep.sum() == 0:
                    continue
                index = keep.nonzero(as_tuple=False)
                level_scores = scores[index[:, 0], index[:, 1]]
                # topk 加速
                if level_scores.numel() > topk:
                    top = level_scores.topk(topk).indices
                    index = index[top]
                    level_scores = level_scores[top]
                cls_at = cls_logits[:, index[:, 0], index[:, 1]].T  # (N, C)
                best_cls, best_label = cls_at.max(dim=1)
                reg_at = reg[:, index[:, 0], index[:, 1]].T  # (N, 4)
                ltrb = F.softplus(reg_at) * self.reg_scale
                cx = index[:, 1].float() + 0.5
                cy = index[:, 0].float() + 0.5
                x1 = cx - ltrb[:, 0]
                y1 = cy - ltrb[:, 1]
                x2 = cx + ltrb[:, 2]
                y2 = cy + ltrb[:, 3]
                # 特征网格坐标 → 图像像素：位置与 ltrb 距离都按 stride 放大。
                # 直接返回网格坐标的话，调用方（评估/可视化）会拿到 0~45 的
                # "像素框"，全部挤在画面左上角。
                boxes_all.append(
                    torch.stack([x1, y1, x2, y2], dim=1) * float(stride)
                )
                scores_all.append(level_scores * best_cls.sigmoid())
                labels_all.append(best_label)
            if not boxes_all:
                results.append(
                    {
                        "boxes": torch.zeros((0, 4), device=voxel.device),
                        "scores": torch.zeros((0,), device=voxel.device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=voxel.device),
                    }
                )
                continue
            boxes = torch.cat(boxes_all)
            scores = torch.cat(scores_all)
            labels = torch.cat(labels_all)
            # 按类别分别 NMS：跨类 NMS 会让大车框压掉重叠的行人/摩托框
            keep_parts = []
            for class_index in labels.unique():
                class_mask = labels == class_index
                class_keep = nms(
                    boxes[class_mask], scores[class_mask], nms_threshold
                )
                keep_parts.append(class_mask.nonzero(as_tuple=False)[class_keep, 0])
            keep = torch.cat(keep_parts)
            # 按分数降序排（各类合并后顺序乱了）
            keep = keep[scores[keep].argsort(descending=True)]
            selected_boxes = boxes[keep].clone()
            # 裁剪到画面内：未训练的模型会给出任意框，下游（评估/可视化）
            # 拿到越界坐标会画到画面外
            selected_boxes[:, 0].clamp_(min=0)
            selected_boxes[:, 1].clamp_(min=0)
            selected_boxes[:, 2].clamp_(max=float(image_width))
            selected_boxes[:, 3].clamp_(max=float(image_height))
            results.append(
                {
                    "boxes": selected_boxes,
                    "scores": scores[keep],
                    "labels": labels[keep],
                }
            )
        return results


def nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """类别内 NMS（xyxy 框）。"""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        current = int(order[0])
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(x1[current], x1[rest])
        yy1 = torch.max(y1[current], y1[rest])
        xx2 = torch.min(x2[current], x2[rest])
        yy2 = torch.min(y2[current], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[current] + areas[rest] - inter).clamp(min=1e-6)
        order = rest[iou <= threshold]
    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)


def build_model(config: dict[str, Any] | None = None) -> LightVoxelDetector:
    """按配置构造模型。"""
    config = config or {}
    return LightVoxelDetector(
        in_channels=int(config.get("in_channels", 18)),
        num_classes=int(config.get("num_classes", 3)),
        width=int(config.get("width", 64)),
    )
