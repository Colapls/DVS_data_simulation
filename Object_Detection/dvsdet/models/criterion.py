"""检测损失：focal（分类）+ GIoU（回归）+ BCE（centerness）。

对应方案 §4.4：focal loss γ=2 α=0.25（背景事件远多于前景）、GIoU 回归、
类别权重 car/moto/ped = 1.0/1.5/2.0（摩托是新增类、行人是难点类）。

正样本分配用 FCOS 的 center sampling：GT 框中心附近的邻域内的位置都算
正样本（远处小目标可能只占一两个格子，逐点判定会让它们没有正样本）。

**回归量的语义**（与 detector.predict 的解码严格一致）：每个位置回归它
到所属 GT 框四边的距离 (l, t, r, b)，经 softplus×REG_SCALE 缩放。训练
和推理必须用同一套缩放——此前训练用 ×4、推理用 ×[4,8,16]，粗尺度层的
框会系统性放大 2~4 倍。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

#: 回归距离的缩放常数。训练与推理共用（detector.predict 同样引用）。
REG_SCALE = 4.0

#: center sampling 半径（特征格子数）。
CENTER_RADIUS = 1.5


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """逐位置 focal loss（targets: 0/1 前景掩码，形状与 logits 一致）。"""
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if class_weights is not None:
        # 前景位置按其类别加权（car=1.0 / moto=1.5 / ped=2.0）
        weight = 1.0 + (class_weights - 1.0).view(1, -1, 1, 1) * targets
        loss = loss * weight
    return loss.sum()


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU（xyxy）。"""
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    union = (area1 + area2 - inter).clamp(min=1e-6)
    iou = inter / union
    enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enclose = (enclose_x2 - enclose_x1).clamp(min=0) * (
        enclose_y2 - enclose_y1
    ).clamp(min=0)
    return iou - (enclose - union) / enclose.clamp(min=1e-6)


def assign_targets(
    boxes_cxcywh: torch.Tensor,
    labels: torch.Tensor,
    feature_sizes: list[tuple[int, int]],
) -> list[dict[str, torch.Tensor]]:
    """逐尺度生成训练目标。按 GT 循环（每图约 10 个），不按像素循环。

    **boxes_cxcywh 的格式是归一化 cxcywh**（dataloader 的输出格式）——
    传成 xyxy 会被再转换一次，框就错位了，训练永远学不会定位。

    labels: (B, N)，-1 为填充。返回每个尺度一个 dict：
        cls_target  (B, H, W)     0=背景，k+1=第 k 类
        ltrb_target (B, H, W, 4)  位置到 GT 四边的距离（特征格子单位）
        ctr_target  (B, H, W)     FCOS centerness ∈ [0,1]
    重叠 GT 时后写的覆盖先写的（FCOS 原版按面积小者优先；数据集里框
    重叠少，简化处理，已知上限）。

    **性能的三次教训**（改这里前先读完）：
    1. GPU 上：每 GT 十几次小 kernel launch，batch 8 × 15 GT × 3 尺度
       ≈ 数千次 launch，实测 644 ms/batch（前向才 38 ms）。
    2. CPU + torch：每个微型 torch 操作仍有 ~80µs 的派发开销，实测
       仍要 375 ms。
    3. CPU + numpy：微型数组操作 ~1µs，同样循环 <10 ms。当前实现。
    """
    device = boxes_cxcywh.device
    batch = boxes_cxcywh.shape[0]
    # 全程 numpy（小操作无派发开销），最后一次性转 torch
    boxes_np = boxes_cxcywh.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    results = []
    for height, width in feature_sizes:
        cls_np = np.zeros((batch, height, width), dtype=np.float32)
        ltrb_np = np.zeros((batch, height, width, 4), dtype=np.float32)
        ctr_np = np.zeros((batch, height, width), dtype=np.float32)
        for b in range(batch):
            valid = labels_np[b] >= 0
            if not valid.any():
                continue
            gt = boxes_np[b][valid]  # (G, 4) 归一化 cxcywh
            gt_labels = labels_np[b][valid] + 1  # 1 起的类别
            # cxcywh → 该尺度的格子坐标 xyxy
            gx1 = (gt[:, 0] - gt[:, 2] / 2) * width
            gy1 = (gt[:, 1] - gt[:, 3] / 2) * height
            gx2 = (gt[:, 0] + gt[:, 2] / 2) * width
            gy2 = (gt[:, 1] + gt[:, 3] / 2) * height
            centers_x = (gx1 + gx2) / 2
            centers_y = (gy1 + gy2) / 2
            for index in range(gt.shape[0]):
                y0 = int(np.clip(centers_y[index] - CENTER_RADIUS, 0, height - 1))
                y1 = int(np.clip(centers_y[index] + CENTER_RADIUS, 0, height - 1))
                x0 = int(np.clip(centers_x[index] - CENTER_RADIUS, 0, width - 1))
                x1 = int(np.clip(centers_x[index] + CENTER_RADIUS, 0, width - 1))
                if y1 < y0 or x1 < x0:
                    continue
                ys = np.arange(y0, y1 + 1, dtype=np.float32) + 0.5
                xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
                grid_y = ys[:, None]
                grid_x = xs[None, :]
                # 位置到 GT 四边的距离（格子单位）
                dist_l = np.broadcast_to(grid_x - gx1[index], (ys.size, xs.size))
                dist_t = np.broadcast_to(grid_y - gy1[index], (ys.size, xs.size))
                dist_r = np.broadcast_to(gx2[index] - grid_x, (ys.size, xs.size))
                dist_b = np.broadcast_to(gy2[index] - grid_y, (ys.size, xs.size))
                lr = np.stack([dist_l, dist_r])
                tb = np.stack([dist_t, dist_b])
                centerness = np.sqrt(
                    np.clip(
                        (lr.min(0) / np.maximum(lr.max(0), 1e-6))
                        * (tb.min(0) / np.maximum(tb.max(0), 1e-6)),
                        0,
                        None,
                    )
                )
                region = (slice(y0, y1 + 1), slice(x0, x1 + 1))
                cls_np[b][region] = gt_labels[index]
                ltrb_np[b][region + (slice(None),)] = np.stack(
                    [dist_l, dist_t, dist_r, dist_b], axis=-1
                )
                ctr_np[b][region] = centerness
        results.append(
            {
                "cls_target": torch.from_numpy(cls_np).to(device),
                "ltrb_target": torch.from_numpy(ltrb_np).to(device),
                "ctr_target": torch.from_numpy(ctr_np).to(device),
            }
        )
    return results


class DetectionCriterion(torch.nn.Module):
    """检测 + 识别联合损失。"""

    def __init__(
        self,
        num_classes: int = 3,
        strides: tuple[int, ...] = (8, 16, 32),
        class_weights: tuple[float, ...] = (1.0, 1.5, 2.0),
        alpha: float = 0.25,
        gamma: float = 2.0,
        reg_weight: float = 1.0,
        ctr_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.ctr_weight = ctr_weight
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32),
        )

    def forward(
        self,
        outputs: dict[str, list[torch.Tensor]],
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cls_logits = outputs["cls_logits"]
        reg_preds = outputs["reg_preds"]
        ctr_logits = outputs["ctr_logits"]
        feature_sizes = [tuple(logits.shape[-2:]) for logits in cls_logits]
        # 目标分配放到 CPU：这段是逐 GT 的小张量操作，在 GPU 上 kernel
        # launch 开销会盖过一切（实测 644 ms vs CPU 上的十几 ms）。
        targets = assign_targets(
            boxes.detach().cpu(), labels.detach().cpu(), feature_sizes
        )
        device = cls_logits[0].device
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]
        batch = boxes.shape[0]

        zero = cls_logits[0].new_zeros(())
        total_cls = zero
        total_reg = zero
        total_ctr = zero
        num_positives = 0
        for level, target in enumerate(targets):
            logits = cls_logits[level]
            cls_target = target["cls_target"]
            positive = (cls_target > 0).unsqueeze(1)
            onehot = torch.zeros_like(logits)
            # 先减 1 再 clamp：背景位置的 -1 必须先 clamp 掉，
            # scatter_ 收到负索引是未定义行为。
            onehot.scatter_(
                1, (cls_target.long() - 1).clamp(min=0).unsqueeze(1), 1.0
            )
            onehot = onehot * positive
            total_cls = total_cls + sigmoid_focal_loss(
                logits,
                onehot,
                self.alpha,
                self.gamma,
                self.class_weights,
            )

            total_ctr = total_ctr + F.binary_cross_entropy_with_logits(
                ctr_logits[level].squeeze(1),
                target["ctr_target"],
                reduction="sum",
            )

            mask = cls_target > 0
            if mask.any():
                num_positives += int(mask.sum())
                positions = mask.nonzero(as_tuple=False)
                b = positions[:, 0]
                rows = positions[:, 1]  # y
                cols = positions[:, 2]  # x
                # 用扁平索引取位置，避免 4D 张量上混用张量索引的维度歧义
                # （[b, c][rows, cols] 会把 batch 维也展开成 N）。
                height, width = logits.shape[-2:]
                flat_index = b * (height * width) + rows * width + cols
                reg_flat = reg_preds[level].reshape(batch, 4, -1)
                pred_ltrb = F.softplus(
                    torch.stack(
                        [reg_flat[:, c].flatten()[flat_index] for c in range(4)],
                        dim=1,
                    )
                ) * REG_SCALE
                # ltrb_target 是 (B, H, W, 4)：必须先把通道维 permute 到
                # 前面再展平，否则 (H, W, 4) 的内存布局被当成 (4, H*W)，
                # 取出来的目标框是错位/退化的 → GIoU 恒为 0，回归学不动。
                target_flat = (
                    target["ltrb_target"].permute(0, 3, 1, 2).reshape(batch, 4, -1)
                )
                target_ltrb = torch.stack(
                    [target_flat[:, c].flatten()[flat_index] for c in range(4)],
                    dim=1,
                )
                cx = cols.float() + 0.5
                cy = rows.float() + 0.5
                pred_xyxy = torch.stack(
                    [
                        cx - pred_ltrb[:, 0],
                        cy - pred_ltrb[:, 1],
                        cx + pred_ltrb[:, 2],
                        cy + pred_ltrb[:, 3],
                    ],
                    dim=1,
                )
                target_xyxy = torch.stack(
                    [
                        cx - target_ltrb[:, 0],
                        cy - target_ltrb[:, 1],
                        cx + target_ltrb[:, 2],
                        cy + target_ltrb[:, 3],
                    ],
                    dim=1,
                )
                giou = generalized_box_iou(pred_xyxy, target_xyxy)
                total_reg = total_reg + (1 - giou).sum()

        # 归一化：正样本数（防 0）
        count = max(num_positives, 1)
        loss_cls = total_cls / count
        loss_reg = self.reg_weight * total_reg / count
        loss_ctr = self.ctr_weight * total_ctr / count
        return {
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "loss_ctr": loss_ctr,
            "loss": loss_cls + loss_reg + loss_ctr,
        }
