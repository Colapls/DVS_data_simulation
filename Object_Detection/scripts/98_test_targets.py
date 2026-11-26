#!/usr/bin/env python3
"""验证正样本分配与回归目标的正确性（单元级，不依赖数据集）。

目标：确认 assign_targets 产出的 ltrb_target 真的等于"该位置到 GT 框
四边的距离"、cls_target 是正确类别、ctr_target 在框中心最大。
这是训练能不能学起来的地基——回归目标错了，训多久都学不会。

    python scripts/98_test_targets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dvsdet.models.criterion import assign_targets  # noqa: E402


def test_ltrb_target_geometry() -> None:
    """ltrb_target 在正样本位置必须等于到 GT 四边的真实距离。"""
    # 1 个 GT：归一化 xyxy (0.4, 0.3, 0.7, 0.7)（cxcywh 格式输入）
    cx, cy, bw, bh = 0.55, 0.5, 0.3, 0.4
    boxes = torch.tensor([[[cx, cy, bw, bh]]], dtype=torch.float32)
    labels = torch.tensor([[2]], dtype=torch.long)
    height, width = 45, 80
    targets = assign_targets(boxes, labels, [(height, width)])
    target = targets[0]

    # 取 GT 中心最近的整数格子
    center_x = cx * width
    center_y = cy * height
    px = int(round(center_x - 0.5))
    py = int(round(center_y - 0.5))
    ltrb = target["ltrb_target"][0, py, px]
    # 期望：中心到四边的距离（格子单位）
    gx1, gy1 = (cx - bw / 2) * width, (cy - bh / 2) * height
    gx2, gy2 = (cx + bw / 2) * width, (cy + bh / 2) * height
    expect = torch.tensor(
        [(px + 0.5) - gx1, (py + 0.5) - gy1, gx2 - (px + 0.5), gy2 - (py + 0.5)]
    )
    assert torch.allclose(ltrb, expect, atol=0.51), (
        f"ltrb_target 不符：实际 {ltrb.tolist()}，期望 {expect.tolist()}"
    )
    assert target["cls_target"][0, py, px] == 3, (
        f"cls_target 应为类别+1=3，实际 {int(target['cls_target'][0,py,px])}"
    )
    assert target["ctr_target"][0, py, px] > 0.9, (
        f"框中心的 centerness 应接近 1，实际 {float(target['ctr_target'][0,py,px]):.3f}"
    )
    print(
        f"OK  ltrb_target 几何正确：中心格子 ltrb={ltrb.tolist()} "
        f"（期望 {expect.tolist()}），cls=3，centerness="
        f"{float(target['ctr_target'][0,py,px]):.3f}"
    )


def test_background_is_untouched() -> None:
    """GT 之外的格子必须是背景（cls=0，ltrb=0，centerness=0）。"""
    boxes = torch.tensor([[[0.55, 0.5, 0.3, 0.4]]], dtype=torch.float32)
    labels = torch.tensor([[2]], dtype=torch.long)
    targets = assign_targets(boxes, labels, [(45, 80)])
    cls_target = targets[0]["cls_target"]
    ltrb = targets[0]["ltrb_target"]
    ctr = targets[0]["ctr_target"]
    positive = cls_target > 0
    assert int(positive.sum()) > 0
    assert float(ltrb[~positive].abs().max()) == 0.0, "背景位置的 ltrb 应为 0"
    assert float(ctr[~positive].max()) == 0.0, "背景位置的 centerness 应为 0"
    print(
        f"OK  背景保持干净：{int(positive.sum())} 个正样本位置，"
        f"其余位置 ltrb/centerness 全为 0"
    )


def test_edge_boxes() -> None:
    """贴边的框：正样本不能越界，四边距离仍要对。"""
    # 框贴左上角：cxcywh = (0.1, 0.1, 0.2, 0.2)
    boxes = torch.tensor([[[0.1, 0.1, 0.2, 0.2]]], dtype=torch.float32)
    labels = torch.tensor([[1]], dtype=torch.long)
    targets = assign_targets(boxes, labels, [(45, 80)])
    cls_target = targets[0]["cls_target"]
    positive = cls_target > 0
    assert int(positive.sum()) > 0, "贴边框也应产生正样本"
    # 正样本位置的 ltrb 全部非负（位置在框内）
    ltrb = targets[0]["ltrb_target"][positive]
    assert float(ltrb.min()) >= -0.01, (
        f"贴边框的 ltrb 出现负值：{float(ltrb.min()):.3f}"
    )
    # 类别正确
    assert set(cls_target[positive].unique().tolist()) == {2}, (
        f"贴边框类别错：{cls_target[positive].unique().tolist()}"
    )
    print(f"OK  贴边框：{int(positive.sum())} 个正样本，ltrb 非负，类别正确")


def test_cxcywh_contract() -> None:
    """契约检查：输入按 cxcywh 解释（传 xyxy 会得到不同结果）。

    把这一条钉死，是为了让"格式传错"这种错误在测试里就暴露，
    而不是表现为"训了 50 轮 mAP 还是 0"。
    """
    cx, cy, bw, bh = 0.55, 0.5, 0.3, 0.4
    labels = torch.tensor([[2]], dtype=torch.long)
    as_cxcywh = torch.tensor([[[cx, cy, bw, bh]]], dtype=torch.float32)
    as_xyxy = torch.tensor(
        [[[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]]],
        dtype=torch.float32,
    )
    first = assign_targets(as_cxcywh, labels, [(45, 80)])[0]
    second = assign_targets(as_xyxy, labels, [(45, 80)])[0]
    # 契约是 cxcywh：两种输入必须给出不同结果（说明确实按 cxcywh 解读，
    # 而不是恰好相同导致测试无效）
    assert not torch.allclose(
        first["ltrb_target"], second["ltrb_target"], atol=1e-3
    ), "两种格式产出相同结果，测试无法区分格式"
    # 且 cxcywh 输入的结果必须与手工算的 xyxy 一致
    height, width = 45, 80
    gx1, gy1 = (cx - bw / 2) * width, (cy - bh / 2) * height
    gx2, gy2 = (cx + bw / 2) * width, (cy + bh / 2) * height
    px, py = int(round(cx * width - 0.5)), int(round(cy * height - 0.5))
    ltrb = first["ltrb_target"][0, py, px]
    expect = torch.tensor(
        [(px + 0.5) - gx1, (py + 0.5) - gy1, gx2 - (px + 0.5), gy2 - (py + 0.5)]
    )
    assert torch.allclose(ltrb, expect, atol=0.51), (
        f"cxcywh 输入被解读错了：{ltrb.tolist()} vs {expect.tolist()}"
    )
    print("OK  输入契约是归一化 cxcywh（传 xyxy 会得到不同结果，格式错能被测出）")


def test_reg_flatten_matches_permute() -> None:
    """训练端取回归目标用的 reshape 与 permute 必须等价。

    ltrb_target 是 (B,H,W,4)，直接 reshape(B,4,-1) 会按错误内存布局解释
    （H/W 交错在通道维之前），目标框退化成点 → GIoU 恒为 0，回归学不动。
    """
    batch, height, width = 2, 45, 80
    target = torch.rand(batch, height, width, 4)
    wrong = target.reshape(batch, 4, -1)
    right = target.permute(0, 3, 1, 2).reshape(batch, 4, -1)
    assert not torch.allclose(wrong, right), "两种取法结果相同，测试没意义"
    # 正确取法：right[:, c] 的第 (y*width+x) 项就是 target[b, y, x, c]
    for b in range(batch):
        for c in range(4):
            assert torch.isclose(
                right[b, c, 10 * width + 20], target[b, 10, 20, c]
            ), "permute 后的索引与原始 (b,y,x,c) 不对应"
    print("OK  permute 取法与原始 (b,y,x,c) 索引一一对应（reshape 直取是错的）")


def main() -> int:
    torch.manual_seed(0)
    test_ltrb_target_geometry()
    test_background_is_untouched()
    test_edge_boxes()
    test_cxcywh_contract()
    test_reg_flatten_matches_permute()
    print("\n正样本分配与回归目标自检全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
