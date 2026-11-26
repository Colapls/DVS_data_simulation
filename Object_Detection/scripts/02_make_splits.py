#!/usr/bin/env python3
"""生成 train/val/test 划分（方案 §3.2 的防泄漏红线）。

**按序列划分，绝不按帧随机切**——同一序列的相邻窗高度相关，按帧切会让
train/val 泄漏，指标虚高。

两种用法：

固定模式（默认）：手工指定 val/test 各一条序列，其余全部训练。

    python scripts/02_make_splits.py --val-sequence sequence_000021 \
        --test-sequence sequence_000026

比例模式（网页端用）：用户选序列范围 + 划分比例，脚本把**整条序列**
贪心分配到三个 split，使窗数比例最接近设定值——比例作用在序列粒度上，
序列绝不跨 split。

    python scripts/02_make_splits.py \
        --sequences sequence_000019,sequence_000020,sequence_000021,sequence_000022 \
        --ratios 70,15,15 --seed 0

默认排除"无行人"序列（修复行人生成 bug 之前采的那批），可用
--keep-zero-pedestrian 保留。
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def scan(slices_root: Path) -> dict[str, list[Path]]:
    """按序列聚合切片样本路径。

    目录层级是 slices/<sequence_id>/<camera_id>/<win>.npz，
    所以序列名在 parents[1]（parents[2] 是 slices 根目录本身）。
    """
    per_sequence: dict[str, list[Path]] = defaultdict(list)
    for npz in sorted(slices_root.rglob("*.npz")):
        per_sequence[npz.parents[1].name].append(npz)
    return dict(per_sequence)


def sequence_stats(samples: list[Path]) -> dict:
    """统计一条序列的地图/天气与各类框数（只读一遍文件）。"""
    counter: dict[int, int] = defaultdict(int)
    info: dict = {}
    for sample in samples:
        with np.load(sample, allow_pickle=True) as data:
            if not info:
                info = {
                    "map": str(data["map_name"]),
                    "weather": str(data["weather"]),
                    "sequence_id": str(data["sequence_id"]),
                }
            for label in data["labels"]:
                counter[int(label)] += 1
    info["classes"] = dict(counter)
    info["windows"] = len(samples)
    return info


def split_by_ratio(
    sequences: list[str],
    stats: dict[str, dict],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[str]]:
    """把序列分到 train/val/test，窗数比例最接近 ratios。

    **按「地图+天气」分组再分配**：同一组（同一地图同一天气）的序列
    绝不跨 split，否则同场景同天气泄漏、指标虚高（方案 §3.2 红线）。

    组内子集用 0/1 背包的 DP 求解：状态 = 累计窗数，保留"用组数最少、
    组序最小"的解，目标是让窗数离目标最近（窗数都是整百，DP 状态很少，
    精确解不贵）。
    """
    groups: dict[tuple[str, str], list[str]] = {}
    for seq in sequences:
        key = (stats[seq]["map"], stats[seq]["weather"])
        groups.setdefault(key, []).append(seq)
    items = [
        (key, sorted(seqs), sum(stats[s]["windows"] for s in seqs))
        for key, seqs in groups.items()
    ]
    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=lambda item: item[0])  # 稳定排序，seed 只影响同键情况

    total = sum(item[2] for item in items)
    target_test = total * ratios[2] / 100.0
    target_val = total * ratios[1] / 100.0

    def best_subset(pool, target):
        """在 pool 里挑窗数最接近 target 的子集（DP，精确解）。"""
        # sum -> (用组数, 组下标列表)
        table: dict[int, tuple[int, list[int]]] = {0: (0, [])}
        for index, (_key, _seqs, windows) in enumerate(pool):
            for total_windows, (count, picked) in list(table.items()):
                new_sum = total_windows + windows
                candidate = (count + 1, picked + [index])
                current = table.get(new_sum)
                if current is None or candidate[0] < current[0]:
                    table[new_sum] = candidate
        if not pool:
            return []
        usable = [(abs(s - target), n, s, picked) for s, (n, picked) in table.items() if s > 0]
        if not usable:
            return []
        usable.sort()
        return [pool[i] for i in usable[0][3]]

    test_groups = best_subset(items, target_test)
    remaining = [item for item in items if item not in test_groups]
    val_groups = best_subset(remaining, target_val) if ratios[1] > 0 else []
    train_groups = [item for item in remaining if item not in val_groups]

    if not train_groups:  # 比例过于极端时的兜底
        if val_groups:
            train_groups = [val_groups.pop()]
        elif test_groups:
            train_groups = [test_groups.pop()]

    def sequences_of(group_list):
        out: list[str] = []
        for _key, seqs, _w in group_list:
            out.extend(seqs)
        return sorted(out)

    return {
        "train": sequences_of(train_groups),
        "val": sequences_of(val_groups),
        "test": sequences_of(test_groups),
    }


def main() -> int:
    # Windows 控制台默认 GBK，打印 ✓/⚠ 会炸——统一改成 UTF-8 容错
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="生成 train/val/test 划分")
    parser.add_argument("--slices", type=Path, default=ROOT / "data" / "slices")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--val-sequence", default="sequence_000021", help="[固定模式] 验证集序列")
    parser.add_argument(
        "--test-sequence", default="sequence_000026", help="[固定模式] 测试集序列（冻结）"
    )
    parser.add_argument(
        "--sequences",
        default="all",
        help="[比例模式] 序列范围：逗号分隔的序列 ID，或 all（全部）",
    )
    parser.add_argument(
        "--ratios",
        default=None,
        help="[比例模式] train,val,test 的百分比，如 70,15,15（和不必为 100，"
        "按比例归一）；给了这个参数就走比例模式",
    )
    parser.add_argument("--seed", type=int, default=0, help="[比例模式] 随机种子")
    parser.add_argument(
        "--keep-zero-pedestrian",
        action="store_true",
        help="保留没有行人标注的序列（默认排除，避免稀释行人占比）",
    )
    args = parser.parse_args()

    per_sequence = scan(args.slices)
    if not per_sequence:
        print(f"{args.slices} 下没有切片样本，先跑 01_slice_events.py")
        return 1

    stats = {seq: sequence_stats(samples) for seq, samples in per_sequence.items()}
    print(f"切片目录共 {len(per_sequence)} 条序列：")
    for sequence_id, info in sorted(stats.items()):
        classes = info["classes"]
        pedestrian = classes.get(2, 0)
        mark = "  ← 无行人" if pedestrian == 0 else ""
        print(
            f"  {sequence_id}: {info['windows']:5d} 窗  "
            f"{info['map']:14s} {info['weather']:14s} "
            f"car={classes.get(0,0):6d} moto={classes.get(1,0):6d} ped={pedestrian:6d}{mark}"
        )

    excluded = set()
    if not args.keep_zero_pedestrian:
        excluded = {
            seq for seq, info in stats.items() if info["classes"].get(2, 0) == 0
        }
        if excluded:
            print(
                f"\n排除 {len(excluded)} 条无行人序列（--keep-zero-pedestrian 可保留）："
                + ", ".join(sorted(excluded))
            )

    use_ratio = args.ratios is not None
    if use_ratio:
        # ---- 比例模式：用户选范围 + 比例，整条序列贪心分配 ----
        if args.sequences.strip().lower() == "all":
            selected = sorted(per_sequence)
        else:
            selected = [
                s.strip() for s in args.sequences.split(",") if s.strip()
            ]
        missing = [s for s in selected if s not in per_sequence]
        if missing:
            print(f"不存在的序列：{', '.join(missing)}")
            return 1
        dropped = [s for s in selected if s in excluded]
        if dropped:
            print(f"范围内被排除的无行人序列：{', '.join(dropped)}")
            selected = [s for s in selected if s not in excluded]
        if not selected:
            print("范围内没有可用序列")
            return 1
        try:
            parts = [float(v) for v in args.ratios.split(",")]
        except ValueError:
            print(f"--ratios 格式不对：{args.ratios}（应为 70,15,15）")
            return 1
        if len(parts) != 3 or any(v < 0 for v in parts):
            print(f"--ratios 必须是三个非负数：{args.ratios}")
            return 1
        total_ratio = sum(parts)
        ratios = tuple(v / total_ratio * 100.0 for v in parts)
        assignment = split_by_ratio(selected, stats, ratios, args.seed)
        print(f"\n比例模式（train:val:test = {ratios[0]:.0f}:{ratios[1]:.0f}:{ratios[2]:.0f}）：")
        for name in ("train", "val", "test"):
            seqs = assignment[name]
            windows = sum(stats[s]["windows"] for s in seqs)
            print(f"  {name:5s}: {len(seqs)} 条序列 {windows} 窗  [{', '.join(seqs) or '空'}]")
    else:
        # ---- 固定模式：手工指定 val/test ----
        val_sequence = args.val_sequence
        test_sequence = args.test_sequence
        for name, sequence_id in (("val", val_sequence), ("test", test_sequence)):
            if sequence_id not in per_sequence:
                print(f"指定的 {name} 序列 {sequence_id} 不存在，请按上面的清单调整")
                return 1
            if sequence_id in excluded:
                print(
                    f"指定的 {name} 序列 {sequence_id} 属于无行人序列，"
                    "请换一个或加 --keep-zero-pedestrian"
                )
                return 1
        assignment = {
            "train": [
                seq
                for seq in sorted(per_sequence)
                if seq not in (val_sequence, test_sequence) and seq not in excluded
            ],
            "val": [val_sequence],
            "test": [test_sequence],
        }
        print(f"\n固定模式：val={val_sequence}，test={test_sequence}")

    splits = {
        name: [sample for seq in seqs for sample in per_sequence[seq]]
        for name, seqs in assignment.items()
    }

    args.out.mkdir(parents=True, exist_ok=True)
    # 把序列分配也存档，网页端与复现实验都能读到
    import json

    (args.out / "assignment.json").write_text(
        json.dumps(
            {
                "mode": "ratio" if use_ratio else "fixed",
                "ratios": list(ratios) if use_ratio else None,
                "train": assignment["train"],
                "val": assignment["val"],
                "test": assignment["test"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name, samples in splits.items():
        path = args.out / f"{name}.txt"
        path.write_text(
            "\n".join(str(sample) for sample in samples) + ("\n" if samples else ""),
            encoding="utf-8",
        )
        print(f"  写出 {name}: {len(samples)} 窗 → {path}")

    # ---- 防泄漏断言 ----
    train_seq = set(assignment["train"])
    val_seq = set(assignment["val"])
    test_seq = set(assignment["test"])
    assert not (train_seq & val_seq), "train/val 有序列重叠"
    assert not (train_seq & test_seq), "train/test 有序列重叠"
    assert not (val_seq & test_seq), "val/test 有序列重叠"
    print("\n✓ 三份划分的序列互不重叠")

    def groups_of(seqs: set[str]) -> set[tuple[str, str]]:
        return {(stats[s]["map"], stats[s]["weather"]) for s in seqs}

    train_groups = groups_of(train_seq)
    for name, seqs in (("val", val_seq), ("test", test_seq)):
        if not seqs:
            continue
        overlap = train_groups & groups_of(seqs)
        if overlap:
            print(
                f"⚠️  train 与 {name} 共享「地图+天气」组合 {sorted(overlap)}——"
                "同场景同天气泄漏风险，指标会虚高"
            )
        else:
            print(f"✓ train 与 {name} 无「地图+天气」重叠")

    # ---- 类别统计（从 sequence_stats 聚合，不再重读文件）----
    print("\n每类框数统计（方案 §3.4 DoD：行人:汽车 ≥ 1:1.5）：")
    for name, seqs in splits.items():
        counter: dict[int, int] = defaultdict(int)
        for seq in assignment[name]:
            for label, count in stats[seq]["classes"].items():
                counter[int(label)] += count
        total = sum(counter.values())
        parts = " ".join(f"{label}={counter[label]}" for label in sorted(counter))
        ratio = (
            f"，行人:汽车 = {counter[2]/counter[0]:.3f}"
            if counter.get(0) and counter.get(2)
            else ""
        )
        print(f"  {name:6s} 共 {total:7d} 框  {parts}{ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
