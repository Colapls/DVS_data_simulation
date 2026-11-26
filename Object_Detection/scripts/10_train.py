#!/usr/bin/env python3
"""训练入口：体素网格 → 目标框 + 三分类（检测 + 识别同训）。

    python scripts/10_train.py --config configs/train/light_detector.yaml

每轮在 val 上评估 mAP@0.5 并保存 checkpoint；实验记录追加到
reports/experiment_log.csv（方案 §4.3 的训练纪律）。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dvsdet.augment.event_augment import EventAugment  # noqa: E402
from dvsdet.datasets.carla_dvs import build_dataloader  # noqa: E402
from dvsdet.datasets.label_map import CLASS_NAMES  # noqa: E402
from dvsdet.eval.detection_eval import (  # noqa: E402
    build_coco_ground_truth,
    detections_to_coco,
    evaluate_coco,
    format_report,
)
from dvsdet.models.criterion import DetectionCriterion  # noqa: E402
from dvsdet.models.detector import build_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"划分文件不存在：{path}")
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    grid_width: int,
    grid_height: int,
    samples: list[Path],
    score_threshold: float = 0.01,
) -> dict:
    """在 val 上跑一遍并做 COCO 评估。"""
    model.eval()
    ground_truth = build_coco_ground_truth(samples, grid_width, grid_height)
    image_ids = [image["id"] for image in ground_truth["images"]]
    detections: list[dict] = []
    offset = 0
    for batch in loader:
        voxel = batch["voxel"].to(device, non_blocking=True)
        results = model.predict(
            voxel,
            score_threshold=score_threshold,
            image_width=grid_width,
            image_height=grid_height,
        )
        for result in results:
            # predict 已返回图像像素坐标（xyxy），不再缩放
            detections.append(
                {
                    "boxes": result["boxes"].cpu().numpy(),
                    "scores": result["scores"].cpu().numpy(),
                    "labels": result["labels"].cpu().numpy(),
                }
            )
    return evaluate_coco(ground_truth, detections_to_coco(detections, image_ids))


def main() -> int:
    # Windows 控制台默认 GBK，中文/符号打印会炸——统一 UTF-8 容错
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="训练轻量事件检测器")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="覆盖配置里的轮数")
    parser.add_argument("--batch-size", type=int, help="覆盖配置里的批大小")
    parser.add_argument("--lr", type=float, help="覆盖配置里的学习率")
    parser.add_argument(
        "--experiment",
        help="实验名（输出目录 runs/<experiment>，网页端按它读训练进度）",
    )
    parser.add_argument("--resume", type=Path, help="从 checkpoint 续训")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(config.get("seed", 42)))

    data_cfg = config["data"]
    train_cfg = config.get("train", {})
    grid_width = int(data_cfg.get("grid_width", 640))
    grid_height = int(data_cfg.get("grid_height", 360))
    in_channels = 2 * int(data_cfg.get("num_time_channels", 9))

    train_samples = read_split(ROOT / data_cfg["split_file"])
    val_samples = read_split(ROOT / data_cfg["val_split_file"])
    print(f"train {len(train_samples)} 窗 / val {len(val_samples)} 窗", flush=True)

    augment = EventAugment(
        temporal_jitter_ratio=float(train_cfg.get("augment", {}).get("temporal_jitter_ratio", 0.1)),
        event_drop_ratio=tuple(train_cfg.get("augment", {}).get("event_drop_ratio", (0.1, 0.3))),
        noise_scale=float(train_cfg.get("augment", {}).get("noise_scale", 1.0)),
        geometry=tuple(train_cfg.get("augment", {}).get("geometry", ("hflip", "translate", "scale"))),
        seed=int(config.get("seed", 42)),
    )
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 16))
    learning_rate = float(args.lr or train_cfg.get("lr", 2e-4))
    train_loader = build_dataloader(
        train_samples,
        batch_size=batch_size,
        grid_size=(grid_height, grid_width),
        training=True,
        augment=augment,
        workers=int(train_cfg.get("workers", 4)),
    )
    val_loader = build_dataloader(
        val_samples,
        batch_size=batch_size,
        grid_size=(grid_height, grid_width),
        training=False,
        workers=int(train_cfg.get("workers", 4)),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not train_cfg.get("force_cpu") else "cpu"
    )
    print(f"设备：{device}" + (f"（{torch.cuda.get_device_name(0)}）" if device.type == "cuda" else ""))

    model = build_model(
        {
            "in_channels": in_channels,
            "num_classes": len(CLASS_NAMES),
            "width": int(config.get("model", {}).get("width", 64)),
        }
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{parameter_count/1e6:.2f} M", flush=True)

    criterion = DetectionCriterion(
        num_classes=len(CLASS_NAMES),
        class_weights=tuple(data_cfg.get("class_weights", (1.0, 1.5, 2.0))),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(args.epochs or train_cfg.get("epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )
    print(
        f"超参：epochs={epochs} batch={batch_size} lr={learning_rate:g}",
        flush=True,
    )

    experiment = args.experiment or config.get("experiment_name", "light_detector")
    run_dir = ROOT / "runs" / experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0
    history: list[dict] = []

    start_epoch = 0
    if args.resume and args.resume.is_file():
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_map = float(checkpoint.get("best_map", -1.0))
        # 历史曲线接上，别把之前的轮次从 history.json 里抹掉
        if (run_dir / "history.json").is_file():
            try:
                history = json.loads(
                    (run_dir / "history.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                history = []
        print(f"从 {args.resume} 续训（epoch {start_epoch}）", flush=True)
    # 续训时把 cosine 调度器推到对应位置：否则 lr 会从初始值重新开始，
    # 等于给训练中间硬插了一次 warm restart。
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
        for group in optimizer.param_groups:
            group["lr"] = scheduler.get_last_lr()[0]
        print(f"续训学习率：{optimizer.param_groups[0]['lr']:.2e}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        started = time.perf_counter()
        totals: dict[str, float] = {}
        batches = 0
        oom_count = 0
        for batch in train_loader:
            voxel = batch["voxel"].to(device, non_blocking=True)
            boxes = batch["boxes"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            try:
                outputs = model(voxel)
                losses = criterion(outputs, boxes, labels)
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            except torch.cuda.OutOfMemoryError:
                # 显存被其他任务挤占时会 OOM：清缓存并跳过这个 batch，
                # 连续 OOM 太多次就中止（否则白跑）
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                oom_count += 1
                print(f"显存不足，跳过该 batch（累计 {oom_count} 次）", flush=True)
                if oom_count >= 20:
                    raise RuntimeError("显存持续不足，训练中止")
                continue
            batches += 1
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        scheduler.step()
        elapsed = time.perf_counter() - started

        metrics = evaluate(
            model, val_loader, device, grid_width, grid_height, val_samples
        )
        train_losses = {key: value / max(batches, 1) for key, value in totals.items()}
        record = {
            "epoch": epoch,
            "seconds": round(elapsed, 1),
            **{f"train_{k}": round(v, 4) for k, v in train_losses.items()},
            "val_mAP@0.5": round(metrics["mAP"].get("@0.5", 0.0), 4),
            "val_mAP@0.75": round(metrics["mAP"].get("@0.75", 0.0), 4),
            "per_class_ap": {k: round(v, 4) for k, v in metrics["per_class_ap"].items()},
            "per_class_recall": {
                k: round(v, 4) for k, v in metrics["per_class_recall"].items()
            },
        }
        history.append(record)
        print(
            f"epoch {epoch:3d} ({elapsed:5.1f}s) "
            f"loss={train_losses.get('loss', 0):.4f} "
            f"cls={train_losses.get('loss_cls', 0):.3f} "
            f"reg={train_losses.get('loss_reg', 0):.3f} "
            f"| val mAP@0.5={record['val_mAP@0.5']:.4f} "
            f"mAP@0.75={record['val_mAP@0.75']:.4f}"
        , flush=True)
        print(
            "    每类召回："
            + " ".join(
                f"{name}={value:.3f}"
                for name, value in metrics["per_class_recall"].items()
            )
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
                "config": config,
            },
            run_dir / "last.pth",
        )
        if metrics["mAP"].get("@0.5", 0.0) > best_map:
            best_map = metrics["mAP"]["@0.5"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_map": best_map,
                    "config": config,
                },
                run_dir / "best.pth",
            )
        (run_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    final = evaluate(model, val_loader, device, grid_width, grid_height, val_samples)
    print("\n最终 val 指标：", flush=True)
    print(format_report(final), flush=True)
    (run_dir / "final_metrics.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 实验记录表（方案 §4.3）
    log_path = ROOT / "reports" / "experiment_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.is_file()
    with log_path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        if write_header:
            writer.writerow(
                [
                    "experiment_name",
                    "epochs",
                    "val_mAP@0.5",
                    "val_mAP@0.75",
                    "car_recall",
                    "motorcycle_recall",
                    "pedestrian_recall",
                    "params_M",
                    "device",
                    "note",
                ]
            )
        writer.writerow(
            [
                experiment,
                epochs,
                round(final["mAP"].get("@0.5", 0.0), 4),
                round(final["mAP"].get("@0.75", 0.0), 4),
                round(final["per_class_recall"].get("car", 0.0), 4),
                round(final["per_class_recall"].get("motorcycle", 0.0), 4),
                round(final["per_class_recall"].get("pedestrian", 0.0), 4),
                round(parameter_count / 1e6, 2),
                str(device),
                "自建轻量检测器（体素网格 + FCOS 风格头）",
            ]
        )
    print(f"\n实验记录已追加：{log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
