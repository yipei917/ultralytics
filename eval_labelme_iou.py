import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ultralytics import YOLO
from ultralytics.models import RTDETR


def normalize_label(label: str) -> str:
    return label.strip().lower()


def load_labelme_boxes(json_path: Path) -> List[Dict]:
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        points = shape.get("points", [])
        if len(points) != 2:
            continue
        (x1, y1), (x2, y2) = points
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        boxes.append(
            {
                "bbox": [x_min, y_min, x_max, y_max],
                "label": normalize_label(shape.get("label", "NG")),
            }
        )
    return boxes


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def greedy_match(
    preds: List[Dict], gts: List[Dict], iou_thr: float
) -> Tuple[List[Dict], List[int], List[int]]:
    matches: List[Dict] = []
    if not preds or not gts:
        return matches, list(range(len(preds))), list(range(len(gts)))

    iou_matrix = np.zeros((len(preds), len(gts)), dtype=float)
    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            if pred["label"] != gt["label"]:
                continue
            iou_matrix[i, j] = box_iou(pred["bbox"], gt["bbox"])

    unmatched_preds = set(range(len(preds)))
    unmatched_gts = set(range(len(gts)))

    while unmatched_preds and unmatched_gts:
        max_iou = 0.0
        max_pair = (-1, -1)
        for i in unmatched_preds:
            for j in unmatched_gts:
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_pair = (i, j)
        if max_iou < iou_thr:
            break
        i_idx, j_idx = max_pair
        matches.append(
            {"pred_idx": i_idx, "gt_idx": j_idx, "iou": max_iou}
        )
        unmatched_preds.remove(i_idx)
        unmatched_gts.remove(j_idx)

    return matches, list(unmatched_preds), list(unmatched_gts)


def collect_images(source: Path) -> List[Path]:
    supported = {".bmp", ".jpg", ".jpeg", ".png"}
    return sorted(
        [
            p
            for p in source.iterdir()
            if p.suffix.lower() in supported and p.is_file()
        ]
    )


def load_model(weights: Path, model_type: Optional[str] = None) -> Union[YOLO, RTDETR]:
    """加载模型，自动检测或使用指定的模型类型。
    
    Args:
        weights: 模型权重路径
        model_type: 模型类型 ('yolo', 'rtdetr', 或 None 自动检测)
    
    Returns:
        加载的模型对象
    """
    weights_str = str(weights)
    
    # 如果指定了模型类型，直接使用
    if model_type:
        model_type = model_type.lower()
        if model_type == 'rtdetr':
            return RTDETR(weights_str)
        elif model_type == 'yolo':
            return YOLO(weights_str)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}，支持 'yolo' 或 'rtdetr'")
    
    # 自动检测：根据路径名称判断
    weights_lower = weights_str.lower()
    if 'rtdetr' in weights_lower:
        try:
            return RTDETR(weights_str)
        except Exception:
            # 如果 RTDETR 加载失败，尝试 YOLO
            pass
    
    # 默认尝试 YOLO
    try:
        return YOLO(weights_str)
    except Exception as e:
        # 如果 YOLO 加载失败，尝试 RTDETR
        try:
            print(f"YOLO 加载失败: {e}，尝试使用 RTDETR")
            return RTDETR(weights_str)
        except Exception as e2:
            raise RuntimeError(f"无法加载模型 (YOLO: {e}, RTDETR: {e2})")


def evaluate(
    weights: Path,
    source_dir: Path,
    iou_thr: float,
    conf_thr: float,
    device: str,
    save_csv: Optional[Path],
    model_type: Optional[str] = None,
):
    model = load_model(weights, model_type)
    model_name = "RTDETR" if isinstance(model, RTDETR) else "YOLO"
    print(f"使用模型类型: {model_name}")
    image_paths = collect_images(source_dir)
    if not image_paths:
        raise FileNotFoundError(f"未在 {source_dir} 找到图像文件")

    summary_rows = []
    totals = {"pred": 0, "gt": 0, "tp": 0}

    for img_path in image_paths:
        result = model.predict(
            source=str(img_path),
            conf=conf_thr,
            device=device,
            verbose=False,
        )[0]

        preds = []
        names = result.names
        if result.boxes is not None:
            cls = result.boxes.cls.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            xyxy = result.boxes.xyxy.cpu().numpy()
            for b, c, s in zip(xyxy, cls, conf):
                class_idx = int(c)
                label = names.get(class_idx, str(class_idx))
                preds.append(
                    {
                        "bbox": b.tolist(),
                        "label": normalize_label(label),
                        "conf": float(s),
                    }
                )

        gt_path = img_path.with_suffix(".json")
        gts = load_labelme_boxes(gt_path)

        matches, unmatched_preds, unmatched_gts = greedy_match(
            preds, gts, iou_thr
        )
        ious = [m["iou"] for m in matches]
        row = {
            "image": img_path.name,
            "pred_count": len(preds),
            "gt_count": len(gts),
            "tp": len(matches),
            "fp": len(unmatched_preds),
            "fn": len(unmatched_gts),
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
        }
        summary_rows.append(row)
        totals["pred"] += len(preds)
        totals["gt"] += len(gts)
        totals["tp"] += len(matches)

    precision = (
        totals["tp"] / totals["pred"] if totals["pred"] > 0 else 0.0
    )
    recall = totals["tp"] / totals["gt"] if totals["gt"] > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print(
        f"\n总计: 图像 {len(image_paths)}, GT {totals['gt']}, "
        f"预测 {totals['pred']}, 命中 {totals['tp']}"
    )
    print(
        f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f} "
        f"(IoU阈值={iou_thr}, conf阈值={conf_thr})"
    )

    if save_csv:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        with save_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(summary_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"详细结果已写入 {save_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="对 Labelme 标注数据执行推理并计算 IoU"
    )
    parser.add_argument(
        "--weights", '-w',
        type=Path,
        default=Path(
            "/home/gdw/object_detection/ultralytics/train/"
            "local_yolo11n_e200/weights/best.pt"
        ),
        help="模型权重路径",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/gdw/object_detection/rack_data"),
        help="包含图像与 Labelme JSON 的目录",
    )
    parser.add_argument(
        "--iou-thr", type=float, default=0.5, help="IoU 匹配阈值"
    )
    parser.add_argument(
        "--conf-thr", type=float, default=0.25, help="置信度阈值"
    )
    parser.add_argument(
        "--device", type=str, default="0", help="推理使用的设备"
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="可选：保存逐图结果的 CSV 路径",
    )
    parser.add_argument(
        "--model-type", '-t',
        type=str,
        default=None,
        choices=["yolo", "rtdetr"],
        help="模型类型：'yolo' 或 'rtdetr'，不指定则自动检测",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        weights=args.weights,
        source_dir=args.source,
        iou_thr=args.iou_thr,
        conf_thr=args.conf_thr,
        device=args.device,
        save_csv=args.save_csv,
        model_type=args.model_type,
    )

