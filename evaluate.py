import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from model import SimpleDetector
from dataset import DetectionDataset, collate_fn
import os
from datetime import datetime, UTC


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / max(union, 1e-6)


def apply_nms(boxes, scores, labels, iou_threshold=0.5):
    """标准NMS实现"""
    if len(boxes) == 0:
        return [], [], []

    # 按分数排序
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]

    keep = []

    while len(boxes) > 0:
        keep.append(0)
        if len(boxes) == 1:
            break

        ious = np.array([calculate_iou(boxes[0], box) for box in boxes[1:]])
        filtered = ious < iou_threshold

        boxes = boxes[1:][filtered]
        scores = scores[1:][filtered]
        labels = labels[1:][filtered]

    return boxes, scores, labels


def filter_predictions(boxes, scores, labels,
                       min_size=0.01, max_size=0.9,
                       min_aspect_ratio=0.2, max_aspect_ratio=5.0,
                       min_score=0.5):
    """基本的预测框过滤"""
    if len(boxes) == 0:
        return [], [], []

    keep = []
    for i, (box, score) in enumerate(zip(boxes, scores)):
        w = box[2] - box[0]
        h = box[3] - box[1]
        area = w * h
        aspect_ratio = w / h if h > 0 else float('inf')

        if (min_size < area < max_size and
                min_aspect_ratio < aspect_ratio < max_aspect_ratio and
                score >= min_score):
            keep.append(i)

    keep = np.array(keep)
    return (boxes[keep] if len(keep) > 0 else np.array([]),
            scores[keep] if len(keep) > 0 else np.array([]),
            labels[keep] if len(keep) > 0 else np.array([]))


def evaluate_model(model, data_loader, device, conf_threshold=0.5, nms_threshold=0.5):
    """评估模型性能"""
    model.eval()
    total_correct = 0
    total_predictions = 0
    total_targets = 0

    print(f"\nStarting evaluation at {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"- Confidence threshold: {conf_threshold}")
    print(f"- NMS threshold: {nms_threshold}")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = images.to(device)

            # 获取模型预测
            pred_classes, pred_boxes = model(images)
            pred_classes = torch.softmax(pred_classes, dim=-1)

            # 处理每张图片
            for i in range(len(targets)):
                img_pred_classes = pred_classes[i].cpu().numpy()
                img_pred_boxes = pred_boxes[i].cpu().numpy()
                target_boxes = targets[i]['boxes'].numpy()
                target_classes = targets[i]['classes'].numpy()

                # 获取预测的类别和置信度
                pred_scores = img_pred_classes.max(axis=1)
                pred_labels = img_pred_classes.argmax(axis=1)

                # 基本过滤
                boxes, scores, labels = filter_predictions(
                    img_pred_boxes, pred_scores, pred_labels,
                    min_score=conf_threshold
                )

                # 应用NMS
                if len(boxes) > 0:
                    boxes, scores, labels = apply_nms(
                        boxes, scores, labels,
                        iou_threshold=nms_threshold
                    )

                # 匹配预测和目标
                matched_pred = set()
                matched_target = set()

                # 遍历目标框
                for t_idx, target_box in enumerate(target_boxes):
                    best_iou = 0.5
                    best_pred = -1

                    for p_idx, pred_box in enumerate(boxes):
                        if p_idx in matched_pred:
                            continue

                        iou = calculate_iou(target_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_pred = p_idx

                    if best_pred != -1:
                        if labels[best_pred] == target_classes[t_idx]:
                            total_correct += 1
                        matched_pred.add(best_pred)
                        matched_target.add(t_idx)

                total_predictions += len(boxes)
                total_targets += len(target_boxes)

            # 显示进度和当前指标
            if batch_idx % 10 == 0:
                precision = total_correct / max(total_predictions, 1)
                recall = total_correct / max(total_targets, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-6)
                print(f"Batch {batch_idx}:")
                print(f"- Current precision: {precision:.4f}")
                print(f"- Current recall: {recall:.4f}")
                print(f"- Current F1: {f1:.4f}")

    # 计算最终指标
    precision = total_correct / max(total_predictions, 1)
    recall = total_correct / max(total_targets, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct': total_correct,
        'predictions': total_predictions,
        'targets': total_targets
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # 创建和加载模型
        model = SimpleDetector(num_classes=2)
        model_path = 'checkpoints/best_model.pth'
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
        else:
            raise FileNotFoundError("Model checkpoint not found")

        # 创建数据集和加载器
        test_dataset = DetectionDataset(
            'D:/ProgramData/PyCharm Community Edition 2024.3.5/PycharmProjects/PythonProject2/OpenDataLab___CrowdHuman/dsdl/dataset_root/Images',
            'D:/ProgramData/PyCharm Community Edition 2024.3.5/PycharmProjects/PythonProject2/OpenDataLab___CrowdHuman/dsdl/dataset_root/annotation_val.odgt',
            max_boxes=5
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        # 评估模型
        metrics = evaluate_model(model, test_loader, device)

        # 输出结果
        current_time = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\nEvaluation completed at: {current_time}")
        print(f"User: {os.getlogin()}")
        print("\nFinal Results:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Total correct detections: {metrics['correct']}")
        print(f"Total predictions: {metrics['predictions']}")
        print(f"Total targets: {metrics['targets']}")

        # 保存结果
        results_dir = 'evaluation_results'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(
            results_dir,
            f'results_{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}.txt'
        )

        with open(results_file, 'w') as f:
            f.write("Evaluation Results\n")
            f.write("=================\n")
            f.write(f"Date: {current_time}\n")
            f.write(f"User: {os.getlogin()}\n\n")
            f.write("Metrics:\n")
            f.write(f"- Precision: {metrics['precision']:.4f}\n")
            f.write(f"- Recall: {metrics['recall']:.4f}\n")
            f.write(f"- F1 Score: {metrics['f1']:.4f}\n\n")
            f.write("Counts:\n")
            f.write(f"- Correct detections: {metrics['correct']}\n")
            f.write(f"- Total predictions: {metrics['predictions']}\n")
            f.write(f"- Total targets: {metrics['targets']}\n")

        print(f"\nResults saved to: {results_file}")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")


if __name__ == '__main__':
    main()