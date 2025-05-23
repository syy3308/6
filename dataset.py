import torch
from torch.utils.data import Dataset
import cv2
import json
import os
import numpy as np
from torchvision import transforms
from PIL import Image


class DetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, max_boxes=5, is_training=True):
        self.image_dir = image_dir
        self.max_boxes = max_boxes
        self.annotations = []
        self.is_training = is_training

        # 训练集和验证集使用不同的变换
        if is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        # 验证目录和文件
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        # 加载图片文件
        print("Scanning image directory...")
        self.image_files = {}
        try:
            for img_file in os.listdir(image_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_id = os.path.splitext(img_file)[0].split(',')[0]
                    self.image_files[img_id] = img_file
            print(f"Found {len(self.image_files)} image files")
        except Exception as e:
            print(f"Error scanning image directory: {e}")
            raise

        # 加载标注
        print("Loading annotations...")
        print(f"Reading annotation file: {annotation_file}")

        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue

                        anno = json.loads(line)
                        img_id = str(anno.get('ID', '')).split(',')[0]

                        if img_id not in self.image_files:
                            continue

                        gtboxes = anno.get('gtboxes', [])
                        valid_boxes = []

                        # 读取图像尺寸
                        img_path = os.path.join(image_dir, self.image_files[img_id])
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        img_height, img_width = img.shape[:2]

                        for box in gtboxes:
                            # 只处理person标签且不忽略的框
                            if (box.get('tag') != 'person' or
                                    box.get('head_attr', {}).get('ignore', 0) == 1):
                                continue

                            # 优先使用全身框
                            if 'fbox' in box:
                                x1, y1, w, h = box['fbox']
                                x2, y2 = x1 + w, y1 + h
                            elif 'hbox' in box:
                                x1, y1, w, h = box['hbox']
                                x2, y2 = x1 + w, y1 + h
                            else:
                                continue

                            # 确保坐标在图像范围内
                            x1 = max(0, min(x1, img_width))
                            y1 = max(0, min(y1, img_height))
                            x2 = max(0, min(x2, img_width))
                            y2 = max(0, min(y2, img_height))

                            # 验证框的有效性
                            if x2 <= x1 or y2 <= y1:
                                continue

                            # 过滤太小的框
                            box_area = (x2 - x1) * (y2 - y1)
                            img_area = img_width * img_height
                            if box_area < 0.001 * img_area:
                                continue

                            valid_boxes.append({
                                'box': [x1, y1, x2, y2],
                                'tag': 'person'
                            })

                        if valid_boxes:  # 只保存有有效框的图像
                            self.annotations.append({
                                'image_id': img_id,
                                'boxes': valid_boxes,
                                'width': img_width,
                                'height': img_height
                            })

                        if line_num % 1000 == 0:
                            print(f"Processed {line_num} lines, found {len(self.annotations)} valid annotations")

                    except Exception as e:
                        print(f"Error processing line {line_num}: {e}")
                        continue

        except Exception as e:
            print(f"Error reading annotation file: {e}")
            raise

        print(f"\nLoaded {len(self.annotations)} valid annotations")
        if len(self.annotations) == 0:
            raise ValueError("No valid annotations found!")

        self._validate_dataset()

    def _validate_dataset(self):
        """验证数据集的完整性"""
        print("\nValidating dataset...")
        total_boxes = sum(len(anno['boxes']) for anno in self.annotations)
        print(f"Total images: {len(self.annotations)}")
        print(f"Total boxes: {total_boxes}")
        print(f"Average boxes per image: {total_boxes / len(self.annotations):.2f}")

        # 显示一些示例标注
        print("\nSample annotations:")
        for i in range(min(3, len(self.annotations))):
            anno = self.annotations[i]
            print(f"\nImage {anno['image_id']}:")
            print(f"Image size: {anno['width']}x{anno['height']}")
            for j, box in enumerate(anno['boxes']):
                x1, y1, x2, y2 = box['box']
                w, h = x2 - x1, y2 - y1
                print(f"Box {j}: pos=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], size={w:.1f}x{h:.1f}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        img_id = anno['image_id']
        img_path = os.path.join(self.image_dir, self.image_files[img_id])

        # 读取图像
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = np.zeros((64, 64, 3), dtype=np.uint8)

        orig_h, orig_w = image.shape[:2]

        # 处理边界框
        boxes = []
        classes = []

        for box_info in anno['boxes'][:self.max_boxes]:
            x1, y1, x2, y2 = box_info['box']

            # 归一化坐标
            x1, x2 = x1 / orig_w, x2 / orig_w
            y1, y2 = y1 / orig_h, y2 / orig_h

            # 确保坐标在[0,1]范围内
            x1, x2 = max(0, min(1, x1)), max(0, min(1, x2))
            y1, y2 = max(0, min(1, y1)), max(0, min(1, y2))

            if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
                boxes.append([x1, y1, x2, y2])
                classes.append(1)  # person class

        # 确保至少有一个框
        if not boxes:
            boxes = [[0.2, 0.2, 0.8, 0.8]]
            classes = [0]  # background class

        # 转换为tensor
        try:
            image = self.transform(image)
        except Exception as e:
            print(f"Transform error for image {img_path}: {e}")
            image = torch.zeros((3, 64, 64))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)

        return image, {'boxes': boxes, 'classes': classes}


def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images)
    return images, targets