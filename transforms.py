import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Dict, Tuple


class CrowdHumanTransform:
    def __init__(self, target_size=(800, 800)):
        self.target_size = target_size

    def __call__(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # 获取原始尺寸
        orig_h, orig_w = image.shape[1:3]

        # 计算缩放比例
        scale_w = self.target_size[1] / orig_w
        scale_h = self.target_size[0] / orig_h

        # 调整图像大小
        image = F.resize(image, self.target_size, antialias=True)

        # 调整边界框坐标
        if 'loc' in target:
            boxes = target['loc']
            # 调整坐标
            boxes[:, [0, 2]] *= scale_w  # x coordinates
            boxes[:, [1, 3]] *= scale_h  # y coordinates

            # 确保坐标在有效范围内
            boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, self.target_size[1])
            boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, self.target_size[0])

            target['loc'] = boxes

        return image, target


class TrainTransform:
    def __init__(self, target_size=(800, 800)):
        self.transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.crowd_transform = CrowdHumanTransform(target_size)

    def __call__(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
        image, target = self.crowd_transform(image, target)
        image = self.transform(image)
        return image, target