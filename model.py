import torch
import torch.nn as nn
import torchvision.models as models


class SimpleDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleDetector, self).__init__()

        # 使用预训练的ResNet18作为特征提取器
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # 特征图大小
        self.feature_size = 512 * 2 * 2

        # 共享特征处理
        self.shared_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # 边界框回归头
        self.box_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.shared_fc, self.cls_head, self.box_head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # 提取特征
        features = self.features(x)

        # 共享特征处理
        shared = self.shared_fc(features)

        # 分类和边界框预测
        cls_pred = self.cls_head(shared)
        box_pred = self.box_head(shared)

        return cls_pred, box_pred

