import os

# 基础路径配置
BASE_DIR = "D:/ProgramData/PyCharm Community Edition 2024.3.5/PycharmProjects/PythonProject2/OpenDataLab___CrowdHuman"

# 数据集路径配置
DATASET_CONFIG = {
    'train': {
        'image_dir': os.path.join(BASE_DIR, "dsdl/dataset_root/Images"),
        'annotation_file': os.path.join(BASE_DIR, "dsdl/dataset_root/annotation_train.odgt"),
    },
    'val': {
        'image_dir': os.path.join(BASE_DIR, "dsdl/dataset_root/Images_val"),
        'annotation_file': os.path.join(BASE_DIR, "dsdl/dataset_root/annotation_val.odgt"),
    },
    'test': {
        'image_dir': os.path.join(BASE_DIR, "dsdl/dataset_root/images_test"),
        'annotation_file': None,  # 测试集可能没有标注文件
    }
}

# 模型配置
MODEL_CONFIG = {
    'num_classes': 2,  # 背景 + 人
    'input_size': (640, 640),  # 输入图像大小
    'batch_size': 16,
    'num_workers': 4,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'num_epochs': 10,
    'save_interval': 1000,  # 每多少批次保存一次检查点
}

# 训练配置
TRAIN_CONFIG = {
    'device': 'cuda',  # 或 'cpu'
    'checkpoint_dir': 'checkpoints',
    'best_model_dir': 'models',
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    'horizontal_flip_prob': 0.5,
    'brightness_range': (0.8, 1.2),
    'contrast_range': (0.8, 1.2),
}

# 创建必要的目录
def create_directories():
    dirs = [
        TRAIN_CONFIG['checkpoint_dir'],
        TRAIN_CONFIG['best_model_dir'],
        'logs'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)