import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import math
from datetime import datetime, UTC
from model import SimpleDetector
from dataset import DetectionDataset, collate_fn
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
            return False


def validate(model, val_loader, device, config, cls_criterion, box_criterion):
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_box_loss = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            batch_size = images.size(0)

            # 准备目标
            cls_targets = torch.zeros(
                (batch_size, config['num_classes']),
                dtype=torch.float32,
                device=device
            )
            box_targets = torch.zeros(
                (batch_size, 4),
                dtype=torch.float32,
                device=device
            )

            # 填充目标值
            for i, target in enumerate(targets):
                if len(target['boxes']) > 0:
                    box_targets[i] = target['boxes'].mean(dim=0).to(device)
                    cls_targets[i, 1] = 1.0
                else:
                    cls_targets[i, 0] = 1.0

            # 前向传播
            cls_pred, box_pred = model(images)

            # 计算损失
            cls_loss = cls_criterion(cls_pred, cls_targets)
            box_loss = box_criterion(box_pred, box_targets)
            loss = config['cls_loss_weight'] * cls_loss + config['box_loss_weight'] * box_loss

            # 更新统计
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_box_loss += box_loss.item()

    # 计算平均损失
    num_batches = len(val_loader)
    return {
        'loss': total_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'box_loss': total_box_loss / num_batches
    }


def train(config):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Batch size: {config['batch_size']}")

    # 创建数据集
    dataset = DetectionDataset(
        config['image_dir'],
        config['annotation_file'],
        max_boxes=config['max_boxes'],
        is_training=True
    )

    # 分割数据集
    val_size = int(len(dataset) * config['val_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )

    # 创建模型
    model = SimpleDetector(num_classes=config['num_classes'])
    model = model.to(device)

    # 定义损失函数
    cls_criterion = nn.BCEWithLogitsLoss()
    box_criterion = nn.SmoothL1Loss()

    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    # 计算总步数和预热步数
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['scheduler_warmup'])

    # 创建带预热的学习率调度器
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 0.5 * (1 + math.cos(
            math.pi * float(step - warmup_steps) / float(total_steps - warmup_steps)
        ))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 早停
    early_stopping = EarlyStopping(patience=config['patience'])

    # 加载检查点
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pth')
    if os.path.exists(checkpoint_path):
        try:
            print(f"Attempting to load checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)

            # 尝试加载模型权重
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Successfully loaded model weights")
            except (RuntimeError, KeyError) as e:
                print(f"Warning: Could not load model weights: {e}")
                print("Training will start from scratch")
                # 重命名旧的检查点文件
                old_checkpoint_path = checkpoint_path + '.old'
                os.rename(checkpoint_path, old_checkpoint_path)
                print(f"Renamed old checkpoint to: {old_checkpoint_path}")
            else:
                # 只有在模型加载成功时才加载其他状态
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch']
                    best_loss = checkpoint['best_loss']
                    print(f"Loaded checkpoint from epoch {start_epoch}")
                except KeyError as e:
                    print(f"Warning: Could not load some checkpoint components: {e}")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training will start from scratch")
            if os.path.exists(checkpoint_path):
                os.rename(checkpoint_path, checkpoint_path + '.corrupt')
                print(f"Renamed corrupt checkpoint file to: {checkpoint_path}.corrupt")

    print(f"Total training samples: {len(train_dataset)}")
    print(f"Training batches per epoch: {len(train_loader)}")

    # 记录训练开始时间
    training_start_time = datetime.now(UTC)
    print(f"\nTraining started at: {training_start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # 训练循环
    try:
        for epoch in range(start_epoch, config['epochs']):
            epoch_start_time = datetime.now(UTC)

            # 训练阶段
            model.train()
            total_loss = 0
            total_cls_loss = 0
            total_box_loss = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["epochs"]}')
            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(device)
                batch_size = images.size(0)

                # 准备目标
                cls_targets = torch.zeros(
                    (batch_size, config['num_classes']),
                    dtype=torch.float32,
                    device=device
                )
                box_targets = torch.zeros(
                    (batch_size, 4),
                    dtype=torch.float32,
                    device=device
                )

                # 填充目标值
                for i, target in enumerate(targets):
                    if len(target['boxes']) > 0:
                        box_targets[i] = target['boxes'].mean(dim=0).to(device)
                        cls_targets[i, 1] = 1.0
                    else:
                        cls_targets[i, 0] = 1.0

                # 前向传播
                cls_pred, box_pred = model(images)

                # 计算损失
                cls_loss = cls_criterion(cls_pred, cls_targets)
                box_loss = box_criterion(box_pred, box_targets)
                loss = config['cls_loss_weight'] * cls_loss + config['box_loss_weight'] * box_loss

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

                optimizer.step()
                scheduler.step()

                # 更新统计
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_box_loss += box_loss.item()

                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cls_loss': f'{cls_loss.item():.4f}',
                    'box_loss': f'{box_loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })

                # 定期保存检查点
                if batch_idx > 0 and batch_idx % config['save_interval'] == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'current_loss': loss.item()
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}_batch{batch_idx}.pth')
                    )

            # 计算训练平均损失
            train_avg_loss = total_loss / len(train_loader)
            train_avg_cls_loss = total_cls_loss / len(train_loader)
            train_avg_box_loss = total_box_loss / len(train_loader)

            # 验证阶段
            val_metrics = validate(model, val_loader, device, config, cls_criterion, box_criterion)

            # 计算epoch用时
            epoch_time = datetime.now(UTC) - epoch_start_time

            # 打印epoch总结
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Time taken: {epoch_time}")
            print("Training Metrics:")
            print(f"- Average Loss: {train_avg_loss:.4f}")
            print(f"- Average Cls Loss: {train_avg_cls_loss:.4f}")
            print(f"- Average Box Loss: {train_avg_box_loss:.4f}")
            print("Validation Metrics:")
            print(f"- Average Loss: {val_metrics['loss']:.4f}")
            print(f"- Average Cls Loss: {val_metrics['cls_loss']:.4f}")
            print(f"- Average Box Loss: {val_metrics['box_loss']:.4f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # 检查早停
            if early_stopping(val_metrics['loss']):
                print("Early stopping triggered")
                break

            # 保存最佳模型
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model with validation loss: {best_loss:.4f}")

            # 保存最新检查点
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'train_loss': train_avg_loss,
                'val_loss': val_metrics['loss'],
                'training_time': str(datetime.now(UTC) - training_start_time)
            }
            torch.save(checkpoint, checkpoint_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    finally:
        # 保存最终状态
        try:
            final_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'training_time': str(datetime.now(UTC) - training_start_time)
            }
            torch.save(final_checkpoint, os.path.join(checkpoint_dir, 'final_model.pth'))
            print("\nSaved final model state")
        except Exception as e:
            print(f"\nError saving final state: {e}")

        # 打印训练总结
        training_time = datetime.now(UTC) - training_start_time
        print(f"\nTraining completed at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Total training time: {training_time}")


def main():
    # 打印开始时间和用户信息
    print(f"Training started at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"User: {os.getlogin()}")

    config = {
        'image_dir': 'D:/ProgramData/PyCharm Community Edition 2024.3.5/PycharmProjects/PythonProject2/OpenDataLab___CrowdHuman/dsdl/dataset_root/Images',
        'annotation_file': 'D:/ProgramData/PyCharm Community Edition 2024.3.5/PycharmProjects/PythonProject2/OpenDataLab___CrowdHuman/dsdl/dataset_root/annotation_train.odgt',
        'num_classes': 2,
        'max_boxes': 5,
        'batch_size': 32,
        'num_workers': 4,
        'learning_rate': 0.0005,
        'weight_decay': 0.0005,
        'epochs': 10,
        'cls_loss_weight': 5.0,
        'box_loss_weight': 1.0,
        'max_grad_norm': 1.0,
        'patience': 7,
        'val_split': 0.1,
        'save_interval': 50,
        'scheduler_warmup': 0.1,
        'clean_checkpoints': True
    }

    # 清理旧的检查点
    if config.get('clean_checkpoints', False):
        checkpoint_dir = 'checkpoints'
        if os.path.exists(checkpoint_dir):
            print("Cleaning old checkpoints...")
            for filename in os.listdir(checkpoint_dir):
                file_path = os.path.join(checkpoint_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.rename(file_path, file_path + '.old')
                        print(f"Renamed {filename} to {filename}.old")
                except Exception as e:
                    print(f"Error while cleaning checkpoint {filename}: {e}")

    try:
        train(config)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise
    finally:
        print(f"\nScript completed at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")


if __name__ == '__main__':
    main()