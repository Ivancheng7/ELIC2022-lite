from re import T
import os
import torch
import torch.nn as nn
from PIL import Image
from compressai.models import Elic2022Chandelier
from Elic2022light import Elic2022ChandelierLite
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from pathlib import Path

# 1. 配置参数
class Config:
    student_quality = 3
    alpha = 0.66
    beta = 0.34
    hyperprior_lambda = 0.15
    lambda_kd = 1
    batch_size = 8
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_max_norm = 1.0
    checkpoint_path = None
    use_distillation = True
    distill_steps = 10000
    lr_stage2 = 5e-5
    unfreeze_entropy_in_stage2 = True

# 2. 改进的蒸馏损失
class DistillationLoss(nn.Module):
    def __init__(self, config, student_channels, teacher_channels, alpha=0.5, beta=0.5):
        super().__init__()
        self.cfg = config
        self.alpha = alpha
        self.beta = beta

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1)
        )

        self.mse = nn.MSELoss()
        self.mgd_alpha = 0.0003
        self.mgd_lambda = 0.5

    def forward(self, preds_S, preds_T, x_hat_S, x_hat_T):
        loss_output = self.cfg.hyperprior_lambda * self.mse(x_hat_S, x_hat_T) * (255 ** 2) * self.alpha

        if torch.isnan(loss_output) or torch.isinf(loss_output):
            print("WARNING: Invalid loss_output value detected!")
            loss_output = torch.zeros_like(loss_output).to(preds_S.device)

        if self.align is not None:
            preds_S_aligned = self.align(preds_S)
        else:
            preds_S_aligned = preds_S

        N, C, H, W = preds_T.shape
        mat = torch.rand((N, C, 1, 1), device=preds_S.device)
        mat = torch.where(mat < self.mgd_lambda, 0, 1)

        masked_fea = torch.mul(preds_S_aligned, mat)
        new_fea = self.generation(masked_fea)
        loss_feature = self.cfg.hyperprior_lambda * self.mse(new_fea, preds_T) * (255 ** 2) * self.beta * self.mgd_alpha

        loss_latent = torch.tensor(0.0, device=preds_S.device)

        if torch.isnan(loss_feature) or torch.isinf(loss_feature):
            print("WARNING: Invalid loss_feature value detected!")
            loss_feature = torch.zeros_like(loss_feature).to(preds_S.device)

        total_distill_loss = loss_output + loss_feature

        return {
            'total_distill': total_distill_loss,
            'L_output': loss_output,
            'L_feature': loss_feature,
            'L_latent': loss_latent
        }

# 3. 完整训练框架
class HyperPriorLoss(nn.Module):
    def __init__(self, lmbda=0.01):
        super().__init__()
        self.lmbda = lmbda
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        bpp_loss = 0.0
        for k, v in output['likelihoods'].items():
            bpp_loss += torch.sum(-torch.log2(v)) / num_pixels
        mse_loss = self.mse(output['x_hat'], target) * (255 ** 2)
        loss = self.lmbda * mse_loss + bpp_loss
        return {
            'loss': loss,
            'bpp_loss': bpp_loss,
            'mse_loss': mse_loss
        }

class DistillationTrainer:
    def __init__(self, config):
        self.cfg = config
        self.writer = SummaryWriter(log_dir='runs/experiment_elic_distill')
        self.teacher = Elic2022Chandelier().eval().to(config.device)
        teacher_ckpt_path = "elic2022_chandelier_pretrained_0150.pth"
        try:
            teacher_state = torch.load(teacher_ckpt_path, map_location=config.device)
            self.teacher.load_state_dict(teacher_state, strict=False)
            print(f"教师模型权重已从 {teacher_ckpt_path} 加载")
        except FileNotFoundError:
            print(f"错误：找不到教师模型权重文件 {teacher_ckpt_path}。请确保路径正确。")
        except Exception as e:
            print(f"加载教师模型权重时出错: {e}")

        self.student = Elic2022ChandelierLite().to(config.device)
        # 将模型结构信息写入txt文件
        with open("model_structures.txt", "w", encoding="utf-8") as f:
            f.write("学生模型结构:\n")
            f.write(str(self.student) + "\n")
            f.write("教师模型结构:\n")
            f.write(str(self.teacher) + "\n")
        self.hyperprior_loss = HyperPriorLoss(lmbda=config.hyperprior_lambda).to(config.device)

        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=config.lr
        )

        self.step_counter = 0
        self.save_interval = 200
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config.checkpoint_path:
            try:
                checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
                self.student.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.step_counter = checkpoint.get('step', 0)
                print(f"已从 {config.checkpoint_path} 恢复模型，从第 {self.step_counter} 步继续训练")
            except FileNotFoundError:
                print(f"警告：找不到检查点文件 {config.checkpoint_path}，将从头开始训练。")
            except Exception as e:
                 print(f"从检查点恢复时出错: {e}，将从头开始训练。")

        for param in self.teacher.parameters():
            param.requires_grad = False

        try:
            with torch.no_grad():
                dummy_input = torch.rand(1, 3, 256, 256).to(config.device)
                # 获取学生和教师模型g_a的输出通道数，直接读取最后一层的out_channels属性
                # 获取学生模型 g_a 输出通道数
                if hasattr(self.student.g_a, 'out_channels' ):
                    student_channels = self.student.g_a.out_channels
                elif isinstance (self.student.g_a, nn.Sequential):
                    last_layer_student = list(self.student.g_a.children())[-1 ]
                    if hasattr(last_layer_student, 'out_channels' ):
                        student_channels = last_layer_student.out_channels
                    else :
                        # 如果最后一个模块没有 out_channels 属性，则通过前向传播获取
                        student_channels = self.student.g_a(dummy_input).shape[1 ]
                else :
                    student_channels = self.student.g_a(dummy_input).shape[1 ]

                # 获取教师模型 g_a 输出通道数
                if hasattr(self.teacher.g_a, 'out_channels' ):
                    teacher_channels = self.teacher.g_a.out_channels
                elif isinstance (self.teacher.g_a, nn.Sequential):
                    last_layer_teacher = list(self.teacher.g_a.children())[-1 ]
                    if hasattr(last_layer_teacher, 'out_channels'): # 检查最后一个模块是否有 out_channels
                        teacher_channels = last_layer_teacher.out_channels
                    else :
                        # 如果最后一个模块（如 AttentionBlock）没有 out_channels 属性，则通过前向传播获取
                        teacher_channels = self.teacher.g_a(dummy_input).shape[1] # 这条路径会被教师模型采用
                else :
                    teacher_channels = self.teacher.g_a(dummy_input).shape[1 ]

                print(f"学生模型 g_a 输出通道数: {student_channels}")
                print(f"教师模型 g_a 输出通道数: {teacher_channels}")

                self.criterion = DistillationLoss(
                    config=self.cfg,
                    student_channels=student_channels,
                    teacher_channels=teacher_channels,
                    alpha=config.alpha,
                    beta=config.beta
                ).to(config.device)
        except Exception as e:
            print(f"初始化DistillationLoss时出错（可能是模型前向传播问题）: {e}")
            print("请检查模型定义和输入数据的兼容性。")
            exit()

    def save_checkpoint(self, filename="checkpoint.pth"):
        filepath = self.checkpoint_dir / filename
        state = {
            'step': self.step_counter,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(state, filepath)
        print(f"检查点已保存至: {filepath}")

    def train_step(self, x):
        if not x.is_cuda:
            x = x.to(self.cfg.device)
        
        self.optimizer.zero_grad()

        losses = {}

        try:
            with torch.no_grad():
                teacher_out = self.teacher(x)
                teacher_out['x_hat'] = teacher_out['x_hat'].clamp(0, 1)
                g_a_teacher = self.teacher.g_a(x)

        except Exception as e:
            print(f"教师模型推理或特征提取时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

        try:
            student_out = self.student(x)
            student_out['x_hat'] = student_out['x_hat'].clamp(0, 1)
            g_a_student = self.student.g_a(x)

        except Exception as e:
            print(f"学生模型推理或特征/潜在表示提取时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

        try:
            hp_loss = self.hyperprior_loss(student_out, x)
            student_weighted_mse = hp_loss['mse_loss'] * self.cfg.hyperprior_lambda
            losses.update({
                'student_total': hp_loss['loss'],
                'student_bpp': hp_loss['bpp_loss'],
                'student_mse': hp_loss['mse_loss'],
                'student_weighted_mse': student_weighted_mse
            })

            dist_loss = self.criterion(
                preds_S=g_a_student,
                preds_T=g_a_teacher,
                x_hat_S=student_out['x_hat'],
                x_hat_T=teacher_out['x_hat']
            )
            losses.update({
                'distill_total': dist_loss['total_distill'],
                'L_output': dist_loss['L_output'],
                'L_feature': dist_loss['L_feature'],
                'L_latent': dist_loss['L_latent']
            })

        except Exception as e:
            print(f"损失计算时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

        total_loss = losses['student_total'] + losses['distill_total']
        total_loss.backward()

        if self.cfg.clip_max_norm:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.clip_max_norm)

        self.optimizer.step()

        self.step_counter += 1

        if self.step_counter % self.save_interval == 0:
            self.save_checkpoint(f"student_step{self.step_counter}.pth")

        return losses


# =====================
# 自定义数据集类，需放在模块顶层，避免多进程DataLoader无法pickle
class ImageDataset(torch.utils.data.Dataset):
    """
    自定义图像数据集类，支持transform。
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # 返回0作为伪标签

if __name__ == '__main__':
    # 配置参数
    config = Config()

    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])

    # 加载训练数据
    train_dataset = ImageDataset(
        root_dir='data/DIV2K_train_p128',
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 初始化训练器
    trainer = DistillationTrainer(config)

    # 训练循环
    print("开始训练...")
    step = 0
    epoch = 0
    try:
        while step < config.distill_steps:
            epoch += 1
            print(f"\nEpoch {epoch}")
            for batch_idx, (x, _) in enumerate(tqdm(train_loader)):
                losses = trainer.train_step(x)
                if losses is None:
                    print(f"警告：第 {step} 步训练失败，跳过此步骤")
                    continue

                step = trainer.step_counter
                if step >= config.distill_steps:
                    break

                # 记录损失到TensorBoard
                for loss_name, loss_value in losses.items():
                    trainer.writer.add_scalar(f'Loss/{loss_name}', loss_value.item(), step)

                # 实时命令行显示loss
                if losses is not None:
                    loss_str = f"Step {step} | " + ", ".join([f"{k}: {v.item():.4f}" for k, v in losses.items()])
                    print("\r" + loss_str, end="", flush=True)

                # 定期保存重建图像
                if step % 100 == 0:
                    with torch.no_grad():
                        trainer.student.eval()
                        x_batch = x[:4]  # 只取前4张图片
                        x_eval = x_batch.to(trainer.cfg.device)  # 确保输入和模型在同一设备
                        out_eval = trainer.student(x_eval)
                        x_hat = out_eval['x_hat'].clamp(0, 1)
                        grid = vutils.make_grid(x_hat.cpu(), normalize=True, scale_each=True)  # 移回CPU
                        trainer.writer.add_image('Reconstruction', grid, step)
                        trainer.student.train()

        print("\n训练完成！")

    except KeyboardInterrupt:
        print("\n训练被用户中断")

    finally:
        # 保存最终模型
        trainer.save_checkpoint(f"student_final.pth")
        trainer.writer.close()