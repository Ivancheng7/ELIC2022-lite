from re import T
import torch
import torch.nn as nn
from compressai.zoo import cheng2020_anchor, cheng2020_attn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
# Add this import at the top with other imports
import time
from tqdm import tqdm
# 1. 配置参数
class Config:
    # 模型配置
    teacher_quality = 3
    student_quality = 3
    beta = 0.66  # 蒸馏损失权重 (生成图MSE占比)

    # HyperPriorLoss配置
    hyperprior_lambda = 0.0016 # HyperPriorLoss中的λ参数
    hyperprior_weight = 0.5  # HyperPriorLoss权重
    distill_weight = 0.5  # 蒸馏损失权重
    # 训练配置
    batch_size = 10
    lr = 3e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 新增恢复训练配置
    checkpoint_path = None  # None表示从头训练，设置路径如"checkpoints/student_step1000.pth"则从checkpoint恢复

# 2. 改进的MGD损失（支持特征对齐和生成图蒸馏）
#这里面的损失函数包括特征损失：MGD损失
# 学生老师输出损失：mse
class DistillationLoss(nn.Module):
    def __init__(self, student_channels, teacher_channels, beta=0.5):
        super().__init__()
        self.beta = beta
        
        # MGD特征损失部分
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
        else:
            self.align = None
            
        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1)
        )
        
        # 损失函数
        self.mse = nn.MSELoss()
        self.mgd_alpha = 0.0003  # MGD原始论文推荐值
        self.mgd_lambda = 0.5    # 掩码比例

    def forward(self, preds_S, preds_T, x_hat_S, x_hat_T):

        
        # 生成图MSE损失 (β权重)
        loss_gen_mse = self.mse(x_hat_S, x_hat_T) * self.beta
        
        # 添加损失值检查
        if torch.isnan(loss_gen_mse) or torch.isinf(loss_gen_mse):
            print("WARNING: Invalid loss_gen_mse value detected!")
            loss_gen_mse = torch.zeros_like(loss_gen_mse)
        
        # MGD特征损失 (1-β权重)
        if self.align is not None:
            preds_S = self.align(preds_S)
        
        # 随机掩码生成
        N, C, H, W = preds_T.shape
        mat = torch.rand((N, C, 1, 1), device=preds_S.device)
        mat = torch.where(mat < self.mgd_lambda, 0, 1)
        
        # 特征蒸馏
        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)
        loss_mgd = self.mse(new_fea, preds_T) * (1 - self.beta) * self.mgd_alpha
        
        return {
            'total': loss_gen_mse + loss_mgd,
            'gen_mse': loss_gen_mse,
            'mgd': loss_mgd
        }

# 3. 完整训练框架
# 包括：数据加载、模型初始化、训练循环、损失计算、优化器更新、模型保存等
# HyperPriorLoss类
# 包括：学生重建的MSE损失和码率损失
# 输入：学生模型输出、真实图像
class HyperPriorLoss(nn.Module):
    def __init__(self, lmbda=0.01):
        super().__init__()
        self.lmbda = lmbda
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        # 1. 计算条件熵 R_y^S
        R_y = torch.mean(-torch.log2(output['likelihoods']['y']))

        # 2. 计算超先验熵 R_z^S
        R_z = torch.mean(-torch.log2(output['likelihoods']['z']))

        # 3. 计算总码率 (R_y + R_z)
        total_rate = R_y + R_z

        # 4. 计算失真（重建误差）
        distortion = self.mse(output['x_hat'], target)

        # 5. 组合损失
        total_loss = self.lmbda * total_rate + distortion

        return {
            'total': total_loss,
            'rate': total_rate,  # R_y + R_z
            'R_y': R_y,          # 条件熵项
            'R_z': R_z,          # 超先验熵项
            'distortion': distortion
        }

class DistillationTrainer:
    def __init__(self, config):
        self.cfg = config
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir='runs/experiment')
        
        # 初始化模型
        self.teacher = cheng2020_attn(
            quality=config.teacher_quality, 
            pretrained=True
        ).eval().to(config.device)
        
        # 初始化学生模型（使用预训练权重）
        self.student = cheng2020_anchor(
            quality=config.student_quality,
            pretrained= True # 加载预训练权重
        ).to(config.device)
        
        # 初始化HyperPriorLoss
        self.hyperprior_loss = HyperPriorLoss(lmbda=config.hyperprior_lambda).to(config.device)
        
        # 基础MSE损失（学生输出 vs GT）
        self.mse_loss = nn.MSELoss()
        
        # 优化器 - 移到这里，确保在加载checkpoint前初始化
        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=config.lr
        )
        
        # 新增训练步数计数器和保存间隔
        self.step_counter = 0
        self.save_interval = 50  # 每50步保存一次
        
        # 从checkpoint恢复（如果配置了路径）
        if config.checkpoint_path:
            checkpoint = torch.load(config.checkpoint_path)
            self.student.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step_counter = checkpoint.get('step', 0)
            print(f"已从 {config.checkpoint_path} 恢复模型，从第 {self.step_counter} 步继续训练")
        
        # 冻结教师模型
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # 损失函数
        # 动态获取通道数
        with torch.no_grad():
            dummy_input = torch.rand(1, 3, 256, 256).to(config.device)
            student_channels = self.student.g_a(dummy_input).shape[1]
            teacher_channels = self.teacher.g_a(dummy_input).shape[1]
            
        self.criterion = DistillationLoss(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            beta=config.beta
            ).to(config.device)

    def train_step(self, x):
        x = x.to(self.cfg.device)
        
        # 教师推理
        with torch.no_grad():
            teacher_out = self.teacher(x)
            teacher_out['x_hat'] = teacher_out['x_hat'].clamp(0, 1)  # 添加约束
            g_a_teacher = self.teacher.g_a(x)
        
        # 学生推理
        student_out = self.student(x)
        student_out['x_hat'] = student_out['x_hat'].clamp(0, 1)  # 添加约束
        g_a_student = self.student.g_a(x)
        
        # 保存重建图像示例
        if not hasattr(self, 'save_counter'):
            self.save_counter = 0
        if self.save_counter % 100 == 0:  # 每100步保存一次
            vutils.save_image(
                torch.cat([x[:4], student_out['x_hat'][:4], teacher_out['x_hat'][:4]], 0),
                f'reconstruction_{self.save_counter}.png',
                normalize=True,
                nrow=4  # 每行4张图像
            )
        self.save_counter += 1
        
        # 计算损失
        losses = {}
        
        # 1. 计算超先验损失
        hyperprior_loss = self.hyperprior_loss(student_out, x)
        losses.update({
            'hyperprior_total': hyperprior_loss['total'],
            'distortion': hyperprior_loss['distortion'],
            'rate': hyperprior_loss['rate'],
            'R_y': hyperprior_loss['R_y'],
            'R_z': hyperprior_loss['R_z']
        })
        
        # 2. 计算蒸馏损失
        dist_loss = self.criterion(
            preds_S=g_a_student,
            preds_T=g_a_teacher,
            x_hat_S=student_out['x_hat'],
            x_hat_T=teacher_out['x_hat']
        )
        losses.update({
            'distill_total': dist_loss['total'],
            'gen_mse': dist_loss['gen_mse'],
            'mgd': dist_loss['mgd']
        })
        
        # 3. 组合损失
        losses['total'] = self.cfg.hyperprior_weight * losses['hyperprior_total'] + self.cfg.distill_weight * losses['distill_total']
        
        # 删除下面重复的损失计算部分（约第240-260行）：
        # 基础MSE损失（学生输出 vs 真实图像） <-- 已包含在HyperPriorLoss的distortion中
        # losses['base_mse'] = self.mse_loss(student_out['x_hat'], x)
        
        # 删除下面重复的蒸馏损失计算 <-- 已经在上方计算过
        # dist_loss = self.criterion(...) 
        
        # 删除重复的总损失计算 <-- 已经在上方计算过
        # losses['total'] = losses['base_mse'] + losses['total']
        
        # 反向传播 (保留这部分)
        self.optimizer.zero_grad()
        losses['total'].backward()
        self.optimizer.step()
        
        # 记录到TensorBoard
        self.writer.add_scalar('HyperPrior/total', losses['hyperprior_total'], self.step_counter)
        self.writer.add_scalar('HyperPrior/distortion', losses['distortion'], self.step_counter) 
        self.writer.add_scalar('HyperPrior/rate', losses['rate'], self.step_counter)
        self.writer.add_scalar('HyperPrior/R_y', losses['R_y'], self.step_counter)
        self.writer.add_scalar('HyperPrior/R_z', losses['R_z'], self.step_counter)
        
        # 更新步数计数器并保存模型
        self.step_counter += 1
        
        # 记录损失到TensorBoard
        self.writer.add_scalar('Loss/total', losses['total'].item(), self.step_counter)
        #self.writer.add_scalar('Loss/base_mse', losses['base_mse'].item(), self.step_counter)
        self.writer.add_scalar('Loss/gen_mse', losses['gen_mse'].item(), self.step_counter)
        self.writer.add_scalar('Loss/mgd', losses['mgd'].item(), self.step_counter)
        
        # 保存重建图像到TensorBoard
        if self.step_counter % 10 == 0:  # 每10步保存一次
            self.writer.add_images('Reconstruction/input', x[:4], self.step_counter)
            self.writer.add_images('Reconstruction/student', student_out['x_hat'][:4], self.step_counter)
            self.writer.add_images('Reconstruction/teacher', teacher_out['x_hat'][:4], self.step_counter)
        
        if self.step_counter % self.save_interval == 0:
            save_path = f"checkpoints/student_step{self.step_counter}.pth"
            torch.save({
                'step': self.step_counter,
                'model_state_dict': self.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': losses['total'].item(),
            }, save_path)
            print(f"模型权重已保存至 {save_path}")
        
        return losses

# 4. 数据加载示例（使用DIV2K子集）
from pathlib import Path
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = sorted(list(Path(img_dir).glob("*.png")))  # 排序保证一致性
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")  # 统一转为RGB
        if self.transform:
            img = self.transform(img)
        return img, 0  # 返回图像和伪标签

def get_dataloader(data_path, batch_size):
    transform = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. 训练循环
# 删除第一个train函数定义（约第290-318行）
# 保留并修改第二个train函数（带tqdm的版本）

def train():
    cfg = Config()
    trainer = DistillationTrainer(cfg)
    dataloader = get_dataloader("data/DIV2K_train_p128/", cfg.batch_size)
    
    # 初始化计时器
    start_time = time.time()
    total_images = 0
    
    for epoch in range(100):
        # 使用tqdm显示进度条
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}', unit='batch')

        for batch_idx, (x, _) in enumerate(pbar):
            batch_size = x.size(0)
            total_images += batch_size
            
            # 训练步骤
            losses = trainer.train_step(x)
            
            # 计算训练速度
            elapsed_time = time.time() - start_time
            images_per_sec = total_images / elapsed_time if elapsed_time > 0 else 0
            
            # 更新进度条描述
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'img/s': f"{images_per_sec:.2f}",
                'time': f"{elapsed_time:.1f}s",
                # 'bpp': f"{losses['rate'].item():.4f}"  # 已注释掉 BPP 显示
            })
            
            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Total Loss: {losses['total'].item():.4f} | "
                    f"HyperPrior: {losses['hyperprior_total'].item():.4f} | "
                    f"Distortion: {losses['distortion'].item():.4f} | "
                    f"熵 (R_y+R_z): {losses['rate'].item():.4f} | "
                    f"R_y: {losses['R_y'].item():.4f} | "
                    f"R_z: {losses['R_z'].item():.4f} | "
                    f"Gen MSE: {losses['gen_mse'].item():.4f} | "
                    f"MGD: {losses['mgd'].item():.6f}"
                )

if __name__ == "__main__":
    train()

