from re import T
import torch
import torch.nn as nn
# 假设你已经定义了Elic2022ChandelierLite类
from Elic2022light import Elic2022ChandelierLite
from compressai.models import Elic2022Chandelier
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
    #teacher_quality = 3
    #student_quality = 3
    alpha = 0.66 # 蒸馏损失权重 (L_output, 生成图MSE占比)
    beta = 0.34  # 蒸馏损失权重 (L_feature, MGD占比)
    # gamma = 0.35 # 蒸馏损失权重 (L_latent, 潜在表示MSE占比)

    # HyperPriorLoss配置
    hyperprior_lambda = 0.0150# HyperPriorLoss中的λ参数
    lambda_kd = 1 # KD损失中的λ参数
    # 训练配置
    batch_size = 16
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_max_norm = 1.0 # <--- 在这里添加梯度裁剪阈值

    # 新增恢复训练配置
    checkpoint_path = None  # None表示从头训练，设置路径如"checkpoints/student_step1000.pth"则从checkpoint恢复

    # 两阶段训练配置
    use_distillation = True       # 是否启用知识蒸馏损失 (第一阶段为True)
    distill_steps = 10000          # 第一阶段（带蒸馏）训练的步数
    lr_stage2 = 3e-5              # 第二阶段（微调）的学习率
    unfreeze_entropy_in_stage2 = True # 是否在第二阶段解冻熵编码模块

# 2. 改进的蒸馏损失（支持特征对齐、生成图蒸馏） # 移除了 ELIC y0潜在表示蒸馏
class DistillationLoss(nn.Module):
    def __init__(self, config, student_channels, teacher_channels, alpha=0.5, beta=0.5): # 移除 gamma, 添加 config
        """
        初始化蒸馏损失模块。

        Args:
            config (Config): 配置对象，包含 hyperprior_lambda 等参数。
            student_channels (int): 学生模型特征图通道数 (用于MGD对齐)。
            teacher_channels (int): 教师模型特征图通道数 (用于MGD对齐)。
            alpha (float): L_output (生成图MSE) 的权重。
            beta (float): L_feature (MGD特征损失) 的权重。
            # gamma (float): L_latent (ELIC y0潜在表示损失) 的权重。
        """
        super().__init__()
        self.cfg = config # 存储配置对象
        self.alpha = alpha # L_output 权重
        self.beta = beta   # L_feature 权重
        # self.gamma = gamma # L_latent 权重

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

        # # L_latent 使用的MSE (在 _group0_distillation 中隐式使用 torch.mean)
        # # self.latent_mse = nn.MSELoss() # 不再直接使用

    # def _calculate_channel_importance(self, y0_T):
    #     """
    #     基于教师激活强度计算通道权重。
    #
    #     Args:
    #         y0_T (torch.Tensor): 教师模型的第一层潜在表示 [B, C, H, W]。
    #
    #     Returns:
    #         torch.Tensor: 每个通道的权重 [1, C, 1, 1]。
    #     """
    #     # [B,C,H,W] -> [1, C, 1, 1] 计算每个通道的平均绝对值，保持维度以便广播
    #     channel_weights = torch.mean(torch.abs(y0_T), dim=[0, 2, 3], keepdim=True)
    #     # 可以选择归一化权重，例如除以所有权重的总和，但这可能不是必需的
    #     # channel_weights = channel_weights / (torch.sum(channel_weights) + 1e-8)
    #     return channel_weights
    #
    # def _group0_distillation(self, y0_S, y0_T):
    #     """
    #     专用于ELIC第一层潜在表示(y0)的蒸馏，使用通道注意力加权MSE。
    #     添加形状兼容性检查。
    #
    #     Args:
    #         y0_S (torch.Tensor): 学生模型的第一层潜在表示或其替代品 [B, C_s, H, W]。
    #         y0_T (torch.Tensor): 教师模型的第一层潜在表示 [B, C_t, H, W]。
    #
    #     Returns:
    #         torch.Tensor: 计算得到的加权MSE损失（标量），如果形状不兼容则返回0。
    #     """
    #     # 检查通道数是否匹配
    #     if y0_S.shape[1] != y0_T.shape[1]:
    #         print(f"警告: L_latent 的通道数不匹配！学生: {y0_S.shape[1]}, 教师: {y0_T.shape[1]}. 跳过 L_latent 计算。")
    #         # 返回一个与输入在同一设备上的零张量
    #         return torch.tensor(0.0, device=y0_S.device)
    #     # 检查空间维度是否匹配 (可选，但推荐)
    #     if y0_S.shape[2:] != y0_T.shape[2:]:
    #          print(f"警告: L_latent 的空间维度不匹配！学生: {y0_S.shape[2:]}, 教师: {y0_T.shape[2:]}. 跳过 L_latent 计算。")
    #          return torch.tensor(0.0, device=y0_S.device)
    #
    #
    #     # 通道注意力加权MSE
    #     channel_weights = self._calculate_channel_importance(y0_T)
    #     # 计算加权平方误差，然后在所有维度上取平均
    #     loss = torch.mean(channel_weights * (y0_S - y0_T)**2)
    #     # 与HyperPrior同量纲缩放
    #     return loss * 0.0016 * (255**2)

    def forward(self, preds_S, preds_T, x_hat_S, x_hat_T): # 移除 y0_S, y0_T
        """
        计算知识蒸馏损失 L_KD = α*L_output + β*L_feature # 移除 + γ*L_latent

        Args:
            preds_S: 学生模型的特征图 (g_a 输出)
            preds_T: 教师模型的特征图 (g_a 输出)
            x_hat_S: 学生模型的重建图像
            x_hat_T: 教师模型的重建图像
            # y0_S: 学生模型的第一层潜在表示 (y0)
            # y0_T: 教师模型的第一层潜在表示 (y0)

        Returns:
            dict: 包含各种损失项和总蒸馏损失
        """

        # 1. L_output: 生成图MSE损失（与HyperPrior对齐的缩放）
        # 使用配置中的 hyperprior_lambda 进行缩放
        loss_output = self.cfg.hyperprior_lambda * self.mse(x_hat_S, x_hat_T) * (255 ** 2) * self.alpha

        # 添加损失值检查
        if torch.isnan(loss_output) or torch.isinf(loss_output):
            print("WARNING: Invalid loss_output value detected!")
            loss_output = torch.zeros_like(loss_output).to(preds_S.device) # 确保在正确的设备上

        # 2. L_feature: MGD特征损失
        if self.align is not None:
            preds_S_aligned = self.align(preds_S)
        else:
            preds_S_aligned = preds_S

        # 随机掩码生成
        N, C, H, W = preds_T.shape
        mat = torch.rand((N, C, 1, 1), device=preds_S.device)
        mat = torch.where(mat < self.mgd_lambda, 0, 1)

        # 特征蒸馏
        masked_fea = torch.mul(preds_S_aligned, mat)
        new_fea = self.generation(masked_fea)
        # 注意：这里的权重是 beta * mgd_alpha
        # 使用配置中的 hyperprior_lambda 进行缩放
        loss_feature = self.cfg.hyperprior_lambda * self.mse(new_fea, preds_T) * (255 ** 2) * self.beta * self.mgd_alpha

        # # 3. L_latent: 第一层分组潜在表示蒸馏 (使用新方法)
        # loss_latent = self._group0_distillation(y0_S, y0_T) * self.gamma # 假设 gamma 已被注释或移除
        loss_latent = torch.tensor(0.0, device=preds_S.device) # 显式设置为0

        # 添加损失值检查
        if torch.isnan(loss_feature) or torch.isinf(loss_feature):
            print("WARNING: Invalid loss_feature value detected!")
            loss_feature = torch.zeros_like(loss_feature).to(preds_S.device)



        # 组合蒸馏损失 L_KD
        total_distill_loss = loss_output + loss_feature # 移除 + loss_latent

        return {
            'total_distill': total_distill_loss, # L_KD
            'L_output': loss_output,          # α * L_output
            'L_feature': loss_feature,        # β * L_feature (MGD)
            'L_latent': loss_latent          # γ * L_latent (y0 distillation) - 保持键名，但值为0
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
        """
        计算与官方 RateDistortionLoss 对齐的损失（不含 MS-SSIM 部分）
        Args:
            output: 模型输出，需包含 'likelihoods' 字典和 'x_hat'
            target: 原始图像
        Returns:
            dict: 包含 loss, bpp_loss, mse_loss
        """
        N, _, H, W = target.size()
        num_pixels = N * H * W
        # 计算所有 likelihoods 的 bpp_loss
        bpp_loss = 0.0
        for k, v in output['likelihoods'].items():
            bpp_loss += torch.sum(-torch.log2(v)) / num_pixels
        # 计算 MSE 损失并乘以 255^2
        mse_loss = self.mse(output['x_hat'], target) * (255 ** 2)
        # 总损失
        loss = self.lmbda * mse_loss + bpp_loss
        return {
            'loss': loss,
            'bpp_loss': bpp_loss,
            'mse_loss': mse_loss
        }
        

class DistillationTrainer:
    def __init__(self, config):
        """
        初始化蒸馏训练器。

        Args:
            config (Config): 包含所有配置参数的对象。
        """
        self.cfg = config

        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir='runs/experiment_elic_distill') # 修改日志目录以区分实验

        # 初始化模型
        # 教师模型替换为Elic2022Chandelier，并加载本地预训练权重
        self.teacher = Elic2022Chandelier().eval().to(config.device)
        #teacher_ckpt_path = "elic2022_chandelier_pretrained.pth"  # 请根据实际路径修改
        teacher_ckpt_path = "elic2022_chandelier_pretrained_0150.pth" 
        try:
            teacher_state = torch.load(teacher_ckpt_path, map_location=config.device)
            # 尝试加载状态字典，忽略不匹配的键（例如优化器状态等，如果存在）
            self.teacher.load_state_dict(teacher_state, strict=False)
            print(f"教师模型权重已从 {teacher_ckpt_path} 加载")
        except FileNotFoundError:
            print(f"错误：找不到教师模型权重文件 {teacher_ckpt_path}。请确保路径正确。")
            # 可以选择退出或继续（如果允许无预训练教师）
            # exit()
        except Exception as e:
            print(f"加载教师模型权重时出错: {e}")
            # exit()

        # 初始化学生模型（使用预训练权重）
        self.student = Elic2022ChandelierLite().to(config.device)

        # +++ 新增：冻结熵编码相关模块 +++
        print("开始冻结学生模型的熵编码部分...")
        frozen_modules_count = 0
        # 冻结超先验网络 h_a 和 h_s
        if hasattr(self.student, 'h_a'):
            for param in self.student.h_a.parameters():
                param.requires_grad = False
            print("- 已冻结 h_a")
            frozen_modules_count += 1
        if hasattr(self.student, 'h_s'):
            for param in self.student.h_s.parameters():
                param.requires_grad = False
            print("- 已冻结 h_s")
            frozen_modules_count += 1

        # 冻结高斯条件熵模型（GaussianConditional）
        if hasattr(self.student, 'gaussian_conditional'):
            for param in self.student.gaussian_conditional.parameters():
                param.requires_grad = False
            print("- 已冻结 gaussian_conditional")
            frozen_modules_count += 1

        # 冻结熵瓶颈层 (EntropyBottleneck)
        if hasattr(self.student, 'entropy_bottleneck'):
            for param in self.student.entropy_bottleneck.parameters():
                param.requires_grad = False
            print("- 已冻结 entropy_bottleneck")
            frozen_modules_count += 1

        # 检查是否有任何模块被冻结
        if frozen_modules_count == 0:
            print("警告：未能找到或冻结任何指定的熵编码模块 (h_a, h_s, gaussian_conditional, entropy_bottleneck)。请检查学生模型结构。")
        else:
            print(f"总共冻结了 {frozen_modules_count} 个主要的熵编码相关模块。")

        # 添加检查代码确保参数已冻结
        print("\n冻结参数统计：")
        total_params = 0
        frozen_params = 0
        trainable_params = 0
        frozen_param_names = []
        trainable_param_names = []

        for name, param in self.student.named_parameters():
            total_params += param.numel()
            if not param.requires_grad:
                frozen_params += param.numel()
                frozen_param_names.append(name)
            else:
                trainable_params += param.numel()
                trainable_param_names.append(name)

        # 打印详细信息
        print(f"总参数量: {total_params}")
        print(f"冻结参数量: {frozen_params} ({100. * frozen_params / total_params:.2f}%)")
        print(f"可训练参数量: {trainable_params} ({100. * trainable_params / total_params:.2f}%)")
        self.hyperprior_loss = HyperPriorLoss(lmbda=config.hyperprior_lambda).to(config.device)

        # 基础MSE损失（学生输出 vs GT） - 不再直接使用，由HyperPriorLoss处理
        #self.mse_loss = nn.MSELoss()

        # 优化器 - 移到这里，确保在加载checkpoint前初始化
        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=config.lr
        )

        # 新增训练步数计数器和保存间隔
        self.step_counter = 0
        self.save_interval = 200  # 每50步保存一次
        # 新增检查点保存目录
        self.checkpoint_dir = Path("checkpoints") # 定义检查点保存目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True) # 创建目录

        # 从checkpoint恢复（如果配置了路径）
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


        # 冻结教师模型
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 损失函数
        # 动态获取通道数 (用于MGD对齐)
        try:
            with torch.no_grad():
                dummy_input = torch.rand(1, 3, 256, 256).to(config.device)
                # 确保学生和教师模型都能处理输入
                _ = self.student(dummy_input)
                _ = self.teacher(dummy_input)
                # 获取 g_a 输出通道数
                student_channels = self.student.g_a(dummy_input).shape[1]
                teacher_channels = self.teacher.g_a(dummy_input).shape[1]

            self.criterion = DistillationLoss(
                config=self.cfg, # <--- 传递 config 对象
                student_channels=student_channels,
                teacher_channels=teacher_channels,
                alpha=config.alpha, # 传递 alpha
                beta=config.beta   # 传递 beta
                # gamma=config.gamma  # 传递 gamma
            ).to(config.device)
        except Exception as e:
            print(f"初始化DistillationLoss时出错（可能是模型前向传播问题）: {e}")
            print("请检查模型定义和输入数据的兼容性。")
            exit()


    def save_checkpoint(self, filename="checkpoint.pth"):
        """
        保存模型检查点，包括学生模型状态、优化器状态和当前步数。

        Args:
            filename (str): 保存检查点文件的名称。
        """
        # 确保文件名包含目录路径
        filepath = self.checkpoint_dir / filename
        state = {
            'step': self.step_counter,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 可以选择性地保存其他信息，例如配置
            # 'config': self.cfg
        }
        torch.save(state, filepath)
        print(f"检查点已保存至: {filepath}")

    def train_step(self, x):
        """
        执行单个训练步骤。

        Args:
            x (torch.Tensor): 输入图像批次。

        Returns:
            dict: 包含各种损失值的字典，如果发生错误则返回 None。
        """
        # 确保输入数据在正确的设备上
        if not x.is_cuda:
            x = x.to(self.cfg.device)
            
        self.optimizer.zero_grad()

        # 存储损失值
        losses = {}

        # --- 教师模型处理 ---
        try:
            with torch.no_grad():
                # 教师推理，获取重建图像和似然
                teacher_out = self.teacher(x)
                teacher_out['x_hat'] = teacher_out['x_hat'].clamp(0, 1)
                # 获取教师的特征图 (g_a 输出, 即 y)
                g_a_teacher = self.teacher.g_a(x)


        except Exception as e:
            print(f"教师模型推理或特征提取时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

        # --- 学生模型处理 ---
        try:
            # 学生推理
            student_out = self.student(x)
            student_out['x_hat'] = student_out['x_hat'].clamp(0, 1)
            # 获取学生的特征图 (g_a 输出)
            g_a_student = self.student.g_a(x)


        except Exception as e:
            print(f"学生模型推理或特征/潜在表示提取时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

        # --- 计算损失 ---
        try:
            # 1. 计算学生模型的 HyperPrior 损失 (重建损失 + 码率损失)
            hp_loss = self.hyperprior_loss(student_out, x)
            # 计算加权的 MSE 损失
            student_weighted_mse = hp_loss['mse_loss'] * self.cfg.hyperprior_lambda
            losses.update({
                'student_total': hp_loss['loss'],
                'student_bpp': hp_loss['bpp_loss'],
                'student_mse': hp_loss['mse_loss'],
                'student_weighted_mse': student_weighted_mse # 添加加权 MSE
            })

            # 2. 计算蒸馏损失 (L_output, L_feature, L_latent)
            # L_latent 的计算现在会在 DistillationLoss 内部检查形状兼容性
            dist_loss = self.criterion(
                preds_S=g_a_student,          # 学生特征图
                preds_T=g_a_teacher,          # 教师特征图 (用于 MGD)
                x_hat_S=student_out['x_hat'], # 学生重建图像
                x_hat_T=teacher_out['x_hat'] # 教师重建图像
            )
            losses.update({
                'distill_total': dist_loss['total_distill'],
                'L_output': dist_loss['L_output'],
                'L_feature': dist_loss['L_feature'],
                'L_latent': dist_loss['L_latent'] # Loss内部会处理不兼容情况
            })

            # 3. 组合总损失
            # 总损失 = 学生自身损失 (+ λ_KD * 蒸馏损失，如果启用)
            total_loss = hp_loss['loss']
            if self.cfg.use_distillation:
                total_loss += self.cfg.lambda_kd * dist_loss['total_distill']
            else:
                # 确保蒸馏损失相关键存在且为0，用于日志记录
                dist_loss['total_distill'] = torch.tensor(0.0, device=x.device)
                dist_loss['L_output'] = torch.tensor(0.0, device=x.device)
                dist_loss['L_feature'] = torch.tensor(0.0, device=x.device)
                dist_loss['L_latent'] = torch.tensor(0.0, device=x.device)
            
            # 更新 losses 字典，确保所有键都存在
            losses.update({
                'distill_total': dist_loss['total_distill'],
                'L_output': dist_loss['L_output'],
                'L_feature': dist_loss['L_feature'],
                'L_latent': dist_loss['L_latent']
            })
            losses['total_train_loss'] = total_loss

        except Exception as e:
            print(f"计算损失时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

        # --- 反向传播和优化 ---
        try:
            total_loss.backward()
            # 梯度裁剪 (可选但推荐)
            if self.cfg.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.clip_max_norm)
            self.optimizer.step()
        except Exception as e:
            print(f"反向传播或优化器步骤出错: {e}")
            import traceback
            traceback.print_exc()
            return None # 发生错误，返回 None

        # 增加步数计数器
        self.step_counter += 1

        # --- 定期保存模型 ---
        if self.step_counter % self.save_interval == 0:
            # 使用定义好的方法保存检查点
            self.save_checkpoint(f'student_step_{self.step_counter}.pth') # 保持文件名格式一致


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

def get_dataloader(data_path, batch_size, device):
    """
    获取数据加载器，确保数据在指定设备上
    
    Args:
        data_path: 数据路径
        batch_size: 批次大小
        device: 目标设备(cpu/cuda)
    """
    transform = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        # 添加设备转移
        lambda x: x.to(device)  
    ])
    dataset = CustomDataset(data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. 训练循环
def train():
    """
    主训练循环函数，支持两阶段训练。
    """
    cfg = Config()
    trainer = DistillationTrainer(cfg)
    # 打印初始化信息
    print(f"Initialized DistillationLoss with alpha = {trainer.criterion.alpha}, beta = {trainer.criterion.beta}")
    print(f"Initialized HyperPriorLoss with lambda = {trainer.hyperprior_loss.lmbda}")
    print(f"Using device: {cfg.device}")
    print(f"两阶段训练配置: distill_steps={cfg.distill_steps}, lr_stage2={cfg.lr_stage2}, unfreeze_entropy={cfg.unfreeze_entropy_in_stage2}")
    
    dataloader = get_dataloader(r"data/DIV2K_train_p128/", cfg.batch_size, cfg.device)

    num_epochs = 1000 # 或者从配置中读取
    start_epoch = trainer.step_counter // len(dataloader) # 根据恢复的步数计算起始 epoch
    stage1_completed = not cfg.use_distillation # 如果初始配置就是False，则认为第一阶段已完成

    print(f"开始训练，从 Epoch {start_epoch}，Step {trainer.step_counter} 开始...")
    print(f"当前阶段: {'1 (蒸馏)' if cfg.use_distillation else '2 (微调)'}")

    for epoch in range(start_epoch, num_epochs):
        trainer.student.train() # 确保学生模型处于训练模式
        total_batches = len(dataloader)
        
        with tqdm(total=total_batches, desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
            for i, (batch_data, _) in enumerate(dataloader):
                # --- 检查并执行阶段切换 --- 
                if not stage1_completed and trainer.step_counter >= cfg.distill_steps:
                    print(f"\n达到步数 {cfg.distill_steps}，切换到第二阶段（微调）...")
                    cfg.use_distillation = False # 修改 trainer 持有的 cfg 对象
                    stage1_completed = True

                    # 调整学习率
                    print(f"将学习率从 {trainer.optimizer.param_groups[0]['lr']:.1e} 调整为 {cfg.lr_stage2:.1e}")
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] = cfg.lr_stage2

                    # 可选：解冻熵编码模块
                    if cfg.unfreeze_entropy_in_stage2:
                        print("解冻熵编码模块...")
                        modules_to_unfreeze = ['h_a', 'h_s', 'gaussian_conditional', 'entropy_bottleneck']
                        unfrozen_count = 0
                        for name in modules_to_unfreeze:
                            if hasattr(trainer.student, name):
                                module = getattr(trainer.student, name)
                                for param in module.parameters():
                                    param.requires_grad = True
                                print(f"- 已解冻 {name}")
                                unfrozen_count += 1
                        if unfrozen_count > 0:
                            print(f"总共解冻了 {unfrozen_count} 个模块。")
                            # 验证解冻效果 (可选)
                            trainable_params_after_unfreeze = sum(p.numel() for p in trainer.student.parameters() if p.requires_grad)
                            total_params = sum(p.numel() for p in trainer.student.parameters())
                            print(f"解冻后可训练参数量: {trainable_params_after_unfreeze} / {total_params} ({100. * trainable_params_after_unfreeze / total_params:.2f}%)")
                        else:
                            print("未找到可解冻的熵编码模块。")
                    else:
                        print("根据配置，保持熵编码模块冻结状态。")

                    # 切换后，确保模型仍在训练模式
                    trainer.student.train()
                # --- 阶段切换结束 ---

                batch_data = batch_data.to(cfg.device)
                losses = trainer.train_step(batch_data)

                if losses is not None:
                    # 更新进度条显示损失信息
                    postfix_dict = {
                        'loss': f"{losses['total_train_loss']:.4f}",
                        'stu_loss': f"{losses['student_total']:.4f}",
                        'stu_bpp': f"{losses['student_bpp']:.4f}",
                        'stu_mse': f"{losses['student_mse']:.4f}",
                        'lr': f"{trainer.optimizer.param_groups[0]['lr']:.1e}",
                        'step': trainer.step_counter,
                        'phase': 1 if cfg.use_distillation else 2
                    }
                    # 只在第一阶段显示有效的蒸馏损失
                    if cfg.use_distillation:
                        postfix_dict['kd_loss'] = f"{losses['distill_total']:.4f}"
                        # postfix_dict['L_out'] = f"{losses['L_output']:.4f}"
                        # postfix_dict['L_feat'] = f"{losses['L_feature']:.4f}"
                    pbar.set_postfix(postfix_dict)

                    # 记录到 TensorBoard
                    global_step = trainer.step_counter # 使用 trainer 维护的全局步数
                    trainer.writer.add_scalar('Loss/Total_Train', losses['total_train_loss'], global_step)
                    trainer.writer.add_scalar('Loss/Student_Total', losses['student_total'], global_step)
                    trainer.writer.add_scalar('Loss/Student_BPP', losses['student_bpp'], global_step)
                    trainer.writer.add_scalar('Loss/Student_MSE', losses['student_mse'], global_step)
                    # 只有在第一阶段记录有效的蒸馏损失值
                    if cfg.use_distillation:
                        trainer.writer.add_scalar('Loss/Distill_Total', losses['distill_total'], global_step)
                        trainer.writer.add_scalar('Loss/Distill_L_output', losses['L_output'], global_step)
                        trainer.writer.add_scalar('Loss/Distill_L_feature', losses['L_feature'], global_step)
                        # trainer.writer.add_scalar('Loss/Distill_L_latent', losses['L_latent'], global_step)
                    trainer.writer.add_scalar('Learning_Rate', trainer.optimizer.param_groups[0]['lr'], global_step)
                    trainer.writer.add_scalar('Training_Phase', 1 if cfg.use_distillation else 2, global_step)

                else:
                    print(f"跳过 Epoch {epoch}, Batch {i} 由于训练步骤错误。")

                pbar.update(1) # 更新进度条

        print(f"Epoch {epoch} 完成 | 当前阶段: {'1 (蒸馏)' if cfg.use_distillation else '2 (微调)'}")
    trainer.writer.close() # 训练结束后关闭 writer
    print("训练完成。")

# 运行训练
if __name__ == "__main__":
    train()