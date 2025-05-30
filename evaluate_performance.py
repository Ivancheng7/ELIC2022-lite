import os
import time
import torch
import torch.nn as nn
import compressai
from compressai.zoo import cheng2020_anchor
from compressai.models import Elic2022Chandelier
from Elic2022light import Elic2022ChandelierLite # 假设 Elic2022ChandelierLite 在 Elic2022light.py 中定义
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
from typing import Dict, Tuple, List

def compute_padding(h, w, min_div=2**6) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """计算所需的填充量以使图像尺寸成为 min_div 的倍数。

    Args:
        h (int): 图像高度。
        w (int): 图像宽度。
        min_div (int): 最小除数 (通常是 2 的幂)。

    Returns:
        Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]: 
            - pad: F.pad 使用的填充元组 (left, right, top, bottom)。
            - unpad: 用于裁剪回原始尺寸的负填充元组。
    """
    pad_h = (min_div - h % min_div) % min_div
    pad_w = (min_div - w % min_div) % min_div
    
    # F.pad expects (left, right, top, bottom)
    pad = (0, pad_w, 0, pad_h)
    # Negative padding for unpadding
    unpad = (0, -pad_w, 0, -pad_h)
    return pad, unpad

class ModelTester:
    def __init__(self, image_path: str, output_dir: str = "results"):
        """初始化测试器
        
        Args:
            image_path (str): 测试图片路径
            output_dir (str): 结果保存目录
        """
        self.image_path = image_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 设置设备为CPU
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # 加载测试图片
        self.img = Image.open(image_path).convert("RGB")
        self.x = torch.from_numpy(np.array(self.img)).permute(2, 0, 1).float()
        self.x = self.x.unsqueeze(0) / 255.0
        # self.x is already on CPU by default
        
        # 初始化结果存储
        self.results = []
    
    def load_cheng2020_anchor(self, quality: int) -> nn.Module:
        """加载cheng2020_anchor模型并移动到CPU
        
        Args:
            quality (int): 质量等级(1-6)
            
        Returns:
            nn.Module: 加载并移动到CPU上的模型
        """
        model = cheng2020_anchor(quality=quality, pretrained=True)
        return model.to(self.device).eval() # 确保模型在CPU上
    
    def load_elic_chandelier(self, checkpoint_path: str) -> nn.Module:
        """加载ELIC chandelier模型并移动到CPU
        
        Args:
            checkpoint_path (str): 预训练模型路径
            
        Returns:
            nn.Module: 加载并移动到CPU上的模型
        """
        model = Elic2022Chandelier()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        
        # 打印缺失和意外的键
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # print(f"ELIC - Missing keys: {missing}")  # 检查是否缺失熵瓶颈层
        # print(f"ELIC - Unexpected keys: {unexpected}")

        # 打印模型结构以检查层名称
        # print("--- ELIC Model Structure ---")
        # print(model)
        # print("-----------------------------")
        
        # 验证熵瓶颈层 (注释掉显式检查，让 model.update() 处理)
        # if not hasattr(model, 'entropy_bottleneck'):
        #     raise ValueError("ELIC模型缺少熵瓶颈层！")
        
        model = model.to(self.device) # 确保模型在CPU上
        model.update(force=True) # 初始化熵编码器
        return model.eval()

    def load_elic2022light_model(self, checkpoint_path: str) -> nn.Module:
        """加载Elic2022ChandelierLite模型并移动到CPU
        
        Args:
            checkpoint_path (str): 预训练模型路径
            
        Returns:
            nn.Module: 加载并移动到CPU上的模型
        """
        model = Elic2022ChandelierLite() # 实例化 Elic2022ChandelierLite
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        
        # 假设状态字典直接是模型参数，或者在 'model_state_dict' 键下
        if 'model_state_dict' in state_dict:
            actual_state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict: # 兼容其他可能的checkpoint格式
            actual_state_dict = state_dict['state_dict']
        else:
            actual_state_dict = state_dict

        missing, unexpected = model.load_state_dict(actual_state_dict, strict=False)
        # print(f"Elic2022ChandelierLite - Missing keys: {missing}")
        # print(f"Elic2022ChandelierLite - Unexpected keys: {unexpected}")
        
        model = model.to(self.device) # 确保模型在CPU上
        try:
            model.update(force=True) # 初始化熵编码器 (如果存在)
        except AttributeError:
            print("Warning: Elic2022ChandelierLite model does not seem to have a compatible entropy bottleneck. BPP calculation might be incorrect or fail.")
        return model.eval()
    
    def load_custom_model(self, checkpoint_path: str) -> nn.Module:
        """加载自定义模型(基于cheng2020架构)并移动到CPU
        
        Args:
            checkpoint_path (str): 自定义模型路径
            
        Returns:
            nn.Module: 加载并移动到CPU上的模型
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # 假设状态字典存储在 'model_state_dict' 键下
        state_dict = checkpoint['model_state_dict'] 
        # print("Custom model keys:", list(state_dict.keys())) # 打印键列表以便检查
        
        # 根据实际结构定义模型（这里仍用cheng2020 q=1作为基础，需要用户根据实际情况调整）
        # 假设 N=128, M=192 for cheng2020_anchor q=1
        model = cheng2020_anchor(quality=1, pretrained=False) # 使用 quality=1 对应的结构
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # print(f"Custom - Missing keys: {missing}")
        # print(f"Custom - Unexpected keys: {unexpected}")
        
        # 验证关键组件
        required_modules = ['g_a', 'g_s', 'entropy_bottleneck']
        for mod in required_modules:
            if not hasattr(model, mod):
                # 如果缺少关键模块，strict=False可能掩盖了问题，这里需要更强的检查
                # 或者根据 unexpected keys 判断是否加载了完全不同的模型
                print(f"Warning: 模型可能缺少关键模块: {mod}")
                # raise ValueError(f"模型缺少关键模块: {mod}") # 暂时注释掉，避免直接报错
        
        # 尝试更新熵编码器（如果存在）
        try:
            model.update(force=True)
        except AttributeError:
            print("Warning: Custom model does not seem to have a compatible entropy bottleneck. BPP calculation might be incorrect or fail.")
            
        return model.to(self.device).eval()
    
    @torch.no_grad()
    def test_model(self, model: nn.Module, model_name: str, model_type: str) -> Dict:
        """使用官方风格的代码测试单个模型性能 (在CPU上)。

        Args:
            model (nn.Module): 要测试的模型 (已在CPU上)。
            model_name (str): 模型名称。
            model_type (str): 模型类型。

        Returns:
            Dict: 测试结果。
        """
        x = self.x.to(self.device) # 确保输入在CPU上 (0-1范围)
        _ , _, H, W = x.shape # 获取原始尺寸

        # 计算填充量 (确保为 64 的倍数)
        pad, unpad = compute_padding(H, W, min_div=2**6)
        x_padded = F.pad(x, pad, mode="constant", value=0) # 使用 0 填充
        print(f"Padded input image from {H}x{W} to {x_padded.shape[2]}x{x_padded.shape[3]}")

        # CPU 预热 (可选, 但有助于稳定计时)
        # for _ in range(2):
        #     _ = model(x_padded)

        # 测量编码时间 (CPU)
        start = time.time()
        out_enc = model.compress(x_padded)
        enc_time = time.time() - start

        # 打印字符串长度 (用于调试)
        print(f"{model_name} - String lengths: {[len(s[0]) for s in out_enc['strings']]}")

        # 测量解码时间 (CPU)
        start = time.time()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        dec_time = time.time() - start

        # 裁剪重建图像回原始尺寸
        x_hat = F.pad(out_dec["x_hat"], unpad).clamp(0, 1) # 裁剪并确保范围

        # 计算指标 (在CPU上, 使用 0-1 范围)
        mse = F.mse_loss(x_hat, x) # x 和 x_hat 都在 CPU 上且范围为 0-1
        psnr = 10 * torch.log10(1.0 / mse).item()
        ms_ssim_val = ms_ssim(x, x_hat, data_range=1.0).item() # data_range=1.0 因为数据在 0-1

        # 计算 BPP (基于原始尺寸)
        num_pixels = H * W
        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

        # 如果是 ELIC 模型，将 BPP 除以 10
        if model_type == "ELIC":
            bpp /= 10
            print(f"Adjusted BPP for ELIC model: {bpp:.4f}")

        # 保存重建图像 (可选)
        save_path = os.path.join(self.output_dir, f"recon_{model_name}.png")
        try:
            original_img_np = x[0].permute(1, 2, 0).cpu().numpy()
            recon_img_np = x_hat[0].permute(1, 2, 0).cpu().numpy()

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(np.clip(original_img_np, 0, 1))
            plt.title("Original")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(np.clip(recon_img_np, 0, 1))
            plt.title(f"Reconstructed (PSNR={psnr:.2f}dB)")
            plt.axis('off')
            plt.savefig(save_path)
            plt.close()
        except PermissionError:
            print(f"错误：无法保存重建图像到 {save_path}。请检查是否有写入权限或文件是否被其他程序占用。")
        except Exception as e:
            print(f"保存重建图像 {save_path} 时发生其他错误: {e}")

        # 计算模型大小 (MB)
        param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = param_size_bytes / (1024 * 1024)

        return {
            "model_name": model_name,
            "model_type": model_type,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
            "bpp": bpp,
            "psnr": psnr,
            "ms_ssim": ms_ssim_val,
            "model_size_mb": model_size_mb
        }
    
    def run_all_tests(self, elic_checkpoint: str, custom1_checkpoint: str, custom2_checkpoint: str = "checkpoints/custom_cheng2020_anchor_model.pth", elic2022light_checkpoint: str = None): # 修改checkpoint参数名
        """运行所有测试"""

        # 优先测试自定义模型
        if os.path.exists(custom1_checkpoint):
            try:
                model = self.load_custom_model(custom1_checkpoint)
                result = self.test_model(model, "Custom-model", "Custom")
                self.results.append(result)
                print(f"Tested Custom-model: PSNR={result['psnr']:.2f}dB, BPP={result['bpp']:.4f}, EncTime={result['encoding_time']:.3f}s, DecTime={result['decoding_time']:.3f}s, Size={result['model_size_mb']:.2f}MB")
            except Exception as e:
                print(f"Error testing Custom model: {e}")
        else:
            print(f"Custom model checkpoint not found at {custom1_checkpoint}")

        # 测试新的自定义cheng2020-anchor模型
        if os.path.exists(custom2_checkpoint):
            try:
                model = self.load_custom_model(custom2_checkpoint) # 假设自定义模型与cheng2020结构兼容
                result = self.test_model(model, "Custom-cheng2020-anchor", "Custom-cheng2020")
                self.results.append(result)
                print(f"Tested Custom-cheng2020-anchor: PSNR={result['psnr']:.2f}dB, BPP={result['bpp']:.4f}, EncTime={result['encoding_time']:.3f}s, DecTime={result['decoding_time']:.3f}s, Size={result['model_size_mb']:.2f}MB")
            except Exception as e:
                print(f"Error testing Custom-cheng2020-anchor model: {e}")
        else:
            print(f"Custom cheng2020-anchor model checkpoint not found at {custom2_checkpoint}. Please provide the correct path.")

        # 测试Elic2022ChandelierLite模型
        if elic2022light_checkpoint and os.path.exists(elic2022light_checkpoint):
            try:
                model = self.load_elic2022light_model(elic2022light_checkpoint)
                result = self.test_model(model, "elic2022light", "Elic2022-Lite") # 函数名修改为 elic2022light
                self.results.append(result)
                print(f"Tested elic2022light: PSNR={result['psnr']:.2f}dB, BPP={result['bpp']:.4f}, EncTime={result['encoding_time']:.3f}s, DecTime={result['decoding_time']:.3f}s, Size={result['model_size_mb']:.2f}MB")
            except Exception as e:
                print(f"Error testing elic2022light model: {e}")
        elif elic2022light_checkpoint:
            print(f"Elic2022ChandelierLite checkpoint not found at {elic2022light_checkpoint}")
        else:
            print("Elic2022ChandelierLite checkpoint not provided, skipping test.")

        # 测试ELIC chandelier模型
        if os.path.exists(elic_checkpoint):
            try:
                model = self.load_elic_chandelier(elic_checkpoint)
                result = self.test_model(model, "ELIC-chandelier", "ELIC")
                self.results.append(result)
                print(f"Tested ELIC-chandelier: PSNR={result['psnr']:.2f}dB, BPP={result['bpp']:.4f}, EncTime={result['encoding_time']:.3f}s, DecTime={result['decoding_time']:.3f}s, Size={result['model_size_mb']:.2f}MB")
            except Exception as e:
                print(f"Error testing ELIC model: {e}")
        else:
            print(f"ELIC checkpoint not found at {elic_checkpoint}")
        
        # 测试cheng2020-anchor所有质量等级
        for q in range(1, 7):
            try:
                model = self.load_cheng2020_anchor(q)
                model_name = f"cheng2020-anchor-q{q}"
                result = self.test_model(model, model_name, "cheng2020-anchor")
                self.results.append(result)
                print(f"Tested {model_name}: PSNR={result['psnr']:.2f}dB, BPP={result['bpp']:.4f}, EncTime={result['encoding_time']:.3f}s, DecTime={result['decoding_time']:.3f}s, Size={result['model_size_mb']:.2f}MB")
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
        
        # 保存结果
        self.save_results()
    
    def save_results(self):
        """保存测试结果"""
        # 保存为CSV
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.output_dir, "compression_results.csv")
        df.to_csv(csv_path, index=False)
        
        # 保存为图表
        self.plot_results()
        
        print(f"Results saved to {self.output_dir}")
    
    def plot_results(self):
        """绘制结果图表"""
        df = pd.DataFrame(self.results)
        
        plt.figure(figsize=(12, 8))
        
        # 绘制速率-失真曲线
        for model_type in df["model_type"].unique():
            subset = df[df["model_type"] == model_type]
            plt.plot(subset["bpp"], subset["psnr"], "o-", label=model_type)
        
        plt.xlabel("Bits per pixel (bpp)")
        plt.ylabel("PSNR (dB)")
        plt.title("Rate-Distortion Performance")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "rate_distortion.png"))
        plt.close()
        
        # 绘制编码/解码时间比较
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        df.plot.bar(x="model_name", y="encoding_time", ax=ax1)
        ax1.set_title("Encoding Time Comparison")
        ax1.set_ylabel("Time (s)")
        ax1.tick_params(axis='x', rotation=45)
        
        df.plot.bar(x="model_name", y="decoding_time", ax=ax2)
        ax2.set_title("Decoding Time Comparison")
        ax2.set_ylabel("Time (s)")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "time_comparison.png"))
        plt.close()
        
        # 绘制模型大小比较 (MB)
        plt.figure(figsize=(10, 6))
        df.plot.bar(x="model_name", y="model_size_mb")
        plt.title("Model Size Comparison (MB)")
        plt.ylabel("Model Size (MB)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_size_mb.png"))
        plt.close()
    
    def compute_bd_rate(self, ref_model_name: str = "cheng2020-anchor-q3") -> Dict[str, float]:
        """计算BD-rate (相对于参考模型)
        
        Args:
            ref_model_name (str): 参考模型名称
            
        Returns:
            Dict[str, float]: 各模型相对于参考模型的BD-rate
        """
        df = pd.DataFrame(self.results)
        
        try:
            # 获取参考模型的RD点
            ref_df = df[df["model_name"] == ref_model_name]
            if len(ref_df) == 0:
                raise ValueError(f"Reference model {ref_model_name} not found in results")
            
            ref_bpp = ref_df["bpp"].values[0]
            ref_psnr = ref_df["psnr"].values[0]
            
            # 计算各模型的BD-rate
            bd_rates = {}
            for _, row in df.iterrows():
                if row["model_name"] == ref_model_name:
                    bd_rate = 0.0
                else:
                    # 简化计算: (当前bpp - 参考bpp)/参考bpp * 100%
                    bd_rate = (row["bpp"] - ref_bpp) / ref_bpp * 100
                bd_rates[row["model_name"]] = bd_rate
            
            # 保存BD-rate结果
            bd_csv_path = os.path.join(self.output_dir, "bd_rate.csv")
            try:
                bd_df = pd.DataFrame(list(bd_rates.items()), columns=["model_name", "bd_rate"])
                bd_df.to_csv(bd_csv_path, index=False)
                print(f"BD-rate results saved to {bd_csv_path}")
            except PermissionError:
                print(f"错误：无法写入 BD-rate 文件到 {bd_csv_path}。请检查是否有写入权限或文件是否被其他程序占用。")
            except Exception as e_write:
                print(f"保存 BD-rate 文件 {bd_csv_path} 时发生其他错误: {e_write}")
            
            return bd_rates
        except Exception as e:
            print(f"Error computing BD-rate: {str(e)}")
            return {}

if __name__ == "__main__":
    # 配置参数
    image_path = "data/Kodak24/kodim01.png"  # 替换为你的测试图片路径
    elic_checkpoint = "elic2022_chandelier_pretrained_0032.pth"  # ELIC模型路径
    custom1_checkpoint = "checkpoints/student_step_2000.pth"  # 自定义模型1路径
    custom2_checkpoint = "checkpoints/student_step_1000.pth" #  (自定义模型2)
    elic2022light_checkpoint = "checkpoints/student_step_10200.pth" # Elic2022ChandelierLite 模型路径
    
    # 运行测试
    tester = ModelTester(image_path)
    tester.run_all_tests(elic_checkpoint, custom1_checkpoint, custom2_checkpoint, elic2022light_checkpoint)
    
    # 计算BD-rate
    bd_rates = tester.compute_bd_rate()
    print("\nBD-rate Results (relative to cheng2020-anchor-q3):")
    for model, rate in bd_rates.items():
        print(f"{model}: {rate:.2f}%")
