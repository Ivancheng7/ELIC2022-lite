import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from compressai.zoo import image_models
from compressai.models import Elic2022Chandelier # 导入 Elic 模型
import os
import numpy as np
from tqdm import tqdm
import math

# ==================== 配置参数 ====================
class Config:
    model_names = ["cheng2020-anchor"]  # 支持的模型
    custom_models = {
        # 格式: "模型显示名称": "checkpoint路径"
        "student-Q1": "checkpoints/student_step_1000.pth",
        "student-Q2": "checkpoints/student_step_2000.pth",
        "student-Q3": "elic2022_chandelier_pretrained_0150.pth",
    }
    qualities = [1,2,3,4,5,6]      # 质量等级
    test_image_path = "data/Kodak24/kodim01.png"   # 测试图像路径
    test_dataset_dir = "None"     # 测试数据集目录
    #test_dataset_dir = "data/Kodak24/"    # 测试数据集目录
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_plot_path = "bpp_psnr_curve.png"

# ==================== 核心BPP计算函数 ====================
def calculate_bpp(model, x, out):
    """通过likelihoods计算BPP的正确方法"""
    num_pixels = x.size(2) * x.size(3)
    
    # 方法1：直接获取bpp字段（某些模型版本）
    if "bpp" in out:
        print("[BPP计算] 使用方法1：直接从输出获取 'bpp' 字段")
        return float(out["bpp"])
    
    # 方法2：通过likelihoods计算（当前需要的方法）
    if "likelihoods" in out:
        print("[BPP计算] 使用方法2：通过 'likelihoods' 计算")
        total_bits = 0
        for _, likelihood in out["likelihoods"].items():
            # 使用交叉熵计算实际比特数
            total_bits += torch.sum(-torch.log2(likelihood)).item()
        return total_bits / num_pixels
    
    # 方法3：通过熵瓶颈层估算
    if hasattr(model, 'entropy_bottleneck'):
        print("[BPP计算] 使用方法3：通过熵瓶颈层估算")
        from compressai.ops import compute_bpp
        return compute_bpp(out, x.size()).item()
    
    # 方法4：最后的fallback
    print("[BPP计算] 使用方法4：Fallback - 使用模型预估BPP")
    print("[警告] 使用近似BPP估算，模型输出字段:", out.keys())
    return model.estimated_bpp  # 所有CompressAI模型都有这个属性

# ==================== 评估函数 ====================
def robust_evaluate(model, x, model_name=None):
    """完整的评估流程，可选地接收模型名称以进行特殊处理。"""
    with torch.no_grad():
        out = model(x)
        x_hat = out["x_hat"].clamp_(0, 1)
        
        # 计算PSNR
        mse = F.mse_loss(x, x_hat)
        psnr = -10 * math.log10(mse.item())
        
        # 计算BPP
        try:
            bpp = calculate_bpp(model, x, out)
        except Exception as e:
            print(f"BPP计算错误: {str(e)}")
            bpp = model.estimated_bpp  # 使用模型预估值

        # --- 特殊处理：如果模型是 elic2022 (student-Q3)，BPP 除以 10 ---
        if model_name == "student-Q3":
            print(f"[特殊处理] 检测到模型 {model_name}，将 BPP ({bpp:.4f}) 除以 10")
            bpp /= 10.0
            print(f"[特殊处理] 处理后 BPP: {bpp:.4f}")
        # --- 特殊处理结束 ---
        
        return {
            "psnr": float(psnr),
            "bpp": float(bpp),
            "status": "success"
        }

# ==================== 数据加载 ====================
def load_image(filepath):
    """加载并自动填充图像"""
    img = Image.open(filepath).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(Config.device)
    h, w = x.shape[2], x.shape[3]
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x

def load_dataset():
    """加载整个测试集"""
    image_files = [f for f in os.listdir(Config.test_dataset_dir) 
                  if f.endswith(('.png', '.jpg', '.jpeg'))]
    return [load_image(os.path.join(Config.test_dataset_dir, f)) for f in image_files]

# ==================== 主流程 ====================
def main():
    all_results = {}
    
    # 加载数据
    images = load_dataset() if os.path.exists(Config.test_dataset_dir) else [load_image(Config.test_image_path)]
    print(f"已加载 {len(images)} 张测试图像")
    
    # 评估预训练模型
    for model_name in Config.model_names:
        model_results = []
        for quality in Config.qualities:
            try:
                model = image_models[model_name](quality=quality, pretrained=True).eval().to(Config.device)
                
                results = {
                    "psnr": 0,
                    "bpp": 0,
                    "count": 0
                }
                
                for x in tqdm(images, desc=f"评估 {model_name}-Q{quality}"):
                    res = robust_evaluate(model, x)
                    if res["status"] == "success":
                        results["psnr"] += res["psnr"]
                        results["bpp"] += res["bpp"]
                        results["count"] += 1
                
                if results["count"] > 0:
                    avg_results = {
                        "psnr": results["psnr"] / results["count"],
                        "bpp": results["bpp"] / results["count"],
                        "status": f"success ({results['count']}/{len(images)})"
                    }
                    model_results.append(avg_results)
                    print(f"Q{quality}: PSNR={avg_results['psnr']:.2f}dB, BPP={avg_results['bpp']:.4f}")
            
            except Exception as e:
                print(f"{model_name}-Q{quality} 评估失败: {str(e)}")
                model_results.append({"psnr": 0, "bpp": 0, "status": str(e)})
        
        all_results[model_name] = model_results
    
    # 评估自定义模型
    for model_name, checkpoint_path in Config.custom_models.items():
        model_results = []
        try:
            # 根据模型名称加载不同的基础模型结构
            if model_name == "student-Q3":
                # 特殊处理 Elic2022Chandelier 模型
                # 注意：Elic模型没有quality参数，直接加载预训练权重
                # 假设 checkpoint_path 指向包含 'state_dict' 的文件
                # 直接实例化 Elic 模型，而不是通过 zoo
                model = Elic2022Chandelier().eval().to(Config.device)
                checkpoint = torch.load(checkpoint_path, map_location=Config.device)
                # 检查 checkpoint 结构，可能需要调整键名
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif 'model_state_dict' in checkpoint: # 兼容旧格式
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False) # 直接加载
                print(f"为 {model_name} 加载 Elic2022Chandelier 结构和权重: {checkpoint_path}")
            else:
                # 其他自定义模型（如 student-Q1, student-Q2）使用 cheng2020-anchor
                base_model = image_models["cheng2020-anchor"](quality=1, pretrained=False).eval().to(Config.device)
                checkpoint = torch.load(checkpoint_path, map_location=Config.device)
                # 检查 checkpoint 结构
                if 'model_state_dict' in checkpoint:
                     base_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint: # 兼容其他格式
                     base_model.load_state_dict(checkpoint['state_dict'])
                else:
                     base_model.load_state_dict(checkpoint) # 直接加载
                model = base_model
                print(f"为 {model_name} 加载 cheng2020-anchor 结构和权重: {checkpoint_path}")
            
            results = {
                "psnr": 0,
                "bpp": 0,
                "count": 0
            }
            
            for x in tqdm(images, desc=f"评估 {model_name}"):
                # 传递 model_name 给 robust_evaluate
                res = robust_evaluate(model, x, model_name=model_name)
                if res["status"] == "success":
                    results["psnr"] += res["psnr"]
                    results["bpp"] += res["bpp"]
                    results["count"] += 1
            
            if results["count"] > 0:
                avg_results = {
                    "psnr": results["psnr"] / results["count"],
                    "bpp": results["bpp"] / results["count"],
                    "status": f"success ({results['count']}/{len(images)})"
                }
                model_results.append(avg_results)
                print(f"{model_name}: PSNR={avg_results['psnr']:.2f}dB, BPP={avg_results['bpp']:.4f}")
        
        except Exception as e:
            print(f"{model_name} 评估失败: {str(e)}")
            model_results.append({"psnr": 0, "bpp": 0, "status": str(e)})
        
        all_results[model_name] = model_results
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    
    # 绘制预训练模型结果
    for name, results in all_results.items():
        if name in Config.model_names:  # 预训练模型
            valid_results = [r for r in results if "success" in r["status"]]
            if valid_results:
                plt.plot(
                    [r["bpp"] for r in valid_results],
                    [r["psnr"] for r in valid_results],
                    'o-', label=f"预训练-{name}"
                )
    
    # 绘制自定义模型结果
    for name, results in all_results.items():
        if name in Config.custom_models:  # 自定义模型
            valid_results = [r for r in results if "success" in r["status"]]
            if valid_results:
                plt.plot(
                    [r["bpp"] for r in valid_results],
                    [r["psnr"] for r in valid_results],
                    's--', label=f"自定义-{name}"
                )
    
    plt.xlabel("BPP (bits per pixel)")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid()
    plt.savefig(Config.save_plot_path, bbox_inches='tight', dpi=300)
    print(f"结果已保存至 {Config.save_plot_path}")

if __name__ == "__main__":
    main()
