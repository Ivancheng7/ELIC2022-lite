import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

class ResultsVisualizer:
    def __init__(self, bd_rate_csv_path: str, compression_results_csv_path: str, output_dir: str = "results"):
        """初始化结果可视化器

        Args:
            bd_rate_csv_path (str): 包含BD-rate数据的CSV文件路径。
            compression_results_csv_path (str): 包含BPP和PSNR数据的CSV文件路径。
            output_dir (str): 图表保存目录。
        """
        self.bd_rate_csv_path = bd_rate_csv_path
        self.compression_results_csv_path = compression_results_csv_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.bd_rate_df = None
        self.compression_results_df = None

    def load_bd_rate_results(self):
        """从CSV文件加载BD-rate结果"""
        if not os.path.exists(self.bd_rate_csv_path):
            print(f"错误：BD-rate CSV文件未找到于 {self.bd_rate_csv_path}")
            return False
        try:
            self.bd_rate_df = pd.read_csv(self.bd_rate_csv_path)
            print(f"成功从 {self.bd_rate_csv_path} 加载BD-rate数据。")
            return True
        except Exception as e:
            print(f"加载BD-rate CSV文件 {self.bd_rate_csv_path} 时发生错误: {e}")
            return False

    def load_compression_results(self):
        """从CSV文件加载压缩结果 (BPP, PSNR)"""
        if not os.path.exists(self.compression_results_csv_path):
            print(f"错误：压缩结果CSV文件未找到于 {self.compression_results_csv_path}")
            return False
        try:
            self.compression_results_df = pd.read_csv(self.compression_results_csv_path)
            print(f"成功从 {self.compression_results_csv_path} 加载压缩结果数据。")
            return True
        except Exception as e:
            print(f"加载压缩结果CSV文件 {self.compression_results_csv_path} 时发生错误: {e}")
            return False

    def plot_bd_rate_comparison(self):
        """绘制BD-rate比较图表"""
        if self.bd_rate_df is None or self.bd_rate_df.empty:
            print("错误：没有BD-rate数据可供绘图。请先调用load_bd_rate_results()。")
            return

        if 'model_name' not in self.bd_rate_df.columns or 'bd_rate' not in self.bd_rate_df.columns:
            print("错误：BD-rate CSV文件必须包含 'model_name' 和 'bd_rate' 列。")
            return

        plt.figure(figsize=(10, 6))
        # 根据bd_rate值对数据进行排序，以便更好地可视化
        sorted_df = self.bd_rate_df.sort_values(by='bd_rate')
        
        bars = plt.bar(sorted_df["model_name"], sorted_df["bd_rate"], color='skyblue')
        plt.xlabel("模型名称")
        plt.ylabel("BD-Rate (%)")
        plt.title("模型 BD-Rate 比较 (相对于参考模型)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        
        # 在条形图上显示数值
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', va='bottom' if yval >=0 else 'top', ha='center')

        save_path = os.path.join(self.output_dir, "bd_rate_comparison.png")
        try:
            plt.savefig(save_path)
            print(f"BD-rate比较图表已保存到 {save_path}")
        except PermissionError:
            print(f"错误：无法保存BD-rate图表到 {save_path}。请检查是否有写入权限或文件是否被其他程序占用。")
        except Exception as e:
            print(f"保存BD-rate图表 {save_path} 时发生其他错误: {e}")
        plt.close()

    def plot_bpp_psnr_curve(self):
        """绘制BPP-PSNR曲线图"""
        if self.compression_results_df is None or self.compression_results_df.empty:
            print("错误：没有压缩结果数据可供绘图。请先调用load_compression_results()。")
            return

        required_cols = ['model_name', 'model_type', 'bpp', 'psnr']
        for col in required_cols:
            if col not in self.compression_results_df.columns:
                print(f"错误：压缩结果CSV文件必须包含 '{col}' 列。")
                return

        plt.figure(figsize=(12, 8))
        
        # 绘制速率-失真曲线
        for model_type in self.compression_results_df["model_type"].unique():
            subset = self.compression_results_df[self.compression_results_df["model_type"] == model_type]
            # 按BPP排序以确保线条正确连接
            subset = subset.sort_values(by='bpp') 
            plt.plot(subset["bpp"], subset["psnr"], "o-", label=model_type)
        
        plt.xlabel("Bits per pixel (bpp)")
        plt.ylabel("PSNR (dB)")
        plt.title("Rate-Distortion Performance")
        plt.grid(True)
        plt.legend()
        
        save_path = os.path.join(self.output_dir, "bpp_psnr_curve_from_csv.png")
        try:
            plt.savefig(save_path)
            print(f"BPP-PSNR曲线图已保存到 {save_path}")
        except PermissionError:
            print(f"错误：无法保存BPP-PSNR图表到 {save_path}。请检查是否有写入权限或文件是否被其他程序占用。")
        except Exception as e:
            print(f"保存BPP-PSNR图表 {save_path} 时发生其他错误: {e}")
        plt.close()

    def run_evaluation(self):
        """运行评估流程：加载数据并绘图"""
        if self.load_bd_rate_results():
            self.plot_bd_rate_comparison()
        
        if self.load_compression_results():
            self.plot_bpp_psnr_curve()

if __name__ == "__main__":
    # 配置参数
    bd_rate_csv_file = "results/bd_rate.csv"
    # 假设包含BPP和PSNR数据的文件名为 compression_results.csv，并且在同一results目录下
    compression_results_csv_file = "results/compression_results.csv"
    output_directory = "results" # 图表输出目录

    # 检查BD-rate CSV文件是否存在
    if not os.path.exists(bd_rate_csv_file):
        print(f"错误: BD-rate CSV 文件 '{bd_rate_csv_file}' 未找到.")
        print("请确保文件路径正确，或者该文件由 evaluate_performance.py 生成.")
        # 如果BD-rate文件不存在，我们可能仍然希望尝试绘制BPP-PSNR图

    # 检查压缩结果CSV文件是否存在
    if not os.path.exists(compression_results_csv_file):
        print(f"错误: 压缩结果CSV文件 '{compression_results_csv_file}' 未找到.")
        print("请确保文件路径正确，或者该文件由 evaluate_performance.py 生成.")
        # 如果压缩结果文件不存在，我们可能仍然希望尝试绘制BD-rate图

    # 即使某个文件不存在，也尝试运行，让类内部的检查处理
    visualizer = ResultsVisualizer(
        bd_rate_csv_path=bd_rate_csv_file, 
        compression_results_csv_path=compression_results_csv_file, 
        output_dir=output_directory
    )
    visualizer.run_evaluation()