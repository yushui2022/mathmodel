"""
可视化模块 - 为数学建模论文生成各种图表
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os
from matplotlib import font_manager

# 设置图表样式
try:
    if 'seaborn-v0_8-darkgrid' in plt.style.available:
        plt.style.use('seaborn-v0_8-darkgrid')
    elif 'seaborn-darkgrid' in plt.style.available:
        plt.style.use('seaborn-darkgrid')
    else:
        plt.style.use('default')
except:
    plt.style.use('default')

# =========================
# 字体：自动选择可用中文字体（注意：放在 style.use 之后，避免被样式覆盖）
# =========================
def _pick_available_chinese_font() -> str | None:
    """
    从系统已安装字体中挑选一个可用的中文字体名称（返回字体 family name）。
    若找不到则返回 None。
    """
    # 常见中文字体优先级（Windows 优先微软雅黑/黑体，Mac/Linux 也做了兼容）
    preferred = [
        "Microsoft YaHei",
        "微软雅黑",
        "SimHei",
        "黑体",
        "FangSong",
        "KaiTi",
        "宋体",
        "STHeiti",
        "PingFang SC",
        "Hiragino Sans GB",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]

    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    for name in preferred:
        if name in available:
            return name
    return None


_ch_font = _pick_available_chinese_font()
if _ch_font:
    matplotlib.rcParams["font.family"] = "sans-serif"
    # 优先把检测到的字体放第一位，后面再加兜底
    matplotlib.rcParams["font.sans-serif"] = [_ch_font, "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
else:
    matplotlib.rcParams["font.family"] = "sans-serif"
    # 如果系统没有任何中文字体（极少见/精简系统），至少不让程序崩
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

# 解决负号显示问题
matplotlib.rcParams["axes.unicode_minus"] = False


class ChartGenerator:
    """图表生成器 - 生成各种类型的图表"""
    
    def __init__(self, output_dir: str = "charts"):
        """
        初始化图表生成器
        :param output_dir: 图表保存目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.chart_counter = 0
    
    def _get_next_filename(self, prefix: str = "chart", extension: str = "png") -> str:
        """生成下一个图表文件名"""
        self.chart_counter += 1
        return os.path.join(self.output_dir, f"{prefix}_{self.chart_counter}.{extension}")
    
    def plot_time_series(self, df: pd.DataFrame, time_col: str = 'time', 
                        value_col: str = 'value', title: str = "时间序列图",
                        show_trend: bool = True, show_seasonal: bool = False) -> str:
        """
        绘制时间序列图
        :param df: DataFrame
        :param time_col: 时间列名
        :param value_col: 数值列名
        :param title: 图表标题
        :param show_trend: 是否显示趋势线
        :param show_seasonal: 是否显示季节性成分
        :return: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df[time_col], df[value_col], label='原始数据', linewidth=2, alpha=0.7)
        
        if show_trend and 'trend' in df.columns:
            ax.plot(df[time_col], df['trend'], label='趋势线', linestyle='--', linewidth=2)
        
        if show_seasonal and 'seasonal' in df.columns:
            ax.plot(df[time_col], df['seasonal'], label='季节性成分', linestyle=':', linewidth=1.5)
        
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        filename = self._get_next_filename("timeseries")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str,
                    title: str = "散点图", show_regression: bool = True) -> str:
        """
        绘制散点图（可选回归线）
        :param df: DataFrame
        :param x_col: X轴列名
        :param y_col: Y轴列名
        :param title: 图表标题
        :param show_regression: 是否显示回归线
        :return: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        if show_regression:
            from scipy import stats
            x = df[x_col].values
            y = df[y_col].values
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            line_x = np.linspace(x.min(), x.max(), 100)
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'r--', linewidth=2, 
                   label=f'回归线 (R²={r_value**2:.3f})')
            ax.legend()
        
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        filename = self._get_next_filename("scatter")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_histogram(self, df: pd.DataFrame, column: str, bins: int = 30,
                      title: str = "直方图", show_stats: bool = True) -> str:
        """
        绘制直方图
        :param df: DataFrame
        :param column: 列名
        :param bins: 分组数
        :param title: 图表标题
        :param show_stats: 是否显示统计信息
        :return: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n, bins, patches = ax.hist(df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
        
        if show_stats:
            mean_val = df[column].mean()
            median_val = df[column].median()
            ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'均值: {mean_val:.2f}')
            ax.axvline(median_val, color='g', linestyle='--', linewidth=2, label=f'中位数: {median_val:.2f}')
            ax.legend()
        
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        filename = self._get_next_filename("histogram")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_boxplot(self, df: pd.DataFrame, column: Optional[str] = None,
                    by: Optional[str] = None, title: str = "箱线图") -> str:
        """
        绘制箱线图
        :param df: DataFrame
        :param column: 数值列名（如果为None，绘制所有数值列）
        :param by: 分组列名（可选）
        :param title: 图表标题
        :return: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if column:
            if by:
                df.boxplot(column=column, by=by, ax=ax)
            else:
                ax.boxplot(df[column].dropna(), labels=[column])
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ax.boxplot([df[col].dropna() for col in numeric_cols], labels=numeric_cols)
        
        ax.set_ylabel('数值', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        filename = self._get_next_filename("boxplot")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, title: str = "相关性热力图") -> str:
        """
        绘制相关性热力图
        :param df: DataFrame
        :param title: 图表标题
        :return: 保存的文件路径
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # 设置刻度标签
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # 添加数值标注
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax, label='相关系数')
        
        filename = self._get_next_filename("heatmap")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str,
                      title: str = "柱状图", horizontal: bool = False) -> str:
        """
        绘制柱状图
        :param df: DataFrame
        :param x_col: X轴列名
        :param y_col: Y轴列名
        :param title: 图表标题
        :param horizontal: 是否水平显示
        :return: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if horizontal:
            ax.barh(df[x_col], df[y_col], edgecolor='black', alpha=0.7)
            ax.set_xlabel(y_col, fontsize=12)
            ax.set_ylabel(x_col, fontsize=12)
        else:
            ax.bar(df[x_col], df[y_col], edgecolor='black', alpha=0.7)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            plt.xticks(rotation=45, ha='right')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        filename = self._get_next_filename("barchart")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_line_comparison(self, data_dict: Dict[str, pd.Series], 
                           title: str = "对比折线图", xlabel: str = "X轴",
                           ylabel: str = "Y轴") -> str:
        """
        绘制多条折线对比图
        :param data_dict: 数据字典 {标签: Series}
        :param title: 图表标题
        :param xlabel: X轴标签
        :param ylabel: Y轴标签
        :return: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for label, series in data_dict.items():
            ax.plot(series.values, label=label, linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        filename = self._get_next_filename("comparison")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_3d_surface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       title: str = "3D曲面图") -> str:
        """
        绘制3D曲面图
        :param x: X坐标数组
        :param y: Y坐标数组
        :param z: Z值数组（2D）
        :param title: 图表标题
        :return: 保存的文件路径
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, z, cmap='viridis', alpha=0.8, edgecolor='none')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        filename = self._get_next_filename("3d_surface")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename


# 导出主要类
__all__ = ['ChartGenerator']

