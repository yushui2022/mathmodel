"""
数据分析模块 - 为数学建模论文生成示例数据和统计分析
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json


class DataGenerator:
    """数据生成器 - 根据问题类型生成模拟数据"""
    
    @staticmethod
    def generate_time_series_data(n_points: int = 100, trend: str = "linear", 
                                  noise_level: float = 0.1, seasonality: bool = True) -> pd.DataFrame:
        """
        生成时间序列数据
        :param n_points: 数据点数量
        :param trend: 趋势类型 ('linear', 'exponential', 'polynomial')
        :param noise_level: 噪声水平
        :param seasonality: 是否包含季节性
        :return: DataFrame with 'time' and 'value' columns
        """
        time = np.arange(n_points)
        
        if trend == "linear":
            base = 100 + 2 * time
        elif trend == "exponential":
            base = 100 * np.exp(0.02 * time)
        elif trend == "polynomial":
            base = 100 + 0.1 * time**2
        else:
            base = 100 + time
        
        if seasonality:
            seasonal = 10 * np.sin(2 * np.pi * time / 12)
        else:
            seasonal = 0
        
        noise = np.random.normal(0, noise_level * np.mean(base), n_points)
        value = base + seasonal + noise
        
        return pd.DataFrame({
            'time': time,
            'value': value,
            'trend': base,
            'seasonal': seasonal if seasonality else 0
        })
    
    @staticmethod
    def generate_multivariate_data(n_samples: int = 200, n_features: int = 5, 
                                   correlation: float = 0.5) -> pd.DataFrame:
        """
        生成多变量数据（可用于回归、分类等）
        :param n_samples: 样本数量
        :param n_features: 特征数量
        :param correlation: 特征间相关性
        :return: DataFrame
        """
        # 生成相关特征
        mean = np.zeros(n_features)
        cov = np.eye(n_features) * (1 - correlation) + np.ones((n_features, n_features)) * correlation
        cov[np.diag_indices_from(cov)] = 1
        
        data = np.random.multivariate_normal(mean, cov, n_samples)
        
        # 添加特征名称
        columns = [f'特征{i+1}' for i in range(n_features)]
        df = pd.DataFrame(data, columns=columns)
        
        # 生成目标变量（基于特征的线性组合）
        target = np.sum(data[:, :3], axis=1) + np.random.normal(0, 0.5, n_samples)
        df['目标变量'] = target
        
        return df
    
    @staticmethod
    def generate_category_data(categories: List[str], n_per_category: int = 50) -> pd.DataFrame:
        """
        生成分类数据
        :param categories: 类别列表
        :param n_per_category: 每个类别的样本数
        :return: DataFrame
        """
        data = []
        for cat in categories:
            for _ in range(n_per_category):
                data.append({
                    '类别': cat,
                    '数值1': np.random.normal(50 + hash(cat) % 20, 10),
                    '数值2': np.random.normal(30 + hash(cat) % 15, 8),
                    '数值3': np.random.uniform(0, 100)
                })
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_optimization_data(n_items: int = 20) -> pd.DataFrame:
        """
        生成优化问题数据（如资源分配、背包问题等）
        :param n_items: 物品数量
        :return: DataFrame
        """
        return pd.DataFrame({
            '物品编号': range(1, n_items + 1),
            '价值': np.random.uniform(10, 100, n_items),
            '成本': np.random.uniform(5, 50, n_items),
            '权重': np.random.uniform(1, 10, n_items),
            '需求量': np.random.randint(1, 20, n_items)
        })


class StatisticalAnalyzer:
    """统计分析器 - 对数据进行统计分析"""
    
    @staticmethod
    def basic_statistics(df: pd.DataFrame, column: str) -> Dict:
        """
        计算基本统计量
        :param df: DataFrame
        :param column: 列名
        :return: 统计量字典
        """
        return {
            '均值': float(df[column].mean()),
            '中位数': float(df[column].median()),
            '标准差': float(df[column].std()),
            '最小值': float(df[column].min()),
            '最大值': float(df[column].max()),
            '四分位数Q1': float(df[column].quantile(0.25)),
            '四分位数Q3': float(df[column].quantile(0.75)),
            '偏度': float(df[column].skew()),
            '峰度': float(df[column].kurtosis())
        }
    
    @staticmethod
    def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """
        相关性分析
        :param df: DataFrame
        :return: 相关性矩阵
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].corr()
    
    @staticmethod
    def regression_analysis(df: pd.DataFrame, x_col: str, y_col: str) -> Dict:
        """
        简单线性回归分析
        :param df: DataFrame
        :param x_col: 自变量列名
        :param y_col: 因变量列名
        :return: 回归结果字典
        """
        from scipy import stats
        
        x = df[x_col].values
        y = df[y_col].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return {
            '斜率': float(slope),
            '截距': float(intercept),
            '相关系数R': float(r_value),
            'R²': float(r_value ** 2),
            'p值': float(p_value),
            '标准误差': float(std_err)
        }
    
    @staticmethod
    def time_series_analysis(df: pd.DataFrame, time_col: str = 'time', value_col: str = 'value') -> Dict:
        """
        时间序列分析
        :param df: DataFrame
        :param time_col: 时间列名
        :param value_col: 数值列名
        :return: 分析结果字典
        """
        values = df[value_col].values
        
        # 计算增长率
        growth_rates = np.diff(values) / values[:-1] * 100
        avg_growth_rate = np.mean(growth_rates)
        
        # 计算移动平均
        window = min(7, len(values) // 10)
        moving_avg = pd.Series(values).rolling(window=window).mean().values
        
        return {
            '平均增长率(%)': float(avg_growth_rate),
            '波动率': float(np.std(growth_rates)),
            '趋势': '上升' if avg_growth_rate > 0 else '下降',
            '移动平均': moving_avg.tolist() if len(moving_avg) > 0 else []
        }


class DataProcessor:
    """数据处理器 - 数据清洗和预处理"""
    
    @staticmethod
    def normalize_data(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        数据标准化（Z-score）
        :param df: DataFrame
        :param columns: 要标准化的列，None表示所有数值列
        :return: 标准化后的DataFrame
        """
        df_copy = df.copy()
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            mean = df_copy[col].mean()
            std = df_copy[col].std()
            if std > 0:
                df_copy[col] = (df_copy[col] - mean) / std
        
        return df_copy
    
    @staticmethod
    def min_max_scale(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        最小-最大归一化
        :param df: DataFrame
        :param columns: 要归一化的列
        :return: 归一化后的DataFrame
        """
        df_copy = df.copy()
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            if max_val > min_val:
                df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
        
        return df_copy
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        处理缺失值
        :param df: DataFrame
        :param strategy: 策略 ('mean', 'median', 'mode', 'drop')
        :return: 处理后的DataFrame
        """
        df_copy = df.copy()
        
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_copy[col].isna().any():
                if strategy == 'mean':
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                elif strategy == 'median':
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
                elif strategy == 'mode':
                    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    df_copy.dropna(subset=[col], inplace=True)
        
        return df_copy


# 导出主要类
__all__ = ['DataGenerator', 'StatisticalAnalyzer', 'DataProcessor']

