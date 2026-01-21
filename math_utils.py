"""
数学工具模块 - 提供各种数学计算和模型求解功能
"""
import numpy as np
from scipy import optimize, stats, linalg
from scipy.optimize import minimize, linprog
from typing import List, Tuple, Dict, Optional, Callable
import pandas as pd


class OptimizationSolver:
    """优化求解器 - 解决各种优化问题"""
    
    @staticmethod
    def linear_programming(c: np.ndarray, A_ub: Optional[np.ndarray] = None,
                          b_ub: Optional[np.ndarray] = None,
                          A_eq: Optional[np.ndarray] = None,
                          b_eq: Optional[np.ndarray] = None,
                          bounds: Optional[List[Tuple]] = None) -> Dict:
        """
        线性规划求解
        :param c: 目标函数系数向量
        :param A_ub: 不等式约束矩阵
        :param b_ub: 不等式约束向量
        :param A_eq: 等式约束矩阵
        :param b_eq: 等式约束向量
        :param bounds: 变量边界 [(min, max), ...]
        :return: 求解结果字典
        """
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        
        return {
            'success': result.success,
            '最优值': float(result.fun) if result.success else None,
            '最优解': result.x.tolist() if result.success else None,
            '状态': result.message
        }
    
    @staticmethod
    def nonlinear_optimization(objective: Callable, x0: np.ndarray,
                              method: str = 'BFGS', bounds: Optional[List[Tuple]] = None,
                              constraints: Optional[List] = None) -> Dict:
        """
        非线性优化求解
        :param objective: 目标函数
        :param x0: 初始值
        :param method: 优化方法
        :param bounds: 变量边界
        :param constraints: 约束条件
        :return: 求解结果字典
        """
        result = minimize(objective, x0, method=method, bounds=bounds, constraints=constraints)
        
        return {
            'success': result.success,
            '最优值': float(result.fun) if result.success else None,
            '最优解': result.x.tolist() if result.success else None,
            '迭代次数': result.nit,
            '状态': result.message
        }
    
    @staticmethod
    def quadratic_programming(P: np.ndarray, q: np.ndarray,
                             A: Optional[np.ndarray] = None,
                             b: Optional[np.ndarray] = None,
                             lb: Optional[np.ndarray] = None,
                             ub: Optional[np.ndarray] = None) -> Dict:
        """
        二次规划求解（简化版，使用非线性优化）
        :param P: 二次项系数矩阵
        :param q: 一次项系数向量
        :param A: 约束矩阵
        :param b: 约束向量
        :param lb: 下界
        :param ub: 上界
        :return: 求解结果字典
        """
        n = len(q)
        x0 = np.zeros(n)
        
        def objective(x):
            return 0.5 * x.T @ P @ x + q.T @ x
        
        bounds = [(lb[i] if lb is not None else None, 
                  ub[i] if ub is not None else None) for i in range(n)]
        
        constraints = []
        if A is not None and b is not None:
            constraints.append({'type': 'eq', 'fun': lambda x: A @ x - b})
        
        return OptimizationSolver.nonlinear_optimization(objective, x0, bounds=bounds, constraints=constraints)


class StatisticalModel:
    """统计模型 - 各种统计分析和建模"""
    
    @staticmethod
    def linear_regression(X: np.ndarray, y: np.ndarray) -> Dict:
        """
        多元线性回归
        :param X: 特征矩阵 (n_samples, n_features)
        :param y: 目标向量 (n_samples,)
        :return: 回归结果字典
        """
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # 最小二乘求解
        try:
            beta = linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            y_pred = X_with_intercept @ beta
            
            # 计算统计量
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            mse = ss_res / len(y)
            rmse = np.sqrt(mse)
            
            return {
                '系数': beta.tolist(),
                '截距': float(beta[0]),
                'R²': float(r_squared),
                'RMSE': float(rmse),
                'MSE': float(mse),
                '预测值': y_pred.tolist()
            }
        except:
            return {'error': '矩阵求解失败'}
    
    @staticmethod
    def polynomial_regression(x: np.ndarray, y: np.ndarray, degree: int = 2) -> Dict:
        """
        多项式回归
        :param x: 自变量
        :param y: 因变量
        :param degree: 多项式次数
        :return: 回归结果字典
        """
        # 构建多项式特征矩阵
        X_poly = np.column_stack([x ** i for i in range(degree + 1)])
        
        beta = linalg.lstsq(X_poly, y, rcond=None)[0]
        y_pred = X_poly @ beta
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            '多项式系数': beta.tolist(),
            'R²': float(r_squared),
            '预测值': y_pred.tolist()
        }
    
    @staticmethod
    def hypothesis_test(data1: np.ndarray, data2: Optional[np.ndarray] = None,
                       test_type: str = 't-test') -> Dict:
        """
        假设检验
        :param data1: 第一组数据
        :param data2: 第二组数据（可选，用于双样本检验）
        :param test_type: 检验类型 ('t-test', 'wilcoxon', 'mannwhitney')
        :return: 检验结果字典
        """
        if test_type == 't-test':
            if data2 is None:
                # 单样本t检验（与0比较）
                statistic, p_value = stats.ttest_1samp(data1, 0)
            else:
                # 双样本t检验
                statistic, p_value = stats.ttest_ind(data1, data2)
            
            return {
                '统计量': float(statistic),
                'p值': float(p_value),
                '显著性': '显著' if p_value < 0.05 else '不显著'
            }
        
        elif test_type == 'wilcoxon' and data2 is not None:
            statistic, p_value = stats.wilcoxon(data1, data2)
            return {
                '统计量': float(statistic),
                'p值': float(p_value),
                '显著性': '显著' if p_value < 0.05 else '不显著'
            }
        
        elif test_type == 'mannwhitney' and data2 is not None:
            statistic, p_value = stats.mannwhitneyu(data1, data2)
            return {
                '统计量': float(statistic),
                'p值': float(p_value),
                '显著性': '显著' if p_value < 0.05 else '不显著'
            }
        
        return {'error': '不支持的检验类型'}


class NumericalMethods:
    """数值方法 - 各种数值计算"""
    
    @staticmethod
    def solve_ode(func: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        求解常微分方程（使用欧拉法）
        :param func: 微分方程 dy/dt = func(t, y)
        :param y0: 初始条件
        :param t: 时间点数组
        :return: 解数组
        """
        from scipy.integrate import odeint
        solution = odeint(func, y0, t)
        return solution
    
    @staticmethod
    def numerical_integration(func: Callable, a: float, b: float, n: int = 1000) -> float:
        """
        数值积分（辛普森法）
        :param func: 被积函数
        :param a: 积分下限
        :param b: 积分上限
        :param n: 分段数
        :return: 积分值
        """
        from scipy.integrate import quad
        result, error = quad(func, a, b)
        return result
    
    @staticmethod
    def root_finding(func: Callable, x0: float, method: str = 'newton') -> Dict:
        """
        求根
        :param func: 函数
        :param x0: 初始值
        :param method: 方法 ('newton', 'brentq', 'bisect')
        :return: 根和相关信息
        """
        if method == 'newton':
            from scipy.optimize import newton
            root = newton(func, x0)
        elif method == 'brentq':
            root = optimize.brentq(func, x0 - 1, x0 + 1)
        elif method == 'bisect':
            root = optimize.bisect(func, x0 - 1, x0 + 1)
        else:
            return {'error': '不支持的方法'}
        
        return {
            '根': float(root),
            '函数值': float(func(root))
        }


class MatrixOperations:
    """矩阵运算 - 各种矩阵操作"""
    
    @staticmethod
    def eigenvalue_decomposition(A: np.ndarray) -> Dict:
        """
        特征值分解
        :param A: 方阵
        :return: 特征值和特征向量
        """
        eigenvals, eigenvecs = linalg.eig(A)
        
        return {
            '特征值': eigenvals.tolist(),
            '特征向量': eigenvecs.tolist(),
            '特征值（实数部分）': np.real(eigenvals).tolist()
        }
    
    @staticmethod
    def svd_decomposition(A: np.ndarray) -> Dict:
        """
        奇异值分解
        :param A: 矩阵
        :return: U, S, V矩阵
        """
        U, s, Vt = linalg.svd(A)
        
        return {
            'U矩阵': U.tolist(),
            '奇异值': s.tolist(),
            'V转置矩阵': Vt.tolist()
        }
    
    @staticmethod
    def matrix_inverse(A: np.ndarray) -> np.ndarray:
        """
        矩阵求逆
        :param A: 方阵
        :return: 逆矩阵
        """
        try:
            return linalg.inv(A)
        except:
            return None


class ModelEvaluator:
    """模型评估器 - 评估模型性能"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        计算回归模型评估指标
        :param y_true: 真实值
        :param y_pred: 预测值
        :return: 评估指标字典
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 平均绝对百分比误差
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R²': float(r_squared),
            'MAPE(%)': float(mape)
        }
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None) -> np.ndarray:
        """
        计算混淆矩阵（分类问题）
        :param y_true: 真实标签
        :param y_pred: 预测标签
        :param labels: 标签列表
        :return: 混淆矩阵
        """
        from sklearn.metrics import confusion_matrix as cm
        return cm(y_true, y_pred, labels=labels)


# 导出主要类
__all__ = ['OptimizationSolver', 'StatisticalModel', 'NumericalMethods', 
           'MatrixOperations', 'ModelEvaluator']






