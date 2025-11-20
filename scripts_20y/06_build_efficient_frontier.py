#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建有效前沿脚本

功能：
1. 使用Markowitz均值-方差优化
2. 扫描不同目标收益率，求解最小方差组合
3. 生成有效前沿曲线数据点
4. 计算最小方差组合(GMV)

输入：
- 处理后数据_20y/summary_stats.csv
- 处理后数据_20y/covariance.csv

输出：
- 处理后数据_20y/efficient_frontier.csv
- 处理后数据_20y/gmv_portfolio.json (全局最小方差组合)
- 处理后数据_20y/frontier_数据分析报告.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize
import json
import warnings

warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = Path(__file__).parent.parent
INPUT_STATS = BASE_DIR / "处理后数据_20y" / "05_统计特征" / r"summary_stats.csv"
INPUT_COV = BASE_DIR / "处理后数据_20y" / "05_统计特征" / r"covariance.csv"
OUTPUT_FRONTIER = BASE_DIR / "处理后数据_20y" / "06_有效前沿" / r"efficient_frontier.csv"
OUTPUT_GMV = BASE_DIR / "处理后数据_20y" / "06_有效前沿" / r"gmv_portfolio.json"
OUTPUT_REPORT = BASE_DIR / "处理后数据_20y" / "06_有效前沿" / r"frontier_数据分析报告.txt"

# 参数配置
NUM_PORTFOLIOS = 50  # 有效前沿点数
ALLOW_SHORT_SELLING = False  # 是否允许卖空
USE_POSITIVE_RETURN_ONLY = False  # 使用全部股票（包括负收益）
TOP_N_STOCKS = None  # 使用全部2092只股票


def portfolio_variance(weights, cov_matrix):
    """计算组合方差"""
    return weights.T @ cov_matrix @ weights


def portfolio_return(weights, mean_returns):
    """计算组合收益"""
    return weights.T @ mean_returns


def minimize_variance_for_target_return(target_return, mean_returns, cov_matrix, allow_short=False):
    """
    对给定目标收益率，求解最小方差组合

    Args:
        target_return: 目标收益率
        mean_returns: 期望收益率向量
        cov_matrix: 协方差矩阵
        allow_short: 是否允许卖空

    Returns:
        result: 优化结果
    """
    n_assets = len(mean_returns)

    # 初始猜测（等权重）
    init_weights = np.ones(n_assets) / n_assets

    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
        {'type': 'eq', 'fun': lambda w: portfolio_return(w, mean_returns) - target_return}  # 目标收益
    ]

    # 边界条件
    if allow_short:
        bounds = tuple((-1, 1) for _ in range(n_assets))  # 允许卖空，但限制范围
    else:
        bounds = tuple((0, 1) for _ in range(n_assets))  # 不允许卖空

    # 优化
    result = minimize(
        lambda w: portfolio_variance(w, cov_matrix),
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    return result


def find_global_minimum_variance(mean_returns, cov_matrix, allow_short=False):
    """
    求解全局最小方差组合(GMV)

    不考虑收益率约束，只求最小方差
    """
    n_assets = len(mean_returns)
    init_weights = np.ones(n_assets) / n_assets

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 权重和为1
    ]

    if allow_short:
        bounds = tuple((-1, 1) for _ in range(n_assets))
    else:
        bounds = tuple((0, 1) for _ in range(n_assets))

    result = minimize(
        lambda w: portfolio_variance(w, cov_matrix),
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    return result


def main():
    print("=" * 60)
    print("构建有效前沿")
    print("=" * 60)
    print()

    # 1. 读取数据
    print("【1. 读取数据】")
    print(f"输入文件:")
    print(f"  - {INPUT_STATS}")
    print(f"  - {INPUT_COV}")
    print()

    # 读取统计数据
    df_stats_full = pd.read_csv(INPUT_STATS)
    print(f"  原始股票数: {len(df_stats_full):,} 只")

    # 第一步筛选：正期望收益（如果启用）
    selected_indices = None
    if USE_POSITIVE_RETURN_ONLY:
        print(f"\n  【筛选1：正期望收益】")
        print(f"  筛选标准: 只使用正期望收益股票（E(R) > 0）")

        negative_count = (df_stats_full['Mean_Return_Annual'] <= 0).sum()
        positive_count = (df_stats_full['Mean_Return_Annual'] > 0).sum()

        print(f"  负收益股票: {negative_count} 只 ({negative_count/len(df_stats_full)*100:.1f}%)")
        print(f"  正收益股票: {positive_count} 只 ({positive_count/len(df_stats_full)*100:.1f}%)")

        # 保留原始索引位置
        df_stats_full = df_stats_full[df_stats_full['Mean_Return_Annual'] > 0]
        selected_indices = df_stats_full.index.tolist()  # 记录筛选后的原始位置索引
        df_stats_full = df_stats_full.reset_index(drop=True)
        print(f"  筛选后股票数: {len(df_stats_full)} 只")

    # 第二步筛选：按夏普比率（如果需要）
    if TOP_N_STOCKS is not None and len(df_stats_full) > TOP_N_STOCKS:
        print(f"\n  【筛选2：夏普比率】")
        print(f"  筛选标准: 夏普比率前{TOP_N_STOCKS}只")
        # 保留原始索引位置
        df_stats = df_stats_full.nlargest(TOP_N_STOCKS, 'Sharpe_Ratio')

        # 如果第一步已经筛选过，需要将第二步的索引映射回原始索引
        if selected_indices is not None:
            second_step_indices = df_stats.index.tolist()
            selected_indices = [selected_indices[i] for i in second_step_indices]
        else:
            selected_indices = df_stats.index.tolist()

        df_stats = df_stats.reset_index(drop=True)
        print(f"  筛选后股票数: {len(df_stats)} 只")
        print()
    else:
        print(f"\n  最终使用股票数: {len(df_stats_full):,} 只")
        print()
        df_stats = df_stats_full.copy()

    symbols = df_stats['Symbol'].values
    mean_returns = df_stats['Mean_Return_Annual'].values

    # 读取协方差矩阵并筛选
    df_cov = pd.read_csv(INPUT_COV, index_col=0)

    # 将columns转换为整数（与index类型一致）
    df_cov.columns = df_cov.columns.astype(int)

    # 使用股票代码筛选协方差矩阵（行和列）
    df_cov = df_cov.loc[symbols, symbols]

    cov_matrix = df_cov.values

    print(f"  最终股票数: {len(symbols):,} 只")
    print(f"  期望收益率范围: {mean_returns.min():.4f} 至 {mean_returns.max():.4f}")
    print(f"  协方差矩阵维度: {cov_matrix.shape}")
    print(f"  是否允许卖空: {'是' if ALLOW_SHORT_SELLING else '否'}")
    print()

    # 2. 求解全局最小方差组合(GMV)
    print("【2. 求解全局最小方差组合(GMV)】")
    print("优化中...")

    gmv_result = find_global_minimum_variance(mean_returns, cov_matrix, ALLOW_SHORT_SELLING)

    if gmv_result.success:
        gmv_weights = gmv_result.x
        gmv_return = portfolio_return(gmv_weights, mean_returns)
        gmv_variance = portfolio_variance(gmv_weights, cov_matrix)
        gmv_std = np.sqrt(gmv_variance)

        print(f"✓ 优化成功")
        print(f"  期望收益率: {gmv_return:.6f} ({gmv_return * 100:.2f}%)")
        print(f"  标准差(风险): {gmv_std:.6f} ({gmv_std * 100:.2f}%)")
        print(f"  方差: {gmv_variance:.6f}")
        print(f"  夏普比率(Rf=0): {gmv_return / gmv_std:.4f}")

        # 找出权重最大的前10只股票
        top_10_idx = np.argsort(gmv_weights)[-10:][::-1]
        print(f"\n  权重最大的前10只股票:")
        for idx in top_10_idx:
            if gmv_weights[idx] > 0.001:  # 只显示权重>0.1%的
                print(f"    {symbols[idx]}: {gmv_weights[idx] * 100:.2f}%")

        # 保存GMV组合
        gmv_data = {
            'type': 'Global Minimum Variance Portfolio',
            'return': float(gmv_return),
            'std': float(gmv_std),
            'variance': float(gmv_variance),
            'sharpe_ratio': float(gmv_return / gmv_std),
            'weights': {str(symbols[i]): float(gmv_weights[i]) for i in range(len(symbols)) if gmv_weights[i] > 1e-6}
        }

        with open(OUTPUT_GMV, 'w', encoding='utf-8') as f:
            json.dump(gmv_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 已保存: {OUTPUT_GMV}")
    else:
        print(f"✗ 优化失败: {gmv_result.message}")
        gmv_return = None
        gmv_std = None

    print()

    # 3. 构建有效前沿
    print(f"【3. 构建有效前沿】")
    print(f"生成 {NUM_PORTFOLIOS} 个组合...")
    print()

    # 确定收益率扫描范围
    if gmv_return is not None:
        min_return = gmv_return
    else:
        min_return = mean_returns.min()

    max_return = mean_returns.max() * 0.95

    target_returns = np.linspace(min_return, max_return, NUM_PORTFOLIOS)

    # 存储结果
    frontier_results = []
    successful_count = 0
    failed_count = 0

    print(f"目标收益率范围: {min_return:.4f} 至 {max_return:.4f}")
    print(f"优化进度:")

    for i, target_return in enumerate(target_returns):
        # 显示进度
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{NUM_PORTFOLIOS} ({(i + 1) / NUM_PORTFOLIOS * 100:.0f}%)")

        result = minimize_variance_for_target_return(
            target_return, mean_returns, cov_matrix, ALLOW_SHORT_SELLING
        )

        if result.success:
            weights = result.x
            ret = portfolio_return(weights, mean_returns)
            var = portfolio_variance(weights, cov_matrix)
            std = np.sqrt(var)

            frontier_results.append({
                'Return': ret,
                'Std': std,
                'Variance': var,
                'Sharpe_Ratio': ret / std if std > 0 else 0
            })
            successful_count += 1
        else:
            failed_count += 1

    print(f"\n优化结果:")
    print(f"  成功: {successful_count} 个")
    print(f"  失败: {failed_count} 个")
    print()

    # 4. 保存有效前沿
    if len(frontier_results) > 0:
        print("【4. 保存有效前沿】")

        df_frontier = pd.DataFrame(frontier_results)

        # 排序（按标准差）
        df_frontier = df_frontier.sort_values('Std').reset_index(drop=True)

        # 保存
        df_frontier.to_csv(OUTPUT_FRONTIER, index=False, encoding='utf-8-sig')

        print(f"✓ 已保存: {OUTPUT_FRONTIER}")
        print(f"  数据点: {len(df_frontier)} 个")
        print()

        # 统计
        print(f"有效前沿统计:")
        print(f"  收益率范围: {df_frontier['Return'].min():.6f} 至 {df_frontier['Return'].max():.6f}")
        print(f"  风险(标准差)范围: {df_frontier['Std'].min():.6f} 至 {df_frontier['Std'].max():.6f}")
        print(f"  最大夏普比率: {df_frontier['Sharpe_Ratio'].max():.4f}")
        print()

        # 5. 生成报告
        print("【5. 生成分析报告】")

        with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("有效前沿分析报告\n")
            f.write("=" * 70 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 一、优化设置
            f.write("一、优化设置\n")
            f.write("-" * 70 + "\n")
            f.write(f"股票数: {len(symbols):,} 只\n")
            f.write(f"优化方法: Markowitz均值-方差模型\n")
            f.write(f"优化算法: SLSQP (Sequential Least Squares Programming)\n")
            f.write(f"是否允许卖空: {'是' if ALLOW_SHORT_SELLING else '否'}\n")
            f.write(f"有效前沿点数: {NUM_PORTFOLIOS} 个\n")
            f.write(f"成功优化: {successful_count} 个 ({successful_count / NUM_PORTFOLIOS * 100:.1f}%)\n\n")

            # 二、全局最小方差组合
            if gmv_result.success:
                f.write("二、全局最小方差组合(GMV)\n")
                f.write("-" * 70 + "\n")
                f.write(f"期望收益率: {gmv_return:.6f} ({gmv_return * 100:.2f}%)\n")
                f.write(f"标准差: {gmv_std:.6f} ({gmv_std * 100:.2f}%)\n")
                f.write(f"方差: {gmv_variance:.6f}\n")
                f.write(f"夏普比率(Rf=0): {gmv_return / gmv_std:.4f}\n\n")

                f.write(f"权重>1%的股票:\n")
                for i in range(len(symbols)):
                    if gmv_weights[i] > 0.01:
                        f.write(f"  {symbols[i]}: {gmv_weights[i] * 100:.2f}%\n")
                f.write("\n")

            # 三、有效前沿统计
            f.write("三、有效前沿统计\n")
            f.write("-" * 70 + "\n")
            f.write(f"数据点数: {len(df_frontier)} 个\n\n")

            f.write(f"收益率统计:\n")
            f.write(f"  最小值: {df_frontier['Return'].min():.6f} ({df_frontier['Return'].min() * 100:.2f}%)\n")
            f.write(f"  最大值: {df_frontier['Return'].max():.6f} ({df_frontier['Return'].max():.2f}%)\n")
            f.write(f"  均值: {df_frontier['Return'].mean():.6f} ({df_frontier['Return'].mean() * 100:.2f}%)\n\n")

            f.write(f"风险(标准差)统计:\n")
            f.write(f"  最小值: {df_frontier['Std'].min():.6f} ({df_frontier['Std'].min() * 100:.2f}%)\n")
            f.write(f"  最大值: {df_frontier['Std'].max():.6f} ({df_frontier['Std'].max() * 100:.2f}%)\n")
            f.write(f"  均值: {df_frontier['Std'].mean():.6f} ({df_frontier['Std'].mean() * 100:.2f}%)\n\n")

            f.write(f"夏普比率统计:\n")
            f.write(f"  最小值: {df_frontier['Sharpe_Ratio'].min():.4f}\n")
            f.write(f"  最大值: {df_frontier['Sharpe_Ratio'].max():.4f}\n")
            f.write(f"  均值: {df_frontier['Sharpe_Ratio'].mean():.4f}\n\n")

            # 四、输出文件
            f.write("四、输出文件\n")
            f.write("-" * 70 + "\n")
            f.write(f"有效前沿数据: {OUTPUT_FRONTIER.name}\n")
            f.write(f"  大小: {OUTPUT_FRONTIER.stat().st_size / 1024:.2f} KB\n")
            f.write(f"  列: Return, Std, Variance, Sharpe_Ratio\n\n")

            f.write(f"GMV组合: {OUTPUT_GMV.name}\n")
            f.write(f"  大小: {OUTPUT_GMV.stat().st_size / 1024:.2f} KB\n")
            f.write(f"  格式: JSON\n\n")

            # 五、说明
            f.write("五、说明\n")
            f.write("-" * 70 + "\n")
            f.write("1. 有效前沿: 给定风险水平下收益最大的组合集合\n")
            f.write("2. GMV组合: 风险最小的组合，不考虑收益约束\n")
            f.write("3. 夏普比率: 简化版本假设无风险利率Rf=0\n")
            f.write("4. 优化约束: 权重和为1，" + ("允许卖空" if ALLOW_SHORT_SELLING else "不允许卖空") + "\n")
            f.write("5. 数据用途: 用于绘制有效前沿曲线和确定资本市场线\n\n")

        print(f"  报告文件: {OUTPUT_REPORT}")
        print()

    # 6. 完成
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - {OUTPUT_FRONTIER}")
    print(f"  - {OUTPUT_GMV}")
    print(f"  - {OUTPUT_REPORT}")
    print()


if __name__ == "__main__":
    main()
