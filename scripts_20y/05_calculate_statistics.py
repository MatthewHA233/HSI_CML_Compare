#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算统计特征脚本

功能：
1. 计算期望收益率向量 μ（年化）
2. 计算协方差矩阵 Σ（年化）
3. 计算相关系数矩阵
4. 验证协方差矩阵正定性

输入：
- 处理后数据_20y/returns.csv

输出：
- 处理后数据_20y/summary_stats.csv (描述性统计)
- 处理后数据_20y/covariance.csv (协方差矩阵)
- 处理后数据_20y/correlation.csv (相关系数矩阵)
- 处理后数据_20y/statistics_数据分析报告.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = Path(__file__).parent.parent
INPUT_RETURNS = BASE_DIR / "处理后数据_20y" / "04_收益率" / r"returns.csv"
OUTPUT_STATS = BASE_DIR / "处理后数据_20y" / "05_统计特征" / r"summary_stats.csv"
OUTPUT_COV = BASE_DIR / "处理后数据_20y" / "05_统计特征" / r"covariance.csv"
OUTPUT_CORR = BASE_DIR / "处理后数据_20y" / "05_统计特征" / r"correlation.csv"
OUTPUT_REPORT = BASE_DIR / "处理后数据_20y" / "05_统计特征" / r"statistics_数据分析报告.txt"

# 参数配置
TRADING_DAYS_PER_YEAR = 252


def check_positive_definite(matrix, name="Matrix"):
    """检查矩阵是否正定"""
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        min_eigenvalue = eigenvalues.min()
        is_positive_definite = np.all(eigenvalues > 0)

        print(f"{name}正定性检查:")
        print(f"  最小特征值: {min_eigenvalue:.6e}")
        print(f"  最大特征值: {eigenvalues.max():.6e}")
        print(f"  是否正定: {'是' if is_positive_definite else '否'}")

        if not is_positive_definite:
            print(f"  警告: {name}不是正定的！可能影响优化求解。")
            if min_eigenvalue > -1e-10:
                print(f"  建议: 特征值接近零，可能是数值精度问题，可尝试添加正则化项。")

        return is_positive_definite, min_eigenvalue, eigenvalues.max()

    except np.linalg.LinAlgError as e:
        print(f"  错误: 无法计算特征值 - {e}")
        return False, None, None


def main():
    print("=" * 60)
    print("计算统计特征")
    print("=" * 60)
    print()

    # 1. 读取收益率数据
    print("【1. 读取收益率数据】")
    print(f"输入文件: {INPUT_RETURNS}")

    df_returns = pd.read_csv(INPUT_RETURNS)
    info_cols = ['Symbol', 'InstitutionID', 'SecurityID']
    date_cols = [col for col in df_returns.columns if col not in info_cols]

    print(f"  股票数: {len(df_returns):,} 只")
    print(f"  交易日数: {len(date_cols):,} 天")
    print(f"  日期范围: {date_cols[0]} 至 {date_cols[-1]}")
    print()

    # 提取收益率矩阵（T×N：交易日×股票）
    symbols = df_returns['Symbol'].values
    returns_data = df_returns[date_cols].T  # 转置为 日期×股票
    returns_data.columns = symbols  # 设置列名为股票代码

    print(f"收益率矩阵维度: {returns_data.shape[0]} 天 × {returns_data.shape[1]} 只股票")
    print()

    # 2. 计算期望收益率向量（年化）
    print("【2. 计算期望收益率向量】")

    # 日均收益率
    mean_returns_daily = returns_data.mean(axis=0)

    # 年化期望收益率
    mean_returns_annual = mean_returns_daily * TRADING_DAYS_PER_YEAR

    print(f"期望收益率统计（年化）:")
    print(f"  最小值: {mean_returns_annual.min():.6f} ({mean_returns_annual.min() * 100:.2f}%)")
    print(f"  最大值: {mean_returns_annual.max():.6f} ({mean_returns_annual.max() * 100:.2f}%)")
    print(f"  均值: {mean_returns_annual.mean():.6f} ({mean_returns_annual.mean() * 100:.2f}%)")
    print(f"  中位数: {mean_returns_annual.median():.6f} ({mean_returns_annual.median() * 100:.2f}%)")
    print(f"  标准差: {mean_returns_annual.std():.6f}")
    print()

    # 3. 计算协方差矩阵（年化）
    print("【3. 计算协方差矩阵】")

    # 日协方差矩阵
    cov_matrix_daily = returns_data.cov()

    # 年化协方差矩阵
    cov_matrix_annual = cov_matrix_daily * TRADING_DAYS_PER_YEAR

    print(f"协方差矩阵维度: {cov_matrix_annual.shape[0]} × {cov_matrix_annual.shape[1]}")
    print(f"协方差统计（年化）:")
    print(f"  对角线均值（方差）: {np.diag(cov_matrix_annual).mean():.6f}")
    print(f"  对角线最小值: {np.diag(cov_matrix_annual).min():.6f}")
    print(f"  对角线最大值: {np.diag(cov_matrix_annual).max():.6f}")
    print()

    # 检查正定性
    is_pd, min_eig, max_eig = check_positive_definite(cov_matrix_annual, "协方差矩阵")
    print()

    # 如果不是正定，添加正则化
    if not is_pd and min_eig is not None and min_eig > -1e-6:
        print("添加正则化项使矩阵正定...")
        regularization = abs(min_eig) + 1e-8
        cov_matrix_annual += np.eye(len(cov_matrix_annual)) * regularization
        print(f"  正则化系数: {regularization:.6e}")

        # 重新检查
        is_pd_new, _, _ = check_positive_definite(cov_matrix_annual, "正则化后协方差矩阵")
        print()

    # 4. 计算相关系数矩阵
    print("【4. 计算相关系数矩阵】")

    corr_matrix = returns_data.corr()

    print(f"相关系数矩阵维度: {corr_matrix.shape[0]} × {corr_matrix.shape[1]}")

    # 提取上三角（不含对角线）
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlations = upper_triangle.stack().values

    print(f"相关系数统计:")
    print(f"  最小值: {correlations.min():.6f}")
    print(f"  最大值: {correlations.max():.6f}")
    print(f"  均值: {correlations.mean():.6f}")
    print(f"  中位数: {np.median(correlations):.6f}")
    print()

    # 高度相关的股票对
    high_corr_threshold = 0.9
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if corr_matrix.iloc[i, j] > high_corr_threshold:
                high_corr_pairs.append((symbols[i], symbols[j], corr_matrix.iloc[i, j]))

    print(f"高度相关的股票对 (>0.9): {len(high_corr_pairs)} 对")
    if len(high_corr_pairs) > 0:
        print(f"  前5对:")
        for pair in sorted(high_corr_pairs, key=lambda x: -x[2])[:5]:
            print(f"    {pair[0]} - {pair[1]}: {pair[2]:.4f}")
    print()

    # 5. 计算波动率（标准差）
    print("【5. 计算波动率】")

    # 日波动率
    std_daily = returns_data.std(axis=0)

    # 年化波动率
    std_annual = std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    print(f"波动率统计（年化）:")
    print(f"  最小值: {std_annual.min():.6f} ({std_annual.min() * 100:.2f}%)")
    print(f"  最大值: {std_annual.max():.6f} ({std_annual.max() * 100:.2f}%)")
    print(f"  均值: {std_annual.mean():.6f} ({std_annual.mean() * 100:.2f}%)")
    print(f"  中位数: {std_annual.median():.6f} ({std_annual.median() * 100:.2f}%)")
    print()

    # 6. 生成汇总统计表
    print("【6. 生成汇总统计表】")

    summary_stats = pd.DataFrame({
        'Symbol': symbols,
        'Mean_Return_Annual': mean_returns_annual.values,
        'Std_Annual': std_annual.values,
        'Sharpe_Ratio': mean_returns_annual.values / std_annual.values,  # 简化夏普比率（假设Rf=0）
        'Min_Return_Daily': returns_data.min(axis=0).values,
        'Max_Return_Daily': returns_data.max(axis=0).values
    })

    # 排序（按年化收益率降序）
    summary_stats = summary_stats.sort_values('Mean_Return_Annual', ascending=False)

    # 保存
    summary_stats.to_csv(OUTPUT_STATS, index=False, encoding='utf-8-sig')
    print(f"✓ 已保存: {OUTPUT_STATS}")
    print()

    # 7. 保存协方差矩阵
    print("【7. 保存协方差矩阵】")

    # cov_matrix_annual 已经是DataFrame，直接保存
    # 确保index和columns都是股票代码
    cov_matrix_annual.index = symbols
    cov_matrix_annual.columns = symbols
    cov_matrix_annual.to_csv(OUTPUT_COV, encoding='utf-8-sig')

    print(f"✓ 已保存: {OUTPUT_COV}")
    print(f"  文件大小: {OUTPUT_COV.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    # 8. 保存相关系数矩阵
    print("【8. 保存相关系数矩阵】")

    corr_df = pd.DataFrame(corr_matrix, index=symbols, columns=symbols)
    corr_df.to_csv(OUTPUT_CORR, encoding='utf-8-sig')

    print(f"✓ 已保存: {OUTPUT_CORR}")
    print(f"  文件大小: {OUTPUT_CORR.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    # 9. 生成分析报告
    print("【9. 生成分析报告】")

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("统计特征分析报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 一、输入数据
        f.write("一、输入数据\n")
        f.write("-" * 70 + "\n")
        f.write(f"输入文件: {INPUT_RETURNS.name}\n")
        f.write(f"股票数: {len(df_returns):,} 只\n")
        f.write(f"交易日数: {len(date_cols):,} 天\n")
        f.write(f"日期范围: {date_cols[0]} 至 {date_cols[-1]}\n\n")

        # 二、期望收益率
        f.write("二、期望收益率向量（年化）\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'统计量':<15} {'数值':>15} {'百分比':>15}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'最小值':<15} {mean_returns_annual.min():>15.6f} {mean_returns_annual.min() * 100:>14.2f}%\n")
        f.write(f"{'25%分位数':<15} {mean_returns_annual.quantile(0.25):>15.6f} {mean_returns_annual.quantile(0.25) * 100:>14.2f}%\n")
        f.write(f"{'中位数':<15} {mean_returns_annual.median():>15.6f} {mean_returns_annual.median() * 100:>14.2f}%\n")
        f.write(f"{'均值':<15} {mean_returns_annual.mean():>15.6f} {mean_returns_annual.mean() * 100:>14.2f}%\n")
        f.write(f"{'75%分位数':<15} {mean_returns_annual.quantile(0.75):>15.6f} {mean_returns_annual.quantile(0.75) * 100:>14.2f}%\n")
        f.write(f"{'最大值':<15} {mean_returns_annual.max():>15.6f} {mean_returns_annual.max() * 100:>14.2f}%\n")
        f.write(f"{'标准差':<15} {mean_returns_annual.std():>15.6f}\n\n")

        # 三、协方差矩阵
        f.write("三、协方差矩阵（年化）\n")
        f.write("-" * 70 + "\n")
        f.write(f"维度: {cov_matrix_annual.shape[0]} × {cov_matrix_annual.shape[1]}\n")
        f.write(f"对角线统计（方差）:\n")
        f.write(f"  最小值: {np.diag(cov_matrix_annual).min():.6f}\n")
        f.write(f"  最大值: {np.diag(cov_matrix_annual).max():.6f}\n")
        f.write(f"  均值: {np.diag(cov_matrix_annual).mean():.6f}\n")
        f.write(f"  中位数: {np.median(np.diag(cov_matrix_annual)):.6f}\n\n")

        f.write(f"正定性检查:\n")
        if is_pd:
            f.write(f"  状态: 正定 ✓\n")
        else:
            f.write(f"  状态: 非正定 (已添加正则化)\n")
        f.write(f"  最小特征值: {min_eig:.6e}\n")
        f.write(f"  最大特征值: {max_eig:.6e}\n")
        f.write(f"  条件数: {max_eig / abs(min_eig) if min_eig != 0 else float('inf'):.2e}\n\n")

        # 四、相关系数矩阵
        f.write("四、相关系数矩阵\n")
        f.write("-" * 70 + "\n")
        f.write(f"维度: {corr_matrix.shape[0]} × {corr_matrix.shape[1]}\n")
        f.write(f"相关系数统计:\n")
        f.write(f"  最小值: {correlations.min():.6f}\n")
        f.write(f"  25%分位数: {np.percentile(correlations, 25):.6f}\n")
        f.write(f"  中位数: {np.median(correlations):.6f}\n")
        f.write(f"  均值: {correlations.mean():.6f}\n")
        f.write(f"  75%分位数: {np.percentile(correlations, 75):.6f}\n")
        f.write(f"  最大值: {correlations.max():.6f}\n\n")

        f.write(f"高度相关股票对 (>0.9): {len(high_corr_pairs)} 对\n")
        if len(high_corr_pairs) > 0:
            f.write(f"前10对:\n")
            for idx, pair in enumerate(sorted(high_corr_pairs, key=lambda x: -x[2])[:10], 1):
                f.write(f"  {idx:2}. {pair[0]} - {pair[1]}: {pair[2]:.6f}\n")
        f.write("\n")

        # 五、波动率
        f.write("五、波动率（年化标准差）\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'统计量':<15} {'数值':>15} {'百分比':>15}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'最小值':<15} {std_annual.min():>15.6f} {std_annual.min() * 100:>14.2f}%\n")
        f.write(f"{'25%分位数':<15} {std_annual.quantile(0.25):>15.6f} {std_annual.quantile(0.25) * 100:>14.2f}%\n")
        f.write(f"{'中位数':<15} {std_annual.median():>15.6f} {std_annual.median() * 100:>14.2f}%\n")
        f.write(f"{'均值':<15} {std_annual.mean():>15.6f} {std_annual.mean() * 100:>14.2f}%\n")
        f.write(f"{'75%分位数':<15} {std_annual.quantile(0.75):>15.6f} {std_annual.quantile(0.75) * 100:>14.2f}%\n")
        f.write(f"{'最大值':<15} {std_annual.max():>15.6f} {std_annual.max() * 100:>14.2f}%\n\n")

        # 六、输出文件
        f.write("六、输出文件\n")
        f.write("-" * 70 + "\n")
        f.write(f"汇总统计: {OUTPUT_STATS.name}\n")
        f.write(f"  大小: {OUTPUT_STATS.stat().st_size / 1024:.2f} KB\n")
        f.write(f"  行数: {len(summary_stats):,} 只股票\n\n")

        f.write(f"协方差矩阵: {OUTPUT_COV.name}\n")
        f.write(f"  大小: {OUTPUT_COV.stat().st_size / 1024 / 1024:.2f} MB\n")
        f.write(f"  维度: {cov_matrix_annual.shape[0]} × {cov_matrix_annual.shape[1]}\n\n")

        f.write(f"相关系数矩阵: {OUTPUT_CORR.name}\n")
        f.write(f"  大小: {OUTPUT_CORR.stat().st_size / 1024 / 1024:.2f} MB\n")
        f.write(f"  维度: {corr_matrix.shape[0]} × {corr_matrix.shape[1]}\n\n")

        # 七、说明
        f.write("七、说明\n")
        f.write("-" * 70 + "\n")
        f.write("1. 年化处理: 期望收益×252，协方差×252，标准差×√252\n")
        f.write("2. 协方差矩阵: 必须正定才能进行Markowitz优化\n")
        f.write("3. 相关系数: 衡量股票间线性相关性，范围[-1, 1]\n")
        f.write("4. 夏普比率: 简化版本假设无风险利率Rf=0\n")
        f.write("5. 数据用途: 本数据将用于构建有效前沿和资本市场线\n\n")

    print(f"  报告文件: {OUTPUT_REPORT}")
    print()

    # 10. 完成
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - {OUTPUT_STATS}")
    print(f"  - {OUTPUT_COV}")
    print(f"  - {OUTPUT_CORR}")
    print(f"  - {OUTPUT_REPORT}")
    print()


if __name__ == "__main__":
    main()
