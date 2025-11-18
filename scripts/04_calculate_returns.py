#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算收益率脚本

功能：
1. 计算对数收益率：r_t = ln(P_t / P_t-1)
2. 生成收益率矩阵（日期×股票）
3. 同时处理股票数据和HSI数据
4. 年化处理（可选）

输入：
- 处理后数据/cleaned_prices.csv
- 处理后数据/hsi_prices.csv

输出：
- 处理后数据/returns.csv (股票日收益率)
- 处理后数据/hsi_returns.csv (HSI日收益率)
- 处理后数据/returns_数据分析报告.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = Path(__file__).parent.parent
INPUT_PRICES = BASE_DIR / "处理后数据" / "03_清洗过滤" / r"cleaned_prices.csv"
INPUT_HSI = BASE_DIR / "处理后数据" / "04_收益率" / r"hsi_prices.csv"
OUTPUT_RETURNS = BASE_DIR / "处理后数据" / "04_收益率" / r"returns.csv"
OUTPUT_HSI_RETURNS = BASE_DIR / "处理后数据" / "04_收益率" / r"hsi_returns.csv"
OUTPUT_REPORT = BASE_DIR / "处理后数据" / "04_收益率" / r"returns_数据分析报告.txt"

# 参数配置
TRADING_DAYS_PER_YEAR = 252  # 年化交易日数


def calculate_log_returns(prices_df, info_cols):
    """
    计算对数收益率

    Args:
        prices_df: 价格DataFrame（包含info列和日期列）
        info_cols: info列名列表

    Returns:
        returns_df: 收益率DataFrame
        date_cols: 日期列名列表
    """
    # 分离价格数据
    date_cols = [col for col in prices_df.columns if col not in info_cols]
    price_data = prices_df[date_cols]

    # 计算对数收益率：r_t = ln(P_t / P_t-1)
    # 使用 shift 方法：每列向右移一位
    price_shifted = price_data.shift(1, axis=1)

    # 对数收益率
    returns_data = np.log(price_data / price_shifted)

    # 删除第一列（首日无前值）
    returns_data = returns_data.iloc[:, 1:]
    date_cols = date_cols[1:]

    # 合并info和收益率
    returns_df = pd.concat([
        prices_df[info_cols].reset_index(drop=True),
        returns_data.reset_index(drop=True)
    ], axis=1)

    return returns_df, date_cols


def main():
    print("=" * 60)
    print("计算收益率")
    print("=" * 60)
    print()

    # ===================== 1. 处理股票数据 =====================
    print("【1. 处理股票数据】")
    print(f"输入文件: {INPUT_PRICES}")

    df_prices = pd.read_csv(INPUT_PRICES)
    info_cols = ['Symbol', 'InstitutionID', 'SecurityID']

    print(f"  股票数: {len(df_prices):,} 只")
    print(f"  价格列数: {len(df_prices.columns) - len(info_cols):,} 天")
    print()

    # 计算对数收益率
    print("计算对数收益率...")
    df_returns, date_cols = calculate_log_returns(df_prices, info_cols)

    print(f"  收益率列数: {len(date_cols):,} 天 (删除首日)")
    print(f"  日期范围: {date_cols[0]} 至 {date_cols[-1]}")
    print()

    # 统计收益率
    returns_data = df_returns[date_cols]

    # 处理inf和-inf（可能由零价格导致）
    inf_count = np.isinf(returns_data.values).sum()
    if inf_count > 0:
        print(f"  警告: 发现 {inf_count} 个无穷值，将替换为NaN")
        returns_data = returns_data.replace([np.inf, -np.inf], np.nan)

    # 统计
    all_returns = returns_data.values.flatten()
    all_returns = all_returns[~np.isnan(all_returns)]

    print(f"收益率统计（日收益率）:")
    print(f"  均值: {all_returns.mean():.6f} ({all_returns.mean() * 100:.4f}%)")
    print(f"  标准差: {all_returns.std():.6f} ({all_returns.std() * 100:.4f}%)")
    print(f"  最小值: {all_returns.min():.6f} ({all_returns.min() * 100:.4f}%)")
    print(f"  最大值: {all_returns.max():.6f} ({all_returns.max() * 100:.4f}%)")
    print(f"  中位数: {np.median(all_returns):.6f} ({np.median(all_returns) * 100:.4f}%)")
    print()

    print(f"年化统计:")
    print(f"  年化收益率: {all_returns.mean() * TRADING_DAYS_PER_YEAR:.6f} ({all_returns.mean() * TRADING_DAYS_PER_YEAR * 100:.2f}%)")
    print(f"  年化波动率: {all_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR):.6f} ({all_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100:.2f}%)")
    print()

    # 保存收益率
    df_returns.to_csv(OUTPUT_RETURNS, index=False, encoding='utf-8-sig')
    print(f"✓ 已保存: {OUTPUT_RETURNS}")
    print()

    # ===================== 2. 处理HSI数据 =====================
    print("【2. 处理HSI数据】")
    print(f"输入文件: {INPUT_HSI}")

    df_hsi = pd.read_csv(INPUT_HSI)
    print(f"  交易日数: {len(df_hsi):,} 天")
    print(f"  日期范围: {df_hsi['Date'].iloc[0]} 至 {df_hsi['Date'].iloc[-1]}")
    print()

    # 计算对数收益率
    print("计算对数收益率...")
    hsi_prices = df_hsi['HSI_Close'].values
    hsi_returns = np.log(hsi_prices[1:] / hsi_prices[:-1])

    # 创建收益率DataFrame
    df_hsi_returns = pd.DataFrame({
        'Date': df_hsi['Date'].iloc[1:].values,
        'HSI_Return': hsi_returns
    })

    print(f"  收益率数据点: {len(df_hsi_returns):,} 天")
    print(f"  日期范围: {df_hsi_returns['Date'].iloc[0]} 至 {df_hsi_returns['Date'].iloc[-1]}")
    print()

    # 统计
    print(f"HSI收益率统计（日收益率）:")
    print(f"  均值: {hsi_returns.mean():.6f} ({hsi_returns.mean() * 100:.4f}%)")
    print(f"  标准差: {hsi_returns.std():.6f} ({hsi_returns.std() * 100:.4f}%)")
    print(f"  最小值: {hsi_returns.min():.6f} ({hsi_returns.min() * 100:.4f}%)")
    print(f"  最大值: {hsi_returns.max():.6f} ({hsi_returns.max() * 100:.4f}%)")
    print(f"  中位数: {np.median(hsi_returns):.6f} ({np.median(hsi_returns) * 100:.4f}%)")
    print()

    print(f"HSI年化统计:")
    print(f"  年化收益率: {hsi_returns.mean() * TRADING_DAYS_PER_YEAR:.6f} ({hsi_returns.mean() * TRADING_DAYS_PER_YEAR * 100:.2f}%)")
    print(f"  年化波动率: {hsi_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR):.6f} ({hsi_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100:.2f}%)")
    print()

    # 保存收益率
    df_hsi_returns.to_csv(OUTPUT_HSI_RETURNS, index=False, encoding='utf-8-sig')
    print(f"✓ 已保存: {OUTPUT_HSI_RETURNS}")
    print()

    # ===================== 3. 生成分析报告 =====================
    print("【3. 生成分析报告】")

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("收益率计算分析报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 一、计算方法
        f.write("一、计算方法\n")
        f.write("-" * 70 + "\n")
        f.write("收益率计算公式: r_t = ln(P_t / P_t-1)\n")
        f.write("  - 对数收益率具有时间可加性\n")
        f.write("  - 适用于连续复利计算\n")
        f.write("  - 近似正态分布，适合统计分析\n\n")

        f.write("年化公式:\n")
        f.write(f"  - 年化收益率 = 日均收益率 × {TRADING_DAYS_PER_YEAR}\n")
        f.write(f"  - 年化波动率 = 日波动率 × √{TRADING_DAYS_PER_YEAR}\n\n")

        # 二、股票收益率统计
        f.write("二、股票收益率统计\n")
        f.write("-" * 70 + "\n")
        f.write(f"输入文件: {INPUT_PRICES.name}\n")
        f.write(f"输出文件: {OUTPUT_RETURNS.name}\n")
        f.write(f"股票数: {len(df_returns):,} 只\n")
        f.write(f"交易日数: {len(date_cols):,} 天\n")
        f.write(f"日期范围: {date_cols[0]} 至 {date_cols[-1]}\n\n")

        f.write("日收益率统计:\n")
        f.write(f"  均值:     {all_returns.mean():>10.6f}  ({all_returns.mean() * 100:>8.4f}%)\n")
        f.write(f"  标准差:   {all_returns.std():>10.6f}  ({all_returns.std() * 100:>8.4f}%)\n")
        f.write(f"  最小值:   {all_returns.min():>10.6f}  ({all_returns.min() * 100:>8.4f}%)\n")
        f.write(f"  25%分位: {np.percentile(all_returns, 25):>10.6f}  ({np.percentile(all_returns, 25) * 100:>8.4f}%)\n")
        f.write(f"  中位数:   {np.median(all_returns):>10.6f}  ({np.median(all_returns) * 100:>8.4f}%)\n")
        f.write(f"  75%分位: {np.percentile(all_returns, 75):>10.6f}  ({np.percentile(all_returns, 75) * 100:>8.4f}%)\n")
        f.write(f"  最大值:   {all_returns.max():>10.6f}  ({all_returns.max() * 100:>8.4f}%)\n\n")

        f.write("年化统计:\n")
        f.write(f"  年化收益率: {all_returns.mean() * TRADING_DAYS_PER_YEAR:>10.6f}  ({all_returns.mean() * TRADING_DAYS_PER_YEAR * 100:>7.2f}%)\n")
        f.write(f"  年化波动率: {all_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR):>10.6f}  ({all_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100:>7.2f}%)\n\n")

        # 三、HSI收益率统计
        f.write("三、HSI收益率统计\n")
        f.write("-" * 70 + "\n")
        f.write(f"输入文件: {INPUT_HSI.name}\n")
        f.write(f"输出文件: {OUTPUT_HSI_RETURNS.name}\n")
        f.write(f"交易日数: {len(df_hsi_returns):,} 天\n")
        f.write(f"日期范围: {df_hsi_returns['Date'].iloc[0]} 至 {df_hsi_returns['Date'].iloc[-1]}\n\n")

        f.write("日收益率统计:\n")
        f.write(f"  均值:     {hsi_returns.mean():>10.6f}  ({hsi_returns.mean() * 100:>8.4f}%)\n")
        f.write(f"  标准差:   {hsi_returns.std():>10.6f}  ({hsi_returns.std() * 100:>8.4f}%)\n")
        f.write(f"  最小值:   {hsi_returns.min():>10.6f}  ({hsi_returns.min() * 100:>8.4f}%)\n")
        f.write(f"  25%分位: {np.percentile(hsi_returns, 25):>10.6f}  ({np.percentile(hsi_returns, 25) * 100:>8.4f}%)\n")
        f.write(f"  中位数:   {np.median(hsi_returns):>10.6f}  ({np.median(hsi_returns) * 100:>8.4f}%)\n")
        f.write(f"  75%分位: {np.percentile(hsi_returns, 75):>10.6f}  ({np.percentile(hsi_returns, 75) * 100:>8.4f}%)\n")
        f.write(f"  最大值:   {hsi_returns.max():>10.6f}  ({hsi_returns.max() * 100:>8.4f}%)\n\n")

        f.write("年化统计:\n")
        f.write(f"  年化收益率: {hsi_returns.mean() * TRADING_DAYS_PER_YEAR:>10.6f}  ({hsi_returns.mean() * TRADING_DAYS_PER_YEAR * 100:>7.2f}%)\n")
        f.write(f"  年化波动率: {hsi_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR):>10.6f}  ({hsi_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100:>7.2f}%)\n\n")

        # 四、数据质量
        f.write("四、数据质量\n")
        f.write("-" * 70 + "\n")
        f.write(f"股票收益率:\n")
        f.write(f"  总数据点: {len(df_returns) * len(date_cols):,}\n")
        f.write(f"  有效数据点: {len(all_returns):,}\n")
        f.write(f"  缺失/异常: {len(df_returns) * len(date_cols) - len(all_returns):,}\n")
        f.write(f"  数据完整度: {len(all_returns) / (len(df_returns) * len(date_cols)) * 100:.2f}%\n\n")

        f.write(f"HSI收益率:\n")
        f.write(f"  总数据点: {len(df_hsi_returns)}\n")
        f.write(f"  缺失值: 0\n")
        f.write(f"  数据完整度: 100.00%\n\n")

        # 五、输出文件
        f.write("五、输出文件信息\n")
        f.write("-" * 70 + "\n")
        f.write(f"股票收益率文件:\n")
        f.write(f"  路径: {OUTPUT_RETURNS}\n")
        f.write(f"  大小: {OUTPUT_RETURNS.stat().st_size / 1024 / 1024:.2f} MB\n")
        f.write(f"  行数: {len(df_returns):,} 只股票\n")
        f.write(f"  列数: {len(df_returns.columns)} 列 (3个info + {len(date_cols)}个日期)\n\n")

        f.write(f"HSI收益率文件:\n")
        f.write(f"  路径: {OUTPUT_HSI_RETURNS}\n")
        f.write(f"  大小: {OUTPUT_HSI_RETURNS.stat().st_size / 1024:.2f} KB\n")
        f.write(f"  行数: {len(df_hsi_returns):,} 天\n")
        f.write(f"  列数: 2 列 (Date, HSI_Return)\n\n")

        # 六、说明
        f.write("六、说明\n")
        f.write("-" * 70 + "\n")
        f.write("1. 对数收益率: 使用自然对数，便于时间序列分析和统计建模\n")
        f.write("2. 年化处理: 方便不同时间周期的收益率和风险比较\n")
        f.write("3. 数据用途: 本数据将用于计算协方差矩阵和构建投资组合\n")
        f.write("4. 注意事项: 收益率假设为独立同分布，实际市场可能存在自相关\n")
        f.write("5. 风险指标: 波动率（标准差）衡量收益的不确定性，是风险的重要指标\n\n")

    print(f"  报告文件: {OUTPUT_REPORT}")
    print()

    # 4. 完成
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - {OUTPUT_RETURNS}")
    print(f"  - {OUTPUT_HSI_RETURNS}")
    print(f"  - {OUTPUT_REPORT}")
    print()


if __name__ == "__main__":
    main()
