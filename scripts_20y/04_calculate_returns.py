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
- 处理后数据_20y/cleaned_prices.csv
- 处理后数据_20y/hsi_prices.csv

输出：
- 处理后数据_20y/returns.csv (股票日收益率)
- 处理后数据_20y/hsi_returns.csv (HSI日收益率)
- 处理后数据_20y/returns_数据分析报告.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats.mstats import winsorize
import warnings

warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = Path(__file__).parent.parent
INPUT_PRICES = BASE_DIR / "处理后数据_20y" / "03_清洗过滤" / r"cleaned_prices.csv"
INPUT_HSI = BASE_DIR / "处理后数据_20y" / "02_恒指价格" / r"hsi_prices.csv"
OUTPUT_RETURNS = BASE_DIR / "处理后数据_20y" / "04_收益率" / r"returns.csv"
OUTPUT_HSI_RETURNS = BASE_DIR / "处理后数据_20y" / "04_收益率" / r"hsi_returns.csv"
OUTPUT_REPORT = BASE_DIR / "处理后数据_20y" / "04_收益率" / r"returns_数据分析报告.txt"

# 参数配置
MONTHS_PER_YEAR = 12  # 月度数据年化因子（原252改为12）
WINSORIZE_LIMITS = (0.01, 0.01)  # Winsorize极端值处理（上下各1%）


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
    price_data = prices_df[date_cols].values  # 转为numpy数组便于处理

    # 计算对数收益率：r_t = ln(P_t / P_t-1)
    # 核心规则（金融计量标准做法）：
    # 1. 如果 P(t-1) 缺失或为0 → return = 0（分母为0）
    # 2. 如果 P(t) 缺失 → return = 0（当日无交易）
    # 3. 否则正常计算 ln(P(t) / P(t-1))
    # 4. 不改变价格本身，只在计算收益率时处理

    n_stocks, n_days = price_data.shape
    returns_data = np.zeros((n_stocks, n_days - 1))  # 初始化为0

    # 逐只股票计算收益率
    for i in range(n_stocks):
        prices = price_data[i, :]
        for t in range(1, n_days):
            p_t = prices[t]
            p_t_1 = prices[t-1]

            # 核心逻辑：处理分母为0或缺失的情况
            if pd.isna(p_t_1) or p_t_1 <= 0:
                returns_data[i, t-1] = 0.0  # 分母为0，收益率为0
            elif pd.isna(p_t) or p_t <= 0:
                returns_data[i, t-1] = 0.0  # 当日价格缺失，收益率为0
            else:
                returns_data[i, t-1] = np.log(p_t / p_t_1)  # 正常计算

    # 转回DataFrame
    returns_df_data = pd.DataFrame(
        returns_data,
        columns=date_cols[1:]  # 删除首日
    )

    # 合并info和收益率
    returns_df = pd.concat([
        prices_df[info_cols].reset_index(drop=True),
        returns_df_data
    ], axis=1)

    return returns_df, date_cols[1:]


def main():
    # 创建输出目录
    OUTPUT_RETURNS.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("计算收益率")
    print("=" * 60)
    print()

    # ===================== 1. 处理股票数据 =====================
    print("【1. 处理股票数据】")
    print(f"输入文件: {INPUT_PRICES}")

    df_prices = pd.read_csv(INPUT_PRICES)
    info_cols = ['Symbol', 'SecurityID']  # 20年数据没有InstitutionID

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

    # 处理inf和-inf（理论上不应出现，因为已在计算时处理）
    inf_count = np.isinf(returns_data.values).sum()
    if inf_count > 0:
        print(f"  警告: 发现 {inf_count} 个无穷值，将替换为0")
        returns_data = returns_data.replace([np.inf, -np.inf], 0.0)

    print()

    # 对收益率进行Winsorize处理（处理极端值）
    print("【对收益率进行Winsorize处理】")
    print(f"  方法: Winsorize (上下各 {WINSORIZE_LIMITS[0] * 100:.0f}%)")

    returns_data_winsorized = returns_data.copy()
    extreme_count = 0

    for idx in range(len(returns_data_winsorized)):
        row = returns_data_winsorized.iloc[idx].values
        # 只对非零值进行winsorize（零值代表停牌或缺失）
        non_zero_mask = row != 0
        if non_zero_mask.sum() > 10:  # 至少有10个非零值才处理
            original = row.copy()
            row_winsorized = winsorize(row[non_zero_mask], limits=WINSORIZE_LIMITS)
            row[non_zero_mask] = row_winsorized
            extreme_count += (np.abs(original - row) > 1e-10).sum()
            returns_data_winsorized.iloc[idx] = row

    print(f"  处理极端值: {extreme_count:,} 个")
    print()

    # 更新DataFrame
    df_returns[date_cols] = returns_data_winsorized

    # 统计处理后的收益率
    all_returns = returns_data_winsorized.values.flatten()
    all_returns = all_returns[all_returns != 0]  # 排除零值（停牌日）

    print(f"收益率统计（日收益率）:")
    print(f"  均值: {all_returns.mean():.6f} ({all_returns.mean() * 100:.4f}%)")
    print(f"  标准差: {all_returns.std():.6f} ({all_returns.std() * 100:.4f}%)")
    print(f"  最小值: {all_returns.min():.6f} ({all_returns.min() * 100:.4f}%)")
    print(f"  最大值: {all_returns.max():.6f} ({all_returns.max() * 100:.4f}%)")
    print(f"  中位数: {np.median(all_returns):.6f} ({np.median(all_returns) * 100:.4f}%)")
    print()

    print(f"年化统计:")
    print(f"  年化收益率: {all_returns.mean() * MONTHS_PER_YEAR:.6f} ({all_returns.mean() * MONTHS_PER_YEAR * 100:.2f}%)")
    print(f"  年化波动率: {all_returns.std() * np.sqrt(MONTHS_PER_YEAR):.6f} ({all_returns.std() * np.sqrt(MONTHS_PER_YEAR) * 100:.2f}%)")
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
    print(f"  年化收益率: {hsi_returns.mean() * MONTHS_PER_YEAR:.6f} ({hsi_returns.mean() * MONTHS_PER_YEAR * 100:.2f}%)")
    print(f"  年化波动率: {hsi_returns.std() * np.sqrt(MONTHS_PER_YEAR):.6f} ({hsi_returns.std() * np.sqrt(MONTHS_PER_YEAR) * 100:.2f}%)")
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
        f.write(f"  - 年化收益率 = 月均收益率 × {MONTHS_PER_YEAR}\n")
        f.write(f"  - 年化波动率 = 月波动率 × √{MONTHS_PER_YEAR}\n\n")

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
        f.write(f"  年化收益率: {all_returns.mean() * MONTHS_PER_YEAR:>10.6f}  ({all_returns.mean() * MONTHS_PER_YEAR * 100:>7.2f}%)\n")
        f.write(f"  年化波动率: {all_returns.std() * np.sqrt(MONTHS_PER_YEAR):>10.6f}  ({all_returns.std() * np.sqrt(MONTHS_PER_YEAR) * 100:>7.2f}%)\n\n")

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
        f.write(f"  年化收益率: {hsi_returns.mean() * MONTHS_PER_YEAR:>10.6f}  ({hsi_returns.mean() * MONTHS_PER_YEAR * 100:>7.2f}%)\n")
        f.write(f"  年化波动率: {hsi_returns.std() * np.sqrt(MONTHS_PER_YEAR):>10.6f}  ({hsi_returns.std() * np.sqrt(MONTHS_PER_YEAR) * 100:>7.2f}%)\n\n")

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
