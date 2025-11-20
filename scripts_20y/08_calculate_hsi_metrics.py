#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算HSI指标脚本

功能：
1. 计算HSI年化收益率
2. 计算HSI年化波动率（风险）
3. 计算HSI夏普比率
4. 计算HSI在风险-收益坐标系中的位置

输入：
- 处理后数据_20y/hsi_returns.csv
- 处理后数据_20y/hsi_prices.csv

输出：
- 处理后数据_20y/hsi_metrics.json
- 处理后数据_20y/hsi_metrics_数据分析报告.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = Path(__file__).parent.parent
INPUT_RETURNS = BASE_DIR / "处理后数据_20y" / "02_恒指价格" / r"hsi_returns.csv"
INPUT_PRICES = BASE_DIR / "处理后数据_20y" / "02_恒指价格" / r"hsi_prices.csv"
OUTPUT_METRICS = BASE_DIR / "处理后数据_20y" / "08_恒指指标" / r"hsi_metrics.json"
OUTPUT_REPORT = BASE_DIR / "处理后数据_20y" / "08_恒指指标" / r"hsi_metrics_数据分析报告.txt"

# 参数配置
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.025  # 假设无风险利率为2.5%（HKMA 3个月国债收益率）


def main():
    # 创建输出目录
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("计算HSI指标")
    print("=" * 60)
    print()

    # 1. 读取数据
    print("【1. 读取数据】")
    print(f"输入文件:")
    print(f"  - {INPUT_RETURNS}")
    print(f"  - {INPUT_PRICES}")
    print()

    # 读取收益率
    df_returns = pd.read_csv(INPUT_RETURNS)
    hsi_returns = df_returns['HSI_Return'].values

    # 读取价格
    df_prices = pd.read_csv(INPUT_PRICES)

    print(f"  收益率数据点: {len(hsi_returns):,} 天")
    print(f"  价格数据点: {len(df_prices):,} 天")
    print(f"  日期范围: {df_returns['Date'].iloc[0]} 至 {df_returns['Date'].iloc[-1]}")
    print()

    # 2. 计算年化收益率
    print("【2. 计算年化收益率】")

    # 日均收益率
    mean_daily_return = hsi_returns.mean()

    # 年化收益率
    annualized_return = mean_daily_return * TRADING_DAYS_PER_YEAR

    print(f"  日均收益率: {mean_daily_return:.6f} ({mean_daily_return * 100:.4f}%)")
    print(f"  年化收益率: {annualized_return:.6f} ({annualized_return * 100:.2f}%)")
    print()

    # 3. 计算年化波动率（风险）
    print("【3. 计算年化波动率】")

    # 日波动率
    daily_std = hsi_returns.std()

    # 年化波动率
    annualized_std = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    print(f"  日波动率: {daily_std:.6f} ({daily_std * 100:.4f}%)")
    print(f"  年化波动率: {annualized_std:.6f} ({annualized_std * 100:.2f}%)")
    print()

    # 4. 计算夏普比率
    print("【4. 计算夏普比率】")
    print(f"  无风险利率: {RISK_FREE_RATE:.4f} ({RISK_FREE_RATE * 100:.2f}%)")

    # 夏普比率 = (年化收益率 - 无风险利率) / 年化波动率
    sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_std

    print(f"  夏普比率: {sharpe_ratio:.4f}")
    print()

    # 5. 计算累计收益
    print("【5. 计算累计收益】")

    start_price = df_prices['ClosePrice'].iloc[0]
    end_price = df_prices['ClosePrice'].iloc[-1]
    total_return = (end_price - start_price) / start_price

    # 年化复合收益率（CAGR）
    n_years = len(df_prices) / TRADING_DAYS_PER_YEAR
    cagr = (end_price / start_price) ** (1 / n_years) - 1

    print(f"  期初价格: {start_price:,.2f}")
    print(f"  期末价格: {end_price:,.2f}")
    print(f"  累计收益率: {total_return:.4f} ({total_return * 100:.2f}%)")
    print(f"  年化复合收益率(CAGR): {cagr:.4f} ({cagr * 100:.2f}%)")
    print(f"  投资期限: {n_years:.2f} 年")
    print()

    # 6. 计算其他风险指标
    print("【6. 计算其他风险指标】")

    # 最大回撤
    prices = df_prices['ClosePrice'].values
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax
    max_drawdown = drawdown.min()

    # 下行波动率（只考虑负收益）
    negative_returns = hsi_returns[hsi_returns < 0]
    downside_std_daily = negative_returns.std() if len(negative_returns) > 0 else 0
    downside_std_annual = downside_std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sortino比率（使用下行波动率）
    sortino_ratio = (annualized_return - RISK_FREE_RATE) / downside_std_annual if downside_std_annual > 0 else 0

    # VaR (95%置信度)
    var_95 = np.percentile(hsi_returns, 5)
    var_95_annual = var_95 * np.sqrt(TRADING_DAYS_PER_YEAR)

    print(f"  最大回撤: {max_drawdown:.4f} ({max_drawdown * 100:.2f}%)")
    print(f"  下行波动率(年化): {downside_std_annual:.4f} ({downside_std_annual * 100:.2f}%)")
    print(f"  Sortino比率: {sortino_ratio:.4f}")
    print(f"  VaR(95%): {var_95:.6f} ({var_95 * 100:.4f}% daily)")
    print()

    # 7. 保存指标
    print("【7. 保存指标】")

    metrics = {
        'index': 'HSI',
        'name': '恒生指数',
        'date_range': {
            'start': df_returns['Date'].iloc[0],
            'end': df_returns['Date'].iloc[-1],
            'trading_days': len(hsi_returns),
            'years': float(n_years)
        },
        'returns': {
            'mean_daily': float(mean_daily_return),
            'annualized': float(annualized_return),
            'cagr': float(cagr),
            'total_return': float(total_return)
        },
        'risk': {
            'std_daily': float(daily_std),
            'std_annualized': float(annualized_std),
            'downside_std_annualized': float(downside_std_annual),
            'max_drawdown': float(max_drawdown),
            'var_95': float(var_95)
        },
        'risk_adjusted_metrics': {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'risk_free_rate': float(RISK_FREE_RATE)
        },
        'position': {
            'return': float(annualized_return),
            'risk': float(annualized_std),
            'description': '在风险-收益坐标系中的位置'
        }
    }

    with open(OUTPUT_METRICS, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✓ 已保存: {OUTPUT_METRICS}")
    print()

    # 8. 生成报告
    print("【8. 生成分析报告】")

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("HSI指标分析报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 一、基本信息
        f.write("一、基本信息\n")
        f.write("-" * 70 + "\n")
        f.write(f"指数代码: HSI\n")
        f.write(f"指数名称: 恒生指数\n")
        f.write(f"日期范围: {df_returns['Date'].iloc[0]} 至 {df_returns['Date'].iloc[-1]}\n")
        f.write(f"交易日数: {len(hsi_returns):,} 天\n")
        f.write(f"投资期限: {n_years:.2f} 年\n")
        f.write(f"无风险利率: {RISK_FREE_RATE:.4f} ({RISK_FREE_RATE * 100:.2f}%)\n\n")

        # 二、收益率指标
        f.write("二、收益率指标\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'指标':<25} {'数值':>15} {'百分比':>15}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'日均收益率':<25} {mean_daily_return:>15.6f} {mean_daily_return * 100:>14.4f}%\n")
        f.write(f"{'年化收益率':<25} {annualized_return:>15.6f} {annualized_return * 100:>14.2f}%\n")
        f.write(f"{'年化复合收益率(CAGR)':<25} {cagr:>15.6f} {cagr * 100:>14.2f}%\n")
        f.write(f"{'累计收益率':<25} {total_return:>15.6f} {total_return * 100:>14.2f}%\n\n")

        # 三、风险指标
        f.write("三、风险指标\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'指标':<25} {'数值':>15} {'百分比':>15}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'日波动率':<25} {daily_std:>15.6f} {daily_std * 100:>14.4f}%\n")
        f.write(f"{'年化波动率':<25} {annualized_std:>15.6f} {annualized_std * 100:>14.2f}%\n")
        f.write(f"{'下行波动率(年化)':<25} {downside_std_annual:>15.6f} {downside_std_annual * 100:>14.2f}%\n")
        f.write(f"{'最大回撤':<25} {max_drawdown:>15.6f} {max_drawdown * 100:>14.2f}%\n")
        f.write(f"{'VaR(95%,日)':<25} {var_95:>15.6f} {var_95 * 100:>14.4f}%\n\n")

        # 四、风险调整后指标
        f.write("四、风险调整后指标\n")
        f.write("-" * 70 + "\n")
        f.write(f"夏普比率: {sharpe_ratio:.4f}\n")
        f.write(f"  计算公式: (年化收益率 - 无风险利率) / 年化波动率\n")
        f.write(f"  含义: 每承受1单位风险获得的超额收益\n\n")
        f.write(f"Sortino比率: {sortino_ratio:.4f}\n")
        f.write(f"  计算公式: (年化收益率 - 无风险利率) / 下行波动率\n")
        f.write(f"  含义: 只考虑下行风险的风险调整收益\n\n")

        # 五、坐标位置
        f.write("五、风险-收益坐标系位置\n")
        f.write("-" * 70 + "\n")
        f.write(f"X轴(风险): {annualized_std:.6f} ({annualized_std * 100:.2f}%)\n")
        f.write(f"Y轴(收益): {annualized_return:.6f} ({annualized_return * 100:.2f}%)\n")
        f.write(f"斜率(夏普比率): {sharpe_ratio:.4f}\n\n")

        # 六、输出文件
        f.write("六、输出文件\n")
        f.write("-" * 70 + "\n")
        f.write(f"指标文件: {OUTPUT_METRICS.name}\n")
        f.write(f"  大小: {OUTPUT_METRICS.stat().st_size / 1024:.2f} KB\n")
        f.write(f"  格式: JSON\n\n")

        # 七、说明
        f.write("七、说明\n")
        f.write("-" * 70 + "\n")
        f.write("1. 年化收益率: 反映指数的长期收益能力\n")
        f.write("2. 年化波动率: 衡量指数价格波动的不确定性，是风险的主要指标\n")
        f.write("3. 夏普比率: 衡量风险调整后的收益，数值越高越好\n")
        f.write("4. 最大回撤: 从历史最高点到最低点的最大跌幅\n")
        f.write("5. 数据用途: 本数据将用于与有效前沿进行对比分析\n\n")

    print(f"  报告文件: {OUTPUT_REPORT}")
    print()

    # 9. 完成
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"\nHSI关键指标:")
    print(f"  年化收益率: {annualized_return * 100:>7.2f}%")
    print(f"  年化波动率: {annualized_std * 100:>7.2f}%")
    print(f"  夏普比率:   {sharpe_ratio:>7.4f}")
    print()
    print(f"输出文件:")
    print(f"  - {OUTPUT_METRICS}")
    print(f"  - {OUTPUT_REPORT}")
    print()


if __name__ == "__main__":
    main()
