#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制有效前沿、CML、HSI对比图

生成论文级别的可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
BASE_DIR = Path(__file__).parent.parent
INPUT_FRONTIER = BASE_DIR / "处理后数据_20y" / "06_有效前沿" / r"efficient_frontier.csv"
INPUT_GMV = BASE_DIR / "处理后数据_20y" / "06_有效前沿" / r"gmv_portfolio.json"
INPUT_MARKET = BASE_DIR / "处理后数据_20y" / "07_资本市场线" / r"market_portfolio.json"
INPUT_HSI = BASE_DIR / "处理后数据_20y" / "08_恒指指标" / r"hsi_metrics.json"
OUTPUT_PLOT = BASE_DIR / "处理后数据_20y" / "10_可视化" / r"ef_cml_plot.png"

# 参数配置
RISK_FREE_RATE = 0.025
FIG_SIZE = (12, 8)
DPI = 300


def main():
    print("=" * 60)
    # 创建输出目录
    BASE_DIR = Path(__file__).parent.parent
    output_dirs = [
        BASE_DIR / "处理后数据_20y" / "04_收益率",
        BASE_DIR / "处理后数据_20y" / "05_统计特征",
        BASE_DIR / "处理后数据_20y" / "06_有效前沿",
        BASE_DIR / "处理后数据_20y" / "07_资本市场线",
        BASE_DIR / "处理后数据_20y" / "08_HSI指标",
        BASE_DIR / "处理后数据_20y" / "09_绩效对比",
        BASE_DIR / "处理后数据_20y" / "10_可视化",
    ]
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("绘制有效前沿、CML、HSI对比图")
    print("=" * 60)
    print()

    # 1. 读取数据
    print("【1. 读取数据】")

    # 有效前沿
    df_frontier = pd.read_csv(INPUT_FRONTIER)
    print(f"  有效前沿: {len(df_frontier)} 个点")

    # GMV
    with open(INPUT_GMV, 'r', encoding='utf-8') as f:
        gmv_data = json.load(f)
    gmv_return = gmv_data['return']
    gmv_std = gmv_data['std']

    # 市场组合M
    with open(INPUT_MARKET, 'r', encoding='utf-8') as f:
        market_data = json.load(f)
    m_return = market_data['return']
    m_std = market_data['std']
    cml_slope = market_data['cml']['slope']

    # HSI
    with open(INPUT_HSI, 'r', encoding='utf-8') as f:
        hsi_data = json.load(f)
    hsi_return = hsi_data['returns']['annualized']
    hsi_std = hsi_data['risk']['std_annualized']

    print(f"  GMV: 收益{gmv_return * 100:.2f}%, 风险{gmv_std * 100:.2f}%")
    print(f"  市场组合M: 收益{m_return * 100:.2f}%, 风险{m_std * 100:.2f}%")
    print(f"  HSI: 收益{hsi_return * 100:.2f}%, 风险{hsi_std * 100:.2f}%")
    print()

    # 2. 创建图表
    print("【2. 创建图表】")

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    # 绘制有效前沿
    ax.plot(df_frontier['Std'] * 100, df_frontier['Return'] * 100,
            'b-', linewidth=2.5, label='Efficient Frontier', alpha=0.8)

    # 绘制CML
    # CML从无风险利率出发，经过M点，延伸到更高风险
    cml_std_range = np.linspace(0, max(hsi_std, m_std) * 1.3, 100)
    cml_return_range = (RISK_FREE_RATE + cml_slope * cml_std_range) * 100
    ax.plot(cml_std_range * 100, cml_return_range,
            'r--', linewidth=2.5, label='Capital Market Line (CML)', alpha=0.8)

    # 绘制无风险利率点
    ax.plot(0, RISK_FREE_RATE * 100, 'gs', markersize=12, label='Risk-free Rate', zorder=5)

    # 绘制GMV点
    ax.plot(gmv_std * 100, gmv_return * 100, 'mo', markersize=12,
            label='Global Minimum Variance (GMV)', zorder=5)

    # 绘制市场组合M点
    ax.plot(m_std * 100, m_return * 100, 'r*', markersize=20,
            label='Market Portfolio (M)', zorder=5, markeredgecolor='darkred', markeredgewidth=1.5)

    # 绘制HSI点
    ax.plot(hsi_std * 100, hsi_return * 100, 'ko', markersize=14,
            label='Hang Seng Index (HSI)', zorder=5, markerfacecolor='yellow', markeredgewidth=2)

    # 添加标注
    offset = 1.5
    ax.annotate(f'Rf={RISK_FREE_RATE * 100:.1f}%',
                xy=(0, RISK_FREE_RATE * 100),
                xytext=(offset, RISK_FREE_RATE * 100 + 2),
                fontsize=10, ha='left')

    ax.annotate(f'GMV\n({gmv_std * 100:.1f}%, {gmv_return * 100:.1f}%)',
                xy=(gmv_std * 100, gmv_return * 100),
                xytext=(gmv_std * 100 + offset, gmv_return * 100 - 3),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

    ax.annotate(f'M\n({m_std * 100:.1f}%, {m_return * 100:.1f}%)\nSharpe={market_data["sharpe_ratio"]:.2f}',
                xy=(m_std * 100, m_return * 100),
                xytext=(m_std * 100 + offset, m_return * 100 + 5),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.annotate(f'HSI\n({hsi_std * 100:.1f}%, {hsi_return * 100:.1f}%)\nSharpe={hsi_data["risk_adjusted_metrics"]["sharpe_ratio"]:.2f}',
                xy=(hsi_std * 100, hsi_return * 100),
                xytext=(hsi_std * 100 + offset, hsi_return * 100 - 8),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 设置坐标轴
    ax.set_xlabel('Risk (Standard Deviation, %)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Expected Return (%)', fontsize=14, fontweight='bold')
    ax.set_title('Efficient Frontier, CML, and HSI Comparison\n港股市场有效前沿与恒生指数对比',
                 fontsize=16, fontweight='bold', pad=20)

    # 网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 图例
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # 设置坐标轴范围
    x_min = -1
    x_max = max(hsi_std, m_std) * 110  # 转换为百分比
    y_min = min(hsi_return, RISK_FREE_RATE, gmv_return) * 110
    y_max = max(m_return, df_frontier['Return'].max()) * 110

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 添加水平线和垂直线辅助对比
    ax.axhline(y=hsi_return * 100, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=hsi_std * 100, color='gray', linestyle=':', alpha=0.3)

    # 紧凑布局
    plt.tight_layout()

    # 保存
    plt.savefig(OUTPUT_PLOT, dpi=DPI, bbox_inches='tight')
    print(f"✓ 已保存: {OUTPUT_PLOT}")
    print(f"  分辨率: {FIG_SIZE[0]}×{FIG_SIZE[1]} 英寸, {DPI} DPI")
    print()

    # 显示统计信息
    print("【3. 关键指标对比】")
    print()
    print(f"{'指标':<20} {'HSI':>15} {'市场组合M':>15} {'差异':>15}")
    print("-" * 70)
    print(f"{'年化收益率 (%)':<20} {hsi_return * 100:>15.2f} {m_return * 100:>15.2f} {(hsi_return - m_return) * 100:>15.2f}")
    print(f"{'年化波动率 (%)':<20} {hsi_std * 100:>15.2f} {m_std * 100:>15.2f} {(hsi_std - m_std) * 100:>15.2f}")
    print(f"{'夏普比率':<20} {hsi_data['risk_adjusted_metrics']['sharpe_ratio']:>15.2f} {market_data['sharpe_ratio']:>15.2f} {hsi_data['risk_adjusted_metrics']['sharpe_ratio'] - market_data['sharpe_ratio']:>15.2f}")
    print()

    # 完成
    print("=" * 60)
    print("绘图完成！")
    print("=" * 60)
    print()
    print(f"图表已保存至: {OUTPUT_PLOT}")
    print(f"可用于论文、报告展示")
    print()


if __name__ == "__main__":
    main()
