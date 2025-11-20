#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建资本市场线(CML)脚本

功能：
1. 从有效前沿中找到最大夏普比率的组合（切点组合M）
2. 构建资本市场线CML
3. 计算市场组合M的详细信息

输入：
- 处理后数据_20y/efficient_frontier.csv
- 处理后数据_20y/summary_stats.csv
- 处理后数据_20y/covariance.csv

输出：
- 处理后数据_20y/market_portfolio.json
- 处理后数据_20y/cml_数据分析报告.txt
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
INPUT_FRONTIER = BASE_DIR / "处理后数据_20y" / "06_有效前沿" / r"efficient_frontier.csv"
INPUT_STATS = BASE_DIR / "处理后数据_20y" / "05_统计特征" / r"summary_stats.csv"
INPUT_COV = BASE_DIR / "处理后数据_20y" / "05_统计特征" / r"covariance.csv"
OUTPUT_MARKET = BASE_DIR / "处理后数据_20y" / "07_资本市场线" / r"market_portfolio.json"
OUTPUT_REPORT = BASE_DIR / "处理后数据_20y" / "07_资本市场线" / r"cml_数据分析报告.txt"

# 参数配置
RISK_FREE_RATE = 0.025  # 无风险利率2.5%
USE_POSITIVE_RETURN_ONLY = False  # 使用全部股票（与步骤6保持一致）
TOP_N_STOCKS = None  # 使用全部2092只股票


def main():
    # 创建输出目录
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("构建资本市场线(CML)")
    print("=" * 60)
    print()

    # 1. 读取有效前沿数据
    print("【1. 读取有效前沿数据】")
    print(f"输入文件: {INPUT_FRONTIER}")

    df_frontier = pd.read_csv(INPUT_FRONTIER)

    print(f"  有效前沿点数: {len(df_frontier)} 个")
    print(f"  收益率范围: {df_frontier['Return'].min():.4f} 至 {df_frontier['Return'].max():.4f}")
    print(f"  风险范围: {df_frontier['Std'].min():.4f} 至 {df_frontier['Std'].max():.4f}")
    print()

    # 2. 找到切点组合M（最大夏普比率）
    print("【2. 找到切点组合M（最大夏普比率）】")
    print(f"  无风险利率: {RISK_FREE_RATE:.4f} ({RISK_FREE_RATE * 100:.2f}%)")
    print()

    # 计算每个点的夏普比率（使用无风险利率）
    sharpe_ratios_with_rf = (df_frontier['Return'] - RISK_FREE_RATE) / df_frontier['Std']

    # 找到最大夏普比率的点
    max_sharpe_idx = sharpe_ratios_with_rf.idxmax()
    market_portfolio = df_frontier.loc[max_sharpe_idx]

    m_return = market_portfolio['Return']
    m_std = market_portfolio['Std']
    m_sharpe = sharpe_ratios_with_rf[max_sharpe_idx]

    print(f"市场组合M:")
    print(f"  期望收益率: {m_return:.6f} ({m_return * 100:.2f}%)")
    print(f"  标准差: {m_std:.6f} ({m_std * 100:.2f}%)")
    print(f"  夏普比率: {m_sharpe:.4f}")
    print()

    # 3. 重新优化找到M的权重
    print("【3. 重新优化求解M组合的权重】")

    # 读取统计数据
    df_stats_full = pd.read_csv(INPUT_STATS)

    # 筛选（与步骤6一致）
    # 第一步：正期望收益筛选
    if USE_POSITIVE_RETURN_ONLY:
        df_stats_full = df_stats_full[df_stats_full['Mean_Return_Annual'] > 0].reset_index(drop=True)

    # 第二步：按夏普比率筛选（如果需要）
    if TOP_N_STOCKS is not None and len(df_stats_full) > TOP_N_STOCKS:
        df_stats = df_stats_full.nlargest(TOP_N_STOCKS, 'Sharpe_Ratio')
        selected_indices = df_stats.index.tolist()
        df_stats = df_stats.reset_index(drop=True)
    else:
        df_stats = df_stats_full.copy()
        selected_indices = None

    symbols = df_stats['Symbol'].values
    mean_returns = df_stats['Mean_Return_Annual'].values

    # 读取协方差矩阵并筛选
    df_cov = pd.read_csv(INPUT_COV, index_col=0)

    # 将columns转换为整数（与index类型一致）
    df_cov.columns = df_cov.columns.astype(int)

    # 使用股票代码筛选协方差矩阵（与步骤6保持一致）
    df_cov = df_cov.loc[symbols, symbols]

    cov_matrix = df_cov.values

    print(f"  股票数: {len(symbols)} 只")

    # 优化求解最大夏普比率组合
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        """负夏普比率（用于最小化）"""
        portfolio_return = weights @ mean_returns
        portfolio_std = np.sqrt(weights @ cov_matrix @ weights)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std
        return -sharpe  # 最小化负夏普比率 = 最大化夏普比率

    # 初始猜测
    init_weights = np.ones(len(symbols)) / len(symbols)

    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 权重和为1
    ]

    # 边界条件
    bounds = tuple((0, 1) for _ in range(len(symbols)))

    # 优化
    result = minimize(
        neg_sharpe_ratio,
        init_weights,
        args=(mean_returns, cov_matrix, RISK_FREE_RATE),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    if result.success:
        m_weights = result.x
        m_return_calc = m_weights @ mean_returns
        m_std_calc = np.sqrt(m_weights @ cov_matrix @ m_weights)
        m_sharpe_calc = (m_return_calc - RISK_FREE_RATE) / m_std_calc

        print(f"✓ 优化成功")
        print(f"  期望收益率: {m_return_calc:.6f} ({m_return_calc * 100:.2f}%)")
        print(f"  标准差: {m_std_calc:.6f} ({m_std_calc * 100:.2f}%)")
        print(f"  夏普比率: {m_sharpe_calc:.4f}")
        print()

        # 找出权重最大的前20只股票
        print(f"  权重最大的前20只股票:")
        top_20_idx = np.argsort(m_weights)[-20:][::-1]
        for idx in top_20_idx:
            if m_weights[idx] > 0.001:  # 只显示权重>0.1%的
                print(f"    {symbols[idx]}: {m_weights[idx] * 100:.2f}%")
        print()

    else:
        print(f"✗ 优化失败: {result.message}")
        m_weights = None
        m_return_calc = m_return
        m_std_calc = m_std
        m_sharpe_calc = m_sharpe

    # 4. 构建CML
    print("【4. 构建CML】")

    # CML方程：E(Rp) = Rf + [(E(Rm) - Rf) / σm] * σp
    # 斜率 = (E(Rm) - Rf) / σm
    cml_slope = (m_return_calc - RISK_FREE_RATE) / m_std_calc

    print(f"CML方程:")
    print(f"  E(Rp) = {RISK_FREE_RATE:.4f} + {cml_slope:.4f} * σp")
    print(f"  截距(Rf): {RISK_FREE_RATE:.4f}")
    print(f"  斜率: {cml_slope:.4f}")
    print()

    # 生成CML上的点（用于绘图）
    cml_std_range = np.linspace(0, m_std_calc * 2, 100)
    cml_return_range = RISK_FREE_RATE + cml_slope * cml_std_range

    # 5. 保存市场组合数据
    print("【5. 保存市场组合数据】")

    market_data = {
        'type': 'Market Portfolio (Tangency Portfolio)',
        'risk_free_rate': float(RISK_FREE_RATE),
        'return': float(m_return_calc),
        'std': float(m_std_calc),
        'variance': float(m_std_calc ** 2),
        'sharpe_ratio': float(m_sharpe_calc),
        'cml': {
            'slope': float(cml_slope),
            'intercept': float(RISK_FREE_RATE),
            'equation': f"E(Rp) = {RISK_FREE_RATE:.4f} + {cml_slope:.4f} * σp"
        },
        'weights': {
            str(symbols[i]): float(m_weights[i])
            for i in range(len(symbols))
            if m_weights is not None and m_weights[i] > 1e-6
        } if m_weights is not None else {},
        'top_holdings': [
            {
                'symbol': str(symbols[i]),
                'weight': float(m_weights[i]),
                'percentage': f"{m_weights[i] * 100:.2f}%"
            }
            for i in np.argsort(m_weights)[-20:][::-1]
            if m_weights is not None and m_weights[i] > 0.001
        ] if m_weights is not None else []
    }

    with open(OUTPUT_MARKET, 'w', encoding='utf-8') as f:
        json.dump(market_data, f, indent=2, ensure_ascii=False)

    print(f"✓ 已保存: {OUTPUT_MARKET}")
    print()

    # 6. 生成报告
    print("【6. 生成分析报告】")

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("资本市场线(CML)分析报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 一、理论基础
        f.write("一、理论基础\n")
        f.write("-" * 70 + "\n")
        f.write("资本市场线(CML):\n")
        f.write("  - 由Sharpe(1964)提出的资本资产定价模型(CAPM)核心概念\n")
        f.write("  - 连接无风险资产和市场组合的直线\n")
        f.write("  - 代表所有有效组合的风险-收益关系\n\n")

        f.write("切点组合(市场组合M):\n")
        f.write("  - 有效前沿与从无风险利率出发的切线的切点\n")
        f.write("  - 在给定风险水平下提供最高夏普比率的组合\n")
        f.write("  - 理论上应该包含所有风险资产，按市值加权\n\n")

        # 二、无风险利率
        f.write("二、无风险利率\n")
        f.write("-" * 70 + "\n")
        f.write(f"Rf = {RISK_FREE_RATE:.4f} ({RISK_FREE_RATE * 100:.2f}%)\n")
        f.write("  来源: HKMA 3个月期国债收益率（假设值）\n")
        f.write("  说明: 无风险利率是CML的截距，代表零风险时的收益\n\n")

        # 三、市场组合M
        f.write("三、市场组合M（切点组合）\n")
        f.write("-" * 70 + "\n")
        f.write(f"期望收益率: {m_return_calc:.6f} ({m_return_calc * 100:.2f}%)\n")
        f.write(f"标准差（风险）: {m_std_calc:.6f} ({m_std_calc * 100:.2f}%)\n")
        f.write(f"方差: {m_std_calc**2:.6f}\n")
        f.write(f"夏普比率: {m_sharpe_calc:.4f}\n\n")

        f.write(f"说明:\n")
        f.write(f"  - 夏普比率{m_sharpe_calc:.4f}是所有有效组合中的最大值\n")
        f.write(f"  - 每承受1单位风险，可获得{m_sharpe_calc:.4f}单位超额收益\n\n")

        # 四、CML方程
        f.write("四、资本市场线(CML)方程\n")
        f.write("-" * 70 + "\n")
        f.write(f"E(Rp) = {RISK_FREE_RATE:.4f} + {cml_slope:.4f} * σp\n\n")

        f.write(f"参数说明:\n")
        f.write(f"  截距: Rf = {RISK_FREE_RATE:.4f} (无风险利率)\n")
        f.write(f"  斜率: {cml_slope:.4f} (风险的市场价格)\n")
        f.write(f"  E(Rp): 组合的期望收益率\n")
        f.write(f"  σp: 组合的标准差（风险）\n\n")

        f.write(f"斜率含义:\n")
        f.write(f"  - 斜率{cml_slope:.4f}表示风险的市场价格\n")
        f.write(f"  - 每增加1单位风险，期望收益增加{cml_slope:.4f}\n")
        f.write(f"  - 等于市场组合的夏普比率\n\n")

        # 五、组合权重
        if m_weights is not None and len(market_data['top_holdings']) > 0:
            f.write("五、市场组合权重分布\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'股票代码':<10} {'权重':>10} {'百分比':>10}\n")
            f.write("-" * 70 + "\n")
            for holding in market_data['top_holdings']:
                f.write(f"{holding['symbol']:<10} {holding['weight']:>10.6f} {holding['percentage']:>10}\n")
            f.write(f"\n总持仓数: {len([w for w in m_weights if w > 1e-6])} 只股票\n")
            f.write(f"前10大持仓占比: {sum([h['weight'] for h in market_data['top_holdings'][:10]]) * 100:.2f}%\n\n")

        # 六、输出文件
        f.write("六、输出文件\n")
        f.write("-" * 70 + "\n")
        f.write(f"市场组合数据: {OUTPUT_MARKET.name}\n")
        f.write(f"  大小: {OUTPUT_MARKET.stat().st_size / 1024:.2f} KB\n")
        f.write(f"  格式: JSON\n")
        f.write(f"  包含: 收益、风险、夏普比率、CML方程、持仓权重\n\n")

        # 七、说明
        f.write("七、说明\n")
        f.write("-" * 70 + "\n")
        f.write("1. CML理论意义: 所有理性投资者应持有市场组合M和无风险资产的组合\n")
        f.write("2. 风险偏好: 风险厌恶者持有更多无风险资产，风险偏好者借入资金投资M\n")
        f.write("3. 市场效率检验: HSI是否在CML上是检验港股市场效率的关键\n")
        f.write("4. 局限性: 实际中M难以精确确定，本研究用夏普比率最大组合近似\n")
        f.write("5. 数据用途: 用于与HSI进行对比，评估恒指相对于理论最优的偏离\n\n")

    print(f"  报告文件: {OUTPUT_REPORT}")
    print()

    # 7. 完成
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"\n市场组合M关键指标:")
    print(f"  期望收益率: {m_return_calc * 100:>7.2f}%")
    print(f"  标准差:     {m_std_calc * 100:>7.2f}%")
    print(f"  夏普比率:   {m_sharpe_calc:>7.4f}")
    print(f"\nCML方程:")
    print(f"  E(Rp) = {RISK_FREE_RATE:.4f} + {cml_slope:.4f} * σp")
    print()
    print(f"输出文件:")
    print(f"  - {OUTPUT_MARKET}")
    print(f"  - {OUTPUT_REPORT}")
    print()


if __name__ == "__main__":
    main()
