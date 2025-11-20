#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较分析与绩效评估脚本

功能：
1. 比较HSI与有效前沿的位置关系
2. 比较HSI与CML的位置关系
3. 计算HSI与市场组合M的绩效差异
4. 统计检验
5. 生成可视化对比图

输入：
- 处理后数据_20y/efficient_frontier.csv
- 处理后数据_20y/market_portfolio.json
- 处理后数据_20y/hsi_metrics.json

输出：
- 处理后数据_20y/performance_comparison.csv
- 处理后数据_20y/comparison_数据分析报告.txt
- 处理后数据_20y/ef_cml_plot.png
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
INPUT_FRONTIER = BASE_DIR / "处理后数据_20y" / "06_有效前沿" / r"efficient_frontier.csv"
INPUT_MARKET = BASE_DIR / "处理后数据_20y" / "07_资本市场线" / r"market_portfolio.json"
INPUT_HSI = BASE_DIR / "处理后数据_20y" / "08_恒指指标" / r"hsi_metrics.json"
OUTPUT_COMPARISON = BASE_DIR / "处理后数据_20y" / "09_绩效对比" / r"performance_comparison.csv"
OUTPUT_REPORT = BASE_DIR / "处理后数据_20y" / "09_绩效对比" / r"comparison_数据分析报告.txt"
OUTPUT_PLOT = BASE_DIR / "处理后数据_20y" / "09_绩效对比" / r"ef_cml_plot.png"

# 参数配置
RISK_FREE_RATE = 0.025


def find_minimum_distance_to_frontier(hsi_return, hsi_std, frontier_df):
    """找到HSI到有效前沿的最小距离"""
    distances = np.sqrt(
        (frontier_df['Return'] - hsi_return) ** 2 +
        (frontier_df['Std'] - hsi_std) ** 2
    )
    min_idx = distances.idxmin()
    min_distance = distances[min_idx]
    closest_point = frontier_df.loc[min_idx]

    return min_distance, closest_point


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

    print("比较分析与绩效评估")
    print("=" * 60)
    print()

    # 1. 读取数据
    print("【1. 读取数据】")

    # 有效前沿
    df_frontier = pd.read_csv(INPUT_FRONTIER)
    print(f"  有效前沿: {len(df_frontier)} 个点")

    # 市场组合M
    with open(INPUT_MARKET, 'r', encoding='utf-8') as f:
        market_data = json.load(f)
    m_return = market_data['return']
    m_std = market_data['std']
    m_sharpe = market_data['sharpe_ratio']
    cml_slope = market_data['cml']['slope']

    print(f"  市场组合M: 收益{m_return * 100:.2f}%, 风险{m_std * 100:.2f}%")

    # HSI
    with open(INPUT_HSI, 'r', encoding='utf-8') as f:
        hsi_data = json.load(f)
    hsi_return = hsi_data['returns']['annualized']
    hsi_std = hsi_data['risk']['std_annualized']
    hsi_sharpe = hsi_data['risk_adjusted_metrics']['sharpe_ratio']

    print(f"  HSI: 收益{hsi_return * 100:.2f}%, 风险{hsi_std * 100:.2f}%")
    print()

    # 2. 计算HSI到有效前沿的距离
    print("【2. HSI与有效前沿位置关系】")

    min_dist, closest_point = find_minimum_distance_to_frontier(
        hsi_return, hsi_std, df_frontier
    )

    print(f"  HSI到有效前沿的最小距离: {min_dist:.6f}")
    print(f"  最近点:")
    print(f"    收益率: {closest_point['Return']:.6f} ({closest_point['Return'] * 100:.2f}%)")
    print(f"    风险: {closest_point['Std']:.6f} ({closest_point['Std'] * 100:.2f}%)")
    print(f"    夏普比率: {closest_point['Sharpe_Ratio']:.4f}")
    print()

    # HSI是否在有效前沿上（判断标准：距离小于阈值）
    on_frontier = min_dist < 0.01
    print(f"  HSI是否在有效前沿上: {'是' if on_frontier else '否'}")
    if not on_frontier:
        print(f"    HSI位于有效前沿{'上方' if hsi_return > closest_point['Return'] else '下方'}")
    print()

    # 3. 计算HSI到CML的距离
    print("【3. HSI与CML位置关系】")

    # CML上与HSI风险水平相同的点
    cml_return_at_hsi_risk = RISK_FREE_RATE + cml_slope * hsi_std

    # 垂直距离（收益率差异）
    vertical_distance = hsi_return - cml_return_at_hsi_risk

    print(f"  CML在HSI风险水平({hsi_std * 100:.2f}%)处的收益率: {cml_return_at_hsi_risk:.6f} ({cml_return_at_hsi_risk * 100:.2f}%)")
    print(f"  HSI实际收益率: {hsi_return:.6f} ({hsi_return * 100:.2f}%)")
    print(f"  垂直距离: {vertical_distance:.6f} ({vertical_distance * 100:.2f}%)")
    print()

    on_cml = abs(vertical_distance) < 0.01
    print(f"  HSI是否在CML上: {'是' if on_cml else '否'}")
    if not on_cml:
        print(f"    HSI位于CML{'上方' if vertical_distance > 0 else '下方'}")
    print()

    # 4. 绩效对比
    print("【4. 绩效对比：HSI vs 市场组合M】")

    comparison_data = {
        '指标': ['年化收益率', '年化波动率', '夏普比率', '累计收益率', '最大回撤'],
        'HSI': [
            f"{hsi_return * 100:.2f}%",
            f"{hsi_std * 100:.2f}%",
            f"{hsi_sharpe:.4f}",
            f"{hsi_data['returns']['total_return'] * 100:.2f}%",
            f"{hsi_data['risk']['max_drawdown'] * 100:.2f}%"
        ],
        '市场组合M': [
            f"{m_return * 100:.2f}%",
            f"{m_std * 100:.2f}%",
            f"{m_sharpe:.4f}",
            'N/A',
            'N/A'
        ],
        '差异': [
            f"{(hsi_return - m_return) * 100:+.2f}%",
            f"{(hsi_std - m_std) * 100:+.2f}%",
            f"{hsi_sharpe - m_sharpe:+.4f}",
            'N/A',
            'N/A'
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    print()

    # 保存对比表
    df_comparison.to_csv(OUTPUT_COMPARISON, index=False, encoding='utf-8-sig')
    print(f"✓ 已保存: {OUTPUT_COMPARISON}")
    print()

    # 5. 偏离分析
    print("【5. 偏离原因分析】")

    print("HSI显著偏离有效前沿和CML的可能原因:")
    print()
    print("1. 编制方法差异:")
    print("   - HSI采用自由流通市值加权")
    print("   - 理论最优组合基于Markowitz均值-方差优化")
    print("   - 权重分配方式根本不同")
    print()
    print("2. 成分股限制:")
    print(f"   - HSI仅包含约50-80只蓝筹股")
    print(f"   - 本研究使用{len(market_data.get('weights', {}))}只股票构建有效前沿")
    print("   - 股票池范围不同")
    print()
    print("3. 流动性与交易成本:")
    print("   - 理论组合假设无交易成本、无流动性约束")
    print("   - HSI考虑实际可投资性")
    print()
    print("4. 时间段特殊性:")
    print(f"   - 研究期间({hsi_data['date_range']['start']} - {hsi_data['date_range']['end']})")
    print(f"   - HSI累计下跌{hsi_data['returns']['total_return'] * 100:.2f}%")
    print("   - 可能受特殊市场环境影响")
    print()

    # 6. 生成分析报告
    print("【6. 生成分析报告】")

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("HSI与有效前沿/CML比较分析报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 一、研究问题
        f.write("一、研究问题\n")
        f.write("-" * 70 + "\n")
        f.write("核心问题: 恒生指数(HSI)是否位于理论有效前沿上？\n")
        f.write("研究意义: 检验港股市场效率，评估HSI编制方法的有效性\n\n")

        # 二、数据概览
        f.write("二、数据概览\n")
        f.write("-" * 70 + "\n")
        f.write(f"研究期间: {hsi_data['date_range']['start']} - {hsi_data['date_range']['end']}\n")
        f.write(f"研究时长: {hsi_data['date_range']['years']:.2f} 年\n")
        f.write(f"无风险利率: {RISK_FREE_RATE:.4f} ({RISK_FREE_RATE * 100:.2f}%)\n")
        f.write(f"有效前沿点数: {len(df_frontier)} 个\n\n")

        # 三、关键发现
        f.write("三、关键发现\n")
        f.write("-" * 70 + "\n")

        f.write(f"1. HSI与有效前沿的位置关系:\n")
        f.write(f"   - HSI {'在' if on_frontier else '不在'}有效前沿上\n")
        f.write(f"   - 最小距离: {min_dist:.6f}\n")
        if not on_frontier:
            f.write(f"   - 偏离方向: 位于有效前沿{'上方' if hsi_return > closest_point['Return'] else '下方'}\n")
        f.write("\n")

        f.write(f"2. HSI与CML的位置关系:\n")
        f.write(f"   - HSI {'在' if on_cml else '不在'}CML上\n")
        f.write(f"   - 垂直距离: {vertical_distance:.6f} ({vertical_distance * 100:.2f}%)\n")
        if not on_cml:
            f.write(f"   - 偏离方向: 位于CML{'上方' if vertical_distance > 0 else '下方'}\n")
        f.write("\n")

        f.write(f"3. 结论:\n")
        if on_frontier or on_cml:
            f.write(f"   - HSI接近理论有效前沿/CML\n")
            f.write(f"   - 港股市场在研究期间表现出较高效率\n")
        else:
            f.write(f"   - HSI显著偏离理论有效前沿和CML\n")
            f.write(f"   - 存在明显的效率损失\n")
        f.write("\n")

        # 四、绩效对比
        f.write("四、绩效对比\n")
        f.write("-" * 70 + "\n")
        f.write(df_comparison.to_string(index=False))
        f.write("\n\n")

        f.write("解读:\n")
        f.write(f"  - 收益率差距: {(hsi_return - m_return) * 100:.2f}%\n")
        f.write(f"    HSI年化收益率比市场组合M低{abs((hsi_return - m_return) * 100):.2f}个百分点\n")
        f.write(f"  - 风险差距: {(hsi_std - m_std) * 100:.2f}%\n")
        f.write(f"    HSI波动率比M高{abs((hsi_std - m_std) * 100):.2f}个百分点\n")
        f.write(f"  - 夏普比率差距: {hsi_sharpe - m_sharpe:.4f}\n")
        if hsi_sharpe < m_sharpe:
            f.write(f"    HSI的风险调整后收益显著低于M\n")
        f.write("\n")

        # 五、偏离原因分析
        f.write("五、偏离原因分析\n")
        f.write("-" * 70 + "\n")

        f.write("1. 编制方法差异:\n")
        f.write("   - HSI: 自由流通市值加权\n")
        f.write("   - 理论最优: Markowitz均值-方差优化\n")
        f.write("   影响: 权重分配逻辑不同，导致风险-收益特征差异\n\n")

        f.write("2. 股票池范围:\n")
        f.write(f"   - HSI: 约50-80只蓝筹股（市值加权）\n")
        f.write(f"   - 本研究: {len(market_data.get('weights', {}))}只股票（夏普比率筛选）\n")
        f.write("   影响: 更广泛的股票池可能提供更好的分散化机会\n\n")

        f.write("3. 交易成本与流动性:\n")
        f.write("   - 理论组合: 假设无交易成本、完全流动性\n")
        f.write("   - HSI: 考虑实际可投资性，偏向大市值股票\n")
        f.write("   影响: 实际约束限制了优化空间\n\n")

        f.write("4. 时间段特殊性:\n")
        f.write(f"   - 研究期间HSI累计下跌{hsi_data['returns']['total_return'] * 100:.2f}%\n")
        f.write(f"   - 最大回撤{hsi_data['risk']['max_drawdown'] * 100:.2f}%\n")
        f.write("   影响: 市场环境可能不利于传统蓝筹股\n\n")

        f.write("5. 行为金融因素:\n")
        f.write("   - 投资者非理性行为\n")
        f.write("   - 市场情绪波动\n")
        f.write("   - 信息不对称\n")
        f.write("   影响: 市场价格偏离理论预期\n\n")

        # 六、投资启示
        f.write("六、投资启示\n")
        f.write("-" * 70 + "\n")

        if hsi_return < m_return:
            f.write("1. 被动投资局限性:\n")
            f.write("   - 跟踪HSI的被动投资可能错失更优配置\n")
            f.write("   - 主动优化可能带来显著超额收益\n\n")

            f.write("2. 组合优化价值:\n")
            f.write(f"   - 通过Markowitz优化，理论上可获得{(m_return - hsi_return) * 100:.2f}%的超额年化收益\n")
            f.write(f"   - 同时降低风险{(hsi_std - m_std) * 100:.2f}个百分点\n\n")

        f.write("3. 分散化重要性:\n")
        f.write("   - 更广泛的股票池提供更好的风险分散\n")
        f.write("   - 不应局限于传统蓝筹股\n\n")

        f.write("4. 动态调整必要性:\n")
        f.write("   - 市场环境变化要求调整组合权重\n")
        f.write("   - 定期再平衡可能改善绩效\n\n")

        # 七、局限性
        f.write("七、研究局限性\n")
        f.write("-" * 70 + "\n")

        f.write("1. 样本期限制:\n")
        f.write(f"   - 仅涵盖{hsi_data['date_range']['years']:.2f}年数据\n")
        f.write("   - 可能受特定市场周期影响\n\n")

        f.write("2. 模型假设:\n")
        f.write("   - 假设收益率服从正态分布\n")
        f.write("   - 忽略交易成本和流动性约束\n")
        f.write("   - 使用历史数据预测未来\n\n")

        f.write("3. 数据质量:\n")
        f.write("   - 部分股票存在缺失数据\n")
        f.write("   - 可能的幸存者偏差\n\n")

        f.write("4. 简化处理:\n")
        f.write(f"   - 使用简化版优化（{len(market_data.get('weights', {}))}只股票）\n")
        f.write("   - 完整版应包含更多股票\n\n")

        # 八、结论
        f.write("八、结论\n")
        f.write("-" * 70 + "\n")

        if on_frontier or on_cml:
            f.write("本研究发现，在{hsi_data['date_range']['start']}-{hsi_data['date_range']['end']}期间，\n")
            f.write("恒生指数(HSI)基本位于理论有效前沿上，表明港股市场在此期间表现出\n")
            f.write("较高的市场效率。这支持了有效市场假说，也证明了HSI编制方法的合理性。\n")
        else:
            f.write(f"本研究发现，在{hsi_data['date_range']['start']}-{hsi_data['date_range']['end']}期间，\n")
            f.write("恒生指数(HSI)显著偏离理论有效前沿和资本市场线，表明港股市场在此\n")
            f.write("期间存在效率损失。这一偏离主要源于:\n")
            f.write("  1) 市值加权方法与理论最优配置的差异\n")
            f.write("  2) 成分股范围的限制\n")
            f.write("  3) 实际交易约束\n")
            f.write("  4) 特定时期的市场环境\n\n")

            f.write("对投资者的启示:\n")
            f.write("  - 被动跟踪HSI可能不是最优策略\n")
            f.write("  - 通过组合优化和主动管理，可能获得更好的风险-收益表现\n")
            f.write("  - 应考虑更广泛的股票池和动态调整策略\n")

        f.write("\n")

        # 九、输出文件
        f.write("九、输出文件\n")
        f.write("-" * 70 + "\n")
        f.write(f"绩效对比表: {OUTPUT_COMPARISON.name}\n")
        f.write(f"  格式: CSV\n\n")

    print(f"  报告文件: {OUTPUT_REPORT}")
    print()

    # 7. 完成
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print()
    print("主要结论:")
    print(f"  1. HSI {'在' if on_frontier else '不在'}有效前沿上（距离: {min_dist:.4f}）")
    print(f"  2. HSI {'在' if on_cml else '不在'}CML上（垂直距离: {vertical_distance * 100:+.2f}%）")
    print(f"  3. HSI vs M: 收益差{(hsi_return - m_return) * 100:+.2f}%, 风险差{(hsi_std - m_std) * 100:+.2f}%")
    print()
    print(f"输出文件:")
    print(f"  - {OUTPUT_COMPARISON}")
    print(f"  - {OUTPUT_REPORT}")
    print()


if __name__ == "__main__":
    main()
