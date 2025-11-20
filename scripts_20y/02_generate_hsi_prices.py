#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数(HSI)月度价格数据提取脚本（20年数据）

功能：
1. 从Hang Seng Historical Data.csv中提取HSI恒生指数数据
2. 筛选日期范围：2005-01-01 至 2024-12-31
3. 月度重采样：取每月最后一个交易日价格
4. 生成hsi_prices.csv文件（月度）
5. 生成数据分析报告

输入：
- 下载数据集_20年/Hang Seng Historical Data.csv

输出：
- 处理后数据_20y/02_恒指价格/hsi_prices.csv
- 处理后数据_20y/02_恒指价格/hsi_数据分析报告.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "下载数据集_20年" / "Hang Seng Historical Data.csv"
OUTPUT_DIR = BASE_DIR / "处理后数据_20y" / "02_恒指价格"
OUTPUT_CSV = OUTPUT_DIR / "hsi_prices.csv"
OUTPUT_REPORT = OUTPUT_DIR / "hsi_数据分析报告.txt"

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("恒生指数(HSI)月度价格数据提取（20年）")
    print("=" * 60)
    print()

    # 1. 读取CSV文件
    print("【1. 读取原始数据】")
    print(f"输入文件: {INPUT_FILE}")

    # 读取CSV，处理逗号分隔的数字
    df = pd.read_csv(INPUT_FILE, thousands=',')

    print(f"  总记录数: {len(df):,} 条")
    print(f"  列名: {list(df.columns)}")
    print()

    # 2. 数据预处理
    print("【2. 数据预处理】")

    # 只保留需要的列
    hsi_df = df[['Date', 'Price']].copy()
    hsi_df.columns = ['TradingDate', 'ClosePrice']

    # 转换日期格式（MM/DD/YYYY → datetime）
    hsi_df['TradingDate'] = pd.to_datetime(hsi_df['TradingDate'], format='%m/%d/%Y')

    # 按日期升序排序（CSV是倒序的）
    hsi_df = hsi_df.sort_values('TradingDate').reset_index(drop=True)

    print(f"  日期范围: {hsi_df['TradingDate'].min().date()} 至 {hsi_df['TradingDate'].max().date()}")
    print(f"  日度数据: {len(hsi_df):,} 个交易日")
    print()

    # 3. 筛选日期范围（2005-01-01 至 2024-12-31）
    print("【3. 筛选日期范围】")
    start_date = pd.to_datetime('2005-01-01')
    end_date = pd.to_datetime('2024-12-31')

    before_filter = len(hsi_df)
    hsi_df = hsi_df[(hsi_df['TradingDate'] >= start_date) &
                    (hsi_df['TradingDate'] <= end_date)]
    after_filter = len(hsi_df)

    print(f"  目标范围: 2005-01-01 至 2024-12-31")
    print(f"  筛选前: {before_filter:,} 条")
    print(f"  筛选后: {after_filter:,} 条")
    if before_filter != after_filter:
        print(f"  移除数据: {before_filter - after_filter} 条")
    print()

    # 4. 月度重采样
    print("【4. 月度重采样】")
    print(f"  日度数据: {len(hsi_df)} 个交易日")

    # 设置日期为索引
    hsi_df = hsi_df.set_index('TradingDate')

    # 取每月最后一个交易日
    hsi_monthly = hsi_df.resample('M').last()

    # 删除缺失值（某些月份可能没有数据）
    hsi_monthly = hsi_monthly.dropna()

    print(f"  月度数据: {len(hsi_monthly)} 个月")
    print(f"  月度范围: {hsi_monthly.index.min().date()} 至 {hsi_monthly.index.max().date()}")
    print()

    # 5. 数据质量检查
    print("【5. 数据质量检查】")

    # 检查缺失值
    missing_prices = hsi_monthly['ClosePrice'].isnull().sum()
    zero_prices = (hsi_monthly['ClosePrice'] == 0).sum()
    negative_prices = (hsi_monthly['ClosePrice'] < 0).sum()

    print(f"  缺失价格: {missing_prices} 条")
    print(f"  零价格: {zero_prices} 条")
    print(f"  负价格: {negative_prices} 条")

    if missing_prices + zero_prices + negative_prices == 0:
        print("  ✓ 数据质量良好")
    else:
        print("  ⚠ 发现异常数据")
    print()

    # 6. 生成输出格式
    print("【6. 生成输出文件】")

    # 转换为宽格式：(Date, Price)
    output_df = pd.DataFrame({
        'Date': hsi_monthly.index.strftime('%Y-%m-%d'),
        'ClosePrice': hsi_monthly['ClosePrice'].values
    })

    # 保存CSV
    output_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"  ✓ 已保存: {OUTPUT_CSV}")
    print(f"  文件大小: {OUTPUT_CSV.stat().st_size / 1024:.2f} KB")
    print()

    # 7. 统计分析
    print("【7. 统计分析】")

    prices = hsi_monthly['ClosePrice']

    print(f"  数据点数: {len(prices)} 个月")
    print(f"  最小值: {prices.min():,.2f}")
    print(f"  最大值: {prices.max():,.2f}")
    print(f"  平均值: {prices.mean():,.2f}")
    print(f"  中位数: {prices.median():,.2f}")
    print(f"  标准差: {prices.std():,.2f}")
    print()

    # 计算月度收益率
    monthly_returns = np.log(prices / prices.shift(1)).dropna()

    print(f"  月度收益率统计:")
    print(f"    均值: {monthly_returns.mean():.4f} ({monthly_returns.mean() * 100:.2f}%)")
    print(f"    标准差: {monthly_returns.std():.4f} ({monthly_returns.std() * 100:.2f}%)")
    print(f"    最小值: {monthly_returns.min():.4f} ({monthly_returns.min() * 100:.2f}%)")
    print(f"    最大值: {monthly_returns.max():.4f} ({monthly_returns.max() * 100:.2f}%)")

    # 年化指标
    annual_return = monthly_returns.mean() * 12
    annual_volatility = monthly_returns.std() * np.sqrt(12)

    print(f"\n  年化指标:")
    print(f"    年化收益率: {annual_return:.4f} ({annual_return * 100:.2f}%)")
    print(f"    年化波动率: {annual_volatility:.4f} ({annual_volatility * 100:.2f}%)")
    print()

    # 8. 生成分析报告
    print("【8. 生成分析报告】")

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("恒生指数(HSI)月度数据分析报告（20年：2005-2024）\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入文件: {INPUT_FILE}\n")
        f.write(f"输出文件: {OUTPUT_CSV}\n\n")

        f.write("-" * 70 + "\n")
        f.write("一、数据范围\n")
        f.write("-" * 70 + "\n\n")
        f.write(f"日度数据: {after_filter:,} 个交易日\n")
        f.write(f"月度数据: {len(hsi_monthly):,} 个月\n")
        f.write(f"起始日期: {hsi_monthly.index.min().strftime('%Y-%m-%d')}\n")
        f.write(f"结束日期: {hsi_monthly.index.max().strftime('%Y-%m-%d')}\n")
        f.write(f"跨度: {(hsi_monthly.index.max() - hsi_monthly.index.min()).days} 天 "
                f"(约 {len(hsi_monthly) / 12:.1f} 年)\n\n")

        f.write("-" * 70 + "\n")
        f.write("二、价格统计\n")
        f.write("-" * 70 + "\n\n")
        f.write(f"最低点: {prices.min():,.2f} ({prices.idxmin().strftime('%Y-%m-%d')})\n")
        f.write(f"最高点: {prices.max():,.2f} ({prices.idxmax().strftime('%Y-%m-%d')})\n")
        f.write(f"平均价格: {prices.mean():,.2f}\n")
        f.write(f"中位数: {prices.median():,.2f}\n")
        f.write(f"标准差: {prices.std():,.2f}\n")
        f.write(f"变异系数: {prices.std() / prices.mean():.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("三、收益率分析\n")
        f.write("-" * 70 + "\n\n")
        f.write(f"月度收益率:\n")
        f.write(f"  样本数: {len(monthly_returns)} 个月\n")
        f.write(f"  均值: {monthly_returns.mean():.4f} ({monthly_returns.mean() * 100:.2f}%)\n")
        f.write(f"  标准差: {monthly_returns.std():.4f} ({monthly_returns.std() * 100:.2f}%)\n")
        f.write(f"  最小值: {monthly_returns.min():.4f} ({monthly_returns.min() * 100:.2f}%)\n")
        f.write(f"  最大值: {monthly_returns.max():.4f} ({monthly_returns.max() * 100:.2f}%)\n\n")

        f.write(f"年化指标:\n")
        f.write(f"  年化收益率: {annual_return:.4f} ({annual_return * 100:.2f}%)\n")
        f.write(f"  年化波动率: {annual_volatility:.4f} ({annual_volatility * 100:.2f}%)\n")
        f.write(f"  年化夏普比率(Rf=0): {annual_return / annual_volatility:.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("四、数据频率说明\n")
        f.write("-" * 70 + "\n\n")
        f.write("本数据为月度频率（取每月最后一个交易日收盘价）\n")
        f.write("- 长期研究（20年）使用月度数据是学术界标准做法\n")
        f.write("- 月度数据减少微观结构噪音，提高统计稳健性\n")
        f.write("- 年化因子：收益率 × 12，波动率 × √12\n\n")

        f.write("=" * 70 + "\n")
        f.write("报告结束\n")
        f.write("=" * 70 + "\n")

    print(f"  ✓ 已保存: {OUTPUT_REPORT}")
    print()

    print("=" * 60)
    print("✓ HSI月度数据提取完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
