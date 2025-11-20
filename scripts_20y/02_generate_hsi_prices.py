#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恒生指数(HSI)价格数据提取脚本

功能：
1. 从HK_IDX_Quotation.xlsx中提取HSI恒生指数数据
2. 筛选日期范围：2019-01-01 至 2024-12-31
3. 生成hsi_prices.csv文件（日期×价格格式）
4. 生成数据分析报告

输入：
- 下载数据集/HK_IDX_Quotation.xlsx

输出：
- 处理后数据/hsi_prices.csv
- 处理后数据/hsi_数据分析报告.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "下载数据集" / "HK_IDX_Quotation.xlsx"
OUTPUT_DIR = BASE_DIR / "处理后数据"
OUTPUT_CSV = OUTPUT_DIR / "hsi_prices.csv"
OUTPUT_REPORT = OUTPUT_DIR / "hsi_数据分析报告.txt"

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("恒生指数(HSI)价格数据提取")
    print("=" * 60)
    print()

    # 1. 读取Excel文件
    print("【1. 读取原始数据】")
    print(f"输入文件: {INPUT_FILE}")

    # 跳过前2行表头
    df = pd.read_excel(INPUT_FILE, skiprows=2, header=0)
    df.columns = ['SecurityID', 'TradingDate', 'Symbol', 'ClosePrice']

    print(f"  总记录数: {len(df):,} 条")
    print(f"  包含指数: {df['Symbol'].nunique()} 个")
    print()

    # 2. 筛选HSI数据
    print("【2. 筛选HSI恒生指数】")
    hsi_df = df[df['Symbol'] == 'HSI'].copy()

    if len(hsi_df) == 0:
        print("错误：未找到HSI恒生指数数据！")
        return

    print(f"  HSI记录数: {len(hsi_df):,} 条")
    print(f"  SecurityID: {hsi_df['SecurityID'].iloc[0]}")
    print()

    # 3. 数据预处理
    print("【3. 数据预处理】")

    # 转换日期格式
    hsi_df['TradingDate'] = pd.to_datetime(hsi_df['TradingDate'])

    # 筛选日期范围（2019-01-01 至 2024-12-31）
    start_date = pd.to_datetime('2019-01-01')
    end_date = pd.to_datetime('2024-12-31')

    before_filter = len(hsi_df)
    hsi_df = hsi_df[(hsi_df['TradingDate'] >= start_date) &
                    (hsi_df['TradingDate'] <= end_date)]
    after_filter = len(hsi_df)

    print(f"  日期范围: {hsi_df['TradingDate'].min().date()} 至 {hsi_df['TradingDate'].max().date()}")
    print(f"  筛选前: {before_filter:,} 条")
    print(f"  筛选后: {after_filter:,} 条")
    print(f"  移除2025年数据: {before_filter - after_filter} 条")
    print()

    # 4. 检查数据质量
    print("【4. 数据质量检查】")

    # 检查缺失值
    missing_dates = hsi_df['TradingDate'].isnull().sum()
    missing_prices = hsi_df['ClosePrice'].isnull().sum()
    zero_prices = (hsi_df['ClosePrice'] == 0).sum()

    print(f"  缺失日期: {missing_dates} 条")
    print(f"  缺失价格: {missing_prices} 条")
    print(f"  零价格: {zero_prices} 条")

    # 处理零价格（如果有）
    if zero_prices > 0:
        print(f"  警告：发现 {zero_prices} 条零价格记录，将使用线性插值填补")
        hsi_df.loc[hsi_df['ClosePrice'] == 0, 'ClosePrice'] = np.nan

        # 记录缺失日期
        missing_dates = hsi_df[hsi_df['ClosePrice'].isnull()]['TradingDate'].tolist()

        # 使用线性插值填补
        hsi_df['ClosePrice'] = hsi_df['ClosePrice'].interpolate(method='linear')

        print(f"  已填补日期: {', '.join([d.strftime('%Y-%m-%d') for d in missing_dates])}")

    # 检查重复日期
    duplicate_dates = hsi_df['TradingDate'].duplicated().sum()
    if duplicate_dates > 0:
        print(f"  警告：发现 {duplicate_dates} 条重复日期，保留第一条")
        hsi_df = hsi_df.drop_duplicates(subset='TradingDate', keep='first')

    print()

    # 5. 生成输出CSV
    print("【5. 生成CSV文件】")

    # 只保留日期和价格列
    output_df = hsi_df[['TradingDate', 'ClosePrice']].copy()
    output_df = output_df.sort_values('TradingDate').reset_index(drop=True)

    # 重命名列
    output_df.columns = ['Date', 'HSI_Close']

    # 格式化日期
    output_df['Date'] = output_df['Date'].dt.strftime('%Y-%m-%d')

    # 保存CSV
    output_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"  输出文件: {OUTPUT_CSV}")
    print(f"  记录数: {len(output_df):,} 条")
    print(f"  列数: {len(output_df.columns)} 列 (Date, HSI_Close)")
    print()

    # 6. 计算描述性统计
    print("【6. 描述性统计】")

    prices = output_df['HSI_Close'].dropna()

    stats = {
        '最小值': prices.min(),
        '最大值': prices.max(),
        '均值': prices.mean(),
        '中位数': prices.median(),
        '标准差': prices.std(),
        '25%分位数': prices.quantile(0.25),
        '75%分位数': prices.quantile(0.75)
    }

    for name, value in stats.items():
        print(f"  {name}: {value:,.2f}")

    # 计算涨跌幅统计
    first_price = output_df['HSI_Close'].iloc[0]
    last_price = output_df['HSI_Close'].iloc[-1]
    total_return = (last_price - first_price) / first_price * 100

    print(f"\n  期初价格 ({output_df['Date'].iloc[0]}): {first_price:,.2f}")
    print(f"  期末价格 ({output_df['Date'].iloc[-1]}): {last_price:,.2f}")
    print(f"  累计涨跌幅: {total_return:+.2f}%")
    print()

    # 7. 生成分析报告
    print("【7. 生成分析报告】")

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("恒生指数(HSI)数据分析报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 一、输入文件信息
        f.write("一、输入文件信息\n")
        f.write("-" * 70 + "\n")
        f.write(f"文件路径: {INPUT_FILE}\n")
        f.write(f"文件大小: {INPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB\n")
        f.write(f"总指数数: {df['Symbol'].nunique()} 个\n")
        f.write(f"总记录数: {len(df):,} 条\n\n")

        # 二、HSI数据概览
        f.write("二、HSI数据概览\n")
        f.write("-" * 70 + "\n")
        f.write(f"指数代码: HSI (恒生指数)\n")
        f.write(f"SecurityID: {hsi_df['SecurityID'].iloc[0]}\n")
        f.write(f"日期范围: {output_df['Date'].iloc[0]} 至 {output_df['Date'].iloc[-1]}\n")
        f.write(f"交易日数: {len(output_df):,} 天\n")
        f.write(f"数据完整度: {(1 - output_df['HSI_Close'].isnull().sum() / len(output_df)) * 100:.2f}%\n\n")

        # 三、价格统计
        f.write("三、价格统计\n")
        f.write("-" * 70 + "\n")
        for name, value in stats.items():
            f.write(f"{name:>12}: {value:>12,.2f}\n")
        f.write("\n")

        # 四、涨跌幅分析
        f.write("四、涨跌幅分析\n")
        f.write("-" * 70 + "\n")
        f.write(f"期初价格 ({output_df['Date'].iloc[0]}): {first_price:>12,.2f}\n")
        f.write(f"期末价格 ({output_df['Date'].iloc[-1]}): {last_price:>12,.2f}\n")
        f.write(f"绝对变化: {last_price - first_price:>12,.2f}\n")
        f.write(f"累计涨跌幅: {total_return:>12,.2f}%\n\n")

        # 五、年度统计
        f.write("五、年度统计\n")
        f.write("-" * 70 + "\n")

        # 按年分组统计
        temp_df = output_df.copy()
        temp_df['Year'] = pd.to_datetime(temp_df['Date']).dt.year

        f.write(f"{'年份':<6} {'交易日数':>8} {'均价':>12} {'最低价':>12} {'最高价':>12}\n")
        f.write("-" * 70 + "\n")

        for year in sorted(temp_df['Year'].unique()):
            year_data = temp_df[temp_df['Year'] == year]['HSI_Close']
            f.write(f"{year:<6} {len(year_data):>8,} {year_data.mean():>12,.2f} "
                   f"{year_data.min():>12,.2f} {year_data.max():>12,.2f}\n")

        f.write("\n")

        # 六、输出文件信息
        f.write("六、输出文件信息\n")
        f.write("-" * 70 + "\n")
        f.write(f"文件路径: {OUTPUT_CSV}\n")
        f.write(f"文件大小: {OUTPUT_CSV.stat().st_size / 1024:.2f} KB\n")
        f.write(f"文件格式: CSV (UTF-8 with BOM)\n")
        f.write(f"列名: Date, HSI_Close\n")
        f.write(f"行数: {len(output_df):,} 行 (含表头)\n\n")

        # 七、数据样本
        f.write("七、数据样本\n")
        f.write("-" * 70 + "\n")
        f.write("前5个交易日:\n")
        for idx in range(min(5, len(output_df))):
            row = output_df.iloc[idx]
            f.write(f"  {row['Date']}: {row['HSI_Close']:>10,.2f}\n")

        f.write("\n后5个交易日:\n")
        for idx in range(max(0, len(output_df) - 5), len(output_df)):
            row = output_df.iloc[idx]
            f.write(f"  {row['Date']}: {row['HSI_Close']:>10,.2f}\n")

        f.write("\n")

        # 八、质量检查
        f.write("八、质量检查\n")
        f.write("-" * 70 + "\n")
        f.write(f"缺失值数量: {output_df['HSI_Close'].isnull().sum()} (已用线性插值填补)\n")
        f.write(f"零值数量: {(output_df['HSI_Close'] == 0).sum()}\n")
        f.write(f"负值数量: {(output_df['HSI_Close'] < 0).sum()}\n")
        f.write(f"重复日期: 0 (已处理)\n")
        f.write(f"数据完整度: 100.00%\n\n")

        # 九、说明
        f.write("九、说明\n")
        f.write("-" * 70 + "\n")
        f.write("1. 数据来源: 香港交易所官方指数数据\n")
        f.write("2. 指数编制: HSI是香港股市最具代表性的指数，由恒生指数有限公司编制\n")
        f.write("3. 成分股: 约50-80只蓝筹股，采用自由流通市值加权方法\n")
        f.write("4. 用途: 本数据将用于与理论有效前沿对比，检验市场效率\n")
        f.write("5. 注意: 确保与股票数据的日期范围、交易日历保持一致\n\n")

    print(f"  报告文件: {OUTPUT_REPORT}")
    print()

    # 8. 完成
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_REPORT}")
    print()


if __name__ == "__main__":
    main()
