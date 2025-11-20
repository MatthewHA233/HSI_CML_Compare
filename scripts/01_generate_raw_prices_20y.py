"""
并行处理20年港股CSV数据，生成raw_prices.csv
- 处理5个时间段（2005-2008, 2009-2012, 2013-2016, 2017-2020, 2021-2024）
- 每个时段包含多个CSV文件（HK_STK_DQUOTE.csv, HK_STK_DQUOTE1.csv等）
- 长格式转宽格式：(日期,股票代码,价格) → 日期×股票矩阵
- 最后合并成完整的20年数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time
import glob


def process_period_folder(folder_info):
    """
    处理单个时期文件夹的所有CSV文件

    Args:
        folder_info: (folder_path, period_name, output_csv_path) 元组

    Returns:
        (output_csv_path, stock_count, date_count, processing_time)
    """
    folder_path, period_name, output_csv_path = folder_info
    start_time = time.time()

    print(f"[{period_name}] 开始处理...")

    # 查找所有HK_STK_DQUOTE*.csv文件
    csv_files = sorted(glob.glob(str(folder_path / "HK_STK_DQUOTE*.csv")))
    # 排除描述文件
    csv_files = [f for f in csv_files if '[DES]' not in f]

    print(f"[{period_name}] 找到 {len(csv_files)} 个CSV文件")

    # 读取并合并所有CSV
    df_list = []
    for csv_file in csv_files:
        # 读取CSV（注意BOM头）
        df_part = pd.read_csv(csv_file, encoding='utf-8-sig')
        df_list.append(df_part)
        print(f"[{period_name}]   读取: {Path(csv_file).name} - {len(df_part):,} 行")

    # 纵向合并（拼接所有行）
    df = pd.concat(df_list, ignore_index=True)
    print(f"[{period_name}] 合并后: {len(df):,} 行")

    # 数据预处理
    df['TradingDate'] = pd.to_datetime(df['TradingDate'])
    df['Symbol'] = df['Symbol'].astype(str).str.zfill(5)  # 股票代码补齐为5位

    # 处理零价格和负价格
    invalid_prices = (df['ClosePrice'] <= 0) | (df['ClosePrice'].isna())
    if invalid_prices.any():
        print(f"[{period_name}] 处理无效价格: {invalid_prices.sum():,} 条记录")
        df.loc[invalid_prices, 'ClosePrice'] = np.nan

    # 去重（同一天同一股票可能有重复记录）
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['Symbol', 'TradingDate'], keep='first')
    after_dedup = len(df)
    if before_dedup != after_dedup:
        print(f"[{period_name}] 去重: {before_dedup - after_dedup:,} 条重复记录")

    # 获取股票基础信息
    stock_info = df.groupby('Symbol').agg({
        'SecurityID': 'first'
    }).reset_index()

    # 创建透视表（长格式→宽格式）
    print(f"[{period_name}] 转换为宽格式...")
    price_pivot = df.pivot_table(
        index='Symbol',
        columns='TradingDate',
        values='ClosePrice',
        aggfunc='first'
    )

    # 格式化日期列名
    price_pivot.columns = price_pivot.columns.strftime('%Y-%m-%d')

    # 合并信息和价格
    result = stock_info.merge(price_pivot, left_on='Symbol', right_index=True, how='left')

    # 保存单独CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    processing_time = time.time() - start_time
    stock_count = len(result)
    date_count = len(price_pivot.columns)

    print(f"[{period_name}] 完成 - {stock_count}只股票 × {date_count}个交易日 - {processing_time:.1f}秒")

    return output_csv_path, stock_count, date_count, processing_time


def main():
    print("=" * 80)
    print("生成 raw_prices.csv（20年数据，2005-2024）")
    print("=" * 80)

    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # 定义所有时期文件夹
    period_folders = [
        ('港股历史行情表2005-01-01 至 2008-12-31', '2005-2008'),
        ('港股历史行情表2009-01-01至2012-12-31', '2009-2012'),
        ('港股历史行情表2013-01-01至2016-12-31', '2013-2016'),
        ('港股历史行情表2017-01-01至2020-12-31', '2017-2020'),
        ('港股历史行情2021-01-01至2024-12-31', '2021-2024'),
    ]

    # 准备所有任务
    tasks = []
    for folder_name, period_label in period_folders:
        folder_path = project_root / "下载数据集_20年" / folder_name
        output_csv_path = project_root / "处理后数据" / "单独csv_20年" / f"{period_label}.csv"

        tasks.append((folder_path, period_label, output_csv_path))

    print(f"\n总共 {len(tasks)} 个时期需要处理")
    print(f"使用并行处理加速...\n")

    # 并行处理所有时期
    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=min(5, len(tasks))) as executor:
        futures = {executor.submit(process_period_folder, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = futures[future]
                print(f"[错误] 处理 {task[1]} 失败: {e}")
                import traceback
                traceback.print_exc()

    total_processing_time = time.time() - start_time

    print(f"\n{'=' * 80}")
    print("并行处理完成！")
    print(f"总耗时: {total_processing_time:.1f}秒")
    print(f"{'=' * 80}\n")

    # 合并所有时期的CSV文件
    print("开始合并所有时期...")
    merge_start = time.time()

    # 读取所有时期的CSV
    period_dfs = []
    for csv_path, stock_count, date_count, _ in sorted(results):
        df = pd.read_csv(csv_path)
        print(f"  读取: {csv_path.name} - {stock_count}只股票 × {date_count}天")
        period_dfs.append(df)

    # 获取所有股票的并集
    all_symbols = set()
    for df in period_dfs:
        all_symbols.update(df['Symbol'].unique())

    print(f"\n所有时期股票总数（并集）: {len(all_symbols)} 只")

    # 逐步合并（按Symbol外连接）
    print("\n逐步合并各时期...")
    merged_df = period_dfs[0]

    for i, df in enumerate(period_dfs[1:], 1):
        print(f"  合并时期 {i+1}...")
        merged_df = merged_df.merge(df, on='Symbol', how='outer', suffixes=('', f'_p{i+1}'))

        # 处理SecurityID的合并（优先使用第一个有效值）
        if f'SecurityID_p{i+1}' in merged_df.columns:
            merged_df['SecurityID'] = merged_df['SecurityID'].fillna(merged_df[f'SecurityID_p{i+1}'])
            merged_df = merged_df.drop(columns=[f'SecurityID_p{i+1}'])

    print(f"  合并完成: {len(merged_df)} 只股票")

    # 只保留到2024-12-31的数据
    print("\n筛选数据到2024-12-31...")
    info_cols = ['Symbol', 'SecurityID']
    date_cols_all = [col for col in merged_df.columns if col not in info_cols]

    # 筛选日期列
    date_cols_filtered = [col for col in date_cols_all if pd.to_datetime(col) <= pd.to_datetime('2024-12-31')]

    # 重组DataFrame
    merged_df = merged_df[info_cols + date_cols_filtered]

    print(f"  原交易日: {len(date_cols_all)} 天")
    print(f"  筛选后: {len(date_cols_filtered)} 天")
    if len(date_cols_all) != len(date_cols_filtered):
        print(f"  移除2025年数据: {len(date_cols_all) - len(date_cols_filtered)} 天")

    # 保存最终合并文件
    output_folder = project_root / "处理后数据" / "01_原始价格"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "raw_prices.csv"

    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    merge_time = time.time() - merge_start

    print(f"\n{'=' * 80}")
    print("最终结果")
    print(f"{'=' * 80}")
    print(f"输出文件: {output_file}")
    print(f"文件大小: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"总股票数: {len(merged_df)}")

    # 获取所有日期列（排除信息列）
    date_cols = [col for col in merged_df.columns if col not in info_cols]
    print(f"总交易日: {len(date_cols)}")

    if date_cols:
        print(f"日期范围: {date_cols[0]} → {date_cols[-1]}")

    # 数据完整度统计
    total_cells = len(merged_df) * len(date_cols)
    missing_cells = merged_df[date_cols].isna().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0

    print(f"\n数据完整度:")
    print(f"  总单元格数: {total_cells:,}")
    print(f"  缺失单元格: {missing_cells:,}")
    print(f"  缺失率: {missing_pct:.2f}%")
    print(f"  完整率: {100 - missing_pct:.2f}%")

    print(f"\n合并耗时: {merge_time:.1f}秒")
    print(f"总耗时: {total_processing_time + merge_time:.1f}秒")

    print(f"\n{'=' * 80}")
    print("✓ 全部完成！")
    print(f"{'=' * 80}")

    print(f"\n生成文件:")
    print(f"  - 单独CSV: {len(results)} 个文件在 处理后数据/单独csv_20年/")
    print(f"  - 合并文件: {output_file}")

    # 生成分析报告
    print("\n生成分析报告...")
    report_file = output_folder / "raw_数据分析报告.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("港股历史行情数据处理分析报告（20年数据：2005-2024）\n")
        f.write("=" * 80 + "\n\n")

        # 处理信息
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理耗时: {total_processing_time + merge_time:.1f}秒\n")
        f.write(f"  - 并行处理: {total_processing_time:.1f}秒\n")
        f.write(f"  - 合并数据: {merge_time:.1f}秒\n\n")

        # 输入文件信息
        f.write("-" * 80 + "\n")
        f.write("一、输入数据信息\n")
        f.write("-" * 80 + "\n\n")
        f.write("数据源文件夹:\n")
        for i, (folder_name, period_label) in enumerate(period_folders, 1):
            f.write(f"  {i}. {folder_name}\n")
        f.write(f"\n总时期数: {len(period_folders)} 个（2005-2024，共20年）\n\n")

        # 各时期统计
        f.write("-" * 80 + "\n")
        f.write("二、各时期数据统计\n")
        f.write("-" * 80 + "\n\n")

        for i, (csv_path, stock_count, date_count, proc_time) in enumerate(sorted(results), 1):
            f.write(f"{i}. {csv_path.name}\n")
            f.write(f"   股票数量: {stock_count:,} 只\n")
            f.write(f"   交易日数: {date_count:,} 天\n")
            f.write(f"   处理时间: {proc_time:.1f} 秒\n")
            f.write(f"   文件大小: {csv_path.stat().st_size / 1024 / 1024:.2f} MB\n\n")

        # 合并后统计
        f.write("-" * 80 + "\n")
        f.write("三、合并文件统计 (raw_prices.csv)\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"文件路径: {output_file}\n")
        f.write(f"文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB\n\n")

        f.write(f"数据维度:\n")
        f.write(f"  总股票数: {len(merged_df):,} 只\n")
        f.write(f"  总交易日: {len(date_cols):,} 天\n")
        f.write(f"  信息列数: {len(info_cols)} 列 (Symbol, SecurityID)\n")
        f.write(f"  总列数: {len(merged_df.columns):,} 列\n\n")

        if date_cols:
            f.write(f"时间范围:\n")
            f.write(f"  起始日期: {date_cols[0]}\n")
            f.write(f"  结束日期: {date_cols[-1]}\n")
            f.write(f"  跨度天数: {(pd.to_datetime(date_cols[-1]) - pd.to_datetime(date_cols[0])).days + 1} 天\n\n")

        # 股票代码范围
        all_symbols = merged_df['Symbol'].tolist()
        f.write(f"股票代码范围:\n")
        f.write(f"  最小代码: {min(all_symbols)}\n")
        f.write(f"  最大代码: {max(all_symbols)}\n\n")

        # 数据完整度
        f.write("-" * 80 + "\n")
        f.write("四、数据完整度分析\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"总单元格数: {total_cells:,}\n")
        f.write(f"有效数据量: {total_cells - missing_cells:,}\n")
        f.write(f"缺失数据量: {missing_cells:,}\n")
        f.write(f"数据完整率: {100 - missing_pct:.2f}%\n")
        f.write(f"数据缺失率: {missing_pct:.2f}%\n\n")

        f.write("说明:\n")
        f.write("  - 20年数据中存在大量缺失属于正常现象，主要原因:\n")
        f.write("    1. 不同股票在不同时期上市（前期缺失）或退市（后期缺失）\n")
        f.write("    2. 部分股票存在长期停牌或暂停交易\n")
        f.write("    3. 流动性较差的股票交易不活跃，存在间歇性停牌\n")
        f.write("    4. 2005年上市的股票较少，后期逐步增加\n\n")

        # 按股票统计缺失率
        stock_missing = merged_df[date_cols].isna().sum(axis=1) / len(date_cols) * 100
        f.write("股票层面缺失率分布:\n")
        f.write(f"  完整数据股票(缺失率<10%): {(stock_missing < 10).sum():,} 只 ({(stock_missing < 10).sum() / len(merged_df) * 100:.2f}%)\n")
        f.write(f"  部分缺失股票(10%≤缺失率<50%): {((stock_missing >= 10) & (stock_missing < 50)).sum():,} 只 ({((stock_missing >= 10) & (stock_missing < 50)).sum() / len(merged_df) * 100:.2f}%)\n")
        f.write(f"  大量缺失股票(50%≤缺失率<90%): {((stock_missing >= 50) & (stock_missing < 90)).sum():,} 只 ({((stock_missing >= 50) & (stock_missing < 90)).sum() / len(merged_df) * 100:.2f}%)\n")
        f.write(f"  几乎无数据股票(缺失率≥90%): {(stock_missing >= 90).sum():,} 只 ({(stock_missing >= 90).sum() / len(merged_df) * 100:.2f}%)\n\n")

        # 文件列表
        f.write("-" * 80 + "\n")
        f.write("五、生成文件清单\n")
        f.write("-" * 80 + "\n\n")

        f.write("各时期CSV文件 (处理后数据/单独csv_20年/):\n")
        for csv_path, _, _, _ in sorted(results):
            f.write(f"  - {csv_path.name}\n")

        f.write(f"\n合并CSV文件 (处理后数据/01_原始价格/):\n")
        f.write(f"  - raw_prices.csv\n")
        f.write(f"  - raw_数据分析报告.txt (本报告)\n\n")

        f.write("=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")

    print(f"✓ 分析报告已保存: {report_file}")


if __name__ == '__main__':
    main()
