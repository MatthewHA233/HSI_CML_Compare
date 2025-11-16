"""
并行处理所有Excel文件，生成raw_prices.csv
- 并行读取5个Excel文件（2019-2022三个 + 2023-2025两个）
- 每个文件单独保存到"单独csv"文件夹
- 最后合并成完整的raw_prices.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time

# 列名定义
COLUMN_NAMES = ['TradingDate', 'Symbol', 'InstitutionID', 'SecurityID', 'ClosePrice', 'Change', 'ChangeRatio']


def process_excel_file(file_info):
    """
    处理单个Excel文件，返回宽格式DataFrame

    Args:
        file_info: (file_path, output_csv_path, file_name) 元组

    Returns:
        (output_csv_path, stock_count, date_count, processing_time)
    """
    file_path, output_csv_path, file_name = file_info
    start_time = time.time()

    print(f"[{file_name}] 开始读取...")

    # 读取Excel（跳过前3行表头）
    df = pd.read_excel(file_path, skiprows=3, header=None, names=COLUMN_NAMES)

    # 数据预处理
    df['TradingDate'] = pd.to_datetime(df['TradingDate'])
    df['Symbol'] = df['Symbol'].apply(lambda x: f"{x:05d}")

    # 处理零价格
    zero_prices = (df['ClosePrice'] == 0)
    if zero_prices.any():
        df.loc[zero_prices, 'ClosePrice'] = np.nan

    # 去重
    df = df.drop_duplicates(subset=['Symbol', 'TradingDate'], keep='first')

    # 获取股票基础信息
    stock_info = df.groupby('Symbol').agg({
        'InstitutionID': 'first',
        'SecurityID': 'first'
    }).reset_index()

    # 创建透视表
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

    print(f"[{file_name}] 完成 - {stock_count}只股票 × {date_count}个交易日 - {processing_time:.1f}秒")

    return output_csv_path, stock_count, date_count, processing_time


def main():
    print("=" * 80)
    print("生成 raw_prices.csv（并行处理所有文件）")
    print("=" * 80)

    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # 定义所有输入文件
    file_configs = [
        # 2019-2022 文件夹（3个文件）
        {
            'folder': '港股历史行情2019.1.1-2022.12.31',
            'files': [
                'HK_STK_DQUOTE.xlsx',
                'HK_STK_DQUOTE1.xlsx',
                'HK_STK_DQUOTE2.xlsx'
            ]
        },
        # 2023-2025 文件夹（2个文件）
        {
            'folder': '港股历史行情2023.1.1-2025.6.30',
            'files': [
                'HK_STK_DQUOTE.xlsx',
                'HK_STK_DQUOTE1.xlsx'
            ]
        }
    ]

    # 准备所有任务
    tasks = []
    for config in file_configs:
        folder_path = project_root / "下载数据集" / config['folder']
        period = config['folder'].replace('港股历史行情', '').replace('.', '_')

        for i, filename in enumerate(config['files'], 1):
            file_path = folder_path / filename
            output_csv_name = f"{period}_part{i}.csv"
            output_csv_path = project_root / "处理后数据" / "单独csv" / output_csv_name

            tasks.append((file_path, output_csv_path, filename))

    print(f"\n总共 {len(tasks)} 个文件需要处理")
    print(f"使用并行处理加速...\n")

    # 并行处理所有文件
    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=min(5, len(tasks))) as executor:
        futures = {executor.submit(process_excel_file, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = futures[future]
                print(f"[错误] 处理 {task[2]} 失败: {e}")

    total_processing_time = time.time() - start_time

    print(f"\n{'=' * 80}")
    print("并行处理完成！")
    print(f"总耗时: {total_processing_time:.1f}秒")
    print(f"{'=' * 80}\n")

    # 合并所有CSV文件
    print("开始合并所有CSV文件...")
    merge_start = time.time()

    # 按时间段分组
    df_2019_list = []
    df_2023_list = []

    for csv_path, stock_count, date_count, _ in sorted(results):
        df = pd.read_csv(csv_path)
        print(f"  读取: {csv_path.name} - {stock_count}只股票")

        if '2019' in csv_path.name:
            df_2019_list.append(df)
        else:
            df_2023_list.append(df)

    # 先在各时间段内合并（按行concat，不同股票）
    print("\n分时段合并...")
    df_2019 = pd.concat(df_2019_list, ignore_index=True) if df_2019_list else None
    df_2023 = pd.concat(df_2023_list, ignore_index=True) if df_2023_list else None

    # 在各时段内去重（因为边界股票可能在相邻part文件中重复）
    if df_2019 is not None:
        before_dedup_2019 = len(df_2019)
        df_2019 = df_2019.drop_duplicates(subset='Symbol', keep='first')
        after_dedup_2019 = len(df_2019)
        if before_dedup_2019 != after_dedup_2019:
            print(f"  2019-2022时段去重: {before_dedup_2019 - after_dedup_2019} 条")

    if df_2023 is not None:
        before_dedup_2023 = len(df_2023)
        df_2023 = df_2023.drop_duplicates(subset='Symbol', keep='first')
        after_dedup_2023 = len(df_2023)
        if before_dedup_2023 != after_dedup_2023:
            print(f"  2023-2025时段去重: {before_dedup_2023 - after_dedup_2023} 条")

    print(f"  2019-2022时段: {len(df_2019) if df_2019 is not None else 0} 只股票")
    print(f"  2023-2025时段: {len(df_2023) if df_2023 is not None else 0} 只股票")

    # 按Symbol合并两个时间段（在列方向合并）
    print("\n合并不同时间段...")
    if df_2019 is not None and df_2023 is not None:
        # 获取两个时间段的所有股票
        all_symbols = set(df_2019['Symbol'].unique()) | set(df_2023['Symbol'].unique())
        print(f"  总股票数（去重后）: {len(all_symbols)} 只")

        # 使用merge合并，on='Symbol'
        merged_df = df_2019.merge(df_2023, on='Symbol', how='outer', suffixes=('', '_2023'))

        # 处理InstitutionID和SecurityID的合并
        # 如果2019数据有，用2019的；否则用2023的
        if 'InstitutionID_2023' in merged_df.columns:
            merged_df['InstitutionID'] = merged_df['InstitutionID'].fillna(merged_df['InstitutionID_2023'])
            merged_df['SecurityID'] = merged_df['SecurityID'].fillna(merged_df['SecurityID_2023'])
            merged_df = merged_df.drop(columns=['InstitutionID_2023', 'SecurityID_2023'])

        print(f"  合并后: {len(merged_df)} 只股票")
    elif df_2019 is not None:
        merged_df = df_2019
        print("  只有2019-2022数据")
    else:
        merged_df = df_2023
        print("  只有2023-2025数据")

    # 只保留到2024-12-31的数据
    print("\n筛选数据到2024-12-31...")
    info_cols = ['Symbol', 'InstitutionID', 'SecurityID']
    date_cols_all = [col for col in merged_df.columns if col not in info_cols]

    # 筛选日期列
    date_cols_filtered = [col for col in date_cols_all if pd.to_datetime(col) <= pd.to_datetime('2024-12-31')]

    # 重组DataFrame
    merged_df = merged_df[info_cols + date_cols_filtered]

    print(f"  原交易日: {len(date_cols_all)} 天")
    print(f"  筛选后: {len(date_cols_filtered)} 天")
    print(f"  移除2025年数据: {len(date_cols_all) - len(date_cols_filtered)} 天")

    # 保存最终合并文件
    output_file = project_root / "处理后数据" / "raw_prices.csv"
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    merge_time = time.time() - merge_start

    print(f"\n{'=' * 80}")
    print("最终结果")
    print(f"{'=' * 80}")
    print(f"输出文件: {output_file}")
    print(f"文件大小: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"总股票数: {len(merged_df)}")

    # 获取所有日期列（排除信息列）
    info_cols = ['Symbol', 'InstitutionID', 'SecurityID']
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
    print(f"  - 单独CSV: {len(results)} 个文件在 处理后数据/单独csv/")
    print(f"  - 合并文件: 处理后数据/raw_prices.csv")

    # 生成分析报告
    print("\n生成分析报告...")
    report_file = project_root / "处理后数据" / "raw_数据分析报告.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("港股历史行情数据处理分析报告\n")
        f.write("=" * 80 + "\n\n")

        # 处理信息
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理耗时: {total_processing_time + merge_time:.1f}秒\n")
        f.write(f"  - 并行处理: {total_processing_time:.1f}秒\n")
        f.write(f"  - 合并数据: {merge_time:.1f}秒\n\n")

        # 输入文件信息
        f.write("-" * 80 + "\n")
        f.write("一、输入文件信息\n")
        f.write("-" * 80 + "\n\n")
        f.write("数据源文件夹:\n")
        f.write("  1. 港股历史行情2019.1.1-2022.12.31/ (3个Excel文件)\n")
        f.write("  2. 港股历史行情2023.1.1-2025.6.30/ (2个Excel文件)\n\n")
        f.write(f"总文件数: {len(results)} 个Excel文件\n\n")

        # 单独CSV统计
        f.write("-" * 80 + "\n")
        f.write("二、单独CSV文件统计\n")
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
        f.write(f"  信息列数: {len(info_cols)} 列 (Symbol, InstitutionID, SecurityID)\n")
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
        f.write("  - 存在缺失数据属于正常现象，主要原因:\n")
        f.write("    1. 不同股票在不同时期上市（前期缺失）或退市（后期缺失）\n")
        f.write("    2. 部分股票存在长期停牌或暂停交易\n")
        f.write("    3. 流动性较差的股票交易不活跃，存在间歇性停牌\n")
        f.write("    4. 数据源本身可能存在个别日期或股票的数据缺失\n\n")

        # 按股票统计缺失率
        stock_missing = merged_df[date_cols].isna().sum(axis=1) / len(date_cols) * 100
        f.write("股票层面缺失率分布:\n")
        f.write(f"  完整数据股票(缺失率<10%): {(stock_missing < 10).sum():,} 只 ({(stock_missing < 10).sum() / len(merged_df) * 100:.2f}%)\n")
        f.write(f"  部分缺失股票(10%≤缺失率<50%): {((stock_missing >= 10) & (stock_missing < 50)).sum():,} 只 ({((stock_missing >= 10) & (stock_missing < 50)).sum() / len(merged_df) * 100:.2f}%)\n")
        f.write(f"  大量缺失股票(50%≤缺失率<90%): {((stock_missing >= 50) & (stock_missing < 90)).sum():,} 只 ({((stock_missing >= 50) & (stock_missing < 90)).sum() / len(merged_df) * 100:.2f}%)\n")
        f.write(f"  几乎无数据股票(缺失率≥90%): {(stock_missing >= 90).sum():,} 只 ({(stock_missing >= 90).sum() / len(merged_df) * 100:.2f}%)\n\n")

        # 缺失数据模式分析
        f.write("缺失数据模式分析:\n")

        # 分析每只股票的数据模式
        price_data = merged_df[date_cols]

        # 计算每只股票首次和末次有数据的位置
        first_valid = price_data.apply(lambda row: row.first_valid_index(), axis=1)
        last_valid = price_data.apply(lambda row: row.last_valid_index(), axis=1)

        # 分类统计
        category_counts = {
            '新上市': 0,
            '已退市': 0,
            '中途停牌': 0,
            '完整交易': 0,
            '几乎无数据': 0
        }

        for idx in range(len(merged_df)):
            row = price_data.iloc[idx]
            missing_pct = stock_missing.iloc[idx]

            # 几乎无数据
            if missing_pct >= 90:
                category_counts['几乎无数据'] += 1
                continue

            # 获取首末有效数据的位置
            first_idx = first_valid.iloc[idx]
            last_idx = last_valid.iloc[idx]

            if pd.isna(first_idx) or pd.isna(last_idx):
                category_counts['几乎无数据'] += 1
                continue

            first_pos = date_cols.index(first_idx)
            last_pos = date_cols.index(last_idx)

            # 前期缺失超过20% → 新上市
            if first_pos > len(date_cols) * 0.2:
                category_counts['新上市'] += 1
            # 后期缺失超过20% → 已退市
            elif last_pos < len(date_cols) * 0.8:
                category_counts['已退市'] += 1
            # 计算中间段的缺失
            elif missing_pct > 10:
                category_counts['中途停牌'] += 1
            else:
                category_counts['完整交易'] += 1

        total_stocks = len(merged_df)
        f.write(f"  1. 完整交易股票: {category_counts['完整交易']:,} 只 ({category_counts['完整交易']/total_stocks*100:.2f}%)\n")
        f.write(f"     - 数据基本完整，全时段有交易记录\n\n")

        f.write(f"  2. 新上市股票: {category_counts['新上市']:,} 只 ({category_counts['新上市']/total_stocks*100:.2f}%)\n")
        f.write(f"     - 前期缺失，2019-2024年间上市\n\n")

        f.write(f"  3. 已退市股票: {category_counts['已退市']:,} 只 ({category_counts['已退市']/total_stocks*100:.2f}%)\n")
        f.write(f"     - 后期缺失，2019-2024年间退市\n\n")

        f.write(f"  4. 中途停牌股票: {category_counts['中途停牌']:,} 只 ({category_counts['中途停牌']/total_stocks*100:.2f}%)\n")
        f.write(f"     - 中间有较长时段缺失，可能停牌或流动性差\n\n")

        f.write(f"  5. 几乎无数据: {category_counts['几乎无数据']:,} 只 ({category_counts['几乎无数据']/total_stocks*100:.2f}%)\n")
        f.write(f"     - 数据严重缺失(>90%)，后续分析应剔除\n\n")

        # 文件列表
        f.write("-" * 80 + "\n")
        f.write("五、生成文件清单\n")
        f.write("-" * 80 + "\n\n")

        f.write("单独CSV文件 (处理后数据/单独csv/):\n")
        for csv_path, _, _, _ in sorted(results):
            f.write(f"  - {csv_path.name}\n")

        f.write(f"\n合并CSV文件 (处理后数据/):\n")
        f.write(f"  - raw_prices.csv\n")
        f.write(f"  - raw_数据分析报告.txt (本报告)\n\n")

        f.write("=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")

    print(f"✓ 分析报告已保存: {report_file}")


if __name__ == '__main__':
    main()
