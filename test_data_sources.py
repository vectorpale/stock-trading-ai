"""
数据源基准测试脚本
Benchmark script for all supported data sources

运行方式 / Usage:
    cd stock_assistant
    python test_data_sources.py

测试内容 / Tests:
  - 每个数据源对美股/港股/A股各抓一只样本
  - 输出成功率、数据量、日期范围、耗时
  - 最终给出推荐配置
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# 把项目根目录加到路径
sys.path.insert(0, str(Path(__file__).parent))

# 只显示 WARNING 以上，避免刷屏
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

import pandas as pd

from src.data.fetcher import (
    fetch_yfinance,
    fetch_akshare_us,
    fetch_akshare_hk,
    fetch_akshare_cn,
    fetch_baostock,
    detect_market,
)

# ── 测试用股票代码 ──
TEST_CASES = [
    # (symbol,   label,          market)
    ('AAPL',    'Apple (US)',   'US'),
    ('MSFT',    'Microsoft (US)', 'US'),
    ('0700.HK', '腾讯 (HK)',    'HK'),
    ('9988.HK', '阿里 (HK)',    'HK'),
    ('600519',  '贵州茅台 (A股)', 'CN_SH'),
    ('000001',  '平安银行 (A股)', 'CN_SZ'),
]

# ── 数据源 → 测试配置 ──
SOURCES = {
    'yfinance':   {'fn': fetch_yfinance,   'markets': ['US', 'HK']},
    'akshare_us': {'fn': fetch_akshare_us, 'markets': ['US']},
    'akshare_hk': {'fn': fetch_akshare_hk, 'markets': ['HK']},
    'akshare_cn': {'fn': fetch_akshare_cn, 'markets': ['CN_SH', 'CN_SZ']},
    'baostock':   {'fn': fetch_baostock,   'markets': ['CN_SH', 'CN_SZ']},
}

PERIOD   = '1y'
INTERVAL = '1d'


def run_test(source_name: str, fn, symbol: str, label: str) -> dict:
    """运行单个测试，返回结果字典"""
    t0 = time.time()
    try:
        df = fn(symbol, PERIOD, INTERVAL)
        elapsed = time.time() - t0

        if df is not None and not df.empty:
            return {
                'source':  source_name,
                'symbol':  symbol,
                'label':   label,
                'status':  '✓',
                'rows':    len(df),
                'start':   str(df.index[0].date()),
                'end':     str(df.index[-1].date()),
                'elapsed': round(elapsed, 1),
                'error':   '',
            }
        else:
            return {
                'source':  source_name,
                'symbol':  symbol,
                'label':   label,
                'status':  '✗',
                'rows':    0,
                'start':   '-',
                'end':     '-',
                'elapsed': round(elapsed, 1),
                'error':   'empty',
            }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            'source':  source_name,
            'symbol':  symbol,
            'label':   label,
            'status':  '✗',
            'rows':    0,
            'start':   '-',
            'end':     '-',
            'elapsed': round(elapsed, 1),
            'error':   str(e)[:60],
        }


def check_installed(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def print_separator(char='-', width=90):
    print(char * width)


def main():
    print()
    print_separator('=')
    print(' 股票数据源基准测试  /  Data Source Benchmark')
    print(f' 测试时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f' 测试周期: {PERIOD}  周期: {INTERVAL}')
    print_separator('=')

    # ── 依赖检查 ──
    deps = {
        'yfinance': check_installed('yfinance'),
        'akshare':  check_installed('akshare'),
        'baostock': check_installed('baostock'),
    }
    print('\n依赖包检查:')
    for pkg, ok in deps.items():
        icon = '✓' if ok else '✗  (pip install ' + pkg + ')'
        print(f'  {icon}  {pkg}')
    print()

    # ── 运行测试 ──
    all_results = []

    for source_name, cfg in SOURCES.items():
        fn = cfg['fn']
        applicable_markets = cfg['markets']

        for symbol, label, market in TEST_CASES:
            if market not in applicable_markets:
                continue

            print(f'  [{source_name:12s}]  {symbol:10s}  {label}...', end=' ', flush=True)
            result = run_test(source_name, fn, symbol, label)
            all_results.append(result)

            if result['status'] == '✓':
                print(f"✓  {result['rows']} 行  {result['start']} ~ {result['end']}  ({result['elapsed']}s)")
            else:
                print(f"✗  {result['error']}  ({result['elapsed']}s)")

    # ── 汇总表 ──
    print()
    print_separator('=')
    print(' 汇总  /  Summary')
    print_separator('=')

    df_res = pd.DataFrame(all_results)

    # 每个数据源的成功率
    print(f'\n{"数据源":<14} {"成功":<6} {"失败":<6} {"成功率":<8} {"平均耗时":<10} {"平均行数"}')
    print_separator('-', 60)
    for source_name in SOURCES:
        sub = df_res[df_res['source'] == source_name]
        if sub.empty:
            continue
        ok    = (sub['status'] == '✓').sum()
        total = len(sub)
        avg_t = sub['elapsed'].mean()
        avg_r = sub[sub['status'] == '✓']['rows'].mean() if ok > 0 else 0
        rate  = f'{ok / total * 100:.0f}%' if total > 0 else '-'
        print(f'{source_name:<14} {ok:<6} {total - ok:<6} {rate:<8} {avg_t:.1f}s       {avg_r:.0f}')

    # 每个市场的推荐数据源
    print()
    print_separator('-', 60)
    print('推荐配置 / Recommended config:')
    print_separator('-', 60)
    market_groups = {
        'US':    ('美股', ['yfinance', 'akshare_us']),
        'HK':    ('港股', ['akshare_hk', 'yfinance']),
        'CN_SH': ('A股沪', ['akshare_cn', 'baostock']),
        'CN_SZ': ('A股深', ['akshare_cn', 'baostock']),
    }
    for market, (market_label, candidates) in market_groups.items():
        market_symbols = {s for s, _, m in TEST_CASES if m == market}
        scores = {}
        for src in candidates:
            sub = df_res[(df_res['source'] == src) & (df_res['symbol'].isin(market_symbols))]
            if sub.empty:
                continue
            ok    = (sub['status'] == '✓').sum()
            total = len(sub)
            avg_t = sub[sub['status'] == '✓']['elapsed'].mean() if ok > 0 else 999
            scores[src] = (ok / total if total > 0 else 0, -avg_t)
        if scores:
            best = max(scores, key=lambda x: scores[x])
            print(f'  {market_label:6s}: {best}  (成功率={scores[best][0]*100:.0f}%, 速度={-scores[best][1]:.1f}s)')
        else:
            print(f'  {market_label:6s}: 无可用数据源')

    print()
    print_separator('=')
    print(' 测试完成 / Test complete')
    print_separator('=')
    print()


if __name__ == '__main__':
    main()
