#!/usr/bin/env python3
"""
命令行信号生成器
CLI Signal Runner

用法示例：
  python run_signals.py                    # 使用配置文件中的股票池
  python run_signals.py --symbols AAPL MSFT NVDA
  python run_signals.py --no-ai            # 禁用 AI 分析（更快）
  python run_signals.py --capital 50000    # 指定资金量
  python run_signals.py --backtest AAPL   # 对指定股票运行回测
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 设置路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import load_config, format_signal_output
from src.signals.generator import SignalGenerator
from src.backtest.engine import BacktestEngine
from src.backtest.fees import FutuFeeCalculator
from src.data.fetcher import DataFetcher
from src.indicators.technical import TechnicalIndicators
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.multi_factor import MultiFactorStrategy


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )


def run_signals(args):
    """生成今日交易信号"""
    config = load_config()

    # 确定股票池
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        us_stocks = config.get('market', {}).get('us_stocks', [])
        hk_stocks = config.get('market', {}).get('hk_stocks', [])
        symbols = us_stocks + hk_stocks

    # 是否启用 AI
    ai_enabled = not args.no_ai
    if ai_enabled and not os.environ.get('ANTHROPIC_API_KEY'):
        print("[警告] 未设置 ANTHROPIC_API_KEY，AI 分析已自动禁用")
        print("       请在 .env 文件中配置 ANTHROPIC_API_KEY=your_key")
        ai_enabled = False

    # 更新 AI 配置
    config['ai']['enabled'] = ai_enabled

    print(f"\n{'='*60}")
    print(f"  股票交易策略助手 - 信号生成")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  股票: {', '.join(symbols)}")
    print(f"  AI分析: {'启用' if ai_enabled else '禁用'}")
    print(f"  资金: ${args.capital:,.0f}")
    print(f"{'='*60}\n")

    generator = SignalGenerator(config)
    signals = generator.generate_for_watchlist(
        symbols,
        include_ai=ai_enabled,
        capital=args.capital
    )

    # 打印格式化信号
    output = format_signal_output(signals)
    print(output)

    # 保存到文件
    if args.save:
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        filename = report_dir / f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\n报告已保存: {filename}")

    return signals


def run_backtest(args):
    """运行回测"""
    config = load_config()
    fetcher = DataFetcher()

    symbol = args.backtest.upper()
    market = 'hk' if symbol.endswith('.HK') else 'us'

    print(f"\n{'='*60}")
    print(f"  策略回测 - {symbol}")
    print(f"{'='*60}\n")

    df = fetcher.fetch_ohlcv(symbol, period="2y", interval="1d")
    if df is None or len(df) < 100:
        print(f"错误：{symbol} 数据不足，无法回测")
        return

    # 三策略分别回测
    strategies = {
        '动量策略': MomentumStrategy(),
        '均值回归策略': MeanReversionStrategy(),
        '多因子策略': MultiFactorStrategy(),
    }

    engine = BacktestEngine(
        initial_capital=args.capital,
        market=market,
        fee_config=config.get('fees', {}),
        stop_loss_pct=config.get('risk', {}).get('stop_loss_pct', 0.07),
        take_profit_pct=config.get('risk', {}).get('take_profit_pct', 0.20),
    )

    for name, strategy in strategies.items():
        print(f"\n--- {name} ---")
        result = engine.run(
            symbol=symbol,
            df=df.copy(),
            strategy_fn=lambda df, sym, s=strategy: s.generate_signal(
                TechnicalIndicators.compute_all(df), sym
            ),
        )

        if 'error' in result:
            print(f"  回测失败: {result['error']}")
            continue

        metrics = result['metrics']
        print(f"  年化收益率: {metrics.get('年化收益率(%)', 0):.2f}%"
              f"  (目标: 30%)")
        print(f"  最大回撤:   {metrics.get('最大回撤(%)', 0):.2f}%")
        print(f"  夏普比率:   {metrics.get('夏普比率', 0):.3f}")
        print(f"  交易次数:   {metrics.get('交易次数', 0)} 次")
        print(f"  胜率:       {metrics.get('胜率(%)', 0):.1f}%")
        print(f"  超额收益:   {metrics.get('超额收益(%)', 0):+.2f}%")
        print(f"  基准(买持): {metrics.get('基准年化(%)', 0):.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='股票交易策略助手 - 信号生成与回测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python run_signals.py
  python run_signals.py --symbols AAPL MSFT TSLA
  python run_signals.py --symbols 0700.HK 9988.HK --capital 500000
  python run_signals.py --no-ai --save
  python run_signals.py --backtest NVDA
        """
    )
    parser.add_argument(
        '--symbols', nargs='+',
        help='指定股票代码（空格分隔），默认使用配置文件中的股票池'
    )
    parser.add_argument(
        '--capital', type=float, default=100000.0,
        help='可用资金量（默认 100000）'
    )
    parser.add_argument(
        '--no-ai', action='store_true',
        help='禁用 AI 分析（更快，不需要 API Key）'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='将信号报告保存到 reports/ 目录'
    )
    parser.add_argument(
        '--backtest', type=str,
        help='对指定股票运行历史回测（如 --backtest AAPL）'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='显示详细日志'
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # 加载 .env（如果存在）
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if args.backtest:
        run_backtest(args)
    else:
        run_signals(args)


if __name__ == '__main__':
    main()
