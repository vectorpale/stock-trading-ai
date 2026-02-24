"""
快速向量化回测（VectorBT）
Fast Vectorized Backtesting via VectorBT

适用场景：快速筛选大批量股票，找出历史表现较好的标的。
与主回测引擎的区别：
  - 主引擎 (engine.py)  ：事件驱动，支持富途精确费率，适合单标的深度分析
  - 快速引擎 (此文件)   ：向量化并行，适合批量初筛，速度快 10-100x

策略：MA 金叉/死叉（可扩展）

安装依赖：pip install vectorbt
注意：若未安装 vectorbt，调用时会给出提示，不影响主程序运行。
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def screen_ma_crossover(
    symbols_data: Dict[str, pd.DataFrame],
    fast_ma: int = 20,
    slow_ma: int = 50,
    fee_rate: float = 0.001,
    sort_by: str = 'sharpe_ratio',
    top_n: int = None,
) -> pd.DataFrame:
    """
    用 MA 金叉/死叉策略对多只股票快速回测，返回绩效排行。

    Args:
        symbols_data : {symbol: OHLCV DataFrame} 字典
        fast_ma      : 快线周期（默认 20 日）
        slow_ma      : 慢线周期（默认 50 日）
        fee_rate     : 单边费率（默认 0.1%）
        sort_by      : 排序指标（'sharpe_ratio' / 'total_return' / 'max_drawdown'）
        top_n        : 只返回前 N 名，None 返回全部

    Returns:
        DataFrame，每行一只股票，包含关键绩效指标，按 sort_by 降序排列
    """
    try:
        import vectorbt as vbt
    except ImportError:
        logger.warning(
            "[fast_backtest] vectorbt 未安装。\n"
            "安装方法: pip install vectorbt\n"
            "功能降级：返回空 DataFrame"
        )
        return pd.DataFrame()

    if not symbols_data:
        return pd.DataFrame()

    results = []

    for symbol, df in symbols_data.items():
        try:
            if df is None or len(df) < slow_ma + 10:
                continue

            close = df['Close'].dropna()
            if len(close) < slow_ma + 10:
                continue

            fast = close.rolling(fast_ma).mean()
            slow = close.rolling(slow_ma).mean()

            # 金叉买入，死叉卖出
            entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
            exits   = (fast < slow) & (fast.shift(1) >= slow.shift(1))

            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                freq='1D',
                init_cash=100_000,
                fees=fee_rate,
            )

            stats = pf.stats()

            total_return = stats.get('Total Return [%]', 0)
            sharpe       = stats.get('Sharpe Ratio', 0)
            max_dd       = stats.get('Max Drawdown [%]', 0)
            win_rate     = stats.get('Win Rate [%]', 0)
            n_trades     = stats.get('Total Trades', 0)
            # Calmar = annualized return / max drawdown
            calmar = (total_return / abs(max_dd)) if max_dd and max_dd != 0 else 0

            results.append({
                'symbol':       symbol,
                '总收益率(%)':  round(float(total_return), 2),
                '夏普比率':     round(float(sharpe) if sharpe and not np.isnan(float(sharpe)) else 0, 3),
                '最大回撤(%)':  round(float(max_dd), 2),
                '胜率(%)':      round(float(win_rate) if win_rate else 0, 1),
                '交易次数':     int(n_trades) if n_trades else 0,
                'Calmar比率':   round(calmar, 3),
                'sharpe_ratio': float(sharpe) if sharpe and not np.isnan(float(sharpe)) else 0,
                'total_return': float(total_return),
                'max_drawdown': float(max_dd),
            })

        except Exception as e:
            logger.debug(f"[fast_backtest] {symbol} 回测失败: {e}")
            continue

    if not results:
        return pd.DataFrame()

    df_result = pd.DataFrame(results)

    # 排序
    sort_col = {
        'sharpe_ratio': 'sharpe_ratio',
        'total_return': 'total_return',
        'max_drawdown': 'max_drawdown',
    }.get(sort_by, 'sharpe_ratio')

    ascending = sort_col == 'max_drawdown'  # 最大回撤越小越好
    df_result = df_result.sort_values(sort_col, ascending=ascending)

    # 隐藏内部排序列
    display_cols = ['symbol', '总收益率(%)', '夏普比率', '最大回撤(%)', '胜率(%)', '交易次数', 'Calmar比率']
    df_result = df_result[display_cols].reset_index(drop=True)

    if top_n:
        df_result = df_result.head(top_n)

    return df_result
