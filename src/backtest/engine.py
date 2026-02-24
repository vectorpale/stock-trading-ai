"""
回测引擎
Backtesting Engine

支持单策略和多策略融合回测
自动扣除富途平台费率
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from .fees import FutuFeeCalculator
from ..indicators.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    事件驱动回测引擎

    特性：
    - 精确扣除富途费率（美股/港股）
    - 支持止损/止盈自动执行
    - 计算多维度绩效指标
    - 生成详细的交易记录
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        market: str = 'us',
        fee_config: Dict = None,
        max_position_pct: float = 0.15,
        stop_loss_pct: float = 0.07,
        take_profit_pct: float = 0.20,
    ):
        self.initial_capital = initial_capital
        self.market = market
        self.fee_calc = FutuFeeCalculator(fee_config)
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        strategy_fn: Callable,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, Any]:
        """
        执行单标的回测

        Args:
            symbol: 股票代码
            df: 原始 OHLCV 数据
            strategy_fn: 策略函数，接收带指标的 df 和 symbol，返回信号字典
            start_date: 回测开始日期（YYYY-MM-DD）
            end_date: 回测结束日期（YYYY-MM-DD）

        Returns:
            回测结果字典，包含绩效指标和交易记录
        """
        # 计算技术指标
        data = TechnicalIndicators.compute_all(df)

        # 日期过滤
        if start_date:
            data = data[data.index >= pd.Timestamp(start_date)]
        if end_date:
            data = data[data.index <= pd.Timestamp(end_date)]

        if len(data) < 60:
            return {'error': f'回测区间数据不足，只有 {len(data)} 条记录'}

        # ─── 初始化回测状态 ───
        capital = self.initial_capital
        position = 0          # 当前持股数
        entry_price = 0.0     # 买入价格
        entry_date = None

        equity_curve = []     # 净值曲线
        trades = []           # 交易记录

        # ─── 逐日回测 ───
        for i in range(60, len(data)):
            current_slice = data.iloc[:i+1]
            row = data.iloc[i]
            date = data.index[i]
            price = float(row['Close'])

            if price <= 0:
                equity_curve.append({'date': date, 'equity': capital + position * price})
                continue

            # 检查止损止盈（有持仓时）
            if position > 0 and entry_price > 0:
                pnl_pct = (price - entry_price) / entry_price

                if pnl_pct <= -self.stop_loss_pct:
                    # 触发止损
                    capital, trade = self._close_position(
                        capital, position, price, entry_price, date, symbol, "止损"
                    )
                    trades.append(trade)
                    position = 0
                    entry_price = 0.0

                elif pnl_pct >= self.take_profit_pct:
                    # 触发止盈
                    capital, trade = self._close_position(
                        capital, position, price, entry_price, date, symbol, "止盈"
                    )
                    trades.append(trade)
                    position = 0
                    entry_price = 0.0

            # 生成策略信号
            try:
                signal = strategy_fn(current_slice, symbol)
            except Exception as e:
                logger.debug(f"信号生成异常 {symbol}@{date}: {e}")
                equity_curve.append({'date': date, 'equity': capital + position * price})
                continue

            action = signal.get('action', 'HOLD')

            # ─── 执行信号 ───
            if action == 'BUY' and position == 0:
                # 计算可买股数
                invest_pct = min(signal.get('position_pct', 0.10), self.max_position_pct)
                invest_amount = capital * invest_pct
                shares = int(invest_amount / price)

                if shares >= 1:
                    # 计算买入费用
                    fees = self._calc_fees(shares, price, is_sell=False)
                    total_cost = shares * price + fees

                    if total_cost <= capital:
                        capital -= total_cost
                        position = shares
                        entry_price = price
                        entry_date = date

                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'fees': fees,
                            'total_cost': total_cost,
                            'capital_after': capital,
                            'reason': signal.get('reason', ''),
                        })

            elif action == 'SELL' and position > 0:
                capital, trade = self._close_position(
                    capital, position, price, entry_price, date, symbol,
                    signal.get('reason', '策略信号')
                )
                trades.append(trade)
                position = 0
                entry_price = 0.0

            # 记录净值
            equity = capital + position * price
            equity_curve.append({'date': date, 'equity': equity})

        # ─── 平仓（回测结束时）───
        if position > 0:
            final_price = float(data['Close'].iloc[-1])
            capital, trade = self._close_position(
                capital, position, final_price, entry_price,
                data.index[-1], symbol, "回测结束平仓"
            )
            trades.append(trade)

        # ─── 计算绩效指标 ───
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        metrics = self._calc_metrics(equity_df, trades, data)

        return {
            'symbol': symbol,
            'metrics': metrics,
            'equity_curve': equity_df,
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
        }

    def _close_position(
        self,
        capital: float,
        position: int,
        price: float,
        entry_price: float,
        date,
        symbol: str,
        reason: str
    ):
        """执行平仓操作，返回新资金量和交易记录"""
        fees = self._calc_fees(position, price, is_sell=True)
        proceeds = position * price - fees
        pnl = proceeds - position * entry_price
        pnl_pct = pnl / (position * entry_price) if entry_price > 0 else 0

        capital += proceeds

        trade = {
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'shares': position,
            'price': price,
            'entry_price': entry_price,
            'fees': fees,
            'proceeds': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'capital_after': capital,
            'reason': reason,
        }
        return capital, trade

    def _calc_fees(self, shares: int, price: float, is_sell: bool) -> float:
        """计算交易费用"""
        if self.market == 'us':
            return self.fee_calc.calc_us_fees(shares, price, is_sell)['total']
        else:
            return self.fee_calc.calc_hk_fees(shares, price, is_sell)['total']

    def _calc_metrics(
        self,
        equity_df: pd.DataFrame,
        trades: List[Dict],
        price_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """计算全面的绩效指标"""
        if equity_df.empty:
            return {}

        equity = equity_df['equity']
        total_return = (equity.iloc[-1] / self.initial_capital - 1) * 100

        # 年化收益率
        n_days = (equity.index[-1] - equity.index[0]).days
        n_years = max(n_days / 365, 0.01)
        annual_return = ((equity.iloc[-1] / self.initial_capital) ** (1 / n_years) - 1) * 100

        # 最大回撤
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # 夏普比率（假设无风险利率 4%）
        daily_returns = equity.pct_change().dropna()
        risk_free_daily = 0.04 / 252
        excess_returns = daily_returns - risk_free_daily
        sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                  if excess_returns.std() > 0 else 0)

        # Sortino 比率
        downside_returns = excess_returns[excess_returns < 0]
        sortino = (excess_returns.mean() / downside_returns.std() * np.sqrt(252)
                   if len(downside_returns) > 0 and downside_returns.std() > 0 else 0)

        # 交易统计
        sell_trades = [t for t in trades if t.get('action') == 'SELL' and 'pnl' in t]
        n_trades = len(sell_trades)
        win_trades = [t for t in sell_trades if t['pnl'] > 0]
        win_rate = len(win_trades) / n_trades * 100 if n_trades > 0 else 0

        avg_win = np.mean([t['pnl_pct'] for t in win_trades]) * 100 if win_trades else 0
        lose_trades = [t for t in sell_trades if t['pnl'] <= 0]
        avg_loss = np.mean([t['pnl_pct'] for t in lose_trades]) * 100 if lose_trades else 0

        profit_factor = (
            sum(t['pnl'] for t in win_trades) / abs(sum(t['pnl'] for t in lose_trades))
            if lose_trades and sum(t['pnl'] for t in lose_trades) != 0 else float('inf')
        )

        # 基准对比（买入持有）
        bh_return = (float(price_df['Close'].iloc[-1]) / float(price_df['Close'].iloc[60]) - 1) * 100
        bh_annual = ((1 + bh_return/100) ** (1/n_years) - 1) * 100

        metrics = {
            '总收益率(%)': round(total_return, 2),
            '年化收益率(%)': round(annual_return, 2),
            '最大回撤(%)': round(max_drawdown, 2),
            '夏普比率': round(sharpe, 3),
            'Sortino比率': round(sortino, 3),
            '交易次数': n_trades,
            '胜率(%)': round(win_rate, 1),
            '平均盈利(%)': round(avg_win, 2),
            '平均亏损(%)': round(avg_loss, 2),
            '盈亏比': round(profit_factor, 2) if profit_factor != float('inf') else '∞',
            '基准买持收益率(%)': round(bh_return, 2),
            '基准年化(%)': round(bh_annual, 2),
            '超额收益(%)': round(annual_return - bh_annual, 2),
            '回测天数': n_days,
        }

        # ── empyrical 专业指标（可选，需 pip install empyrical）──
        try:
            import empyrical as ep
            returns = daily_returns  # 已在上面计算

            calmar = ep.calmar_ratio(returns)
            omega  = ep.omega_ratio(returns)
            var95  = ep.value_at_risk(returns, cutoff=0.05)
            cvar95 = ep.conditional_value_at_risk(returns, cutoff=0.05)
            stability = ep.stability_of_timeseries(returns)
            tail_ratio = ep.tail_ratio(returns)

            def _fmt(v, pct=False, decimals=3):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return 'N/A'
                return round(v * 100 if pct else v, decimals)

            metrics.update({
                'Calmar比率':    _fmt(calmar),
                'Omega比率':     _fmt(omega),
                'VaR(95%, 日%)': _fmt(var95, pct=True, decimals=2),
                'CVaR(95%, 日%)':_fmt(cvar95, pct=True, decimals=2),
                '收益稳定性R²':  _fmt(stability),
                '尾部比率':      _fmt(tail_ratio),
            })
        except ImportError:
            pass  # empyrical 未安装，跳过

        return metrics
