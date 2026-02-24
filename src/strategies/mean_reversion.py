"""
均值回归策略
Mean Reversion Strategy

基于布林带、RSI、价格偏离度的均值回归策略
适合震荡市场，在超买超卖区间操作
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

from .base import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略

    核心逻辑：
    1. 布林带下轨附近买入，上轨附近卖出
    2. RSI 超卖区间（<30）确认买入，超买（>70）确认卖出
    3. 价格偏离均线程度
    4. 历史波动率过滤（避开剧烈波动期）
    5. KDJ 随机指标辅助确认
    """

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if df is None or len(df) < 30:
            return self._empty_signal(symbol, "数据不足30日")

        g = self._get_latest

        price = g(df, 'Close', 0) or 0
        if price <= 0:
            return self._empty_signal(symbol, "价格异常")

        score = 0.0
        reasons = []

        # ─── 1. 布林带位置 ───
        bb_pct = g(df, 'bb_pct', 0.5) or 0.5   # 0=下轨, 1=上轨
        bb_width = g(df, 'bb_width', 0.1) or 0.1

        # 布林带收窄时，均值回归信号更可靠
        width_factor = 1.0 + min(bb_width * 2, 0.5)

        if bb_pct <= 0.05:
            score += 40 * width_factor; reasons.append("触及布林带下轨（超卖）")
        elif bb_pct <= 0.15:
            score += 25; reasons.append("接近布林带下轨")
        elif bb_pct >= 0.95:
            score -= 40 * width_factor; reasons.append("触及布林带上轨（超买）")
        elif bb_pct >= 0.85:
            score -= 25; reasons.append("接近布林带上轨")
        elif 0.45 <= bb_pct <= 0.55:
            score += 0  # 中性区域，无信号

        # ─── 2. RSI 超买超卖 ───
        rsi = g(df, 'rsi_14', 50) or 50
        rsi_6 = g(df, 'rsi_6', 50) or 50

        if rsi < 25:
            score += 30; reasons.append(f"RSI 极度超卖（{rsi:.0f}）")
        elif rsi < 35:
            score += 20; reasons.append(f"RSI 超卖（{rsi:.0f}）")
        elif rsi > 75:
            score -= 30; reasons.append(f"RSI 极度超买（{rsi:.0f}）")
        elif rsi > 65:
            score -= 20; reasons.append(f"RSI 超买（{rsi:.0f}）")

        # 短周期RSI快速反转信号
        if rsi_6 < 20 and rsi > 30:
            score += 15; reasons.append("短期RSI反转信号")

        # ─── 3. 价格偏离均线 ───
        close_vs_sma20 = g(df, 'close_vs_sma20', 0) or 0
        close_vs_sma50 = g(df, 'close_vs_sma50', 0) or 0

        if close_vs_sma20 < -0.08:
            score += 20; reasons.append(f"价格低于20日均线{abs(close_vs_sma20)*100:.1f}%")
        elif close_vs_sma20 < -0.04:
            score += 10
        elif close_vs_sma20 > 0.08:
            score -= 20; reasons.append(f"价格高于20日均线{close_vs_sma20*100:.1f}%")

        # ─── 4. KDJ 随机指标 ───
        stoch_k = g(df, 'stoch_k', 50) or 50
        stoch_d = g(df, 'stoch_d', 50) or 50

        if stoch_k < 20 and stoch_d < 20:
            score += 15; reasons.append("KDJ 超卖区域")
        elif stoch_k > 80 and stoch_d > 80:
            score -= 15; reasons.append("KDJ 超买区域")

        # KDJ 金叉/死叉
        prev_k = df['stoch_k'].iloc[-2] if len(df) >= 2 else stoch_k
        prev_d = df['stoch_d'].iloc[-2] if len(df) >= 2 else stoch_d
        if stoch_k > stoch_d and prev_k <= prev_d and stoch_k < 40:
            score += 20; reasons.append("KDJ 低位金叉")
        elif stoch_k < stoch_d and prev_k >= prev_d and stoch_k > 60:
            score -= 20; reasons.append("KDJ 高位死叉")

        # ─── 5. 波动率过滤（避免追高期间均值回归）───
        volatility_20 = g(df, 'volatility_20', 0.3) or 0.3
        if volatility_20 > 0.6:
            score *= 0.6  # 高波动期间，均值回归信号衰减
            reasons.append("波动率偏高，信号衰减")

        # ─── 6. 趋势过滤：下跌趋势中不做均值回归多头 ───
        sma_50 = g(df, 'sma_50', price) or price
        sma_200 = g(df, 'sma_200', price) or price
        if price < sma_200 * 0.95 and score > 0:
            score *= 0.5
            reasons.append("价格低于200日均线（趋势偏弱）")

        score = np.clip(score, -100, 100)

        atr = g(df, 'atr_14', price * 0.02) or price * 0.02

        return self._build_signal(
            symbol=symbol,
            score=score,
            price=price,
            atr=atr,
            reasons=reasons
        )

    def _build_signal(
        self,
        symbol: str,
        score: float,
        price: float,
        atr: float,
        reasons: list
    ) -> Dict[str, Any]:
        if score >= 45:
            action = 'BUY'
            stop_loss = price - 1.5 * atr
            take_profit = price + 2.5 * atr
            position_pct = min(0.12, 0.04 + (score - 45) / 550)
        elif score <= -45:
            action = 'SELL'
            stop_loss = price + 1.5 * atr
            take_profit = price - 2.5 * atr
            position_pct = 0
        elif abs(score) >= 25:
            action = 'WATCH'
            stop_loss = price - 1.5 * atr
            take_profit = price + 2.0 * atr
            position_pct = 0
        else:
            action = 'HOLD'
            stop_loss = price - 2.0 * atr
            take_profit = price + 2.0 * atr
            position_pct = 0

        return {
            'symbol': symbol,
            'strategy': 'mean_reversion',
            'score': round(score, 1),
            'action': action,
            'confidence': min(abs(score), 100),
            'price': round(price, 4),
            'stop_loss': round(max(stop_loss, 0), 4),
            'take_profit': round(take_profit, 4),
            'position_pct': position_pct,
            'reason': '；'.join(reasons) if reasons else '信号中性',
        }

    def _empty_signal(self, symbol: str, reason: str) -> Dict[str, Any]:
        return {
            'symbol': symbol,
            'strategy': 'mean_reversion',
            'score': 0,
            'action': 'HOLD',
            'confidence': 0,
            'price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'position_pct': 0,
            'reason': reason,
        }
