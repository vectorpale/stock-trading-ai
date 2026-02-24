"""
动量策略
Momentum Strategy

基于价格趋势、MACD、RSI 的动量追踪策略
适合趋势市场，捕捉持续上涨的强势股
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

from .base import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    动量策略

    核心逻辑：
    1. 多周期动量（5日、20日、60日）综合评分
    2. MACD 金叉/死叉信号
    3. 均线排列（多头排列看涨）
    4. 成交量确认（量价配合）
    5. RSI 动量确认（避开超买区域买入）
    """

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if df is None or len(df) < 60:
            return self._empty_signal(symbol, "数据不足60日")

        g = self._get_latest  # 简写

        # ─── 当前价格 ───
        price = g(df, 'Close', 0)
        if price <= 0:
            return self._empty_signal(symbol, "价格异常")

        score = 0.0
        reasons = []

        # ─── 1. 多周期动量评分 ───
        mom_5  = g(df, 'mom_5', 0)   or 0
        mom_20 = g(df, 'mom_20', 0)  or 0
        mom_60 = g(df, 'mom_60', 0)  or 0

        # 短期动量（权重 20%）
        if mom_5 > 0.03:
            score += 20; reasons.append("短期价格上涨")
        elif mom_5 < -0.03:
            score -= 20; reasons.append("短期价格下跌")

        # 中期动量（权重 30%）
        if mom_20 > 0.08:
            score += 30; reasons.append("中期强劲上涨")
        elif mom_20 > 0.03:
            score += 15
        elif mom_20 < -0.08:
            score -= 30; reasons.append("中期明显下跌")
        elif mom_20 < -0.03:
            score -= 15

        # 长期动量（权重 20%）
        if mom_60 > 0.15:
            score += 20; reasons.append("长期趋势向上")
        elif mom_60 > 0.05:
            score += 10
        elif mom_60 < -0.15:
            score -= 20; reasons.append("长期趋势向下")

        # ─── 2. MACD 信号 ───
        macd = g(df, 'macd', 0) or 0
        macd_signal = g(df, 'macd_signal', 0) or 0
        macd_hist = g(df, 'macd_hist', 0) or 0
        prev_macd_hist = df['macd_hist'].iloc[-2] if len(df) >= 2 else 0

        if macd_hist > 0 and prev_macd_hist <= 0:
            score += 25; reasons.append("MACD 金叉（买入信号）")
        elif macd_hist < 0 and prev_macd_hist >= 0:
            score -= 25; reasons.append("MACD 死叉（卖出信号）")
        elif macd > macd_signal and macd_hist > 0:
            score += 10; reasons.append("MACD 多头持续")
        elif macd < macd_signal and macd_hist < 0:
            score -= 10

        # ─── 3. 均线排列 ───
        sma_20 = g(df, 'sma_20', 0) or 0
        sma_50 = g(df, 'sma_50', 0) or 0
        sma_200 = g(df, 'sma_200', 0) or 0

        if sma_200 > 0:
            # 多头排列：价格 > 20MA > 50MA > 200MA
            if price > sma_20 > sma_50 > sma_200:
                score += 20; reasons.append("均线多头排列")
            elif price > sma_200:
                score += 10; reasons.append("价格在200日均线上方")
            elif price < sma_200:
                score -= 15; reasons.append("价格在200日均线下方（弱势）")

        # ─── 4. 成交量确认 ───
        volume_ratio = g(df, 'volume_ratio', 1) or 1

        if mom_20 > 0 and volume_ratio > 1.5:
            score += 10; reasons.append("放量上涨（成交量确认）")
        elif mom_20 > 0 and volume_ratio < 0.7:
            score -= 5  # 缩量上涨，信号较弱

        # ─── 5. RSI 过滤 ───
        rsi = g(df, 'rsi_14', 50) or 50
        if rsi > 75:
            score *= 0.7  # 超买，降低信号强度
            reasons.append("RSI 超买（>75），信号衰减")
        elif rsi < 30:
            score *= 0.7  # 超卖但动量策略不抄底
            reasons.append("RSI 超卖（<30）")
        elif 50 < rsi < 70:
            score *= 1.1  # 动量健康区间

        score = np.clip(score, -100, 100)

        # ─── 生成最终信号 ───
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
        """根据分数构建完整信号"""
        if score >= 50:
            action = 'BUY'
            stop_loss = price - 2.0 * atr
            take_profit = price + 3.0 * atr
            position_pct = min(0.15, 0.05 + (score - 50) / 500)
        elif score <= -50:
            action = 'SELL'
            stop_loss = price + 2.0 * atr
            take_profit = price - 3.0 * atr
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
            'strategy': 'momentum',
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
            'strategy': 'momentum',
            'score': 0,
            'action': 'HOLD',
            'confidence': 0,
            'price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'position_pct': 0,
            'reason': reason,
        }
