"""
多因子策略
Multi-Factor Strategy

结合动量、价值、质量、成长因子的综合评分模型
参考学术界和业界广泛使用的因子投资框架
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from .base import BaseStrategy

logger = logging.getLogger(__name__)


class MultiFactorStrategy(BaseStrategy):
    """
    多因子模型策略

    五大因子评分：
    1. 动量因子（Momentum Factor）- 价格趋势
    2. 趋势强度因子（Trend Factor）- ADX + 均线
    3. 波动率因子（Volatility Factor）- 低波动溢价
    4. 成交量因子（Volume Factor）- 量价关系
    5. 技术因子（Technical Factor）- 综合技术形态
    """

    # 五大因子权重（总和=1）
    FACTOR_WEIGHTS = {
        'momentum': 0.30,
        'trend': 0.25,
        'volatility': 0.15,
        'volume': 0.15,
        'technical': 0.15,
    }

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if df is None or len(df) < 100:
            return self._empty_signal(symbol, "数据不足100日（多因子模型需要更长历史）")

        g = self._get_latest

        price = g(df, 'Close', 0) or 0
        if price <= 0:
            return self._empty_signal(symbol, "价格异常")

        # ─── 各因子评分（-100 到 +100）───
        momentum_score = self._calc_momentum_factor(df, g)
        trend_score = self._calc_trend_factor(df, g)
        volatility_score = self._calc_volatility_factor(df, g, price)
        volume_score = self._calc_volume_factor(df, g)
        technical_score = self._calc_technical_factor(df, g)

        factor_scores = {
            'momentum': momentum_score,
            'trend': trend_score,
            'volatility': volatility_score,
            'volume': volume_score,
            'technical': technical_score,
        }

        # ─── 加权综合评分 ───
        composite_score = sum(
            score * self.FACTOR_WEIGHTS[name]
            for name, score in factor_scores.items()
        )

        reasons = self._summarize_factors(factor_scores)

        atr = g(df, 'atr_14', price * 0.02) or price * 0.02

        return self._build_signal(
            symbol=symbol,
            composite_score=composite_score,
            factor_scores=factor_scores,
            price=price,
            atr=atr,
            reasons=reasons
        )

    def _calc_momentum_factor(self, df, g) -> float:
        """动量因子：多周期价格变化率"""
        score = 0
        mom_5  = g(df, 'mom_5', 0) or 0
        mom_20 = g(df, 'mom_20', 0) or 0
        mom_60 = g(df, 'mom_60', 0) or 0

        # 跨周期动量加权
        score += np.tanh(mom_5 / 0.05) * 20    # 5日动量
        score += np.tanh(mom_20 / 0.10) * 40   # 20日动量
        score += np.tanh(mom_60 / 0.20) * 40   # 60日动量

        return float(np.clip(score, -100, 100))

    def _calc_trend_factor(self, df, g) -> float:
        """趋势因子：ADX + 均线排列"""
        score = 0
        adx = g(df, 'adx', 20) or 20
        price = g(df, 'Close', 0) or 1
        sma_20 = g(df, 'sma_20', price) or price
        sma_50 = g(df, 'sma_50', price) or price
        sma_200 = g(df, 'sma_200', price) or price

        # ADX 趋势强度（>25 为强趋势）
        if adx > 40:
            adx_boost = 1.5
        elif adx > 25:
            adx_boost = 1.0
        else:
            adx_boost = 0.5  # 弱趋势，降低权重

        # 均线排列
        if price > sma_20 > sma_50 > sma_200:
            score += 60 * adx_boost  # 多头排列
        elif price > sma_50 > sma_200:
            score += 35 * adx_boost
        elif price > sma_200:
            score += 20
        elif price < sma_200:
            score -= 30
        if price < sma_50 < sma_20:
            score -= 50  # 空头排列

        return float(np.clip(score, -100, 100))

    def _calc_volatility_factor(self, df, g, price) -> float:
        """
        波动率因子：低波动股票在长期有超额收益
        同时规避过高波动率的高风险股
        """
        vol_20 = g(df, 'volatility_20', 0.3) or 0.3
        atr_14 = g(df, 'atr_14', price * 0.02) or price * 0.02
        atr_pct = atr_14 / price if price > 0 else 0.02

        score = 0
        # 低波动性（年化波动率<20%）给正分
        if vol_20 < 0.20:
            score = 60
        elif vol_20 < 0.30:
            score = 30
        elif vol_20 < 0.50:
            score = 0
        elif vol_20 < 0.70:
            score = -20
        else:
            score = -50  # 高波动性惩罚

        return float(np.clip(score, -100, 100))

    def _calc_volume_factor(self, df, g) -> float:
        """成交量因子：量价配合度"""
        score = 0
        volume_ratio = g(df, 'volume_ratio', 1) or 1
        mom_5 = g(df, 'mom_5', 0) or 0
        obv = g(df, 'obv', 0) or 0

        # OBV 趋势（近期是否在上升）
        if len(df) >= 10:
            obv_now = df['obv'].iloc[-1]
            obv_10d = df['obv'].iloc[-10]
            obv_trend = (obv_now - obv_10d) / (abs(obv_10d) + 1)

            if obv_trend > 0.05:
                score += 40  # OBV 持续流入
            elif obv_trend < -0.05:
                score -= 40  # OBV 持续流出

        # 量价配合
        if mom_5 > 0 and volume_ratio > 1.5:
            score += 30  # 放量上涨
        elif mom_5 > 0 and volume_ratio < 0.8:
            score -= 10  # 缩量上涨，弱信号
        elif mom_5 < 0 and volume_ratio > 1.5:
            score -= 30  # 放量下跌
        elif mom_5 < 0 and volume_ratio < 0.8:
            score += 10  # 缩量下跌，弱信号

        return float(np.clip(score, -100, 100))

    def _calc_technical_factor(self, df, g) -> float:
        """综合技术因子：MACD + RSI + 价格位置"""
        score = 0

        # MACD
        macd_hist = g(df, 'macd_hist', 0) or 0
        if macd_hist > 0:
            score += 25
        else:
            score -= 25

        # RSI（中性区间得分最高）
        rsi = g(df, 'rsi_14', 50) or 50
        if 50 < rsi < 65:
            score += 20  # 健康多头区间
        elif 35 < rsi <= 50:
            score += 5   # 轻微弱势但不超卖
        elif rsi <= 30:
            score -= 15  # 超卖（多因子模型中不直接买）
        elif rsi >= 75:
            score -= 25  # 超买

        # 52周高低位置（Momentum premium near highs）
        pct_from_high = g(df, 'pct_from_52w_high', -0.1) or -0.1
        if pct_from_high > -0.05:
            score += 25  # 接近52周高点，强势
        elif pct_from_high > -0.15:
            score += 10
        elif pct_from_high < -0.40:
            score -= 20  # 远离52周高点，弱势

        return float(np.clip(score, -100, 100))

    def _summarize_factors(self, factor_scores: Dict[str, float]) -> List[str]:
        """总结各因子信号"""
        reasons = []
        labels = {
            'momentum': '动量因子',
            'trend': '趋势因子',
            'volatility': '波动率因子',
            'volume': '成交量因子',
            'technical': '技术因子'
        }
        for name, score in factor_scores.items():
            if score >= 40:
                reasons.append(f"{labels[name]}强势({score:+.0f})")
            elif score <= -40:
                reasons.append(f"{labels[name]}弱势({score:+.0f})")
        return reasons

    def _build_signal(
        self,
        symbol: str,
        composite_score: float,
        factor_scores: Dict,
        price: float,
        atr: float,
        reasons: List[str]
    ) -> Dict[str, Any]:
        composite_score = float(np.clip(composite_score, -100, 100))

        if composite_score >= 40:
            action = 'BUY'
            stop_loss = price - 2.0 * atr
            take_profit = price + 3.5 * atr
            position_pct = min(0.15, 0.05 + (composite_score - 40) / 600)
        elif composite_score <= -40:
            action = 'SELL'
            stop_loss = price + 2.0 * atr
            take_profit = price - 3.0 * atr
            position_pct = 0
        elif abs(composite_score) >= 20:
            action = 'WATCH'
            stop_loss = price - 2.0 * atr
            take_profit = price + 2.5 * atr
            position_pct = 0
        else:
            action = 'HOLD'
            stop_loss = price - 2.0 * atr
            take_profit = price + 2.0 * atr
            position_pct = 0

        return {
            'symbol': symbol,
            'strategy': 'multi_factor',
            'score': round(composite_score, 1),
            'factor_scores': {k: round(v, 1) for k, v in factor_scores.items()},
            'action': action,
            'confidence': min(abs(composite_score), 100),
            'price': round(price, 4),
            'stop_loss': round(max(stop_loss, 0), 4),
            'take_profit': round(take_profit, 4),
            'position_pct': position_pct,
            'reason': '；'.join(reasons) if reasons else '各因子综合中性',
        }

    def _empty_signal(self, symbol: str, reason: str) -> Dict[str, Any]:
        return {
            'symbol': symbol,
            'strategy': 'multi_factor',
            'score': 0,
            'factor_scores': {},
            'action': 'HOLD',
            'confidence': 0,
            'price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'position_pct': 0,
            'reason': reason,
        }
