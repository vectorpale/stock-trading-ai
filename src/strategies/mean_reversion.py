"""
均值回归策略
Mean Reversion Strategy

基于布林带、RSI、价格偏离度的均值回归策略
适合震荡市场，在超买超卖区间操作

优化项（v2）:
- Z-score 标准化偏离度（比固定百分比更自适应）
- Hurst 指数过滤（确认标的具有均值回归特性）
- ADF 平稳性检验（可选，需 statsmodels）
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

    @staticmethod
    def _hurst_exponent(close_series: pd.Series, max_lag: int = 20) -> float:
        """
        Hurst 指数（无需外部库，仅用 numpy）
        H < 0.45 → 均值回归    H ≈ 0.5 → 随机游走    H > 0.55 → 趋势
        """
        ts = close_series.dropna().values
        if len(ts) < max_lag + 5:
            return 0.5
        lags = range(2, max_lag)
        tau = [float(np.std(ts[lag:] - ts[:-lag])) for lag in lags]
        tau = [t if t > 0 else 1e-8 for t in tau]
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(reg[0])

    @staticmethod
    def _zscore(close_series: pd.Series, window: int = 20) -> float:
        """价格相对滚动均值的 Z-score（标准差倍数）"""
        if len(close_series) < window:
            return 0.0
        recent = close_series.iloc[-window:]
        mean = recent.mean()
        std = recent.std()
        if std == 0:
            return 0.0
        return float((close_series.iloc[-1] - mean) / std)

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if df is None or len(df) < 30:
            return self._empty_signal(symbol, "数据不足30日")

        g = self._get_latest

        price = g(df, 'Close', 0) or 0
        if price <= 0:
            return self._empty_signal(symbol, "价格异常")

        score = 0.0
        reasons = []

        # ─── 0. Hurst 指数：确认均值回归特性 ───
        hurst = self._hurst_exponent(df['Close'], max_lag=min(20, len(df) // 3))
        hurst_multiplier = 1.0
        if hurst < 0.40:
            hurst_multiplier = 1.3   # 强均值回归特性，加强信号
            reasons.append(f"Hurst={hurst:.2f}（强均值回归）")
        elif hurst < 0.48:
            hurst_multiplier = 1.1
        elif hurst > 0.60:
            hurst_multiplier = 0.5   # 趋势型股票，均值回归策略不适用
            reasons.append(f"Hurst={hurst:.2f}（偏趋势型，信号衰减）")

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

        # ─── 3. Z-score 标准化偏离（替代固定百分比，更自适应）───
        zscore = self._zscore(df['Close'], window=20)

        if zscore < -2.0:
            score += 30; reasons.append(f"Z-score={zscore:.2f}（极度低估）")
        elif zscore < -1.5:
            score += 20; reasons.append(f"Z-score={zscore:.2f}（明显低估）")
        elif zscore < -1.0:
            score += 10
        elif zscore > 2.0:
            score -= 30; reasons.append(f"Z-score={zscore:.2f}（极度高估）")
        elif zscore > 1.5:
            score -= 20; reasons.append(f"Z-score={zscore:.2f}（明显高估）")
        elif zscore > 1.0:
            score -= 10

        # ADF 平稳性检验（可选，statsmodels）
        try:
            from statsmodels.tsa.stattools import adfuller
            if len(df) >= 30:
                adf_result = adfuller(df['Close'].iloc[-60:].dropna(), autolag='AIC')
                adf_pvalue = adf_result[1]
                if adf_pvalue < 0.05:
                    hurst_multiplier = min(hurst_multiplier * 1.15, 1.5)
                    reasons.append(f"ADF平稳(p={adf_pvalue:.3f})")
                elif adf_pvalue > 0.5:
                    hurst_multiplier = max(hurst_multiplier * 0.85, 0.4)
        except ImportError:
            pass

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

        # ─── 应用 Hurst 指数调整因子 ───
        score *= hurst_multiplier

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
