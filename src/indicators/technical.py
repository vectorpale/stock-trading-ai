"""
技术指标计算模块
Technical Indicators Calculator

使用 pandas-ta 库计算多种技术指标
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    技术指标计算器
    包含趋势、动量、波动率、成交量等多类指标
    """

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        一次性计算所有技术指标

        Args:
            df: 包含 OHLCV 的 DataFrame

        Returns:
            添加了所有技术指标列的 DataFrame
        """
        data = df.copy()

        # --- 趋势类指标 ---
        # 移动平均线
        data['sma_20'] = data['Close'].rolling(20).mean()
        data['sma_50'] = data['Close'].rolling(50).mean()
        data['sma_200'] = data['Close'].rolling(200).mean()

        # 指数移动平均
        data['ema_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['ema_50'] = data['Close'].ewm(span=50, adjust=False).mean()

        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']

        # --- 动量类指标 ---
        # RSI (14日)
        data['rsi_14'] = TechnicalIndicators._rsi(data['Close'], 14)
        data['rsi_6'] = TechnicalIndicators._rsi(data['Close'], 6)

        # 随机指标 KDJ
        data['stoch_k'], data['stoch_d'] = TechnicalIndicators._stochastic(
            data['High'], data['Low'], data['Close']
        )

        # 动量（价格变化率）
        data['mom_5'] = data['Close'].pct_change(5)   # 5日动量
        data['mom_20'] = data['Close'].pct_change(20)  # 20日动量
        data['mom_60'] = data['Close'].pct_change(60)  # 60日动量

        # --- 波动率类指标 ---
        # 布林带 (20日, 2倍标准差)
        bb_mid = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['bb_upper'] = bb_mid + 2 * bb_std
        data['bb_lower'] = bb_mid - 2 * bb_std
        data['bb_mid'] = bb_mid
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / bb_mid
        data['bb_pct'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

        # ATR (平均真实波动范围)
        data['atr_14'] = TechnicalIndicators._atr(
            data['High'], data['Low'], data['Close'], 14
        )

        # 历史波动率（20日年化）
        daily_returns = data['Close'].pct_change()
        data['volatility_20'] = daily_returns.rolling(20).std() * np.sqrt(252)

        # --- 成交量类指标 ---
        data['volume_sma_20'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma_20']

        # OBV (能量潮)
        data['obv'] = TechnicalIndicators._obv(data['Close'], data['Volume'])

        # 成交量加权均价 VWAP (日内，此处用近似20日)
        data['vwap'] = (data['Close'] * data['Volume']).rolling(20).sum() / \
                       data['Volume'].rolling(20).sum()

        # --- 趋势强度 ---
        # ADX (平均方向指数)
        data['adx'] = TechnicalIndicators._adx(
            data['High'], data['Low'], data['Close'], 14
        )

        # --- 价格位置指标 ---
        # 52周高低位置
        data['high_52w'] = data['Close'].rolling(252).max()
        data['low_52w'] = data['Close'].rolling(252).min()
        data['pct_from_52w_high'] = (data['Close'] - data['high_52w']) / data['high_52w']

        # 价格相对于布林带的位置（百分比B）
        data['close_vs_sma20'] = (data['Close'] - data['sma_20']) / data['sma_20']
        data['close_vs_sma50'] = (data['Close'] - data['sma_50']) / data['sma_50']

        return data

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """计算 RSI"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ):
        """计算随机振荡器 %K 和 %D"""
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(d_period).mean()
        return stoch_k, stoch_d

    @staticmethod
    def _atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """计算 ATR（平均真实波动范围）"""
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        return atr

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算 OBV（能量潮）"""
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    @staticmethod
    def _adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """计算 ADX（平均方向指数）"""
        # 上升方向运动和下降方向运动
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # 计算真实波动范围
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        # 平滑
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(
            alpha=1/period, min_periods=period, adjust=False
        ).mean() / atr
        minus_di = 100 * minus_dm.ewm(
            alpha=1/period, min_periods=period, adjust=False
        ).mean() / atr

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        return adx

    @staticmethod
    def get_latest_values(df: pd.DataFrame) -> dict:
        """获取最新一行的所有技术指标值"""
        if df.empty:
            return {}
        row = df.iloc[-1]
        return row.to_dict()
