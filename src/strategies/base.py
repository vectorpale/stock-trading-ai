"""
策略基类
Base Strategy Class
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseStrategy(ABC):
    """所有策略的抽象基类"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        生成交易信号

        Returns:
            {
                'symbol': str,
                'score': float,       # -100 到 +100，正数看多，负数看空
                'action': str,        # 'BUY', 'SELL', 'HOLD', 'WATCH'
                'confidence': float,  # 0-100 的信心分数
                'reason': str,        # 信号原因说明
                'price': float,       # 当前价格
                'stop_loss': float,   # 建议止损价
                'take_profit': float, # 建议止盈价
                'position_pct': float # 建议仓位比例 0-1
            }
        """
        pass

    def _get_latest(self, df: pd.DataFrame, col: str, default=None):
        """安全获取最新值"""
        try:
            val = df[col].iloc[-1]
            import numpy as np
            if pd.isna(val) or (isinstance(val, float) and np.isinf(val)):
                return default
            return val
        except Exception:
            return default
