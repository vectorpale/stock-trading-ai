"""
富途（Futu）交易费率计算器
Futu Brokerage Fee Calculator

基于富途牛牛（2024年）实际收费标准
包含美股、港股完整费用结构
"""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class FutuFeeCalculator:
    """
    富途平台费率计算器

    美股费用（买入/卖出均收）：
    - 佣金：$0.0049/股，最低 $0.99
    - 平台费：$0.0049/股，最低 $0.99
    - 卖出时额外收费：
        - SEC 费：成交额 × 0.0000278
        - FINRA TAF：$0.000145/股，最高 $7.27

    港股费用（买入/卖出均收）：
    - 佣金：成交额 × 0.03%，最低 HK$3
    - 平台费：成交额 × 0.03%，最低 HK$3
    - 印花税：成交额 × 0.1%（买卖双方均收）
    - 证监会征费：成交额 × 0.0027%
    - 联交所交易费：成交额 × 0.00565%
    """

    def __init__(self, config: Dict = None):
        cfg = config or {}
        us_cfg = cfg.get('us', {})
        hk_cfg = cfg.get('hk', {})

        # 美股费率
        self.us_commission_per_share = us_cfg.get('commission_per_share', 0.0049)
        self.us_min_commission = us_cfg.get('min_commission', 0.99)
        self.us_platform_per_share = us_cfg.get('platform_fee_per_share', 0.0049)
        self.us_min_platform = us_cfg.get('min_platform_fee', 0.99)
        self.us_sec_fee_rate = us_cfg.get('sec_fee_rate', 0.0000278)
        self.us_finra_per_share = us_cfg.get('finra_taf_per_share', 0.000145)
        self.us_finra_max = us_cfg.get('finra_taf_max', 7.27)

        # 港股费率
        self.hk_commission_rate = hk_cfg.get('commission_rate', 0.0003)
        self.hk_min_commission = hk_cfg.get('min_commission_hkd', 3.0)
        self.hk_platform_rate = hk_cfg.get('platform_fee_rate', 0.0003)
        self.hk_min_platform = hk_cfg.get('min_platform_fee_hkd', 3.0)
        self.hk_stamp_duty_rate = hk_cfg.get('stamp_duty_rate', 0.001)
        self.hk_levy_rate = hk_cfg.get('transaction_levy_rate', 0.000027)
        self.hk_trading_fee_rate = hk_cfg.get('trading_fee_rate', 0.0000565)

    def calc_us_fees(
        self,
        shares: int,
        price: float,
        is_sell: bool = False
    ) -> Dict[str, float]:
        """
        计算美股单笔交易费用

        Args:
            shares: 股数（正整数）
            price: 每股价格（USD）
            is_sell: 是否为卖出（卖出多收 SEC 费和 FINRA）

        Returns:
            费用明细字典，total 为总费用（USD）
        """
        trade_value = shares * price

        # 佣金
        commission = max(shares * self.us_commission_per_share, self.us_min_commission)

        # 平台使用费
        platform_fee = max(shares * self.us_platform_per_share, self.us_min_platform)

        # 卖出时的监管费
        sec_fee = 0.0
        finra_taf = 0.0
        if is_sell:
            sec_fee = trade_value * self.us_sec_fee_rate
            finra_taf = min(shares * self.us_finra_per_share, self.us_finra_max)

        total = commission + platform_fee + sec_fee + finra_taf

        return {
            'commission': round(commission, 4),
            'platform_fee': round(platform_fee, 4),
            'sec_fee': round(sec_fee, 4),
            'finra_taf': round(finra_taf, 4),
            'total': round(total, 4),
            'fee_rate': total / trade_value if trade_value > 0 else 0,
        }

    def calc_hk_fees(
        self,
        shares: int,
        price: float,
        is_sell: bool = False
    ) -> Dict[str, float]:
        """
        计算港股单笔交易费用

        Args:
            shares: 股数
            price: 每股价格（HKD）
            is_sell: 是否为卖出

        Returns:
            费用明细字典，total 为总费用（HKD）
        """
        trade_value = shares * price

        # 佣金
        commission = max(trade_value * self.hk_commission_rate, self.hk_min_commission)

        # 平台费
        platform_fee = max(trade_value * self.hk_platform_rate, self.hk_min_platform)

        # 印花税（买卖双方均收）
        stamp_duty = trade_value * self.hk_stamp_duty_rate

        # 证监会征费
        levy = trade_value * self.hk_levy_rate

        # 联交所交易费
        trading_fee = trade_value * self.hk_trading_fee_rate

        total = commission + platform_fee + stamp_duty + levy + trading_fee

        return {
            'commission': round(commission, 4),
            'platform_fee': round(platform_fee, 4),
            'stamp_duty': round(stamp_duty, 4),
            'levy': round(levy, 4),
            'trading_fee': round(trading_fee, 4),
            'total': round(total, 4),
            'fee_rate': total / trade_value if trade_value > 0 else 0,
        }

    def calc_round_trip_cost(
        self,
        market: str,
        shares: int,
        buy_price: float,
        sell_price: float
    ) -> Dict[str, float]:
        """
        计算完整一个来回（买入+卖出）的总成本

        Args:
            market: 'us' 或 'hk'
            shares: 股数
            buy_price: 买入价格
            sell_price: 卖出价格

        Returns:
            含买入费、卖出费、总费用、费率等信息
        """
        if market == 'us':
            buy_fees = self.calc_us_fees(shares, buy_price, is_sell=False)
            sell_fees = self.calc_us_fees(shares, sell_price, is_sell=True)
        else:
            buy_fees = self.calc_hk_fees(shares, buy_price, is_sell=False)
            sell_fees = self.calc_hk_fees(shares, sell_price, is_sell=True)

        total_cost = buy_fees['total'] + sell_fees['total']
        trade_value = shares * buy_price

        return {
            'buy_fees': buy_fees,
            'sell_fees': sell_fees,
            'total_cost': round(total_cost, 4),
            'total_fee_rate': total_cost / trade_value if trade_value > 0 else 0,
        }

    def get_effective_fee_rate(self, market: str, price: float, shares: int = 100) -> float:
        """
        获取给定价格和股数下的有效费率（双边）
        用于回测中快速估算交易成本
        """
        result = self.calc_round_trip_cost(
            market=market,
            shares=shares,
            buy_price=price,
            sell_price=price * 1.01  # 假设1%盈利卖出
        )
        return result['total_fee_rate']
