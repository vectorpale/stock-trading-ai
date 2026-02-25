"""
模拟交易组合跟踪器
Paper Trading Portfolio Tracker

管理一个模拟账户（默认 $1,000,000），根据交易信号执行买卖操作，
持久化保存持仓和交易历史。
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_CAPITAL = 1_000_000.0
PORTFOLIO_FILE = "data/portfolio.json"


class PortfolioTracker:
    """
    模拟交易组合管理器

    持久化存储：data/portfolio.json
    记录：现金余额、持仓明细、交易历史、每日净值快照
    """

    def __init__(self, portfolio_path: str = None, initial_capital: float = None):
        self.path = Path(portfolio_path or PORTFOLIO_FILE)
        self.initial_capital = initial_capital or DEFAULT_INITIAL_CAPITAL

        if self.path.exists():
            self._load()
        else:
            self._init_new()

    def _init_new(self):
        """初始化一个新的模拟账户"""
        self.data = {
            "created_at": datetime.now().isoformat(),
            "initial_capital": self.initial_capital,
            "cash": self.initial_capital,
            "positions": {},      # symbol -> {shares, avg_cost, total_cost, first_buy}
            "trade_history": [],   # [{date, symbol, action, shares, price, amount, fees, cash_after}]
            "daily_snapshots": [], # [{date, cash, market_value, total_value}]
        }
        self._save()
        logger.info(f"新模拟账户已创建，初始资金: ${self.initial_capital:,.0f}")

    def _load(self):
        """从文件加载"""
        with open(self.path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        logger.info(f"模拟账户已加载: 现金 ${self.cash:,.2f}, 持仓 {len(self.positions)} 只")

    def _save(self):
        """保存到文件"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # ─── 属性访问 ───

    @property
    def cash(self) -> float:
        return self.data["cash"]

    @property
    def positions(self) -> Dict:
        return self.data["positions"]

    @property
    def trade_history(self) -> List[Dict]:
        return self.data["trade_history"]

    @property
    def daily_snapshots(self) -> List[Dict]:
        return self.data["daily_snapshots"]

    # ─── 交易操作 ───

    def buy(self, symbol: str, shares: int, price: float, fees: float = 0.0,
            reason: str = "") -> Dict:
        """
        买入股票

        Returns:
            交易记录 dict，或含 error 的 dict
        """
        amount = shares * price + fees

        if amount > self.cash:
            return {"error": f"资金不足: 需要 ${amount:,.2f}, 可用 ${self.cash:,.2f}"}

        if shares <= 0:
            return {"error": "股数必须大于0"}

        # 扣减现金
        self.data["cash"] -= amount

        # 更新持仓
        pos = self.positions.get(symbol)
        if pos:
            # 加仓：更新均价
            old_total = pos["shares"] * pos["avg_cost"]
            new_total = old_total + shares * price
            pos["shares"] += shares
            pos["avg_cost"] = new_total / pos["shares"]
            pos["total_cost"] += shares * price
        else:
            self.positions[symbol] = {
                "shares": shares,
                "avg_cost": price,
                "total_cost": shares * price,
                "first_buy": datetime.now().strftime("%Y-%m-%d"),
            }

        # 记录交易
        trade = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "symbol": symbol,
            "action": "BUY",
            "shares": shares,
            "price": price,
            "amount": round(shares * price, 2),
            "fees": round(fees, 2),
            "cash_after": round(self.cash, 2),
            "reason": reason,
        }
        self.trade_history.append(trade)
        self._save()

        logger.info(f"买入 {symbol} {shares}股 @ ${price:.2f}, 费用 ${fees:.2f}")
        return trade

    def sell(self, symbol: str, shares: int, price: float, fees: float = 0.0,
             reason: str = "") -> Dict:
        """
        卖出股票

        Returns:
            交易记录 dict（含 pnl 盈亏），或含 error 的 dict
        """
        pos = self.positions.get(symbol)
        if not pos:
            return {"error": f"未持有 {symbol}"}

        if shares > pos["shares"]:
            return {"error": f"{symbol} 持仓 {pos['shares']} 股，不足 {shares} 股"}

        if shares <= 0:
            return {"error": "股数必须大于0"}

        # 计算盈亏
        sell_amount = shares * price
        cost_basis = shares * pos["avg_cost"]
        pnl = sell_amount - cost_basis - fees
        pnl_pct = pnl / cost_basis if cost_basis > 0 else 0

        # 增加现金
        self.data["cash"] += sell_amount - fees

        # 更新持仓
        pos["shares"] -= shares
        pos["total_cost"] -= shares * pos["avg_cost"]
        if pos["shares"] <= 0:
            del self.positions[symbol]

        # 记录交易
        trade = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "symbol": symbol,
            "action": "SELL",
            "shares": shares,
            "price": price,
            "amount": round(sell_amount, 2),
            "fees": round(fees, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "cash_after": round(self.cash, 2),
            "reason": reason,
        }
        self.trade_history.append(trade)
        self._save()

        logger.info(f"卖出 {symbol} {shares}股 @ ${price:.2f}, 盈亏 ${pnl:+,.2f}")
        return trade

    def execute_signals(self, signals: List[Dict], current_prices: Dict[str, float] = None,
                        fee_calculator=None) -> List[Dict]:
        """
        根据信号批量执行交易

        Args:
            signals: 信号列表 (由 SignalGenerator 生成)
            current_prices: {symbol: price} 实时价格（可选，不传则用信号中的 price）
            fee_calculator: FutuFeeCalculator 实例（可选）

        Returns:
            执行的交易记录列表
        """
        trades = []

        for sig in signals:
            symbol = sig.get("symbol", "")
            action = sig.get("action", "")
            price = (current_prices or {}).get(symbol, sig.get("price", 0))

            if price <= 0:
                continue

            if action == "BUY":
                # 使用信号中的建议仓位
                position_pct = sig.get("position_pct", 0) / 100.0  # 信号中是百分比
                if position_pct <= 0:
                    continue

                # 已持有则跳过
                if symbol in self.positions:
                    logger.info(f"已持有 {symbol}，跳过买入")
                    continue

                investment = self.cash * position_pct
                shares = int(investment / price)
                if shares <= 0:
                    continue

                # 计算费用
                fees = 0.0
                if fee_calculator:
                    market = 'hk' if symbol.endswith('.HK') else 'us'
                    if market == 'us':
                        fee_detail = fee_calculator.calc_us_fees(shares, price, is_sell=False)
                    else:
                        fee_detail = fee_calculator.calc_hk_fees(shares, price, is_sell=False)
                    fees = fee_detail['total']

                reason = sig.get("reason", "")[:200]
                result = self.buy(symbol, shares, price, fees, reason=reason)
                if "error" not in result:
                    trades.append(result)

            elif action == "SELL":
                pos = self.positions.get(symbol)
                if not pos:
                    continue

                shares = pos["shares"]
                fees = 0.0
                if fee_calculator:
                    market = 'hk' if symbol.endswith('.HK') else 'us'
                    if market == 'us':
                        fee_detail = fee_calculator.calc_us_fees(shares, price, is_sell=True)
                    else:
                        fee_detail = fee_calculator.calc_hk_fees(shares, price, is_sell=True)
                    fees = fee_detail['total']

                reason = sig.get("reason", "")[:200]
                result = self.sell(symbol, shares, price, fees, reason=reason)
                if "error" not in result:
                    trades.append(result)

        return trades

    def take_snapshot(self, current_prices: Dict[str, float]) -> Dict:
        """
        记录每日净值快照

        Args:
            current_prices: {symbol: current_price}

        Returns:
            快照 dict
        """
        market_value = 0.0
        for symbol, pos in self.positions.items():
            curr_price = current_prices.get(symbol, pos["avg_cost"])
            market_value += pos["shares"] * curr_price

        total_value = self.cash + market_value
        snapshot = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "cash": round(self.cash, 2),
            "market_value": round(market_value, 2),
            "total_value": round(total_value, 2),
            "return_pct": round((total_value / self.data["initial_capital"] - 1) * 100, 2),
            "positions_count": len(self.positions),
        }

        # 避免同日重复快照
        if self.daily_snapshots and self.daily_snapshots[-1]["date"] == snapshot["date"]:
            self.daily_snapshots[-1] = snapshot
        else:
            self.daily_snapshots.append(snapshot)

        self._save()
        return snapshot

    def get_positions_with_market_data(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        获取持仓详情（含现价、盈亏等）

        Args:
            current_prices: {symbol: current_price}

        Returns:
            持仓列表，每个元素包含完整信息
        """
        result = []
        for symbol, pos in self.positions.items():
            curr_price = current_prices.get(symbol, pos["avg_cost"])
            shares = pos["shares"]
            avg_cost = pos["avg_cost"]
            total_cost = shares * avg_cost
            market_value = shares * curr_price
            pnl = market_value - total_cost
            pnl_pct = (curr_price / avg_cost - 1) * 100 if avg_cost > 0 else 0

            result.append({
                "symbol": symbol,
                "shares": shares,
                "avg_cost": round(avg_cost, 2),
                "current_price": round(curr_price, 2),
                "total_cost": round(total_cost, 2),
                "market_value": round(market_value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "first_buy": pos.get("first_buy", ""),
                "weight": 0,  # 由调用方计算
            })

        # 计算持仓权重
        total_market_value = sum(p["market_value"] for p in result)
        for p in result:
            p["weight"] = round(p["market_value"] / total_market_value * 100, 1) if total_market_value > 0 else 0

        # 按市值排序
        result.sort(key=lambda x: x["market_value"], reverse=True)
        return result

    def get_account_summary(self, current_prices: Dict[str, float]) -> Dict:
        """
        获取账户概览

        Args:
            current_prices: {symbol: current_price}

        Returns:
            账户概览 dict
        """
        positions_detail = self.get_positions_with_market_data(current_prices)
        market_value = sum(p["market_value"] for p in positions_detail)
        total_value = self.cash + market_value
        total_pnl = total_value - self.data["initial_capital"]
        total_return = (total_value / self.data["initial_capital"] - 1) * 100

        # 统计盈亏持仓数
        winning = sum(1 for p in positions_detail if p["pnl"] > 0)
        losing = sum(1 for p in positions_detail if p["pnl"] < 0)

        return {
            "initial_capital": self.data["initial_capital"],
            "cash": round(self.cash, 2),
            "market_value": round(market_value, 2),
            "total_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return, 2),
            "positions_count": len(self.positions),
            "winning_positions": winning,
            "losing_positions": losing,
            "total_trades": len(self.trade_history),
            "created_at": self.data.get("created_at", ""),
            "positions": positions_detail,
        }

    def reset(self, initial_capital: float = None):
        """重置账户（清空所有持仓和历史）"""
        self.initial_capital = initial_capital or self.data.get("initial_capital", DEFAULT_INITIAL_CAPITAL)
        self._init_new()
        logger.info(f"账户已重置，初始资金: ${self.initial_capital:,.0f}")
