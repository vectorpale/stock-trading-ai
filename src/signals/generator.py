"""
综合信号生成器
Comprehensive Signal Generator

融合多策略信号 + AI 情绪分析 → 最终买卖指令
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..data.fetcher import DataFetcher
from ..indicators.technical import TechnicalIndicators
from ..strategies.momentum import MomentumStrategy
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.multi_factor import MultiFactorStrategy
from ..ai.llm_analyzer import LLMAnalyzer
from ..utils.helpers import get_market_type, calculate_position_size

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    综合交易信号生成器

    流程：
    1. 获取股票数据（自动缓存）
    2. 计算全部技术指标
    3. 运行三个量化策略（动量/均值回归/多因子）
    4. AI 情绪分析（新闻）
    5. AI 技术信号解读
    6. 综合评分 → 最终信号
    7. 计算仓位建议和止损止盈
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        data_cfg = config.get('data', {})
        ai_cfg = config.get('ai', {})
        risk_cfg = config.get('risk', {})

        self.fetcher = DataFetcher(cache_dir=data_cfg.get('cache_dir', 'data/cache'))

        # 策略权重
        weights = config.get('strategy_weights', {})
        self.weights = {
            'momentum': weights.get('momentum', 0.35),
            'mean_reversion': weights.get('mean_reversion', 0.25),
            'multi_factor': weights.get('multi_factor', 0.40),
        }

        # AI 情绪权重
        self.ai_enabled = ai_cfg.get('enabled', True)
        self.sentiment_weight = ai_cfg.get('sentiment_weight', 0.20)

        # 风险参数
        self.max_position_size = risk_cfg.get('max_position_size', 0.15)
        self.max_total_positions = risk_cfg.get('max_total_positions', 8)
        self.stop_loss_pct = risk_cfg.get('stop_loss_pct', 0.07)
        self.take_profit_pct = risk_cfg.get('take_profit_pct', 0.20)

        # 实例化策略
        self.momentum = MomentumStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.multi_factor = MultiFactorStrategy()

        # AI 分析器
        if self.ai_enabled:
            self.llm = LLMAnalyzer(model=ai_cfg.get('model', 'claude-opus-4-5'))
        else:
            self.llm = None

    def generate_for_symbol(
        self,
        symbol: str,
        include_ai: bool = True,
        capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        为单只股票生成完整交易信号

        Args:
            symbol: 股票代码
            include_ai: 是否启用 AI 分析
            capital: 当前可用资金

        Returns:
            完整的信号字典
        """
        logger.info(f"正在分析 {symbol}...")
        market = get_market_type(symbol)

        # ─── 1. 获取数据 ───
        df = self.fetcher.fetch_ohlcv(symbol, period="2y", interval="1d")
        if df is None or len(df) < 60:
            return {
                'symbol': symbol,
                'action': 'SKIP',
                'reason': f'数据不足（只有 {len(df) if df is not None else 0} 条）',
                'final_score': 0,
                'final_confidence': 0,
            }

        # ─── 2. 计算技术指标 ───
        df_with_indicators = TechnicalIndicators.compute_all(df)
        latest_indicators = TechnicalIndicators.get_latest_values(df_with_indicators)
        price = float(df['Close'].iloc[-1])

        # ─── 3. 量化策略信号 ───
        try:
            mom_sig = self.momentum.generate_signal(df_with_indicators, symbol)
            mr_sig = self.mean_reversion.generate_signal(df_with_indicators, symbol)
            mf_sig = self.multi_factor.generate_signal(df_with_indicators, symbol)
        except Exception as e:
            logger.error(f"{symbol} 策略运行失败: {e}")
            mom_sig = mr_sig = mf_sig = {'score': 0, 'action': 'HOLD', 'reason': '策略异常'}

        strategy_signals = {
            'momentum': mom_sig,
            'mean_reversion': mr_sig,
            'multi_factor': mf_sig,
        }

        # ─── 4. 量化综合评分 ───
        quant_score = (
            mom_sig.get('score', 0) * self.weights['momentum'] +
            mr_sig.get('score', 0) * self.weights['mean_reversion'] +
            mf_sig.get('score', 0) * self.weights['multi_factor']
        )

        # ─── 5. AI 分析 ───
        ai_result = None
        sentiment_result = None
        final_score = quant_score

        if include_ai and self.llm is not None:
            # 5a. 新闻情绪分析
            try:
                ticker_info = self.fetcher.fetch_ticker_info(symbol)
                news = self.fetcher.fetch_news(symbol, max_items=10)
                if news:
                    sentiment_result = self.llm.analyze_news_sentiment(
                        symbol, news,
                        company_name=ticker_info.get('name', symbol)
                    )
            except Exception as e:
                logger.warning(f"{symbol} 情绪分析失败: {e}")

            # 5b. AI 技术分析
            try:
                ai_result = self.llm.analyze_technical_signals(
                    symbol, strategy_signals, latest_indicators, market
                )
            except Exception as e:
                logger.warning(f"{symbol} AI 技术分析失败: {e}")

            # 5c. 融合 AI 评分
            ai_score = 0.0
            if ai_result:
                ai_score += ai_result.get('score', 0) * 0.15
            if sentiment_result:
                ai_score += sentiment_result.get('sentiment_score', 0) * self.sentiment_weight

            # 加权融合（量化80%，AI 20%）
            final_score = quant_score * (1 - self.sentiment_weight) + ai_score

        final_score = float(np.clip(final_score, -100, 100))

        # ─── 6. 确定最终动作 ───
        action, confidence = self._score_to_action(final_score, strategy_signals)

        # ─── 7. 计算止损止盈 ───
        atr = latest_indicators.get('atr_14', price * 0.02) or price * 0.02
        stop_loss = round(price * (1 - self.stop_loss_pct), 4)
        take_profit = round(price * (1 + self.take_profit_pct), 4)

        # ─── 8. 仓位建议 ───
        position_pct = 0.0
        if action == 'BUY':
            position_pct = min(
                self.max_position_size,
                0.05 + (final_score - 30) / 700
            )
        position_info = calculate_position_size(
            capital, price, position_pct, self.max_position_size
        )

        # ─── 9. 汇总原因 ───
        all_reasons = []
        for name, sig in strategy_signals.items():
            strategy_label = {'momentum': '动量', 'mean_reversion': '均值回归', 'multi_factor': '多因子'}[name]
            if sig.get('action') in ('BUY', 'SELL') and sig.get('reason'):
                all_reasons.append(f"[{strategy_label}] {sig['reason']}")
        if ai_result and ai_result.get('reasoning'):
            all_reasons.append(f"[AI] {ai_result['reasoning'][:80]}...")

        return {
            'symbol': symbol,
            'market': market,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'action': action,
            'final_score': round(final_score, 1),
            'final_confidence': round(confidence, 1),
            'price': round(price, 4),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_pct': round(position_pct * 100, 1),
            'position_shares': position_info['shares'],
            'position_investment': round(position_info['investment'], 2),
            'reason': '；'.join(all_reasons[:3]),

            # 详细分解
            'strategy_signals': strategy_signals,
            'quant_score': round(quant_score, 1),
            'sentiment': sentiment_result,
            'ai_analysis': ai_result,
        }

    def generate_for_watchlist(
        self,
        symbols: List[str],
        include_ai: bool = True,
        capital: float = 100000.0,
        max_signals: int = 5
    ) -> List[Dict]:
        """
        批量生成信号并排序

        Returns:
            按信号强度排序的信号列表（最强的 max_signals 个）
        """
        all_signals = []

        for symbol in symbols:
            try:
                sig = self.generate_for_symbol(symbol, include_ai=include_ai, capital=capital)
                if sig.get('action') != 'SKIP':
                    all_signals.append(sig)
            except Exception as e:
                logger.error(f"处理 {symbol} 时出错: {e}")

        # 排序：买入信号按分数降序，卖出信号按分数升序
        buy_signals = sorted(
            [s for s in all_signals if s['action'] == 'BUY'],
            key=lambda x: x['final_score'], reverse=True
        )
        sell_signals = sorted(
            [s for s in all_signals if s['action'] == 'SELL'],
            key=lambda x: x['final_score']
        )
        watch_signals = sorted(
            [s for s in all_signals if s['action'] == 'WATCH'],
            key=lambda x: abs(x['final_score']), reverse=True
        )
        other_signals = [s for s in all_signals if s['action'] == 'HOLD']

        # 买入信号最多 max_signals 个
        result = buy_signals[:max_signals] + sell_signals + watch_signals[:3] + other_signals

        logger.info(
            f"信号生成完成: {len(buy_signals)} 买入, "
            f"{len(sell_signals)} 卖出, {len(watch_signals)} 观察"
        )
        return result

    def _score_to_action(
        self,
        score: float,
        strategy_signals: Dict
    ) -> tuple:
        """
        将综合分数转换为动作

        需要策略信号具有一定的一致性（减少噪声）
        """
        # 策略一致性检查：至少2个策略同向
        buy_votes = sum(1 for s in strategy_signals.values() if s.get('action') == 'BUY')
        sell_votes = sum(1 for s in strategy_signals.values() if s.get('action') == 'SELL')

        if score >= 45 and buy_votes >= 2:
            return 'BUY', min(score, 95)
        elif score >= 30 and buy_votes >= 1:
            return 'WATCH', score * 0.8
        elif score <= -45 and sell_votes >= 2:
            return 'SELL', min(abs(score), 95)
        elif score <= -30 and sell_votes >= 1:
            return 'WATCH', abs(score) * 0.8
        elif abs(score) >= 25:
            return 'WATCH', abs(score)
        else:
            return 'HOLD', max(20, abs(score))
