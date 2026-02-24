"""
综合信号生成器
Comprehensive Signal Generator

融合多策略信号 + AI 情绪分析 → 最终买卖指令

AI 模式选择（通过 config.yaml ai.mode 控制）：
  - 'debate'  : 5智能体辩论引擎（多头/空头/量化/宏观/风险官 + CIO 裁决）[推荐]
  - 'simple'  : 单智能体技术分析（原 LLMAnalyzer，速度更快，费用更低）
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
from ..ai.advanced_debate import AdvancedDebateEngine, DecisionMemory
from ..utils.helpers import get_market_type, calculate_position_size

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    综合交易信号生成器

    流程：
    1. 获取股票数据（自动缓存）
    2. 计算全部技术指标
    3. 运行三个量化策略（动量/均值回归/多因子）
    4. AI 分析（两种模式）：
       - debate 模式：5智能体辩论（多头/空头/量化/宏观/风险官）→ CIO 裁决
       - simple 模式：单智能体技术分析 + 新闻情绪
    5. 综合评分 → 最终信号
    6. 计算仓位建议和止损止盈
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

        # AI 配置
        self.ai_enabled = ai_cfg.get('enabled', True)
        self.ai_mode = ai_cfg.get('mode', 'debate')       # 'debate' | 'simple'
        self.sentiment_weight = ai_cfg.get('sentiment_weight', 0.20)
        self.debate_rounds = ai_cfg.get('debate_rounds', 1)

        # 风险参数
        self.max_position_size = risk_cfg.get('max_position_size', 0.15)
        self.max_total_positions = risk_cfg.get('max_total_positions', 8)
        self.stop_loss_pct = risk_cfg.get('stop_loss_pct', 0.07)
        self.take_profit_pct = risk_cfg.get('take_profit_pct', 0.20)

        # 实例化策略
        self.momentum = MomentumStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.multi_factor = MultiFactorStrategy()

        # AI 分析器（按模式初始化）
        self.llm = None
        self.debate_engine = None
        if self.ai_enabled:
            import os
            api_key = os.getenv('ANTHROPIC_API_KEY', '')
            model = ai_cfg.get('model', 'claude-opus-4-6')
            if self.ai_mode == 'debate':
                memory_dir = data_cfg.get('memory_dir', 'data/memory')
                self.debate_engine = AdvancedDebateEngine(
                    api_key=api_key,
                    model=model,
                    memory=DecisionMemory(memory_dir=memory_dir),
                    convergence_threshold=ai_cfg.get('convergence_threshold', 0.85),
                    max_debate_rounds=ai_cfg.get('max_debate_rounds', 3),
                )
                logger.info(f"[AI] 多智能体辩论引擎已初始化（{self.debate_rounds} 轮辩论）")
            else:
                self.llm = LLMAnalyzer(model=model)
                logger.info("[AI] 单智能体模式已初始化")

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
        debate_result = None
        final_score = quant_score

        if include_ai and self.ai_enabled:

            if self.ai_mode == 'debate' and self.debate_engine is not None:
                # ── 5A. 多智能体辩论模式 ─────────────────────────
                # 先获取新闻情绪（可选，作为辩论输入）
                try:
                    ticker_info = self.fetcher.fetch_ticker_info(symbol)
                    news = self.fetcher.fetch_news(symbol, max_items=10)
                    if news and self.llm:
                        sentiment_result = self.llm.analyze_news_sentiment(
                            symbol, news,
                            company_name=ticker_info.get('name', symbol)
                        )
                except Exception as e:
                    logger.warning(f"{symbol} 情绪分析失败: {e}")

                try:
                    debate_result = self.debate_engine.run_debate(
                        symbol=symbol,
                        market=market.lower(),
                        price=price,
                        strategy_signals=strategy_signals,
                        indicators=latest_indicators,
                        sentiment=sentiment_result,
                        debate_rounds=self.debate_rounds,
                    )
                    cio = debate_result.get('cio_decision', {})
                    ai_score = cio.get('score', 0) * self.sentiment_weight
                    # 融合：量化 80% + 辩论引擎 CIO 20%
                    final_score = quant_score * (1 - self.sentiment_weight) + ai_score
                except Exception as e:
                    logger.warning(f"{symbol} 多智能体辩论失败: {e}")

            elif self.llm is not None:
                # ── 5B. 单智能体模式（原 simple 逻辑）────────────
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

                try:
                    ai_result = self.llm.analyze_technical_signals(
                        symbol, strategy_signals, latest_indicators, market
                    )
                except Exception as e:
                    logger.warning(f"{symbol} AI 技术分析失败: {e}")

                ai_score = 0.0
                if ai_result:
                    ai_score += ai_result.get('score', 0) * 0.15
                if sentiment_result:
                    ai_score += sentiment_result.get('sentiment_score', 0) * self.sentiment_weight
                final_score = quant_score * (1 - self.sentiment_weight) + ai_score

        final_score = float(np.clip(final_score, -100, 100))

        # ─── 6. 确定最终动作 ───
        # debate 模式：优先使用 CIO 裁决的 action
        if debate_result:
            cio = debate_result.get('cio_decision', {})
            action = cio.get('action', 'HOLD')
            confidence = float(cio.get('confidence', 50))
            # 风险委员会强制 VETO → HOLD
            if debate_result.get('risk_review', {}).get('verdict') == 'VETO':
                action = 'HOLD'
                confidence = 0.0
        else:
            action, confidence = self._score_to_action(final_score, strategy_signals)

        # ─── 7. 计算止损止盈 ───
        atr = latest_indicators.get('atr_14', price * 0.02) or price * 0.02
        # debate 模式使用风险委员会给出的止损幅度
        if debate_result:
            cio = debate_result.get('cio_decision', {})
            ps = cio.get('position_sizing', {})
            stop_loss_pct = ps.get('stop_loss_pct', self.stop_loss_pct)
            take_profit_pct = ps.get('take_profit_pct', self.take_profit_pct)
        else:
            stop_loss_pct = self.stop_loss_pct
            take_profit_pct = self.take_profit_pct

        stop_loss = round(price * (1 - stop_loss_pct), 4)
        take_profit = round(price * (1 + take_profit_pct), 4)

        # ─── 8. 仓位建议 ───
        position_pct = 0.0
        if action == 'BUY':
            if debate_result:
                cio = debate_result.get('cio_decision', {})
                ps = cio.get('position_sizing', {})
                risk_max = debate_result.get('risk_review', {}).get('max_position_pct', 15)
                suggested = ps.get('suggested_pct', 5)
                position_pct = min(suggested, risk_max, self.max_position_size * 100) / 100
            else:
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
        if debate_result:
            cio = debate_result.get('cio_decision', {})
            if cio.get('verdict'):
                all_reasons.append(f"[CIO] {cio['verdict']}")
            consensus = debate_result.get('agent_consensus', '')
            if consensus:
                all_reasons.append(f"[辩论共识] {consensus}")
        elif ai_result and ai_result.get('reasoning'):
            all_reasons.append(f"[AI] {ai_result['reasoning'][:80]}...")

        result = {
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

        # debate 模式附加辩论详情
        if debate_result:
            result['debate'] = {
                'cio_decision': debate_result.get('cio_decision', {}),
                'agent_consensus': debate_result.get('agent_consensus', ''),
                'convergence_score': debate_result.get('convergence_score', 0),
                'actual_debate_rounds': debate_result.get('actual_debate_rounds', 0),
                'risk_review': debate_result.get('risk_review', {}),
                'agents_summary': {
                    k: {
                        'name': v.get('agent_name', k),
                        'position': v.get('position', '?'),
                        'score': v.get('score', 0),
                        'confidence': v.get('confidence', 0),
                        'core_argument': v.get('core_argument', ''),
                    }
                    for k, v in debate_result.get('agents', {}).items()
                },
            }

        return result

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
