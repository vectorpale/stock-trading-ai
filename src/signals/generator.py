"""
综合信号生成器
Comprehensive Signal Generator

两阶段架构（debate 模式）：
  阶段一：量化初筛 — 三个量化策略快速评分，过滤明显无信号的标的
  阶段二：CIO 全权裁决 — 通过初筛的标的进入多智能体辩论，CIO 作出最终决策

CIO 是唯一的权威决策者。量化信号仅作为辩论的数据输入，不参与最终评分的加权。

AI 模式（config.yaml ai.mode）：
  - 'debate' : 两阶段架构，CIO 全权裁决 [推荐]
  - 'simple' : 单智能体技术分析（速度快，费用低）
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
    综合交易信号生成器（CIO 全权模式）

    debate 模式流程：
    1. 获取数据 + 计算技术指标
    2. 三个量化策略评分 → 初筛（过滤低分标的，节省 API 调用）
    3. 通过初筛的标的 → 5智能体辩论 → CIO 全权裁决
    4. CIO 的 action / score / position_sizing 为最终输出，不再与量化混合

    simple 模式流程：
    1-2 同上
    3. 单智能体 LLMAnalyzer 分析（80%量化 + 20% AI 混合）
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        data_cfg = config.get('data', {})
        ai_cfg = config.get('ai', {})
        risk_cfg = config.get('risk', {})

        self.fetcher = DataFetcher(cache_dir=data_cfg.get('cache_dir', 'data/cache'))

        # 策略权重（量化初筛用）
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

        # 量化初筛阈值：quant_score 低于此值直接跳过辩论（debate 模式专用）
        # 负值表示"不做初筛，全部进入辩论"
        self.quant_prefilter_threshold = ai_cfg.get('quant_prefilter_threshold', 15)

        # 风险参数
        self.max_position_size = risk_cfg.get('max_position_size', 0.12)
        self.max_total_positions = risk_cfg.get('max_total_positions', 10)
        self.stop_loss_pct = risk_cfg.get('stop_loss_pct', 0.07)
        self.take_profit_pct = risk_cfg.get('take_profit_pct', 0.20)

        # CIO 最低置信度门槛：低于此值的 BUY 降级为 WATCH
        self.min_cio_confidence = risk_cfg.get('min_cio_confidence', 65)

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
            # 分角色模型（CIO 用 Opus，分析师用 Sonnet，情绪用 Haiku）
            model_analyst   = ai_cfg.get('model_analyst',   ai_cfg.get('model', 'claude-sonnet-4-6'))
            model_cio       = ai_cfg.get('model_cio',       'claude-opus-4-6')
            model_sentiment = ai_cfg.get('model_sentiment', 'claude-haiku-4-5-20251001')

            if self.ai_mode == 'debate':
                memory_dir = data_cfg.get('memory_dir', 'data/memory')
                self.debate_engine = AdvancedDebateEngine(
                    api_key=api_key,
                    model=model_analyst,
                    model_cio=model_cio,
                    memory=DecisionMemory(memory_dir=memory_dir),
                    convergence_threshold=ai_cfg.get('convergence_threshold', 0.85),
                    max_debate_rounds=ai_cfg.get('max_debate_rounds', 3),
                )
                # 情绪分析用轻量模型（Haiku）
                self.llm = LLMAnalyzer(model=model_sentiment)
                logger.info(
                    f"[AI] CIO 全权模式已初始化 | "
                    f"分析师={model_analyst} | CIO={model_cio} | 情绪={model_sentiment} | "
                    f"辩论轮数={self.debate_rounds} | 初筛阈值={self.quant_prefilter_threshold} | "
                    f"最低置信度={self.min_cio_confidence}"
                )
            else:
                self.llm = LLMAnalyzer(model=model_analyst)
                logger.info(f"[AI] 单智能体模式已初始化（model={model_analyst}）")

    def generate_for_symbol(
        self,
        symbol: str,
        include_ai: bool = True,
        capital: float = 100000.0,
        skip_prefilter: bool = False,   # 强制跳过初筛，直接进入辩论
    ) -> Dict[str, Any]:
        """
        为单只股票生成完整交易信号

        Args:
            symbol: 股票代码
            include_ai: 是否启用 AI 分析
            capital: 当前可用资金
            skip_prefilter: True 时跳过量化初筛（用于直接指定的重点标的）

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

        # ─── 3. 量化策略评分（初筛基础）───
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

        quant_score = (
            mom_sig.get('score', 0) * self.weights['momentum'] +
            mr_sig.get('score', 0) * self.weights['mean_reversion'] +
            mf_sig.get('score', 0) * self.weights['multi_factor']
        )

        # ─── 4. 量化初筛（debate 模式）───
        # 量化评分过低，不值得消耗 API 进行深度辩论 → 直接返回 HOLD
        if (
            include_ai
            and self.ai_mode == 'debate'
            and not skip_prefilter
            and self.quant_prefilter_threshold > 0
            and abs(quant_score) < self.quant_prefilter_threshold
        ):
            logger.info(
                f"  [{symbol}] 量化评分 {quant_score:.1f} 低于初筛阈值 "
                f"{self.quant_prefilter_threshold}，跳过辩论"
            )
            return self._build_quant_only_result(
                symbol, market, price, quant_score,
                strategy_signals, latest_indicators, capital,
                reason='量化评分不足，无需深度辩论'
            )

        # ─── 5. AI 分析 ───
        ai_result = None
        sentiment_result = None
        debate_result = None
        final_score = quant_score
        action = 'HOLD'
        confidence = 0.0

        if include_ai and self.ai_enabled:

            if self.ai_mode == 'debate' and self.debate_engine is not None:
                # ── 5A. CIO 全权模式 ─────────────────────────────
                # 新闻情绪作为辩论输入（可选）
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

                    # CIO 全权：直接使用 CIO 的评分，不与量化混合
                    final_score = float(cio.get('score', quant_score))
                    action = cio.get('action', 'HOLD')
                    confidence = float(cio.get('confidence', 0))

                    # 风险委员会 VETO → 强制 HOLD
                    if debate_result.get('risk_review', {}).get('verdict') == 'VETO':
                        action = 'HOLD'
                        confidence = 0.0
                        logger.info(f"  [{symbol}] 风险委员会否决，强制 HOLD")

                    # CIO 置信度不足 → BUY 降级为 WATCH
                    elif action == 'BUY' and confidence < self.min_cio_confidence:
                        logger.info(
                            f"  [{symbol}] CIO 置信度 {confidence:.0f} < {self.min_cio_confidence}，"
                            f"BUY 降级为 WATCH"
                        )
                        action = 'WATCH'

                except Exception as e:
                    logger.warning(f"{symbol} 多智能体辩论失败: {e}")
                    # 辩论失败时回退到量化信号
                    action, confidence = self._score_to_action(quant_score, strategy_signals)
                    final_score = quant_score

            elif self.llm is not None:
                # ── 5B. 单智能体模式 ─────────────────────────────
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
                action, confidence = self._score_to_action(final_score, strategy_signals)

        else:
            # AI 未启用 → 纯量化
            action, confidence = self._score_to_action(quant_score, strategy_signals)
            final_score = quant_score

        final_score = float(np.clip(final_score, -100, 100))

        # ─── 6. 止损止盈 ───
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

        # ─── 7. 仓位建议（CIO 全权时由 CIO 给出，并受风险委员会约束）───
        position_pct = 0.0
        if action == 'BUY':
            if debate_result:
                cio = debate_result.get('cio_decision', {})
                ps = cio.get('position_sizing', {})
                risk_max = debate_result.get('risk_review', {}).get('max_position_pct', 12)
                suggested = ps.get('suggested_pct', 5)
                # 三重约束：CIO建议 / 风险委员会上限 / 系统全局上限
                position_pct = min(suggested, risk_max, self.max_position_size * 100) / 100
            else:
                position_pct = min(
                    self.max_position_size,
                    0.05 + (final_score - 30) / 700
                )

        position_info = calculate_position_size(
            capital, price, position_pct, self.max_position_size
        )

        # ─── 8. 汇总原因 ───
        all_reasons = []
        for name, sig in strategy_signals.items():
            label = {'momentum': '动量', 'mean_reversion': '均值回归', 'multi_factor': '多因子'}[name]
            if sig.get('action') in ('BUY', 'SELL') and sig.get('reason'):
                all_reasons.append(f"[{label}] {sig['reason']}")
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
        max_buy_positions: int = 5,     # 最多返回的 BUY 数量（强集中持仓）
        max_watch_signals: int = 3,     # 最多返回的 WATCH 数量
    ) -> List[Dict]:
        """
        两阶段批量分析：量化初筛 → CIO 深度裁决

        debate 模式下，只有通过量化初筛的标的才进入辩论（节省 API）。
        最终 BUY 信号严格限制在 max_buy_positions 个以内，确保集中持仓。

        Args:
            symbols: 股票代码列表
            include_ai: 是否启用 AI
            capital: 可用资金
            max_buy_positions: 最终输出的最大 BUY 数量（默认5，对应集中持仓策略）
            max_watch_signals: 最终输出的最大 WATCH 数量

        Returns:
            [BUY 信号（按置信度排序）] + [SELL 信号] + [WATCH 信号（前N个）]
        """
        all_signals = []

        for symbol in symbols:
            try:
                sig = self.generate_for_symbol(
                    symbol, include_ai=include_ai, capital=capital
                )
                if sig.get('action') != 'SKIP':
                    all_signals.append(sig)
            except Exception as e:
                logger.error(f"处理 {symbol} 时出错: {e}")

        # debate 模式：按 CIO 置信度排序；simple 模式：按 final_score 排序
        sort_key = 'final_confidence' if self.ai_mode == 'debate' else 'final_score'

        buy_signals = sorted(
            [s for s in all_signals if s['action'] == 'BUY'],
            key=lambda x: x[sort_key], reverse=True
        )
        sell_signals = sorted(
            [s for s in all_signals if s['action'] == 'SELL'],
            key=lambda x: x['final_score']
        )
        watch_signals = sorted(
            [s for s in all_signals if s['action'] == 'WATCH'],
            key=lambda x: x[sort_key], reverse=True
        )

        # 严格限制买入数量（集中持仓核心约束）
        final_buy = buy_signals[:max_buy_positions]

        logger.info(
            f"信号生成完成: 候选BUY={len(buy_signals)} → 精选BUY={len(final_buy)}, "
            f"SELL={len(sell_signals)}, WATCH={len(watch_signals)}"
        )

        return final_buy + sell_signals + watch_signals[:max_watch_signals]

    def _build_quant_only_result(
        self,
        symbol: str,
        market: str,
        price: float,
        quant_score: float,
        strategy_signals: Dict,
        indicators: Dict,
        capital: float,
        reason: str = '',
    ) -> Dict[str, Any]:
        """量化初筛未通过时，返回轻量级 HOLD 结果（不调用 AI）"""
        atr = indicators.get('atr_14', price * 0.02) or price * 0.02
        return {
            'symbol': symbol,
            'market': market,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'action': 'HOLD',
            'final_score': round(quant_score, 1),
            'final_confidence': round(abs(quant_score) * 0.5, 1),
            'price': round(price, 4),
            'stop_loss': round(price - 2 * atr, 4),
            'take_profit': round(price + 2 * atr, 4),
            'position_pct': 0.0,
            'position_shares': 0,
            'position_investment': 0.0,
            'reason': reason or '量化评分不足',
            'strategy_signals': strategy_signals,
            'quant_score': round(quant_score, 1),
            'sentiment': None,
            'ai_analysis': None,
        }

    def _score_to_action(
        self,
        score: float,
        strategy_signals: Dict
    ) -> tuple:
        """
        simple 模式：将综合分数转换为动作（需至少2个策略同向）
        """
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
