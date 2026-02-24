"""
Claude LLM 智能分析模块
Claude LLM Intelligent Analysis Module

使用 Anthropic Claude API 进行：
1. 新闻情绪分析
2. 技术面综合解读
3. 交易决策辅助建议
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

import anthropic

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """
    基于 Claude 的 AI 分析器

    功能：
    - 新闻标题情绪打分（-100 到 +100）
    - 综合技术指标解读（自然语言）
    - 多策略信号融合建议
    """

    def __init__(self, model: str = "claude-opus-4-5"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("未设置 ANTHROPIC_API_KEY，AI 分析功能将不可用")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=api_key)

        self.model = model

    def analyze_news_sentiment(
        self,
        symbol: str,
        news_items: List[Dict],
        company_name: str = ""
    ) -> Dict[str, Any]:
        """
        分析新闻情绪

        Args:
            symbol: 股票代码
            news_items: 新闻列表（每条含 title, summary, published_at）
            company_name: 公司名称（可选）

        Returns:
            {
                'sentiment_score': float,  # -100 到 +100
                'sentiment_label': str,    # 'bullish'/'bearish'/'neutral'
                'key_themes': list,        # 主要主题
                'analysis': str,           # 详细分析
                'confidence': float        # AI 置信度 0-100
            }
        """
        if not self.client:
            return self._neutral_sentiment()

        if not news_items:
            return self._neutral_sentiment()

        # 构建新闻摘要
        news_text = "\n".join([
            f"[{item.get('published_at', '')}] {item.get('title', '')}: {item.get('summary', '')[:200]}"
            for item in news_items[:10]
        ])

        prompt = f"""你是一位专业的股票分析师，请分析以下关于 {company_name or symbol} ({symbol}) 的新闻情绪。

新闻内容：
{news_text}

请以 JSON 格式返回分析结果，包含以下字段：
- sentiment_score: 情绪分数（-100极度看空 到 +100极度看多，0中性）
- sentiment_label: 情绪标签（"bullish"/"bearish"/"neutral"）
- key_themes: 主要主题列表（最多3条，每条不超过15字）
- analysis: 简短分析（50-100字，中文）
- confidence: 置信度（0-100，基于新闻数量和清晰度）

只返回 JSON，不要其他文字。"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()

            # 清理可能的 markdown 包裹
            if raw.startswith('```'):
                raw = raw.split('```')[1]
                if raw.startswith('json'):
                    raw = raw[4:]

            result = json.loads(raw)
            result['sentiment_score'] = float(result.get('sentiment_score', 0))
            result['confidence'] = float(result.get('confidence', 50))
            logger.info(f"{symbol} 情绪分析完成: {result['sentiment_score']:.0f}")
            return result

        except Exception as e:
            logger.error(f"情绪分析失败 {symbol}: {e}")
            return self._neutral_sentiment()

    def analyze_technical_signals(
        self,
        symbol: str,
        strategy_signals: Dict[str, Dict],
        indicators: Dict[str, float],
        market: str = 'us'
    ) -> Dict[str, Any]:
        """
        综合分析技术信号，给出 AI 辅助建议

        Args:
            symbol: 股票代码
            strategy_signals: 各策略信号字典
            indicators: 关键技术指标值
            market: 'us' 或 'hk'

        Returns:
            {
                'recommendation': str,  # 'BUY'/'SELL'/'HOLD'/'WATCH'
                'confidence': float,    # 0-100
                'score': float,         # -100 到 +100
                'reasoning': str,       # 详细推理（中文）
                'risk_factors': list,   # 主要风险点
                'key_levels': dict,     # 关键价位
            }
        """
        if not self.client:
            return self._default_technical_result()

        # 整理各策略信号摘要
        signals_summary = []
        for strategy_name, sig in strategy_signals.items():
            signals_summary.append(
                f"- {strategy_name}: {sig.get('action','?')} "
                f"（分数 {sig.get('score',0):+.0f}，{sig.get('reason','')}）"
            )

        # 整理技术指标
        ind_summary = self._format_indicators(indicators)
        market_label = "美股" if market == 'us' else "港股"

        prompt = f"""你是一位资深量化交易分析师，精通技术分析和风险管理。
请综合分析以下 {market_label} 股票 {symbol} 的策略信号和技术指标，给出交易建议。

【各策略信号】
{chr(10).join(signals_summary)}

【关键技术指标】
{ind_summary}

请以 JSON 格式返回分析结果：
{{
  "recommendation": "BUY或SELL或HOLD或WATCH",
  "confidence": 置信度0-100,
  "score": 综合评分-100到100,
  "reasoning": "详细推理（100-150字中文，需提及信号一致性、风险）",
  "risk_factors": ["风险1（15字内）", "风险2（15字内）"],
  "key_levels": {{
    "support": 支撑价格,
    "resistance": 压力价格,
    "stop_loss": 止损建议价格
  }},
  "time_horizon": "持仓周期建议（如：1-2周、2-4周）"
}}

只返回 JSON，不要其他文字。"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=768,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()

            if raw.startswith('```'):
                raw = raw.split('```')[1]
                if raw.startswith('json'):
                    raw = raw[4:]

            result = json.loads(raw)
            result['confidence'] = float(result.get('confidence', 50))
            result['score'] = float(result.get('score', 0))
            logger.info(
                f"{symbol} AI 技术分析完成: {result.get('recommendation','?')} "
                f"({result['score']:+.0f})"
            )
            return result

        except Exception as e:
            logger.error(f"技术信号 AI 分析失败 {symbol}: {e}")
            return self._default_technical_result()

    def generate_daily_summary(
        self,
        signals: List[Dict],
        portfolio_value: float = None
    ) -> str:
        """
        生成每日交易信号摘要（自然语言报告）

        Args:
            signals: 所有股票的综合信号列表
            portfolio_value: 当前组合市值（可选）

        Returns:
            自然语言报告字符串（中文）
        """
        if not self.client:
            return "AI 功能未启用（请配置 ANTHROPIC_API_KEY）"

        if not signals:
            return "今日无交易信号。"

        buy_signals = [s for s in signals if s.get('action') == 'BUY']
        sell_signals = [s for s in signals if s.get('action') == 'SELL']

        signals_text = []
        for s in signals:
            signals_text.append(
                f"{s.get('symbol','?')}: {s.get('action','?')} "
                f"（综合分数 {s.get('final_score',0):+.0f}，"
                f"置信度 {s.get('final_confidence',0):.0f}）"
            )

        prompt = f"""你是一位专业的股票交易助手，请根据今日量化策略信号，生成一份简洁、专业的每日报告。

今日信号汇总：
{chr(10).join(signals_text)}

买入信号数量：{len(buy_signals)}
卖出信号数量：{len(sell_signals)}
{f'当前组合市值：${portfolio_value:,.0f}' if portfolio_value else ''}

请生成一份150-200字的中文日报，需包含：
1. 今日市场信号整体偏多/偏空/中性
2. 最值得关注的1-2个机会（或风险）
3. 操作建议概要（保守/中性/积极）
4. 简短的风险提示

语气专业客观，不要过于乐观，强调风险管理。"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"日报生成失败: {e}")
            return f"今日共有 {len(buy_signals)} 个买入信号，{len(sell_signals)} 个卖出信号。AI 日报生成失败，请手动查看信号详情。"

    def _format_indicators(self, indicators: Dict[str, float]) -> str:
        """格式化关键技术指标为可读文本"""
        key_inds = [
            ('Close', '当前价格'),
            ('rsi_14', 'RSI(14)'),
            ('macd_hist', 'MACD柱'),
            ('bb_pct', '布林带位置(0=下轨,1=上轨)'),
            ('adx', 'ADX趋势强度'),
            ('volume_ratio', '成交量比率'),
            ('volatility_20', '20日年化波动率'),
            ('mom_20', '20日价格涨跌幅'),
            ('sma_20', '20日均线'),
            ('sma_50', '50日均线'),
        ]
        lines = []
        for key, label in key_inds:
            val = indicators.get(key)
            if val is not None and not (isinstance(val, float) and (val != val)):
                if key in ('mom_20', 'volatility_20', 'bb_pct'):
                    lines.append(f"  {label}: {val*100:.1f}%")
                elif key in ('rsi_14', 'adx', 'volume_ratio', 'bb_pct'):
                    lines.append(f"  {label}: {val:.1f}")
                else:
                    lines.append(f"  {label}: {val:.4f}")
        return "\n".join(lines)

    def _neutral_sentiment(self) -> Dict[str, Any]:
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'key_themes': [],
            'analysis': 'AI 情绪分析不可用或无新闻数据',
            'confidence': 0.0,
        }

    def _default_technical_result(self) -> Dict[str, Any]:
        return {
            'recommendation': 'HOLD',
            'confidence': 0.0,
            'score': 0.0,
            'reasoning': 'AI 分析不可用',
            'risk_factors': [],
            'key_levels': {},
            'time_horizon': 'N/A',
        }
