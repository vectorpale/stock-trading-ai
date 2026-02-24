"""
多智能体辩论引擎
Multi-Agent Debate Engine for Investment Decisions

设计参考：
- Du et al. (ICML 2024): "Improving Factuality and Reasoning in Language Models through
  Multiagent Debate" — 3 agents × 2 rounds，每轮互读他人观点后更新立场
- TradingAgents (arXiv 2412.20138): Bull/Bear Researcher + 专家分析师 + Trader + Fund Manager
  的层级结构，以及 RiskyGoal/SafeGoal/NeutralGoal 的角色分化设计
- FinCon (NeurIPS 2024): Manager-Analyst 层级，Manager 作为唯一决策者，
  Conceptual Verbal Reinforcement 机制（自我批评 + 知识更新）

流程：
  Phase 1: 三位分析师（多头/空头/量化）独立给出初始判断
  Phase 2: 辩论轮（可选）—— 各分析师阅读对方观点并反驳/补充
  Phase 3: CIO 综合所有观点，作出最终裁决
"""

import json
import logging
from typing import Dict, Any, List, Optional

import anthropic

logger = logging.getLogger(__name__)

# ─── 分析师角色定义（参考 TradingAgents 的角色分化） ───
AGENT_PERSONAS = {
    'bull': {
        'name': '多头研究员 Alex',
        'role': '成长型多头分析师',
        'goal': 'RiskyGoal',  # TradingAgents 的目标类型
        'system': (
            "你是资深的多头研究员 Alex，专注于挖掘上涨机会和成长动能。"
            "你的职责是从数据中寻找买入理由：技术突破、动量加速、估值支撑、"
            "成交量放大等正面信号。你会用积极的视角解读数据，但论据必须基于实际数据，"
            "不能无中生有。你的目标是给出有说服力的做多理由。"
        ),
    },
    'bear': {
        'name': '空头研究员 Morgan',
        'role': '风险导向空头分析师',
        'goal': 'SafeGoal',  # TradingAgents 的目标类型
        'system': (
            "你是谨慎的空头研究员 Morgan，专注于识别风险和潜在下行因素。"
            "你的职责是质疑多头逻辑：超买信号、趋势疲软、量价背离、宏观风险、"
            "止损位置等负面信号。你会从防御角度解读数据，"
            "寻找反对做多或建议做空的理由。论据必须有数据支撑。"
        ),
    },
    'quant': {
        'name': '量化分析师 Sam',
        'role': '中性量化研究员',
        'goal': 'NeutralGoal',  # TradingAgents 的目标类型
        'system': (
            "你是严谨的量化分析师 Sam，完全依赖数学模型和统计规律，"
            "不受情绪偏见影响。你关注信号统计显著性、因子暴露、"
            "历史回测规律、风险收益比，以及各策略之间的共识程度。"
            "你给出中性、客观的数据驱动分析。"
        ),
    },
}


def _parse_json_response(raw: str) -> dict:
    """健壮地解析 LLM 返回的 JSON（处理 markdown 包裹）"""
    text = raw.strip()
    # 去掉 ```json ... ``` 包裹
    if text.startswith('```'):
        parts = text.split('```')
        # parts[0] 为空，parts[1] 为内容
        text = parts[1] if len(parts) > 1 else text
        if text.startswith('json'):
            text = text[4:]
    return json.loads(text.strip())


class DebateEngine:
    """
    多智能体投资辩论引擎

    Phase 1: 三位分析师独立分析（参考 Du et al. 的 Society of Minds）
    Phase 2: 辩论轮，互读立场并反驳（参考 Du et al. 的 Read-Critique-Update 机制）
    Phase 3: CIO 综合裁决（参考 FinCon 的 Manager-Analyst 层级）
    """

    def __init__(self, api_key: str, model: str = 'claude-opus-4-5'):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    # ──────────────────────────────────────────────
    # Phase 1: 各分析师初始判断
    # ──────────────────────────────────────────────

    def _agent_initial_analysis(
        self,
        persona_key: str,
        symbol: str,
        market: str,
        strategy_signals: Dict[str, Dict],
        indicators: Dict[str, float],
        sentiment: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        单个分析师的初始独立分析。

        参考 Du et al. 的做法：各 agent 先独立生成答案（不互看），
        再在 Phase 2 读取彼此答案并更新。
        """
        persona = AGENT_PERSONAS[persona_key]
        market_label = '美股' if market == 'us' else '港股'

        ind_text = _format_indicators(indicators)
        sig_text = _format_strategy_signals(strategy_signals)

        news_section = ''
        if sentiment and sentiment.get('sentiment_score', 0) != 0:
            news_section = (
                f"\n【新闻情绪】\n"
                f"  情绪分数: {sentiment.get('sentiment_score', 0):+.0f} "
                f"({sentiment.get('sentiment_label', 'neutral')})\n"
                f"  主要主题: {', '.join(sentiment.get('key_themes', []))}\n"
                f"  情绪摘要: {sentiment.get('analysis', '')}"
            )

        prompt = (
            f"你正在分析 {market_label} 股票 **{symbol}**。\n\n"
            f"【技术指标】\n{ind_text}\n\n"
            f"【量化策略信号】\n{sig_text}"
            f"{news_section}\n\n"
            "请从你的角色视角给出独立判断，以 JSON 格式返回：\n"
            "{\n"
            '  "position": "BUY 或 SELL 或 HOLD 或 WATCH",\n'
            '  "confidence": 0到100的置信度,\n'
            '  "score": -100到100的综合评分（正数看多，负数看空）,\n'
            '  "core_argument": "你最核心的论点（50字以内）",\n'
            '  "arguments": ["支持你立场的论据1", "论据2", "论据3"],\n'
            '  "key_risk": "你认为最大的风险或不确定性（30字以内）"\n'
            "}\n\n"
            "只返回 JSON，不要其他文字。"
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=persona['system'],
                messages=[{"role": "user", "content": prompt}]
            )
            result = _parse_json_response(resp.content[0].text)
            result['agent_key'] = persona_key
            result['agent_name'] = persona['name']
            result['agent_role'] = persona['role']
            result['score'] = float(result.get('score', 0))
            result['confidence'] = float(result.get('confidence', 50))
            logger.info(
                f"  [{persona['name']}] {result.get('position','?')} "
                f"score={result['score']:+.0f} conf={result['confidence']:.0f}"
            )
            return result
        except Exception as e:
            logger.warning(f"分析师 {persona_key} 初始分析失败: {e}")
            return _default_agent_result(persona_key)

    # ──────────────────────────────────────────────
    # Phase 2: 辩论轮（Read-Critique-Update）
    # ──────────────────────────────────────────────

    def _agent_rebuttal(
        self,
        persona_key: str,
        own_analysis: Dict,
        others: Dict[str, Dict],
    ) -> str:
        """
        参考 Du et al. 的 Read-Critique-Update 机制：
        每位分析师读取他人立场，给出针对性反驳或补充。
        限制在 120 字以内，保持辩论的锐度。
        """
        persona = AGENT_PERSONAS[persona_key]

        other_views = '\n'.join([
            f"- **{v['agent_name']}**（{v['position']}，{v['confidence']:.0f}分）："
            f"{v['core_argument']}"
            for k, v in others.items()
            if k != persona_key
        ])

        prompt = (
            f"你之前的立场是：**{own_analysis['position']}**\n"
            f"核心论点：{own_analysis['core_argument']}\n\n"
            f"其他分析师的观点：\n{other_views}\n\n"
            "请在 120 字以内，针对他们的论点给出你的反驳或关键补充。"
            "直接切入要点，不要重复你之前的所有论据。只返回纯文本。"
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=persona['system'],
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text.strip()
        except Exception as e:
            logger.warning(f"分析师 {persona_key} 辩论轮失败: {e}")
            return ''

    # ──────────────────────────────────────────────
    # Phase 3: CIO 最终裁决（FinCon Manager 角色）
    # ──────────────────────────────────────────────

    def _cio_decision(
        self,
        symbol: str,
        market: str,
        price: float,
        agents: Dict[str, Dict],
        rebuttals: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        CIO 综合裁决。

        参考 FinCon 的 Manager-Analyst 层级：
        - Manager（CIO）是唯一的最终决策者
        - 明确评估各方论据的说服力（Conceptual Verbal Reinforcement）
        - 给出可执行的交易决策，包含风险控制参数
        """
        market_label = '美股' if market == 'us' else '港股'

        # 构建辩论记录
        debate_transcript = []
        for key in ['bull', 'bear', 'quant']:
            if key not in agents:
                continue
            a = agents[key]
            section = (
                f"**{a['agent_name']}**（{a['role']}）\n"
                f"  立场: {a.get('position','?')} | "
                f"评分: {a.get('score', 0):+.0f} | "
                f"置信度: {a.get('confidence', 0):.0f}\n"
                f"  核心论点: {a.get('core_argument', '')}\n"
                f"  详细论据: {' | '.join(a.get('arguments', []))}\n"
                f"  主要风险: {a.get('key_risk', '')}"
            )
            if rebuttals and key in rebuttals and rebuttals[key]:
                section += f"\n  辩论补充: {rebuttals[key]}"
            debate_transcript.append(section)

        # 计算三位分析师的平均评分（作为参考基准）
        scores = [agents[k].get('score', 0) for k in agents]
        avg_score = sum(scores) / len(scores) if scores else 0

        cio_system = (
            "你是首席投资官（CIO），负责综合所有分析师的辩论意见并作出最终投资裁决。"
            "你的决策必须：\n"
            "1. 明确评估每位分析师论据的说服力，不能简单取平均\n"
            "2. 识别多空双方论据中的关键分歧点\n"
            "3. 给出清晰可执行的最终决策及其理由\n"
            "4. 严格控制风险，给出具体的仓位和止损建议\n"
            "你的决策风格：在论据充分时果断，在分歧明显时保守。"
        )

        prompt = (
            f"请对 {market_label} 股票 **{symbol}**（当前价格: {price:.2f}）"
            "作出最终投资裁决。\n\n"
            "以下是三位分析师经过辩论后的完整意见：\n\n"
            + '\n\n'.join(debate_transcript)
            + f"\n\n三方平均评分: {avg_score:+.1f}\n\n"
            "请以 JSON 格式给出你的最终裁决：\n"
            "{\n"
            '  "action": "BUY 或 SELL 或 HOLD 或 WATCH",\n'
            '  "confidence": 0到100,\n'
            '  "score": -100到100（你的综合评分，不必等于平均分）,\n'
            '  "verdict": "一句话裁决（60字以内）",\n'
            '  "reasoning": "详细决策推理（120-150字）：说明你如何权衡各方，'
            '哪方论据更有说服力，为何做出此决定",\n'
            '  "key_disagreement": "多空方最关键的分歧点（40字以内）",\n'
            '  "adopted_views": ["主要采纳了哪位分析师的观点，简述原因"],\n'
            '  "risk_factors": ["最重要的风险1", "风险2"],\n'
            '  "position_sizing": {\n'
            '    "suggested_pct": 0到15（建议仓位百分比），\n'
            '    "stop_loss_pct": 止损幅度如0.07,\n'
            '    "take_profit_pct": 目标收益幅度如0.20\n'
            "  },\n"
            '  "time_horizon": "建议持仓周期"\n'
            "}\n\n"
            "只返回 JSON，不要其他文字。"
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=cio_system,
                messages=[{"role": "user", "content": prompt}]
            )
            result = _parse_json_response(resp.content[0].text)
            result['score'] = float(result.get('score', avg_score))
            result['confidence'] = float(result.get('confidence', 50))
            logger.info(
                f"  [CIO] 最终裁决: {result.get('action','?')} "
                f"score={result['score']:+.0f}"
            )
            return result
        except Exception as e:
            logger.error(f"CIO 裁决失败: {e}")
            return _default_cio_result()

    # ──────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────

    def run_debate(
        self,
        symbol: str,
        market: str,
        price: float,
        strategy_signals: Dict[str, Dict],
        indicators: Dict[str, float],
        sentiment: Optional[Dict] = None,
        debate_rounds: int = 1,
    ) -> Dict[str, Any]:
        """
        执行完整辩论流程。

        Args:
            symbol: 股票代码
            market: 'us' 或 'hk'
            price: 当前价格（供 CIO 参考）
            strategy_signals: 量化策略信号字典
            indicators: 技术指标字典
            sentiment: 新闻情绪分析结果（可选）
            debate_rounds: 辩论轮数（0=仅初始判断+CIO, 1=加辩论轮）

        Returns:
            {
                'agents': {               # Phase 1 各分析师初始判断
                    'bull': {...},
                    'bear': {...},
                    'quant': {...},
                },
                'rebuttals': {            # Phase 2 辩论轮（debate_rounds>0 时）
                    'bull': '...',
                    'bear': '...',
                    'quant': '...',
                },
                'cio_decision': {         # Phase 3 CIO 最终裁决
                    'action': str,
                    'confidence': float,
                    'score': float,
                    'verdict': str,
                    'reasoning': str,
                    ...
                },
                'debate_rounds': int,
                'agent_consensus': str,   # 三方是否达成共识
                'avg_agent_score': float, # 三方平均评分
            }
        """
        logger.info(f"[辩论] 开始分析 {symbol}（{debate_rounds} 轮辩论）...")

        # ── Phase 1: 独立初始分析 ──
        logger.info("[辩论] Phase 1: 各分析师独立给出初始判断...")
        agents: Dict[str, Dict] = {}
        for key in ['bull', 'bear', 'quant']:
            agents[key] = self._agent_initial_analysis(
                key, symbol, market, strategy_signals, indicators, sentiment
            )

        # ── Phase 2: 辩论轮（可选）──
        rebuttals: Dict[str, str] = {}
        if debate_rounds > 0:
            logger.info("[辩论] Phase 2: 辩论轮，各方互读立场并反驳...")
            for key in ['bull', 'bear', 'quant']:
                rebuttals[key] = self._agent_rebuttal(key, agents[key], agents)

        # ── Phase 3: CIO 最终裁决 ──
        logger.info("[辩论] Phase 3: CIO 综合裁决...")
        cio = self._cio_decision(
            symbol, market, price, agents,
            rebuttals if debate_rounds > 0 else None
        )

        # ── 辅助统计 ──
        positions = [agents[k].get('position', 'HOLD') for k in agents]
        avg_score = sum(agents[k].get('score', 0) for k in agents) / len(agents)
        buy_count = positions.count('BUY')
        sell_count = positions.count('SELL')
        if buy_count == 3:
            consensus = '三方看多'
        elif sell_count == 3:
            consensus = '三方看空'
        elif buy_count >= 2:
            consensus = '多数看多'
        elif sell_count >= 2:
            consensus = '多数看空'
        else:
            consensus = '意见分歧'

        return {
            'agents': agents,
            'rebuttals': rebuttals,
            'cio_decision': cio,
            'debate_rounds': debate_rounds,
            'agent_consensus': consensus,
            'avg_agent_score': round(avg_score, 1),
        }


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _format_indicators(indicators: Dict[str, float]) -> str:
    """格式化技术指标为可读文本"""
    mapping = [
        ('Close',        '当前价格',         '{:.2f}'),
        ('rsi_14',       'RSI(14)',           '{:.1f}'),
        ('rsi_6',        'RSI(6)',            '{:.1f}'),
        ('macd_hist',    'MACD柱状值',        '{:.4f}'),
        ('bb_pct',       '布林带位置(0下轨/1上轨)', '{:.2f}'),
        ('adx',          'ADX趋势强度',       '{:.1f}'),
        ('volume_ratio', '量比',              '{:.2f}'),
        ('volatility_20','年化波动率',        '{:.1%}'),
        ('mom_20',       '20日涨跌幅',        '{:.1%}'),
        ('mom_5',        '5日涨跌幅',         '{:.1%}'),
        ('sma_20',       'SMA(20)',           '{:.2f}'),
        ('sma_50',       'SMA(50)',           '{:.2f}'),
        ('sma_200',      'SMA(200)',          '{:.2f}'),
    ]
    lines = []
    for key, label, fmt in mapping:
        val = indicators.get(key)
        if val is not None and not (isinstance(val, float) and val != val):
            try:
                lines.append(f"  {label}: {fmt.format(float(val))}")
            except Exception:
                pass
    return '\n'.join(lines) if lines else '指标数据不足'


def _format_strategy_signals(strategy_signals: Dict[str, Dict]) -> str:
    """格式化量化策略信号"""
    names = {
        'momentum': '动量策略',
        'mean_reversion': '均值回归',
        'multi_factor': '多因子',
    }
    lines = []
    for key, label in names.items():
        if key in strategy_signals:
            sig = strategy_signals[key]
            reason = sig.get('reason', '')[:60]
            lines.append(
                f"  {label}: {sig.get('action', 'N/A')} "
                f"(评分: {sig.get('score', 0):+.1f}) — {reason}"
            )
    return '\n'.join(lines) if lines else '无策略信号'


def _default_agent_result(persona_key: str) -> Dict[str, Any]:
    persona = AGENT_PERSONAS.get(persona_key, {})
    return {
        'agent_key': persona_key,
        'agent_name': persona.get('name', persona_key),
        'agent_role': persona.get('role', ''),
        'position': 'HOLD',
        'confidence': 0.0,
        'score': 0.0,
        'core_argument': '分析失败，无法获取结果',
        'arguments': [],
        'key_risk': '分析不可用',
    }


def _default_cio_result() -> Dict[str, Any]:
    return {
        'action': 'HOLD',
        'confidence': 0.0,
        'score': 0.0,
        'verdict': 'CIO 裁决失败',
        'reasoning': '辩论引擎遇到错误，无法完成分析',
        'key_disagreement': '未知',
        'adopted_views': [],
        'risk_factors': [],
        'position_sizing': {
            'suggested_pct': 0,
            'stop_loss_pct': 0.07,
            'take_profit_pct': 0.20,
        },
        'time_horizon': '未知',
    }
