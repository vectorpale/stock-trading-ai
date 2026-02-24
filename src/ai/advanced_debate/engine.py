"""
高级多智能体辩论引擎 - Advanced Multi-Agent Debate Engine
Version 2.0

算法设计参考 / Algorithm References:
  [1] Du et al. (ICML 2024, arXiv:2305.14325)
      "Improving Factuality and Reasoning in Language Models through Multiagent Debate"
      — 核心机制：Society of Minds 独立初始化 + Read-Critique-Update 迭代辩论
      — 收敛判断：当各方立场差距低于阈值时提前停止

  [2] Xiao et al. (arXiv:2412.20138, TradingAgents)
      "TradingAgents: Multi-Agents LLM Financial Trading Framework"
      — 层级结构：Bull/Bear Researcher → Expert Analysts → Trader → Fund Manager
      — 角色分化：RiskyGoal / SafeGoal / NeutralGoal

  [3] Yu et al. (NeurIPS 2024, arXiv:2407.06567, FinCon)
      "FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement"
      — Manager-Analyst 层级；CVR（概念化语言强化）机制
      — Manager 作为唯一决策者，Analyst 提供多元化论据

  [4] Zhang et al. (2024, arXiv:2402.18485, FinAgent)
      "FinAgent: A Multimodal Foundation Agent for Financial Trading"
      — 多模态多维度分析框架（技术面 + 基本面 + 宏观面融合）

  [5] Shinn et al. (NeurIPS 2023, arXiv:2303.11366, Reflexion, ~1520 引用)
      "Reflexion: Language Agents with Verbal Reinforcement Learning"
      — 通过语言形式的反思从历史失败中学习（无需梯度更新）

  [6] Gou et al. (ICLR 2024, arXiv:2305.11738, CRITIC, ~500 引用)
      "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing"
      — LLM 与工具交互式自我批评，迭代检测并修正错误

流程：
  Phase 1  独立初始分析  [Du et al. Society of Minds]
           各分析师独立给出初始判断（互不可见）
           注入历史记忆上下文 [FinMem / Reflexion]

  Phase 2  辩论轮（可选，支持多轮）[Du et al. Read-Critique-Update]
           各方读取他人论点，给出针对性反驳/补充
           轮后计算收敛度，满足阈值则提前停止

  Phase 3  风险委员会审查 [FinCon 风险层]
           风险官对辩论结果进行风险评估，可触发一票否决

  Phase 4  CIO 综合裁决 [FinCon Manager + TradingAgents Fund Manager]
           加权汇总各方意见，作出最终交易决策
           决策自动存入记忆库 [FinMem]
"""

import logging
from typing import Dict, Any, List, Optional

import anthropic

from .agents import AGENT_PERSONAS, CIO_SYSTEM_PROMPT
from .memory import DecisionMemory
from .utils import (
    parse_json_response,
    format_indicators,
    format_strategy_signals,
    compute_consensus,
    compute_convergence_score,
    weighted_score_aggregation,
)

logger = logging.getLogger(__name__)


class AdvancedDebateEngine:
    """
    高级多智能体辩论引擎 v2.0

    相比 v1 的改进：
    - 5 个智能体角色（新增 Macro + Risk，基于 FinAgent 多维度框架）
    - 多轮辩论 + 收敛检测（Du et al. 2024）
    - 风险否决机制（FinCon 风险层）
    - 分层记忆注入（FinMem + Reflexion）
    - 置信度加权聚合（TradingAgents 层级加权）
    - 分角色模型：CIO 使用 Opus，分析师使用 Sonnet，降低成本
    """

    def __init__(
        self,
        api_key: str,
        model: str = 'claude-sonnet-4-6',           # 分析师默认模型
        model_cio: str = 'claude-opus-4-6',          # CIO 使用更强模型
        memory: Optional[DecisionMemory] = None,
        convergence_threshold: float = 0.85,
        max_debate_rounds: int = 3,
    ):
        """
        Args:
            api_key: Anthropic API Key
            model: 各分析师使用的模型（Phase 1/2/3）
            model_cio: CIO 最终裁决使用的模型（Phase 4，建议 Opus）
            memory: 决策记忆模块（可选，默认创建新实例）
            convergence_threshold: 辩论收敛阈值（超过则停止辩论）
            max_debate_rounds: 最大辩论轮数上限
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model              # 分析师模型（Sonnet，性价比高）
        self.model_cio = model_cio      # CIO 模型（Opus，最强推理）
        self.memory = memory or DecisionMemory()
        self.convergence_threshold = convergence_threshold
        self.max_debate_rounds = max_debate_rounds

    # ══════════════════════════════════════════════════════════
    # Phase 1: 独立初始分析（Society of Minds）
    # ══════════════════════════════════════════════════════════

    def _agent_initial_analysis(
        self,
        persona_key: str,
        symbol: str,
        market: str,
        strategy_signals: Dict[str, Dict],
        indicators: Dict[str, float],
        sentiment: Optional[Dict] = None,
        memory_context: str = '',
    ) -> Dict[str, Any]:
        """
        单个智能体的独立初始分析。

        参考 Du et al. (2024) Society of Minds：
        各 agent 先独立生成答案（互不可见），避免锚定效应。
        历史记忆上下文注入参考 FinMem (Yu et al. 2024) 的记忆检索机制。
        """
        persona = AGENT_PERSONAS[persona_key]
        market_label = '美股' if market == 'us' else ('港股' if market == 'hk' else 'A股')

        ind_text = format_indicators(indicators)
        sig_text = format_strategy_signals(strategy_signals)

        # 新闻情绪部分（可选）
        news_section = ''
        if sentiment and sentiment.get('sentiment_score', 0) != 0:
            news_section = (
                f"\n【新闻情绪】\n"
                f"  情绪分数: {sentiment.get('sentiment_score', 0):+.0f} "
                f"({sentiment.get('sentiment_label', 'neutral')})\n"
                f"  主要主题: {', '.join(sentiment.get('key_themes', []))}\n"
                f"  摘要: {sentiment.get('analysis', '')}"
            )

        # 历史记忆上下文注入（FinMem 机制）
        memory_section = ''
        if memory_context:
            memory_section = f"\n{memory_context}\n"

        # 宏观/风险分析师特殊视角引导
        role_specific_prompt = _get_role_specific_prompt(persona_key)

        prompt = (
            f"你正在分析 {market_label} 股票 **{symbol}**。\n\n"
            f"【技术指标】\n{ind_text}\n\n"
            f"【量化策略信号】\n{sig_text}"
            f"{news_section}"
            f"{memory_section}"
            f"{role_specific_prompt}\n\n"
            "请从你的角色视角给出独立判断，以 JSON 格式返回：\n"
            "{\n"
            '  "position": "BUY 或 SELL 或 HOLD 或 WATCH",\n'
            '  "confidence": 0到100的置信度,\n'
            '  "score": -100到100的综合评分（正数看多，负数看空）,\n'
            '  "core_argument": "你最核心的论点（60字以内）",\n'
            '  "arguments": ["支持你立场的论据1", "论据2", "论据3"],\n'
            '  "key_risk": "最大的风险或不确定性（40字以内）",\n'
            '  "suggested_position_pct": 0到15（建议仓位，基于你的分析）\n'
            "}\n\n"
            "只返回 JSON，不要其他文字。"
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                system=persona['system'],
                messages=[{"role": "user", "content": prompt}]
            )
            result = parse_json_response(resp.content[0].text)
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

    # ══════════════════════════════════════════════════════════
    # Phase 2: 辩论轮（Read-Critique-Update）
    # ══════════════════════════════════════════════════════════

    def _agent_rebuttal(
        self,
        persona_key: str,
        own_analysis: Dict,
        others: Dict[str, Dict],
        round_num: int = 1,
    ) -> Dict[str, Any]:
        """
        辩论轮：每个智能体读取他人立场，更新自己的判断。

        参考 Du et al. (2024) Read-Critique-Update 机制：
        1. Read: 读取所有其他 agent 的论点
        2. Critique: 识别他人论点的优缺点
        3. Update: 更新自己的立场和置信度（允许被说服）
        """
        persona = AGENT_PERSONAS[persona_key]

        other_views = '\n'.join([
            f"- **{v['agent_name']}**（{v.get('position','?')}，"
            f"评分{v.get('score',0):+.0f}，置信度{v.get('confidence',50):.0f}）：\n"
            f"  论点: {v.get('core_argument','')}\n"
            f"  风险提示: {v.get('key_risk','')}"
            for k, v in others.items()
            if k != persona_key
        ])

        prompt = (
            f"第 {round_num} 轮辩论。你之前的立场：**{own_analysis['position']}**\n"
            f"你的核心论点：{own_analysis['core_argument']}\n\n"
            f"其他分析师的观点如下：\n{other_views}\n\n"
            "请以 JSON 格式返回更新后的判断：\n"
            "{\n"
            '  "position": "更新后的立场（可与之前相同或改变）",\n'
            '  "confidence": 更新后的置信度0-100,\n'
            '  "score": 更新后的评分-100到100,\n'
            '  "rebuttal": "对其他方论点的反驳或补充（80字以内）",\n'
            '  "position_changed": true或false（立场是否改变）,\n'
            '  "why_changed": "如果立场改变，简述原因（否则填null）"\n'
            "}\n\n"
            "只返回 JSON，不要其他文字。"
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=persona['system'],
                messages=[{"role": "user", "content": prompt}]
            )
            result = parse_json_response(resp.content[0].text)
            result['agent_key'] = persona_key
            result['agent_name'] = persona['name']
            result['score'] = float(result.get('score', own_analysis.get('score', 0)))
            result['confidence'] = float(result.get('confidence', own_analysis.get('confidence', 50)))

            if result.get('position_changed'):
                logger.info(
                    f"  [{persona['name']}] 立场更新: "
                    f"{own_analysis['position']} → {result.get('position','?')}"
                )
            return result
        except Exception as e:
            logger.warning(f"分析师 {persona_key} 辩论轮 {round_num} 失败: {e}")
            return {
                'agent_key': persona_key,
                'agent_name': AGENT_PERSONAS[persona_key]['name'],
                'position': own_analysis.get('position', 'HOLD'),
                'score': own_analysis.get('score', 0),
                'confidence': own_analysis.get('confidence', 50),
                'rebuttal': '',
                'position_changed': False,
                'why_changed': None,
            }

    # ══════════════════════════════════════════════════════════
    # Phase 3: 风险委员会审查（FinCon 风险层）
    # ══════════════════════════════════════════════════════════

    def _risk_committee_review(
        self,
        symbol: str,
        market: str,
        price: float,
        agents: Dict[str, Dict],
        weighted_score: float,
    ) -> Dict[str, Any]:
        """
        风险委员会审查。

        参考 FinCon 的风险管理层：Risk Manager 作为 Manager 决策前的最后防线。
        风险官（已在 agents['risk'] 中）评估整体方案的可行性，
        可返回 veto（一票否决）或 approve（批准，带条件）。
        """
        risk_agent = agents.get('risk', {})
        market_label = '美股' if market == 'us' else ('港股' if market == 'hk' else 'A股')

        # 聚合其他 agent 的方向
        other_agents_summary = '\n'.join([
            f"- {v.get('agent_name','?')}：{v.get('position','?')} "
            f"(评分{v.get('score',0):+.0f}, 置信度{v.get('confidence',50):.0f})"
            for k, v in agents.items() if k != 'risk'
        ])

        prompt = (
            f"风险审查：{market_label} 股票 **{symbol}**（当前价格: {price:.2f}）\n\n"
            f"各分析师当前方向：\n{other_agents_summary}\n\n"
            f"加权综合评分: {weighted_score:+.1f}\n"
            f"你自己的风险分析：{risk_agent.get('core_argument', '待评估')}\n\n"
            "请以 JSON 格式给出风险委员会审查结论：\n"
            "{\n"
            '  "verdict": "APPROVE 或 APPROVE_WITH_CONDITIONS 或 VETO",\n'
            '  "risk_level": "LOW 或 MEDIUM 或 HIGH 或 EXTREME",\n'
            '  "max_position_pct": 建议最大仓位0-15,\n'
            '  "mandatory_stop_loss_pct": 强制止损幅度如0.07,\n'
            '  "conditions": ["如果批准但有条件，列出风险控制条件"],\n'
            '  "veto_reason": "如果否决，填否决理由（否则填null）"\n'
            "}\n\n"
            "只返回 JSON。若风险极高（EXTREME）或流动性不足，使用 VETO。"
        )

        risk_system = AGENT_PERSONAS['risk']['system']

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=risk_system,
                messages=[{"role": "user", "content": prompt}]
            )
            result = parse_json_response(resp.content[0].text)
            logger.info(
                f"  [风险委员会] {result.get('verdict','?')} "
                f"risk_level={result.get('risk_level','?')}"
            )
            return result
        except Exception as e:
            logger.warning(f"风险委员会审查失败: {e}")
            return {
                'verdict': 'APPROVE_WITH_CONDITIONS',
                'risk_level': 'MEDIUM',
                'max_position_pct': 8,
                'mandatory_stop_loss_pct': 0.07,
                'conditions': ['风险审查模块异常，建议保守操作'],
                'veto_reason': None,
            }

    # ══════════════════════════════════════════════════════════
    # Phase 4: CIO 最终裁决（FinCon Manager）
    # ══════════════════════════════════════════════════════════

    def _cio_decision(
        self,
        symbol: str,
        market: str,
        price: float,
        agents: Dict[str, Dict],
        rebuttals_history: List[Dict[str, Dict]],
        risk_review: Dict[str, Any],
        weighted_score: float,
        reflection_context: str = '',
    ) -> Dict[str, Any]:
        """
        CIO 最终投资裁决。

        参考 FinCon Manager + TradingAgents Fund Manager：
        - CIO 是唯一的最终决策者
        - 采用 CVR（概念化语言强化）机制：明确评估各方论据说服力
        - 风险委员会的硬约束（VETO / 最大仓位）必须遵守
        - 反射层历史教训注入（Reflexion）
        """
        market_label = '美股' if market == 'us' else ('港股' if market == 'hk' else 'A股')

        # 构建完整辩论记录
        debate_transcript = []
        for key in AGENT_PERSONAS:
            if key not in agents:
                continue
            a = agents[key]
            section = (
                f"**{a['agent_name']}**（{a['agent_role']}）\n"
                f"  最终立场: {a.get('position','?')} | "
                f"评分: {a.get('score', 0):+.0f} | "
                f"置信度: {a.get('confidence', 0):.0f}\n"
                f"  核心论点: {a.get('core_argument', '')}\n"
                f"  风险提示: {a.get('key_risk', '')}"
            )
            debate_transcript.append(section)

        # 附加辩论轮摘要
        rounds_summary = ''
        if rebuttals_history:
            rounds_summary = f"\n辩论共进行了 {len(rebuttals_history)} 轮。"
            position_changes = sum(
                1 for round_rebuttals in rebuttals_history
                for key, r in round_rebuttals.items()
                if r.get('position_changed', False)
            )
            if position_changes > 0:
                rounds_summary += f" 过程中有 {position_changes} 次立场更新（说明辩论有实质分歧）。"

        # 风险委员会约束
        risk_constraints = (
            f"\n【风险委员会裁定】\n"
            f"  审查结论: {risk_review.get('verdict','?')}\n"
            f"  风险等级: {risk_review.get('risk_level','?')}\n"
            f"  最大允许仓位: {risk_review.get('max_position_pct', 10)}%\n"
            f"  强制止损: {risk_review.get('mandatory_stop_loss_pct', 0.07):.0%}"
        )
        if risk_review.get('verdict') == 'VETO':
            risk_constraints += f"\n  否决理由: {risk_review.get('veto_reason','')}"
        if risk_review.get('conditions'):
            risk_constraints += f"\n  条件: {'; '.join(risk_review.get('conditions', []))}"

        # 历史反思注入（Reflexion）
        reflection_section = ''
        if reflection_context:
            reflection_section = f"\n{reflection_context}\n"

        prompt = (
            f"请对 {market_label} 股票 **{symbol}**（当前价格: {price:.2f}）"
            f"作出最终投资裁决。\n\n"
            "【各分析师最终立场】\n"
            + '\n\n'.join(debate_transcript)
            + f"\n\n加权综合评分（置信度加权）: {weighted_score:+.1f}"
            + rounds_summary
            + risk_constraints
            + reflection_section
            + "\n\n"
            "注意：\n"
            "1. 若风险委员会 VETO，则 action 必须为 HOLD，position_pct 为 0\n"
            "2. position_pct 不得超过风险委员会设定的最大仓位\n"
            "3. stop_loss_pct 不得低于风险委员会要求的强制止损\n\n"
            "请以 JSON 格式给出最终裁决：\n"
            "{\n"
            '  "action": "BUY 或 SELL 或 HOLD 或 WATCH",\n'
            '  "confidence": 0到100,\n'
            '  "score": -100到100（你的综合评分）,\n'
            '  "verdict": "一句话裁决（70字以内）",\n'
            '  "reasoning": "详细决策推理（120-160字）：如何权衡各方，哪方论据更有说服力",\n'
            '  "key_disagreement": "辩论中最关键的分歧点（40字以内）",\n'
            '  "adopted_views": ["主要采纳了哪位分析师的观点，简述原因"],\n'
            '  "risk_factors": ["最重要的风险1", "风险2", "风险3"],\n'
            '  "position_sizing": {\n'
            '    "suggested_pct": 0到max_position_pct（建议仓位，受风险委员会约束）,\n'
            '    "stop_loss_pct": 止损幅度（不得低于强制止损）,\n'
            '    "take_profit_pct": 目标收益幅度\n'
            "  },\n"
            '  "time_horizon": "建议持仓周期"\n'
            "}\n\n"
            "只返回 JSON，不要其他文字。"
        )

        try:
            resp = self.client.messages.create(
                model=self.model_cio,   # CIO 使用更强模型（Opus）
                max_tokens=1200,
                system=CIO_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            result = parse_json_response(resp.content[0].text)
            result['score'] = float(result.get('score', weighted_score))
            result['confidence'] = float(result.get('confidence', 50))

            # 强制遵守风险委员会否决
            if risk_review.get('verdict') == 'VETO':
                result['action'] = 'HOLD'
                result['position_sizing'] = {
                    'suggested_pct': 0,
                    'stop_loss_pct': risk_review.get('mandatory_stop_loss_pct', 0.07),
                    'take_profit_pct': 0.0,
                }
                result['verdict'] = f"[风险否决] {result.get('verdict', '')}"

            logger.info(
                f"  [CIO] 最终裁决: {result.get('action','?')} "
                f"score={result['score']:+.0f} conf={result['confidence']:.0f}"
            )
            return result
        except Exception as e:
            logger.error(f"CIO 裁决失败: {e}")
            return _default_cio_result()

    # ══════════════════════════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════════════════════════

    def run_debate(
        self,
        symbol: str,
        market: str,
        price: float,
        strategy_signals: Dict[str, Dict],
        indicators: Dict[str, float],
        sentiment: Optional[Dict] = None,
        debate_rounds: int = 1,
        store_to_memory: bool = True,
    ) -> Dict[str, Any]:
        """
        执行完整辩论流程（Phase 1 → Phase 2 → Phase 3 → Phase 4）。

        Args:
            symbol: 股票代码
            market: 'us' / 'hk' / 'cn'
            price: 当前价格
            strategy_signals: 量化策略信号字典（来自三个量化策略）
            indicators: 技术指标字典
            sentiment: 新闻情绪分析结果（可选）
            debate_rounds: 期望辩论轮数（0=仅初始+CIO，实际轮数受收敛控制）
            store_to_memory: 是否将结果存入记忆库

        Returns:
            完整辩论结果字典（含 agents、rebuttals、risk_review、cio_decision 等）
        """
        logger.info(f"[辩论] 开始分析 {symbol}（最多 {debate_rounds} 轮辩论）...")

        # 从记忆库检索历史上下文（FinMem 机制）
        memory_context = self.memory.get_context_for_debate(symbol)
        reflection_context = self.memory.get_reflection_summary(symbol)
        if memory_context:
            logger.info(f"  [Memory] 注入历史记忆上下文 ({len(memory_context)} 字)")

        # ── Phase 1: 独立初始分析 ─────────────────────────────
        logger.info("[辩论] Phase 1: 各分析师独立给出初始判断...")
        agents: Dict[str, Dict] = {}
        for key in AGENT_PERSONAS:
            agents[key] = self._agent_initial_analysis(
                key, symbol, market, strategy_signals, indicators,
                sentiment, memory_context
            )

        # ── Phase 2: 辩论轮（多轮 + 收敛检测）──────────────────
        rebuttals_history: List[Dict[str, Dict]] = []
        actual_rounds = 0

        for round_num in range(1, min(debate_rounds, self.max_debate_rounds) + 1):
            # 收敛检测（Du et al. 2024 提前停止机制）
            convergence = compute_convergence_score(agents)
            if convergence >= self.convergence_threshold:
                logger.info(
                    f"  [辩论] 第 {round_num} 轮前检测到收敛 "
                    f"(convergence={convergence:.2f})，停止辩论"
                )
                break

            logger.info(
                f"[辩论] Phase 2 Round {round_num}: "
                f"Read-Critique-Update (convergence={convergence:.2f})..."
            )
            round_rebuttals: Dict[str, Dict] = {}
            for key in AGENT_PERSONAS:
                if key not in agents:
                    continue
                rebuttal = self._agent_rebuttal(key, agents[key], agents, round_num)
                round_rebuttals[key] = rebuttal
                # 用辩论结果更新 agents（立场可能改变）
                agents[key]['position'] = rebuttal.get('position', agents[key]['position'])
                agents[key]['score'] = rebuttal.get('score', agents[key]['score'])
                agents[key]['confidence'] = rebuttal.get('confidence', agents[key]['confidence'])

            rebuttals_history.append(round_rebuttals)
            actual_rounds = round_num

        # ── 加权评分聚合（TradingAgents 层级加权）────────────────
        weighted_score = weighted_score_aggregation(agents)
        agent_consensus = compute_consensus(agents)
        convergence_final = compute_convergence_score(agents)

        # ── Phase 3: 风险委员会审查 ───────────────────────────
        logger.info("[辩论] Phase 3: 风险委员会审查...")
        risk_review = self._risk_committee_review(symbol, market, price, agents, weighted_score)

        # ── Phase 4: CIO 最终裁决 ────────────────────────────
        logger.info("[辩论] Phase 4: CIO 综合裁决...")
        cio = self._cio_decision(
            symbol, market, price, agents,
            rebuttals_history, risk_review, weighted_score, reflection_context
        )

        result = {
            'agents': agents,
            'rebuttals_history': rebuttals_history,
            'cio_decision': cio,
            'risk_review': risk_review,
            'actual_debate_rounds': actual_rounds,
            'convergence_score': convergence_final,
            'agent_consensus': agent_consensus,
            'weighted_score': round(weighted_score, 1),
        }

        # ── 存入记忆库（FinMem）─────────────────────────────
        if store_to_memory:
            self.memory.store_decision(symbol, market, result, indicators, price)

        return result


# ─── 辅助函数 ──────────────────────────────────────────────────

def _get_role_specific_prompt(persona_key: str) -> str:
    """为不同角色提供额外的分析角度引导"""
    prompts = {
        'bull': "\n重点关注：近期是否有技术突破、动量加速、成交量配合等正面信号。",
        'bear': "\n重点关注：是否存在超买、顶背离、量能萎缩、支撑破位等警示信号。",
        'quant': "\n重点关注：各策略信号的一致性、统计显著性和历史胜率。",
        'macro': (
            "\n重点关注：当前宏观环境（利率/美元/VIX）对本股票板块的影响，"
            "该股票处于顺风还是逆风周期。"
        ),
        'risk': (
            "\n重点关注：当前价格的风险收益比、合理止损位置、最大可承受损失。"
            "如果各项风险指标超过警戒线，直接建议 HOLD 或 WATCH。"
        ),
    }
    return prompts.get(persona_key, '')


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
        'suggested_position_pct': 0,
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
