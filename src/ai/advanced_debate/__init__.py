"""
高级多智能体辩论引擎包
Advanced Multi-Agent Debate Engine Package

5智能体 + 分层记忆 + 风险委员会 + CIO 裁决
"""

from .engine import AdvancedDebateEngine
from .memory import DecisionMemory
from .agents import AGENT_PERSONAS, CIO_SYSTEM_PROMPT
from .utils import (
    parse_json_response,
    format_indicators,
    format_strategy_signals,
    compute_consensus,
    compute_convergence_score,
    weighted_score_aggregation,
    kelly_position_size,
)

__all__ = [
    'AdvancedDebateEngine',
    'DecisionMemory',
    'AGENT_PERSONAS',
    'CIO_SYSTEM_PROMPT',
    'parse_json_response',
    'format_indicators',
    'format_strategy_signals',
    'compute_consensus',
    'compute_convergence_score',
    'weighted_score_aggregation',
    'kelly_position_size',
]
