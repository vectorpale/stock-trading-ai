"""
辅助工具函数 - Utility Functions
"""

import json
import logging
import math
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def parse_json_response(raw: str) -> dict:
    """
    健壮地解析 LLM 返回的 JSON（处理 markdown 包裹、尾部文字等边界情况）
    """
    text = raw.strip()
    # 去掉 ```json ... ``` 包裹
    if text.startswith('```'):
        parts = text.split('```')
        text = parts[1] if len(parts) > 1 else text
        if text.startswith('json'):
            text = text[4:]
    # 截取第一个完整 JSON 对象
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text.strip())


def format_indicators(indicators: Dict[str, float]) -> str:
    """格式化技术指标为可读文本"""
    mapping = [
        ('Close',        '当前价格',              '{:.2f}'),
        ('rsi_14',       'RSI(14)',               '{:.1f}'),
        ('rsi_6',        'RSI(6)',                '{:.1f}'),
        ('macd_hist',    'MACD柱状值',             '{:.4f}'),
        ('bb_pct',       '布林带位置(0下轨/1上轨)', '{:.2f}'),
        ('adx',          'ADX趋势强度',            '{:.1f}'),
        ('volume_ratio', '量比',                  '{:.2f}'),
        ('volatility_20','年化波动率',             '{:.1%}'),
        ('mom_20',       '20日涨跌幅',             '{:.1%}'),
        ('mom_5',        '5日涨跌幅',              '{:.1%}'),
        ('sma_20',       'SMA(20)',               '{:.2f}'),
        ('sma_50',       'SMA(50)',               '{:.2f}'),
        ('sma_200',      'SMA(200)',              '{:.2f}'),
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


def format_strategy_signals(strategy_signals: Dict[str, Dict]) -> str:
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
            reason = sig.get('reason', '')[:80]
            lines.append(
                f"  {label}: {sig.get('action', 'N/A')} "
                f"(评分: {sig.get('score', 0):+.1f}) — {reason}"
            )
    return '\n'.join(lines) if lines else '无策略信号'


def compute_consensus(agents: Dict[str, Dict]) -> str:
    """计算多方共识状态"""
    positions = [agents[k].get('position', 'HOLD') for k in agents]
    buy_count = positions.count('BUY')
    sell_count = positions.count('SELL')
    total = len(positions)
    if buy_count == total:
        return '全员看多'
    elif sell_count == total:
        return '全员看空'
    elif buy_count / total >= 0.8:
        return '强多头共识'
    elif sell_count / total >= 0.8:
        return '强空头共识'
    elif buy_count > sell_count:
        return '多数看多'
    elif sell_count > buy_count:
        return '多数看空'
    else:
        return '意见分歧'


def compute_convergence_score(
    agents: Dict[str, Dict],
    prev_agents: Optional[Dict[str, Dict]] = None,
) -> float:
    """
    计算辩论收敛度（参考 Du et al. 2024 收敛判断）

    基于各 agent score 的标准差来衡量分歧程度。
    score 标准差越小（各方评分越接近），收敛度越高（0~1）。

    Returns:
        float: 0 = 完全分歧，1 = 完全共识
    """
    scores = [float(agents[k].get('score', 0)) for k in agents]
    if not scores:
        return 0.0

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(variance)

    # 将 std（0~100 范围）映射到 0~1 的收敛度
    # std=0 → convergence=1.0；std=100 → convergence=0.0
    convergence = max(0.0, 1.0 - std / 100.0)
    return round(convergence, 3)


def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_position: float = 0.15,
    kelly_fraction: float = 0.25,    # 用 1/4 Kelly 降低方差（实践常用）
) -> float:
    """
    Kelly 准则仓位计算（参考量化风险管理最佳实践）

    Args:
        win_rate: 历史胜率 (0~1)
        avg_win: 平均盈利幅度（如 0.10 = 10%）
        avg_loss: 平均亏损幅度（如 0.07 = 7%）
        max_position: 仓位上限
        kelly_fraction: Kelly 缩放系数（Full Kelly 波动大，实践多用 1/4）

    Returns:
        float: 建议仓位比例
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.05  # 信息不足时给最小仓位

    # Kelly 公式: f = (p*b - q) / b，其中 b = avg_win/avg_loss, q = 1-p
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    kelly = (p * b - q) / b

    if kelly <= 0:
        return 0.0  # 负期望，不下注

    # 缩放 Kelly，不超过上限
    position = min(kelly * kelly_fraction, max_position)
    return round(position, 3)


def weighted_score_aggregation(
    agents: Dict[str, Dict],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    加权汇总各 agent 评分（参考 TradingAgents 层级加权决策）

    Args:
        agents: 各 agent 结果字典
        weights: 各 agent 权重（默认使用 AGENT_PERSONAS 中定义的权重）
    """
    from .agents import AGENT_PERSONAS

    total_weight = 0.0
    weighted_sum = 0.0

    for key, agent in agents.items():
        w = (weights or {}).get(key) or AGENT_PERSONAS.get(key, {}).get('weight', 1.0)
        # 置信度调整权重（参考 FinCon：高置信度 agent 权重更大）
        confidence_factor = agent.get('confidence', 50) / 100.0
        effective_weight = w * confidence_factor
        weighted_sum += agent.get('score', 0) * effective_weight
        total_weight += effective_weight

    return round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0
