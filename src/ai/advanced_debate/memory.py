"""
决策记忆模块 - Decision Memory Module

设计参考 / Design Reference:
  - FinMem (Yu et al. IEEE TBD 2024, arXiv:2311.13743):
      "FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design"
      分层记忆：短期（近期交易）+ 中期（财报事件）+ 长期（宏观规律）+ 反射层（自我批评）
  - Reflexion (Shinn et al. NeurIPS 2023, arXiv:2303.11366):
      语言强化学习——通过语言形式的反思从失败中学习，无需梯度更新（~1520 引用）
  - CRITIC (Gou et al. ICLR 2024, arXiv:2305.11738):
      LLM 与外部工具交互式自我批评——检测错误并迭代修正

功能：
  1. 短期记忆（Short-term）: 最近 N 次同一股票的辩论决策
  2. 长期记忆（Long-term）: 各股票的历史信号模式和规律总结
  3. 反射记忆（Reflection）: 基于真实收益的事后反思（FinMem 反射层 + Reflexion）
  4. 记忆检索：为当前辩论提供相关历史上下文注入
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class DecisionMemory:
    """
    分层决策记忆系统（参考 FinMem 三层记忆架构）

    Layer 1 - 短期记忆 (Short-term): 最近 N 次辩论记录
    Layer 2 - 长期记忆 (Long-term): 按股票/日期归档的历史信号
    Layer 3 - 反射记忆 (Reflection): 事后反思，从结果中学习规律
    """

    def __init__(
        self,
        memory_dir: str = "data/memory",       # 统一存放在 data/ 目录下
        short_term_window: int = 5,            # 短期记忆窗口（最近N次）
        long_term_window: int = 30,            # 长期记忆窗口（最近N天）
    ):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.short_term_window = short_term_window
        self.long_term_window = long_term_window

        # 内存缓存（避免频繁读文件）
        self._cache: Dict[str, List[Dict]] = {}

    # ── 写入记忆 ─────────────────────────────────────────────

    def store_decision(
        self,
        symbol: str,
        market: str,
        debate_result: Dict[str, Any],
        indicators: Dict[str, float],
        price: float,
    ) -> None:
        """
        存储一次辩论决策到记忆库（FinMem 短期/长期层）

        Args:
            symbol: 股票代码
            market: 'us' 或 'hk'
            debate_result: AdvancedDebateEngine.run_debate() 的完整返回结果
            indicators: 技术指标快照
            price: 决策时价格
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'market': market,
            'price_at_decision': price,
            'action': debate_result.get('cio_decision', {}).get('action', 'HOLD'),
            'confidence': debate_result.get('cio_decision', {}).get('confidence', 0),
            'score': debate_result.get('cio_decision', {}).get('score', 0),
            'verdict': debate_result.get('cio_decision', {}).get('verdict', ''),
            'agent_consensus': debate_result.get('agent_consensus', ''),
            'avg_agent_score': debate_result.get('avg_agent_score', 0),
            # 快照关键指标（用于后续反思对比）
            'indicators_snapshot': {
                k: v for k, v in indicators.items()
                if k in ('rsi_14', 'macd_hist', 'adx', 'bb_pct', 'mom_20', 'volume_ratio')
            },
            # 占位：后续可通过 update_outcome() 更新实际收益
            'actual_return': None,
            'outcome_date': None,
            'reflection': None,
        }

        # 读取历史 → 追加 → 写回
        records = self._load_records(symbol)
        records.append(record)
        self._save_records(symbol, records)

        # 更新内存缓存
        self._cache[symbol] = records
        logger.debug(f"[Memory] 存储 {symbol} 决策记录: {record['action']}")

    def update_outcome(
        self,
        symbol: str,
        decision_timestamp: str,
        actual_return: float,
        reflection_text: Optional[str] = None,
    ) -> None:
        """
        更新实际收益结果（用于 Reflexion 自我反思）

        Args:
            symbol: 股票代码
            decision_timestamp: 决策时间戳（用于定位记录）
            actual_return: 实际收益率（如 0.05 = +5%）
            reflection_text: 可选，手动填入的反思文字
        """
        records = self._load_records(symbol)
        for r in records:
            if r['timestamp'] == decision_timestamp:
                r['actual_return'] = actual_return
                r['outcome_date'] = datetime.now().isoformat()
                if reflection_text:
                    r['reflection'] = reflection_text
                break
        self._save_records(symbol, records)
        self._cache[symbol] = records

    # ── 检索记忆 ─────────────────────────────────────────────

    def get_context_for_debate(
        self,
        symbol: str,
        max_records: Optional[int] = None,
    ) -> str:
        """
        为辩论引擎生成历史上下文注入文本（参考 FinMem 记忆检索机制）

        返回格式化的历史决策摘要，注入到 Phase 1 提示词中，
        让各分析师能参考"我们上次对这只股票的判断和结果"。

        Args:
            symbol: 股票代码
            max_records: 最多引用多少条记录（默认用 short_term_window）

        Returns:
            格式化文本字符串，空字符串表示无历史记录
        """
        n = max_records or self.short_term_window
        records = self._load_records(symbol)
        recent = records[-n:] if len(records) >= n else records

        if not recent:
            return ''

        lines = [f"【{symbol} 历史辩论记录（最近 {len(recent)} 次）】"]
        for r in recent:
            date_str = r['timestamp'][:10]
            outcome_str = ''
            if r.get('actual_return') is not None:
                ret = r['actual_return']
                outcome_str = f" → 实际收益: {ret:+.1%}"
                if r.get('reflection'):
                    outcome_str += f"（反思: {r['reflection'][:50]}）"
            lines.append(
                f"  {date_str}: {r['action']} "
                f"score={r.get('score', 0):+.0f} "
                f"conf={r.get('confidence', 0):.0f} "
                f"共识={r.get('agent_consensus', '?')}"
                f"{outcome_str}"
            )

        # 如果有足够的有结果记录，生成胜率统计
        resolved = [r for r in records if r.get('actual_return') is not None]
        if len(resolved) >= 3:
            buy_records = [r for r in resolved if r['action'] == 'BUY']
            if buy_records:
                wins = sum(1 for r in buy_records if r['actual_return'] > 0)
                avg_ret = sum(r['actual_return'] for r in buy_records) / len(buy_records)
                lines.append(
                    f"  [历史统计] 买入信号胜率: {wins}/{len(buy_records)} "
                    f"平均收益: {avg_ret:+.1%}"
                )

        return '\n'.join(lines)

    def get_reflection_summary(self, symbol: str) -> str:
        """
        生成反射层摘要（参考 Reflexion 的语言反思机制）
        提炼该股票的历史教训，注入给 CIO 作为决策参考。
        """
        records = self._load_records(symbol)
        resolved = [r for r in records if r.get('actual_return') is not None]

        if len(resolved) < 3:
            return ''

        # 分析误判模式
        false_bulls = [
            r for r in resolved
            if r['action'] == 'BUY' and r['actual_return'] < -0.03
        ]
        false_bears = [
            r for r in resolved
            if r['action'] == 'SELL' and r['actual_return'] > 0.03
        ]

        lines = [f"【{symbol} 反射层摘要】"]
        if false_bulls:
            lines.append(f"  误判买入 {len(false_bulls)} 次（平均亏损 "
                         f"{sum(r['actual_return'] for r in false_bulls)/len(false_bulls):.1%}）")
            # 提炼常见误判场景
            for r in false_bulls[-2:]:
                ind = r.get('indicators_snapshot', {})
                lines.append(
                    f"    示例: RSI={ind.get('rsi_14','?'):.0f} "
                    f"MACD={ind.get('macd_hist','?'):.4f} 时买入亏损"
                )
        if false_bears:
            lines.append(f"  误判卖出 {len(false_bears)} 次（错失收益 "
                         f"{sum(r['actual_return'] for r in false_bears)/len(false_bears):.1%}）")

        if not false_bulls and not false_bears:
            lines.append("  近期无显著误判记录，历史信号质量良好。")

        return '\n'.join(lines)

    # ── 内部 I/O ─────────────────────────────────────────────

    def _load_records(self, symbol: str) -> List[Dict]:
        """从磁盘加载记录（优先内存缓存）"""
        if symbol in self._cache:
            return self._cache[symbol]

        path = self.memory_dir / f"{symbol.replace('.', '_')}.json"
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                self._cache[symbol] = records
                return records
            except Exception as e:
                logger.warning(f"[Memory] 加载 {symbol} 记录失败: {e}")
        return []

    def _save_records(self, symbol: str, records: List[Dict]) -> None:
        """写回磁盘"""
        path = self.memory_dir / f"{symbol.replace('.', '_')}.json"
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Memory] 写入 {symbol} 记录失败: {e}")
