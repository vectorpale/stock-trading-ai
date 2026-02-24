"""
工具函数模块
Utility helper functions
"""

import os
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    if config_path is None:
        # 默认路径：项目根目录下的 config/config.yaml
        base_dir = Path(__file__).parent.parent.parent
        config_path = base_dir / "config" / "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f"配置文件已加载: {config_path}")
    return config


def format_signal_output(signals: List[Dict]) -> str:
    """将信号列表格式化为易读的文本输出"""
    if not signals:
        return "今日无交易信号。\n"

    lines = []
    lines.append("=" * 60)
    lines.append(f"  交易信号报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    # 分类显示
    buy_signals = [s for s in signals if s.get('action') == 'BUY']
    sell_signals = [s for s in signals if s.get('action') == 'SELL']
    watch_signals = [s for s in signals if s.get('action') == 'WATCH']

    if buy_signals:
        lines.append("\n🟢 买入信号 (BUY):")
        lines.append("-" * 40)
        for s in buy_signals:
            lines.append(_format_single_signal(s))

    if sell_signals:
        lines.append("\n🔴 卖出信号 (SELL):")
        lines.append("-" * 40)
        for s in sell_signals:
            lines.append(_format_single_signal(s))

    if watch_signals:
        lines.append("\n🟡 关注信号 (WATCH):")
        lines.append("-" * 40)
        for s in watch_signals:
            lines.append(_format_single_signal(s))

    lines.append("\n" + "=" * 60)
    lines.append("⚠️  以上信号仅供参考，请结合市场情况人工判断")
    lines.append("=" * 60)

    return "\n".join(lines)


def _format_single_signal(signal: Dict) -> str:
    """格式化单条信号"""
    symbol = signal.get('symbol', 'N/A')
    action = signal.get('action', 'N/A')
    price = signal.get('price', 0)
    confidence = signal.get('confidence', 0)
    reason = signal.get('reason', '')
    stop_loss = signal.get('stop_loss', 0)
    take_profit = signal.get('take_profit', 0)
    position_pct = signal.get('position_pct', 0)

    lines = [
        f"  【{symbol}】",
        f"    当前价格: {price:.2f}",
        f"    建议仓位: {position_pct:.0f}% 的可用资金",
        f"    止损价位: {stop_loss:.2f} ({((stop_loss/price)-1)*100:.1f}%)",
        f"    目标价位: {take_profit:.2f} ({((take_profit/price)-1)*100:.1f}%)",
        f"    信号强度: {confidence:.0f}/100",
        f"    信号原因: {reason}",
    ]
    return "\n".join(lines)


def get_market_type(symbol: str) -> str:
    """判断股票市场类型"""
    if symbol.endswith('.HK'):
        return 'hk'
    else:
        return 'us'


def calculate_position_size(
    capital: float,
    price: float,
    position_pct: float,
    max_position_pct: float = 0.15
) -> Dict[str, Any]:
    """
    计算建议的仓位大小

    Args:
        capital: 总资金
        price: 当前股价
        position_pct: 建议仓位比例 (0-1)
        max_position_pct: 最大仓位比例

    Returns:
        包含股数、金额等信息的字典
    """
    # 限制不超过最大仓位
    actual_pct = min(position_pct, max_position_pct)
    investment = capital * actual_pct
    shares = int(investment / price)

    # 港股以手为单位（通常1手=100股或1000股，此处简化为100股）
    return {
        'shares': shares,
        'investment': shares * price,
        'position_pct': actual_pct * 100
    }
