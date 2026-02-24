"""
股票交易策略助手 - Streamlit Web 界面
Stock Trading Strategy Assistant - Streamlit Web UI

运行方式: streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import load_config, format_signal_output
from src.data.fetcher import DataFetcher
from src.indicators.technical import TechnicalIndicators
from src.signals.generator import SignalGenerator
from src.backtest.engine import BacktestEngine
from src.backtest.fees import FutuFeeCalculator
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.multi_factor import MultiFactorStrategy

# ─── 页面配置 ───
st.set_page_config(
    page_title="股票交易策略助手",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── 自定义样式 ───
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}
.buy-signal { color: #00d09c; font-weight: bold; font-size: 1.2em; }
.sell-signal { color: #ef4444; font-weight: bold; font-size: 1.2em; }
.watch-signal { color: #f59e0b; font-weight: bold; font-size: 1.2em; }
.hold-signal { color: #6b7280; font-weight: bold; font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_config():
    """加载配置（缓存）"""
    try:
        return load_config()
    except Exception as e:
        st.error(f"配置文件加载失败: {e}")
        return {}


@st.cache_resource
def get_fetcher():
    return DataFetcher(cache_dir="data/cache")


def get_signal_generator(config):
    """获取信号生成器（不缓存，因为可能切换AI开关）"""
    return SignalGenerator(config)


def action_color(action: str) -> str:
    colors = {'BUY': '#00d09c', 'SELL': '#ef4444', 'WATCH': '#f59e0b', 'HOLD': '#6b7280'}
    return colors.get(action, '#6b7280')


def action_emoji(action: str) -> str:
    emojis = {'BUY': '🟢 买入', 'SELL': '🔴 卖出', 'WATCH': '🟡 关注', 'HOLD': '⚪ 持有'}
    return emojis.get(action, action)


# ─── 侧边栏 ───
def sidebar():
    st.sidebar.title("⚙️ 设置")

    # API Key
    st.sidebar.subheader("AI 配置")
    api_key = st.sidebar.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="填写后启用 Claude AI 分析功能"
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    ai_enabled = st.sidebar.toggle("启用 AI 分析", value=bool(api_key))

    st.sidebar.divider()

    # 资金设置
    st.sidebar.subheader("资金设置")
    capital = st.sidebar.number_input(
        "可用资金 (USD/HKD)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000,
        format="%d"
    )

    st.sidebar.divider()

    # 快速市场选择
    st.sidebar.subheader("自定义股票池")
    custom_symbols = st.sidebar.text_area(
        "股票代码（每行一个）",
        placeholder="AAPL\nMSFT\n0700.HK",
        height=120
    )

    return {
        'ai_enabled': ai_enabled,
        'capital': capital,
        'custom_symbols': [s.strip().upper() for s in custom_symbols.split('\n') if s.strip()],
    }


# ─── 页面：今日信号 ───
def page_signals(config, sidebar_params):
    st.header("📊 今日交易信号")

    cfg = get_config()
    if not cfg:
        st.error("无法加载配置文件，请检查 config/config.yaml")
        return

    # 合并自定义股票池
    us_stocks = cfg.get('market', {}).get('us_stocks', [])
    hk_stocks = cfg.get('market', {}).get('hk_stocks', [])
    default_symbols = us_stocks + hk_stocks

    custom = sidebar_params.get('custom_symbols', [])
    symbols = list(set(custom + default_symbols)) if custom else default_symbols

    # 控制栏
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_symbols = st.multiselect(
            "选择分析的股票",
            options=symbols,
            default=symbols[:6],
            help="最多建议选择10只"
        )
    with col2:
        capital = sidebar_params['capital']
        st.metric("可用资金", f"${capital:,.0f}")
    with col3:
        ai_status = "✅ 已启用" if sidebar_params['ai_enabled'] else "❌ 未启用"
        st.metric("AI 分析", ai_status)

    if not selected_symbols:
        st.info("请选择至少一只股票")
        return

    if st.button("🚀 生成信号", type="primary", use_container_width=True):
        # 更新配置以匹配 AI 开关
        cfg_copy = dict(cfg)
        cfg_copy['ai'] = dict(cfg.get('ai', {}))
        cfg_copy['ai']['enabled'] = sidebar_params['ai_enabled']

        generator = get_signal_generator(cfg_copy)

        with st.spinner("正在分析市场数据，请稍候..."):
            signals = generator.generate_for_watchlist(
                selected_symbols,
                include_ai=sidebar_params['ai_enabled'],
                capital=capital
            )

        if not signals:
            st.warning("未能生成任何信号，请检查网络连接")
            return

        st.session_state['signals'] = signals
        st.session_state['signal_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')

    # 显示已生成的信号
    if 'signals' in st.session_state:
        signals = st.session_state['signals']
        st.caption(f"信号生成时间：{st.session_state.get('signal_time', '')}")

        # AI 日报
        if sidebar_params['ai_enabled'] and os.environ.get("ANTHROPIC_API_KEY"):
            with st.expander("📝 AI 每日市场简报", expanded=True):
                from src.ai.llm_analyzer import LLMAnalyzer
                llm = LLMAnalyzer()
                summary = llm.generate_daily_summary(signals, capital)
                st.write(summary)

        # 信号卡片
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        sell_signals = [s for s in signals if s['action'] == 'SELL']
        watch_signals = [s for s in signals if s['action'] == 'WATCH']

        # 汇总统计
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("买入信号", len(buy_signals), delta="做多")
        m2.metric("卖出信号", len(sell_signals), delta="减仓")
        m3.metric("关注信号", len(watch_signals))
        m4.metric("分析股票", len(signals))

        st.divider()

        # 信号详情
        for sig in signals:
            if sig['action'] == 'SKIP':
                continue

            with st.expander(
                f"{action_emoji(sig['action'])}  **{sig['symbol']}**  "
                f"— 综合评分 {sig['final_score']:+.0f}  |  "
                f"置信度 {sig['final_confidence']:.0f}%",
                expanded=(sig['action'] in ('BUY', 'SELL'))
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("当前价格", f"{sig['price']:.2f}")
                c2.metric("止损价", f"{sig['stop_loss']:.2f}",
                          delta=f"{((sig['stop_loss']/sig['price'])-1)*100:.1f}%",
                          delta_color="inverse")
                c3.metric("目标价", f"{sig['take_profit']:.2f}",
                          delta=f"+{((sig['take_profit']/sig['price'])-1)*100:.1f}%")
                c4.metric("建议仓位", f"{sig['position_pct']:.1f}%")

                if sig['action'] == 'BUY' and sig['position_shares'] > 0:
                    st.info(
                        f"💡 建议买入 **{sig['position_shares']} 股**，"
                        f"约 **${sig['position_investment']:,.0f}**"
                    )

                st.caption(f"**信号依据：** {sig.get('reason', '无')}")

                # 策略信号分解
                if sig.get('strategy_signals'):
                    st.markdown("**各策略评分：**")
                    sc1, sc2, sc3 = st.columns(3)
                    strat_sigs = sig['strategy_signals']
                    sc1.metric(
                        "动量策略",
                        f"{strat_sigs['momentum'].get('score', 0):+.0f}",
                        strat_sigs['momentum'].get('action', 'HOLD')
                    )
                    sc2.metric(
                        "均值回归",
                        f"{strat_sigs['mean_reversion'].get('score', 0):+.0f}",
                        strat_sigs['mean_reversion'].get('action', 'HOLD')
                    )
                    sc3.metric(
                        "多因子模型",
                        f"{strat_sigs['multi_factor'].get('score', 0):+.0f}",
                        strat_sigs['multi_factor'].get('action', 'HOLD')
                    )

                # AI 分析结果
                if sig.get('ai_analysis') and sig['ai_analysis'].get('reasoning'):
                    st.markdown("**AI 分析：**")
                    st.info(sig['ai_analysis']['reasoning'])
                    if sig['ai_analysis'].get('risk_factors'):
                        st.warning("⚠️ 风险因素：" + "、".join(sig['ai_analysis']['risk_factors']))


# ─── 页面：K线图表 ───
def page_chart(config):
    st.header("📈 K线图表与技术指标")

    fetcher = get_fetcher()
    cfg = get_config()

    all_symbols = (
        cfg.get('market', {}).get('us_stocks', []) +
        cfg.get('market', {}).get('hk_stocks', [])
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.selectbox("选择股票", options=all_symbols, index=0)
    with col2:
        period = st.selectbox("时间范围", ["6mo", "1y", "2y"], index=1)

    df = fetcher.fetch_ohlcv(symbol, period=period, interval="1d", use_cache=True)

    if df is None or df.empty:
        st.error("无法获取数据，请检查股票代码和网络连接")
        return

    # 计算指标
    df_ind = TechnicalIndicators.compute_all(df)

    # ─── K线图 ───
    fig = go.Figure()

    # 蜡烛图
    fig.add_trace(go.Candlestick(
        x=df_ind.index,
        open=df_ind['Open'],
        high=df_ind['High'],
        low=df_ind['Low'],
        close=df_ind['Close'],
        name='K线',
        increasing_line_color='#00d09c',
        decreasing_line_color='#ef4444'
    ))

    # 均线
    for ma, color, name in [
        ('sma_20', '#60a5fa', '20日均线'),
        ('sma_50', '#f59e0b', '50日均线'),
        ('sma_200', '#a78bfa', '200日均线'),
    ]:
        if ma in df_ind.columns:
            fig.add_trace(go.Scatter(
                x=df_ind.index,
                y=df_ind[ma],
                mode='lines',
                name=name,
                line=dict(color=color, width=1.5),
                opacity=0.8
            ))

    # 布林带
    if 'bb_upper' in df_ind.columns:
        fig.add_trace(go.Scatter(
            x=df_ind.index,
            y=df_ind['bb_upper'],
            mode='lines',
            name='布林上轨',
            line=dict(color='rgba(100,180,255,0.5)', width=1, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=df_ind.index,
            y=df_ind['bb_lower'],
            mode='lines',
            name='布林下轨',
            line=dict(color='rgba(100,180,255,0.5)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(100,180,255,0.05)',
        ))

    fig.update_layout(
        title=f"{symbol} 价格走势",
        yaxis_title="价格",
        xaxis_rangeslider_visible=False,
        height=500,
        template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── 副图：RSI + MACD ───
    col_rsi, col_macd = st.columns(2)

    with col_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind['rsi_14'],
            mode='lines', name='RSI(14)',
            line=dict(color='#60a5fa', width=2)
        ))
        fig_rsi.add_hline(y=70, line_dash='dash', line_color='#ef4444', annotation_text='超买70')
        fig_rsi.add_hline(y=30, line_dash='dash', line_color='#00d09c', annotation_text='超卖30')
        fig_rsi.update_layout(
            title='RSI 相对强弱指标',
            yaxis=dict(range=[0, 100]),
            height=250, template='plotly_dark'
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col_macd:
        fig_macd = go.Figure()
        colors = ['#00d09c' if v >= 0 else '#ef4444' for v in df_ind['macd_hist'].fillna(0)]
        fig_macd.add_trace(go.Bar(
            x=df_ind.index, y=df_ind['macd_hist'],
            name='MACD 柱', marker_color=colors
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind['macd'],
            mode='lines', name='MACD', line=dict(color='#60a5fa', width=1.5)
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind['macd_signal'],
            mode='lines', name='信号线', line=dict(color='#f59e0b', width=1.5)
        ))
        fig_macd.update_layout(
            title='MACD 指标',
            height=250, template='plotly_dark'
        )
        st.plotly_chart(fig_macd, use_container_width=True)

    # 当前技术指标表
    latest = df_ind.iloc[-1]
    st.subheader("当前技术指标值")
    indicators_data = {
        '指标': ['RSI(14)', 'RSI(6)', 'MACD柱', 'KDJ-K', 'ADX', '布林带位置', '成交量比率', '年化波动率'],
        '当前值': [
            f"{latest.get('rsi_14', 0):.1f}",
            f"{latest.get('rsi_6', 0):.1f}",
            f"{latest.get('macd_hist', 0):.4f}",
            f"{latest.get('stoch_k', 0):.1f}",
            f"{latest.get('adx', 0):.1f}",
            f"{latest.get('bb_pct', 0)*100:.1f}%",
            f"{latest.get('volume_ratio', 1):.2f}x",
            f"{latest.get('volatility_20', 0)*100:.1f}%",
        ],
        '信号解读': [
            '超买' if latest.get('rsi_14', 50) > 70 else ('超卖' if latest.get('rsi_14', 50) < 30 else '中性'),
            '超买' if latest.get('rsi_6', 50) > 70 else ('超卖' if latest.get('rsi_6', 50) < 30 else '中性'),
            '多头' if latest.get('macd_hist', 0) > 0 else '空头',
            '超买' if latest.get('stoch_k', 50) > 80 else ('超卖' if latest.get('stoch_k', 50) < 20 else '中性'),
            '强趋势' if latest.get('adx', 20) > 25 else '弱趋势',
            '接近上轨' if latest.get('bb_pct', 0.5) > 0.8 else ('接近下轨' if latest.get('bb_pct', 0.5) < 0.2 else '中段'),
            '放量' if latest.get('volume_ratio', 1) > 1.5 else ('缩量' if latest.get('volume_ratio', 1) < 0.7 else '正常'),
            '高波动' if latest.get('volatility_20', 0.3) > 0.5 else '正常',
        ]
    }
    st.dataframe(pd.DataFrame(indicators_data), use_container_width=True, hide_index=True)


# ─── 页面：回测 ───
def page_backtest(config):
    st.header("🔬 策略回测（含富途费率）")

    cfg = get_config()
    fetcher = get_fetcher()

    all_symbols = (
        cfg.get('market', {}).get('us_stocks', []) +
        cfg.get('market', {}).get('hk_stocks', [])
    )

    # 回测参数
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        symbol = st.selectbox("股票代码", all_symbols)
    with col2:
        strategy_name = st.selectbox(
            "回测策略",
            ["动量策略", "均值回归", "多因子模型"],
            index=2
        )
    with col3:
        initial_capital = st.number_input("初始资金", min_value=10000, value=100000, step=10000)
    with col4:
        backtest_period = st.selectbox("回测周期", ["1年", "2年", "3年"], index=1)

    period_map = {"1年": "1y", "2年": "2y", "3年": "3y"}
    market = 'hk' if symbol.endswith('.HK') else 'us'

    if st.button("▶ 开始回测", type="primary"):
        with st.spinner("正在进行回测计算..."):
            df = fetcher.fetch_ohlcv(symbol, period=period_map[backtest_period], interval="1d")

            if df is None or len(df) < 100:
                st.error("数据不足，无法回测")
                return

            # 选择策略
            strategy_map = {
                "动量策略": MomentumStrategy(),
                "均值回归": MeanReversionStrategy(),
                "多因子模型": MultiFactorStrategy(),
            }
            strategy = strategy_map[strategy_name]

            engine = BacktestEngine(
                initial_capital=initial_capital,
                market=market,
                fee_config=cfg.get('fees', {}),
                stop_loss_pct=cfg.get('risk', {}).get('stop_loss_pct', 0.07),
                take_profit_pct=cfg.get('risk', {}).get('take_profit_pct', 0.20),
            )

            result = engine.run(
                symbol=symbol,
                df=df,
                strategy_fn=lambda df, sym: strategy.generate_signal(
                    TechnicalIndicators.compute_all(df), sym
                ),
            )

        if 'error' in result:
            st.error(result['error'])
            return

        metrics = result['metrics']
        equity_curve = result['equity_curve']
        trades_df = result['trades']

        # ─── 绩效指标卡片 ───
        st.subheader("回测结果")
        m1, m2, m3, m4, m5 = st.columns(5)

        annual_return = metrics.get('年化收益率(%)', 0)
        m1.metric(
            "年化收益率",
            f"{annual_return:.1f}%",
            delta=f"目标 30%" if annual_return < 30 else "✅ 达标",
            delta_color="normal" if annual_return >= 30 else "inverse"
        )
        m2.metric("最大回撤", f"{metrics.get('最大回撤(%)', 0):.1f}%")
        m3.metric("夏普比率", f"{metrics.get('夏普比率', 0):.2f}")
        m4.metric("胜率", f"{metrics.get('胜率(%)', 0):.1f}%")
        m5.metric("超额收益", f"{metrics.get('超额收益(%)', 0):.1f}%")

        # 详细指标表
        with st.expander("查看全部绩效指标"):
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['指标', '数值'])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # ─── 净值曲线图 ───
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve['equity'],
            mode='lines',
            name='策略净值',
            line=dict(color='#60a5fa', width=2),
            fill='tozeroy',
            fillcolor='rgba(96,165,250,0.1)'
        ))

        # 标记交易点
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']

            if not buy_trades.empty:
                # 近似净值
                buy_equities = []
                for d in buy_trades['date']:
                    idx = equity_curve.index.get_indexer([d], method='nearest')[0]
                    buy_equities.append(equity_curve['equity'].iloc[idx])

                fig.add_trace(go.Scatter(
                    x=buy_trades['date'],
                    y=buy_equities,
                    mode='markers',
                    name='买入点',
                    marker=dict(color='#00d09c', size=8, symbol='triangle-up')
                ))

        fig.update_layout(
            title=f"{symbol} {strategy_name} 净值曲线",
            yaxis_title="净值 ($)",
            height=400,
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

        # ─── 交易记录 ───
        if not trades_df.empty:
            st.subheader("交易记录")
            display_cols = [c for c in
                ['date', 'action', 'shares', 'price', 'fees', 'pnl', 'pnl_pct', 'reason']
                if c in trades_df.columns
            ]
            st.dataframe(
                trades_df[display_cols].style.format({
                    'price': '{:.2f}',
                    'fees': '{:.2f}',
                    'pnl': '{:.2f}',
                    'pnl_pct': '{:.2%}',
                }),
                use_container_width=True
            )


# ─── 页面：费率计算器 ───
def page_fees():
    st.header("💰 富途费率计算器")
    st.caption("基于富途牛牛（2024年）实际收费标准")

    cfg = get_config()
    fee_calc = FutuFeeCalculator(cfg.get('fees', {}))

    tab1, tab2 = st.tabs(["美股费率", "港股费率"])

    with tab1:
        st.subheader("美股单笔费用")
        c1, c2, c3 = st.columns(3)
        with c1:
            us_shares = st.number_input("股数", min_value=1, value=100, key="us_shares")
        with c2:
            us_price = st.number_input("股价 (USD)", min_value=0.01, value=150.0, key="us_price")
        with c3:
            us_is_sell = st.checkbox("卖出方向", key="us_sell")

        fees = fee_calc.calc_us_fees(int(us_shares), float(us_price), us_is_sell)
        trade_value = us_shares * us_price

        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("交易金额", f"${trade_value:,.2f}")
        m2.metric("佣金", f"${fees['commission']:.2f}")
        m3.metric("平台费", f"${fees['platform_fee']:.2f}")
        m4.metric("SEC/FINRA费", f"${fees['sec_fee'] + fees['finra_taf']:.2f}")
        m5.metric("**总费用**", f"**${fees['total']:.2f}**")

        st.info(f"综合费率：{fees['fee_rate']*100:.4f}% | 往返费率约：{fees['fee_rate']*200:.3f}%")

    with tab2:
        st.subheader("港股单笔费用")
        c1, c2, c3 = st.columns(3)
        with c1:
            hk_shares = st.number_input("股数（按手，1手通常=100股）", min_value=100, value=1000, step=100, key="hk_shares")
        with c2:
            hk_price = st.number_input("股价 (HKD)", min_value=0.01, value=350.0, key="hk_price")
        with c3:
            hk_is_sell = st.checkbox("卖出方向", key="hk_sell")

        fees_hk = fee_calc.calc_hk_fees(int(hk_shares), float(hk_price), hk_is_sell)
        trade_value_hk = hk_shares * hk_price

        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("交易金额", f"HK${trade_value_hk:,.0f}")
        m2.metric("佣金+平台", f"HK${fees_hk['commission']+fees_hk['platform_fee']:.2f}")
        m3.metric("印花税", f"HK${fees_hk['stamp_duty']:.2f}")
        m4.metric("监管费", f"HK${fees_hk['levy']+fees_hk['trading_fee']:.2f}")
        m5.metric("**总费用**", f"**HK${fees_hk['total']:.2f}**")

        st.info(f"综合费率：{fees_hk['fee_rate']*100:.4f}% | 往返费率约：{fees_hk['fee_rate']*200:.3f}%")


# ─── 主入口 ───
def main():
    # 加载配置
    config = get_config()

    # 侧边栏
    sidebar_params = sidebar()

    # 主导航
    st.title("📈 股票交易策略助手")
    st.caption("美股 & 港股 | 多策略融合 | AI 辅助决策 | 富途平台")

    tabs = st.tabs(["🎯 今日信号", "📊 图表分析", "🔬 策略回测", "💰 费率计算器"])

    with tabs[0]:
        page_signals(config, sidebar_params)

    with tabs[1]:
        page_chart(config)

    with tabs[2]:
        page_backtest(config)

    with tabs[3]:
        page_fees()


if __name__ == "__main__":
    main()
