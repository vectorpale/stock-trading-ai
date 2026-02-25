# 📈 Stock Trading AI — 多智能体辩论交易系统

> 5 个 AI 智能体激烈辩论，CIO 拍板决策 —— 学术级量化 + AI 深度推理的股票信号系统

---

## 核心架构：多智能体辩论引擎

本系统的核心不是传统量化策略，而是一套受 6 篇顶级学术论文启发的 **多智能体 AI 辩论决策系统**。量化策略只是输入层，最终交易决策由 AI 辩论裁决。

### 两阶段信号管线

```
阶段一：量化初筛                          阶段二：AI 辩论裁决
┌─────────────────────────┐          ┌──────────────────────────────────┐
│  数据获取（多源自动降级）  │          │  Phase 1: 5 智能体独立分析        │
│  → 20+ 技术指标计算       │          │  Phase 2: 多轮辩论 + 收敛检测     │
│  → 3 策略并行评分         │          │  Phase 3: 风险委员会审查（一票否决）│
│  → 综合分 > 15 → 进入辩论 │  ──────→ │  Phase 4: CIO 最终裁决（Opus 模型）│
└─────────────────────────┘          └──────────────────────────────────┘
   动量 35% + 均值回归 25%                  CIO 决策完全覆盖量化评分
   + 多因子 40%                             不是加权混合，而是独立裁决
```

### 5 个智能体角色

| 角色 | 代号 | 职责 | 权重 |
|------|------|------|------|
| **多头研究员** | Alex (Bull) | 寻找上涨信号：突破、动量、放量 | 1.0 |
| **空头研究员** | Morgan (Bear) | 识别下行风险：超买、趋势衰竭、背离 | 1.0 |
| **量化分析师** | Sam (Quant) | 纯数学评估：因子暴露、胜率、统计显著性 | 1.2 |
| **宏观分析师** | Kai (Macro) | 宏观环境：美联储政策、美元、板块轮动 | 0.9 |
| **风险官** | Chris (Risk) | 风控把关：Kelly 仓位、VaR、集中度 | 1.1 |

### CIO 决策权威

- 使用 **Claude Opus**（最强推理模型）做最终决策
- 拥有**唯一决策权**：综合 5 位分析师观点，但不受其约束
- 受风险委员会硬约束：VETO（一票否决）时 CIO 必须执行
- 低置信度保护：BUY 置信度 < 65 → 自动降级为 WATCH

### 模型分工

| 角色 | 模型 | 用途 |
|------|------|------|
| CIO（Phase 4） | `claude-opus-4-6` | 最终交易决策 — 最强推理 |
| 分析师（Phase 1/2/3） | `claude-sonnet-4-6` | 独立分析和多轮辩论 |
| 情绪分析 | `claude-haiku-4-5-20251001` | 新闻情绪摘要 — 快速低成本 |

### FinMem 记忆系统

基于 Reflexion（NeurIPS 2023）的三层决策记忆，让系统从历史中学习：

- **短期记忆**：同一只股票最近 5 次决策
- **长期记忆**：30 天历史模式
- **反思层**：分析历史失误（假阳性、错过的卖出信号），注入 CIO 上下文

### 学术参考

| 论文 | 来源 | 本系统中的应用 |
|------|------|---------------|
| Du et al. — Multi-Agent Debate | ICML 2024 | 多轮辩论 + 收敛检测 |
| TradingAgents (Xiao et al.) | 2024 | Bull/Bear 研究员角色设计 |
| FinCon (Yu et al.) | 2024 | Manager-Analyst 层级决策 |
| FinAgent (Zhang et al.) | 2024 | 多模态分析框架 |
| Reflexion (Shinn et al.) | NeurIPS 2023 | 语言反思式学习（记忆系统） |
| CRITIC (Gou et al.) | ICLR 2024 | 自我验证与批判机制 |

---

## 功能概览

| 功能 | 说明 |
|------|------|
| **多智能体辩论** | 5 Agent 辩论 + CIO 裁决（推荐，默认模式） |
| **量化策略融合** | 动量 + 均值回归 + 多因子模型，三策略加权 |
| **模拟交易账户** | $1,000,000 模拟资金，根据信号自动建仓，实时跟踪盈亏 |
| **多数据源** | yfinance / AkShare / Tushare / BaoStock，自动降级 |
| **回测引擎** | 历史数据验证，精确扣除富途费率 |
| **盘前定时信号** | GitHub Actions 每日自动生成（港股 07:30 / 美股 20:30 北京时间） |
| **Web 界面** | Streamlit 交互式 UI，支持手机浏览器 |
| **费率计算器** | 富途美股/港股完整费用结构 |
| **支持市场** | 美股（NASDAQ/NYSE）+ 港股（HKEX）+ A 股（沪深） |

---

## 快速开始

### 安装

```bash
git clone https://github.com/vectorpale/stock-trading-ai.git
cd stock-trading-ai

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # 编辑填入 ANTHROPIC_API_KEY（可选）
```

### 运行

```bash
# Web 界面（推荐）
streamlit run app.py

# 命令行 — 纯量化（不需要 API Key）
python run_signals.py --symbols NVDA AAPL MSFT --no-ai

# 命令行 — 完整辩论模式（需要 API Key）
python run_signals.py --symbols NVDA AAPL MSFT

# 回测
python run_signals.py --backtest NVDA --capital 50000
```

### AI 模式

| 模式 | 配置 | 特点 |
|------|------|------|
| **debate**（默认，推荐） | `ai.mode: debate` | 5 Agent 辩论 + CIO 决策，最深度 |
| **simple** | `ai.mode: simple` | 单 Agent 分析（80% 量化 + 20% 情绪），更快 |
| **quant-only** | `--no-ai` | 纯量化策略，不调用 API，免费 |

---

## Web 界面 Tab

| Tab | 功能 |
|-----|------|
| **今日信号** | 生成交易信号，查看各策略评分和 AI 分析 |
| **模拟账户** | $1M 模拟账户，一键信号交易，持仓/成本/盈亏/净值跟踪 |
| **图表分析** | K 线 + 均线 + 布林带 + RSI + MACD 可视化 |
| **策略回测** | 历史回测，净值曲线，绩效指标 |
| **费率计算器** | 富途美股/港股费用明细 |

---

## 信号格式

```
============================================================
  交易信号报告 - 2025-01-15 21:45
============================================================

  买入信号 (BUY):
----------------------------------------
  【NVDA】
    当前价格: 875.43
    建议仓位: 12% 的可用资金
    止损价位: 814.15 (-7.0%)
    目标价位: 1050.52 (+20.0%)
    信号强度: 78/100
    信号原因: [动量] MACD金叉；均线多头排列；[多因子] 动量因子强势(+72)
```

| 信号 | 含义 | 触发条件 |
|------|------|----------|
| **BUY** | 买入 | CIO 判定看多，置信度 >= 65 |
| **SELL** | 卖出 | CIO 判定看空 |
| **WATCH** | 关注 | 信号接近阈值，或置信度不足 |
| **HOLD** | 持有 | 无明显方向 |

---

## 量化策略

### 动量策略（35%）
多周期价格动量 + MACD 金叉/死叉 + 均线多头排列 + 成交量确认

### 均值回归策略（25%）
布林带超买超卖 + RSI 极值反转 + KDJ 辅助确认

### 多因子模型（40%）

| 因子 | 权重 | 指标 |
|------|------|------|
| 动量 | 30% | 多周期价格变化率 |
| 趋势 | 25% | ADX + 均线排列 |
| 波动率 | 15% | 低波动溢价 |
| 成交量 | 15% | OBV + 量价配合 |
| 技术 | 15% | MACD + RSI + 52 周高位 |

---

## 数据源

| 市场 | 优先级 | 说明 |
|------|--------|------|
| 美股 | yfinance → AkShare | yfinance 速度快，AkShare 国内稳定 |
| 港股 | AkShare → yfinance | AkShare 国内优先 |
| A 股 | AkShare → Tushare → BaoStock | Tushare 需 Token，数据质量更高 |

---

## 风控参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 单只最大仓位 | 12% | CIO 受此约束 |
| 最大持仓数 | 10 | 集中持仓 |
| CIO 最低置信度 | 65 | 低于此 BUY → WATCH |
| 止损线 | 7% | CIO 未指定时默认 |
| 止盈线 | 20% | CIO 未指定时默认 |
| 组合回撤暂停 | 15% | 超过暂停交易 |

---

## 云端部署

| 方式 | 文档 | 适合 |
|------|------|------|
| **GitHub Actions** | [CLOUD_SETUP.md](CLOUD_SETUP.md) | 全自动，手机看结果 |
| **阿里云无影** | [DEPLOY_CHINA_CLOUD.md](DEPLOY_CHINA_CLOUD.md) | 自有服务器，需要 Web UI |

---

## 配置

所有参数在 `config/config.yaml`：
- 股票池（美股 ~50 只 + 港股 ~110 只）
- 策略权重
- 风控参数
- AI 模型选择和辩论参数
- 费率结构

Secrets 在 `.env`（不提交）：`ANTHROPIC_API_KEY`、`NEWSAPI_KEY`、`TUSHARE_TOKEN`

---

## 项目结构

```
stock-trading-ai/
├── app.py                      # Streamlit Web UI（含模拟账户）
├── run_signals.py              # CLI 信号生成器
├── test_data_sources.py        # 数据源基准测试
├── requirements.txt            # 依赖
├── config/
│   └── config.yaml             # 主配置
├── src/
│   ├── ai/
│   │   ├── advanced_debate/    # ★ 核心：多智能体辩论引擎 v2.0
│   │   │   ├── engine.py       #   4 阶段辩论流程编排
│   │   │   ├── agents.py       #   5 智能体人设 + CIO 系统提示词
│   │   │   ├── memory.py       #   FinMem 三层决策记忆
│   │   │   └── utils.py        #   JSON 解析、共识计算、Kelly 仓位
│   │   ├── llm_analyzer.py     #   Simple 模式单 Agent 分析
│   │   └── debate_engine.py    #   Legacy v1.0（已弃用）
│   ├── data/
│   │   └── fetcher.py          # 多源数据获取（yfinance/AkShare/Tushare/BaoStock）
│   ├── indicators/
│   │   └── technical.py        # 20+ 技术指标
│   ├── strategies/
│   │   ├── base.py             # 策略抽象基类
│   │   ├── momentum.py         # 动量策略
│   │   ├── mean_reversion.py   # 均值回归策略
│   │   └── multi_factor.py     # 多因子模型
│   ├── signals/
│   │   └── generator.py        # 信号编排器（两阶段管线）
│   ├── portfolio/
│   │   └── tracker.py          # 模拟交易组合管理器
│   ├── backtest/
│   │   ├── engine.py           # 回测引擎
│   │   └── fees.py             # 富途费率计算器
│   └── utils/
│       └── helpers.py          # 配置加载、格式化、市场检测
├── .github/workflows/
│   ├── daily_signals.yml       # 每日盘前信号（港股 07:30 + 美股 20:30）
│   └── backtest.yml            # 手动触发回测
├── CLOUD_SETUP.md              # GitHub Actions 部署指南
├── DEPLOY_CHINA_CLOUD.md       # 阿里云无影部署指南
└── CLAUDE.md                   # AI 开发助手上下文
```

---

## 免责声明

> **本工具仅用于辅助决策，不构成任何投资建议。**
>
> 股票投资具有风险，过去表现不代表未来收益。请在充分了解风险的前提下，结合个人判断做出投资决策。建议从模拟账户开始验证策略，再逐步增加实盘资金。
