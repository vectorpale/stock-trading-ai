# 📈 股票交易策略助手

> 美股 & 港股 & A股 | 多策略融合量化模型 | Claude AI 辅助决策 | 富途平台适配

---

## 一、项目简介

本系统是面向个人投资者的**量化辅助交易工具**，整合了多种量化策略与大语言模型（Claude AI），在每个交易日（或半交易日）自动生成清晰的**买入/卖出/观察信号**，供您在富途牛牛平台手动执行。

### 核心目标
- 年化收益目标：30%+（在可控风险下）
- 信号频率：每日或半日生成一次
- 执行方式：人工手动操作（非自动化交易）
- 支持市场：美股（NASDAQ/NYSE）+ 港股（HKEX）+ A股（沪深）

### 核心功能

| 功能 | 说明 |
|------|------|
| 多策略融合 | 动量策略 + 均值回归 + 多因子模型（三策略加权综合） |
| 多数据源 | AkShare / yfinance / BaoStock 自动降级，美港A股全覆盖 |
| AI 情绪分析 | Claude 分析股票新闻情绪（需 API Key） |
| AI 技术解读 | Claude 解读技术指标，给出建议理由 |
| 回测引擎 | 历史数据验证，精确扣除富途费率 |
| 可视化界面 | Streamlit Web UI，适合新手操作 |
| 费率计算器 | 实时计算富途美股/港股交易费用 |

---

## 二、环境准备（新手必读）

### 2.1 安装 Python

1. 访问 [python.org](https://www.python.org/downloads/)，下载 **Python 3.10 或以上版本**
2. 安装时勾选 **"Add Python to PATH"**
3. 打开终端（Windows: 命令提示符 / Mac: 终端），验证：
   ```bash
   python --version
   # 应显示 Python 3.10.x 或更高
   ```

### 2.2 下载本项目

```bash
# 克隆仓库（或直接下载 ZIP 解压）
git clone https://github.com/vectorpale/stock-trading-ai.git
cd stock-trading-ai
```

### 2.3 安装依赖包

```bash
# 创建虚拟环境（推荐，避免包冲突）
python -m venv venv

# 激活虚拟环境
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

> 安装过程约需 3-5 分钟，请耐心等待。

### 2.4 配置 API Key（可选但推荐）

AI 分析功能需要 Claude API Key：

1. 访问 [console.anthropic.com](https://console.anthropic.com/) 注册并获取 API Key
2. 在项目目录下创建 `.env` 文件：
   ```bash
   cp .env.example .env
   ```
3. 用文本编辑器打开 `.env`，填写：
   ```
   ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
   ```

> **不配置 API Key 也可以正常使用**，只是 AI 分析功能会禁用，量化策略信号仍然正常工作。

---

## 三、快速开始

### 方式一：Web 界面（推荐新手）

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，界面包含：

- **今日信号**：点击"生成信号"按钮，获取所有股票的买卖信号
- **图表分析**：K线图 + RSI + MACD 可视化
- **策略回测**：验证策略历史表现
- **费率计算器**：查询富途实际交易费用

### 方式二：命令行（适合定期运行）

```bash
# 生成今日所有股票信号（使用配置文件中的股票池）
python run_signals.py

# 指定股票（美股）
python run_signals.py --symbols AAPL MSFT NVDA TSLA

# 指定港股
python run_signals.py --symbols 0700.HK 9988.HK --capital 500000

# 不使用 AI（更快，适合快速查看）
python run_signals.py --no-ai

# 保存报告到文件
python run_signals.py --save

# 运行回测
python run_signals.py --backtest NVDA
```

---

## 四、信号解读指南

### 4.1 信号类型

| 信号 | 含义 | 建议操作 |
|------|------|----------|
| 🟢 **BUY（买入）** | 多策略综合看多，综合分数 ≥ 45 | 考虑按建议仓位买入 |
| 🔴 **SELL（卖出）** | 多策略综合看空，综合分数 ≤ -45 | 考虑减仓或清仓 |
| 🟡 **WATCH（关注）** | 信号接近阈值，需进一步确认 | 加入自选，等待更强信号 |
| ⚪ **HOLD（持有）** | 无明显方向信号 | 维持现有仓位 |

### 4.2 信号分数说明

- **综合评分**：-100 到 +100，正数看多，负数看空
- **置信度**：0-100，越高代表策略一致性越好
- **建议仓位**：占可用资金的百分比（系统已限制最大单仓 15%）

### 4.3 止损止盈

- **止损价**：跌破此价格建议止损（默认跌幅 7%）
- **目标价**：涨幅达 20% 可考虑止盈
- 实际操作可根据个人风险承受能力调整

---

## 五、自定义配置

编辑 `config/config.yaml` 文件：

```yaml
# 修改股票池
market:
  us_stocks:
    - AAPL
    - MSFT
    - 你想加入的股票代码

# 修改风险参数
risk:
  stop_loss_pct: 0.07      # 止损线（7%）
  take_profit_pct: 0.20    # 止盈线（20%）
  max_position_size: 0.15  # 单只股票最大仓位 15%

# 修改策略权重（三个权重之和必须 = 1）
strategy_weights:
  momentum: 0.35           # 动量策略
  mean_reversion: 0.25     # 均值回归
  multi_factor: 0.40       # 多因子模型
```

---

## 六、策略说明

### 6.1 动量策略（35% 权重）

捕捉已形成趋势的股票，核心逻辑：
- 多周期价格动量（5日/20日/60日）
- MACD 金叉/死叉信号
- 均线多头排列（价格 > 20MA > 50MA > 200MA）
- 成交量确认（放量上涨更可靠）

**适合**：趋势明确的强势股、牛市环境

### 6.2 均值回归策略（25% 权重）

在超买超卖区间反向操作，核心逻辑：
- 布林带下轨买入，上轨卖出
- RSI 超卖（<30）买入，超买（>70）卖出
- KDJ 随机指标辅助确认

**适合**：震荡市、蓝筹股、相对稳定的股票

### 6.3 多因子模型（40% 权重）

参考学术界成熟的因子投资框架，综合评估：

| 因子 | 权重 | 说明 |
|------|------|------|
| 动量因子 | 30% | 价格变化率（多周期） |
| 趋势因子 | 25% | ADX + 均线排列 |
| 波动率因子 | 15% | 低波动溢价效应 |
| 成交量因子 | 15% | OBV + 量价配合 |
| 技术因子 | 15% | MACD + RSI + 52周高位 |

### 6.4 AI 增强

Claude AI 在量化信号基础上提供：
- **新闻情绪**：自动分析最新新闻，判断市场情绪偏向
- **技术解读**：将复杂指标转化为自然语言建议
- **每日简报**：生成市场概况和操作要点

---

## 七、富途费率说明

系统回测时精确扣除以下费用：

### 美股费率
| 费用项目 | 标准 |
|----------|------|
| 佣金 | $0.0049/股，最低 $0.99 |
| 平台使用费 | $0.0049/股，最低 $0.99 |
| SEC 费（卖出） | 成交额 × 0.00278% |
| FINRA TAF（卖出） | $0.000145/股，最高 $7.27 |

### 港股费率
| 费用项目 | 标准 |
|----------|------|
| 佣金 | 0.03%，最低 HK$3 |
| 平台使用费 | 0.03%，最低 HK$3 |
| 印花税（买卖均收） | 0.1% |
| 证监会征费 | 0.0027% |
| 联交所交易费 | 0.00565% |

---

## 八、重要免责声明

> ⚠️ **本工具仅用于辅助决策，不构成任何投资建议。**
>
> - 股票投资具有风险，过去表现不代表未来收益
> - 30% 年化收益目标是策略设计目标，实际收益受市场环境影响
> - 请在充分了解风险的前提下，结合个人判断做出投资决策
> - 建议从小仓位开始验证策略，再逐步增加资金

---

## 九、常见问题

**Q: 安装 requirements.txt 时报错？**
A: 尝试 `pip install -r requirements.txt --upgrade`，或检查 Python 版本是否 ≥ 3.10。

**Q: 获取数据失败？**
A: 系统会自动切换数据源（AkShare → yfinance → BaoStock）。若所有来源失败，请检查网络连接。国内用户推荐优先使用 AkShare，无需代理。

**Q: AI 分析不可用？**
A: 检查 `.env` 文件中的 `ANTHROPIC_API_KEY` 是否正确填写。

**Q: A股数据如何获取？**
A: 直接填 6 位数字代码即可（如 `600519` 贵州茅台），系统自动识别沪深市场，使用 AkShare 或 BaoStock 获取。

**Q: 港股数据格式？**
A: 港股代码格式为数字+`.HK`，例如腾讯是 `0700.HK`，注意前面的 `0`。

**Q: 如何添加新股票？**
A: 编辑 `config/config.yaml`，在对应市场的列表中添加股票代码即可。

---

## 十、项目结构

```
stock-trading-ai/
├── app.py                    # Streamlit Web 界面（主入口）
├── run_signals.py            # 命令行信号生成器
├── test_data_sources.py      # 数据源基准测试工具
├── requirements.txt          # Python 依赖包
├── .env.example              # 环境变量模板
├── config/
│   └── config.yaml           # 主配置文件（股票池、费率、策略参数）
└── src/
    ├── data/
    │   └── fetcher.py        # 多数据源获取（AkShare / yfinance / BaoStock，自动降级）
    ├── indicators/
    │   └── technical.py      # 技术指标计算（20+ 指标）
    ├── strategies/
    │   ├── momentum.py       # 动量策略
    │   ├── mean_reversion.py # 均值回归策略
    │   └── multi_factor.py   # 多因子模型
    ├── ai/
    │   ├── llm_analyzer.py   # Claude AI 分析模块
    │   └── debate_engine.py  # 多智能体辩论引擎
    ├── backtest/
    │   ├── engine.py         # 回测引擎
    │   └── fees.py           # 富途费率计算器
    ├── signals/
    │   └── generator.py      # 综合信号生成器
    └── utils/
        └── helpers.py        # 工具函数
```
