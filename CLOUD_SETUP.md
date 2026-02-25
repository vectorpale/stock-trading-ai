# 云端运行指南

---

## 方案一：GitHub Actions 全自动运行（最省心，手机看结果）

GitHub 的云端服务器每天自动运行信号生成，你只需要在手机/网页上查看结果。

### 第一步：创建你自己的 GitHub 仓库（只需做一次）

1. 打开 [github.com](https://github.com)，点击右上角 **"+"** → **"New repository"**
2. 填写仓库名称（如 `stock-trading-ai`），选 **Private（私有）**
3. 点击 **"Create repository"**

### 第二步：将代码推送到你的 GitHub

```bash
# 1. 克隆本项目
git clone https://github.com/vectorpale/stock-trading-ai.git
cd stock-trading-ai

# 2. 改为指向你自己的仓库（替换为你的 GitHub 用户名）
git remote set-url origin https://github.com/你的用户名/stock-trading-ai.git

# 3. 推送
git push -u origin main
```

> 代码已包含 `.github/workflows/` 工作流文件，推送后 Actions 自动生效。

### 第三步：配置 Secrets（API Key）

在你的 GitHub 仓库页面：

1. 点击 **Settings** → **Secrets and variables** → **Actions**
2. 点击 **"New repository secret"**，添加以下内容：

| Secret 名称 | 值 | 是否必填 |
|---|---|---|
| `ANTHROPIC_API_KEY` | 你的 Claude API Key（`sk-ant-...`） | 可选（不填则 AI 分析自动禁用，量化信号正常） |
| `NEWSAPI_KEY` | NewsAPI Key | 可选（增强新闻情绪分析） |
| `TUSHARE_TOKEN` | Tushare Token | 可选（A股数据源） |

### 第四步：手动触发或等待自动运行

**自动运行**：每个交易日 UTC 21:30（北京时间次日 05:30）自动执行，对应美股收盘后。

**立即手动触发**：
1. 在你的 GitHub 仓库，点击顶部 **"Actions"** 标签
2. 左侧选择 **"每日股票信号"**
3. 点击右侧 **"Run workflow"**
4. 可选填参数：
   - **股票代码**：空格分隔，如 `NVDA AAPL MSFT`（留空则跑全部股票池）
   - **AI 分析**：选 `true` 启用多智能体辩论（需要 API Key，更慢但更深入）
   - **资金量**：默认 100000 USD
5. 点击绿色 **"Run workflow"** 按钮

**手动触发回测**：
1. Actions → 左侧选 **"策略回测"**
2. Run workflow → 填写股票代码（如 `NVDA`）和资金量
3. 运行完成后下载 Artifact 查看三策略回测对比

### 第五步：查看信号结果（3 种方式）

**方法 A（最方便）**：打开仓库根目录的 `latest_signals.txt` 文件，每次运行后自动更新。手机浏览器打开 GitHub 就能看。

**方法 B**：Actions → 选择某次运行 → 展开 **"打印信号到日志"** 步骤，直接看日志输出。

**方法 C**：Actions 运行结果页底部，下载 **Artifacts** 压缩包，里面有完整报告文件。

### 工作流文件说明

项目已包含两个工作流：

| 文件 | 功能 | 触发方式 |
|---|---|---|
| `.github/workflows/daily_signals.yml` | 每日信号生成 | 定时（周一至周五 UTC 21:30）+ 手动 |
| `.github/workflows/backtest.yml` | 策略回测 | 仅手动 |

---

## 方案二：无影云电脑本地运行

详见 [DEPLOY_CHINA_CLOUD.md](DEPLOY_CHINA_CLOUD.md)，包含完整的环境配置、crontab 定时任务、手机远程查看等。

适合：
- 想在自己的服务器上运行
- 需要 Streamlit Web UI
- 网络环境需要特殊配置（代理等）

---

## 信号结果示例

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

  关注信号 (WATCH):
----------------------------------------
  【AAPL】
    当前价格: 215.30
    ...

============================================================
  以上信号仅供参考，请结合市场情况人工判断
============================================================
```

---

## 常见问题

**Q: GitHub Actions 免费吗？**
A: 公开仓库完全免费。私有仓库每月有 2000 分钟免费额度（本程序每次运行约 3-5 分钟，一个月约 100-150 分钟，完全够用）。

**Q: 信号生成失败了怎么看原因？**
A: 进入 Actions → 点击失败的运行 → 展开报红的步骤，查看错误信息。

**Q: 怎么修改股票池？**
A: 编辑 `config/config.yaml`，在 `us_stocks` 或 `hk_stocks` 下添加/删除股票代码，提交到 GitHub 即可。

**Q: 可以修改自动运行时间吗？**
A: 编辑 `.github/workflows/daily_signals.yml`，修改 `cron` 那行。格式：`'分 时 日 月 星期'`，均为 UTC 时间。

**Q: AI 分析默认关闭？**
A: 是的。定时任务默认 `--no-ai`（纯量化，快速免费）。手动触发时可以选择开启 AI 辩论分析。开启后每只股票约消耗 $0.05-0.15 API 费用。
