# 云端运行指南（GitHub Actions + Openclaw）

---

## 方案一：GitHub Actions 全自动运行（最省心）

GitHub 的云端服务器每天自动运行，你只需要在网页上查看结果。

### 第一步：创建 GitHub 仓库（只需做一次）

1. 打开 [github.com](https://github.com)，点击右上角 **"+"** → **"New repository"**
2. 填写仓库名称（如 `stock-assistant`），选 **Private（私有）**
3. 点击 **"Create repository"**，记住你的仓库地址（格式：`https://github.com/你的用户名/stock-assistant`）

### 第二步：将代码推送到你的 GitHub

在你的**云电脑终端**或本地任意有 git 的电脑上执行：

```bash
# 1. 克隆现有仓库（包含已开发的代码）
git clone https://github.com/vectorpale/All.git
cd All

# 2. 将 stock_assistant 单独推送到你自己的仓库
# （以下用你自己的 GitHub 用户名和仓库名替换）
git subtree push --prefix stock_assistant \
  https://github.com/你的用户名/stock-assistant.git main
```

> **更简单的方式（推荐新手）**：
> 1. 在 GitHub 网页上点击 `vectorpale/All` → `stock_assistant` 文件夹
> 2. 点击右上角 "..." → **"Download ZIP"**，解压后手动上传到新仓库

### 第三步：添加 GitHub Actions 工作流

将 `.github/workflows/` 目录整体复制到你自己的新仓库根目录。

如果是手动上传的，需要在你的新仓库里创建：
- `.github/workflows/daily_signals.yml`（从本项目复制）
- `.github/workflows/backtest.yml`（从本项目复制）

### 第四步：配置 Claude API Key（Secret）

在你的 GitHub 仓库页面：

1. 点击 **Settings**（设置）→ **Secrets and variables** → **Actions**
2. 点击 **"New repository secret"**
3. 名称填写：`ANTHROPIC_API_KEY`
4. 值填写：你的 Claude API Key（`sk-ant-...`）
5. 点击 **"Add secret"**

> 如果没有 API Key，跳过这步也可以，AI 功能会自动关闭，量化信号仍然正常。

### 第五步：手动触发或等待自动运行

**自动运行时间**：每个交易日 UTC 21:30（北京时间次日 05:30）自动执行，对应美股收盘后。

**立即手动触发**：
1. 在你的 GitHub 仓库，点击顶部 **"Actions"** 标签
2. 左侧选择 **"每日股票信号"**
3. 点击右侧 **"Run workflow"** → 填写参数 → 点击绿色按钮

### 第六步：查看信号结果

运行完成后，有三个地方可以看结果：

**方法 A（最方便）**：仓库根目录的 `latest_signals.txt` 文件，每次运行后自动更新

**方法 B**：点击 Actions → 选择某次运行 → 滚动到 **"Print signals to log"** 步骤，直接看日志

**方法 C**：在 Actions 运行结果页面下方，下载 **Artifacts** 压缩包，解压查看报告

---

## 方案二：通过 Openclaw 助手按需运行

在你的云电脑上，让 Openclaw 助手执行以下命令：

### 初始化（只需做一次）

告诉你的 Openclaw 助手：

```
请帮我执行以下命令，完成股票助手的初始化：

git clone https://github.com/vectorpale/All.git /home/user/stock_assistant_app
cd /home/user/stock_assistant_app/stock_assistant
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=你的API_KEY" > .env
```

### 每次生成信号时

告诉 Openclaw 助手：

```
请帮我运行股票信号，命令如下：
cd /home/user/stock_assistant_app/stock_assistant && python run_signals.py --no-ai
```

或者带 AI 分析：

```
cd /home/user/stock_assistant_app/stock_assistant && python run_signals.py
```

### 更新代码（有新版本时）

```
cd /home/user/stock_assistant_app && git pull origin claude/stock-trading-assistant-xd8ed
```

---

## 信号结果示例

```
============================================================
  交易信号报告 - 2025-01-15 21:45
============================================================

🟢 买入信号 (BUY):
----------------------------------------
  【NVDA】
    当前价格: 875.43
    建议仓位: 12% 的可用资金
    止损价位: 814.15 (-7.0%)
    目标价位: 1050.52 (+20.0%)
    信号强度: 78/100
    信号原因: [动量] MACD金叉；均线多头排列；[多因子] 动量因子强势(+72)

🟡 关注信号 (WATCH):
----------------------------------------
  【AAPL】
    当前价格: 215.30
    ...

============================================================
⚠️  以上信号仅供参考，请结合市场情况人工判断
============================================================
```

---

## 常见问题

**Q: GitHub Actions 免费吗？**
A: 公开仓库完全免费，私有仓库每月有 2000 分钟免费额度（本程序每次运行约 3-5 分钟，一个月约用 100-150 分钟，完全够用）。

**Q: 信号生成失败了怎么看原因？**
A: 进入 Actions → 点击失败的运行 → 展开报红的步骤，查看错误信息。

**Q: 怎么修改股票池？**
A: 编辑 `config/config.yaml`，在 `us_stocks` 或 `hk_stocks` 下添加/删除股票代码，提交到 GitHub 即可。

**Q: 可以修改自动运行时间吗？**
A: 编辑 `.github/workflows/daily_signals.yml`，修改 `cron` 那行。格式：`'分 时 日 月 星期'`，均为 UTC 时间。
