# 阿里云无影云电脑部署指南

在无影云电脑上一键部署股票交易策略助手，手机随时查看信号。

---

## 一、环境准备（只需做一次）

### 1. 打开终端

在无影云电脑桌面上打开 **终端（Terminal）**。
如果是 Linux 桌面，找到「终端」应用即可。

### 2. 安装 Python（如果还没有）

```bash
# 查看是否已安装
python3 --version

# 如果没有（Ubuntu/Debian 系）：
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

# 如果没有（CentOS/Alinux 系）：
sudo yum install -y python3 python3-pip git
```

要求 **Python 3.10+**。

### 3. 下载代码

```bash
cd ~
git clone https://github.com/vectorpale/stock-trading-ai.git
cd stock-trading-ai
```

### 4. 创建虚拟环境并安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> 如果 `pandas-ta` 安装失败，可以先跳过，不影响核心功能：
> ```bash
> pip install $(grep -v pandas-ta requirements.txt)
> ```

### 5. 配置 API Key

```bash
cp .env.example .env
```

用编辑器打开 `.env`，填入你的 Claude API Key：

```bash
# 用 nano 编辑（或任何你习惯的编辑器）
nano .env
```

内容改为：
```
ANTHROPIC_API_KEY=sk-ant-你的真实key
```

保存退出（nano 中按 `Ctrl+O` 保存，`Ctrl+X` 退出）。

> **没有 API Key？** 去 https://console.anthropic.com/ 注册获取。
> 也可以先不配置，用 `--no-ai` 模式只跑量化分析。

---

## 二、运行信号

### 快速试跑（纯量化，不需要 API Key）

```bash
cd ~/stock-trading-ai
source venv/bin/activate
python run_signals.py --symbols NVDA AAPL MSFT --no-ai
```

### 完整运行（量化 + AI 辩论分析）

```bash
python run_signals.py --symbols NVDA AAPL MSFT
```

### 运行全部股票池

```bash
# 跑配置文件中所有 ~160 只股票（耗时较长）
python run_signals.py --no-ai

# 只跑美股
python run_signals.py --symbols NVDA AMD AVGO MSFT GOOGL AMZN META AAPL TSLA --no-ai
```

### 保存报告到文件

```bash
python run_signals.py --symbols NVDA AAPL --save
# 报告保存在 reports/ 目录
```

### 运行回测

```bash
python run_signals.py --backtest NVDA --capital 50000
```

### 启动 Web 界面（浏览器操作）

```bash
streamlit run app.py
# 会显示一个地址如 http://localhost:8501
# 在无影云电脑的浏览器里打开即可
```

---

## 三、设置每日自动运行（推荐）

让系统每天自动生成信号，你只需要查看结果。

### 方式 A：crontab 定时任务

```bash
# 编辑定时任务
crontab -e
```

添加以下行（每天北京时间 5:30 运行，对应美股收盘后）：

```cron
30 5 * * 1-5 cd /root/stock-trading-ai && /root/stock-trading-ai/venv/bin/python run_signals.py --save >> /root/stock-trading-ai/logs/daily.log 2>&1
```

> 注意：如果你的用户目录不是 `/root`，请替换为实际路径（用 `echo $HOME` 查看）。

创建日志目录：

```bash
mkdir -p ~/stock-trading-ai/logs
```

### 方式 B：写一个快捷脚本

创建一键运行脚本：

```bash
cat > ~/run_stock.sh << 'EOF'
#!/bin/bash
cd ~/stock-trading-ai
source venv/bin/activate
echo "========================================"
echo "  股票信号 - $(date '+%Y-%m-%d %H:%M')"
echo "========================================"
python run_signals.py --symbols NVDA AMD AVGO MSFT GOOGL AMZN META AAPL TSLA --no-ai
EOF
chmod +x ~/run_stock.sh
```

之后每次只需要：

```bash
~/run_stock.sh
```

---

## 四、手机远程查看

### 方法 1：无影 App（最直接）

手机下载 **阿里云无影** App，直接连到你的云电脑桌面操作。

### 方法 2：SSH 远程连接

如果你的无影开了公网 IP，可以用手机 SSH 工具（Termux / JuiceSSH）连接：

```bash
ssh root@你的公网IP
cd ~/stock-trading-ai && source venv/bin/activate
python run_signals.py --symbols NVDA AAPL --no-ai
```

### 方法 3：查看定时任务的输出

如果配置了 crontab，直接看日志：

```bash
cat ~/stock-trading-ai/logs/daily.log
# 或者只看最新的
tail -100 ~/stock-trading-ai/logs/daily.log
```

---

## 五、更新代码

当有新版本时：

```bash
cd ~/stock-trading-ai
git pull origin main
source venv/bin/activate
pip install -r requirements.txt  # 如果依赖有变化
```

---

## 六、常见问题

**Q: 运行提示 `ModuleNotFoundError`？**
A: 确保已激活虚拟环境：`source venv/bin/activate`

**Q: yfinance 获取数据失败？**
A: 国内网络可能访问 Yahoo Finance 不稳定，程序会自动切换到 akshare（国内源）。如果都失败，检查网络或加代理。

**Q: AI 分析太慢/太贵？**
A: 用 `--no-ai` 只跑量化分析，速度快且免费。AI 辩论模式每只股票约消耗 $0.05-0.15（取决于模型）。

**Q: 想修改股票池？**
A: 编辑 `config/config.yaml`，在 `us_stocks` 或 `hk_stocks` 下增删股票代码。

**Q: 想调整策略参数？**
A: 编辑 `config/config.yaml`，可调整：
- `strategy_weights` — 三个量化策略的权重
- `risk` — 止损/止盈/最大仓位等
- `ai.mode` — 切换 `debate`（多智能体辩论）或 `simple`（单智能体）模式

**Q: 怎么看港股信号？**
A:
```bash
python run_signals.py --symbols 0700.HK 9988.HK 3690.HK 1810.HK --no-ai
```

**Q: 无影云电脑重启后需要重新配置吗？**
A: 不需要。代码、虚拟环境、`.env` 配置、crontab 都保存在磁盘上，重启后一切照旧。
