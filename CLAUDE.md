# CLAUDE.md — Stock Trading AI

## Project Overview

Quantitative stock trading strategy assistant that generates BUY/SELL/WATCH/HOLD signals for US stocks (NASDAQ/NYSE), Hong Kong stocks (HKEX), and A-shares (Shanghai/Shenzhen). Combines three quantitative strategies with a multi-agent Claude AI debate system for decision support.

**Primary language:** Python 3.10+
**Primary documentation language:** Chinese (中文), code comments mix Chinese and English

## Repository Structure

```
stock-trading-ai/
├── app.py                  # Streamlit web UI entry point
├── run_signals.py          # CLI signal generator entry point
├── test_data_sources.py    # Data source benchmark tool
├── requirements.txt        # Python dependencies
├── config/
│   └── config.yaml         # Main configuration (stock pools, fees, risk, AI settings)
├── data/                   # Runtime data cache & memory (gitignored)
├── src/
│   ├── data/
│   │   └── fetcher.py      # Multi-source data fetcher (yfinance/akshare/tushare/baostock)
│   ├── indicators/
│   │   └── technical.py    # 20+ technical indicators (SMA, EMA, RSI, MACD, Bollinger, etc.)
│   ├── strategies/
│   │   ├── base.py         # Abstract base class for strategies
│   │   ├── momentum.py     # Momentum strategy (35% weight)
│   │   ├── mean_reversion.py  # Mean reversion strategy (25% weight)
│   │   └── multi_factor.py # Multi-factor model (40% weight)
│   ├── signals/
│   │   └── generator.py    # Signal orchestrator — two-stage pipeline
│   ├── ai/
│   │   ├── llm_analyzer.py     # Simple mode: single-agent analyzer
│   │   ├── debate_engine.py    # Legacy debate engine wrapper
│   │   └── advanced_debate/
│   │       ├── engine.py   # Advanced multi-agent debate engine (4 phases)
│   │       ├── agents.py   # 5 agent personas (Bull, Bear, Quant, Macro, Risk)
│   │       ├── memory.py   # FinMem-style layered decision memory
│   │       └── utils.py    # JSON parsing, consensus, convergence, Kelly sizing
│   ├── backtest/
│   │   ├── engine.py       # Event-driven backtesting engine
│   │   └── fees.py         # Futu brokerage fee calculator (US/HK)
│   └── utils/
│       └── helpers.py      # Config loading, formatting, market detection
├── .env.example            # Environment variable template
├── .gitignore
├── README.md               # User guide (Chinese)
└── CLOUD_SETUP.md          # GitHub Actions deployment guide
```

## How to Run

### Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Then fill in ANTHROPIC_API_KEY
```

### Entry Points
```bash
# Web UI (interactive)
streamlit run app.py

# CLI signal generation
python run_signals.py                           # All configured stocks
python run_signals.py --symbols AAPL MSFT NVDA  # Specific symbols
python run_signals.py --no-ai                   # Quant-only (no API calls)
python run_signals.py --backtest AAPL --capital 50000

# Data source validation
python test_data_sources.py
```

## Architecture

### Two-Stage Signal Pipeline (debate mode)

```
Data Fetcher → Technical Indicators → 3 Quant Strategies (parallel)
    → Composite Score → Prefilter (score > 15 passes)
    → Phase 1: 5 agents give independent analysis
    → Phase 2: Multi-round debate with convergence check
    → Phase 3: Risk committee review (potential veto)
    → Phase 4: CIO final decision (sole authority)
    → Final Signal: action, score, confidence, position size
```

### AI Model Allocation
| Role | Model | Purpose |
|------|-------|---------|
| CIO (Phase 4) | `claude-opus-4-6` | Final trading decisions — strongest reasoning |
| Analysts (Phase 1/2/3) | `claude-sonnet-4-6` | Independent analysis and debate |
| Sentiment | `claude-haiku-4-5-20251001` | News summarization — fast and cheap |

### AI Modes
- **`debate`** (default, recommended): Full multi-agent debate with CIO authority
- **`simple`**: Single-agent analysis (80% quant + 20% sentiment)

## Key Conventions

### Signal Output Format
Signals use a standard dict structure:
```python
{
    'symbol': str,
    'action': 'BUY' | 'SELL' | 'WATCH' | 'HOLD',
    'score': float,       # -100 to +100
    'confidence': float,  # 0 to 100
    'price': float,
    'stop_loss': float,
    'take_profit': float,
    'position_pct': float,  # 0.0 to 1.0
    'reason': str,
    'debate_result': dict,  # Only in debate mode
}
```

### Market Detection
Symbols are auto-detected by format:
- **US stocks:** Plain tickers like `AAPL`, `NVDA`
- **HK stocks:** `XXXX.HK` format like `0700.HK`
- **A-shares:** 6-digit codes like `600519` (Shanghai) or `000001` (Shenzhen)

### Data Source Fallback
The fetcher tries multiple data sources with automatic fallback:
- US/HK: yfinance → akshare
- A-shares: akshare → tushare → baostock

### Strategy Pattern
All quant strategies extend `BaseStrategy` (in `src/strategies/base.py`) and implement `generate_signal()`. The weighted composite is: momentum 35% + mean_reversion 25% + multi_factor 40%.

### Configuration
- All tunable parameters live in `config/config.yaml`
- Stock pools, fee structures, risk limits, strategy weights, AI settings are all config-driven
- Secrets go in `.env` (never committed): `ANTHROPIC_API_KEY`, `NEWSAPI_KEY`, `TUSHARE_TOKEN`

### Risk Parameters (from config)
- Max single position: 12% of portfolio
- Max total positions: 10
- Min CIO confidence for BUY: 65 (below → downgraded to WATCH)
- Default stop loss: 7%, take profit: 20%
- Portfolio drawdown halt: 15%

## Development Notes

### No Formal Test Suite
There is no pytest/unittest configuration. Validation is done through:
- `test_data_sources.py` for data source reliability
- Manual testing via CLI (`run_signals.py`) and Streamlit UI
- Backtesting engine for historical strategy validation

### No Linting/Formatting Tools
No black, ruff, flake8, or similar tools are configured. Follow existing code style:
- Standard Python conventions (PEP 8 generally)
- Chinese comments are common and expected
- Type hints used sparingly

### Git Conventions
- Commit messages use `feat:` prefix for features, written in Chinese
- The `data/` and `reports/` directories are gitignored (runtime artifacts)
- Never commit `.env` files

### Dependencies
All in `requirements.txt` — no pyproject.toml or setup.py. Install with `pip install -r requirements.txt`.

### Key Academic References (architecture design)
The multi-agent debate system is inspired by:
- Du et al. (ICML 2024) — Multi-agent debate
- TradingAgents (Xiao et al. 2024) — Bull/Bear researcher roles
- FinCon (Yu et al. 2024) — Manager-Analyst hierarchy
- FinAgent (Zhang et al. 2024) — Multi-modal analysis
- Reflexion (Shinn et al. NeurIPS 2023) — Language-based reinforcement learning
