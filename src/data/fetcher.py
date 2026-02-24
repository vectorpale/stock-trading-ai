"""
数据获取模块 - 多数据源支持，自动降级
Data Fetcher - Multi-source with automatic fallback

支持的数据源 / Supported Sources:
  - AkShare  : A股 / 港股 / 美股，国内最稳定，无需 API Key
  - yfinance : 美股 / 港股，速度快但偶有网络限制
  - BaoStock : A股专用，历史数据完整，免费

市场→优先级 / Market Priority:
  美股 US   : yfinance → akshare_us
  港股 HK   : akshare_hk → yfinance
  A股 CN    : akshare_cn → baostock
"""

import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 辅助函数 / Helper functions
# ─────────────────────────────────────────────

def detect_market(symbol: str) -> str:
    """
    根据股票代码自动识别市场类型

    Returns:
        'US' | 'HK' | 'CN_SH' | 'CN_SZ'
    """
    s = symbol.upper().strip()

    # 显式后缀
    if s.endswith('.HK'):
        return 'HK'
    if s.endswith('.SS') or s.endswith('.SH'):
        return 'CN_SH'
    if s.endswith('.SZ'):
        return 'CN_SZ'

    # 纯数字 → A股
    clean = s.replace('.', '')
    if clean.isdigit():
        n = int(clean)
        # 沪市: 6xxxxx / 9xxxxx
        if n >= 600000:
            return 'CN_SH'
        # 深市: 0xxxxx / 3xxxxx
        return 'CN_SZ'

    # 默认美股
    return 'US'


def period_to_dates(period: str) -> Tuple[datetime, datetime]:
    """将 period 字符串转为 (start, end) datetime"""
    end = datetime.now()
    mapping = {
        '1d': 1, '5d': 5,
        '1mo': 30, '3mo': 90, '6mo': 180,
        '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
    }
    days = mapping.get(period, 365)
    return end - timedelta(days=days), end


def period_to_date_str(period: str) -> Tuple[str, str]:
    """将 period 转为 ('YYYYMMDD', 'YYYYMMDD') 字符串对（AkShare 格式）"""
    start, end = period_to_dates(period)
    return start.strftime('%Y%m%d'), end.strftime('%Y%m%d')


def normalize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    将任意来源的 DataFrame 标准化为 Open/High/Low/Close/Volume 格式，
    以日期为 index，数值列为 float64。
    """
    if df is None or df.empty:
        return None

    # ── 列名归一化（大小写 + 中文）──
    rename = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ('开盘', 'open', 'openprice'):
            rename[col] = 'Open'
        elif cl in ('最高', 'high', 'highprice'):
            rename[col] = 'High'
        elif cl in ('最低', 'low', 'lowprice'):
            rename[col] = 'Low'
        elif cl in ('收盘', 'close', 'closeprice'):
            rename[col] = 'Close'
        elif cl in ('成交量', 'volume', 'vol'):
            rename[col] = 'Volume'
        elif cl in ('日期', 'date', 'tradedate', 'trade_date'):
            rename[col] = 'Date'
    df = df.rename(columns=rename)

    # ── 将 Date 列设为 index ──
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.set_index('Date')

    # ── 只保留 OHLCV ──
    keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    if 'Close' not in keep:
        return None
    df = df[keep].copy()

    # ── 转 float，清洗 ──
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.index = pd.to_datetime(df.index, errors='coerce')
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    df = df.sort_index()
    df.dropna(subset=['Close'], inplace=True)
    df = df[df['Close'] > 0]

    return df if not df.empty else None


# ─────────────────────────────────────────────
# 各数据源抓取函数
# ─────────────────────────────────────────────

def fetch_yfinance(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """yfinance：美股 / 港股，支持日线和日内数据"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        return normalize_df(df)
    except Exception as e:
        logger.debug(f"[yfinance] {symbol} 失败: {e}")
        return None



def fetch_akshare_us(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """AkShare 美股日线"""
    if interval not in ('1d', '1wk', '1mo'):
        return None
    try:
        import akshare as ak
        freq_map = {'1d': 'daily', '1wk': 'weekly', '1mo': 'monthly'}
        freq = freq_map.get(interval, 'daily')
        start_str, end_str = period_to_date_str(period)
        df = ak.stock_us_hist(
            symbol=symbol, period=freq,
            start_date=start_str, end_date=end_str,
            adjust='qfq'
        )
        return normalize_df(df)
    except Exception as e:
        logger.debug(f"[akshare_us] {symbol} 失败: {e}")
        return None


def fetch_akshare_hk(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """AkShare 港股日线（港股代码如 '0700.HK' → '00700'）"""
    if interval not in ('1d', '1wk', '1mo'):
        return None
    try:
        import akshare as ak
        # 统一转为 5 位数字代码
        sym = symbol.upper().replace('.HK', '').lstrip('0').zfill(5)
        freq_map = {'1d': 'daily', '1wk': 'weekly', '1mo': 'monthly'}
        freq = freq_map.get(interval, 'daily')
        start_str, end_str = period_to_date_str(period)
        df = ak.stock_hk_hist(
            symbol=sym, period=freq,
            start_date=start_str, end_date=end_str,
            adjust='qfq'
        )
        return normalize_df(df)
    except Exception as e:
        logger.debug(f"[akshare_hk] {symbol} 失败: {e}")
        return None


def fetch_akshare_cn(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """AkShare A股日线（支持 '600519' / '600519.SH' 格式）"""
    if interval not in ('1d', '1wk', '1mo'):
        return None
    try:
        import akshare as ak
        sym = symbol.split('.')[0]
        freq_map = {'1d': 'daily', '1wk': 'weekly', '1mo': 'monthly'}
        freq = freq_map.get(interval, 'daily')
        start_str, end_str = period_to_date_str(period)
        df = ak.stock_zh_a_hist(
            symbol=sym, period=freq,
            start_date=start_str, end_date=end_str,
            adjust='qfq'
        )
        return normalize_df(df)
    except Exception as e:
        logger.debug(f"[akshare_cn] {symbol} 失败: {e}")
        return None


# BaoStock 需要登录/注销，用模块级状态管理
_baostock_session: bool = False


def _ensure_baostock_login() -> bool:
    global _baostock_session
    if _baostock_session:
        return True
    try:
        import baostock as bs
        result = bs.login()
        if result.error_code == '0':
            _baostock_session = True
            return True
        logger.warning(f"[baostock] 登录失败: {result.error_msg}")
        return False
    except Exception as e:
        logger.debug(f"[baostock] 登录异常: {e}")
        return False


def fetch_baostock(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """BaoStock A股（历史数据完整，免费无限制）"""
    if interval not in ('1d', '1wk', '1mo'):
        return None
    if not _ensure_baostock_login():
        return None
    try:
        import baostock as bs
        sym_clean = symbol.split('.')[0]
        market = detect_market(symbol)
        prefix = 'sh' if market == 'CN_SH' else 'sz'
        bs_symbol = f'{prefix}.{sym_clean}'

        freq_map = {'1d': 'd', '1wk': 'w', '1mo': 'm'}
        freq = freq_map.get(interval, 'd')
        start, end = period_to_dates(period)

        rs = bs.query_history_k_data_plus(
            bs_symbol,
            "date,open,high,low,close,volume",
            start_date=start.strftime('%Y-%m-%d'),
            end_date=end.strftime('%Y-%m-%d'),
            frequency=freq,
            adjustflag='2'  # 后复权
        )
        rows = []
        while rs.error_code == '0' and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=rs.fields)
        return normalize_df(df)
    except Exception as e:
        logger.debug(f"[baostock] {symbol} 失败: {e}")
        return None


# ─────────────────────────────────────────────
# DataFetcher 主类
# ─────────────────────────────────────────────

# 各市场的数据源优先级
_SOURCE_PRIORITY: Dict[str, List[str]] = {
    'US':    ['yfinance', 'akshare_us'],
    'HK':    ['akshare_hk', 'yfinance'],
    'CN_SH': ['akshare_cn', 'baostock'],
    'CN_SZ': ['akshare_cn', 'baostock'],
}

_FETCH_FN = {
    'yfinance':   fetch_yfinance,
    'akshare_us': fetch_akshare_us,
    'akshare_hk': fetch_akshare_hk,
    'akshare_cn': fetch_akshare_cn,
    'baostock':   fetch_baostock,
}


class DataFetcher:
    """
    多数据源股票数据获取器，按市场类型自动选择最优来源，失败时自动降级。

    市场 → 数据源优先级:
        美股 US   : yfinance → akshare_us
        港股 HK   : akshare_hk → yfinance
        A股 CN    : akshare_cn → baostock
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str, period: str, interval: str) -> Path:
        safe = symbol.replace('.', '_').replace('/', '_')
        return self.cache_dir / f"{safe}_{period}_{interval}.parquet"

    def _try_sources(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        market = detect_market(symbol)
        sources = _SOURCE_PRIORITY.get(market, ['yfinance'])

        for source in sources:
            fn = _FETCH_FN.get(source)
            if fn is None:
                continue
            logger.info(f"[{source}] 正在获取 {symbol} ({market})...")
            t0 = time.time()
            df = fn(symbol, period, interval)
            elapsed = time.time() - t0
            if df is not None and not df.empty:
                logger.info(
                    f"[{source}] ✓ {symbol} 成功: {len(df)} 条, "
                    f"范围 {df.index[0].date()} ~ {df.index[-1].date()}, "
                    f"耗时 {elapsed:.1f}s"
                )
                return df
            logger.warning(f"[{source}] ✗ {symbol} 失败 ({elapsed:.1f}s)，尝试下一个源")

        logger.error(f"所有数据源均无法获取 {symbol}")
        return None

    # ── 公开接口（与旧版兼容）──

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        获取 OHLCV 行情数据（自动多源降级）

        Args:
            symbol  : 股票代码（美股 'AAPL'，港股 '0700.HK'，A股 '600519'）
            period  : 历史时长（'1y', '2y', '6mo' 等）
            interval: K 线周期（'1d'；日内仅 yfinance 支持）
            use_cache: 是否使用 6h 本地缓存
        """
        cache_file = self._get_cache_path(symbol, period, interval)

        if use_cache and cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if (datetime.now() - mtime).seconds < 3600 * 6:
                try:
                    df = pd.read_parquet(cache_file)
                    logger.debug(f"[cache] ✓ {symbol} 命中缓存，{len(df)} 条")
                    return df
                except Exception:
                    pass

        df = self._try_sources(symbol, period, interval)
        if df is not None:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass
        return df

    def fetch_multiple(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """批量获取多只股票，返回 {symbol: DataFrame}"""
        result = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, period=period, interval=interval)
            if df is not None and not df.empty:
                result[symbol] = df
            else:
                logger.warning(f"跳过 {symbol}（所有数据源失败）")
        return result

    def fetch_ticker_info(self, symbol: str) -> Dict:
        """获取股票基本信息（优先 yfinance，A股降级到 AkShare）"""
        market = detect_market(symbol)

        # A股从 AkShare 获取基本面
        if market in ('CN_SH', 'CN_SZ'):
            return self._fetch_info_akshare_cn(symbol)

        # 美股/港股用 yfinance
        try:
            import yfinance as yf
            info = yf.Ticker(symbol).info
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'eps': info.get('trailingEps'),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
            }
        except Exception as e:
            logger.error(f"fetch_ticker_info {symbol} 失败: {e}")
            return {'symbol': symbol, 'name': symbol}

    def _fetch_info_akshare_cn(self, symbol: str) -> Dict:
        """AkShare A股基本面信息"""
        try:
            import akshare as ak
            sym = symbol.split('.')[0]
            df = ak.stock_individual_info_em(symbol=sym)
            # df 是两列: item / value
            info_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            return {
                'symbol': symbol,
                'name': info_dict.get('股票简称', symbol),
                'sector': info_dict.get('行业', 'N/A'),
                'industry': info_dict.get('行业', 'N/A'),
                'market_cap': 0,
                'pe_ratio': None,
                'pb_ratio': None,
                'eps': None,
                'dividend_yield': 0,
                'beta': 1.0,
                'fifty_two_week_high': 0,
                'fifty_two_week_low': 0,
                'avg_volume': 0,
            }
        except Exception as e:
            logger.debug(f"akshare_cn info {symbol} 失败: {e}")
            return {'symbol': symbol, 'name': symbol}

    def fetch_news(self, symbol: str, max_items: int = 10) -> List[Dict]:
        """获取股票相关新闻（yfinance）"""
        try:
            import yfinance as yf
            news = yf.Ticker(symbol).news or []
            return [
                {
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'publisher': item.get('publisher', ''),
                    'published_at': datetime.fromtimestamp(
                        item.get('providerPublishTime', 0)
                    ).strftime('%Y-%m-%d %H:%M'),
                    'url': item.get('link', ''),
                }
                for item in news[:max_items]
            ]
        except Exception as e:
            logger.error(f"fetch_news {symbol} 失败: {e}")
            return []

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新收盘价"""
        df = self.fetch_ohlcv(symbol, period="5d", interval="1d", use_cache=False)
        if df is not None and not df.empty:
            return float(df['Close'].iloc[-1])
        return None
