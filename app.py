from __future__ import annotations
from typing import Dict, List
from datetime import datetime, timezone, timedelta
import math
import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="Options Chart AI Bot", page_icon="📈", layout="wide")
st.title("📈 Options Chart AI Bot — Streamlit")
st.caption("Educational only. Not financial advice. Options involve substantial risk.")

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def _rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / float(length)
    return series.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = _rma(up, length)
    avg_loss = _rma(down, length)
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = _true_range(df)
    return _rma(tr, length)

def _bbands(close: pd.Series, length: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    return pd.DataFrame({'BB_M': mid, 'BB_U': upper, 'BB_L': lower})

@st.cache_data(show_spinner=False)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed; upload a CSV instead.")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("No data returned — try a longer period or different interval.")
    df.rename(columns=str.capitalize, inplace=True)
    return df.dropna()

@st.cache_data(show_spinner=False)
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EMA20'] = _ema(df['Close'], 20)
    df['EMA50'] = _ema(df['Close'], 50)
    df['EMA200'] = _ema(df['Close'], 200)
    df['RSI'] = _rsi(df['Close'], 14)
    df['ATR'] = _atr(df, 14)
    bb = _bbands(df['Close'], 20, 2.0)
    df = pd.concat([df, bb], axis=1)
    tr = _true_range(df)
    df['RV'] = (tr / df['Close']).rolling(21, min_periods=21).std(ddof=0) * np.sqrt(252)
    df['EMA20_slope'] = df['EMA20'].diff(5)
    return df.dropna()


def key_levels(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    recent = df.tail(50)
    swing_high = recent['High'].rolling(10, min_periods=1).max().iloc[-1]
    swing_low = recent['Low'].rolling(10, min_periods=1).min().iloc[-1]
    if 'Volume' in df.columns:
        denom = float(recent['Volume'].sum())
        vwap = (recent['Close'] * recent['Volume']).sum() / (denom if denom != 0 else 1.0)
    else:
        vwap = float('nan')
    return {
        'last_close': float(last['Close']),
        'ema20': float(last['EMA20']),
        'ema50': float(last['EMA50']),
        'ema200': float(last['EMA200']),
        'swing_high_10': float(swing_high),
        'swing_low_10': float(swing_low),
        'bb_upper': float(last['BB_U']),
        'bb_lower': float(last['BB_L']),
        'vwap50': float(vwap) if not math.isnan(vwap) else float('nan'),
    }


def detect_regime(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    price = last['Close']
    ema20, ema50, ema200 = last['EMA20'], last['EMA50'], last['EMA200']
    rsi = last['RSI']
    slope = last['EMA20_slope']
    rv = float(df['RV'].iloc[-1])
    rv_series = df['RV'].dropna()
    if len(rv_series) >= 60:
        iv_rank = float((rv_series <= rv).mean())
    else:
        iv_rank = float('nan')
    if price > ema20 > ema50 > ema200 and rsi >= 55 and slope > 0:
        trend = 'bull'
        comment = 'Uptrend: stacked EMAs, positive momentum.'
    elif price < ema20 < ema50 < ema200 and rsi <= 45 and slope < 0:
        trend = 'bear'
        comment = 'Downtrend: stacked EMAs, negative momentum.'
    else:
        trend = 'range'
        comment = 'Mean-reverting / sideways conditions.'
    return {
        'trend': trend,
        'iv_proxy': rv,
        'iv_rank': iv_rank,
        'rsi': float(rsi),
        'comment': comment,
    }

@st.cache_data(show_spinner=False)
def get_next_events(ticker: str):
    t = yf.Ticker(ticker)
    next_earn = None
    try:
        edf = t.get_earnings_dates(limit=12)
        if edf is not None and not edf.empty:
            col = [c for c in edf.columns if 'Earnings' in c][0]
            future = edf[edf[col] >= pd.Timestamp.utcnow().tz_localize(None)]
            if not future.empty:
                next_earn = pd.to_datetime(future.iloc[0][col]).to_pydatetime()
    except Exception:
        pass
    next_div = None
    try:
        divs = t.dividends
        if divs is not None and not divs.empty:
            future_divs = divs[divs.index >= pd.Timestamp.utcnow().tz_localize(None)]
            if not future_divs.empty:
                next_div = future_divs.index[0].to_pydatetime()
    except Exception:
        pass
    return next_earn, next_div


def _bs_d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T) + 1e-12)

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_delta_call(S, K, T, sigma, r=0.0):
    d1 = _bs_d1(S, K, T, r, sigma)
    return _norm_cdf(d1)

def bs_delta_put(S, K, T, sigma, r=0.0):
    return bs_delta_call(S, K, T, sigma, r) - 1.0

@st.cache_data(show_spinner=False)
def fetch_option_chain(ticker: str, target_dte_low: int = 30, target_dte_high: int = 45):
    t = yf.Ticker(ticker)
    expiries = t.options
    if not expiries:
        raise RuntimeError("No option expirations found.")
    today = datetime.now(timezone.utc).date()
    target_mid = (target_dte_low + target_dte_high) / 2
    best = None
    best_diff = 10**9
    for e in expiries:
        try:
            d = datetime.fromisoformat(e).date()
        except Exception:
            continue
        dte = (d - today).days
        if dte <= 0:
            continue
        diff = abs(dte - target_mid)
        if (target_dte_low <= dte <= target_dte_high and diff < best_diff) or (best is None or diff < best_diff):
            best = (e, dte)
            best_diff = diff
    if best is None:
        future = []
        for e in expiries:
            try:
                d = datetime.fromisoformat(e).date()
            except Exception:
                continue
            dte = (d - today).days
            if dte > 0:
                future.append((e, dte))
        if not future:
            raise RuntimeError("No future expirations available.")
        best = min(future, key=lambda x: x[1])
    expiry, dte = best
    chain = t.option_chain(expiry)
    calls = chain.calls.rename(columns=str.lower)
    puts = chain.puts.rename(columns=str.lower)
    return expiry, dte, calls, puts


def pick_bull_put_spread(price: float, puts_df: pd.DataFrame, dte: int, target_delta=0.28, width=5):
    T = max(dte/365.0, 1/365)
    df = puts_df.copy()
    if df.empty:
        raise RuntimeError("Empty puts chain")
    if not {'impliedvolatility','strike','bid','ask'}.issubset(df.columns):
        raise RuntimeError("Missing columns in puts chain")
    df = df[df['impliedvolatility'].notna()]
    if df.empty:
        raise RuntimeError("No IV data in puts chain")
    df['delta'] = df.apply(lambda r: bs_delta_put(price, r['strike'], T, max(r['impliedvolatility'], 1e-6)), axis=1)
    df['abs_delta'] = df['delta'].abs()
    short = df.iloc[(df['abs_delta'] - target_delta).abs().argsort()].head(1).iloc[0]
    short_k = float(short['strike'])
    long_k = max(0.01, short_k - width)
    short_mid = (short['bid'] + short['ask'])/2
    long_row = df.iloc[(df['strike'] - long_k).abs().argsort()].head(1).iloc[0]
    long_mid = (long_row['bid'] + long_row['ask'])/2
    credit = float(max(0.01, short_mid - long_mid))
    max_loss = max(0.01, width - credit) * 100
    return {
        'short_strike': short_k,
        'long_strike': float(long_row['strike']),
        'credit_mid': float(round(credit, 2)),
        'width': width,
        'max_loss_per_contract': float(round(max_loss, 2)),
        'short_delta': float(round(short['delta'], 2))
    }


def pick_bear_call_spread(price: float, calls_df: pd.DataFrame, dte: int, target_delta=0.28, width=5):
    T = max(dte/365.0, 1/365)
    df = calls_df.copy()
    if df.empty:
        raise RuntimeError("Empty calls chain")
    if not {'impliedvolatility','strike','bid','ask'}.issubset(df.columns):
        raise RuntimeError("Missing columns in calls chain")
    df = df[df['impliedvolatility'].notna()]
    if df.empty:
        raise RuntimeError("No IV data in calls chain")
    df['delta'] = df.apply(lambda r: bs_delta_call(price, r['strike'], T, max(r['impliedvolatility'], 1e-6)), axis=1)
    df['abs_delta'] = df['delta'].abs()
    short = df.iloc[(df['delta'] - (1 - target_delta)).abs().argsort()].head(1).iloc[0]
    short_k = float(short['strike'])
    long_k = short_k + width
    short_mid = (short['bid'] + short['ask'])/2
    long_row = df.iloc[(df['strike'] - long_k).abs().argsort()].head(1).iloc[0]
    long_mid = (long_row['bid'] + long_row['ask'])/2
    credit = float(max(0.01, short_mid - long_mid))
    max_loss = max(0.01, width - credit) * 100
    return {
        'short_strike': short_k,
        'long_strike': float(long_row['strike']),
        'credit_mid': float(round(credit, 2)),
        'width': width,
        'max_loss_per_contract': float(round(max_loss, 2)),
        'short_delta': float(round(short['delta'], 2))
    }


def position_size(account_equity: float, risk_pct: float, max_loss_per_contract: float):
    risk_dollars = account_equity * (risk_pct/100.0)
    if max_loss_per_contract <= 0:
        return 0, risk_dollars
    contracts = int(max(0, math.floor(risk_dollars / max_loss_per_contract)))
    return contracts, risk_dollars

@st.cache_data(show_spinner=False)
def compute_betas(tickers: List[str], lookback_days=90):
    if yf is None:
        raise RuntimeError("yfinance not installed")
    if not tickers:
        raise RuntimeError("No tickers provided")
    base = [t for t in tickers if t]
    base = [t.upper() for t in base]
    if 'SPY' not in base:
        base.append('SPY')
    base = list(dict.fromkeys(base))
    data = yf.download(base, period=f"{lookback_days}d", interval='1d', auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise RuntimeError("No price data for betas")
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        if 'Close' not in data.columns:
            raise RuntimeError("No Close prices in data")
        close = data[['Close']]
        close.columns = [base[0]]
    close = close.dropna(how='all')
    rets = close.pct_change().dropna(how='all')
    if 'SPY' not in rets.columns:
        raise RuntimeError("SPY returns unavailable for beta calc")
    spy = rets['SPY']
    betas = {}
    for t in base:
        if t == 'SPY' or t not in rets.columns:
            continue
        x = rets[t].dropna()
        y = spy.reindex_like(x).dropna()
        m = min(len(x), len(y))
        if m < 2:
            continue
        x = x.tail(m)
        y = y.tail(m)
        cov = np.cov(x, y)[0,1]
        var = np.var(y)
        beta = cov / (var + 1e-12)
        betas[t] = float(beta)
    avail = [t for t in base if t in rets.columns]
    corr = rets[avail].corr() if len(avail) >= 2 else None
    return betas, corr

with st.sidebar:
    st.header("Inputs")
    mode = st.radio("Data Source", ["Ticker (yfinance)", "Upload CSV"], index=0)
    if mode == "Ticker (yfinance)":
        ticker = st.text_input("Ticker", value="SPY").upper().strip()
        period = st.selectbox("Period", ['1mo','3mo','6mo','1y','2y','5y','ytd','max'], index=2)
        interval = st.selectbox("Interval", ['1d','1h','30m','15m','5m'], index=0)
    else:
        ticker = st.text_input("Label for your CSV (e.g., SPY)", value="CSV").upper().strip()
        uploaded = st.file_uploader("Upload CSV (Datetime,Open,High,Low,Close,Volume)", type=["csv"]) 
        period = "custom"
        interval = "custom"
    st.divider()
    st.subheader("Strike Suggestions")
    target_dte_low = st.number_input("Target DTE (low)", value=30, min_value=7, max_value=120, step=1)
    target_dte_high = st.number_input("Target DTE (high)", value=45, min_value=7, max_value=180, step=1)
    target_delta_cr = st.slider("Target short Δ (credit spreads)", min_value=0.10, max_value=0.40, value=0.28, step=0.01)
    spread_width = st.number_input("Spread width ($)", value=5, min_value=1, max_value=50, step=1)
    st.divider()
    st.subheader("Position Sizing")
    acct_equity = st.number_input("Account equity ($)", value=10000, min_value=1000, step=500)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
    st.divider()
    st.subheader("Correlation Guardrails")
    portfolio_text = st.text_area("Current tickers (comma‑sep)", value="AAPL, MSFT, NVDA")
    direction_map_input = st.text_input("Directions (bullish/bearish, comma‑sep)", value="bullish, bullish, bullish")
    beta_warn = st.slider("Beta stack warning threshold (abs sum)", 0.5, 5.0, 3.0, 0.1)
    st.divider()
    st.subheader("Corporate Event Filter")
    use_blackout = st.checkbox("Skip trades near earnings/dividends", value=True)
    blackout_days = st.number_input("Blackout window (days)", value=7, min_value=1, max_value=30)
    st.divider()
    st.subheader("Diagnostics / Self‑Tests")
    if st.button("Run Self‑Tests"):
        try:
            d_call_atm = bs_delta_call(100, 100, 30/365, 0.3)
            d_put_atm = bs_delta_put(100, 100, 30/365, 0.3)
            assert 0.48 < d_call_atm < 0.52
            assert -0.52 < d_put_atm < -0.48
            cts1, riskd1 = position_size(10000, 1.0, 250)
            assert cts1 == 0 and abs(riskd1 - 100) < 1e-6
            cts2, _ = position_size(25000, 2.0, 250)
            assert cts2 == 2
            dates = pd.date_range('2024-01-01', periods=80)
            close = pd.Series(np.linspace(100, 120, 80))
            df_syn = pd.DataFrame({'Open': close, 'High': close+0.5, 'Low': close-0.5, 'Close': close, 'Volume': 1_000}, index=dates)
            df_syn = compute_indicators(df_syn)
            reg = detect_regime(df_syn)
            assert reg['trend'] in ['bull','range']
            fake_puts = pd.DataFrame({'strike':[95, 100, 105],'bid':[1.0, 2.0, 3.5],'ask':[1.2, 2.2, 3.8],'impliedvolatility':[0.25, 0.30, 0.35]})
            rec_pp = pick_bull_put_spread(110, fake_puts, 30, target_delta=0.28, width=5)
            assert rec_pp['credit_mid'] >= 0.01 and rec_pp['width'] == 5
            rsi_flat = _rsi(pd.Series([100.0]*40), 14).iloc[-1]
            assert 45 <= rsi_flat <= 55
            st.success("Self‑tests passed ✅")
        except AssertionError as ae:
            st.error(f"Self‑test failed: {ae}")
        except Exception as e:
            st.error(f"Self‑test error: {e}")

try:
    if mode == "Ticker (yfinance)":
        raw = load_data(ticker, period, interval)
    else:
        if uploaded is None:
            st.info("Upload a CSV to continue.")
            st.stop()
        df = pd.read_csv(uploaded)
        cols_map = {c.lower(): c for c in df.columns}
        dt_col = None
        for key in ['datetime','date','time','timestamp']:
            if key in cols_map:
                dt_col = cols_map[key]
                break
        if dt_col is None:
            st.error("CSV must include a Datetime/Date column.")
            st.stop()
        df.rename(columns={dt_col: 'Datetime'}, inplace=True)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        for col in ['open','high','low','close','volume']:
            if col in cols_map:
                df.rename(columns={cols_map[col]: col.capitalize()}, inplace=True)
        select_cols = ['Datetime','Open','High','Low','Close']
        if 'Volume' in df.columns:
            select_cols.append('Volume')
        raw = df[select_cols].sort_values('Datetime').set_index('Datetime')
    data = compute_indicators(raw)
    last = data.iloc[-1]
    levels = key_levels(data)
    regime = detect_regime(data)
except Exception as e:
    st.error(f"Error loading/computing data: {e}")
    st.stop()

next_earn, next_div = (None, None)
if mode == "Ticker (yfinance)":
    try:
        next_earn, next_div = get_next_events(ticker)
    except Exception:
        pass

with st.container():
    cols = st.columns(3)
    cols[0].metric("Last Close", f"{last['Close']:.2f}")
    cols[1].metric("RSI(14)", f"{last['RSI']:.1f}")
    iv_rank_disp = "n/a" if math.isnan(regime['iv_rank']) else f"{regime['iv_rank']:.2f}"
    cols[2].metric("IV Rank (proxy)", iv_rank_disp)
    st.write(f"Regime: **{regime['trend'].upper()}** — {regime['comment']}")
    if next_earn:
        st.info(f"Next earnings: {next_earn.strftime('%Y-%m-%d')} (local date)")
    if next_div:
        st.info(f"Next dividend: {next_div.strftime('%Y-%m-%d')}")

blackout_active = False
if use_blackout and (next_earn or next_div):
    now = datetime.now()
    dates = [d for d in [next_earn, next_div] if d is not None]
    soon = any((d - now) <= timedelta(days=int(blackout_days)) and (d - now).days >= 0 for d in dates)
    if soon:
        blackout_active = True
        st.warning("Blackout active — within earnings/dividend window. Strike suggestions are hidden. You can disable the filter in the sidebar.")

with st.container():
    st.subheader(f"{ticker} Chart")
    plot_cols = [c for c in ['Close','EMA20','EMA50','EMA200','BB_U','BB_M','BB_L'] if c in data.columns]
    st.line_chart(data[plot_cols])

strike_tab, sizing_tab, corr_tab, ts_tab = st.tabs(["🎯 Strike Suggestions", "📏 Position Sizing", "🧭 Correlation Guardrails", "🧩 thinkScript Generator"]) 

with strike_tab:
    if blackout_active:
        st.stop()
    try:
        expiry, dte, calls, puts = fetch_option_chain(ticker, int(target_dte_low), int(target_dte_high))
        st.markdown(f"**Selected Expiry:** `{expiry}`  | **DTE:** `{dte}`")
        price = float(last['Close'])
        if regime['trend'] == 'bull':
            rec = pick_bull_put_spread(price, puts, dte, target_delta=target_delta_cr, width=int(spread_width))
            st.markdown("**Suggested Bull Put Credit Spread**")
            st.json(rec)
            tos = f"SELL {ticker} {expiry} {rec['short_strike']:.0f} PUT / BUY {ticker} {expiry} {rec['long_strike']:.0f} PUT — Credit ≈ ${rec['credit_mid']:.2f} (width ${rec['width']}, Δ≈{rec['short_delta']})"
            st.code(tos)
        elif regime['trend'] == 'bear':
            rec = pick_bear_call_spread(price, calls, dte, target_delta=target_delta_cr, width=int(spread_width))
            st.markdown("**Suggested Bear Call Credit Spread**")
            st.json(rec)
            tos = f"SELL {ticker} {expiry} {rec['short_strike']:.0f} CALL / BUY {ticker} {expiry} {rec['long_strike']:.0f} CALL — Credit ≈ ${rec['credit_mid']:.2f} (width ${rec['width']}, Δ≈{rec['short_delta']})"
            st.code(tos)
        else:
            put_leg = pick_bull_put_spread(price, puts, dte, target_delta=target_delta_cr, width=int(spread_width))
            call_leg = pick_bear_call_spread(price, calls, dte, target_delta=target_delta_cr, width=int(spread_width))
            credit_ic = round(put_leg['credit_mid'] + call_leg['credit_mid'], 2)
            st.markdown("**Suggested Iron Condor**")
            ic = {'put_short': put_leg['short_strike'], 'put_long': put_leg['long_strike'], 'put_credit_mid': put_leg['credit_mid'], 'call_short': call_leg['short_strike'], 'call_long': call_leg['long_strike'], 'call_credit_mid': call_leg['credit_mid'], 'total_credit_mid': credit_ic, 'width_each_side': int(spread_width)}
            st.json(ic)
            tos = (f"SELL {ticker} {expiry} {call_leg['short_strike']:.0f} CALL / BUY {ticker} {expiry} {call_leg['long_strike']:.0f} CALL + "
                   f"SELL {ticker} {expiry} {put_leg['short_strike']:.0f} PUT / BUY {ticker} {expiry} {put_leg['long_strike']:.0f} PUT — "
                   f"Total credit ≈ ${credit_ic:.2f} (width ${int(spread_width)} each side)")
            st.code(tos)
    except Exception as e:
        st.warning(f"Could not fetch/compute strike suggestions: {e}")

with sizing_tab:
    if blackout_active:
        st.stop()
    try:
        price = float(last['Close'])
        expiry, dte, calls, puts = fetch_option_chain(ticker, int(target_dte_low), int(target_dte_high))
        if regime['trend'] == 'bear':
            rec = pick_bear_call_spread(price, calls, dte, target_delta=target_delta_cr, width=int(spread_width))
        else:
            rec = pick_bull_put_spread(price, puts, dte, target_delta=target_delta_cr, width=int(spread_width))
        contracts, risk_dollars = position_size(float(acct_equity), float(risk_pct), rec['max_loss_per_contract'])
        st.markdown("**Position Size Suggestion**")
        st.write(f"Account: ${acct_equity:,.0f} | Risk/Trade: {risk_pct:.1f}% → ${risk_dollars:,.2f}")
        st.write(f"Max loss/contract: ${rec['max_loss_per_contract']:.2f}")
        st.subheader(f"→ Contracts: {contracts}")
        if contracts == 0:
            st.info("Risk per trade is smaller than max loss per contract. Reduce width or increase risk %.")
    except Exception as e:
        st.warning(f"Sizing unavailable: {e}")

with corr_tab:
    try:
        tickers = [t.strip().upper() for t in portfolio_text.split(',') if t.strip()]
        if ticker not in tickers:
            tickers.append(ticker)
        betas, corr = compute_betas(tickers)
        dir_vals = [d.strip().lower() for d in direction_map_input.split(',')]
        dirs = {}
        for i, t in enumerate(tickers):
            sign = 1
            if i < len(dir_vals) and dir_vals[i].startswith('bear'):
                sign = -1
            dirs[t] = sign
        beta_stack = 0.0
        rows = []
        for t, b in (betas or {}).items():
            beta_stack += b * dirs.get(t, 1)
            rows.append({'Ticker': t, 'Beta vs SPY': round(b, 2), 'Direction': 'Bull' if dirs.get(t,1)>0 else 'Bear'})
        st.markdown("**Portfolio Betas (incl. candidate)**")
        st.table(pd.DataFrame(rows))
        st.write(f"**Net beta-weighted bias:** {beta_stack:.2f} (threshold {beta_warn:.2f})")
        if abs(beta_stack) > float(beta_warn):
            st.warning("Beta stack exceeds threshold — consider reducing exposure or choosing a lower‑beta/uncorrelated ticker.")
        if corr is not None:
            st.markdown("**Correlation (close‑to‑close returns, 90d)**")
            st.dataframe(corr.round(2))
    except Exception as e:
        st.warning(f"Guardrails unavailable: {e}")

with ts_tab:
    st.subheader("Generate thinkScript")
    ts_delta = target_delta_cr
    ts_tol = 0.03
    ts_minD = int(target_dte_low)
    ts_maxD = int(target_dte_high)
    ts_put_scan = f"""
# Scan: Bull Put short leg near target Δ and DTE window
input targetDelta = {ts_delta:.2f};
input tolerance   = {ts_tol:.2f};
input minDTE      = {ts_minD};
input maxDTE      = {ts_maxD};
# Works on OptionHacker study filter
def d  = AbsValue(Delta());
def dte = DaysToExpiration();
plot scan = IsPut() and d >= targetDelta - tolerance and d <= targetDelta + tolerance and dte >= minDTE and dte <= maxDTE;
"""
    st.code(ts_put_scan, language='thinkscript')
    ts_call_scan = f"""
# Scan: Bear Call short leg near target Δ and DTE window
input targetDelta = {ts_delta:.2f};
input tolerance   = {ts_tol:.2f};
input minDTE      = {ts_minD};
input maxDTE      = {ts_maxD};
# Works on OptionHacker study filter
# For calls, we want OTM short call delta ≈ 1 - targetDelta
plot scan = IsCall() and (1 - Delta()) >= targetDelta - tolerance and (1 - Delta()) <= targetDelta + tolerance and DaysToExpiration() between minDTE and maxDTE;
"""
    st.code(ts_call_scan, language='thinkscript')
    ts_chain_col = f"""
# Column: label 'TARGET' for contracts ~target Δ within DTE window
input targetDelta = {ts_delta:.2f};
input tolerance   = {ts_tol:.2f};
input minDTE      = {ts_minD};
input maxDTE      = {ts_maxD};

def d  = AbsValue(Delta());
def dte = DaysToExpiration();
plot isTarget = d >= targetDelta - tolerance and d <= targetDelta + tolerance and dte >= minDTE and dte <= maxDTE;
AssignBackgroundColor(if isTarget then Color.YELLOW else Color.CURRENT);
AddLabel(isTarget, "TARGET", Color.BLACK);
"""
    st.code(ts_chain_col, language='thinkscript')
    st.download_button("Download thinkScript (3 snippets).txt", data=(ts_put_scan + "\n\n" + ts_call_scan + "\n\n" + ts_chain_col).encode(), file_name=f"thinkscript_targetDelta_{ts_delta:.2f}.txt", mime="text/plain")

st.subheader("Export — thinkorswim Order Plan")
iv_rank_disp_txt = 'n/a' if math.isnan(regime['iv_rank']) else f"{regime['iv_rank']:.2f}"
plan_lines = [
    f"Ticker: {ticker}",
    f"Regime: {regime['trend'].upper()} | IV proxy: {regime['iv_proxy']:.3f} | IV Rank: {iv_rank_disp_txt} | RSI: {regime['rsi']:.1f}",
    f"Context: {regime['comment']}",
]
if next_earn:
    plan_lines.append(f"Next earnings: {next_earn.strftime('%Y-%m-%d')}")
if next_div:
    plan_lines.append(f"Next dividend: {next_div.strftime('%Y-%m-%d')}")
plan_lines += [
    "",
    "thinkorswim staging:",
    "- Trade → All Products → Spread = Vertical (for credits/debits)",
    "- Choose ~30–45 DTE; sort strikes by Delta; right‑click → Analyze trade",
    "- Analyze → Risk Profile: add price slices (Current, −1×ATR, +1×ATR)",
]
try:
    if not blackout_active:
        expiry, dte, calls, puts = fetch_option_chain(ticker, int(target_dte_low), int(target_dte_high))
        price = float(last['Close'])
        if regime['trend'] == 'bear':
            rec = pick_bear_call_spread(price, calls, dte, target_delta=target_delta_cr, width=int(spread_width))
            plan_lines += ["", "Suggested Bear Call Credit Spread:", f"SELL {ticker} {expiry} {rec['short_strike']:.0f} CALL / BUY {ticker} {expiry} {rec['long_strike']:.0f} CALL — credit≈${rec['credit_mid']:.2f}; width ${int(spread_width)}; Δ≈{rec['short_delta']}"]
        elif regime['trend'] == 'bull':
            rec = pick_bull_put_spread(price, puts, dte, target_delta=target_delta_cr, width=int(spread_width))
            plan_lines += ["", "Suggested Bull Put Credit Spread:", f"SELL {ticker} {expiry} {rec['short_strike']:.0f} PUT / BUY {ticker} {expiry} {rec['long_strike']:.0f} PUT — credit≈${rec['credit_mid']:.2f}; width ${int(spread_width)}; Δ≈{rec['short_delta']}"]
        else:
            put_leg = pick_bull_put_spread(price, puts, dte, target_delta=target_delta_cr, width=int(spread_width))
            call_leg = pick_bear_call_spread(price, calls, dte, target_delta=target_delta_cr, width=int(spread_width))
            credit_ic = round(put_leg['credit_mid'] + call_leg['credit_mid'], 2)
            plan_lines += ["", "Suggested Iron Condor:", f"CALL: SELL {ticker} {expiry} {call_leg['short_strike']:.0f}C / BUY {ticker} {expiry} {call_leg['long_strike']:.0f}C", f"PUT : SELL {ticker} {expiry} {put_leg['short_strike']:.0f}P / BUY {ticker} {expiry} {put_leg['long_strike']:.0f}P — total credit≈${credit_ic:.2f}; width ${int(spread_width)} each"]
    else:
        plan_lines += ["", "(Strike suggestions suppressed due to earnings/dividend blackout)"]
except Exception as e:
    plan_lines += ["", f"(Strike suggestions unavailable: {e})"]
plan = "\n".join(plan_lines)
st.code(plan)
st.download_button("Download Order Plan (.txt)", data=plan.encode(), file_name=f"{ticker}_order_plan.txt", mime="text/plain")
st.caption("Tip: In thinkorswim → Analyze → Risk Profile, add price slices (Current, −1×ATR, +1×ATR) to visualize P/L against expected move.")
