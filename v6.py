import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from datetime import datetime
import time
import requests

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="TSLA vs UVXY 三層訊號系統", page_icon="🎯", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  .main { background-color: #0e1117; }

  /* ── Metric cards ── */
  .metric-card {
      background:#1c1f26; border-radius:10px; padding:13px 15px;
      border:1px solid #2d3139; text-align:center; height:100%;
  }
  .metric-label { color:#8b8fa8; font-size:0.68rem; letter-spacing:0.06em; text-transform:uppercase; }
  .metric-value { font-size:1.35rem; font-weight:700; margin-top:4px; line-height:1.2; }
  .metric-sub   { font-size:0.75rem; font-weight:600; margin-top:3px; }

  /* ── Signal boxes ── */
  .sig-layer1 {
      border-radius:12px; padding:14px 20px; margin:6px 0;
      background:#2b2200; border:2px solid #f6c90e; text-align:center;
  }
  .sig-layer2-buy {
      border-radius:12px; padding:16px 20px; margin:6px 0;
      background:#0a2b16; border:2px solid #00d97e; text-align:center;
  }
  .sig-layer2-sell {
      border-radius:12px; padding:16px 20px; margin:6px 0;
      background:#2b0a0a; border:2px solid #e84045; text-align:center;
  }
  .sig-layer3 {
      border-radius:12px; padding:14px 20px; margin:6px 0;
      background:#1a0a2b; border:2px solid #b57bee; text-align:center;
  }
  .sig-normal {
      border-radius:12px; padding:12px 20px; margin:6px 0;
      background:#1c1f26; border:1px solid #2d3139; text-align:center;
  }
  .sig-title  { font-size:1.5rem; font-weight:800; }
  .sig-detail { font-size:0.87rem; color:#c9cdd8; margin-top:5px; line-height:1.5; }

  /* ── Section titles ── */
  .section-title {
      font-size:0.92rem; font-weight:700; color:#c9cdd8;
      border-left:3px solid #5c7cfa; padding-left:8px; margin:18px 0 8px 0;
  }

  /* ── Track bar ── */
  .track-card {
      background:#161920; border-radius:8px; padding:10px 14px;
      border:1px solid #2d3139; margin:4px 0; font-size:0.82rem; color:#c9cdd8;
  }

  /* ── SPY badge ── */
  .spy-on  { background:#0a2020; border:1px solid #00d97e; border-radius:6px;
             padding:4px 10px; color:#00d97e; font-size:0.78rem; font-weight:700; }
  .spy-off { background:#1c1f26; border:1px solid #2d3139; border-radius:6px;
             padding:4px 10px; color:#8b8fa8; font-size:0.78rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════
def send_telegram(message: str):
    try:
        token   = st.secrets["telegram"]["bot_token"]
        chat_id = st.secrets["telegram"]["chat_id"]
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=8,
        )
    except Exception as e:
        st.sidebar.warning(f"Telegram 失敗: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=55)
def fetch_1m(ticker: str, bars: int = 80) -> pd.DataFrame:
    df = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
    if df.empty:
        return df
    df = df.tail(bars).copy()
    df.index = pd.to_datetime(df.index)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

@st.cache_data(ttl=290)
def fetch_5m(ticker: str, bars: int = 30) -> pd.DataFrame:
    df = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=True)
    if df.empty:
        return df
    df = df.tail(bars).copy()
    df.index = pd.to_datetime(df.index)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# TREND ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def calc_trend(prices: pd.Series, consec_required: int = 2) -> dict:
    """
    Linear regression trend analysis.
    Returns: slope (%/bar), r2, direction (+1/-1/0),
             consecutive (tail candles same direction),
             confirmed (bool), pct_move (first→last %)
    """
    n = len(prices)
    if n < 3:
        return dict(slope=0, r2=0, direction=0, consecutive=0, confirmed=False, pct_move=0)
    x = np.arange(n, dtype=float)
    y = prices.values.astype(float)
    slope, intercept, r, p, se = linregress(x, y)
    r2        = r ** 2
    norm_slope = (slope / float(np.mean(y))) * 100
    diffs     = np.diff(y)
    last_sign = np.sign(diffs[-1])
    consec    = 1
    for d in reversed(diffs[:-1]):
        if np.sign(d) == last_sign and last_sign != 0:
            consec += 1
        else:
            break
    direction = int(np.sign(norm_slope))
    confirmed = (consec >= consec_required) and (r2 >= 0.45)
    pct_move  = (y[-1] - y[0]) / y[0] * 100
    return dict(slope=norm_slope, r2=r2, direction=direction,
                consecutive=consec, confirmed=confirmed, pct_move=pct_move)

def resistance_ratio(uvxy_pct: float, tsla_pct: float) -> float:
    """
    How much TSLA resisted UVXY's move.
    UVXY up +1%, TSLA only down -0.1% → ratio = 0.1 (strong resistance)
    Perfect negative correlation → ratio = 1.0
    Returns 0–∞ (lower = stronger resistance when directions oppose)
    """
    if abs(uvxy_pct) < 0.01:
        return 1.0
    expected_tsla = -uvxy_pct          # perfect negative correlation
    actual_tsla   = tsla_pct
    # ratio of actual response vs expected response
    ratio = abs(actual_tsla) / abs(expected_tsla)
    return round(ratio, 3)

def signal_stars(strength: float) -> str:
    if strength >= 4.0:   return "⭐⭐⭐"
    elif strength >= 2.0: return "⭐⭐"
    else:                 return "⭐"

def hs_dynamic_size(corr20: float, uvxy_chg_abs: float,
                    recent_pnls: list, et_hour: int,
                    uvxy_rsi: float = 50.0,
                    uvxy_mom5: float = 0.0,
                    tsla_rsi: float = 50.0,
                    use_filter_a: bool = False,
                    use_filter_b: bool = False,
                    use_filter_c: bool = False) -> tuple:
    """
    Dynamic position sizing for High-Sensitivity mode.

    Base factors (always active):
      Corr -0.7~-0.5 → WR 74.1% → 2x
      Corr -0.5~-0.3 → WR 64.7% → 1.5x
      Corr -0.3~ 0   → WR 54.5% → 1x
      Corr  < -0.7   → WR 45.6% → 0.5x
      Corr  > 0      → WR 37.5% → 0x SKIP

    Optional filters (user-toggled):
      Filter A – UVXY RSI:  <30→×1.3  30-45→×0.7(trap)  >65→×0.8
      Filter B – UVXY 5-bar momentum: 0~0.3%→×1.4  -1~-0.3%→×1.2  >1%→×0.6
      Filter C – TSLA RSI:  40-50→×1.2  50-60→×0.7(trap)  extreme→×0.8

    Returns (multiplier, reason, quality_label)
    """
    import math
    reasons = []
    mult    = 1.0

    # ── Factor 1: 20-bar rolling correlation (always active) ─────────────
    if math.isnan(corr20):
        return 0.0, "相關係數數據不足", "跳過"
    if corr20 > 0:
        return 0.0, f"正相關({corr20:.2f})負相關失效", "跳過"
    elif corr20 > -0.3:
        mult *= 0.5; reasons.append(f"相關偏弱({corr20:.2f})→0.5x")
    elif corr20 > -0.5:
        mult *= 1.0; reasons.append(f"相關正常({corr20:.2f})→1x")
    elif corr20 > -0.7:
        mult *= 2.0; reasons.append(f"相關強({corr20:.2f})→2x⭐")
    else:
        mult *= 0.5; reasons.append(f"相關極強({corr20:.2f})均值回歸→0.5x")

    # ── Factor 2: Recent 10-trade rolling win rate (always active) ───────
    if len(recent_pnls) >= 5:
        rwr = sum(1 for p in recent_pnls[-10:] if p > 0) / min(len(recent_pnls), 10)
        if rwr >= 0.8:
            mult *= 0.5; reasons.append(f"近期過熱WR={rwr*100:.0f}%→0.5x")
        elif rwr < 0.4:
            mult *= 0.5; reasons.append(f"冷場WR={rwr*100:.0f}%→0.5x")
        elif rwr >= 0.6:
            mult *= 1.3; reasons.append(f"熱手WR={rwr*100:.0f}%→1.3x")

    # ── Factor 3: UVXY move magnitude (always active) ────────────────────
    if 0.25 <= uvxy_chg_abs < 0.5:
        mult *= 1.2; reasons.append(f"UVXY大幅{uvxy_chg_abs:.2f}%→1.2x")
    elif uvxy_chg_abs >= 0.5:
        mult *= 0.8; reasons.append(f"UVXY極端{uvxy_chg_abs:.2f}%→0.8x")

    # ── Factor 4: Time-of-day filter (always active) ─────────────────────
    if et_hour in [10, 14]:
        mult *= 0.5; reasons.append(f"{et_hour}:xx ET低勝時段→0.5x")

    # ── Filter A: UVXY RSI filter (user-toggled) ──────────────────────────
    # Real data: RSI<30→WR70%  RSI30-45→WR42%(trap!)  RSI>65→WR45%
    if use_filter_a and not math.isnan(uvxy_rsi):
        if uvxy_rsi < 30:
            mult *= 1.3; reasons.append(f"[A]UVXY超賣RSI={uvxy_rsi:.0f}→1.3x")
        elif uvxy_rsi < 45:
            mult *= 0.7; reasons.append(f"[A]UVXY RSI陷阱={uvxy_rsi:.0f}(30-45)→0.7x")
        elif uvxy_rsi > 65:
            mult *= 0.8; reasons.append(f"[A]UVXY偏熱RSI={uvxy_rsi:.0f}→0.8x")
        else:
            reasons.append(f"[A]UVXY RSI={uvxy_rsi:.0f}正常")

    # ── Filter B: UVXY 5-bar momentum filter (user-toggled) ──────────────
    # Real data: 微升0~0.3%→WR72%  跌-1~-0.3%→WR61%  大升>1%→WR38%
    if use_filter_b and not math.isnan(uvxy_mom5):
        if 0 <= uvxy_mom5 < 0.3:
            mult *= 1.4; reasons.append(f"[B]UVXY緩升{uvxy_mom5:+.2f}%→1.4x⭐")
        elif -1.0 <= uvxy_mom5 < -0.3:
            mult *= 1.2; reasons.append(f"[B]UVXY適度跌{uvxy_mom5:+.2f}%→1.2x")
        elif uvxy_mom5 >= 1.0:
            mult *= 0.6; reasons.append(f"[B]UVXY急升{uvxy_mom5:+.2f}%→0.6x⚠️")
        else:
            reasons.append(f"[B]UVXY動能{uvxy_mom5:+.2f}%正常")

    # ── Filter C: TSLA RSI filter (user-toggled) ──────────────────────────
    # Real data: RSI40-50→WR60%  RSI50-60→WR42%(trap!)  extreme→WR46%
    if use_filter_c and not math.isnan(tsla_rsi):
        if 40 <= tsla_rsi < 50:
            mult *= 1.2; reasons.append(f"[C]TSLA RSI最佳={tsla_rsi:.0f}(40-50)→1.2x")
        elif 50 <= tsla_rsi < 60:
            mult *= 0.7; reasons.append(f"[C]TSLA RSI陷阱={tsla_rsi:.0f}(50-60)→0.7x⚠️")
        elif tsla_rsi < 30 or tsla_rsi > 70:
            mult *= 0.8; reasons.append(f"[C]TSLA RSI極端={tsla_rsi:.0f}→0.8x")
        else:
            reasons.append(f"[C]TSLA RSI={tsla_rsi:.0f}正常")

    # Round to 0.5x steps, cap at 3x
    mult = round(min(max(mult, 0.0), 3.0) * 2) / 2

    # Quality label
    if mult == 0:     quality = "跳過"
    elif mult <= 0.5: quality = "⚠️ 輕倉"
    elif mult <= 1.0: quality = "▪ 正常"
    elif mult <= 1.5: quality = "▲ 加碼"
    elif mult <= 2.0: quality = "⭐ 重倉"
    else:             quality = "🔥 最重"

    return mult, "  |  ".join(reasons), quality


def vol_ratio(df: pd.DataFrame, window: int = 10) -> float:
    """Current bar volume vs rolling average."""
    if len(df) < window + 1:
        return 1.0
    vols = df["Volume"].values.astype(float)
    avg  = float(np.mean(vols[-(window+1):-1]))
    curr = float(vols[-1])
    return curr / avg if avg > 0 else 1.0

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
def _empty_corr():
    return pd.DataFrame({"time": pd.Series(dtype="datetime64[ns]"),
                         "corr": pd.Series(dtype="float64")})

defaults = {
    "corr_history":        _empty_corr(),
    "last_alert_time":     None,
    "last_warn_time":      None,   # Layer 1 cooldown
    "last_exit_time":      None,   # Layer 3 cooldown
    "last_hs_alert_time":  None,   # High-sensitivity cooldown
    "hs_recent_pnls":      [],     # Rolling last 10 HS trade PnLs for dynamic sizing
    "signal_history":      [],
    "active_signal":       None,
    "active_signal_time":  None,
    "active_signal_entry": None,   # entry price when signal fired
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if not isinstance(st.session_state.corr_history, pd.DataFrame):
    st.session_state.corr_history = _empty_corr()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ 控制面板")

    display_bars   = st.slider("K線顯示數量",             10, 40, 20)
    consec_req     = st.slider("UVXY 連續確認根數",         2,  5,  2)
    cooldown_min   = st.slider("Telegram 冷卻（分鐘）",    1, 30,  5)
    signal_timeout = st.slider("訊號追蹤失效（分鐘）",     5, 60, 20)

    st.divider()
    st.markdown("### 📐 靈敏度")
    min_uvxy_slope     = st.slider("UVXY 最低斜率 (%/根)",        0.05, 1.0,  0.15, step=0.05)
    min_tsla_response  = st.slider("TSLA 視為已反應 (%)",          0.1,  2.0,  0.30, step=0.1)
    resistance_thresh  = st.slider("抵抗力比值門檻（Layer1）",     0.05, 0.8,  0.30, step=0.05,
                                    help="TSLA跌幅÷UVXY升幅，低於此值觸發預警")
    rsi_overbought     = st.slider("RSI 超買出場閾值（Layer3）",   65,   85,   70)
    vol_spike_mult     = st.slider("成交量爆增倍數（Layer3）",      1.5,  5.0,  2.5, step=0.5)

    st.divider()

    # ── SPY Filter Toggle ──────────────────────────────────────────────────────
    st.markdown("### 🔍 SPY 大盤過濾")
    use_spy_filter = st.toggle(
        "啟用 SPY 過濾",
        value=False,
        help="開啟後：若 SPY 同步下跌確認，BUY_TSLA 訊號會被降級或過濾，避免大盤拖累假訊號"
    )
    if use_spy_filter:
        spy_filter_strength = st.radio(
            "過濾強度",
            ["寬鬆（降低訊號強度）", "嚴格（直接過濾訊號）"],
            index=0,
        )
        st.markdown("<div class='spy-on'>✅ SPY 過濾已啟用</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='spy-off'>○ SPY 過濾已關閉</div>", unsafe_allow_html=True)

    st.divider()

    # ── High-Sensitivity Mode ──────────────────────────────────────────────────
    st.markdown("### ⚡ 高敏感度模式")
    use_hs_mode = st.toggle(
        "啟用高敏感度偵測",
        value=False,
        help="每分鐘直接比較最新2根K線方向：UVXY升但TSLA未跌，或UVXY跌但TSLA未升，即時發出提醒"
    )
    if use_hs_mode:
        hs_uvxy_min   = st.slider("HS: UVXY最低變動 (%)", 0.05, 1.0, 0.15, step=0.05,
                                   help="UVXY單根K線需變動超過此幅度才觸發")
        hs_tsla_max   = st.slider("HS: TSLA最大容許反應 (%)", 0.0, 1.0, 0.10, step=0.05,
                                   help="TSLA反應低於此幅度視為「未跟隨」")
        hs_cooldown   = st.slider("HS: 冷卻時間（分鐘）", 1, 10, 2)

        st.markdown("**📊 勝率提升過濾器**")
        st.markdown(
            "<small style='color:#8b8fa8'>根據175筆真實回測，逐步提升勝率</small>",
            unsafe_allow_html=True,
        )

        use_filter_a = st.toggle(
            "過濾A：UVXY RSI 陷阱過濾",
            value=False,
            help="RSI<30(超賣)→加碼1.3x WR=70% | RSI30-45(陷阱)→減碼0.7x WR=42% | RSI>65→減碼0.8x | 預期+5~8%勝率",
        )
        _fa_color = "#00d97e" if use_filter_a else "#8b8fa8"
        st.markdown(
            f"<div style='font-size:0.73rem;color:{_fa_color};margin:-6px 0 6px 0'>"
            f"{'✅ 已啟用  WR預期 ~63%' if use_filter_a else '○ 關閉'}</div>",
            unsafe_allow_html=True,
        )

        use_filter_b = st.toggle(
            "過濾B：UVXY 5棒動能過濾",
            value=False,
            help="緩升0~0.3%→加碼1.4x WR=72% | 適度跌-1~-0.3%→加碼1.2x WR=61% | 急升>1%→減碼0.6x WR=38% | 預期+5~10%勝率",
        )
        _fb_color = "#00d97e" if use_filter_b else "#8b8fa8"
        st.markdown(
            f"<div style='font-size:0.73rem;color:{_fb_color};margin:-6px 0 6px 0'>"
            f"{'✅ 已啟用  WR預期 ~67%' if use_filter_b else '○ 關閉'}</div>",
            unsafe_allow_html=True,
        )

        use_filter_c = st.toggle(
            "過濾C：TSLA RSI 陷阱過濾",
            value=False,
            help="RSI40-50(最佳)→加碼1.2x WR=60% | RSI50-60(陷阱)→減碼0.7x WR=42% | 極端→減碼0.8x | 預期+5~8%勝率",
        )
        _fc_color = "#00d97e" if use_filter_c else "#8b8fa8"
        st.markdown(
            f"<div style='font-size:0.73rem;color:{_fc_color};margin:-6px 0 6px 0'>"
            f"{'✅ 已啟用  WR預期 ~75%' if use_filter_c else '○ 關閉'}</div>",
            unsafe_allow_html=True,
        )

        # Summary badge
        active_filters = sum([use_filter_a, use_filter_b, use_filter_c])
        expected_wr = {0: "57%", 1: "~63%", 2: "~70%", 3: "~80%"}[active_filters]
        badge_color = "#00d97e" if active_filters >= 2 else ("#f6c90e" if active_filters == 1 else "#8b8fa8")
        st.markdown(
            f"<div style='background:#111;border:1px solid {badge_color};border-radius:6px;"
            f"padding:6px 10px;font-size:0.75rem;color:{badge_color};margin-top:4px'>"
            f"⚡ 高敏感度  {active_filters}/3 過濾器開啟  預期勝率 {expected_wr}</div>",
            unsafe_allow_html=True,
        )
    else:
        hs_uvxy_min   = 0.15
        hs_tsla_max   = 0.10
        hs_cooldown   = 2
        use_filter_a  = False
        use_filter_b  = False
        use_filter_c  = False

    st.divider()
    auto_refresh = st.toggle("每分鐘自動刷新", value=True)
    if st.button("🔄 立即刷新"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### 📋 訊號紀錄")
    if st.session_state.signal_history:
        for entry in reversed(st.session_state.signal_history[-15:]):
            if "🟢" in entry or "兌現" in entry:
                color = "#00d97e"
            elif "🔴" in entry or "出場" in entry:
                color = "#e84045"
            elif "🟡" in entry:
                color = "#f6c90e"
            elif "紫" in entry or "exit" in entry.lower():
                color = "#b57bee"
            else:
                color = "#8b8fa8"
            st.markdown(
                f"<div style='color:{color};font-size:0.76rem;margin:2px 0'>{entry}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<small style='color:#8b8fa8'>尚無訊號</small>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🎯 TSLA vs UVXY 三層訊號系統")
spy_badge = (
    "<span class='spy-on'>SPY過濾 ON</span>" if use_spy_filter
    else "<span class='spy-off'>SPY過濾 OFF</span>"
)
if use_hs_mode:
    _active_f = sum([use_filter_a, use_filter_b, use_filter_c])
    _f_label  = f"{_active_f}/3過濾" if _active_f > 0 else "無過濾"
    hs_badge = (
        "<span style='background:#1a1a0a;border:1px solid #f6c90e;border-radius:5px;"
        f"padding:3px 8px;color:#f6c90e;font-size:0.76rem;font-weight:700'>"
        f"⚡ 高敏感 {_f_label}</span>"
    )
else:
    hs_badge = (
        "<span style='background:#1c1f26;border:1px solid #2d3139;border-radius:5px;"
        "padding:3px 8px;color:#8b8fa8;font-size:0.76rem'>高敏感度 OFF</span>"
    )
st.markdown(
    f"<small style='color:#8b8fa8'>預警 → 入場 → 出場 三層架構 · 抵抗力偵測 · 多框架趨勢引擎</small>"
    f" &nbsp; {spy_badge} &nbsp; {hs_badge}",
    unsafe_allow_html=True,
)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# FETCH DATA
# ═══════════════════════════════════════════════════════════════════════════════
tickers_to_fetch = ["TSLA", "UVXY"]
if use_spy_filter:
    tickers_to_fetch.append("SPY")

with st.spinner("載入市場數據…"):
    tsla_1m = fetch_1m("TSLA", bars=max(display_bars + 20, 80))
    uvxy_1m = fetch_1m("UVXY", bars=max(display_bars + 20, 80))
    tsla_5m = fetch_5m("TSLA", bars=30)
    uvxy_5m = fetch_5m("UVXY", bars=30)
    spy_1m  = fetch_1m("SPY",  bars=max(display_bars + 20, 80)) if use_spy_filter else pd.DataFrame()

data_ok = not tsla_1m.empty and not uvxy_1m.empty
now     = datetime.now()

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
# Output variables
layer1_active    = False   # Warning: TSLA resisting UVXY rise
layer2_signal    = None    # "BUY_TSLA" | "SELL_TSLA" | None
layer3_exit      = False   # Exit signal
layer2_stars     = ""
layer2_strength  = 0.0
layer1_reason    = ""
layer2_reason    = ""
layer3_reason    = ""
spy_filtered     = False   # Whether SPY filter suppressed a signal

uvxy_trend_1m   = {}
tsla_trend_1m   = {}
uvxy_trend_5m   = {}
tsla_trend_5m   = {}
spy_trend_1m    = {}
corr_value      = None
multi_tf_agree  = False
res_ratio       = 1.0
tsla_rsi1       = None
tsla_vol_ratio  = 1.0

if data_ok:
    # ── Pearson correlation ───────────────────────────────────────────────────
    common_1m = tsla_1m.index.intersection(uvxy_1m.index)
    if len(common_1m) >= 10:
        corr_value = float(tsla_1m.loc[common_1m, "Close"].corr(
                            uvxy_1m.loc[common_1m, "Close"]))
        new_row  = pd.DataFrame({"time": [pd.Timestamp(now)], "corr": [corr_value]})
        existing = st.session_state.corr_history
        if not isinstance(existing, pd.DataFrame):
            existing = _empty_corr()
        st.session_state.corr_history = pd.concat(
            [existing, new_row], ignore_index=True
        ).tail(120).reset_index(drop=True)

    # ── Trend calculations ────────────────────────────────────────────────────
    uvxy_trend_1m = calc_trend(uvxy_1m["Close"].tail(display_bars), consec_req)
    tsla_trend_1m = calc_trend(tsla_1m["Close"].tail(display_bars), consec_req)
    if not tsla_5m.empty and not uvxy_5m.empty:
        uvxy_trend_5m = calc_trend(uvxy_5m["Close"].tail(12), consec_req)
        tsla_trend_5m = calc_trend(tsla_5m["Close"].tail(12), consec_req)
    if use_spy_filter and not spy_1m.empty:
        spy_trend_1m = calc_trend(spy_1m["Close"].tail(display_bars), consec_req)

    uvxy_dir       = uvxy_trend_1m.get("direction", 0)
    uvxy_confirmed = uvxy_trend_1m.get("confirmed", False)
    uvxy_slope_ok  = abs(uvxy_trend_1m.get("slope", 0)) >= min_uvxy_slope
    uvxy_5m_dir    = uvxy_trend_5m.get("direction", 0) if uvxy_trend_5m else 0
    multi_tf_agree = (uvxy_dir != 0) and (uvxy_5m_dir == uvxy_dir)

    tsla_pct       = tsla_trend_1m.get("pct_move", 0)
    uvxy_pct       = uvxy_trend_1m.get("pct_move", 0)
    tsla_responded = (
        abs(tsla_pct) >= min_tsla_response and
        int(np.sign(tsla_pct)) == -uvxy_dir
    )

    # ── Resistance ratio ──────────────────────────────────────────────────────
    # Only meaningful when UVXY is rising and TSLA should be falling
    if uvxy_dir == +1 and uvxy_pct > 0.1:
        res_ratio = resistance_ratio(uvxy_pct, tsla_pct)

    # ── Simple RSI-14 for TSLA ────────────────────────────────────────────────
    closes = tsla_1m["Close"].values.astype(float)
    if len(closes) >= 15:
        deltas = np.diff(closes[-15:])
        gains  = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        avg_g  = float(np.mean(gains))  if len(gains)  > 0 else 0.0
        avg_l  = float(np.mean(losses)) if len(losses) > 0 else 0.0
        if avg_l == 0:
            tsla_rsi1 = 100.0
        else:
            rs        = avg_g / avg_l
            tsla_rsi1 = round(100 - 100 / (1 + rs), 2)

    # ── Volume spike ─────────────────────────────────────────────────────────
    if "Volume" in tsla_1m.columns:
        tsla_vol_ratio = vol_ratio(tsla_1m, window=10)

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 1: WARNING — TSLA resistance while UVXY is rising
    # Condition: UVXY confirmed up + TSLA barely moves (resistance ratio low)
    # ──────────────────────────────────────────────────────────────────────────
    if (uvxy_confirmed and uvxy_slope_ok and uvxy_dir == +1
            and res_ratio < resistance_thresh and not tsla_responded):
        layer1_active = True
        layer1_reason = (
            f"UVXY 升 {uvxy_pct:+.2f}%（斜率{uvxy_trend_1m['slope']:+.3f}%/根 R²={uvxy_trend_1m['r2']:.2f}）"
            f"　TSLA 僅 {tsla_pct:+.2f}%　抵抗力比值={res_ratio:.2f}（< {resistance_thresh}）"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 2: ENTRY SIGNAL
    # Case A (BUY): UVXY confirmed DOWN + TSLA not yet risen
    # Case B (BUY enhanced): UVXY was UP, now turning DOWN (pivot) + TSLA still strong
    # Case C (SELL): UVXY confirmed UP + TSLA not yet fallen
    # ──────────────────────────────────────────────────────────────────────────
    if uvxy_confirmed and uvxy_slope_ok and uvxy_dir != 0:
        base_strength = (
            abs(uvxy_trend_1m["slope"])
            + uvxy_trend_1m["r2"]
            + uvxy_trend_1m["consecutive"] * 0.3
        ) * (1.5 if multi_tf_agree else 1.0)

        if uvxy_dir == -1 and not tsla_responded:
            # Standard BUY: UVXY falling, TSLA lagging
            layer2_signal   = "BUY_TSLA"
            layer2_strength = base_strength
            tf_tag = "【1m+5m✓】" if multi_tf_agree else "【1m】"

            # Boost if Layer 1 was recently active (resistance confirmed before drop)
            if layer1_active or res_ratio < resistance_thresh:
                layer2_strength *= 1.8
                tf_tag += "【抵抗力確認⭐】"

            layer2_reason = (
                f"{tf_tag} UVXY跌 {uvxy_pct:+.2f}%  斜率{uvxy_trend_1m['slope']:+.3f}%/根  "
                f"R²={uvxy_trend_1m['r2']:.2f}  連{uvxy_trend_1m['consecutive']}根 | "
                f"TSLA 僅 {tsla_pct:+.2f}% 尚未反應"
            )

        elif uvxy_dir == +1 and not tsla_responded:
            # SELL: UVXY rising, TSLA not yet falling
            layer2_signal   = "SELL_TSLA"
            layer2_strength = base_strength
            tf_tag = "【1m+5m✓】" if multi_tf_agree else "【1m】"
            layer2_reason = (
                f"{tf_tag} UVXY升 {uvxy_pct:+.2f}%  斜率{uvxy_trend_1m['slope']:+.3f}%/根  "
                f"R²={uvxy_trend_1m['r2']:.2f}  連{uvxy_trend_1m['consecutive']}根 | "
                f"TSLA 僅 {tsla_pct:+.2f}% 尚未跟跌"
            )

    layer2_stars = signal_stars(layer2_strength)

    # ──────────────────────────────────────────────────────────────────────────
    # SPY FILTER — applied to BUY_TSLA only
    # ──────────────────────────────────────────────────────────────────────────
    if use_spy_filter and layer2_signal == "BUY_TSLA" and spy_trend_1m:
        spy_dir       = spy_trend_1m.get("direction", 0)
        spy_confirmed = spy_trend_1m.get("confirmed", False)
        spy_pct       = spy_trend_1m.get("pct_move", 0)

        if spy_confirmed and spy_dir == -1:
            if "嚴格" in spy_filter_strength:
                # Hard filter: cancel signal
                layer2_signal = None
                spy_filtered  = True
                layer2_reason = (
                    f"⛔ SPY過濾（嚴格）：SPY跌 {spy_pct:+.2f}% 確認下跌趨勢，"
                    f"TSLA跌可能是大盤拖累，非純UVXY滯後訊號"
                )
            else:
                # Soft filter: reduce strength and add warning
                layer2_strength *= 0.5
                layer2_stars     = signal_stars(layer2_strength)
                spy_filtered     = True
                layer2_reason   += (
                    f"  ⚠️ SPY過濾（寬鬆）：SPY跌 {spy_pct:+.2f}%，訊號強度已降半"
                )

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 3: EXIT SIGNAL
    # Condition: RSI overbought OR volume spike red candle OR UVXY reversal up
    # ──────────────────────────────────────────────────────────────────────────
    exit_reasons = []

    if tsla_rsi1 is not None and tsla_rsi1 >= rsi_overbought:
        exit_reasons.append(f"RSI={tsla_rsi1:.1f} 超買（>{rsi_overbought}）")

    if tsla_vol_ratio >= vol_spike_mult:
        last_close = float(tsla_1m["Close"].iloc[-1])
        last_open  = float(tsla_1m["Open"].iloc[-1])
        if last_close < last_open:   # red candle
            exit_reasons.append(f"成交量爆增{tsla_vol_ratio:.1f}×且收黑K")

    # UVXY reversal: was falling, now turning up strongly
    if (st.session_state.active_signal == "BUY_TSLA"
            and uvxy_dir == +1 and uvxy_confirmed):
        exit_reasons.append(f"UVXY反轉上升（斜率{uvxy_trend_1m['slope']:+.3f}%/根）")

    if exit_reasons:
        layer3_exit   = True
        layer3_reason = "  ·  ".join(exit_reasons)

    # ──────────────────────────────────────────────────────────────────────────
    # ACTIVE SIGNAL TRACKING
    # ──────────────────────────────────────────────────────────────────────────
    if layer2_signal in ("BUY_TSLA", "SELL_TSLA"):
        if st.session_state.active_signal != layer2_signal:
            st.session_state.active_signal      = layer2_signal
            st.session_state.active_signal_time = now
            st.session_state.active_signal_entry = float(tsla_1m["Close"].iloc[-1])
        else:
            elapsed = (now - st.session_state.active_signal_time).total_seconds() / 60
            t_now   = float(tsla_1m["Close"].iloc[-1])
            pnl     = ((t_now - st.session_state.active_signal_entry)
                       / st.session_state.active_signal_entry * 100
                       * (1 if layer2_signal == "BUY_TSLA" else -1))

            if layer3_exit:
                st.session_state.signal_history.append(
                    f"{now.strftime('%H:%M')} 🟣 出場 "
                    f"{'買' if layer2_signal=='BUY_TSLA' else '賣'}TSLA "
                    f"持{elapsed:.0f}min  P&L≈{pnl:+.2f}%  {layer3_reason[:30]}"
                )
                st.session_state.active_signal = None
            elif elapsed > signal_timeout:
                st.session_state.signal_history.append(
                    f"{now.strftime('%H:%M')} ❌ 失效 "
                    f"{'買' if layer2_signal=='BUY_TSLA' else '賣'}TSLA "
                    f"超{signal_timeout}min未反應"
                )
                st.session_state.active_signal = None
            elif abs(pnl) > 0 and (
                (layer2_signal == "BUY_TSLA"  and pnl > 0) or
                (layer2_signal == "SELL_TSLA" and pnl > 0)
            ):
                # Mark as fulfilled when TSLA has actually moved
                if tsla_responded:
                    st.session_state.signal_history.append(
                        f"{now.strftime('%H:%M')} ✅ 兌現 "
                        f"{'買' if layer2_signal=='BUY_TSLA' else '賣'}TSLA "
                        f"持{elapsed:.0f}min  P&L≈{pnl:+.2f}%"
                    )
    else:
        if layer3_exit and st.session_state.active_signal:
            t_now  = float(tsla_1m["Close"].iloc[-1])
            entry  = st.session_state.active_signal_entry or t_now
            elapsed = (now - (st.session_state.active_signal_time or now)).total_seconds() / 60
            pnl    = (t_now - entry) / entry * 100
            st.session_state.signal_history.append(
                f"{now.strftime('%H:%M')} 🟣 出場 持{elapsed:.0f}min P&L≈{pnl:+.2f}%"
            )
            st.session_state.active_signal = None

    # ──────────────────────────────────────────────────────────────────────────
    # HIGH-SENSITIVITY MODE — single candle divergence detection
    # Logic: compare last 2 closes of TSLA and UVXY
    #   UVXY close[n] > close[n-1]  (UVXY just moved up this minute)
    #   TSLA close[n] <= close[n-1] + hs_tsla_max%  (TSLA did NOT fall)
    #   → Alert: SELL TSLA (TSLA should have fallen but didn't)
    #
    #   UVXY close[n] < close[n-1]  (UVXY just moved down this minute)
    #   TSLA close[n] >= close[n-1] - hs_tsla_max%  (TSLA did NOT rise)
    #   → Alert: BUY TSLA (TSLA should have risen but didn't)
    # ──────────────────────────────────────────────────────────────────────────
    hs_signal      = None    # "HS_BUY" | "HS_SELL"
    hs_reason      = ""
    hs_uvxy_chg    = 0.0
    hs_tsla_chg    = 0.0
    hs_mult        = 0.0     # dynamic position size multiplier
    hs_mult_reason = ""
    hs_quality     = ""

    if use_hs_mode and len(tsla_1m) >= 2 and len(uvxy_1m) >= 2:
        common_last2 = tsla_1m.index.intersection(uvxy_1m.index)
        if len(common_last2) >= 2:
            t_closes = tsla_1m.loc[common_last2, "Close"].iloc[-2:]
            u_closes = uvxy_1m.loc[common_last2, "Close"].iloc[-2:]

            hs_tsla_chg = float((t_closes.iloc[-1] - t_closes.iloc[0]) / t_closes.iloc[0] * 100)
            hs_uvxy_chg = float((u_closes.iloc[-1] - u_closes.iloc[0]) / u_closes.iloc[0] * 100)

            uvxy_up_now   = hs_uvxy_chg >= hs_uvxy_min
            uvxy_down_now = hs_uvxy_chg <= -hs_uvxy_min
            tsla_not_fell = hs_tsla_chg >= -hs_tsla_max
            tsla_not_rose = hs_tsla_chg <= hs_tsla_max

            if uvxy_up_now and tsla_not_fell:
                hs_signal = "HS_SELL"
                hs_reason = (
                    f"UVXY本分鐘升 {hs_uvxy_chg:+.3f}%，"
                    f"TSLA 僅 {hs_tsla_chg:+.3f}%（未跟跌）"
                )
            elif uvxy_down_now and tsla_not_rose:
                hs_signal = "HS_BUY"
                hs_reason = (
                    f"UVXY本分鐘跌 {hs_uvxy_chg:+.3f}%，"
                    f"TSLA 僅 {hs_tsla_chg:+.3f}%（未跟升）"
                )

            # ── Dynamic sizing (runs whenever HS mode is on) ──────────────
            import math as _math
            # Get 20-bar rolling correlation
            _common = tsla_1m.index.intersection(uvxy_1m.index)
            if len(_common) >= 20:
                _tc = tsla_1m.loc[_common, "Close"].iloc[-20:].values.astype(float)
                _uc = uvxy_1m.loc[_common, "Close"].iloc[-20:].values.astype(float)
                _corr20 = float(np.corrcoef(_tc, _uc)[0, 1])
            else:
                _corr20 = float("nan")

            # ET hour
            try:
                import pytz as _pytz
                _et = now.astimezone(_pytz.timezone("America/New_York"))
                _et_hour = _et.hour
            except Exception:
                _et_hour = now.hour  # fallback

            # Fetch UVXY RSI and 5-bar momentum for optional filters
            _uvxy_rsi  = float(uvxy_1m["RSI"].iloc[-1]) if "RSI" in uvxy_1m.columns and not np.isnan(uvxy_1m["RSI"].iloc[-1]) else float("nan")
            _tsla_rsi  = float(tsla_rsi1) if tsla_rsi1 is not None else float("nan")
            _uvxy_mom5 = float(
                (uvxy_1m["Close"].iloc[-1] - uvxy_1m["Close"].iloc[-6]) / uvxy_1m["Close"].iloc[-6] * 100
            ) if len(uvxy_1m) >= 6 else float("nan")

            hs_mult, hs_mult_reason, hs_quality = hs_dynamic_size(
                corr20       = _corr20,
                uvxy_chg_abs = abs(hs_uvxy_chg),
                recent_pnls  = list(st.session_state.hs_recent_pnls),
                et_hour      = _et_hour,
                uvxy_rsi     = _uvxy_rsi,
                uvxy_mom5    = _uvxy_mom5,
                tsla_rsi     = _tsla_rsi,
                use_filter_a = use_filter_a,
                use_filter_b = use_filter_b,
                use_filter_c = use_filter_c,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # TELEGRAM ALERTS
    # ──────────────────────────────────────────────────────────────────────────
    def cooldown_ok(last_time, minutes):
        return (last_time is None or
                (now - last_time).total_seconds() > minutes * 60)

    # Layer 1 warning alert
    if layer1_active and cooldown_ok(st.session_state.last_warn_time, max(1, cooldown_min // 2)):
        t_p = float(tsla_1m["Close"].iloc[-1])
        u_p = float(uvxy_1m["Close"].iloc[-1])
        msg = (
            f"🟡 *TSLA 抵抗力預警（Layer 1）*\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"時間：`{now.strftime('%Y-%m-%d %H:%M')}`\n"
            f"UVXY升 `{uvxy_pct:+.2f}%` 但TSLA僅 `{tsla_pct:+.2f}%`\n"
            f"抵抗力比值：`{res_ratio:.2f}`（門檻<{resistance_thresh}）\n"
            f"TSLA：`${t_p:.2f}`  UVXY：`${u_p:.2f}`\n"
            f"⚡ 若UVXY轉跌，TSLA可能急升"
        )
        send_telegram(msg)
        st.session_state.last_warn_time = now
        st.session_state.signal_history.append(
            f"{now.strftime('%H:%M')} 🟡 預警 UVXY升{uvxy_pct:+.1f}% TSLA抵抗{res_ratio:.2f}"
        )

    # Layer 2 entry alert
    if (layer2_signal in ("BUY_TSLA", "SELL_TSLA")
            and cooldown_ok(st.session_state.last_alert_time, cooldown_min)):
        action = "🟢 買入 TSLA" if layer2_signal == "BUY_TSLA" else "🔴 賣出 TSLA"
        t_p    = float(tsla_1m["Close"].iloc[-1])
        u_p    = float(uvxy_1m["Close"].iloc[-1])
        spy_line = ""
        if use_spy_filter and spy_trend_1m:
            spy_pct_val = spy_trend_1m.get("pct_move", 0)
            spy_line = f"SPY過濾：{'⛔已過濾降級' if spy_filtered else '✅未觸發'}（SPY{spy_pct_val:+.2f}%）\n"
        tf_line = "✅ 1m+5m 雙框架" if multi_tf_agree else "⚠️ 僅1m框架"
        msg = (
            f"*{action}*  {layer2_stars}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"時間：`{now.strftime('%Y-%m-%d %H:%M')}`\n"
            f"框架：{tf_line}\n"
            f"{spy_line}"
            f"UVXY斜率：`{uvxy_trend_1m.get('slope',0):+.3f}%/根`  "
            f"R²=`{uvxy_trend_1m.get('r2',0):.2f}`  "
            f"連續`{uvxy_trend_1m.get('consecutive',0)}`根\n"
            f"TSLA未反應：`{tsla_pct:+.2f}%`\n"
            f"訊號強度：`{layer2_strength:.2f}` {layer2_stars}\n"
            f"入場參考：TSLA `${t_p:.2f}`  UVXY `${u_p:.2f}`"
        )
        send_telegram(msg)
        st.session_state.last_alert_time = now
        spy_tag = " [SPY↓降級]" if spy_filtered else ""
        st.session_state.signal_history.append(
            f"{now.strftime('%H:%M')} "
            f"{'🟢 買TSLA' if layer2_signal=='BUY_TSLA' else '🔴 賣TSLA'} "
            f"{layer2_stars} 強度{layer2_strength:.1f}{spy_tag}"
        )

    # Layer 3 exit alert
    if layer3_exit and cooldown_ok(st.session_state.last_exit_time, 2):
        t_p   = float(tsla_1m["Close"].iloc[-1])
        entry = st.session_state.active_signal_entry
        pnl_str = ""
        if entry:
            pnl = (t_p - entry) / entry * 100
            pnl_str = f"P&L估算：`{pnl:+.2f}%`\n"
        msg = (
            f"🟣 *出場訊號（Layer 3）*\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"時間：`{now.strftime('%Y-%m-%d %H:%M')}`\n"
            f"原因：{layer3_reason}\n"
            f"{pnl_str}"
            f"TSLA現價：`${t_p:.2f}`\n"
            f"RSI：`{tsla_rsi1:.1f}`  成交量倍數：`{tsla_vol_ratio:.1f}×`"
        )
        send_telegram(msg)
        st.session_state.last_exit_time = now
        st.session_state.signal_history.append(
            f"{now.strftime('%H:%M')} 🟣 出場 {layer3_reason[:40]}"
        )

    # High-sensitivity alert (with dynamic sizing)
    if (hs_signal in ("HS_BUY", "HS_SELL")
            and cooldown_ok(st.session_state.last_hs_alert_time, hs_cooldown)):
        t_p   = float(tsla_1m["Close"].iloc[-1])
        u_p   = float(uvxy_1m["Close"].iloc[-1])
        action_hs = "⚡🟢 買入 TSLA（高敏感）" if hs_signal == "HS_BUY" else "⚡🔴 賣出 TSLA（高敏感）"

        if hs_mult == 0:
            # Skipped — still log but send lighter alert
            msg = (
                f"⏭ *高敏感訊號跳過*\n"
                f"時間：`{now.strftime('%Y-%m-%d %H:%M')}`\n"
                f"原因：{hs_mult_reason}\n"
                f"訊號：{'買TSLA' if hs_signal=='HS_BUY' else '賣TSLA'}  "
                f"UVXY{hs_uvxy_chg:+.3f}% TSLA{hs_tsla_chg:+.3f}%"
            )
        else:
            mult_emoji = "🔥" if hs_mult >= 2.5 else ("⭐" if hs_mult >= 2 else ("▲" if hs_mult >= 1.5 else "▪"))
            msg = (
                f"*{action_hs}*\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"時間：`{now.strftime('%Y-%m-%d %H:%M')}`\n"
                f"注碼建議：`{hs_mult}x` {mult_emoji}  {hs_quality}\n"
                f"注碼依據：{hs_mult_reason}\n"
                f"原因：{hs_reason}\n"
                f"UVXY變動：`{hs_uvxy_chg:+.3f}%`  TSLA變動：`{hs_tsla_chg:+.3f}%`\n"
                f"TSLA現價：`${t_p:.2f}`  UVXY現價：`${u_p:.2f}`"
            )
        send_telegram(msg)
        st.session_state.last_hs_alert_time = now
        log = (f"{now.strftime('%H:%M')} "
               f"{'⚡🟢 HS買TSLA' if hs_signal=='HS_BUY' else '⚡🔴 HS賣TSLA'} "
               f"{hs_quality} {hs_mult}x  UVXY{hs_uvxy_chg:+.2f}%")
        st.session_state.signal_history.append(log)

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL DISPLAY — THREE LAYERS
# ═══════════════════════════════════════════════════════════════════════════════

# Layer 3 exit (top priority display)
if layer3_exit:
    st.markdown(f"""
    <div class="sig-layer3">
      <div class="sig-title" style="color:#b57bee">🟣 出場訊號（Layer 3）</div>
      <div class="sig-detail">{layer3_reason}<br>
      RSI={tsla_rsi1:.1f if tsla_rsi1 else '—'}  成交量={tsla_vol_ratio:.1f}×</div>
    </div>
    """, unsafe_allow_html=True)

# Layer 2 entry signal
if layer2_signal == "BUY_TSLA":
    spy_note = f"<br><small style='color:#f6c90e'>⚠️ SPY過濾已降級訊號強度</small>" if spy_filtered else ""
    st.markdown(f"""
    <div class="sig-layer2-buy">
      <div class="sig-title" style="color:#00d97e">🟢 買入 TSLA &nbsp; {layer2_stars}</div>
      <div class="sig-detail">{layer2_reason}{spy_note}</div>
    </div>
    """, unsafe_allow_html=True)
elif layer2_signal == "SELL_TSLA":
    st.markdown(f"""
    <div class="sig-layer2-sell">
      <div class="sig-title" style="color:#e84045">🔴 賣出 TSLA &nbsp; {layer2_stars}</div>
      <div class="sig-detail">{layer2_reason}</div>
    </div>
    """, unsafe_allow_html=True)
elif spy_filtered:
    st.markdown(f"""
    <div class="sig-layer2-buy" style="opacity:0.5">
      <div class="sig-title" style="color:#8b8fa8">⛔ 買入訊號已被 SPY 過濾</div>
      <div class="sig-detail">{layer2_reason}</div>
    </div>
    """, unsafe_allow_html=True)

# Layer 1 warning
if layer1_active:
    st.markdown(f"""
    <div class="sig-layer1">
      <div class="sig-title" style="color:#f6c90e">🟡 預警：TSLA 抵抗力強（Layer 1）</div>
      <div class="sig-detail">{layer1_reason}<br>
      <b>UVXY若轉跌，TSLA極可能急升 → 等待Layer 2確認入場</b></div>
    </div>
    """, unsafe_allow_html=True)

# Normal state
if not layer1_active and not layer2_signal and not layer3_exit:
    uvxy_s = uvxy_trend_1m.get("slope", 0)
    st.markdown(f"""
    <div class="sig-normal">
      <div class="sig-title" style="color:#8b8fa8">⚪ 無訊號</div>
      <div class="sig-detail">UVXY斜率={uvxy_s:+.3f}%/根　確認={uvxy_trend_1m.get('confirmed','—')}　
      趨勢未達觸發條件</div>
    </div>
    """, unsafe_allow_html=True)

# High-sensitivity signal box (shown below the main signal boxes)
if use_hs_mode:
    # Determine colors based on signal and multiplier
    _hs_mult_color = (
        "#b57bee" if hs_mult >= 2.5 else
        "#00d97e" if hs_mult >= 2.0 else
        "#5c7cfa" if hs_mult >= 1.5 else
        "#f6c90e" if hs_mult >= 0.5 else
        "#e84045"
    )
    _hs_mult_label = f"{hs_mult}x  {hs_quality}" if hs_signal else ""

    if hs_signal == "HS_BUY" and hs_mult > 0:
        st.markdown(f"""
        <div style='border-radius:10px;padding:12px 20px;margin:6px 0;
                    background:#0a1a14;border:2px dashed #00d97e;text-align:center'>
          <div style='font-size:1.3rem;font-weight:800;color:#00d97e'>
              ⚡ 高敏感 — 🟢 買入 TSLA
              <span style='font-size:1rem;margin-left:12px;color:{_hs_mult_color}'>
                  注碼 {_hs_mult_label}</span></div>
          <div style='font-size:0.82rem;color:#c9cdd8;margin-top:4px'>{hs_reason}</div>
          <div style='font-size:0.75rem;color:#8b8fa8;margin-top:3px'>
              UVXY {hs_uvxy_chg:+.3f}%　TSLA {hs_tsla_chg:+.3f}%</div>
          <div style='font-size:0.72rem;color:#5c7cfa;margin-top:3px'>{hs_mult_reason}</div>
        </div>
        """, unsafe_allow_html=True)
    elif hs_signal == "HS_SELL" and hs_mult > 0:
        st.markdown(f"""
        <div style='border-radius:10px;padding:12px 20px;margin:6px 0;
                    background:#1a0a0a;border:2px dashed #e84045;text-align:center'>
          <div style='font-size:1.3rem;font-weight:800;color:#e84045'>
              ⚡ 高敏感 — 🔴 賣出 TSLA
              <span style='font-size:1rem;margin-left:12px;color:{_hs_mult_color}'>
                  注碼 {_hs_mult_label}</span></div>
          <div style='font-size:0.82rem;color:#c9cdd8;margin-top:4px'>{hs_reason}</div>
          <div style='font-size:0.75rem;color:#8b8fa8;margin-top:3px'>
              UVXY {hs_uvxy_chg:+.3f}%　TSLA {hs_tsla_chg:+.3f}%</div>
          <div style='font-size:0.72rem;color:#5c7cfa;margin-top:3px'>{hs_mult_reason}</div>
        </div>
        """, unsafe_allow_html=True)
    elif hs_signal and hs_mult == 0:
        st.markdown(f"""
        <div style='border-radius:10px;padding:10px 20px;margin:6px 0;
                    background:#1a1008;border:2px dashed #e84045;text-align:center;opacity:0.7'>
          <div style='font-size:1rem;font-weight:700;color:#e84045'>
              ⏭ 高敏感訊號已跳過（注碼=0）</div>
          <div style='font-size:0.78rem;color:#8b8fa8;margin-top:4px'>{hs_mult_reason}</div>
        </div>
        """, unsafe_allow_html=True)
    elif hs_signal is None:
        _active_f2 = sum([use_filter_a, use_filter_b, use_filter_c])
        _filter_tags = "  ".join([
            f"<span style='color:#00d97e'>A✓</span>" if use_filter_a else "<span style='color:#2d3139'>A○</span>",
            f"<span style='color:#00d97e'>B✓</span>" if use_filter_b else "<span style='color:#2d3139'>B○</span>",
            f"<span style='color:#00d97e'>C✓</span>" if use_filter_c else "<span style='color:#2d3139'>C○</span>",
        ])
        st.markdown(f"""
        <div style='border-radius:10px;padding:10px 20px;margin:6px 0;
                    background:#111316;border:1px dashed #2d3139;text-align:center'>
          <div style='font-size:0.9rem;color:#8b8fa8'>
              ⚡ 高敏感監測中…　UVXY={hs_uvxy_chg:+.3f}%　TSLA={hs_tsla_chg:+.3f}%</div>
          <div style='font-size:0.78rem;margin-top:4px'>
              過濾器 {_filter_tags}
              　<span style='color:#8b8fa8'>注碼評分：</span>
              <span style='color:{_hs_mult_color}'>{hs_mult_reason if hs_mult_reason else "待訊號"}</span></div>
        </div>
        """, unsafe_allow_html=True)

# ── Active signal progress tracker ───────────────────────────────────────────
if st.session_state.active_signal and st.session_state.active_signal_time:
    elapsed_min = (now - st.session_state.active_signal_time).total_seconds() / 60
    remaining   = max(signal_timeout - elapsed_min, 0)
    bar_pct     = min(elapsed_min / signal_timeout * 100, 100)
    bar_color   = "#00d97e" if st.session_state.active_signal == "BUY_TSLA" else "#e84045"
    action_word = "上升" if st.session_state.active_signal == "BUY_TSLA" else "下跌"
    entry_str   = f"${st.session_state.active_signal_entry:.2f}" if st.session_state.active_signal_entry else "—"
    t_now_val   = float(tsla_1m["Close"].iloc[-1]) if data_ok else 0
    pnl_now     = ((t_now_val - (st.session_state.active_signal_entry or t_now_val))
                   / (st.session_state.active_signal_entry or 1) * 100)
    pnl_color   = "#00d97e" if pnl_now >= 0 else "#e84045"
    st.markdown(f"""
    <div class="track-card">
      <b style="color:{bar_color}">📡 訊號追蹤中</b>
      &nbsp;｜&nbsp; 入場價 <b>{entry_str}</b>
      &nbsp;｜&nbsp; 持倉 <b>{elapsed_min:.0f}</b> 分鐘
      &nbsp;｜&nbsp; 失效倒數 <b>{remaining:.0f}</b> 分鐘
      &nbsp;｜&nbsp; 目標：TSLA {action_word} ≥ {min_tsla_response}%
      &nbsp;｜&nbsp; 即時P&L：<b style="color:{pnl_color}">{pnl_now:+.2f}%</b>
      <div style="background:#2d3139;border-radius:4px;height:6px;margin-top:8px">
        <div style="background:{bar_color};width:{bar_pct:.0f}%;height:6px;border-radius:4px"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# METRIC CARDS
# ═══════════════════════════════════════════════════════════════════════════════
tsla_price = float(tsla_1m["Close"].iloc[-1]) if data_ok else None
uvxy_price = float(uvxy_1m["Close"].iloc[-1]) if data_ok else None
spy_price  = float(spy_1m["Close"].iloc[-1])  if use_spy_filter and not spy_1m.empty else None

def fmt_pct(v):
    if v is None: return "—"
    return f"{'▲' if v >= 0 else '▼'} {abs(v):.2f}%"
def pct_color(v):
    return "#8b8fa8" if v is None else ("#00d97e" if v >= 0 else "#e84045")
def slope_color(s):
    return "#8b8fa8" if abs(s) < 0.05 else ("#00d97e" if s > 0 else "#e84045")
corr_color = (
    "#00d97e" if corr_value is not None and corr_value < -0.7 else
    "#f6c90e" if corr_value is not None and corr_value < -0.5 else "#e84045"
)

def metric_card(col, label, value, color, sub=None, sub_color="#8b8fa8"):
    sub_html = f'<div class="metric-sub" style="color:{sub_color}">{sub}</div>' if sub else ""
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color}">{value}</div>
          {sub_html}
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-title">即時指標</div>', unsafe_allow_html=True)

num_cols = 11 if use_spy_filter else 10
cols = st.columns(num_cols)

metric_card(cols[0], "TSLA 價格",
            f"${tsla_price:.2f}" if tsla_price else "—", "#5c7cfa",
            sub=fmt_pct(tsla_trend_1m.get("pct_move")),
            sub_color=pct_color(tsla_trend_1m.get("pct_move")))
metric_card(cols[1], "TSLA 1m斜率",
            f"{tsla_trend_1m.get('slope',0):+.3f}%",
            slope_color(tsla_trend_1m.get("slope", 0)),
            sub=f"R²={tsla_trend_1m.get('r2',0):.2f} 連{tsla_trend_1m.get('consecutive',0)}根")
metric_card(cols[2], "TSLA RSI",
            f"{tsla_rsi1:.1f}" if tsla_rsi1 else "—",
            "#e84045" if (tsla_rsi1 or 0) >= rsi_overbought else
            "#f6c90e" if (tsla_rsi1 or 0) >= rsi_overbought - 10 else "#00d97e",
            sub="超買⚠️" if (tsla_rsi1 or 0) >= rsi_overbought else "正常")
metric_card(cols[3], "UVXY 價格",
            f"${uvxy_price:.2f}" if uvxy_price else "—", "#f6c90e",
            sub=fmt_pct(uvxy_trend_1m.get("pct_move")),
            sub_color=pct_color(uvxy_trend_1m.get("pct_move")))
metric_card(cols[4], "UVXY 1m斜率",
            f"{uvxy_trend_1m.get('slope',0):+.3f}%",
            slope_color(uvxy_trend_1m.get("slope", 0)),
            sub=f"R²={uvxy_trend_1m.get('r2',0):.2f} 連{uvxy_trend_1m.get('consecutive',0)}根")
metric_card(cols[5], "UVXY 5m斜率",
            f"{uvxy_trend_5m.get('slope',0):+.3f}%" if uvxy_trend_5m else "—",
            slope_color(uvxy_trend_5m.get("slope", 0) if uvxy_trend_5m else 0),
            sub=f"{'✅雙框架' if multi_tf_agree else '⚪單框架'}")
metric_card(cols[6], "抵抗力比值",
            f"{res_ratio:.2f}" if uvxy_dir == +1 else "—",
            "#00d97e" if res_ratio < resistance_thresh else "#8b8fa8",
            sub=f"門檻<{resistance_thresh}  {'⚡強抵抗' if res_ratio < resistance_thresh else '正常'}")
metric_card(cols[7], "成交量倍數",
            f"{tsla_vol_ratio:.1f}×",
            "#e84045" if tsla_vol_ratio >= vol_spike_mult else "#8b8fa8",
            sub="爆量⚠️" if tsla_vol_ratio >= vol_spike_mult else "正常")
metric_card(cols[8], "皮爾森係數",
            f"{corr_value:.3f}" if corr_value is not None else "—", corr_color)
metric_card(cols[9], "最後更新", now.strftime("%H:%M:%S"), "#8b8fa8")

if use_spy_filter and spy_price:
    metric_card(cols[10], "SPY 價格",
                f"${spy_price:.2f}", "#a78bfa",
                sub=fmt_pct(spy_trend_1m.get("pct_move") if spy_trend_1m else None),
                sub_color=pct_color(spy_trend_1m.get("pct_move") if spy_trend_1m else None))

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CANDLE CHARTS — 1m with regression overlay
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-title">1 分鐘 K 線圖（最新 {display_bars} 根）</div>',
            unsafe_allow_html=True)

if data_ok:
    tsla_show = tsla_1m.tail(display_bars)
    uvxy_show = uvxy_1m.tail(display_bars)

    def make_candle_fig(df, title, slope, r2, cu="#00d97e", cd="#e84045"):
        closes = df["Close"].values.astype(float)
        x_vals = np.arange(len(closes))
        s, b, *_ = linregress(x_vals, closes) if len(closes) >= 3 else (0, closes[0], None, None, None)
        reg_y = [s * xi + b for xi in x_vals]
        reg_color = "#00d97e" if slope >= 0 else "#e84045"
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color=cu, decreasing_line_color=cd,
            increasing_fillcolor=cu, decreasing_fillcolor=cd,
            name=title, showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=reg_y, mode="lines",
            line=dict(color=reg_color, width=2, dash="dot"),
            name=f"趨勢線 {slope:+.3f}%/根 R²={r2:.2f}",
        ))
        fig.update_layout(
            title=dict(text=f"{title}   斜率={slope:+.3f}%/根   R²={r2:.2f}",
                       font=dict(size=13, color="#c9cdd8")),
            paper_bgcolor="#1c1f26", plot_bgcolor="#1c1f26",
            xaxis=dict(gridcolor="#2d3139", rangeslider=dict(visible=False), color="#8b8fa8"),
            yaxis=dict(gridcolor="#2d3139", color="#8b8fa8"),
            legend=dict(font=dict(color="#8b8fa8", size=10), bgcolor="#1c1f26"),
            margin=dict(l=10, r=10, t=45, b=10), height=320,
        )
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            make_candle_fig(tsla_show, f"TSLA  ${tsla_price:.2f}",
                            tsla_trend_1m.get("slope", 0), tsla_trend_1m.get("r2", 0)),
            use_container_width=True, config={"displayModeBar": False},
        )
    with col2:
        st.plotly_chart(
            make_candle_fig(uvxy_show, f"UVXY  ${uvxy_price:.2f}",
                            uvxy_trend_1m.get("slope", 0), uvxy_trend_1m.get("r2", 0),
                            cu="#f6c90e", cd="#e84045"),
            use_container_width=True, config={"displayModeBar": False},
        )

# ═══════════════════════════════════════════════════════════════════════════════
# 5m CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">5 分鐘趨勢確認（最新12根 ≈ 60分鐘）</div>',
            unsafe_allow_html=True)

if not tsla_5m.empty and not uvxy_5m.empty:
    def make_5m_fig(df, label, slope, color):
        closes    = df["Close"].tail(12)
        x_idx     = np.arange(len(closes))
        s, b, *_  = linregress(x_idx, closes.values.astype(float))
        reg_y     = [s * xi + b for xi in x_idx]
        reg_color = "#00d97e" if slope >= 0 else "#e84045"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=closes.index, y=closes.values,
            mode="lines+markers", line=dict(color=color, width=2),
            marker=dict(size=5), name=label,
        ))
        fig.add_trace(go.Scatter(
            x=closes.index, y=reg_y, mode="lines",
            line=dict(color=reg_color, width=2, dash="dot"),
            name=f"趨勢 {slope:+.3f}%/根",
        ))
        fig.update_layout(
            title=dict(text=f"{label}  5m  斜率={slope:+.3f}%/根",
                       font=dict(size=13, color="#c9cdd8")),
            paper_bgcolor="#1c1f26", plot_bgcolor="#1c1f26",
            xaxis=dict(gridcolor="#2d3139", color="#8b8fa8"),
            yaxis=dict(gridcolor="#2d3139", color="#8b8fa8"),
            legend=dict(font=dict(color="#8b8fa8", size=10), bgcolor="#1c1f26"),
            margin=dict(l=10, r=10, t=45, b=10), height=230,
        )
        return fig

    c5a, c5b = st.columns(2)
    with c5a:
        st.plotly_chart(
            make_5m_fig(tsla_5m, "TSLA 5m",
                        tsla_trend_5m.get("slope", 0) if tsla_trend_5m else 0, "#5c7cfa"),
            use_container_width=True, config={"displayModeBar": False},
        )
    with c5b:
        st.plotly_chart(
            make_5m_fig(uvxy_5m, "UVXY 5m",
                        uvxy_trend_5m.get("slope", 0) if uvxy_trend_5m else 0, "#f6c90e"),
            use_container_width=True, config={"displayModeBar": False},
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PEARSON CORRELATION HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">歷史皮爾森相關係數</div>', unsafe_allow_html=True)

hist = st.session_state.corr_history
if isinstance(hist, pd.DataFrame) and len(hist) >= 2:
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(
        x=hist["time"], y=hist["corr"],
        mode="lines+markers", line=dict(color="#5c7cfa", width=2),
        marker=dict(size=4), fill="tozeroy",
        fillcolor="rgba(92,124,250,0.08)", name="皮爾森係數",
    ))
    fig_c.add_hline(y=-0.5, line_dash="dash", line_color="#f6c90e",
                    annotation_text="警戒線 −0.5", annotation_font_color="#f6c90e")
    fig_c.add_hline(y=0, line_dash="dot", line_color="#8b8fa8")
    fig_c.update_layout(
        paper_bgcolor="#1c1f26", plot_bgcolor="#1c1f26",
        xaxis=dict(gridcolor="#2d3139", color="#8b8fa8"),
        yaxis=dict(gridcolor="#2d3139", color="#8b8fa8", range=[-1.1, 1.1]),
        legend=dict(font=dict(color="#8b8fa8"), bgcolor="#1c1f26"),
        margin=dict(l=10, r=10, t=20, b=10), height=190,
    )
    st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("📈 累積數據中，稍後顯示走勢圖…")

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO REFRESH
# ═══════════════════════════════════════════════════════════════════════════════
if auto_refresh:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
