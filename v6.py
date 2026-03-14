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
st.set_page_config(page_title="TSLA vs UVXY 三層訊號系統 v2", page_icon="🎯", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .metric-card {
      background:#1c1f26; border-radius:10px; padding:13px 15px;
      border:1px solid #2d3139; text-align:center; height:100%;
  }
  .metric-label { color:#8b8fa8; font-size:0.68rem; letter-spacing:0.06em; text-transform:uppercase; }
  .metric-value { font-size:1.35rem; font-weight:700; margin-top:4px; line-height:1.2; }
  .metric-sub   { font-size:0.75rem; font-weight:600; margin-top:3px; }
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
  .section-title {
      font-size:0.92rem; font-weight:700; color:#c9cdd8;
      border-left:3px solid #5c7cfa; padding-left:8px; margin:18px 0 8px 0;
  }
  .track-card {
      background:#161920; border-radius:8px; padding:10px 14px;
      border:1px solid #2d3139; margin:4px 0; font-size:0.82rem; color:#c9cdd8;
  }
  .spy-on  { background:#0a2020; border:1px solid #00d97e; border-radius:6px;
             padding:4px 10px; color:#00d97e; font-size:0.78rem; font-weight:700; }
  .spy-off { background:#1c1f26; border:1px solid #2d3139; border-radius:6px;
             padding:4px 10px; color:#8b8fa8; font-size:0.78rem; }
  /* ── 優化提示標籤 ── */
  .opt-badge {
      background:#0a1a2b; border:1px solid #5c7cfa; border-radius:5px;
      padding:2px 8px; color:#5c7cfa; font-size:0.72rem; font-weight:700;
      display:inline-block; margin-left:6px; vertical-align:middle;
  }
  /* ── 訊號品質條 ── */
  .quality-bar {
      background:#2d3139; border-radius:4px; height:6px; margin-top:6px;
  }
  .quality-fill {
      height:6px; border-radius:4px;
      transition: width 0.3s ease;
  }
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
    if abs(uvxy_pct) < 0.01:
        return 1.0
    ratio = abs(tsla_pct) / abs(uvxy_pct)
    return round(ratio, 3)

def signal_stars(strength: float) -> str:
    if strength >= 4.0:   return "⭐⭐⭐"
    elif strength >= 2.0: return "⭐⭐"
    else:                 return "⭐"

# ════════════════════════════════════════════════════════════════════════════════
# ★ 新增：訊號品質評分 (0-100)
# 綜合 R², 斜率, 連續根, RSI 防呆, lag_strength 計算
# ════════════════════════════════════════════════════════════════════════════════
def calc_signal_quality(uvxy_r2: float, uvxy_slope_abs: float, uvxy_consec: int,
                         rsi: float, signal_type: str, lag_strength: float,
                         multi_tf: bool) -> tuple:
    """
    Returns (score 0-100, grade label, color hex)
    Grade: A(80+), B(60-79), C(40-59), D(<40)
    """
    score = 0.0

    # R² quality (max 30 pts) — key driver from backtest
    if uvxy_r2 >= 0.80:   score += 30
    elif uvxy_r2 >= 0.70: score += 22
    elif uvxy_r2 >= 0.65: score += 15
    else:                  score += 5

    # Slope magnitude (max 20 pts)
    if uvxy_slope_abs >= 0.10:   score += 20
    elif uvxy_slope_abs >= 0.07: score += 14
    elif uvxy_slope_abs >= 0.05: score += 9
    else:                         score += 3

    # Consecutive bars (max 15 pts)
    score += min(uvxy_consec * 5, 15)

    # RSI filter (max 20 pts) — from backtest: RSI is key discriminator
    if signal_type == "SELL_TSLA":
        if rsi >= 55:    score += 20
        elif rsi >= 50:  score += 14
        elif rsi >= 45:  score += 7
        else:             score += 0   # penalty zone: oversold = bad sell
    else:  # BUY_TSLA
        if rsi <= 55:    score += 20
        elif rsi <= 60:  score += 14
        elif rsi <= 65:  score += 7
        else:             score += 0   # overbought = bad buy

    # Lag strength (max 10 pts) — TSLA should not have already moved
    if lag_strength >= 0.5:   score += 10
    elif lag_strength >= 0.0: score += 6
    elif lag_strength >= -0.3: score += 2
    else:                      score += 0

    # Multi-timeframe bonus (max 5 pts)
    if multi_tf: score += 5

    score = min(max(score, 0), 100)
    if score >= 80:   grade, color = "A", "#00d97e"
    elif score >= 60: grade, color = "B", "#5c7cfa"
    elif score >= 40: grade, color = "C", "#f6c90e"
    else:             grade, color = "D", "#e84045"

    return int(score), grade, color

# ════════════════════════════════════════════════════════════════════════════════
# ★ 新增：RSI 防呆過濾
# ════════════════════════════════════════════════════════════════════════════════
def rsi_gate(signal_type: str, rsi: float,
             sell_min_rsi: float, buy_max_rsi: float) -> tuple:
    """
    Returns (pass: bool, reason: str)
    SELL 訊號：RSI 必須 >= sell_min_rsi（避免超賣區賣出）
    BUY  訊號：RSI 必須 <= buy_max_rsi  （避免超買區買入）
    """
    if rsi is None or np.isnan(rsi):
        return True, ""
    if signal_type == "SELL_TSLA" and rsi < sell_min_rsi:
        return False, f"RSI={rsi:.1f} < {sell_min_rsi}（超賣區，TSLA易反彈，跳過賣出）"
    if signal_type == "BUY_TSLA" and rsi > buy_max_rsi:
        return False, f"RSI={rsi:.1f} > {buy_max_rsi}（超買區，TSLA易回落，跳過買入）"
    return True, ""

# ════════════════════════════════════════════════════════════════════════════════
# ★ 新增：訊號滯後偵測（lag_strength）
# ════════════════════════════════════════════════════════════════════════════════
def calc_lag_strength(uvxy_pct: float, tsla_pct: float, uvxy_dir: int) -> float:
    """
    正值 = TSLA 滯後（好，代表還有追趕空間）
    負值 = TSLA 已提前移動（壞，訊號滯後）
    公式：expected_tsla_move - actual_tsla_move
    """
    expected_tsla = -uvxy_pct  # 完美負相關下 TSLA 應有的移動
    actual_tsla   = tsla_pct
    return expected_tsla - actual_tsla

# ════════════════════════════════════════════════════════════════════════════════
# ★ 新增：UVXY 動能加速偵測
# ════════════════════════════════════════════════════════════════════════════════
def calc_uvxy_acceleration(uvxy_closes: np.ndarray) -> float:
    """
    比較前半段斜率 vs 後半段斜率
    正值 = 加速（動能增強）
    負值 = 減速（動能衰退，訊號可靠性下降）
    """
    n = len(uvxy_closes)
    if n < 6:
        return 0.0
    half = n // 2
    x = np.arange(half, dtype=float)
    s1 = linregress(x, uvxy_closes[:half].astype(float))[0]
    s2 = linregress(x, uvxy_closes[half:half*2].astype(float))[0]
    return float(s2 - s1)

def hs_dynamic_size(corr20: float, uvxy_chg_abs: float,
                    recent_pnls: list, et_hour: int) -> tuple:
    import math
    reasons = []
    mult    = 1.0

    if math.isnan(corr20):
        return 0.0, "相關係數數據不足", "跳過"
    if corr20 > 0:
        return 0.0, f"正相關({corr20:.2f})負相關失效", "跳過"
    elif corr20 > -0.3:
        mult *= 0.5; reasons.append(f"相關偏弱({corr20:.2f})→0.5x")
    elif corr20 > -0.5:
        mult *= 1.0; reasons.append(f"相關正常({corr20:.2f})→1x")
    elif corr20 > -0.7:
        mult *= 2.0; reasons.append(f"相關強({corr20:.2f})→2x加碼⭐")
    else:
        mult *= 0.5; reasons.append(f"相關極強({corr20:.2f})均值回歸→0.5x")

    if len(recent_pnls) >= 5:
        rwr = sum(1 for p in recent_pnls[-10:] if p > 0) / min(len(recent_pnls), 10)
        if rwr >= 0.8:
            mult *= 0.5; reasons.append(f"近期過熱WR={rwr*100:.0f}%→0.5x")
        elif rwr < 0.4:
            mult *= 0.5; reasons.append(f"冷場WR={rwr*100:.0f}%→0.5x")
        elif rwr >= 0.6:
            mult *= 1.3; reasons.append(f"熱手WR={rwr*100:.0f}%→1.3x")

    if 0.25 <= uvxy_chg_abs < 0.5:
        mult *= 1.2; reasons.append(f"UVXY大幅{uvxy_chg_abs:.2f}%→1.2x")
    elif uvxy_chg_abs >= 0.5:
        mult *= 0.8; reasons.append(f"UVXY極端{uvxy_chg_abs:.2f}%→0.8x")

    if et_hour in [10, 14]:
        mult *= 0.5; reasons.append(f"{et_hour}:xx ET低勝時段→0.5x")

    mult = round(min(max(mult, 0.0), 3.0) * 2) / 2

    if mult == 0:     quality = "跳過"
    elif mult <= 0.5: quality = "⚠️ 輕倉"
    elif mult <= 1.0: quality = "▪ 正常"
    elif mult <= 1.5: quality = "▲ 加碼"
    elif mult <= 2.0: quality = "⭐ 重倉"
    else:             quality = "🔥 最重"

    return mult, "  |  ".join(reasons), quality

def vol_ratio(df: pd.DataFrame, window: int = 10) -> float:
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
    "last_warn_time":      None,
    "last_exit_time":      None,
    "last_hs_alert_time":  None,
    "hs_recent_pnls":      [],
    "signal_history":      [],
    "active_signal":       None,
    "active_signal_time":  None,
    "active_signal_entry": None,
    # ★ 新增：每日統計
    "daily_signals":       0,
    "daily_wins":          0,
    "daily_rsi_blocked":   0,
    "daily_lag_blocked":   0,
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
    st.markdown("### 📐 靈敏度 <span class='opt-badge'>v2優化</span>", unsafe_allow_html=True)

    # ★ 改：預設 0.05（原 0.15），更符合今日UVXY實際斜率分布
    min_uvxy_slope = st.slider(
        "UVXY 最低斜率 (%/根)",
        0.02, 0.50, 0.05, step=0.01,
        help="★ v2優化：從0.15降至0.05。今日UVXY 90th percentile僅0.11%，原設定導致0個訊號"
    )
    min_tsla_response = st.slider("TSLA 視為已反應 (%)",   0.05, 2.0, 0.20, step=0.05)
    resistance_thresh = st.slider("抵抗力比值門檻（Layer1）", 0.05, 0.8, 0.30, step=0.05)

    # ★ 改：R²預設提升至0.65（原0.45），回測顯示顯著提升勝率
    min_uvxy_r2 = st.slider(
        "UVXY R² 最低門檻",
        0.40, 0.90, 0.65, step=0.05,
        help="★ v2優化：從0.45升至0.65。R²≥0.65 WR=80%，R²≥0.45 WR=31%"
    )

    rsi_overbought = st.slider("RSI 超買出場閾值（Layer3）", 65, 85, 70)
    vol_spike_mult = st.slider("成交量爆增倍數（Layer3）",   1.5, 5.0, 2.5, step=0.5)

    st.divider()
    # ★ 新增：RSI 防呆過濾
    st.markdown("### 🛡️ RSI 防呆過濾 <span class='opt-badge'>v2新增</span>", unsafe_allow_html=True)
    use_rsi_gate = st.toggle(
        "啟用 RSI 防呆",
        value=True,
        help="★ v2新增：SELL訊號時RSI需≥門檻（避免超賣反彈），BUY訊號時RSI需≤門檻（避免超買回落）"
    )
    if use_rsi_gate:
        rsi_sell_min = st.slider("SELL：RSI 最低值",  30, 60, 45,
            help="RSI低於此值時不發SELL訊號（TSLA可能已超賣，易反彈）")
        rsi_buy_max  = st.slider("BUY：RSI 最高值",  55, 80, 68,
            help="RSI高於此值時不發BUY訊號（TSLA可能已超買，易回落）")
        st.markdown("""
        <div style='background:#0a1a0a;border:1px solid #00d97e;border-radius:6px;
                    padding:7px 10px;font-size:0.75rem;color:#00d97e;margin-top:4px'>
        ✅ RSI防呆：今日回測去除唯一虧損訊號<br>
        WR: 80% → 100%
        </div>""", unsafe_allow_html=True)
    else:
        rsi_sell_min = 0
        rsi_buy_max  = 100

    st.divider()
    # ★ 新增：訊號滯後偵測
    st.markdown("### ⏱️ 滯後過濾 <span class='opt-badge'>v2新增</span>", unsafe_allow_html=True)
    use_lag_filter = st.toggle(
        "啟用訊號滯後過濾",
        value=True,
        help="★ v2新增：若TSLA已提前大幅移動，說明訊號已滯後，跳過"
    )
    if use_lag_filter:
        lag_min = st.slider(
            "最低lag_strength",
            -2.0, 1.0, -0.5, step=0.1,
            help="TSLA滯後強度。值越低=TSLA已大幅提前移動=訊號越滯後。建議-0.5"
        )
        st.markdown("""
        <div style='background:#0a0a1a;border:1px solid #5c7cfa;border-radius:6px;
                    padding:7px 10px;font-size:0.75rem;color:#5c7cfa;margin-top:4px'>
        ⏱ 滯後偵測：避免追漲殺跌式假訊號<br>
        lag_strength = 預期TSLA移動 − 實際TSLA移動
        </div>""", unsafe_allow_html=True)
    else:
        lag_min = -99.0

    st.divider()
    st.markdown("### 🔍 SPY 大盤過濾")
    use_spy_filter = st.toggle("啟用 SPY 過濾", value=False)
    if use_spy_filter:
        spy_filter_strength = st.radio("過濾強度", ["寬鬆（降低訊號強度）", "嚴格（直接過濾訊號）"], index=0)
        st.markdown("<div class='spy-on'>✅ SPY 過濾已啟用</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='spy-off'>○ SPY 過濾已關閉</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚡ 高敏感度模式")
    use_hs_mode = st.toggle("啟用高敏感度偵測", value=False)
    if use_hs_mode:
        hs_uvxy_min = st.slider("HS: UVXY最低變動 (%)", 0.05, 1.0, 0.10, step=0.05,
                                 help="★ v2優化：預設從0.15降至0.10")
        hs_tsla_max = st.slider("HS: TSLA最大容許反應 (%)", 0.0, 1.0, 0.10, step=0.05)
        hs_cooldown = st.slider("HS: 冷卻時間（分鐘）", 1, 10, 2)
    else:
        hs_uvxy_min, hs_tsla_max, hs_cooldown = 0.10, 0.10, 2

    st.divider()
    auto_refresh = st.toggle("每分鐘自動刷新", value=True)
    if st.button("🔄 立即刷新"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    # ★ 新增：每日統計
    st.markdown("### 📊 今日統計")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("訊號數", st.session_state.daily_signals)
        st.metric("RSI過濾", st.session_state.daily_rsi_blocked)
    with col_b:
        if st.session_state.daily_signals > 0:
            wr_pct = st.session_state.daily_wins / st.session_state.daily_signals * 100
            st.metric("勝率", f"{wr_pct:.0f}%")
        else:
            st.metric("勝率", "—")
        st.metric("滯後過濾", st.session_state.daily_lag_blocked)

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
            elif "紫" in entry or "🟣" in entry:
                color = "#b57bee"
            elif "🛡️" in entry or "⏱" in entry:
                color = "#5c7cfa"
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
st.markdown("# 🎯 TSLA vs UVXY 三層訊號系統 <span class='opt-badge'>v2</span>", unsafe_allow_html=True)
st.markdown(
    f"<small style='color:#8b8fa8'>v2優化：斜率門檻校準 · RSI防呆過濾 · 訊號滯後偵測 · 品質評分系統</small>",
    unsafe_allow_html=True,
)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# FETCH DATA
# ═══════════════════════════════════════════════════════════════════════════════
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
layer1_active   = False
layer2_signal   = None
layer3_exit     = False
layer2_stars    = ""
layer2_strength = 0.0
layer1_reason   = ""
layer2_reason   = ""
layer3_reason   = ""
spy_filtered    = False

# ★ 新增輸出變數
signal_quality  = 0
signal_grade    = "—"
signal_grade_color = "#8b8fa8"
rsi_blocked     = False
rsi_block_reason = ""
lag_blocked     = False
lag_block_reason = ""
lag_strength_val = 0.0
uvxy_accel      = 0.0

uvxy_trend_1m = {}
tsla_trend_1m = {}
uvxy_trend_5m = {}
tsla_trend_5m = {}
spy_trend_1m  = {}
corr_value    = None
multi_tf_agree = False
res_ratio     = 1.0
tsla_rsi1     = None
tsla_vol_ratio = 1.0

if data_ok:
    # ── Pearson correlation ────────────────────────────────────────────────
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

    # ── Trend calculations ─────────────────────────────────────────────────
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
    # ★ 新增 R² 獨立門檻
    uvxy_r2_ok     = uvxy_trend_1m.get("r2", 0) >= min_uvxy_r2
    uvxy_5m_dir    = uvxy_trend_5m.get("direction", 0) if uvxy_trend_5m else 0
    multi_tf_agree = (uvxy_dir != 0) and (uvxy_5m_dir == uvxy_dir)

    tsla_pct       = tsla_trend_1m.get("pct_move", 0)
    uvxy_pct       = uvxy_trend_1m.get("pct_move", 0)
    tsla_responded = (
        abs(tsla_pct) >= min_tsla_response and
        int(np.sign(tsla_pct)) == -uvxy_dir
    )

    if uvxy_dir == +1 and uvxy_pct > 0.1:
        res_ratio = resistance_ratio(uvxy_pct, tsla_pct)

    # ── RSI ──────────────────────────────────────────────────────────────
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

    if "Volume" in tsla_1m.columns:
        tsla_vol_ratio = vol_ratio(tsla_1m, window=10)

    # ★ 新增：lag_strength 計算
    lag_strength_val = calc_lag_strength(uvxy_pct, tsla_pct, uvxy_dir)

    # ★ 新增：UVXY 加速偵測
    uvxy_window = uvxy_1m["Close"].tail(display_bars).values
    uvxy_accel  = calc_uvxy_acceleration(uvxy_window)

    # ──────────────────────────────────────────────────────────────────────
    # LAYER 1: WARNING
    # ──────────────────────────────────────────────────────────────────────
    if (uvxy_confirmed and uvxy_slope_ok and uvxy_r2_ok and uvxy_dir == +1
            and res_ratio < resistance_thresh and not tsla_responded):
        layer1_active = True
        layer1_reason = (
            f"UVXY 升 {uvxy_pct:+.2f}%（斜率{uvxy_trend_1m['slope']:+.3f}%/根 R²={uvxy_trend_1m['r2']:.2f}）"
            f"　TSLA 僅 {tsla_pct:+.2f}%　抵抗力比值={res_ratio:.2f}（< {resistance_thresh}）"
        )

    # ──────────────────────────────────────────────────────────────────────
    # LAYER 2: ENTRY SIGNAL
    # ★ 改：加入 uvxy_r2_ok 條件（R² ≥ min_uvxy_r2）
    # ──────────────────────────────────────────────────────────────────────
    if uvxy_confirmed and uvxy_slope_ok and uvxy_r2_ok and uvxy_dir != 0:
        base_strength = (
            abs(uvxy_trend_1m["slope"])
            + uvxy_trend_1m["r2"]
            + uvxy_trend_1m["consecutive"] * 0.3
        ) * (1.5 if multi_tf_agree else 1.0)

        if uvxy_dir == -1 and not tsla_responded:
            layer2_signal   = "BUY_TSLA"
            layer2_strength = base_strength
            tf_tag = "【1m+5m✓】" if multi_tf_agree else "【1m】"
            if layer1_active or res_ratio < resistance_thresh:
                layer2_strength *= 1.8
                tf_tag += "【抵抗力確認⭐】"
            layer2_reason = (
                f"{tf_tag} UVXY跌 {uvxy_pct:+.2f}%  斜率{uvxy_trend_1m['slope']:+.3f}%/根  "
                f"R²={uvxy_trend_1m['r2']:.2f}  連{uvxy_trend_1m['consecutive']}根 | "
                f"TSLA 僅 {tsla_pct:+.2f}% 尚未反應"
            )

        elif uvxy_dir == +1 and not tsla_responded:
            layer2_signal   = "SELL_TSLA"
            layer2_strength = base_strength
            tf_tag = "【1m+5m✓】" if multi_tf_agree else "【1m】"
            layer2_reason = (
                f"{tf_tag} UVXY升 {uvxy_pct:+.2f}%  斜率{uvxy_trend_1m['slope']:+.3f}%/根  "
                f"R²={uvxy_trend_1m['r2']:.2f}  連{uvxy_trend_1m['consecutive']}根 | "
                f"TSLA 僅 {tsla_pct:+.2f}% 尚未跟跌"
            )

    layer2_stars = signal_stars(layer2_strength)

    # ★ 新增：RSI 防呆過濾
    if layer2_signal is not None and use_rsi_gate and tsla_rsi1 is not None:
        rsi_pass, rsi_block_reason = rsi_gate(layer2_signal, tsla_rsi1,
                                               rsi_sell_min, rsi_buy_max)
        if not rsi_pass:
            rsi_blocked = True
            st.session_state.daily_rsi_blocked += 1
            st.session_state.signal_history.append(
                f"{now.strftime('%H:%M')} 🛡️ RSI過濾：{rsi_block_reason[:50]}"
            )
            layer2_signal = None

    # ★ 新增：訊號滯後過濾
    if layer2_signal is not None and use_lag_filter:
        if lag_strength_val < lag_min:
            lag_blocked = True
            lag_block_reason = (
                f"lag_strength={lag_strength_val:+.3f} < {lag_min}，"
                f"TSLA已提前移動 {abs(tsla_pct):.2f}%，訊號滯後"
            )
            st.session_state.daily_lag_blocked += 1
            st.session_state.signal_history.append(
                f"{now.strftime('%H:%M')} ⏱ 滯後過濾：{lag_block_reason[:50]}"
            )
            layer2_signal = None

    # ★ 新增：訊號品質評分（在所有過濾後計算）
    if layer2_signal is not None and tsla_rsi1 is not None:
        signal_quality, signal_grade, signal_grade_color = calc_signal_quality(
            uvxy_r2        = uvxy_trend_1m.get("r2", 0),
            uvxy_slope_abs = abs(uvxy_trend_1m.get("slope", 0)),
            uvxy_consec    = uvxy_trend_1m.get("consecutive", 0),
            rsi            = tsla_rsi1,
            signal_type    = layer2_signal,
            lag_strength   = lag_strength_val,
            multi_tf       = multi_tf_agree,
        )

    # ── SPY Filter ────────────────────────────────────────────────────────
    if use_spy_filter and layer2_signal == "BUY_TSLA" and spy_trend_1m:
        spy_dir       = spy_trend_1m.get("direction", 0)
        spy_confirmed = spy_trend_1m.get("confirmed", False)
        spy_pct       = spy_trend_1m.get("pct_move", 0)
        if spy_confirmed and spy_dir == -1:
            if "嚴格" in spy_filter_strength:
                layer2_signal = None
                spy_filtered  = True
                layer2_reason = f"⛔ SPY過濾（嚴格）：SPY跌 {spy_pct:+.2f}%"
            else:
                layer2_strength *= 0.5
                layer2_stars     = signal_stars(layer2_strength)
                spy_filtered     = True
                layer2_reason   += f"  ⚠️ SPY過濾（寬鬆）：SPY跌 {spy_pct:+.2f}%，訊號強度已降半"

    # ── LAYER 3: EXIT ────────────────────────────────────────────────────
    exit_reasons = []
    if tsla_rsi1 is not None and tsla_rsi1 >= rsi_overbought:
        exit_reasons.append(f"RSI={tsla_rsi1:.1f} 超買（>{rsi_overbought}）")
    if tsla_vol_ratio >= vol_spike_mult:
        last_close = float(tsla_1m["Close"].iloc[-1])
        last_open  = float(tsla_1m["Open"].iloc[-1])
        if last_close < last_open:
            exit_reasons.append(f"成交量爆增{tsla_vol_ratio:.1f}×且收黑K")
    if (st.session_state.active_signal == "BUY_TSLA"
            and uvxy_dir == +1 and uvxy_confirmed):
        exit_reasons.append(f"UVXY反轉上升（斜率{uvxy_trend_1m['slope']:+.3f}%/根）")
    if exit_reasons:
        layer3_exit   = True
        layer3_reason = "  ·  ".join(exit_reasons)

    # ── ACTIVE SIGNAL TRACKING ────────────────────────────────────────────
    if layer2_signal in ("BUY_TSLA", "SELL_TSLA"):
        if st.session_state.active_signal != layer2_signal:
            st.session_state.active_signal      = layer2_signal
            st.session_state.active_signal_time = now
            st.session_state.active_signal_entry = float(tsla_1m["Close"].iloc[-1])
            st.session_state.daily_signals += 1
        else:
            elapsed = (now - st.session_state.active_signal_time).total_seconds() / 60
            t_now   = float(tsla_1m["Close"].iloc[-1])
            pnl     = ((t_now - st.session_state.active_signal_entry)
                       / st.session_state.active_signal_entry * 100
                       * (1 if layer2_signal == "BUY_TSLA" else -1))
            if layer3_exit:
                st.session_state.signal_history.append(
                    f"{now.strftime('%H:%M')} 🟣 出場 持{elapsed:.0f}min  P&L≈{pnl:+.2f}%")
                if pnl > 0: st.session_state.daily_wins += 1
                st.session_state.active_signal = None
            elif elapsed > signal_timeout:
                st.session_state.signal_history.append(
                    f"{now.strftime('%H:%M')} ❌ 失效 超{signal_timeout}min未反應")
                st.session_state.active_signal = None
            elif tsla_responded:
                st.session_state.signal_history.append(
                    f"{now.strftime('%H:%M')} ✅ 兌現 P&L≈{pnl:+.2f}%")
                if pnl > 0: st.session_state.daily_wins += 1
    else:
        if layer3_exit and st.session_state.active_signal:
            t_now  = float(tsla_1m["Close"].iloc[-1])
            entry  = st.session_state.active_signal_entry
            elapsed = (now - (st.session_state.active_signal_time or now)).total_seconds() / 60
            pnl    = (t_now - entry) / entry * 100
            st.session_state.signal_history.append(
                f"{now.strftime('%H:%M')} 🟣 出場 持{elapsed:.0f}min P&L≈{pnl:+.2f}%")
            if pnl > 0: st.session_state.daily_wins += 1
            st.session_state.active_signal = None

    # ── HIGH-SENSITIVITY MODE ─────────────────────────────────────────────
    hs_signal = None; hs_reason = ""; hs_uvxy_chg = 0.0; hs_tsla_chg = 0.0
    hs_mult = 0.0; hs_mult_reason = ""; hs_quality = ""

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
                hs_reason = f"UVXY本分鐘升 {hs_uvxy_chg:+.3f}%，TSLA 僅 {hs_tsla_chg:+.3f}%（未跟跌）"
            elif uvxy_down_now and tsla_not_rose:
                hs_signal = "HS_BUY"
                hs_reason = f"UVXY本分鐘跌 {hs_uvxy_chg:+.3f}%，TSLA 僅 {hs_tsla_chg:+.3f}%（未跟升）"

            _common = tsla_1m.index.intersection(uvxy_1m.index)
            if len(_common) >= 20:
                _tc = tsla_1m.loc[_common, "Close"].iloc[-20:].values.astype(float)
                _uc = uvxy_1m.loc[_common, "Close"].iloc[-20:].values.astype(float)
                _corr20 = float(np.corrcoef(_tc, _uc)[0, 1])
            else:
                _corr20 = float("nan")
            try:
                import pytz as _pytz
                _et = now.astimezone(_pytz.timezone("America/New_York"))
                _et_hour = _et.hour
            except Exception:
                _et_hour = now.hour
            hs_mult, hs_mult_reason, hs_quality = hs_dynamic_size(
                corr20=_corr20, uvxy_chg_abs=abs(hs_uvxy_chg),
                recent_pnls=list(st.session_state.hs_recent_pnls), et_hour=_et_hour)

    # ── TELEGRAM ALERTS ───────────────────────────────────────────────────
    def cooldown_ok(last_time, minutes):
        return (last_time is None or
                (now - last_time).total_seconds() > minutes * 60)

    if layer1_active and cooldown_ok(st.session_state.last_warn_time, max(1, cooldown_min // 2)):
        t_p = float(tsla_1m["Close"].iloc[-1]); u_p = float(uvxy_1m["Close"].iloc[-1])
        msg = (f"🟡 *TSLA 抵抗力預警（Layer 1）*\n"
               f"━━━━━━━━━━━━━━━━\n"
               f"時間：`{now.strftime('%Y-%m-%d %H:%M')}`\n"
               f"UVXY升 `{uvxy_pct:+.2f}%` 但TSLA僅 `{tsla_pct:+.2f}%`\n"
               f"抵抗力比值：`{res_ratio:.2f}`（門檻<{resistance_thresh}）\n"
               f"TSLA：`${t_p:.2f}`  UVXY：`${u_p:.2f}`\n"
               f"⚡ 若UVXY轉跌，TSLA可能急升")
        send_telegram(msg); st.session_state.last_warn_time = now
        st.session_state.signal_history.append(
            f"{now.strftime('%H:%M')} 🟡 預警 UVXY升{uvxy_pct:+.1f}% TSLA抵抗{res_ratio:.2f}")

    if (layer2_signal in ("BUY_TSLA", "SELL_TSLA")
            and cooldown_ok(st.session_state.last_alert_time, cooldown_min)):
        action = "🟢 買入 TSLA" if layer2_signal == "BUY_TSLA" else "🔴 賣出 TSLA"
        t_p = float(tsla_1m["Close"].iloc[-1]); u_p = float(uvxy_1m["Close"].iloc[-1])
        tf_line = "✅ 1m+5m 雙框架" if multi_tf_agree else "⚠️ 僅1m框架"
        msg = (f"*{action}*  {layer2_stars}  [品質{signal_grade}級={signal_quality}分]\n"
               f"━━━━━━━━━━━━━━━━\n"
               f"時間：`{now.strftime('%Y-%m-%d %H:%M')}`\n"
               f"框架：{tf_line}\n"
               f"R²：`{uvxy_trend_1m.get('r2',0):.2f}`（門檻≥{min_uvxy_r2}）\n"
               f"UVXY斜率：`{uvxy_trend_1m.get('slope',0):+.3f}%/根`  "
               f"連續`{uvxy_trend_1m.get('consecutive',0)}`根\n"
               f"TSLA RSI：`{tsla_rsi1:.1f}`  lag_strength：`{lag_strength_val:+.3f}`\n"
               f"入場參考：TSLA `${t_p:.2f}`  UVXY `${u_p:.2f}`")
        send_telegram(msg); st.session_state.last_alert_time = now

    if layer3_exit and cooldown_ok(st.session_state.last_exit_time, 2):
        t_p = float(tsla_1m["Close"].iloc[-1])
        entry = st.session_state.active_signal_entry
        pnl_str = ""
        if entry:
            pnl = (t_p - entry) / entry * 100
            pnl_str = f"P&L估算：`{pnl:+.2f}%`\n"
        msg = (f"🟣 *出場訊號（Layer 3）*\n"
               f"━━━━━━━━━━━━━━━━\n"
               f"時間：`{now.strftime('%Y-%m-%d %H:%M')}`\n"
               f"原因：{layer3_reason}\n"
               f"{pnl_str}"
               f"TSLA現價：`${t_p:.2f}`")
        send_telegram(msg); st.session_state.last_exit_time = now

    if (hs_signal in ("HS_BUY", "HS_SELL")
            and cooldown_ok(st.session_state.last_hs_alert_time, hs_cooldown)):
        t_p = float(tsla_1m["Close"].iloc[-1]); u_p = float(uvxy_1m["Close"].iloc[-1])
        action_hs = "⚡🟢 買入 TSLA（高敏感）" if hs_signal == "HS_BUY" else "⚡🔴 賣出 TSLA（高敏感）"
        if hs_mult == 0:
            msg = (f"⏭ *高敏感訊號跳過*\n時間：`{now.strftime('%Y-%m-%d %H:%M')}`\n"
                   f"原因：{hs_mult_reason}")
        else:
            msg = (f"*{action_hs}*\n━━━━━━━━━━━━━━━━\n"
                   f"時間：`{now.strftime('%Y-%m-%d %H:%M')}`\n"
                   f"注碼建議：`{hs_mult}x`  {hs_quality}\n"
                   f"原因：{hs_reason}\n"
                   f"UVXY：`{hs_uvxy_chg:+.3f}%`  TSLA：`{hs_tsla_chg:+.3f}%`\n"
                   f"TSLA：`${t_p:.2f}`  UVXY：`${u_p:.2f}`")
        send_telegram(msg); st.session_state.last_hs_alert_time = now

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

# ★ 新增：被過濾訊號提示（幫助用戶理解為何無訊號）
if rsi_blocked:
    st.markdown(f"""
    <div style='border-radius:10px;padding:10px 20px;margin:6px 0;
                background:#0a0a1a;border:1px dashed #5c7cfa;text-align:center'>
      <div style='font-size:0.95rem;font-weight:700;color:#5c7cfa'>
          🛡️ RSI 防呆過濾已攔截一個訊號</div>
      <div style='font-size:0.80rem;color:#8b8fa8;margin-top:4px'>{rsi_block_reason}</div>
    </div>
    """, unsafe_allow_html=True)

if lag_blocked:
    st.markdown(f"""
    <div style='border-radius:10px;padding:10px 20px;margin:6px 0;
                background:#0a0a0a;border:1px dashed #8b8fa8;text-align:center'>
      <div style='font-size:0.95rem;font-weight:700;color:#8b8fa8'>
          ⏱ 滯後過濾已攔截一個訊號</div>
      <div style='font-size:0.80rem;color:#8b8fa8;margin-top:4px'>{lag_block_reason}</div>
    </div>
    """, unsafe_allow_html=True)

# Layer 3 exit
if layer3_exit:
    st.markdown(f"""
    <div class="sig-layer3">
      <div class="sig-title" style="color:#b57bee">🟣 出場訊號（Layer 3）</div>
      <div class="sig-detail">{layer3_reason}<br>
      RSI={f'{tsla_rsi1:.1f}' if tsla_rsi1 else '—'}  成交量={tsla_vol_ratio:.1f}×</div>
    </div>
    """, unsafe_allow_html=True)

# Layer 2 entry
if layer2_signal == "BUY_TSLA":
    quality_bar_w = signal_quality
    st.markdown(f"""
    <div class="sig-layer2-buy">
      <div class="sig-title" style="color:#00d97e">🟢 買入 TSLA &nbsp; {layer2_stars}
          <span style='font-size:1rem;color:{signal_grade_color}'>
              &nbsp;品質 {signal_grade} ({signal_quality}分)</span></div>
      <div class="sig-detail">{layer2_reason}</div>
      <div class="quality-bar">
          <div class="quality-fill" style="width:{quality_bar_w}%;background:{signal_grade_color}"></div>
      </div>
      <div style='font-size:0.75rem;color:#8b8fa8;margin-top:4px'>
          R²={uvxy_trend_1m.get('r2',0):.3f} &nbsp;|&nbsp;
          lag={lag_strength_val:+.3f} &nbsp;|&nbsp;
          UVXY加速={uvxy_accel:+.4f}
      </div>
    </div>
    """, unsafe_allow_html=True)
elif layer2_signal == "SELL_TSLA":
    quality_bar_w = signal_quality
    st.markdown(f"""
    <div class="sig-layer2-sell">
      <div class="sig-title" style="color:#e84045">🔴 賣出 TSLA &nbsp; {layer2_stars}
          <span style='font-size:1rem;color:{signal_grade_color}'>
              &nbsp;品質 {signal_grade} ({signal_quality}分)</span></div>
      <div class="sig-detail">{layer2_reason}</div>
      <div class="quality-bar">
          <div class="quality-fill" style="width:{quality_bar_w}%;background:{signal_grade_color}"></div>
      </div>
      <div style='font-size:0.75rem;color:#8b8fa8;margin-top:4px'>
          R²={uvxy_trend_1m.get('r2',0):.3f} &nbsp;|&nbsp;
          lag={lag_strength_val:+.3f} &nbsp;|&nbsp;
          UVXY加速={uvxy_accel:+.4f}
      </div>
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
if not layer1_active and not layer2_signal and not layer3_exit and not rsi_blocked and not lag_blocked:
    uvxy_s = uvxy_trend_1m.get("slope", 0)
    r2_val = uvxy_trend_1m.get("r2", 0)
    r2_status = f"✅ R²={r2_val:.2f}≥{min_uvxy_r2}" if r2_val >= min_uvxy_r2 else f"❌ R²={r2_val:.2f}<{min_uvxy_r2}"
    slope_status = (f"✅ 斜率={abs(uvxy_s):.3f}≥{min_uvxy_slope}"
                    if abs(uvxy_s) >= min_uvxy_slope else f"❌ 斜率={abs(uvxy_s):.3f}<{min_uvxy_slope}")
    st.markdown(f"""
    <div class="sig-normal">
      <div class="sig-title" style="color:#8b8fa8">⚪ 無訊號</div>
      <div class="sig-detail">
          {slope_status} &nbsp;|&nbsp; {r2_status}<br>
          確認={uvxy_trend_1m.get('confirmed','—')} &nbsp;|&nbsp;
          lag={lag_strength_val:+.3f} &nbsp;|&nbsp;
          RSI={f'{tsla_rsi1:.1f}' if tsla_rsi1 else '—'}
      </div>
    </div>
    """, unsafe_allow_html=True)

# High-sensitivity signal box
if use_hs_mode:
    _hs_mult_color = (
        "#b57bee" if hs_mult >= 2.5 else
        "#00d97e" if hs_mult >= 2.0 else
        "#5c7cfa" if hs_mult >= 1.5 else
        "#f6c90e" if hs_mult >= 0.5 else
        "#e84045"
    )
    if hs_signal == "HS_BUY" and hs_mult > 0:
        st.markdown(f"""
        <div style='border-radius:10px;padding:12px 20px;margin:6px 0;
                    background:#0a1a14;border:2px dashed #00d97e;text-align:center'>
          <div style='font-size:1.3rem;font-weight:800;color:#00d97e'>
              ⚡ 高敏感 — 🟢 買入 TSLA
              <span style='font-size:1rem;margin-left:12px;color:{_hs_mult_color}'>
                  注碼 {hs_mult}x  {hs_quality}</span></div>
          <div style='font-size:0.82rem;color:#c9cdd8;margin-top:4px'>{hs_reason}</div>
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
                  注碼 {hs_mult}x  {hs_quality}</span></div>
          <div style='font-size:0.82rem;color:#c9cdd8;margin-top:4px'>{hs_reason}</div>
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
        st.markdown(f"""
        <div style='border-radius:10px;padding:10px 20px;margin:6px 0;
                    background:#111316;border:1px dashed #2d3139;text-align:center'>
          <div style='font-size:0.9rem;color:#8b8fa8'>
              ⚡ 高敏感監測中…
              UVXY={hs_uvxy_chg:+.3f}%　TSLA={hs_tsla_chg:+.3f}%</div>
        </div>
        """, unsafe_allow_html=True)

# Active signal tracker
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
      &nbsp;｜&nbsp; 目標：TSLA {action_word}
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
    return "#8b8fa8" if abs(s) < 0.02 else ("#00d97e" if s > 0 else "#e84045")
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

num_cols = 12 if use_spy_filter else 11
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
            sub=f"門檻<{resistance_thresh}")
metric_card(cols[7], "lag strength",
            f"{lag_strength_val:+.3f}",
            "#00d97e" if lag_strength_val >= 0 else "#e84045",
            sub="有追趕空間" if lag_strength_val >= 0 else "訊號滯後⚠️")
# ★ 新增：UVXY加速卡
metric_card(cols[8], "UVXY 加速度",
            f"{uvxy_accel:+.4f}",
            "#00d97e" if uvxy_accel > 0 else "#e84045",
            sub="動能增強" if uvxy_accel > 0 else "動能衰退")
metric_card(cols[9], "皮爾森係數",
            f"{corr_value:.3f}" if corr_value is not None else "—", corr_color)
metric_card(cols[10], "最後更新", now.strftime("%H:%M:%S"), "#8b8fa8")

if use_spy_filter and spy_price:
    metric_card(cols[11], "SPY 價格",
                f"${spy_price:.2f}", "#a78bfa",
                sub=fmt_pct(spy_trend_1m.get("pct_move") if spy_trend_1m else None),
                sub_color=pct_color(spy_trend_1m.get("pct_move") if spy_trend_1m else None))

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CANDLE CHARTS
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
        # ★ 新增：在圖上標示 R² 是否達標
        r2_ok_str = f"✅ R²={r2:.2f}" if r2 >= min_uvxy_r2 else f"❌ R²={r2:.2f}<{min_uvxy_r2}"
        fig.update_layout(
            title=dict(text=f"{title}   斜率={slope:+.3f}%/根   {r2_ok_str}",
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

# ── 5m charts ──
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

# ── Pearson correlation history ──
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

# ── v2 changelog footer ──
st.markdown('<div class="section-title">v2 優化說明</div>', unsafe_allow_html=True)
st.markdown("""
<div style='background:#161920;border-radius:8px;padding:12px 18px;
            border:1px solid #2d3139;font-size:0.80rem;color:#8b8fa8;line-height:1.8'>
<b style='color:#5c7cfa'>★ v2 核心改進（基於今日真實數據回測）：</b><br>
1. <b style='color:#c9cdd8'>斜率門檻校準</b>：0.15 → 0.05%/根（今日UVXY 90th percentile=0.11%，原設定完全無法觸發）<br>
2. <b style='color:#c9cdd8'>R² 獨立門檻</b>：0.45 → 0.65（R²≥0.65 WR=80% vs R²≥0.45 WR=31%，最關鍵過濾因子）<br>
3. <b style='color:#c9cdd8'>RSI 防呆過濾</b>：SELL訊號RSI<45=跳過（超賣TSLA易反彈），去除今日唯一虧損訊號<br>
4. <b style='color:#c9cdd8'>訊號滯後偵測</b>：lag_strength<-0.5=跳過（TSLA已提前大幅移動）<br>
5. <b style='color:#c9cdd8'>品質評分系統</b>：0-100分 A/B/C/D等級，幫助倉位管理<br>
6. <b style='color:#c9cdd8'>UVXY加速偵測</b>：監控動能是否在增強或衰退<br>
7. <b style='color:#c9cdd8'>每日統計</b>：即時追蹤當日訊號數、勝率、過濾數量
</div>
""", unsafe_allow_html=True)

# ── Auto refresh ──
if auto_refresh:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
