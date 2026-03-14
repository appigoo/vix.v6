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
st.set_page_config(page_title="TSLA vs UVXY 三層訊號系統 v3b", page_icon="🎯", layout="wide")

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
  /* ── 時段標籤 ── */
  .tod-hot  { background:#0a2b16; border:1px solid #00d97e; border-radius:5px;
              padding:2px 8px; color:#00d97e; font-size:0.74rem; font-weight:700; display:inline-block; }
  .tod-warm { background:#1a1a0a; border:1px solid #f6c90e; border-radius:5px;
              padding:2px 8px; color:#f6c90e; font-size:0.74rem; font-weight:700; display:inline-block; }
  .tod-cold { background:#1c1f26; border:1px solid #2d3139; border-radius:5px;
              padding:2px 8px; color:#8b8fa8; font-size:0.74rem; display:inline-block; }
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
                         multi_tf: bool, et_hour: int = -1) -> tuple:
    """
    Returns (score 0-100, grade label, color hex)
    Grade: A(80+), B(60-79), C(40-59), D(<40)
    v3 更新：
      - consec 權重大幅提升（5日回測：consec=4 WR=82%，consec=2 WR=44%）
      - SELL RSI 加入上限（RSI≥55 時 SELL WR=0%）
      - 時段加權（10點 WR=72%，其他時段 WR=40-53%）
    """
    score = 0.0

    # R² quality (max 20 pts) — 修正v3b：0.65為基準，非0.80
    if uvxy_r2 >= 0.85:   score += 20
    elif uvxy_r2 >= 0.80: score += 17
    elif uvxy_r2 >= 0.75: score += 13
    elif uvxy_r2 >= 0.70: score += 9
    elif uvxy_r2 >= 0.65: score += 5
    else:                  score += 0

    # Slope magnitude (max 10 pts)
    if uvxy_slope_abs >= 0.15:   score += 10
    elif uvxy_slope_abs >= 0.10: score += 7
    elif uvxy_slope_abs >= 0.07: score += 5
    elif uvxy_slope_abs >= 0.05: score += 3
    else:                         score += 0

    # ★ v3: Consecutive bars 大幅提升至 max 35 pts（5日回測最強因子）
    # consec=2→0pts, consec=3→12pts, consec=4→25pts, consec≥5→35pts
    if uvxy_consec >= 5:   score += 35
    elif uvxy_consec >= 4: score += 25
    elif uvxy_consec >= 3: score += 12
    else:                   score += 0   # consec=2 不加分

    # ★ v3: RSI 雙向門檻（上下限）
    if signal_type == "SELL_TSLA":
        # SELL: RSI 45-55 最佳（WR 67-71%），RSI≥55 WR=0%
        if 45 <= rsi < 55:    score += 20
        elif 40 <= rsi < 45:  score += 12
        elif 35 <= rsi < 40:  score += 8
        elif rsi >= 55:        score += 0   # ★ RSI≥55 SELL 完全無效
        else:                  score += 3   # RSI<35 極少見，謹慎
    else:  # BUY_TSLA
        # BUY: RSI 40-65 合理，RSI<40 反而差（WR=33%）
        if 40 <= rsi < 65:    score += 20
        elif 65 <= rsi < 68:  score += 12
        elif rsi >= 68:        score += 5
        else:                  score += 3   # RSI<40 謹慎

    # Lag strength (max 8 pts)
    if lag_strength >= 3.0:   score += 8
    elif lag_strength >= 1.0: score += 6
    elif lag_strength >= 0.0: score += 4
    elif lag_strength >= -0.5: score += 1
    else:                      score += 0

    # Multi-timeframe bonus (max 4 pts)
    if multi_tf: score += 4

    # ★ v3: 時段加權（5日回測：10點 WR=72%，其他時段 44-54%）
    if et_hour == 10:          score += 3   # 黃金時段
    elif et_hour in [9, 14]:   score += 1
    elif et_hour in [11, 12, 13, 15]: score += 0  # 一般時段不加分

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
             sell_min_rsi: float, sell_max_rsi: float,
             buy_max_rsi: float, buy_min_rsi: float) -> tuple:
    """
    v3 雙向 RSI 門檻：
    SELL：sell_min_rsi ≤ RSI ≤ sell_max_rsi
    BUY ：buy_min_rsi  ≤ RSI ≤ buy_max_rsi
    5日回測發現：SELL RSI≥55 WR=0%，必須加上限
    """
    if rsi is None or np.isnan(rsi):
        return True, ""
    if signal_type == "SELL_TSLA":
        if rsi < sell_min_rsi:
            return False, f"RSI={rsi:.1f} < {sell_min_rsi}（超賣，TSLA易反彈，跳過SELL）"
        if rsi > sell_max_rsi:
            return False, f"RSI={rsi:.1f} > {sell_max_rsi}（TSLA過強，SELL勝率0%，跳過）"
    if signal_type == "BUY_TSLA":
        if rsi > buy_max_rsi:
            return False, f"RSI={rsi:.1f} > {buy_max_rsi}（超買，TSLA易回落，跳過BUY）"
        if rsi < buy_min_rsi:
            return False, f"RSI={rsi:.1f} < {buy_min_rsi}（RSI過低，BUY勝率低，跳過）"
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
    st.markdown("### 📐 靈敏度 <span class='opt-badge'>v3優化</span>", unsafe_allow_html=True)

    min_uvxy_slope = st.slider(
        "UVXY 最低斜率 (%/根)",
        0.02, 0.50, 0.05, step=0.01,
        help="v2校準：0.15→0.05。5日UVXY 90th percentile=0.15~0.21%"
    )
    min_tsla_response = st.slider("TSLA 視為已反應 (%)", 0.05, 2.0, 0.20, step=0.05)
    resistance_thresh = st.slider("抵抗力比值門檻（Layer1）", 0.05, 0.8, 0.30, step=0.05)

    min_uvxy_r2 = st.slider(
        "UVXY R² 最低門檻",
        0.40, 0.95, 0.65, step=0.05,
        help="★ 修正v3：恢復0.65（0.80在高波動日會過濾所有訊號）\n5日回測：R²≥0.65+consec≥3 WR=68% N=41，R²≥0.80+consec≥3 WR=63% N=19"
    )

    # ★ v3 核心：consec 預設改為 3
    consec_req = st.slider(
        "UVXY 連續確認根數",
        2, 6, 3,
        help="★ v3關鍵改進：consec=2 WR=44%，consec=3 WR=61%，consec=4 WR=82%（5日回測）"
    )

    rsi_overbought = st.slider("RSI 超買出場閾值（Layer3）", 65, 85, 70)
    vol_spike_mult = st.slider("成交量爆增倍數（Layer3）", 1.5, 5.0, 2.5, step=0.5)

    st.divider()
    # ★ v3：RSI 雙向門檻
    st.markdown("### 🛡️ RSI 雙向過濾 <span class='opt-badge'>v3升級</span>", unsafe_allow_html=True)
    use_rsi_gate = st.toggle(
        "啟用 RSI 雙向防呆",
        value=True,
        help="v3升級：加入 SELL 上限。5日回測：SELL RSI≥55 WR=0%，必須過濾"
    )
    if use_rsi_gate:
        col_rsi1, col_rsi2 = st.columns(2)
        with col_rsi1:
            rsi_sell_min = st.slider("SELL RSI 下限", 25, 50, 35,
                help="SELL 時 RSI 需高於此值（超賣區 TSLA 易反彈）")
            rsi_buy_min  = st.slider("BUY RSI 下限",  20, 45, 38,
                help="BUY 時 RSI 需高於此值（過低時 BUY WR 僅 33%）")
        with col_rsi2:
            rsi_sell_max = st.slider("SELL RSI 上限", 48, 70, 55,
                help="★ v3新增：SELL RSI≥55 WR=0%，必須設上限")
            rsi_buy_max  = st.slider("BUY RSI 上限",  55, 82, 68,
                help="BUY 時 RSI 需低於此值（超買易回落）")
        st.markdown("""
        <div style='background:#0a1a0a;border:1px solid #00d97e;border-radius:6px;
                    padding:7px 10px;font-size:0.75rem;color:#00d97e;margin-top:4px'>
        ✅ v3雙向RSI：SELL需在下限~上限之間<br>
        5日回測：SELL RSI≥55 WR=0%（6個全輸）
        </div>""", unsafe_allow_html=True)
    else:
        rsi_sell_min, rsi_sell_max = 0, 100
        rsi_buy_min,  rsi_buy_max  = 0, 100

    st.divider()
    # ★ v3：時段過濾
    st.markdown("### 🕙 時段過濾 <span class='opt-badge'>v3新增</span>", unsafe_allow_html=True)
    use_time_filter = st.toggle(
        "啟用時段優先模式",
        value=False,
        help="5日回測：10點 WR=72%，其他時段44-54%。開啟後在非優質時段降低訊號強度"
    )
    if use_time_filter:
        time_filter_mode = st.radio(
            "時段模式",
            ["寬鬆（非優質時段降級）", "嚴格（只在優質時段發訊）"],
            index=0,
        )
        st.markdown("""
        <div style='display:flex;gap:6px;margin-top:4px;flex-wrap:wrap'>
        <span class='tod-hot'>10:00 WR=72%</span>
        <span class='tod-warm'>09:xx WR=50%</span>
        <span class='tod-warm'>14:xx WR=54%</span>
        <span class='tod-cold'>11-13:xx WR=40-53%</span>
        <span class='tod-cold'>15:xx WR=40%</span>
        </div>""", unsafe_allow_html=True)
    else:
        time_filter_mode = "關閉"

    st.divider()
    # ★ v3：lag filter（維持，但5日回測顯示效果有限）
    st.markdown("### ⏱️ 滯後過濾 <span class='opt-badge'>v2新增</span>", unsafe_allow_html=True)
    use_lag_filter = st.toggle(
        "啟用訊號滯後過濾",
        value=False,
        help="5日回測：lag_strength對勝率影響不大（WR差距僅1-2%），v3預設關閉"
    )
    if use_lag_filter:
        lag_min = st.slider(
            "最低lag_strength",
            -2.0, 2.0, 0.0, step=0.1,
            help="正值=TSLA還有追趕空間。5日數據顯示BUY lag 3-5 時 WR=82%"
        )
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
                                 help="v2優化：預設從0.15降至0.10")
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
st.markdown("# 🎯 TSLA vs UVXY 三層訊號系統 <span class='opt-badge'>v3b</span>", unsafe_allow_html=True)
st.markdown(
    f"<small style='color:#8b8fa8'>v3b修正：R²恢復0.65 · consec≥3為核心 · SELL RSI雙向(35-55) · 5日41個訊號WR=68%</small>",
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
time_blocked    = False
time_block_reason = ""
et_hour_now     = now.hour

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

    # ── ET 時區換算（用於時段過濾與品質評分）─────────────────────────────
    try:
        import pytz as _pytz
        _et_now = now.astimezone(_pytz.timezone("America/New_York"))
        et_hour_now = _et_now.hour
    except Exception:
        et_hour_now = now.hour

    # ★ v3：RSI 雙向防呆過濾
    if layer2_signal is not None and use_rsi_gate and tsla_rsi1 is not None:
        rsi_pass, rsi_block_reason = rsi_gate(
            layer2_signal, tsla_rsi1,
            rsi_sell_min, rsi_sell_max,
            rsi_buy_max,  rsi_buy_min,
        )
        if not rsi_pass:
            rsi_blocked = True
            st.session_state.daily_rsi_blocked += 1
            st.session_state.signal_history.append(
                f"{now.strftime('%H:%M')} 🛡️ RSI過濾：{rsi_block_reason[:55]}"
            )
            layer2_signal = None

    # ★ v3：時段過濾
    time_blocked       = False
    time_block_reason  = ""
    if layer2_signal is not None and use_time_filter and time_filter_mode != "關閉":
        # 10點 = 黃金時段，其他時段依模式處理
        if "嚴格" in time_filter_mode and et_hour_now not in [9, 10]:
            time_blocked      = True
            time_block_reason = f"時段過濾（嚴格）：{et_hour_now}:xx WR偏低，僅09-10點發訊"
            st.session_state.signal_history.append(
                f"{now.strftime('%H:%M')} 🕙 時段過濾：{et_hour_now}:xx 非優質時段"
            )
            layer2_signal = None
        elif "寬鬆" in time_filter_mode and et_hour_now not in [9, 10]:
            # 非優質時段：降低訊號強度 50%
            layer2_strength *= 0.5
            layer2_stars     = signal_stars(layer2_strength)
            time_block_reason = f"時段降級（{et_hour_now}:xx）訊號強度×0.5"
            layer2_reason    += f"  🕙 {time_block_reason}"

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
                f"{now.strftime('%H:%M')} ⏱ 滯後過濾：{lag_block_reason[:55]}"
            )
            layer2_signal = None

    # ★ v3：訊號品質評分（所有過濾後計算，加入 et_hour）
    if layer2_signal is not None and tsla_rsi1 is not None:
        signal_quality, signal_grade, signal_grade_color = calc_signal_quality(
            uvxy_r2        = uvxy_trend_1m.get("r2", 0),
            uvxy_slope_abs = abs(uvxy_trend_1m.get("slope", 0)),
            uvxy_consec    = uvxy_trend_1m.get("consecutive", 0),
            rsi            = tsla_rsi1,
            signal_type    = layer2_signal,
            lag_strength   = lag_strength_val,
            multi_tf       = multi_tf_agree,
            et_hour        = et_hour_now,
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
        tf_line  = "✅ 1m+5m 雙框架" if multi_tf_agree else "⚠️ 僅1m框架"
        tod_line = "🌟 黃金時段" if et_hour_now==10 else f"🕙 {et_hour_now}:xx ET"
        co_now   = uvxy_trend_1m.get('consecutive',0)
        co_emoji = "🔥" if co_now>=5 else ("⭐" if co_now>=4 else ("✅" if co_now>=3 else "⚠️"))
        msg = (f"*{action}*  {layer2_stars}  [品質{signal_grade}={signal_quality}分]\n"
               f"━━━━━━━━━━━━━━━━\n"
               f"時間：`{now.strftime('%Y-%m-%d %H:%M')}` {tod_line}\n"
               f"框架：{tf_line}\n"
               f"連續根數：`{co_now}`根 {co_emoji}（門檻≥{consec_req}）\n"
               f"R²：`{uvxy_trend_1m.get('r2',0):.2f}`（門檻≥{min_uvxy_r2}）\n"
               f"UVXY斜率：`{uvxy_trend_1m.get('slope',0):+.3f}%/根`\n"
               f"RSI：`{tsla_rsi1:.1f}`  lag：`{lag_strength_val:+.3f}`\n"
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

# ★ 被過濾訊號提示
if rsi_blocked:
    st.markdown(f"""
    <div style='border-radius:10px;padding:10px 20px;margin:6px 0;
                background:#0a0a1a;border:1px dashed #5c7cfa;text-align:center'>
      <div style='font-size:0.95rem;font-weight:700;color:#5c7cfa'>
          🛡️ RSI 雙向過濾已攔截一個訊號</div>
      <div style='font-size:0.80rem;color:#8b8fa8;margin-top:4px'>{rsi_block_reason}</div>
    </div>
    """, unsafe_allow_html=True)

if time_blocked:
    st.markdown(f"""
    <div style='border-radius:10px;padding:10px 20px;margin:6px 0;
                background:#0a1a0a;border:1px dashed #f6c90e;text-align:center'>
      <div style='font-size:0.95rem;font-weight:700;color:#f6c90e'>
          🕙 時段過濾已攔截一個訊號</div>
      <div style='font-size:0.80rem;color:#8b8fa8;margin-top:4px'>{time_block_reason}</div>
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

# ★ 時段標籤 helper
def _tod_badge(hr: int) -> str:
    if hr == 10:
        return "<span class='tod-hot'>🕙 10:xx 黃金時段 WR=72%</span>"
    elif hr in [9, 14]:
        return f"<span class='tod-warm'>🕙 {hr}:xx 普通時段</span>"
    else:
        return f"<span class='tod-cold'>🕙 {hr}:xx WR偏低</span>"

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
    tod_html = _tod_badge(et_hour_now)
    st.markdown(f"""
    <div class="sig-layer2-buy">
      <div class="sig-title" style="color:#00d97e">🟢 買入 TSLA &nbsp; {layer2_stars}
          <span style='font-size:1rem;color:{signal_grade_color}'>
              &nbsp;品質 {signal_grade} ({signal_quality}分)</span></div>
      <div style='margin:4px 0'>{tod_html}</div>
      <div class="sig-detail">{layer2_reason}</div>
      <div class="quality-bar">
          <div class="quality-fill" style="width:{quality_bar_w}%;background:{signal_grade_color}"></div>
      </div>
      <div style='font-size:0.75rem;color:#8b8fa8;margin-top:4px'>
          consec={uvxy_trend_1m.get('consecutive',0)} &nbsp;|&nbsp;
          R²={uvxy_trend_1m.get('r2',0):.3f} &nbsp;|&nbsp;
          lag={lag_strength_val:+.3f} &nbsp;|&nbsp;
          RSI={f'{tsla_rsi1:.1f}' if tsla_rsi1 else '—'} &nbsp;|&nbsp;
          UVXY加速={uvxy_accel:+.4f}
      </div>
    </div>
    """, unsafe_allow_html=True)
elif layer2_signal == "SELL_TSLA":
    quality_bar_w = signal_quality
    tod_html = _tod_badge(et_hour_now)
    st.markdown(f"""
    <div class="sig-layer2-sell">
      <div class="sig-title" style="color:#e84045">🔴 賣出 TSLA &nbsp; {layer2_stars}
          <span style='font-size:1rem;color:{signal_grade_color}'>
              &nbsp;品質 {signal_grade} ({signal_quality}分)</span></div>
      <div style='margin:4px 0'>{tod_html}</div>
      <div class="sig-detail">{layer2_reason}</div>
      <div class="quality-bar">
          <div class="quality-fill" style="width:{quality_bar_w}%;background:{signal_grade_color}"></div>
      </div>
      <div style='font-size:0.75rem;color:#8b8fa8;margin-top:4px'>
          consec={uvxy_trend_1m.get('consecutive',0)} &nbsp;|&nbsp;
          R²={uvxy_trend_1m.get('r2',0):.3f} &nbsp;|&nbsp;
          lag={lag_strength_val:+.3f} &nbsp;|&nbsp;
          RSI={f'{tsla_rsi1:.1f}' if tsla_rsi1 else '—'} &nbsp;|&nbsp;
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
if not layer1_active and not layer2_signal and not layer3_exit and not rsi_blocked and not lag_blocked and not time_blocked:
    uvxy_s   = uvxy_trend_1m.get("slope", 0)
    r2_val   = uvxy_trend_1m.get("r2", 0)
    co_val   = uvxy_trend_1m.get("consecutive", 0)
    r2_status    = f"✅ R²={r2_val:.2f}≥{min_uvxy_r2}" if r2_val >= min_uvxy_r2 else f"❌ R²={r2_val:.2f}<{min_uvxy_r2}"
    slope_status = (f"✅ 斜率={abs(uvxy_s):.3f}≥{min_uvxy_slope}"
                    if abs(uvxy_s) >= min_uvxy_slope else f"❌ 斜率={abs(uvxy_s):.3f}<{min_uvxy_slope}")
    consec_status = (f"✅ consec={co_val}≥{consec_req}"
                     if co_val >= consec_req else f"❌ consec={co_val}<{consec_req}")
    tod_color = "#00d97e" if et_hour_now == 10 else ("#f6c90e" if et_hour_now in [9,14] else "#8b8fa8")
    st.markdown(f"""
    <div class="sig-normal">
      <div class="sig-title" style="color:#8b8fa8">⚪ 無訊號</div>
      <div class="sig-detail">
          {slope_status} &nbsp;|&nbsp; {r2_status} &nbsp;|&nbsp;
          <span style='color:{"#00d97e" if co_val>=consec_req else "#e84045"}'>{consec_status}</span><br>
          確認={uvxy_trend_1m.get('confirmed','—')} &nbsp;|&nbsp;
          RSI={f'{tsla_rsi1:.1f}' if tsla_rsi1 else '—'} &nbsp;|&nbsp;
          lag={lag_strength_val:+.3f} &nbsp;|&nbsp;
          <span style='color:{tod_color}'>🕙 {et_hour_now}:xx ET</span>
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

# ── v3b changelog footer ──
st.markdown('<div class="section-title">版本歷史（5日1949根回測校準）</div>', unsafe_allow_html=True)
st.markdown("""
<div style='background:#161920;border-radius:8px;padding:12px 18px;
            border:1px solid #2d3139;font-size:0.80rem;color:#8b8fa8;line-height:1.8'>
<b style='color:#5c7cfa'>v2（1日390根）：</b>
斜率0.15→0.05 · R²0.45→0.65 · RSI單邊防呆 · 品質評分系統<br>
<b style='color:#f6c90e'>v3（5日1949根）：</b>
consec升為核心 · RSI雙邊門檻 · 時段過濾 · <span style='color:#e84045'>⚠️ R²錯誤升至0.80（高波動日全無訊號）</span><br>
<b style='color:#00d97e'>v3b修正（今日盤中截圖驗證）：</b><br>
1. <b style='color:#c9cdd8'>R² 恢復 0.65</b>：v3把R²升至0.80導致今日UVXY+6.75%大行情中0個訊號<br>
&nbsp;&nbsp;&nbsp;高波動日UVXY震盪劇烈，R²天生偏低（今日均值0.686），0.80過濾了所有機會<br>
2. <b style='color:#c9cdd8'>consec≥3 保留為核心過濾器</b>：5日回測最強因子，WR隨consec線性提升<br>
3. <b style='color:#c9cdd8'>SELL RSI雙向門檻 35-55</b>：被攔截訊號WR=0% PnL=-0.558%，攔截完全正確<br><br>
<b style='color:#00d97e'>修正v3b 5日最終成績：N=41  WR=68.3%  AvgPnL=+0.077%</b><br>
<span style='color:#8b8fa8'>對比：v1(無過濾) WR=50% | v2 WR=58% | v3錯誤 WR=63%/N=19 | v3b WR=68%/N=41</span>
</div>
""", unsafe_allow_html=True)

# ── Auto refresh ──
if auto_refresh:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
