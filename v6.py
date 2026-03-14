import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title="UVXY Lead → TSLA Trade Alert v2", layout="wide", page_icon="⚡")

st.title("⚡ UVXY Leading TSLA Trade Alert Dashboard (升級版)")
st.markdown("**UVXY作為領先指標 • 偵測不同步 • 明確買/賣TSLA建議 • 已加入最小實體過濾與大K線優先**")

# ── 側邊欄控制 ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")
    n_candles = st.slider("回顧根數 (N)", 3, 15, 7, help="用來判斷趨勢的K線數量")
    trend_threshold = st.slider("趨勢強度門檻 (%)", 60, 90, 70, help="至少多少% K線需同方向才算強趨勢")
    min_body_pct = st.slider("最小實體過濾 (%)", 0.05, 0.5, 0.15, help="實體小於此值不計入趨勢計算")
    big_candle_pct = st.slider("大K線強制門檻 (%)", 0.3, 1.5, 0.5, help="單根變動超過此值視為強趨勢")
    cooldown_min = st.slider("Telegram冷卻時間 (分鐘)", 1, 15, 5)
    auto_refresh = st.checkbox("每60秒自動更新", value=True)
    if st.button("🔄 手動更新"):
        st.rerun()

# ── Telegram 設定 ──────────────────────────────────────────────────────────
try:
    TELEGRAM_TOKEN = st.secrets["telegram"]["bot_token"]
    CHAT_ID = st.secrets["telegram"]["chat_id"]
except:
    st.error("請在 .streamlit/secrets.toml 設定 telegram.bot_token 與 chat_id")
    st.stop()

# 冷卻狀態
if "last_alert" not in st.session_state:
    st.session_state.last_alert = datetime.now(pytz.utc) - timedelta(minutes=10)

# ── 抓取資料 ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=45)
def fetch_data():
    tz = pytz.timezone("America/New_York")
    tsla = yf.download("TSLA", interval="1m", period="5d", progress=False)
    uvxy = yf.download("UVXY", interval="1m", period="5d", progress=False)
    if not tsla.empty:
        tsla.index = tsla.index.tz_convert(tz)
    if not uvxy.empty:
        uvxy.index = uvxy.index.tz_convert(tz)
    return tsla.tail(150), uvxy.tail(150)

tsla_df, uvxy_df = fetch_data()

if tsla_df.empty or uvxy_df.empty:
    st.error("無法取得最新資料，請確認網路或美股交易時段")
    st.stop()

tsla_latest = tsla_df.iloc[-1]
uvxy_latest = uvxy_df.iloc[-1]

# ── 趨勢強度計算（含最小實體過濾 + 大K線優先） ─────────────────────────────
def get_trend_strength(df, n, min_body_pct, big_candle_pct):
    recent = df.tail(n).copy()
    recent['Body'] = abs(recent['Close'] - recent['Open'])
    recent['Pct'] = (recent['Close'] - recent['Open']) / recent['Open'] * 100
    recent['Direction'] = np.sign(recent['Pct'])

    # 過濾小實體
    valid = recent[recent['Body'] / recent['Open'] * 100 >= min_body_pct]

    if len(valid) == 0:
        return "無有效K線", 0, False

    up_count = (valid['Direction'] > 0).sum()
    down_count = (valid['Direction'] < 0).sum()
    total_valid = len(valid)

    up_pct = up_count / total_valid * 100 if total_valid > 0 else 0
    down_pct = down_count / total_valid * 100 if total_valid > 0 else 0

    # 大K線檢查（單根超過門檻視為強趨勢）
    has_big_up = (recent['Pct'] >= big_candle_pct).any()
    has_big_down = (recent['Pct'] <= -big_candle_pct).any()

    if has_big_up:
        return "UP (大陽線優先)", max(up_pct, 80), True
    if has_big_down:
        return "DOWN (大陰線優先)", max(down_pct, 80), True

    if up_pct >= trend_threshold:
        return "UP", up_pct, False
    if down_pct >= trend_threshold:
        return "DOWN", down_pct, False

    return "無明顯趨勢", max(up_pct, down_pct), False

uvxy_trend, uvxy_strength, uvxy_forced_by_big = get_trend_strength(
    uvxy_df, n_candles, min_body_pct, big_candle_pct
)

# TSLA 是否跟隨
tsla_recent = tsla_df.tail(n_candles)
tsla_net_pct = (tsla_recent['Close'].iloc[-1] - tsla_recent['Close'].iloc[0]) / tsla_recent['Close'].iloc[0] * 100

tsla_follow_expected = False
if uvxy_trend == "UP":
    tsla_follow_expected = tsla_net_pct <= -0.12  # 至少小跌
elif uvxy_trend == "DOWN":
    tsla_follow_expected = tsla_net_pct >= 0.12   # 至少小漲

desync = (uvxy_trend in ["UP", "DOWN"]) and not tsla_follow_expected

# ── Pearson 相關係數（輔助確認） ───────────────────────────────────────────
common_idx = tsla_df.index.intersection(uvxy_df.index)
corr = np.nan
corr_hist = []
if len(common_idx) >= 60:
    ret_tsla = tsla_df.loc[common_idx, 'Close'].pct_change().dropna()
    ret_uvxy = uvxy_df.loc[common_idx, 'Close'].pct_change().dropna()
    if len(ret_tsla) == len(ret_uvxy):
        corr_series = ret_tsla.rolling(60).corr(ret_uvxy)
        corr = corr_series.iloc[-1]
        corr_hist = corr_series.tail(60).dropna().values

# ── Telegram 警報 ───────────────────────────────────────────────────────────
now = datetime.now(pytz.utc)
if desync and (now - st.session_state.last_alert).total_seconds() > cooldown_min * 60:
    direction = "立即買入 TSLA" if uvxy_trend == "UP" else "立即賣出/放空 TSLA"
    stop_ref = ""
    if len(tsla_recent) >= 2:
        prev_low = tsla_recent['Low'].iloc[-2]
        prev_high = tsla_recent['High'].iloc[-2]
        stop_ref = f"建議停損參考：{'前低 ' + f'{prev_low:.2f}' if uvxy_trend == 'UP' else '前高 ' + f'{prev_high:.2f}'}"

    msg = f"""🚨 UVXY → TSLA 不同步警報！
UVXY：{uvxy_trend} ({uvxy_strength:.1f}%) {'[大K線強制]' if uvxy_forced_by_big else ''}
TSLA 近{n_candles}分鐘變化：{tsla_net_pct:+.2f}%
→ {direction} 機會！
目前價格：  TSLA ${tsla_latest['Close']:.2f}   UVXY ${uvxy_latest['Close']:.2f}
Pearson 60分滾動：{corr:.3f if not np.isnan(corr) else 'N/A'}
{stop_ref}
時間：{now.astimezone(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S ET')}"""

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
        st.session_state.last_alert = now
        st.success("已發送 Telegram 警報")
    except Exception as e:
        st.warning(f"Telegram 發送失敗：{e}")

# ── 即時看板 ───────────────────────────────────────────────────────────────
cols = st.columns(5)
cols[0].metric("TSLA 最新價", f"${tsla_latest['Close']:.2f}", f"{tsla_latest['Close']-tsla_df['Close'].iloc[-2]:+.2f}")
cols[1].metric("UVXY 最新價", f"${uvxy_latest['Close']:.2f}", f"{uvxy_latest['Close']-uvxy_df['Close'].iloc[-2]:+.2f}")
cols[2].metric("Pearson (60分)", f"{corr:.3f}" if not np.isnan(corr) else "N/A")
cols[3].metric("UVXY 趨勢", uvxy_trend, f"{uvxy_strength:.1f}%")
cols[4].metric("最後更新", datetime.now().strftime("%H:%M:%S"))

if desync:
    st.error(f"⚡ 偵測到不同步！UVXY {uvxy_trend} 但 TSLA 未跟隨 → 建議 {direction}")

# ── 雙K線圖 ────────────────────────────────────────────────────────────────
last15_tsla = tsla_df.tail(15)
last15_uvxy = uvxy_df.tail(15)

fig = make_subplots(rows=1, cols=2, subplot_titles=("TSLA 最近15分鐘", "UVXY 最近15分鐘"))
fig.add_trace(go.Candlestick(
    x=last15_tsla.index, open=last15_tsla.Open, high=last15_tsla.High,
    low=last15_tsla.Low, close=last15_tsla.Close, name="TSLA",
    increasing_line_color="#00cc96", decreasing_line_color="#ff4d4d"), row=1, col=1)
fig.add_trace(go.Candlestick(
    x=last15_uvxy.index, open=last15_uvxy.Open, high=last15_uvxy.High,
    low=last15_uvxy.Low, close=last15_uvxy.Close, name="UVXY",
    increasing_line_color="#ffaa00", decreasing_line_color="#ff4d4d"), row=1, col=2)

fig.update_layout(height=520, template="plotly_dark", showlegend=False, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# ── Pearson 走勢圖 ──────────────────────────────────────────────────────────
if len(corr_hist) > 5:
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(y=corr_hist, mode="lines+markers", name="60分滾動Pearson", line=dict(color="#00aaff")))
    fig_corr.add_hline(y=-0.5, line_dash="dash", line_color="red", annotation_text="-0.5 警戒線")
    fig_corr.update_layout(height=340, template="plotly_dark", title="Pearson 相關係數走勢 (最近60分鐘滾動)", yaxis_title="相關係數")
    st.plotly_chart(fig_corr, use_container_width=True)

# ── 自動更新 ───────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(60)
    st.rerun()

st.caption("資料來源：yfinance • 僅限美股交易時段 • 僅供參考，非投資建議")
