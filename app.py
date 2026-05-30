import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. 網頁基本設定 & 暫存狀態初始化
# ==========================================
st.set_page_config(page_title="高股息策略與質押模擬器", layout="wide")
st.title("🛡️ 台灣高股息 ETF 現金流組合與質押模擬器")

# 初始化暫存記憶體，用來保存使用者紀錄的投資組合
if 'saved_portfolios' not in st.session_state:
    st.session_state.saved_portfolios = []

ETF_DICT = {
    "0056 元大高股息": "0056.TW",
    "00878 國泰永續高股息": "00878.TW",
    "00919 群益台灣精選高息": "00919.TW",
    "00929 復華台灣科技優息": "00929.TW",
    "00713 元大台灣高息低波": "00713.TW"
}

# ==========================================
# 2. 資料抓取函數
# ==========================================
@st.cache_data(ttl=3600)
def load_etf_data(tickers, start_date, end_date):
    if not tickers: return pd.DataFrame()
    df_dict = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(start=start_date, end=end_date)
            if not hist.empty and 'Close' in hist.columns:
                hist.index = hist.index.tz_localize(None)
                df_dict[ticker] = hist['Close']
        except Exception:
            continue
    return pd.DataFrame(df_dict).ffill().dropna()

@st.cache_data(ttl=3600)
def load_dividend_data(tickers):
    if not tickers: return pd.DataFrame()
    div_dict = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            divs = tk.dividends
            if not divs.empty:
                divs.index = divs.index.tz_localize(None)
                annual_divs = divs.groupby(divs.index.year).sum()
                div_dict[ticker] = annual_divs
        except Exception:
            continue
    return pd.DataFrame(div_dict).fillna(0)

# ==========================================
# 3. 數學運算輔助函數
# ==========================================
def calculate_mdd(cum_returns):
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def calculate_metrics(returns_series, name, div_series=None):
    cagr = ((1 + returns_series).prod() ** (252 / len(returns_series))) - 1
    volatility = returns_series.std() * np.sqrt(252)
    sharpe_ratio = (cagr - 0.015) / volatility if volatility != 0 else 0
    cum_returns = (1 + returns_series).cumprod()
    mdd = calculate_mdd(cum_returns)
    
    cv_str = "N/A"
    if div_series is not None and len(div_series[div_series > 0]) >= 2:
        valid_divs = div_series[div_series > 0]
        cv = valid_divs.std() / valid_divs.mean()
        cv_str = f"{cv:.2f}"
    
    return {
        "標的名稱": name,
        "年化報酬率 (%)": float(cagr * 100),
        "年化波動率 (%)": float(volatility * 100),
        "夏普值": float(sharpe_ratio),
        "最大回撤 MDD (%)": float(mdd * 100),
        "配息變異係數 (CV)": cv_str
    }

# ==========================================
# 4. 側邊欄：參數設定
# ==========================================
with st.sidebar:
    st.header("🕒 1. 回測時間區間")
    default_start = datetime.today() - timedelta(days=5*365)
    start_date = st.date_input("開始日期", value=default_start)
    end_date = st.date_input("結束日期", value=datetime.today())
    st.divider()
    
    st.header("⚖️ 2. 資產與權重")
    selected_names = st.multiselect(
        "選擇組成 ETF", 
        list(ETF_DICT.keys()), 
        default=["00713 元大台灣高息低波", "00878 國泰永續高股息"]
    )
    
    weights = []
    if selected_names:
        default_w = 100 // len(selected_names)
        for i, name in enumerate(selected_names):
            val = default_w + (100 % len(selected_names) if i == len(selected_names)-1 else 0)
            w = st.number_input(f"{name[:5]} 權重 (%)", min_value=0, max_value=100, value=val)
            weights.append(w / 100)
            
    st.divider()
    st.header("🔥 3. 質押槓桿設定")
    leverage_pct = st.slider("質押借款比例 (%)", 0, 100, 20)
    leverage_ratio = leverage_pct / 100.0
    borrow_rate = st.number_input("借款年利率 (%)", value=2.5, step=0.1) / 100.0

# ==========================================
# 5. 主畫面運算與渲染
# ==========================================
if selected_names:
    if sum(weights) != 1.0:
        st.error(f"⚠️ 目前權重總和為 {sum(weights)*100:.0f}%，請調整至 100%。")
    else:
        selected_tickers = [ETF_DICT[name] for name in selected_names]
        with st.spinner("載入報價與配息模型中..."):
            df_price = load_etf_data(selected_tickers, start_date, end_date)
            df_div = load_dividend_data(selected_tickers)
            
        if not df_price.empty:
            # --- 核心運算 ---
            daily_returns = df_price.pct_change().dropna()
            port_daily_returns = (daily_returns * weights).sum(axis=1)
            
            daily_borrow_rate = borrow_rate / 252
            lev_daily_returns = port_daily_returns * (1 + leverage_ratio) - (daily_borrow_rate * leverage_ratio)
            
            # --- 當前組合指標計算 ---
            port_div_series = None
            if not df_div.empty:
                valid_div_cols = [c for c in df_div.columns if c in selected_tickers]
                if valid_div_cols:
                    port_div_series = (df_div[valid_div_cols] * [weights[selected_tickers.index(c)] for c in valid_div_cols]).sum(axis=1)
            
            # 建立當前操作面板的原型與槓桿指標
            current_port_metrics = calculate_metrics(port_daily_returns, "🎯 當前組合 (未槓桿)", port_div_series)
            current_lev_metrics = calculate_metrics(lev_daily_returns, f"🔥 當前組合 (質押 {leverage_pct}%)", None) # 槓桿組合 CV 計算較複雜暫略

            # --- 儲存按鈕區塊 ---
            st.subheader("💾 保存當前策略")
            col_save1, col_save2 = st.columns([1, 4])
            with col_save1:
                # 按下按鈕會將當前兩筆資料塞進 session_state
                if st.button("➕ 記錄此組合至總表", type="primary"):
                    st.session_state.saved_portfolios.append(current_port_metrics)
                    if leverage_ratio > 0:
                        st.session_state.saved_portfolios.append(current_lev_metrics)
            with col_save2:
                if st.button("🗑️ 清空歷史紀錄"):
                    st.session_state.saved_portfolios = []
                    
            st.divider()

            # --- 績效比較總表建構 ---
            st.subheader("📋 策略總表：單一 ETF vs 歷史紀錄 vs 當前組合")
            metrics_list = []
            
            # 1. 加入單一 ETF 績效
            for col in daily_returns.columns:
                etf_name = [k for k, v in ETF_DICT.items() if v == col][0]
                etf_div_series = df_div[col] if col in df_div.columns else None
                metrics_list.append(calculate_metrics(daily_returns[col].dropna(), etf_name, etf_div_series))
            
            # 2. 加入歷史儲存的組合
            metrics_list.extend(st.session_state.saved_portfolios)
            
            # 3. 加入目前畫面上正在調整的組合
            metrics_list.append(current_port_metrics)
            if leverage_ratio > 0:
                metrics_list.append(current_lev_metrics)
            
            comparison_df = pd.DataFrame(metrics_list).set_index("標的名稱")
            
            # 定義表格小數點格式與高亮邏輯
            # 控制數值欄位只顯示到小數點第二位
            format_dict = {
                "年化報酬率 (%)": "{:.2f}",
                "年化波動率 (%)": "{:.2f}",
                "夏普值": "{:.2f}",
                "最大回撤 MDD (%)": "{:.2f}"
            }
            
            # 高亮目前正在操作的組合
            def highlight_current(row):
                if row.name.startswith("🎯 當前組合"):
                    return ['background: #117A65; color: white; font-weight: bold'] * len(row)
                elif row.name.startswith("🔥 當前組合"):
                    return ['background: #922B21; color: white; font-weight: bold'] * len(row)
                elif "🔥" in row.name or "🎯" in row.name:
                    return ['background: #2C3E50; color: #D5D8DC; font-style: italic'] * len(row) # 歷史紀錄
                return [''] * len(row)

            # 渲染表格
            st.dataframe(
                comparison_df.style.apply(highlight_current, axis=1).format(format_dict, na_rep="N/A"),
                use_container_width=True
            )
            
            # --- 繪製績效對比圖 ---
            st.subheader("📈 資金曲線：原型組合 vs 質押策略")
            cum_port = (1 + port_daily_returns).cumprod() * 100
            cum_lev = (1 + lev_daily_returns).cumprod() * 100
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=cum_port.index, y=cum_port, mode='lines', name='🎯 當前組合 (未槓桿)', line=dict(color='#2E86C1', width=2)))
            if leverage_ratio > 0:
                fig_perf.add_trace(go.Scatter(x=cum_lev.index, y=cum_lev, mode='lines', name=f'🔥 當前組合 (質押 {leverage_pct}%)', line=dict(color='#E74C3C', width=2)))
            
            fig_perf.update_layout(hovermode="x unified", yaxis_title="累積報酬 (基期=100)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_perf, use_container_width=True)
            
        else:
            st.warning("資料獲取失敗，請確認時間區間內是否有足夠報價。")
else:
    st.info("請從左側欄位選擇至少一檔 ETF 來建立投資組合。")
