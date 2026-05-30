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
# 2. 資料抓取函數 (含息與不含息)
# ==========================================
@st.cache_data(ttl=3600)
def load_etf_data(tickers, start_date, end_date):
    """
    抓取兩種價格：
    'Close': yfinance 預設的 history()，已經是含息還原價 (Total Return)。
    'Price_Only': 我們利用原始收盤價來模擬不含息的走勢 (Price Return)。
    """
    if not tickers: return pd.DataFrame(), pd.DataFrame()
    adj_dict = {}
    price_dict = {}
    
    for ticker in tickers:
        try:
            # history() auto_adjust=True 預設抓取含息還原價
            tk = yf.Ticker(ticker)
            hist = tk.history(start=start_date, end=end_date)
            
            # 抓取未還原的原始收盤價 (不含息走勢)
            raw_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not hist.empty and 'Close' in hist.columns:
                hist.index = hist.index.tz_localize(None)
                adj_dict[ticker] = hist['Close']
                
            if not raw_data.empty:
                # 處理 yfinance 新版多層次索引
                if isinstance(raw_data.columns, pd.MultiIndex):
                    raw_data.columns = raw_data.columns.get_level_values(0)
                if 'Close' in raw_data.columns:
                    raw_data.index = raw_data.index.tz_localize(None)
                    price_dict[ticker] = raw_data['Close']
                    
        except Exception:
            continue
            
    return pd.DataFrame(adj_dict).ffill().dropna(), pd.DataFrame(price_dict).ffill().dropna()

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

def calculate_metrics(adj_returns, price_returns, name, leverage_pct=0, div_series=None, price_series=None, borrow_rate=0.0):
    # 含息指標計算
    cagr_adj = ((1 + adj_returns).prod() ** (252 / len(adj_returns))) - 1
    volatility = adj_returns.std() * np.sqrt(252)
    sharpe_ratio = (cagr_adj - 0.015) / volatility if volatility != 0 else 0
    mdd = calculate_mdd((1 + adj_returns).cumprod())
    
    # 不含息指標計算 (純價差)
    cagr_price = ((1 + price_returns).prod() ** (252 / len(price_returns))) - 1
    
    cv_str = "N/A"
    yield_val = "N/A"
    
    # 計算 CV 與 年化配息率
    if div_series is not None and len(div_series[div_series > 0]) > 0:
        valid_divs = div_series[div_series > 0]
        if len(valid_divs) >= 2:
            cv = valid_divs.std() / valid_divs.mean()
            cv_str = f"{cv:.2f}"
            
        if price_series is not None:
            yearly_yields = []
            for year, div in valid_divs.items():
                p_year = price_series[price_series.index.year == year]
                if not p_year.empty:
                    yearly_yields.append(div / p_year.mean())
            if yearly_yields:
                base_yield = np.mean(yearly_yields)
                lev_ratio = leverage_pct / 100.0
                final_yield = base_yield * (1 + lev_ratio) - (borrow_rate * lev_ratio)
                yield_val = float(final_yield * 100)
    
    # 將所有數值格式化為字串，確保在 Streamlit 表格中能穩定顯示與置中
    return {
        "標的名稱": name,
        "質押": f"{leverage_pct}%",
        "含息年化 (%)": f"{cagr_adj * 100:.2f}",
        "不含息年化 (%)": f"{cagr_price * 100:.2f}",
        "年化配息率 (%)": f"{yield_val:.2f}" if yield_val != "N/A" else yield_val,
        "年化波動率 (%)": f"{volatility * 100:.2f}",
        "最大回撤 (%)": f"{mdd * 100:.2f}",
        "夏普值": f"{sharpe_ratio:.2f}",
        "配息 CV": cv_str
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
    selected_names = st.multiselect("選擇組成 ETF", list(ETF_DICT.keys()), default=["00713 元大台灣高息低波", "00878 國泰永續高股息"])
    
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
            df_adj, df_price = load_etf_data(selected_tickers, start_date, end_date)
            df_div = load_dividend_data(selected_tickers)
            
        if not df_adj.empty and not df_price.empty:
            # --- 核心報酬率計算 ---
            adj_returns = df_adj.pct_change().dropna()
            price_returns = df_price.pct_change().dropna()
            
            # 原型組合的日報酬 (含息與不含息)
            port_adj_returns = (adj_returns * weights).sum(axis=1)
            port_price_returns = (price_returns * weights).sum(axis=1)
            
            # 槓桿組合的日報酬 (含息與不含息，皆需扣除利息)
            daily_borrow_rate = borrow_rate / 252
            lev_adj_returns = port_adj_returns * (1 + leverage_ratio) - (daily_borrow_rate * leverage_ratio)
            lev_price_returns = port_price_returns * (1 + leverage_ratio) - (daily_borrow_rate * leverage_ratio)
            
            # --- 配息資料處理 ---
            port_div_series = None
            port_price_series_for_yield = None
            if not df_div.empty:
                valid_div_cols = [c for c in df_div.columns if c in selected_tickers]
                if valid_div_cols:
                    port_div_series = (df_div[valid_div_cols] * [weights[selected_tickers.index(c)] for c in valid_div_cols]).sum(axis=1)
                    port_price_series_for_yield = (df_adj[valid_div_cols] * [weights[selected_tickers.index(c)] for c in valid_div_cols]).sum(axis=1)
            
            # --- 儲存按鈕區塊 ---
            st.subheader("💾 命名與保存當前策略")
            col_name, col_btn, col_clear = st.columns([2, 1, 1])
            with col_name:
                custom_name = st.text_input("✏️ 為此投資組合命名", value="我的現金流組合", label_visibility="collapsed")
            with col_btn:
                if st.button("➕ 記錄至總表", type="primary", use_container_width=True):
                    p_metrics = calculate_metrics(port_adj_returns, port_price_returns, f"🎯 {custom_name} (原型)", 0, port_div_series, port_price_series_for_yield, borrow_rate)
                    st.session_state.saved_portfolios.append(p_metrics)
                    
                    if leverage_ratio > 0:
                        l_metrics = calculate_metrics(lev_adj_returns, lev_price_returns, f"🔥 {custom_name} (質押)", leverage_pct, port_div_series, port_price_series_for_yield, borrow_rate)
                        st.session_state.saved_portfolios.append(l_metrics)
            with col_clear:
                if st.button("🗑️ 清空紀錄", use_container_width=True):
                    st.session_state.saved_portfolios = []
                    
            st.divider()

            # --- 績效比較總表 ---
            st.subheader("📋 策略總表：單一 ETF vs 歷史紀錄")
            metrics_list = []
            
            for col in adj_returns.columns:
                etf_name = [k for k, v in ETF_DICT.items() if v == col][0]
                etf_div_series = df_div[col] if col in df_div.columns else None
                etf_price_series = df_adj[col]
                # 傳入含息與不含息的報酬率計算單檔 ETF
                metrics_list.append(calculate_metrics(adj_returns[col], price_returns[col], etf_name, 0, etf_div_series, etf_price_series, borrow_rate))
            
            metrics_list.extend(st.session_state.saved_portfolios)
            
            curr_p = calculate_metrics(port_adj_returns, port_price_returns, f"👁️ 預覽: {custom_name} (原型)", 0, port_div_series, port_price_series_for_yield, borrow_rate)
            metrics_list.append(curr_p)
            if leverage_ratio > 0:
                curr_l = calculate_metrics(lev_adj_returns, lev_price_returns, f"👁️ 預覽: {custom_name} (質押)", leverage_pct, port_div_series, port_price_series_for_yield, borrow_rate)
                metrics_list.append(curr_l)
            
            comparison_df = pd.DataFrame(metrics_list).set_index("標的名稱")

            # 強制 HTML 表格渲染，確保完美置中且樣式不會被 Streamlit 覆蓋
            def render_html_table(df):
                html = "<table style='width:100%; text-align:center; border-collapse: collapse; font-family: sans-serif;'>"
                # 表頭
                html += "<tr style='background-color: #1E1E1E; border-bottom: 2px solid #444;'>"
                html += f"<th style='padding: 12px; text-align:left;'>標的名稱</th>"
                for col in df.columns:
                    html += f"<th style='padding: 12px; text-align:center;'>{col}</th>"
                html += "</tr>"
                # 資料列
                for index, row in df.iterrows():
                    bg_color = "transparent"
                    font_weight = "normal"
                    color = "#E0E0E0"
                    
                    if str(index).startswith("👁️ 預覽"):
                        bg_color = "#117A65"
                        font_weight = "bold"
                        color = "white"
                    elif "🔥" in str(index) or "🎯" in str(index):
                        bg_color = "#2C3E50"
                        color = "#D5D8DC"
                    
                    html += f"<tr style='background-color: {bg_color}; border-bottom: 1px solid #333;'>"
                    html += f"<td style='padding: 10px; text-align:left; color:{color}; font-weight:{font_weight};'>{index}</td>"
                    for item in row:
                        html += f"<td style='padding: 10px; text-align:center; color:{color}; font-weight:{font_weight};'>{item}</td>"
                    html += "</tr>"
                html += "</table>"
                return html

            st.markdown(render_html_table(comparison_df), unsafe_allow_html=True)
            st.caption("💡 **不含息年化**代表將配息全數提領不投入的本金成長率。**年化配息率**已扣除質押利息成本。")
            st.divider()

            # --- 資金曲線雙圖表 ---
            st.subheader("📈 資金曲線預覽：含息 (股息再投入) vs 不含息 (提領配息)")
            
            tab1, tab2 = st.tabs(["💰 不含息軌跡 (真實提領生活費)", "📈 含息軌跡 (股息再投入)"])
            
            # 不含息圖表 (純價格走勢)
            with tab1:
                cum_port_price = (1 + port_price_returns).cumprod() * 100
                cum_lev_price = (1 + lev_price_returns).cumprod() * 100
                
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(x=cum_port_price.index, y=cum_port_price, mode='lines', name='🎯 當前組合 (未槓桿)', line=dict(color='#2E86C1', width=2)))
                if leverage_ratio > 0:
                    fig_price.add_trace(go.Scatter(x=cum_lev_price.index, y=cum_lev_price, mode='lines', name=f'🔥 當前組合 (質押 {leverage_pct}%)', line=dict(color='#E74C3C', width=2)))
                
                fig_price.update_layout(hovermode="x unified", yaxis_title="累積報酬 (基期=100)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_price, use_container_width=True)
                st.info("👆 **這張圖是你每年拿走配息去繳生活費與利息後，帳戶內【本金】的真實變化。** 如果曲線長期向上，代表策略成功抵抗了通膨與本金耗損。")
                
            # 含息圖表 (總報酬)
            with tab2:
                cum_port_adj = (1 + port_adj_returns).cumprod() * 100
                cum_lev_adj = (1 + lev_adj_returns).cumprod() * 100
                
                fig_adj = go.Figure()
                fig_adj.add_trace(go.Scatter(x=cum_port_adj.index, y=cum_port_adj, mode='lines', name='🎯 當前組合 (未槓桿)', line=dict(color='#2E86C1', width=2)))
                if leverage_ratio > 0:
                    fig_adj.add_trace(go.Scatter(x=cum_lev_adj.index, y=cum_lev_adj, mode='lines', name=f'🔥 當前組合 (質押 {leverage_pct}%)', line=dict(color='#E74C3C', width=2)))
                
                fig_adj.update_layout(hovermode="x unified", yaxis_title="累積報酬 (基期=100)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_adj, use_container_width=True)
                
        else:
            st.warning("資料獲取失敗，請確認時間區間內是否有足夠報價。")
else:
    st.info("請從左側欄位選擇至少一檔 ETF 來建立投資組合。")
