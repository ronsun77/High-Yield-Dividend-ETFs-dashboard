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

# 初始化 session_state
if 'saved_portfolios' not in st.session_state:
    st.session_state.saved_portfolios = []
if 'portfolio_counter' not in st.session_state:
    st.session_state.portfolio_counter = 0

ETF_DICT = {
    "0056 元大高股息": "0056.TW",
    "00878 國泰永續高股息": "00878.TW",
    "00919 群益台灣精選高息": "00919.TW",
    "00929 復華台灣科技優息": "00929.TW",
    "00713 元大台灣高息低波": "00713.TW"
}

# ==========================================
# 2. 資料抓取與輔助函數
# ==========================================
@st.cache_data(ttl=3600)
def load_etf_data(tickers, start_date, end_date):
    """抓取含息(Close)與不含息(Price)報價"""
    if not tickers: return pd.DataFrame(), pd.DataFrame()
    adj_dict, price_dict = {}, {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(start=start_date, end=end_date)
            raw_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not hist.empty and 'Close' in hist.columns:
                hist.index = hist.index.tz_localize(None)
                adj_dict[ticker] = hist['Close']
            if not raw_data.empty:
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
    """抓取配息紀錄"""
    if not tickers: return pd.DataFrame(), pd.Series()
    div_dict = {}
    div_raw_dict = {} # 保存原始除息日與金額用於填息計算
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            divs = tk.dividends
            if not divs.empty:
                divs.index = divs.index.tz_localize(None)
                div_raw_dict[ticker] = divs
                annual_divs = divs.groupby(divs.index.year).sum()
                div_dict[ticker] = annual_divs
        except Exception:
            continue
    return pd.DataFrame(div_dict).fillna(0), div_raw_dict

def calculate_mdd(cum_returns):
    running_max = cum_returns.cummax()
    return ((cum_returns - running_max) / running_max).min()

def calculate_fill_rate(div_series, price_df, ticker):
    """計算填息成功率與平均天數"""
    if div_series is None or ticker not in div_series or ticker not in price_df.columns:
        return "N/A", "N/A"
    
    divs = div_series[ticker]
    prices = price_df[ticker]
    success_count = 0
    total_days = 0
    valid_divs = 0
    
    for ex_date, div_amount in divs.items():
        # 尋找除息日前一天的收盤價
        pre_ex_dates = prices.index[prices.index < ex_date]
        if pre_ex_dates.empty: continue
        pre_ex_price = prices.loc[pre_ex_dates[-1]]
        
        # 尋找除息日之後的價格走勢
        post_ex_prices = prices[prices.index >= ex_date]
        if post_ex_prices.empty: continue
        
        # 判斷是否填息 (價格 >= 除息前一天價格)
        filled_dates = post_ex_prices[post_ex_prices >= pre_ex_price].index
        if not filled_dates.empty:
            fill_date = filled_dates[0]
            days_to_fill = (fill_date - ex_date).days
            success_count += 1
            total_days += days_to_fill
            valid_divs += 1
            
    if valid_divs == 0: return "N/A", "N/A"
    success_rate = (success_count / len(divs)) * 100
    avg_days = total_days / success_count if success_count > 0 else 0
    return f"{success_rate:.0f}%", f"{avg_days:.0f}"

# ==========================================
# 3. 核心指標與資產軌跡運算
# ==========================================
def calculate_metrics_and_trajectory(adj_returns, price_returns, name, leverage_pct, div_series, price_series, borrow_rate, initial_capital, annual_expense, div_raw, is_preview=False):
    """計算指標並模擬真實提領軌跡"""
    lev_ratio = leverage_pct / 100.0
    daily_borrow_rate = borrow_rate / 252
    
    # 組合日報酬 (不含息，純價差)
    lev_price_returns = price_returns * (1 + lev_ratio) - (daily_borrow_rate * lev_ratio)
    
    # --- 模擬真實資產軌跡 ---
    # 起始總資產 = 本金 + 借款
    total_assets = initial_capital * (1 + lev_ratio)
    debt = initial_capital * lev_ratio
    net_assets = initial_capital
    
    trajectory = []
    current_year = -1
    yearly_div_pool = 0
    
    # 假設這是一個合成的 ETF 組合，我們用加權平均的配息率來模擬每日/每年的配息發放
    # 為了簡化，我們在每年年底進行一次生活費結算與再投入
    
    for date, ret in lev_price_returns.items():
        year = date.year
        if year != current_year:
            # 跨年結算 (提領生活費或再投入)
            if current_year != -1:
                # 取得該年配息總額 (若有資料)
                if div_series is not None and current_year in div_series.index:
                    yield_amount = div_series.loc[current_year]
                    p_year = price_series[price_series.index.year == current_year]
                    if not p_year.empty:
                        # 估算該年實際配息金額
                        actual_div_yield = yield_amount / p_year.mean()
                        cash_received = net_assets * actual_div_yield * (1 + lev_ratio)
                        
                        # 扣除利息
                        interest_paid = debt * borrow_rate
                        net_cash = cash_received - interest_paid
                        
                        # 提領生活費
                        if net_cash >= annual_expense:
                            # 配息有剩，再投入本金
                            reinvest_amount = net_cash - annual_expense
                            total_assets += reinvest_amount
                            net_assets += reinvest_amount
                        else:
                            # 配息不夠，變賣資產補足
                            shortfall = annual_expense - net_cash
                            total_assets -= shortfall
                            net_assets -= shortfall
                            
            current_year = year
            
        # 每日資產隨市場波動
        total_assets = total_assets * (1 + ret)
        net_assets = total_assets - debt
        trajectory.append({'Date': date, 'Net_Assets': net_assets})
        
    traj_df = pd.DataFrame(trajectory).set_index('Date')
    
    # --- 計算其他指標 ---
    cagr_adj = ((1 + adj_returns).prod() ** (252 / len(adj_returns))) - 1
    cagr_price = ((1 + price_returns).prod() ** (252 / len(price_returns))) - 1
    volatility = adj_returns.std() * np.sqrt(252)
    sharpe_ratio = (cagr_adj - 0.015) / volatility if volatility != 0 else 0
    mdd = calculate_mdd((1 + adj_returns).cumprod())
    
    cv_str = "N/A"
    yield_val = "N/A"
    fill_rate_str = "N/A"
    fill_days_str = "N/A"
    
    if div_series is not None and len(div_series[div_series > 0]) > 0:
        valid_divs = div_series[div_series > 0]
        if len(valid_divs) >= 2:
            cv = valid_divs.std() / valid_divs.mean()
            cv_str = f"{cv:.2f}"
            
        if price_series is not None:
            yearly_yields = []
            for y, d in valid_divs.items():
                p_y = price_series[price_series.index.year == y]
                if not p_y.empty:
                    yearly_yields.append(d / p_y.mean())
            if yearly_yields:
                base_yield = np.mean(yearly_yields)
                final_yield = base_yield * (1 + lev_ratio) - (borrow_rate * lev_ratio)
                yield_val = f"{final_yield * 100:.2f}"
                
    # 若是單一 ETF，計算填息資料
    if name in ETF_DICT.keys():
        ticker = ETF_DICT[name]
        fill_rate_str, fill_days_str = calculate_fill_rate(div_raw, price_series, ticker)
                
    metrics = {
        "id": f"id_{datetime.now().timestamp()}" if not is_preview else "preview",
        "標的名稱": name,
        "質押": f"{leverage_pct}%",
        "期末淨資產 (萬)": f"{net_assets / 10000:.0f}",
        "期末總資產 (萬)": f"{total_assets / 10000:.0f}",
        "含息年化 (%)": f"{cagr_adj * 100:.2f}",
        "不含息年化 (%)": f"{cagr_price * 100:.2f}",
        "年化配息率 (%)": yield_val,
        "填息成功率": fill_rate_str,
        "平均填息天數": fill_days_str,
        "年化波動率 (%)": f"{volatility * 100:.2f}",
        "最大回撤 (%)": f"{mdd * 100:.2f}",
        "夏普值": f"{sharpe_ratio:.2f}",
        "配息 CV": cv_str
    }
    
    return metrics, traj_df

# ==========================================
# 4. 側邊欄：參數設定
# ==========================================
with st.sidebar:
    st.header("💰 1. 資金與提領設定")
    initial_capital = st.number_input("初始本金 (元)", value=1000000, step=100000)
    annual_expense = st.number_input("每年生活費需求 (元)", value=60000, step=10000)
    st.divider()
    
    st.header("🕒 2. 回測時間區間")
    default_start = datetime.today() - timedelta(days=5*365)
    start_date = st.date_input("開始日期", value=default_start)
    end_date = st.date_input("結束日期", value=datetime.today())
    st.divider()
    
    st.header("⚖️ 3. 資產與權重")
    selected_names = st.multiselect("選擇組成 ETF", list(ETF_DICT.keys()), default=["00713 元大台灣高息低波", "00878 國泰永續高股息"])
    
    weights = []
    if selected_names:
        default_w = 100 // len(selected_names)
        for i, name in enumerate(selected_names):
            val = default_w + (100 % len(selected_names) if i == len(selected_names)-1 else 0)
            w = st.number_input(f"{name[:5]} 權重 (%)", min_value=0, max_value=100, value=val)
            weights.append(w / 100)
            
    st.divider()
    st.header("🔥 4. 質押槓桿設定")
    leverage_pct = st.slider("質押借款比例 (%)", 0, 100, 20)
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
            df_div_annual, df_div_raw = load_dividend_data(selected_tickers)
            
        if not df_adj.empty and not df_price.empty:
            adj_returns = df_adj.pct_change().dropna()
            price_returns = df_price.pct_change().dropna()
            
            port_adj_returns = (adj_returns * weights).sum(axis=1)
            port_price_returns = (price_returns * weights).sum(axis=1)
            
            port_div_series = None
            port_price_series_for_yield = None
            if not df_div_annual.empty:
                valid_div_cols = [c for c in df_div_annual.columns if c in selected_tickers]
                if valid_div_cols:
                    port_div_series = (df_div_annual[valid_div_cols] * [weights[selected_tickers.index(c)] for c in valid_div_cols]).sum(axis=1)
                    port_price_series_for_yield = (df_adj[valid_div_cols] * [weights[selected_tickers.index(c)] for c in valid_div_cols]).sum(axis=1)
            
            # --- 儲存按鈕區塊 ---
            st.subheader("💾 命名與保存當前策略")
            col_name, col_btn = st.columns([3, 1])
            with col_name:
                custom_name = st.text_input("✏️ 為此投資組合命名", value="我的現金流組合", label_visibility="collapsed")
            with col_btn:
                if st.button("➕ 記錄至績效比較表", type="primary", use_container_width=True):
                    p_metrics, p_traj = calculate_metrics_and_trajectory(port_adj_returns, port_price_returns, f"🎯 {custom_name} (原型)", 0, port_div_series, port_price_series_for_yield, borrow_rate, initial_capital, annual_expense, df_div_raw)
                    st.session_state.saved_portfolios.append({"metrics": p_metrics, "traj": p_traj})
                    
                    if leverage_pct > 0:
                        l_metrics, l_traj = calculate_metrics_and_trajectory(port_adj_returns, port_price_returns, f"🔥 {custom_name} (質押)", leverage_pct, port_div_series, port_price_series_for_yield, borrow_rate, initial_capital, annual_expense, df_div_raw)
                        st.session_state.saved_portfolios.append({"metrics": l_metrics, "traj": l_traj})
            st.divider()

            # --- 績效比較表 ---
            st.subheader("📋 績效比較表")
            
            # 準備顯示資料
            display_data = []
            
            # 1. 單一 ETF
            for col in adj_returns.columns:
                etf_name = [k for k, v in ETF_DICT.items() if v == col][0]
                etf_div_series = df_div_annual[col] if col in df_div_annual.columns else None
                etf_price_series = df_adj[col]
                m, _ = calculate_metrics_and_trajectory(adj_returns[col], price_returns[col], etf_name, 0, etf_div_series, etf_price_series, borrow_rate, initial_capital, annual_expense, df_div_raw)
                m.pop('id', None) # 移除不必要的內部 id
                display_data.append(m)
                
            # 2. 歷史紀錄與刪除功能
            for idx, item in enumerate(st.session_state.saved_portfolios):
                m = item['metrics'].copy()
                item_id = m.pop('id', None)
                
                # 這裡使用 st.columns 來並排顯示刪除按鈕與資料 (僅為示意，更優雅的作法是直接渲染 HTML 表格並綁定 JS，但在 Streamlit 中較難實作)
                # 我們退而求其次，在總表下方提供一個下拉選單來刪除特定紀錄
                display_data.append(m)
                
            # 3. 當前預覽
            curr_p_m, curr_p_t = calculate_metrics_and_trajectory(port_adj_returns, port_price_returns, f"👁️ 預覽: {custom_name} (原型)", 0, port_div_series, port_price_series_for_yield, borrow_rate, initial_capital, annual_expense, df_div_raw, is_preview=True)
            curr_p_m.pop('id', None)
            display_data.append(curr_p_m)
            
            if leverage_pct > 0:
                curr_l_m, curr_l_t = calculate_metrics_and_trajectory(port_adj_returns, port_price_returns, f"👁️ 預覽: {custom_name} (質押)", leverage_pct, port_div_series, port_price_series_for_yield, borrow_rate, initial_capital, annual_expense, df_div_raw, is_preview=True)
                curr_l_m.pop('id', None)
                display_data.append(curr_l_m)
                
            comparison_df = pd.DataFrame(display_data).set_index("標的名稱")
            
            def render_html_table(df):
                html = "<table style='width:100%; text-align:center; border-collapse: collapse; font-family: sans-serif; font-size: 0.9em;'>"
                html += "<tr style='background-color: #1E1E1E; border-bottom: 2px solid #444;'>"
                html += f"<th style='padding: 8px; text-align:left;'>標的名稱</th>"
                for col in df.columns:
                    html += f"<th style='padding: 8px; text-align:center;'>{col}</th>"
                html += "</tr>"
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
                    html += f"<td style='padding: 8px; text-align:left; color:{color}; font-weight:{font_weight};'>{index}</td>"
                    for item in row:
                        html += f"<td style='padding: 8px; text-align:center; color:{color}; font-weight:{font_weight};'>{item}</td>"
                    html += "</tr>"
                html += "</table>"
                return html

            st.markdown(render_html_table(comparison_df), unsafe_allow_html=True)
            
            # --- 刪除特定歷史紀錄介面 ---
            if st.session_state.saved_portfolios:
                st.write("")
                del_col1, del_col2 = st.columns([3, 1])
                with del_col1:
                    options = [f"{i}: {p['metrics']['標的名稱']}" for i, p in enumerate(st.session_state.saved_portfolios)]
                    selected_del = st.selectbox("選擇要刪除的紀錄", options, label_visibility="collapsed")
                with del_col2:
                    if st.button("🗑️ 刪除選取紀錄", use_container_width=True):
                        idx_to_del = int(selected_del.split(":")[0])
                        st.session_state.saved_portfolios.pop(idx_to_del)
                        st.rerun()

            st.caption("💡 **淨資產**為扣除負債後的真實身價。**總資產**包含質押借款。每年年底結算，配息大於生活費則再投入，不足則變賣本金。")
            st.divider()

            # --- 資金曲線雙圖表 ---
            st.subheader("📈 真實提領軌跡預覽 (包含再投入與變賣本金)")
            
            # 這裡繪製的是加入生活費提領邏輯後的 "淨資產" 曲線
            fig_traj = go.Figure()
            # 為了避免 ID 衝突，我們在繪圖時不指定硬體 ID，Plotly 會自動處理
            fig_traj.add_trace(go.Scatter(x=curr_p_t.index, y=curr_p_t['Net_Assets'], mode='lines', name='🎯 當前組合淨資產 (未槓桿)', line=dict(color='#2E86C1', width=2)))
            if leverage_pct > 0:
                fig_traj.add_trace(go.Scatter(x=curr_l_t.index, y=curr_l_t['Net_Assets'], mode='lines', name=f'🔥 當前組合淨資產 (質押 {leverage_pct}%)', line=dict(color='#E74C3C', width=2)))
            
            fig_traj.update_layout(
                hovermode="x unified", 
                yaxis_title="淨資產金額 (元)", 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # 使用唯一的 key 參數來避免 duplicate ID error
            st.plotly_chart(fig_traj, use_container_width=True, key="traj_chart_main")
            st.info(f"👆 這張圖展示了在每年需提領 **{annual_expense} 元**生活費的壓力下，你的淨資產是持續成長（配息 > 生活費，自動再投入）還是逐漸枯竭（配息不足，被迫變賣本金）。")
                
        else:
            st.warning("資料獲取失敗，請確認時間區間內是否有足夠報價。")
else:
    st.info("請從左側欄位選擇至少一檔 ETF 來建立投資組合。")
