import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. 網頁設定 & 暫存初始化
# ==========================================
st.set_page_config(page_title="高股息策略與質押模擬器", layout="wide")
st.title("🛡️ 台灣高股息 ETF 現金流組合與質押模擬器")

if 'saved_portfolios' not in st.session_state:
    st.session_state.saved_portfolios = []
if 'custom_etfs' not in st.session_state:
    st.session_state.custom_etfs = {}

DEFAULT_ETF_DICT = {
    "0056 元大高股息": "0056.TW",
    "00878 國泰永續高股息": "00878.TW",
    "00919 群益台灣精選高息": "00919.TW",
    "00929 復華台灣科技優息": "00929.TW",
    "00713 元大台灣高息低波": "00713.TW"
}

# ==========================================
# 2. 資料抓取與手動還原引擎 (核心修正)
# ==========================================
@st.cache_data(ttl=3600)
def load_and_reconstruct_data(tickers, start_date, end_date):
    """
    抓取原始價格，並利用除息紀錄，手動計算出精確的「含息還原價(Total Return)」。
    徹底解決 yfinance 台股資料含息/不含息相同的問題。
    """
    if not tickers: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    
    raw_prices = {}
    div_raw_dict = {}
    
    for ticker in tickers:
        try:
            # 強制抓取未還原的原始收盤價 (Price Return)
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if 'Close' in df.columns:
                    df.index = df.index.tz_localize(None)
                    raw_prices[ticker] = df['Close']
            
            # 抓取配息紀錄
            tk = yf.Ticker(ticker)
            divs = tk.dividends
            if not divs.empty:
                divs.index = divs.index.tz_localize(None)
                div_raw_dict[ticker] = divs[(divs.index >= pd.Timestamp(start_date)) & (divs.index <= pd.Timestamp(end_date))]
        except Exception:
            continue
            
    df_price = pd.DataFrame(raw_prices).dropna() # 取交集對齊時間
    if df_price.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    
    # === 手動建構含息還原價格 (Total Return Index) ===
    df_adj = df_price.copy()
    for ticker in df_price.columns:
        divs = div_raw_dict.get(ticker, pd.Series(dtype=float))
        adj_series = df_price[ticker].copy()
        
        # 從最後一天往回算，遇到除息日，就將除息日「之前」的所有價格向下調整
        # 這樣才能算出真實的複利總報酬
        sorted_divs = divs.sort_index(ascending=False)
        for ex_date, div_amt in sorted_divs.items():
            if ex_date in adj_series.index:
                pre_dates = adj_series.index[adj_series.index < ex_date]
                if not pre_dates.empty:
                    pre_date = pre_dates[-1]
                    pre_price_raw = df_price[ticker].loc[pre_date] # 使用原始價格計算比例
                    if pre_price_raw > div_amt:
                        adj_factor = (pre_price_raw - div_amt) / pre_price_raw
                        adj_series.loc[:pre_date] = adj_series.loc[:pre_date] * adj_factor
                        
        df_adj[ticker] = adj_series
        
    div_annual_dict = {ticker: divs.groupby(divs.index.year).sum() for ticker, divs in div_raw_dict.items() if not divs.empty}
    df_div_annual = pd.DataFrame(div_annual_dict).fillna(0)
    
    return df_adj, df_price, df_div_annual, div_raw_dict

def calculate_fill_rate(div_series_dict, price_series, ticker):
    if div_series_dict is None or ticker not in div_series_dict or price_series is None or price_series.empty:
        return "N/A", "N/A"
    
    divs = div_series_dict[ticker]
    success_count, total_days, valid_divs = 0, 0, 0
    
    for ex_date, div_amount in divs.items():
        pre_ex_dates = price_series.index[price_series.index < ex_date]
        if pre_ex_dates.empty: continue
        pre_ex_price = price_series.loc[pre_ex_dates[-1]]
        
        post_ex_prices = price_series[price_series.index >= ex_date]
        if post_ex_prices.empty: continue
        
        filled_dates = post_ex_prices[post_ex_prices >= pre_ex_price].index
        if not filled_dates.empty:
            success_count += 1
            total_days += (filled_dates[0] - ex_date).days
            valid_divs += 1
            
    if valid_divs == 0: return "N/A", "N/A"
    return f"{(success_count / len(divs)) * 100:.0f}%", f"{total_days / success_count:.0f}"

# ==========================================
# 3. 核心指標與資產軌跡運算
# ==========================================
def calculate_metrics_and_trajectory(adj_returns, price_returns, name, leverage_pct, div_series, price_series, borrow_rate, initial_capital, annual_expense, div_raw_dict, ticker_for_fill_rate=None):
    lev_ratio = leverage_pct / 100.0
    daily_borrow_rate = borrow_rate / 252
    
    # 槓桿組合日報酬
    lev_adj_returns = adj_returns * (1 + lev_ratio) - (daily_borrow_rate * lev_ratio)
    lev_price_returns = price_returns * (1 + lev_ratio) - (daily_borrow_rate * lev_ratio)
    
    # 模擬提領軌跡與維持率
    total_assets = initial_capital * (1 + lev_ratio)
    debt = initial_capital * lev_ratio
    net_assets = initial_capital
    
    trajectory = []
    current_year = -1
    
    for date, ret in lev_price_returns.items():
        year = date.year
        if year != current_year:
            if current_year != -1:
                if div_series is not None and current_year in div_series.index:
                    yield_amount = div_series.loc[current_year]
                    p_year = price_series[price_series.index.year == current_year] if price_series is not None else pd.Series()
                    if not p_year.empty:
                        actual_div_yield = yield_amount / p_year.mean()
                        cash_received = net_assets * actual_div_yield * (1 + lev_ratio)
                        interest_paid = debt * borrow_rate
                        net_cash = cash_received - interest_paid
                        
                        if net_cash >= annual_expense:
                            reinvest_amount = net_cash - annual_expense
                            total_assets += reinvest_amount
                            net_assets += reinvest_amount
                        else:
                            shortfall = annual_expense - net_cash
                            total_assets -= shortfall
                            net_assets -= shortfall
            current_year = year
            
        total_assets = total_assets * (1 + ret)
        net_assets = total_assets - debt
        
        # 計算當日維持率 = 總資產 / 融資金額
        maintenance_margin = (total_assets / debt * 100) if debt > 0 else float('inf')
        
        trajectory.append({'Date': date, 'Net_Assets': net_assets, 'Maintenance_Margin': maintenance_margin})
        
    traj_df = pd.DataFrame(trajectory).set_index('Date')
    
    # 指標計算
    cagr_adj = ((1 + lev_adj_returns).prod() ** (252 / len(lev_adj_returns))) - 1 if not lev_adj_returns.empty else 0
    cagr_price = ((1 + lev_price_returns).prod() ** (252 / len(lev_price_returns))) - 1 if not lev_price_returns.empty else 0
    volatility = lev_adj_returns.std() * np.sqrt(252) if not lev_adj_returns.empty else 0
    sharpe_ratio = (cagr_adj - 0.015) / volatility if volatility != 0 else 0
    
    cum_adj = (1 + lev_adj_returns).cumprod()
    running_max = cum_adj.cummax()
    mdd = ((cum_adj - running_max) / running_max).min() if not lev_adj_returns.empty else 0
    
    min_maintenance = traj_df['Maintenance_Margin'].min() if not traj_df.empty and debt > 0 else float('inf')
    
    cv_str, yield_val, fill_rate_str, fill_days_str = "N/A", "N/A", "N/A", "N/A"
    
    if div_series is not None and len(div_series[div_series > 0]) > 0:
        valid_divs = div_series[div_series > 0]
        if len(valid_divs) >= 2:
            cv_str = f"{valid_divs.std() / valid_divs.mean():.2f}"
            
        if price_series is not None:
            yearly_yields = [d / price_series[price_series.index.year == y].mean() for y, d in valid_divs.items() if not price_series[price_series.index.year == y].empty]
            if yearly_yields:
                final_yield = np.mean(yearly_yields) * (1 + lev_ratio) - (borrow_rate * lev_ratio)
                yield_val = f"{final_yield * 100:.2f}"
                
    if ticker_for_fill_rate:
        fill_rate_str, fill_days_str = calculate_fill_rate(div_raw_dict, price_series, ticker_for_fill_rate)
                
    metrics = {
        "標的名稱": name,
        "質押": f"{leverage_pct}%",
        "最低維持率": f"{min_maintenance:.0f}%" if min_maintenance != float('inf') else "N/A",
        "期末淨資產(萬)": f"{net_assets / 10000:.2f}",
        "期末總資產(萬)": f"{total_assets / 10000:.2f}",
        "含息年化(%)": f"{cagr_adj * 100:.2f}",
        "不含息年化(%)": f"{cagr_price * 100:.2f}",
        "年化配息率(%)": yield_val,
        "填息成功率": fill_rate_str,
        "平均填息天數": fill_days_str,
        "年化波動率(%)": f"{volatility * 100:.2f}",
        "最大回撤(%)": f"{mdd * 100:.2f}",
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
    current_etf_dict = {**DEFAULT_ETF_DICT, **st.session_state.custom_etfs}
    with st.expander("➕ 新增自訂 ETF"):
        new_etf_code = st.text_input("輸入台股代碼 (例: 006208)")
        new_etf_name = st.text_input("輸入顯示名稱 (例: 006208 富邦台50)")
        if st.button("新增標的"):
            if new_etf_code and new_etf_name:
                ticker_symbol = f"{new_etf_code}.TW" if not new_etf_code.endswith(".TW") else new_etf_code
                st.session_state.custom_etfs[new_etf_name] = ticker_symbol
                st.rerun()
                
    selected_names = st.multiselect("選擇組成 ETF", list(current_etf_dict.keys()), default=["00713 元大台灣高息低波", "00878 國泰永續高股息"])
    weights = []
    if selected_names:
        default_w = 100 // len(selected_names)
        for i, name in enumerate(selected_names):
            val = default_w + (100 % len(selected_names) if i == len(selected_names)-1 else 0)
            w = st.number_input(f"{name[:5]} 權重 (%)", min_value=0, max_value=100, value=val)
            weights.append(w / 100)
            
    st.divider()
    st.header("🔥 4. 質押槓桿設定")
    
    # 支援兩種輸入方式互換
    leverage_mode = st.radio("設定方式", ["依借款比例", "依目標維持率"])
    if leverage_mode == "依借款比例":
        leverage_pct = st.slider("質押借款比例 (%)", 0, 100, 20)
        target_margin = ((1 + leverage_pct/100) / (leverage_pct/100) * 100) if leverage_pct > 0 else float('inf')
        if leverage_pct > 0:
            st.caption(f"推算初始維持率約為: {target_margin:.0f}%")
    else:
        target_margin = st.number_input("目標初始維持率 (%)", min_value=130, max_value=1000, value=166, step=10)
        leverage_pct = round(100 / (target_margin/100 - 1)) if target_margin > 100 else 0
        st.caption(f"推算需借款比例約為: {leverage_pct}%")
        
    borrow_rate = st.number_input("借款年利率 (%)", value=2.5, step=0.1) / 100.0

# ==========================================
# 5. 主畫面運算與渲染
# ==========================================
if selected_names:
    if sum(weights) != 1.0:
        st.error(f"⚠️ 權重總和需為 100%，目前為 {sum(weights)*100:.0f}%")
    else:
        selected_tickers = [current_etf_dict[name] for name in selected_names]
        with st.spinner("載入報價並手動還原真實股息模型中..."):
            df_adj, df_price, df_div_annual, df_div_raw = load_and_reconstruct_data(selected_tickers, start_date, end_date)
            
        if not df_adj.empty and not df_price.empty:
            actual_start = df_adj.index[0].strftime('%Y-%m-%d')
            actual_end = df_adj.index[-1].strftime('%Y-%m-%d')
            
            adj_returns = df_adj.pct_change().dropna()
            price_returns = df_price.pct_change().dropna()
            
            port_adj_returns = (adj_returns * weights).sum(axis=1)
            port_price_returns = (price_returns * weights).sum(axis=1)
            
            port_div_series = None
            port_price_series_for_yield = None
            if not df_div_annual.empty:
                valid_cols = [c for c in df_div_annual.columns if c in selected_tickers]
                if valid_cols:
                    port_div_series = (df_div_annual[valid_cols] * [weights[selected_tickers.index(c)] for c in valid_cols]).sum(axis=1)
                    port_price_series_for_yield = (df_price[valid_cols] * [weights[selected_tickers.index(c)] for c in valid_cols]).sum(axis=1)
            
            # --- 儲存區塊 ---
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
            st.subheader(f"📋 績效比較表 (同期基準: {actual_start} 至 {actual_end})")
            
            display_data = []
            
            # 1. 單一 ETF (無槓桿基準)
            for col in adj_returns.columns:
                etf_name = [k for k, v in current_etf_dict.items() if v == col][0]
                etf_div_series = df_div_annual[col] if col in df_div_annual.columns else None
                etf_price_series = df_price[col]
                m, _ = calculate_metrics_and_trajectory(adj_returns[col], price_returns[col], etf_name, 0, etf_div_series, etf_price_series, borrow_rate, initial_capital, annual_expense, df_div_raw, ticker_for_fill_rate=col)
                display_data.append(m)
                
            # 2. 歷史紀錄
            for item in st.session_state.saved_portfolios:
                display_data.append(item['metrics'].copy())
                
            # 3. 當前預覽
            curr_p_m, curr_p_t = calculate_metrics_and_trajectory(port_adj_returns, port_price_returns, f"👁️ 預覽: {custom_name} (原型)", 0, port_div_series, port_price_series_for_yield, borrow_rate, initial_capital, annual_expense, df_div_raw)
            display_data.append(curr_p_m)
            
            if leverage_pct > 0:
                curr_l_m, curr_l_t = calculate_metrics_and_trajectory(port_adj_returns, port_price_returns, f"👁️ 預覽: {custom_name} (質押)", leverage_pct, port_div_series, port_price_series_for_yield, borrow_rate, initial_capital, annual_expense, df_div_raw)
                display_data.append(curr_l_m)
                
            comparison_df = pd.DataFrame(display_data).set_index("標的名稱")
            
            def render_html_table(df):
                html = "<table style='width:100%; text-align:center; border-collapse: collapse; font-family: sans-serif; font-size: 0.85em;'>"
                html += "<tr style='background-color: #1E1E1E; border-bottom: 2px solid #444;'>"
                html += f"<th style='padding: 6px; text-align:left;'>標的名稱</th>"
                for col in df.columns:
                    html += f"<th style='padding: 6px; text-align:center;'>{col}</th>"
                html += "</tr>"
                for index, row in df.iterrows():
                    bg_color, font_weight, color = "transparent", "normal", "#E0E0E0"
                    if str(index).startswith("👁️ 預覽"):
                        bg_color, font_weight, color = "#117A65", "bold", "white"
                    elif "🔥" in str(index) or "🎯" in str(index):
                        bg_color, color = "#2C3E50", "#D5D8DC"
                    
                    html += f"<tr style='background-color: {bg_color}; border-bottom: 1px solid #333;'>"
                    html += f"<td style='padding: 6px; text-align:left; color:{color}; font-weight:{font_weight};'>{index}</td>"
                    for item in row:
                        html += f"<td style='padding: 6px; text-align:center; color:{color}; font-weight:{font_weight};'>{item}</td>"
                    html += "</tr>"
                html += "</table>"
                return html

            st.markdown(render_html_table(comparison_df), unsafe_allow_html=True)
            
            if st.session_state.saved_portfolios:
                st.write("")
                del_col1, del_col2 = st.columns([3, 1])
                with del_col1:
                    options = [f"第 {i+1} 筆: {p['metrics']['標的名稱']}" for i, p in enumerate(st.session_state.saved_portfolios)]
                    selected_del = st.selectbox("選擇要刪除的歷史紀錄", options, label_visibility="collapsed")
                with del_col2:
                    if st.button("🗑️ 刪除選取的紀錄", use_container_width=True):
                        idx_to_del = int(selected_del.split(":")[0].replace("第 ", "").replace(" 筆", "")) - 1
                        st.session_state.saved_portfolios.pop(idx_to_del)
                        st.rerun()

            st.caption("💡 **最低維持率**評估抗斷頭能力。**不含息年化**代表純價格成長。皆為扣除生活費與利息後之真實模擬。")
            st.divider()

            # --- 雙圖表區塊 ---
            tab_traj, tab_margin = st.tabs(["📉 真實提領軌跡預覽", "🚨 質押維持率壓力監測"])
            
            with tab_traj:
                fig_traj = go.Figure()
                fig_traj.add_trace(go.Scatter(x=curr_p_t.index, y=curr_p_t['Net_Assets'], mode='lines', name='🎯 當前組合淨資產 (未槓桿)', line=dict(color='#2E86C1', width=2)))
                if leverage_pct > 0:
                    fig_traj.add_trace(go.Scatter(x=curr_l_t.index, y=curr_l_t['Net_Assets'], mode='lines', name=f'🔥 當前組合淨資產 (質押 {leverage_pct}%)', line=dict(color='#E74C3C', width=2)))
                fig_traj.update_layout(hovermode="x unified", yaxis_title="淨資產金額 (元)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_traj, use_container_width=True, key="traj_chart_main")
                
            with tab_margin:
                if leverage_pct > 0:
                    fig_margin = go.Figure()
                    fig_margin.add_trace(go.Scatter(x=curr_l_t.index, y=curr_l_t['Maintenance_Margin'], mode='lines', name='當日維持率 (%)', line=dict(color='#F1C40F', width=2)))
                    # 畫警戒線
                    fig_margin.add_hline(y=166, line_dash="dash", line_color="orange", annotation_text="166% (追繳警戒線)", annotation_position="bottom right")
                    fig_margin.add_hline(y=130, line_dash="solid", line_color="red", annotation_text="130% (斷頭死亡線)", annotation_position="bottom right")
                    
                    fig_margin.update_layout(hovermode="x unified", yaxis_title="維持率 (%)", yaxis=dict(range=[100, max(300, curr_l_t['Maintenance_Margin'].max() * 1.1)]))
                    st.plotly_chart(fig_margin, use_container_width=True, key="margin_chart_main")
                else:
                    st.info("未開啟質押槓桿，無維持率風險。")
                
        else:
            st.warning("資料獲取失敗，請確認時間區間內是否有足夠報價。")
else:
    st.info("請從左側欄位選擇至少一檔 ETF 來建立投資組合。")
