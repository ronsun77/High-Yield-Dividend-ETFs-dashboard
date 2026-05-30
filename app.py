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
# 2. 資料抓取引擎 (只抓原始數據)
# ==========================================
@st.cache_data(ttl=3600)
def load_raw_data(tickers, start_date, end_date):
    """
    只負責抓取最純粹的「原始收盤價」與「歷史配息紀錄表」。
    絕不依賴 yfinance 的任何還原邏輯。
    """
    if not tickers: return pd.DataFrame(), {}
    
    raw_prices = {}
    div_raw_dict = {}
    
    for ticker in tickers:
        try:
            # 1. 抓取原始收盤價 (完全不調整)
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if 'Close' in df.columns:
                    df.index = df.index.tz_localize(None)
                    raw_prices[ticker] = df['Close']
            
            # 2. 抓取配息紀錄
            tk = yf.Ticker(ticker)
            divs = tk.dividends
            if not divs.empty:
                divs.index = divs.index.tz_localize(None)
                # 只保留在回測區間內的配息
                div_raw_dict[ticker] = divs[(divs.index >= pd.Timestamp(start_date)) & (divs.index <= pd.Timestamp(end_date))]
        except Exception:
            continue
            
    df_price = pd.DataFrame(raw_prices).dropna() # 取交集對齊時間
    return df_price, div_raw_dict

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
# 3. 核心模擬器 (真實資金池運算)
# ==========================================
def run_simulation(df_price, div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense):
    """
    真實資金池模擬器。
    透過追蹤每天的「持有單位數」與「剩餘現金」，完美分離出含息與不含息的真實差異。
    """
    lev_ratio = leverage_pct / 100.0
    daily_borrow_rate = borrow_rate / 252
    
    # 初始資金池設定
    total_cash_pool = initial_capital * (1 + lev_ratio)
    debt = initial_capital * lev_ratio
    
    # 為每檔 ETF 分配初始單位數
    shares_dict = {}
    for i, ticker in enumerate(df_price.columns):
        alloc_cash = total_cash_pool * weights[i]
        initial_price = df_price[ticker].iloc[0]
        shares_dict[ticker] = alloc_cash / initial_price
        
    trajectory = []
    
    # 用於追蹤純價格成長 (不含息) 的基準點
    initial_base_value = total_cash_pool
    current_year = -1
    year_accumulated_div = 0
    
    # === 每日迴圈模擬 ===
    for date, prices in df_price.iterrows():
        daily_portfolio_value = 0
        daily_base_value = 0
        daily_div_received = 0
        
        # 1. 檢查今日是否有配息
        for ticker in df_price.columns:
            divs = div_raw_dict.get(ticker, pd.Series(dtype=float))
            if date in divs.index:
                # 收到配息 = 持有單位數 * 每股配息金額
                div_amount = divs.loc[date]
                daily_div_received += shares_dict[ticker] * div_amount
                
            # 計算當日市值 (持有單位數 * 當日股價)
            daily_portfolio_value += shares_dict[ticker] * prices[ticker]
            
            # 計算不含息的純價格基準 (初始單位數 * 當日股價)
            initial_shares = (total_cash_pool * weights[df_price.columns.get_loc(ticker)]) / df_price[ticker].iloc[0]
            daily_base_value += initial_shares * prices[ticker]
            
        # 累積今年的配息
        year_accumulated_div += daily_div_received
        
        # 如果今天有配息，立刻啟動 DRIP (股息再投入)
        if daily_div_received > 0:
            for i, ticker in enumerate(df_price.columns):
                reinvest_cash = daily_div_received * weights[i]
                new_shares = reinvest_cash / prices[ticker]
                shares_dict[ticker] += new_shares
                
            # 更新市值 (因為買了新單位)
            daily_portfolio_value = 0
            for ticker in df_price.columns:
                daily_portfolio_value += shares_dict[ticker] * prices[ticker]
                
        # 2. 每日利息扣除 (直接從總市值中扣除)
        daily_interest = debt * daily_borrow_rate
        daily_portfolio_value -= daily_interest
        daily_base_value -= daily_interest
        
        # 3. 跨年結算生活費提領
        year = date.year
        if year != current_year:
            if current_year != -1:
                # 年底結算：用這一年累積的配息去繳生活費
                if year_accumulated_div >= annual_expense:
                    # 配息夠付，年初已經把配息全買成股票了，所以現在要「賣掉」等值的生活費
                    sell_amount = annual_expense
                else:
                    # 配息不夠付，差額要變賣本金
                    shortfall = annual_expense - year_accumulated_div
                    sell_amount = year_accumulated_div + shortfall # 等於賣掉 annual_expense 的市值
                    
                # 按權重變賣單位數
                for i, ticker in enumerate(df_price.columns):
                    sell_cash = sell_amount * weights[i]
                    sell_shares = sell_cash / prices[ticker]
                    shares_dict[ticker] -= sell_shares
                    
                # 重新計算扣除生活費後的市值
                daily_portfolio_value = sum([shares_dict[ticker] * prices[ticker] for ticker in df_price.columns])
                
                # 不含息基準也要扣除生活費 (直接扣現金)
                daily_base_value -= annual_expense
                
                year_accumulated_div = 0
            current_year = year

        # 4. 記錄當日結果
        net_assets = daily_portfolio_value - debt
        net_base_assets = daily_base_value - debt
        maintenance_margin = (daily_portfolio_value / debt * 100) if debt > 0 else float('inf')
        
        trajectory.append({
            'Date': date, 
            'Net_Assets_Adj': net_assets,      # 含息軌跡 (已扣生活費)
            'Net_Assets_Price': net_base_assets, # 不含息軌跡 (純價差，扣生活費)
            'Maintenance_Margin': maintenance_margin
        })
        
    traj_df = pd.DataFrame(trajectory).set_index('Date')
    
    # --- 計算指標 ---
    years = len(traj_df) / 252
    
    # 含息指標
    cagr_adj = (traj_df['Net_Assets_Adj'].iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 and traj_df['Net_Assets_Adj'].iloc[-1] > 0 else 0
    daily_ret_adj = traj_df['Net_Assets_Adj'].pct_change().dropna()
    volatility = daily_ret_adj.std() * np.sqrt(252) if not daily_ret_adj.empty else 0
    sharpe_ratio = (cagr_adj - 0.015) / volatility if volatility != 0 else 0
    
    cum_adj = traj_df['Net_Assets_Adj'] / initial_capital
    running_max = cum_adj.cummax()
    mdd = ((cum_adj - running_max) / running_max).min() if not cum_adj.empty else 0
    
    # 不含息指標
    cagr_price = (traj_df['Net_Assets_Price'].iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 and traj_df['Net_Assets_Price'].iloc[-1] > 0 else 0
    
    # 配息率估算 (以最後一年的總配息 / 均價市值)
    yield_val = "N/A"
    
    min_maintenance = traj_df['Maintenance_Margin'].min() if debt > 0 else float('inf')
    
    return {
        "期末淨資產(萬)": f"{traj_df['Net_Assets_Adj'].iloc[-1] / 10000:.2f}",
        "期末總資產(萬)": f"{(traj_df['Net_Assets_Adj'].iloc[-1] + debt) / 10000:.2f}",
        "含息年化(%)": f"{cagr_adj * 100:.2f}",
        "不含息年化(%)": f"{cagr_price * 100:.2f}",
        "最低維持率": f"{min_maintenance:.0f}%" if debt > 0 else "N/A",
        "年化波動率(%)": f"{volatility * 100:.2f}",
        "最大回撤(%)": f"{mdd * 100:.2f}",
        "夏普值": f"{sharpe_ratio:.2f}"
    }, traj_df


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
        with st.spinner("啟動真實資金池模擬引擎..."):
            df_price, div_raw_dict = load_raw_data(selected_tickers, start_date, end_date)
            
        if not df_price.empty:
            actual_start = df_price.index[0].strftime('%Y-%m-%d')
            actual_end = df_price.index[-1].strftime('%Y-%m-%d')
            
            # --- 儲存區塊 ---
            st.subheader("💾 命名與保存當前策略")
            col_name, col_btn = st.columns([3, 1])
            with col_name:
                custom_name = st.text_input("✏️ 為此投資組合命名", value="我的現金流組合", label_visibility="collapsed")
            with col_btn:
                if st.button("➕ 記錄至績效比較表", type="primary", use_container_width=True):
                    p_metrics, p_traj = run_simulation(df_price, div_raw_dict, weights, initial_capital, 0, borrow_rate, annual_expense)
                    p_metrics.update({"標的名稱": f"🎯 {custom_name} (原型)", "質押": "0%"})
                    st.session_state.saved_portfolios.append({"metrics": p_metrics, "traj": p_traj})
                    
                    if leverage_pct > 0:
                        l_metrics, l_traj = run_simulation(df_price, div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense)
                        l_metrics.update({"標的名稱": f"🔥 {custom_name} (質押)", "質押": f"{leverage_pct}%"})
                        st.session_state.saved_portfolios.append({"metrics": l_metrics, "traj": l_traj})
            st.divider()

            # --- 績效比較表 ---
            st.subheader(f"📋 績效比較表 (同期基準: {actual_start} 至 {actual_end})")
            display_data = []
            
            # 1. 單一 ETF 基準
            for col in df_price.columns:
                etf_name = [k for k, v in current_etf_dict.items() if v == col][0]
                m, _ = run_simulation(df_price[[col]], div_raw_dict, [1.0], initial_capital, 0, borrow_rate, annual_expense)
                m.update({"標的名稱": etf_name, "質押": "0%"})
                
                # 計算單一 ETF 填息與 CV
                fill_rate, fill_days, cv_str = "N/A", "N/A", "N/A"
                if col in div_raw_dict and not div_raw_dict[col].empty:
                    fill_rate, fill_days = calculate_fill_rate(div_raw_dict, df_price[col], col)
                    valid_divs = div_raw_dict[col].groupby(div_raw_dict[col].index.year).sum()
                    if len(valid_divs[valid_divs>0]) >= 2:
                        cv_str = f"{valid_divs.std() / valid_divs.mean():.2f}"
                
                m.update({"填息成功率": fill_rate, "平均填息天數": fill_days, "配息 CV": cv_str})
                display_data.append(m)
                
            # 2. 歷史紀錄
            for item in st.session_state.saved_portfolios:
                display_data.append(item['metrics'].copy())
                
            # 3. 當前預覽
            curr_p_m, curr_p_t = run_simulation(df_price, div_raw_dict, weights, initial_capital, 0, borrow_rate, annual_expense)
            curr_p_m.update({"標的名稱": f"👁️ 預覽: {custom_name} (原型)", "質押": "0%", "填息成功率": "N/A", "平均填息天數": "N/A", "配息 CV": "N/A"})
            display_data.append(curr_p_m)
            
            if leverage_pct > 0:
                curr_l_m, curr_l_t = run_simulation(df_price, div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense)
                curr_l_m.update({"標的名稱": f"👁️ 預覽: {custom_name} (質押)", "質押": f"{leverage_pct}%", "填息成功率": "N/A", "平均填息天數": "N/A", "配息 CV": "N/A"})
                display_data.append(curr_l_m)
                
            # 確保欄位順序美觀
            ordered_cols = ["標的名稱", "質押", "最低維持率", "期末淨資產(萬)", "期末總資產(萬)", "含息年化(%)", "不含息年化(%)", "填息成功率", "平均填息天數", "年化波動率(%)", "最大回撤(%)", "夏普值", "配息 CV"]
            comparison_df = pd.DataFrame(display_data).set_index("標的名稱")
            comparison_df = comparison_df[[c for c in ordered_cols if c in comparison_df.columns]]
            
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

            st.divider()

            # --- 雙圖表區塊 ---
            st.subheader("📈 真實提領軌跡預覽 (包含再投入與變賣本金)")
            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(x=curr_p_t.index, y=curr_p_t['Net_Assets_Adj'], mode='lines', name='🎯 淨資產(未槓桿)', line=dict(color='#2E86C1', width=2)))
            if leverage_pct > 0:
                fig_traj.add_trace(go.Scatter(x=curr_l_t.index, y=curr_l_t['Net_Assets_Adj'], mode='lines', name=f'🔥 淨資產(質押 {leverage_pct}%)', line=dict(color='#E74C3C', width=2)))
            
            # 加入不含息的純價差軌跡作為對照
            fig_traj.add_trace(go.Scatter(x=curr_p_t.index, y=curr_p_t['Net_Assets_Price'], mode='lines', name='單純價差(不含息)', line=dict(color='#7F8C8D', width=1, dash='dash')))
            
            fig_traj.update_layout(hovermode="x unified", yaxis_title="淨資產金額 (元)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_traj, use_container_width=True, key="traj_chart_main")
            
            st.subheader("🚨 質押維持率壓力監測")
            if leverage_pct > 0:
                fig_margin = go.Figure()
                fig_margin.add_trace(go.Scatter(x=curr_l_t.index, y=curr_l_t['Maintenance_Margin'], mode='lines', name='當日維持率 (%)', line=dict(color='#F1C40F', width=2)))
                fig_margin.add_hline(y=166, line_dash="dash", line_color="orange", annotation_text="166% (追繳線)", annotation_position="bottom right")
                fig_margin.add_hline(y=130, line_dash="solid", line_color="red", annotation_text="130% (斷頭線)", annotation_position="bottom right")
                
                fig_margin.update_layout(hovermode="x unified", yaxis_title="維持率 (%)", yaxis=dict(range=[100, max(300, curr_l_t['Maintenance_Margin'].max() * 1.1)]))
                st.plotly_chart(fig_margin, use_container_width=True, key="margin_chart_main")
            else:
                st.info("未開啟質押槓桿，無維持率風險。")
                
        else:
            st.warning("資料獲取失敗，請確認時間區間內是否有足夠報價。")
else:
    st.info("請從左側欄位選擇至少一檔 ETF 來建立投資組合。")
