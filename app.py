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
# 2. 資料抓取引擎 (強制對齊時間軸以修復配息遺漏)
# ==========================================
@st.cache_data(ttl=3600)
def load_raw_data(tickers, start_date, end_date):
    if not tickers: return pd.DataFrame(), {}
    raw_prices = {}
    div_raw_dict = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if 'Close' in df.columns:
                    # 強制時間歸零對齊
                    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                    raw_prices[ticker] = df['Close']
            
            tk = yf.Ticker(ticker)
            divs = tk.dividends
            if not divs.empty:
                divs.index = pd.to_datetime(divs.index).tz_localize(None).normalize()
                div_raw_dict[ticker] = divs[(divs.index >= pd.Timestamp(start_date)) & (divs.index <= pd.Timestamp(end_date))]
        except Exception:
            continue
    return pd.DataFrame(raw_prices).dropna(), div_raw_dict

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
# 3. 三引擎提領模擬器 (Buy, Borrow, Die 真實現金流邏輯)
# ==========================================
def run_simulation(df_price, div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense, enable_rebalance):
    lev_ratio = leverage_pct / 100.0
    daily_borrow_rate = borrow_rate / 252
    
    initial_debt = initial_capital * lev_ratio
    total_initial_assets = initial_capital + initial_debt
    target_margin_ratio = (total_initial_assets / initial_debt) if initial_debt > 0 else float('inf')
    
    # --- 軌跡 1: 理論含息 (無視提領，無限DRIP) ---
    shares_theory = {t: (initial_capital * weights[i]) / df_price[t].iloc[0] for i, t in enumerate(df_price.columns)}
    
    # --- 軌跡 2: 真實淨資產 (真實世界的保證金帳戶) ---
    shares_real = {t: (total_initial_assets * weights[i]) / df_price[t].iloc[0] for i, t in enumerate(df_price.columns)}
    margin_balance = -initial_debt # 負值代表負債，正值代表現金
    
    # --- 軌跡 3: 單純價差 (股數永遠不變，純看底層資產漲跌) ---
    shares_static = shares_real.copy()
    
    trajectory = []
    current_year = -1
    
    for date, prices in df_price.iterrows():
        # 1. 處理配息發放
        daily_div_real = 0
        for i, ticker in enumerate(df_price.columns):
            divs = div_raw_dict.get(ticker, pd.Series(dtype=float))
            if date in divs.index:
                div_amount = divs.loc[date]
                if isinstance(div_amount, pd.Series): div_amount = div_amount.iloc[0]
                
                # 理論含息：配發當天立即買入
                shares_theory[ticker] += (shares_theory[ticker] * div_amount) / prices[ticker]
                
                # 真實淨資產：配發的現金進入保證金帳戶 (抵銷負債)
                daily_div_real += shares_real[ticker] * div_amount
                
        # 結算今日現金流入
        margin_balance += daily_div_real
        
        # 若保證金為正(無負債且有閒錢)，立刻再投入買股 (避免現金拖累)
        if margin_balance > 0:
            for i, ticker in enumerate(df_price.columns):
                shares_real[ticker] += (margin_balance * weights[i]) / prices[ticker]
            margin_balance = 0

        # 2. 扣除每日質押利息 (利息增加負債)
        if margin_balance < 0:
            margin_balance += margin_balance * daily_borrow_rate

        # 3. 跨年結算 (生活費提領 & 維持率再平衡)
        year = date.year
        if year != current_year:
            if current_year != -1:
                # 扣除生活費 (生活費增加負債)
                margin_balance -= annual_expense
                
                # 【傳統策略防呆】：若無質押，不允許負債，必須變賣股票換取生活費
                if leverage_pct == 0 and margin_balance < 0:
                    shortfall = -margin_balance
                    for i, ticker in enumerate(df_price.columns):
                        shares_real[ticker] -= (shortfall * weights[i]) / prices[ticker]
                    margin_balance = 0
                    
                # 【Buy, Borrow, Die 單向再平衡】：若有質押，且開啟再平衡
                if leverage_pct > 0 and enable_rebalance and margin_balance < 0 and target_margin_ratio != float('inf'):
                    val_real = sum([shares_real[t] * prices[t] for t in df_price.columns])
                    debt_real = -margin_balance
                    current_margin = val_real / debt_real
                    
                    # 只有維持率「大於」目標時，才擴張信用買進；小於時裝死不賣。
                    if current_margin > target_margin_ratio:
                        # 數學推導: 新借款額 ΔD = (總市值 - 目標維持率 * 現有負債) / (目標維持率 - 1)
                        delta_debt = (val_real - target_margin_ratio * debt_real) / (target_margin_ratio - 1)
                        if delta_debt > 0:
                            margin_balance -= delta_debt # 增加負債
                            for i, ticker in enumerate(df_price.columns):
                                shares_real[ticker] += (delta_debt * weights[i]) / prices[ticker] # 買入資產
                                
            current_year = year

        # 4. 記錄當日資產價值
        val_theory = sum([shares_theory[t] * prices[t] for t in df_price.columns])
        val_real = sum([shares_real[t] * prices[t] for t in df_price.columns])
        val_static = sum([shares_static[t] * prices[t] for t in df_price.columns])
        
        net_theory = val_theory
        net_real = val_real + margin_balance # 若為負代表扣除負債
        net_static = val_static - initial_debt
        
        maintenance_margin = (val_real / -margin_balance * 100) if margin_balance < 0 else float('inf')
        
        trajectory.append({
            'Date': date, 
            'Net_Theory': net_theory,
            'Net_Real': net_real,
            'Net_Static': net_static,
            'Maintenance_Margin': maintenance_margin
        })
        
    traj_df = pd.DataFrame(trajectory).set_index('Date')
    years = len(traj_df) / 252
    
    # 計算年化指標
    cagr_theory = (traj_df['Net_Theory'].iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 and traj_df['Net_Theory'].iloc[-1] > 0 else 0
    cagr_real = (traj_df['Net_Real'].iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 and traj_df['Net_Real'].iloc[-1] > 0 else 0
    cagr_static = (traj_df['Net_Static'].iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 and traj_df['Net_Static'].iloc[-1] > 0 else 0
    
    daily_ret_real = traj_df['Net_Real'].pct_change().dropna()
    volatility = daily_ret_real.std() * np.sqrt(252) if not daily_ret_real.empty else 0
    sharpe_ratio = (cagr_real - 0.015) / volatility if volatility != 0 else 0
    
    cum_real = traj_df['Net_Real'] / initial_capital
    running_max = cum_real.cummax()
    mdd = ((cum_real - running_max) / running_max).min() if not cum_real.empty else 0
    
    debt_final = -margin_balance if margin_balance < 0 else 0
    min_maintenance = traj_df['Maintenance_Margin'].min() if debt_final > 0 else float('inf')
    
    return {
        "期末淨資產(萬)": f"{traj_df['Net_Real'].iloc[-1] / 10000:.2f}",
        "期末總資產(萬)": f"{(traj_df['Net_Real'].iloc[-1] + debt_final) / 10000:.2f}",
        "理論含息年化(%)": f"{cagr_theory * 100:.2f}",
        "真實淨資產年化(%)": f"{cagr_real * 100:.2f}",
        "單純價差年化(%)": f"{cagr_static * 100:.2f}",
        "最低維持率": f"{min_maintenance:.0f}%" if debt_final > 0 else "N/A",
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
    st.header("🔥 4. 質押槓桿與再平衡")
    leverage_mode = st.radio("設定方式", ["依借款比例", "依目標維持率"])
    if leverage_mode == "依借款比例":
        leverage_pct = st.slider("質押借款比例 (%)", 0, 100, 20)
    else:
        target_margin = st.number_input("目標初始維持率 (%)", min_value=130, max_value=1000, value=166, step=10)
        leverage_pct = round(100 / (target_margin/100 - 1)) if target_margin > 100 else 0
        st.caption(f"推算需借款比例約為: {leverage_pct}%")
    borrow_rate = st.number_input("借款年利率 (%)", value=2.5, step=0.1) / 100.0
    
    enable_rebalance = False
    if leverage_pct > 0:
        enable_rebalance = st.checkbox("⚙️ 單向恆定維持率 (Buy, Borrow, Die)", value=False, help="每年底檢視，若維持率高於初始設定，則擴張信用買入更多資產；若跌破則絕不賣股去槓桿。")

# ==========================================
# 5. 主畫面運算與渲染
# ==========================================
if selected_names:
    if sum(weights) != 1.0:
        st.error(f"⚠️ 權重總和需為 100%，目前為 {sum(weights)*100:.0f}%")
    else:
        selected_tickers = [current_etf_dict[name] for name in selected_names]
        with st.spinner("啟動 BBD 現金流提領引擎..."):
            df_price, div_raw_dict = load_raw_data(selected_tickers, start_date, end_date)
            
        if not df_price.empty:
            actual_start = df_price.index[0].strftime('%Y-%m-%d')
            actual_end = df_price.index[-1].strftime('%Y-%m-%d')
            
            st.subheader("💾 命名與保存當前策略")
            col_name, col_btn = st.columns([3, 1])
            with col_name:
                custom_name = st.text_input("✏️ 為此投資組合命名", value="我的現金流組合", label_visibility="collapsed")
            with col_btn:
                if st.button("➕ 記錄至績效比較表", type="primary", use_container_width=True):
                    p_metrics, p_traj = run_simulation(df_price, div_raw_dict, weights, initial_capital, 0, borrow_rate, annual_expense, False)
                    p_metrics.update({"標的名稱": f"🎯 {custom_name} (無質押)", "質押": "0%"})
                    st.session_state.saved_portfolios.append({"metrics": p_metrics, "traj": p_traj})
                    
                    if leverage_pct > 0:
                        l_metrics, l_traj = run_simulation(df_price, div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense, enable_rebalance)
                        l_metrics.update({"標的名稱": f"🔥 {custom_name} (質押)", "質押": f"{leverage_pct}%"})
                        st.session_state.saved_portfolios.append({"metrics": l_metrics, "traj": l_traj})
            st.divider()

            st.subheader(f"📋 績效比較表 (同期基準: {actual_start} 至 {actual_end})")
            display_data = []
            
            # 1. 單一 ETF 基準
            for col in df_price.columns:
                etf_name = [k for k, v in current_etf_dict.items() if v == col][0]
                m, _ = run_simulation(df_price[[col]], div_raw_dict, [1.0], initial_capital, 0, borrow_rate, annual_expense, False)
                m.update({"標的名稱": etf_name, "質押": "0%"})
                
                fill_rate, fill_days, cv_str, yield_str = "N/A", "N/A", "N/A", "N/A"
                if col in div_raw_dict and not div_raw_dict[col].empty:
                    fill_rate, fill_days = calculate_fill_rate(div_raw_dict, df_price[col], col)
                    valid_divs = div_raw_dict[col].groupby(div_raw_dict[col].index.year).sum()
                    if len(valid_divs[valid_divs>0]) >= 2:
                        cv_str = f"{valid_divs.std() / valid_divs.mean():.2f}"
                    yearly_yields = [d / df_price[col][df_price[col].index.year == y].mean() for y, d in valid_divs.items() if not df_price[col][df_price[col].index.year == y].empty]
                    if yearly_yields:
                        yield_str = f"{np.mean(yearly_yields) * 100:.2f}"
                
                m.update({"年化配息率(%)": yield_str, "填息成功率": fill_rate, "平均填息天數": fill_days, "配息 CV": cv_str})
                display_data.append(m)
                
            # 2. 歷史紀錄
            for item in st.session_state.saved_portfolios:
                display_data.append(item['metrics'].copy())
                
            # 3. 當前預覽
            curr_p_m, curr_p_t = run_simulation(df_price, div_raw_dict, weights, initial_capital, 0, borrow_rate, annual_expense, False)
            curr_p_m.update({"標的名稱": f"👁️ 預覽: {custom_name} (無質押)", "質押": "0%", "年化配息率(%)": "見單檔", "填息成功率": "-", "平均填息天數": "-", "配息 CV": "-"})
            display_data.append(curr_p_m)
            
            if leverage_pct > 0:
                curr_l_m, curr_l_t = run_simulation(df_price, div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense, enable_rebalance)
                curr_l_m.update({"標的名稱": f"👁️ 預覽: {custom_name} (質押)", "質押": f"{leverage_pct}%", "年化配息率(%)": "見單檔", "填息成功率": "-", "平均填息天數": "-", "配息 CV": "-"})
                display_data.append(curr_l_m)
                
            ordered_cols = ["標的名稱", "質押", "最低維持率", "期末淨資產(萬)", "期末總資產(萬)", "理論含息年化(%)", "真實淨資產年化(%)", "單純價差年化(%)", "年化配息率(%)", "填息成功率", "平均填息天數", "年化波動率(%)", "最大回撤(%)", "夏普值", "配息 CV"]
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

            st.caption("💡 **理論含息**代表無提領壓力的無限複利。**真實淨資產**為扣生活費/利息後，自動抵債或借款的真實損益。**單純價差**為股數凍結的價格走勢。")
            st.divider()

            # --- 三軌跡資金曲線圖 ---
            st.subheader("📈 三重真實提領軌跡預覽 (包含現金流 Sweep 與擴張)")
            fig_traj = go.Figure()
            
            fig_traj.add_trace(go.Scatter(x=curr_p_t.index, y=curr_p_t['Net_Theory'], mode='lines', name='🌟 理論含息 (無提領)', line=dict(color='#8E44AD', width=1, dash='dot')))
            fig_traj.add_trace(go.Scatter(x=curr_p_t.index, y=curr_p_t['Net_Real'], mode='lines', name='🎯 真實淨資產(無質押)', line=dict(color='#2E86C1', width=2)))
            if leverage_pct > 0:
                rebalance_tag = " [單向再平衡]" if enable_rebalance else ""
                fig_traj.add_trace(go.Scatter(x=curr_l_t.index, y=curr_l_t['Net_Real'], mode='lines', name=f'🔥 真實淨資產(質押 {leverage_pct}%){rebalance_tag}', line=dict(color='#E74C3C', width=2)))
            fig_traj.add_trace(go.Scatter(x=curr_p_t.index, y=curr_p_t['Net_Static'], mode='lines', name='📏 單純價差基準線(股數不變)', line=dict(color='#7F8C8D', width=1, dash='dash')))
            
            fig_traj.update_layout(hovermode="x unified", yaxis_title="淨資產金額 (元)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_traj, use_container_width=True, key="traj_chart_main")
            
            # --- 維持率圖表 ---
            st.subheader("🚨 質押維持率壓力監測")
            if leverage_pct > 0:
                fig_margin = go.Figure()
                fig_margin.add_trace(go.Scatter(x=curr_l_t.index, y=curr_l_t['Maintenance_Margin'], mode='lines', name='當日維持率 (%)', line=dict(color='#F1C40F', width=2)))
                target_margin_show = ((1 + leverage_pct/100) / (leverage_pct/100) * 100)
                if enable_rebalance:
                    fig_margin.add_hline(y=target_margin_show, line_dash="dash", line_color="green", annotation_text=f"動態擴張線 ({target_margin_show:.0f}%)", annotation_position="top left")
                
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
