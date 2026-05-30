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
st.title("🛡️ 台灣高股息 ETF 買借死 (BBD) 質押模擬器")

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
# 2. 資料抓取引擎 (強制對齊時間軸)
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
        return "-", "-"
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
    if valid_divs == 0: return "-", "-"
    return f"{(success_count / len(divs)) * 100:.0f}%", f"{total_days / success_count:.0f}"

def get_portfolio_yield_cv(tickers, df_price, div_raw_dict, weights, leverage_pct, borrow_rate):
    """計算投資組合的綜合年化配息率與配息變異係數 (CV)"""
    base_yields = []
    port_annual_divs = pd.Series(dtype=float)
    
    for i, t in enumerate(tickers):
        divs = div_raw_dict.get(t, pd.Series(dtype=float))
        if not divs.empty:
            annual_divs = divs.groupby(divs.index.year).sum()
            y_yields = [d / df_price[t][df_price[t].index.year == y].mean() for y, d in annual_divs.items() if not df_price[t][df_price[t].index.year == y].empty]
            avg_yield = np.mean(y_yields) if y_yields else 0.0
            base_yields.append(avg_yield * weights[i])
            
            annual = divs.groupby(divs.index.year).sum() * weights[i]
            if port_annual_divs.empty:
                port_annual_divs = annual
            else:
                port_annual_divs = port_annual_divs.add(annual, fill_value=0)
        else:
            base_yields.append(0.0)
            
    port_base_yield = sum(base_yields)
    no_lev_str = f"{port_base_yield * 100:.2f}"
    
    lev_ratio = leverage_pct / 100.0
    lev_yield = port_base_yield * (1 + lev_ratio) - (borrow_rate * lev_ratio)
    lev_str = f"{lev_yield * 100:.2f}"
    
    cv_str = "-"
    if not port_annual_divs.empty and len(port_annual_divs[port_annual_divs > 0]) >= 2:
        valid = port_annual_divs[port_annual_divs > 0]
        cv_str = f"{valid.std() / valid.mean():.2f}"
        
    return no_lev_str, lev_str, cv_str

# ==========================================
# 3. 真實 BBD 提領模擬器
# ==========================================
def run_simulation(df_price, div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense, enable_rebalance):
    lev_ratio = leverage_pct / 100.0
    daily_borrow_rate = borrow_rate / 252
    
    initial_debt = initial_capital * lev_ratio
    total_initial_assets = initial_capital + initial_debt
    target_margin_ratio = (total_initial_assets / initial_debt) if initial_debt > 0 else float('inf')
    
    shares_theory = {t: (total_initial_assets * weights[i]) / df_price[t].iloc[0] for i, t in enumerate(df_price.columns)}
    debt_theory = initial_debt
    shares_real = {t: (total_initial_assets * weights[i]) / df_price[t].iloc[0] for i, t in enumerate(df_price.columns)}
    debt_real = initial_debt
    shares_static = shares_real.copy()
    
    yearly_expense_accrued = 0
    yearly_interest_accrued = 0
    yearly_div_accrued = 0
    
    trajectory = []
    current_year = -1
    tickers = df_price.columns
    
    for date, prices in df_price.iterrows():
        yearly_expense_accrued += annual_expense / 252
        yearly_interest_accrued += debt_real * daily_borrow_rate
        debt_theory += debt_theory * daily_borrow_rate
        
        for i, t in enumerate(tickers):
            divs = div_raw_dict.get(t, pd.Series(dtype=float))
            if date in divs.index:
                div_amount = divs.loc[date]
                if isinstance(div_amount, pd.Series): div_amount = div_amount.iloc[0]
                shares_theory[t] += (shares_theory[t] * div_amount) / prices[t]
                yearly_div_accrued += shares_real[t] * div_amount

        year = date.year
        if year != current_year:
            if current_year != -1:
                total_bill = yearly_expense_accrued + yearly_interest_accrued
                net_cash = yearly_div_accrued - total_bill
                
                if net_cash > 0:
                    for i, t in enumerate(tickers):
                        shares_real[t] += (net_cash * weights[i]) / prices[t]
                else:
                    shortfall = -net_cash
                    if leverage_pct > 0:
                        debt_real += shortfall
                    else:
                        for i, t in enumerate(tickers):
                            shares_real[t] -= (shortfall * weights[i]) / prices[t]

                if leverage_pct > 0 and enable_rebalance and target_margin_ratio != float('inf'):
                    val_real = sum([shares_real[t] * prices[t] for t in tickers])
                    if debt_real > 0:
                        current_margin = val_real / debt_real
                        if current_margin > target_margin_ratio:
                            delta_debt = (val_real - target_margin_ratio * debt_real) / (target_margin_ratio - 1)
                            if delta_debt > 0:
                                debt_real += delta_debt
                                for i, t in enumerate(tickers):
                                    shares_real[t] += (delta_debt * weights[i]) / prices[t]

                yearly_expense_accrued = 0
                yearly_interest_accrued = 0
                yearly_div_accrued = 0
                
            current_year = year

        val_theory = sum([shares_theory[t] * prices[t] for t in tickers])
        val_real = sum([shares_real[t] * prices[t] for t in tickers])
        val_static = sum([shares_static[t] * prices[t] for t in tickers])
        
        net_theory = val_theory - debt_theory
        net_real = val_real + yearly_div_accrued - debt_real - yearly_interest_accrued - yearly_expense_accrued
        net_static = val_static - initial_debt
        
        maintenance_margin = (val_real / debt_real * 100) if debt_real > 0 else float('inf')
        
        trajectory.append({
            'Date': date, 
            'Net_Theory': net_theory,
            'Net_Real': net_real,
            'Net_Static': net_static,
            'Maintenance_Margin': maintenance_margin
        })
        
    traj_df = pd.DataFrame(trajectory).set_index('Date')
    years = len(traj_df) / 252
    
    cagr_theory = (traj_df['Net_Theory'].iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 and traj_df['Net_Theory'].iloc[-1] > 0 else 0
    cagr_real = (traj_df['Net_Real'].iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 and traj_df['Net_Real'].iloc[-1] > 0 else 0
    cagr_static = (traj_df['Net_Static'].iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 and traj_df['Net_Static'].iloc[-1] > 0 else 0
    
    daily_ret_real = traj_df['Net_Real'].pct_change().dropna()
    volatility = daily_ret_real.std() * np.sqrt(252) if not daily_ret_real.empty else 0
    sharpe_ratio = (cagr_real - 0.015) / volatility if volatility != 0 else 0
    
    cum_real = traj_df['Net_Real'] / initial_capital
    running_max = cum_real.cummax()
    mdd = ((cum_real - running_max) / running_max).min() if not cum_real.empty else 0
    
    min_maintenance = traj_df['Maintenance_Margin'].min() if debt_real > 0 else float('inf')
    rebalance_status = "開啟" if enable_rebalance and leverage_pct > 0 else "-"
    
    final_net_real = traj_df['Net_Real'].iloc[-1]
    final_total_assets = final_net_real + debt_real
    
    return {
        "期末淨資產(萬)": f"{final_net_real / 10000:.2f}",
        "期末總資產(萬)": f"{final_total_assets / 10000:.2f}",
        "理論含息年化(%)": f"{cagr_theory * 100:.2f}",
        "真實淨資產年化(%)": f"{cagr_real * 100:.2f}",
        "單純價差年化(%)": f"{cagr_static * 100:.2f}",
        "最低維持率": f"{min_maintenance:.0f}%" if min_maintenance != float('inf') else "-",
        "年化波動率(%)": f"{volatility * 100:.2f}",
        "最大回撤(%)": f"{mdd * 100:.2f}",
        "夏普值": f"{sharpe_ratio:.2f}",
        "恆定維持率": rebalance_status,
        "質押標籤": "質押" if leverage_pct > 0 else "原型" 
    }, traj_df

# ==========================================
# 4. 側邊欄：參數設定
# ==========================================
with st.sidebar:
    st.header("💰 1. 資金與提領設定")
    initial_capital = st.number_input("初始本金 (元)", value=10000000, step=1000000)
    annual_expense = st.number_input("每年生活費需求 (元)", value=600000, step=10000)
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
    st.header("🔥 4. 質押槓桿與策略")
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
        enable_rebalance = st.checkbox("⚙️ 恆定維持率策略", value=False, help="每年底檢視：維持率超過設定值時，增加質押借款買入資產；低於設定值時，不做任何動作(絕不賣股)。")

# ==========================================
# 5. 主畫面運算與渲染
# ==========================================
if selected_names:
    if sum(weights) != 1.0:
        st.error(f"⚠️ 權重總和需為 100%，目前為 {sum(weights)*100:.0f}%")
    else:
        selected_tickers = [current_etf_dict[name] for name in selected_names]
        with st.spinner("啟動 Buy Borrow Die 提領與擴張模擬引擎..."):
            df_price, div_raw_dict = load_raw_data(selected_tickers, start_date, end_date)
            
        if not df_price.empty:
            actual_start = df_price.index[0].strftime('%Y-%m-%d')
            actual_end = df_price.index[-1].strftime('%Y-%m-%d')
            
            # 計算投資組合整體的 Yield 與 CV
            port_yield_no_lev, port_yield_lev, port_cv = get_portfolio_yield_cv(selected_tickers, df_price, div_raw_dict, weights, leverage_pct, borrow_rate)
            
            st.subheader("💾 命名與保存當前策略")
            col_name, col_btn = st.columns([3, 1])
            with col_name:
                custom_name = st.text_input("✏️ 為此投資組合命名", value="我的現金流組合", label_visibility="collapsed")
            with col_btn:
                if st.button("➕ 記錄至績效比較表", type="primary", use_container_width=True):
                    p_metrics, p_traj = run_simulation(df_price, div_raw_dict, weights, initial_capital, 0, borrow_rate, annual_expense, False)
                    p_metrics.update({"標的名稱": f"🎯 {custom_name} (無質押)", "質押": "0%", "年化配息率(%)": port_yield_no_lev, "填息成功率": "-", "平均填息天數": "-", "配息 CV": port_cv})
                    st.session_state.saved_portfolios.append({"metrics": p_metrics, "traj": p_traj})
                    
                    if leverage_pct > 0:
                        l_metrics, l_traj = run_simulation(df_price, div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense, enable_rebalance)
                        l_metrics.update({"標的名稱": f"🔥 {custom_name} (質押)", "質押": f"{leverage_pct}%", "年化配息率(%)": port_yield_lev, "填息成功率": "-", "平均填息天數": "-", "配息 CV": port_cv})
                        st.session_state.saved_portfolios.append({"metrics": l_metrics, "traj": l_traj})
            st.divider()

            st.subheader(f"📋 績效比較表 (同期基準: {actual_start} 至 {actual_end})")
            display_data = []
            
            # 單一 ETF
            for col in df_price.columns:
                etf_name = [k for k, v in current_etf_dict.items() if v == col][0]
                m, _ = run_simulation(df_price[[col]], div_raw_dict, [1.0], initial_capital, 0, borrow_rate, annual_expense, False)
                m.update({"標的名稱": etf_name, "質押": "0%", "恆定維持率": "-"})
                
                fill_rate, fill_days, cv_str, yield_str = "-", "-", "-", "-"
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
                
            # 歷史紀錄 (加入防呆清洗機制，避免舊的 nan 字串影響版面)
            for item in st.session_state.saved_portfolios:
                metrics_copy = item['metrics'].copy()
                for k, v in metrics_copy.items():
                    if str(v).lower() == 'nan':
                        metrics_copy[k] = "-"
                display_data.append(metrics_copy)
                
            # 當前預覽
            curr_p_m, curr_p_t = run_simulation(df_price, div_raw_dict, weights, initial_capital, 0, borrow_rate, annual_expense, False)
            curr_p_m.update({"標的名稱": f"👁️ 預覽: {custom_name} (無質押)", "質押": "0%", "年化配息率(%)": port_yield_no_lev, "填息成功率": "-", "平均填息天數": "-", "配息 CV": port_cv})
            display_data.append(curr_p_m)
            
            if leverage_pct > 0:
                curr_l_m, curr_l_t = run_simulation(df_price, div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense, enable_rebalance)
                curr_l_m.update({"標的名稱": f"👁️ 預覽: {custom_name} (質押)", "質押": f"{leverage_pct}%", "年化配息率(%)": port_yield_lev, "填息成功率": "-", "平均填息天數": "-", "配息 CV": port_cv})
                display_data.append(curr_l_m)
                
            ordered_cols = ["標的名稱", "質押", "恆定維持率", "最低維持率", "期末淨資產(萬)", "期末總資產(萬)", "理論含息年化(%)", "真實淨資產年化(%)", "單純價差年化(%)", "年化配息率(%)", "填息成功率", "平均填息天數", "年化波動率(%)", "最大回撤(%)", "夏普值", "配息 CV"]
            comparison_df = pd.DataFrame(display_data).set_index("標的名稱")
            comparison_df = comparison_df[[c for c in ordered_cols if c in comparison_df.columns]]
            
            TOOLTIPS = {
                "恆定維持率": "是否開啟策略：維持率超標時自動借款買入資產",
                "最低維持率": "評估抗斷頭能力。歷史回測中遭遇最差狀況時的維持率 (低於130%將斷頭)",
                "期末淨資產(萬)": "已扣除此期間全部生活費、質押利息與剩餘負債後的真實身價",
                "期末總資產(萬)": "期末淨資產 + 期末質押負債餘額 (亦已反映提領扣除)",
                "理論含息年化(%)": "假設不提領生活費，配息100%全額再投入的極限報酬率",
                "真實淨資產年化(%)": "BBD真實帳戶：扣生活費/利息，餘額買股、不足借款的淨身價成長率",
                "單純價差年化(%)": "基準線：假設股數永遠不變(配息剛好抵銷提領)，單純由股價上漲的報酬",
                "夏普值": "衡量承受每單位風險所獲得的超額報酬 (以真實軌跡計算)，越高越好",
                "配息 CV": "變異係數 (標準差÷平均值)。越接近0代表歷年配息金額越平穩"
            }
            
            def render_html_table(df):
                html = "<table style='width:100%; text-align:center; border-collapse: collapse; font-family: sans-serif; font-size: 0.85em;'>"
                html += "<tr style='background-color: #1E1E1E; border-bottom: 2px solid #444;'>"
                html += f"<th style='padding: 6px; text-align:left;'>標的名稱</th>"
                for col in df.columns:
                    tooltip = TOOLTIPS.get(col, "")
                    html += f"<th style='padding: 6px; text-align:center;' title='{tooltip}'>{col}</th>"
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
            
            with st.expander("📊 績效指標說明辭典 (點擊展開)", expanded=False):
                st.markdown("""
                * **期末淨資產 (萬)**：這段期間內付清所有生活費、繳完所有質押利息、並結清剩餘負債後，你口袋裡真正剩下的錢。
                * **期末總資產 (萬)**：期末淨資產 + 尚未還清的質押負債餘額（等同於你帳戶裡的股票總市值）。
                * **最低維持率**：遭遇歷史股災最差狀況時的維持率。小於 130% 代表策略失敗已遭券商斷頭。
                * **理論含息年化 (%)**：假設完全不需要提領生活費，配息 100% 瘋狂再投入的「烏托邦極限報酬率」。
                * **真實淨資產年化 (%)**：**核心指標！** BBD 帳戶真實運作：自動扣除生活費/利息，餘額買股、不足借款的真實身價成長率。
                * **單純價差年化 (%)**：對照組：假設股數永遠凍結不變，純粹靠底層資產漲跌的報酬率。若真實淨資產 > 單純價差，代表你的現金流為正向循環。
                * **夏普值**：衡量每單位風險的超額報酬 (CP值)，數值越高代表承擔相同波動能換取更多報酬。
                * **配息 CV**：配息變異係數 (標準差÷平均值)，越接近 0 代表配息金額越平穩，越適合做為現金流核心。
                """)

            if st.session_state.saved_portfolios:
                st.write("")
                with st.expander("🗑️ 管理與批次刪除歷史紀錄", expanded=False):
                    with st.form("batch_delete_form"):
                        st.write("請勾選您想移除的策略紀錄，支援多選批次刪除：")
                        del_cols = st.columns(2)
                        to_delete = []
                        for i, p in enumerate(st.session_state.saved_portfolios):
                            col = del_cols[i % 2]
                            if col.checkbox(f"第 {i+1} 筆: {p['metrics']['標的名稱']}", key=f"del_chk_{i}"):
                                to_delete.append(i)
                        
                        submit_del = st.form_submit_button("🗑️ 刪除已勾選紀錄", type="primary")
                        if submit_del and to_delete:
                            for idx in sorted(to_delete, reverse=True):
                                st.session_state.saved_portfolios.pop(idx)
                            st.rerun()

            st.caption("💡 **真實淨資產**為BBD運作結果：配息支付生活費與利息，餘額自動買入配息資產，不足額直接轉化為負債，絕不賣股。")
            st.divider()

            st.subheader("📈 競技場：所有儲存組合的真實淨資產比較")
            fig_traj = go.Figure()
            
            fig_traj.add_trace(go.Scatter(x=curr_p_t.index, y=curr_p_t['Net_Theory'], mode='lines', name='[參考] 理論含息 (無視提領/利息)', line=dict(color='#BDC3C7', width=1, dash='dot')))
            fig_traj.add_trace(go.Scatter(x=curr_p_t.index, y=curr_p_t['Net_Static'], mode='lines', name='[參考] 單純價差 (股數不變)', line=dict(color='#7F8C8D', width=1, dash='dash')))
            
            for item in st.session_state.saved_portfolios:
                name = item['metrics']['標的名稱']
                traj = item['traj']
                fig_traj.add_trace(go.Scatter(x=traj.index, y=traj['Net_Real'], mode='lines', name=name, line=dict(width=1.5)))

            fig_traj.add_trace(go.Scatter(x=curr_p_t.index, y=curr_p_t['Net_Real'], mode='lines', name=f'👁️預覽: {custom_name} (無質押)', line=dict(width=3, color='#2ECC71')))
            if leverage_pct > 0:
                rebalance_tag = " [恆定維持率]" if enable_rebalance else " [不還本]"
                fig_traj.add_trace(go.Scatter(x=curr_l_t.index, y=curr_l_t['Net_Real'], mode='lines', name=f'👁️預覽: {custom_name} (質押 {leverage_pct}%){rebalance_tag}', line=dict(width=3, color='#E74C3C')))
            
            fig_traj.update_layout(hovermode="x unified", yaxis_title="淨資產金額 (元)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_traj, use_container_width=True, key="traj_chart_main")
            
            st.subheader("🚨 質押維持率壓力監測競技場")
            fig_margin = go.Figure()
            has_margin_data = False
            
            for item in st.session_state.saved_portfolios:
                if item.get('metrics', {}).get('質押標籤', '原型') == "質押":
                    name = item['metrics']['標的名稱']
                    traj = item['traj']
                    plot_margin = traj['Maintenance_Margin'].replace([np.inf, -np.inf], 1000).clip(upper=1000)
                    fig_margin.add_trace(go.Scatter(x=traj.index, y=plot_margin, mode='lines', name=name, line=dict(width=1.5)))
                    has_margin_data = True

            if leverage_pct > 0:
                plot_margin = curr_l_t['Maintenance_Margin'].replace([np.inf, -np.inf], 1000).clip(upper=1000)
                fig_margin.add_trace(go.Scatter(x=curr_l_t.index, y=plot_margin, mode='lines', name=f'👁️預覽: {custom_name} (質押 {leverage_pct}%)', line=dict(width=3, color='#F1C40F')))
                has_margin_data = True
                
                target_margin_show = ((1 + leverage_pct/100) / (leverage_pct/100) * 100)
                if enable_rebalance:
                    fig_margin.add_hline(y=target_margin_show, line_dash="dash", line_color="green", annotation_text=f"預覽恆定目標線 ({target_margin_show:.0f}%)", annotation_position="top left")

            if has_margin_data:
                fig_margin.add_hline(y=166, line_dash="dash", line_color="orange", annotation_text="166% (追繳線)", annotation_position="bottom right")
                fig_margin.add_hline(y=130, line_dash="solid", line_color="red", annotation_text="130% (斷頭線)", annotation_position="bottom right")
                fig_margin.update_layout(hovermode="x unified", yaxis_title="維持率 (%)", yaxis=dict(range=[100, 1000]))
                st.plotly_chart(fig_margin, use_container_width=True, key="margin_chart_main")
            else:
                st.info("目前清單中沒有包含質押槓桿的策略，無維持率風險。")
                
        else:
            st.warning("資料獲取失敗，請確認時間區間內是否有足夠報價。")
else:
    st.info("請從左側欄位選擇至少一檔 ETF 來建立投資組合。")
