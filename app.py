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
st.title("🛡️ 台灣高股息 ETF 買借死 (BBD) 質押模擬器 V9.2")

with st.expander("📖 BBD 冬眠預備金動態防禦系統：五大運作鐵律 (點擊展開/收起)", expanded=True):
    st.markdown("""
    1. **開局儲備（冬眠機制）：** 系統開局會自動從總本金中鎖定 **N 年的生活費**作為「現金預備金」，此資金不投入股市、不計槓桿，專門用來擊碎退休前幾年大盤崩盤的「報酬順序風險 (SORR)」。
    2. **剩餘資金入市與擴張：** 扣除預備金後的賸餘資金才進入股市建立核心資產，並依據指定的「目標初始維持率」精準推算質押成數，借款放大資產規模。
    3. **股息自動回補幫浦：** 年底結算時，若「總股息收入」大於「生活費＋利息」，多出來的錢**優先存回現金預備金池**。直到預備金補滿水位，剩餘資金才啟動「再投入買股」進行資產複利。
    4. **股災期間絕對不賣股：** 股息不夠付開銷時，**嚴禁在低檔變賣股票**。系統會優先抽血「現金預備金」補缺口；若預備金燒光，則自動動用剩餘的質押借款額度（以債養債）度過寒冬。
    5. **嚴格同期對齊與防破產：** 所有資產強制採用無情交集對齊（以最晚上市的 ETF 為起跑線），杜絕假數據。一旦真實淨資產（股票市值＋預備金－質押負債）小於等於 0，系統立刻宣判破產。
    """)

if 'saved_portfolios' not in st.session_state:
    st.session_state.saved_portfolios = []
elif len(st.session_state.saved_portfolios) > 0 and 'tickers' not in st.session_state.saved_portfolios[0]:
    st.session_state.saved_portfolios = []
if 'custom_etfs' not in st.session_state:
    st.session_state.custom_etfs = {}

DEFAULT_ETF_DICT = {
    "0050 元大台灣50": "0050.TW",
    "0056 元大高股息": "0056.TW",
    "006208 富邦台50": "006208.TW",
    "00878 國泰永續高股息": "00878.TW",
    "00919 群益台灣精選高息": "00919.TW",
    "00929 復華台灣科技優息": "00929.TW",
    "00713 元大台灣高息低波": "00713.TW"
}

# ==========================================
# 2. 資料抓取與嚴格對齊引擎
# ==========================================
@st.cache_data(ttl=3600)
def load_raw_data(tickers, start_date, end_date):
    if not tickers: return pd.DataFrame(), {}
    
    fetch_list = list(set(tickers + ['^TWII']))
    raw_prices = {}
    div_raw_dict = {}
    
    for ticker in fetch_list:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if 'Close' in df.columns:
                    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                    raw_prices[ticker] = df['Close']
            
            if ticker != '^TWII':
                tk = yf.Ticker(ticker)
                divs = tk.dividends
                if not divs.empty:
                    divs.index = pd.to_datetime(divs.index).tz_localize(None).normalize()
                    div_raw_dict[ticker] = divs[(divs.index >= pd.Timestamp(start_date)) & (divs.index <= pd.Timestamp(end_date))]
        except Exception:
            continue
            
    aligned_df = pd.DataFrame(raw_prices).ffill().dropna()
    return aligned_df, div_raw_dict

def get_beta_and_market_mdd(tickers, weights, df_price, market_ticker='^TWII'):
    if market_ticker not in df_price.columns: return 1.0, 0.0
    valid_tickers = [t for t in tickers if t in df_price.columns]
    if not valid_tickers: return 1.0, 0.0
        
    aligned_df = df_price[valid_tickers + [market_ticker]].dropna()
    if aligned_df.empty or len(aligned_df) < 2: return 1.0, 0.0
        
    returns = aligned_df.pct_change().dropna()
    
    adj_weights = []
    for i, t in enumerate(tickers):
        if t in valid_tickers: adj_weights.append(weights[i])
    sum_w = sum(adj_weights)
    if sum_w == 0: return 1.0, 0.0
    adj_weights = [w/sum_w for w in adj_weights]
            
    port_returns = returns[valid_tickers].dot(adj_weights)
    market_returns = returns[market_ticker]
    
    cov_matrix = np.cov(port_returns, market_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1.0
    
    cum_market = (1 + market_returns).cumprod()
    running_max = cum_market.cummax()
    market_mdd = ((cum_market - running_max) / running_max).min()
    
    return beta, market_mdd

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
    base_yields = []
    port_annual_divs = pd.Series(dtype=float)
    
    for i, t in enumerate(tickers):
        divs = div_raw_dict.get(t, pd.Series(dtype=float))
        if not divs.empty:
            annual_divs = divs.groupby(divs.index.year).sum()
            if t in df_price.columns:
                y_yields = [d / df_price[t][df_price[t].index.year == y].mean() for y, d in annual_divs.items() if not df_price[t][df_price[t].index.year == y].empty]
                avg_yield = np.mean(y_yields) if y_yields else 0.0
                base_yields.append(avg_yield * weights[i])
            else:
                base_yields.append(0.0)
            annual = divs.groupby(divs.index.year).sum() * weights[i]
            if port_annual_divs.empty: port_annual_divs = annual
            else: port_annual_divs = port_annual_divs.add(annual, fill_value=0)
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
        
    return no_lev_str, lev_str, cv_str, port_base_yield

def generate_weight_combinations(num_assets, step_pct=10):
    step = step_pct / 100.0
    res = []
    for i in range(num_assets):
        w = [0.0] * num_assets
        w[i] = 1.0
        res.append(w)
    if num_assets > 1:
        def helper(n, target):
            if n == 1: return [[target]]
            temp_res = []
            for i in range(int(round(target/step)) + 1):
                val = round(i * step, 2)
                for rest in helper(n-1, round(target - val, 2)):
                    temp_res.append([val] + rest)
            return temp_res
        combinations = helper(num_assets, 1.0)
        for combo in combinations:
            if combo not in res: res.append(combo)
    return res

# ==========================================
# 3. 真實 BBD 提領模擬器
# ==========================================
def run_simulation(df_price, div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense, enable_rebalance, target_margin_input=None, reserve_years=2):
    lev_ratio = leverage_pct / 100.0
    daily_borrow_rate = borrow_rate / 252
    
    cash_reserve_target = annual_expense * reserve_years
    initial_cash = cash_reserve_target if initial_capital > cash_reserve_target else 0
    investable_capital = initial_capital - initial_cash
    
    initial_debt = investable_capital * lev_ratio
    total_initial_assets = investable_capital + initial_debt
    
    if target_margin_input is not None and enable_rebalance:
        target_margin_ratio = target_margin_input / 100.0
    else:
        target_margin_ratio = (total_initial_assets / initial_debt) if initial_debt > 0 else float('inf')
    
    tickers = [c for c in df_price.columns if c != '^TWII']
    adj_weights = []
    for i, t in enumerate(df_price.columns):
         if t != '^TWII': adj_weights.append(weights[i] if i < len(weights) else 0)
    
    shares_theory = {t: (initial_capital * adj_weights[i]) / df_price[t].iloc[0] for i, t in enumerate(tickers)}
    
    shares_real = {t: (total_initial_assets * adj_weights[i]) / df_price[t].iloc[0] for i, t in enumerate(tickers)}
    debt_real = initial_debt
    cash_real = initial_cash
    
    shares_static = shares_real.copy()
    
    yearly_expense_accrued = 0
    yearly_interest_accrued = 0
    yearly_div_accrued = 0
    
    trajectory = []
    current_year = -1
    is_bankrupt = False 
    
    for date, prices in df_price.iterrows():
        if is_bankrupt:
            val_theory = sum([shares_theory[t] * prices[t] for t in tickers])
            val_static = sum([shares_static[t] * prices[t] for t in tickers])
            for t in tickers:
                divs = div_raw_dict.get(t, pd.Series(dtype=float))
                if date in divs.index:
                    div_amount = divs.loc[date]
                    if isinstance(div_amount, pd.Series): div_amount = div_amount.iloc[0]
                    shares_theory[t] += (shares_theory[t] * div_amount) / prices[t]
            trajectory.append({
                'Date': date, 'Net_Theory': val_theory, 'Net_Real': 0.0,
                'Net_Static': val_static + initial_cash - initial_debt, 'Maintenance_Margin': float('inf') if leverage_pct == 0 else 0.0
            })
            continue

        yearly_expense_accrued += annual_expense / 252
        yearly_interest_accrued += debt_real * daily_borrow_rate
        
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
                    deficit = cash_reserve_target - cash_real
                    if deficit > 0:
                        fill = min(net_cash, deficit)
                        cash_real += fill
                        net_cash -= fill
                    if net_cash > 0:
                        for i, t in enumerate(tickers):
                            shares_real[t] += (net_cash * adj_weights[i]) / prices[t]
                else:
                    shortfall = -net_cash
                    draw = min(shortfall, cash_real)
                    cash_real -= draw
                    shortfall -= draw
                    
                    if shortfall > 0:
                        if leverage_pct > 0:
                            debt_real += shortfall
                        else:
                            for i, t in enumerate(tickers):
                                sell_shares = (shortfall * adj_weights[i]) / prices[t]
                                shares_real[t] -= sell_shares

                if leverage_pct > 0 and enable_rebalance and target_margin_ratio != float('inf'):
                    val_real = sum([shares_real[t] * prices[t] for t in tickers])
                    if debt_real > 0:
                        current_margin = val_real / debt_real
                        if current_margin > target_margin_ratio:
                            delta_debt = (val_real - target_margin_ratio * debt_real) / (target_margin_ratio - 1)
                            if delta_debt > 0:
                                debt_real += delta_debt
                                for i, t in enumerate(tickers):
                                    shares_real[t] += (delta_debt * adj_weights[i]) / prices[t]

                yearly_expense_accrued = 0
                yearly_interest_accrued = 0
                yearly_div_accrued = 0
                
            current_year = year

        val_theory = sum([shares_theory[t] * prices[t] for t in tickers])
        val_real = sum([shares_real[t] * prices[t] for t in tickers])
        val_static = sum([shares_static[t] * prices[t] for t in tickers])
        
        net_theory = val_theory
        net_real = val_real + cash_real + yearly_div_accrued - debt_real - yearly_interest_accrued - yearly_expense_accrued
        net_static = val_static + initial_cash - initial_debt
        
        if net_real <= 0:
            is_bankrupt = True
            net_real = 0.0
            for t in tickers: shares_real[t] = 0 
            debt_real = 0
            cash_real = 0
            
        maintenance_margin = (val_real / debt_real * 100) if debt_real > 0 else float('inf')
        trajectory.append({'Date': date, 'Net_Theory': net_theory, 'Net_Real': net_real, 'Net_Static': net_static, 'Maintenance_Margin': maintenance_margin})
        
    traj_df = pd.DataFrame(trajectory).set_index('Date')
    years = len(traj_df) / 252
    
    cagr_theory = (traj_df['Net_Theory'].iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 and traj_df['Net_Theory'].iloc[-1] > 0 else 0
    cagr_static = (traj_df['Net_Static'].iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 and traj_df['Net_Static'].iloc[-1] > 0 else 0
    
    if is_bankrupt:
        cagr_real = -1.0 
        sharpe_real = -9.99
        mdd = -1.0
        min_maintenance = 0.0
        final_net_real_str = "💀 破產歸零"
        final_total_assets_str = "-"
    else:
        cagr_real = (traj_df['Net_Real'].iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 and traj_df['Net_Real'].iloc[-1] > 0 else 0
        daily_ret_real = traj_df['Net_Real'].pct_change().dropna()
        vol_real = daily_ret_real.std() * np.sqrt(252) if not daily_ret_real.empty else 0
        sharpe_real = (cagr_real - 0.015) / vol_real if vol_real != 0 else 0
        cum_real = traj_df['Net_Real'] / initial_capital
        running_max = cum_real.cummax()
        mdd = ((cum_real - running_max) / running_max).min() if not cum_real.empty else 0
        min_maintenance = traj_df['Maintenance_Margin'].min() if leverage_pct > 0 else float('inf')
        final_net_real_str = f"{traj_df['Net_Real'].iloc[-1] / 10000:.2f}"
        final_total_assets_str = f"{(traj_df['Net_Real'].iloc[-1] + debt_real) / 10000:.2f}"

    daily_ret_theory = traj_df['Net_Theory'].pct_change().dropna()
    vol_theory = daily_ret_theory.std() * np.sqrt(252) if not daily_ret_theory.empty else 0
    sharpe_theory = (cagr_theory - 0.015) / vol_theory if vol_theory != 0 else 0
    
    rebalance_status = "開啟" if enable_rebalance and leverage_pct > 0 else "-"
    
    raw_metrics = {
        "final_net_real": traj_df['Net_Real'].iloc[-1] if not is_bankrupt else 0,
        "cagr_real": cagr_real,
        "mdd": mdd,
        "sharpe_real": sharpe_real,
        "min_maintenance": min_maintenance,
        "is_bankrupt": is_bankrupt
    }
    
    return {
        "期末淨資產(萬)": final_net_real_str,
        "期末總資產(萬)": final_total_assets_str,
        "理論含息年化(%)": f"{cagr_theory * 100:.2f}",
        "真實淨資產年化(%)": f"{cagr_real * 100:.2f}" if not is_bankrupt else "破產",
        "單純價差年化(%)": f"{cagr_static * 100:.2f}",
        "最低維持率": f"{min_maintenance:.0f}%" if leverage_pct > 0 and not is_bankrupt else ("-" if leverage_pct == 0 else "斷頭"),
        "年化波動率(%)": f"{vol_real * 100:.2f}" if not is_bankrupt else "-",
        "最大回撤(%)": f"{mdd * 100:.2f}" if not is_bankrupt else "-100%",
        "理論夏普值": f"{sharpe_theory:.2f}",
        "真實夏普值": f"{sharpe_real:.2f}" if not is_bankrupt else "-",
        "恆定維持率": rebalance_status,
        "質押標籤": "質押" if leverage_pct > 0 else "原型" 
    }, traj_df, raw_metrics

# ==========================================
# 4. 側邊欄：參數動態連動
# ==========================================
with st.sidebar:
    st.header("💰 1. 資金與提領設定 (全局連動)")
    initial_capital = st.number_input("初始本金 (元)", value=8000000, step=1000000)
    annual_expense = st.number_input("每年生活費需求 (元)", value=600000, step=10000)
    reserve_years = st.number_input("冬眠現金預備金 (年)", value=2, min_value=0, max_value=10, help="保留 N 年的生活費作為現金緩衝，不投入股市。遇到股災時優先扣除此現金，絕對避免在低檔賣股。")
    st.divider()
    
    st.header("🕒 2. 回測時間區間")
    default_start = datetime(2011, 1, 1)
    start_date = st.date_input("開始日期", value=default_start)
    end_date = st.date_input("結束日期", value=datetime.today())
    st.divider()
    
    st.header("⚖️ 3. 資產與權重")
    current_etf_dict = {**DEFAULT_ETF_DICT, **st.session_state.custom_etfs}
    
    with st.expander("➕ 新增自訂 ETF", expanded=False):
        new_etf_code = st.text_input("輸入台股代碼 (例: 006208)", help="輸入純代碼即可，系統會嘗試自動抓取名稱。")
        new_etf_name_override = st.text_input("自訂顯示名稱 (選填)")
        if st.button("手動/自動新增標的"):
            if new_etf_code:
                code_clean = new_etf_code.strip()
                ticker_symbol = f"{code_clean}.TW" if code_clean.isdigit() else code_clean
                display_name = new_etf_name_override if new_etf_name_override else code_clean
                
                if not new_etf_name_override:
                    with st.spinner("連線交易所嘗試抓取資料中..."):
                        try:
                            tk = yf.Ticker(ticker_symbol)
                            name = tk.info.get('shortName', '')
                            if name: display_name = f"{code_clean} {name}"
                        except Exception:
                            pass
                st.session_state.custom_etfs[display_name] = ticker_symbol
                st.rerun()
                
    # 預設連動 00713 / 0056 / 00878
    selected_names = st.multiselect("選擇組成 ETF", list(current_etf_dict.keys()), default=["00713 元大台灣高息低波", "0056 元大高股息", "00878 國泰永續高股息"])
    weights = []
    if selected_names:
        default_w = 100 // len(selected_names)
        for i, name in enumerate(selected_names):
            val = default_w + (100 % len(selected_names) if i == len(selected_names)-1 else 0)
            w = st.number_input(f"{name[:5]} 權重 (%)", min_value=0, max_value=100, value=val)
            weights.append(w / 100)
            
    st.divider()
    st.header("🔥 4. 質押槓桿與策略")
    leverage_mode = st.radio("設定方式", ["依借款比例", "依目標維持率"], index=1)
    if leverage_mode == "依借款比例":
        leverage_pct = st.slider("質押借款比例 (%)", 0, 100, 20)
        target_margin = ((1 + leverage_pct/100) / (leverage_pct/100) * 100) if leverage_pct > 0 else float('inf')
    else:
        target_margin = st.number_input("目標初始維持率 (%)", min_value=130, max_value=1000, value=450, step=10)
        leverage_pct = round(100 / (target_margin/100 - 1)) if target_margin > 100 else 0
        st.caption(f"推算需借款比例約為: {leverage_pct}%")
    borrow_rate = st.number_input("借款年利率 (%)", value=2.5, step=0.1) / 100.0
    
    enable_rebalance = st.checkbox("⚙️ 恆定維持率策略", value=True, help="每年底檢視：維持率超過設定值時，增加質押借款買入資產；低於設定值時，不做任何動作(絕不賣股)。")

    st.divider()
    st.header("🎯 5. AI 尋優防禦底線設定")
    ai_min_margin = st.number_input("股災最低容許維持率 (%)", min_value=140, max_value=800, value=350, step=10, help="歷史回測中若維持率跌破此數值，AI 將淘汰該策略。建議 >300% 以保證安穩好眠。")

# ==========================================
# 5. 主畫面運算與渲染
# ==========================================
if selected_names:
    if sum(weights) != 1.0:
        st.error(f"⚠️ 權重總和需為 100%，目前為 {sum(weights)*100:.0f}%")
    else:
        all_required_tickers = set(current_etf_dict[name] for name in selected_names)
        for strat in st.session_state.saved_portfolios:
            all_required_tickers.update(strat['tickers'])
            
        with st.spinner("啟動嚴格同期交集對齊與冬眠預備金引擎..."):
            df_price, div_raw_dict = load_raw_data(list(all_required_tickers), start_date, end_date)
            
        if not df_price.empty:
            actual_start = df_price.index[0].strftime('%Y-%m-%d')
            actual_end = df_price.index[-1].strftime('%Y-%m-%d')
            
            selected_tickers = [current_etf_dict[name] for name in selected_names]
            port_beta, market_mdd = get_beta_and_market_mdd(selected_tickers, weights, df_price)
            port_yield_no_lev, port_yield_lev, port_cv, raw_port_yield = get_portfolio_yield_cv(selected_tickers, df_price, div_raw_dict, weights, leverage_pct, borrow_rate)
            
            with st.sidebar:
                st.divider()
                st.header("🛡️ 戰略評估與大盤連動試算")
                
                cash_reserve_amt = annual_expense * reserve_years
                investable_amt = initial_capital - cash_reserve_amt if initial_capital > cash_reserve_amt else 0
                
                if leverage_pct > 0:
                    debt_amt = investable_amt * (leverage_pct / 100)
                    total_amt = investable_amt + debt_amt
                    
                    drop_to_166 = (1 - (1.66 * debt_amt) / total_amt) * 100 if total_amt > 0 else 0
                    drop_to_130 = (1 - (1.30 * debt_amt) / total_amt) * 100 if total_amt > 0 else 0
                    
                    annual_interest = debt_amt * borrow_rate
                    total_liability = annual_expense + annual_interest
                    breakeven_yield = (total_liability / total_amt) * 100 if total_amt > 0 else 0
                    
                    market_drop_130 = drop_to_130 / port_beta if port_beta > 0 else 0
                    
                    st.info(f"""
                    **🚨 組合斷頭防禦力 (組合 Beta: {port_beta:.2f})**
                    * 組合淨值跌 **-{drop_to_166:.1f}%** 面臨追繳 (166%)
                    * 組合淨值跌 **-{drop_to_130:.1f}%** 面臨斷頭 (130%)
                    
                    *(根據您的 ETF 歷史防禦力換算，**台股大盤需崩跌約 -{market_drop_130:.1f}%** 才會讓您的帳戶觸及斷頭)*
                    
                    **⚖️ 損平殖利率安全檢測**
                    * 扣除預備金後投入股市本金：{investable_amt/10000:.0f} 萬
                    * 現金預備金池水位：{cash_reserve_amt/10000:.0f} 萬
                    * 每年提領與利息總需：{(total_liability)/10000:.1f} 萬
                    * 總資產需有 **{breakeven_yield:.2f}%** 殖利率才能不扣預備金。
                    """)
                else:
                    st.info(f"無質押，無斷頭風險。\n* 扣除預備金後投入本金：{investable_amt/10000:.0f} 萬\n* 目前組合 Beta 值：{port_beta:.2f} (大盤連動度)")

            st.subheader(f"📊 預估首年現金流健康度 (以歷史平均殖利率估算)")
            if leverage_pct > 0:
                est_div = (investable_amt * (1 + leverage_pct/100)) * raw_port_yield
                est_interest = (investable_amt * (leverage_pct/100)) * borrow_rate
                net_cash = est_div - est_interest - annual_expense
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("預估總股息收入", f"{est_div/10000:.1f} 萬")
                col2.metric("生活費提領", f"-{annual_expense/10000:.1f} 萬")
                col3.metric("預估利息支出", f"-{est_interest/10000:.1f} 萬")
                
                if net_cash >= 0:
                    col4.metric("💰 年底可擴張買股餘額", f"+{net_cash/10000:.1f} 萬", "正向循環")
                else:
                    col4.metric("🩸 年底需動用現金預備金", f"{net_cash/10000:.1f} 萬", "消耗預備金", delta_color="inverse")
            else:
                est_div = investable_amt * raw_port_yield
                net_cash = est_div - annual_expense
                
                col1, col2, col3 = st.columns(3)
                col1.metric("預估總股息收入", f"{est_div/10000:.1f} 萬")
                col2.metric("生活費提領", f"-{annual_expense/10000:.1f} 萬")
                if net_cash >= 0:
                    col3.metric("💰 年底可買股餘額", f"+{net_cash/10000:.1f} 萬", "正向循環")
                else:
                    col3.metric("🩸 年底需動用現金預備金", f"{net_cash/10000:.1f} 萬", "消耗預備金", delta_color="inverse")
            st.divider()

            st.subheader("💾 命名與保存當前策略")
            col_name, col_btn = st.columns([3, 1])
            with col_name:
                custom_name = st.text_input("✏️ 為此投資組合命名", value="我的現金流組合", label_visibility="collapsed")
            with col_btn:
                if st.button("➕ 記錄至績效比較表", type="primary", use_container_width=True):
                    st.session_state.saved_portfolios.append({
                        "name": f"🎯 {custom_name} (無質押)", "tickers": selected_tickers, "weights": weights, "leverage_pct": 0, "enable_rebalance": False
                    })
                    if leverage_pct > 0:
                        st.session_state.saved_portfolios.append({
                            "name": f"🔥 {custom_name} (質押)", "tickers": selected_tickers, "weights": weights, "leverage_pct": leverage_pct, "enable_rebalance": enable_rebalance
                        })
                    st.rerun() 
            st.divider()

            st.subheader(f"📋 全境動態績效比較表 (嚴格同期基準: {actual_start} 至 {actual_end})")
            display_data = []
            
            for col in selected_tickers:
                if col in df_price.columns:
                    etf_name = [k for k, v in current_etf_dict.items() if v == col][0]
                    m, _, _ = run_simulation(df_price[[col]], div_raw_dict, [1.0], initial_capital, 0, borrow_rate, annual_expense, False, None, reserve_years)
                    s_beta, _ = get_beta_and_market_mdd([col], [1.0], df_price)
                    
                    m.update({"標的名稱": etf_name, "質押": "0%", "組合 Beta": f"{s_beta:.2f}", "恆定維持率": "-"})
                    
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
                
            for strat in st.session_state.saved_portfolios:
                s_tickers = strat['tickers']
                s_weights = strat['weights']
                if all(t in df_price.columns for t in s_tickers):
                    m, t_df, _ = run_simulation(
                        df_price[s_tickers], div_raw_dict, s_weights, initial_capital, strat['leverage_pct'], borrow_rate, annual_expense, strat['enable_rebalance'], None, reserve_years
                    )
                    y_no_lev, y_lev, cv, _ = get_portfolio_yield_cv(s_tickers, df_price, div_raw_dict, s_weights, strat['leverage_pct'], borrow_rate)
                    s_beta, _ = get_beta_and_market_mdd(s_tickers, s_weights, df_price)
                    m.update({
                        "標的名稱": strat['name'], "質押": f"{strat['leverage_pct']}%" if strat['leverage_pct'] > 0 else "0%", "組合 Beta": f"{s_beta:.2f}",
                        "年化配息率(%)": y_lev if strat['leverage_pct'] > 0 else y_no_lev, "填息成功率": "-", "平均填息天數": "-", "配息 CV": cv
                    })
                    display_data.append(m)
                    strat['computed_traj'] = t_df 
                
            curr_p_m, curr_p_t, _ = run_simulation(df_price[selected_tickers], div_raw_dict, weights, initial_capital, 0, borrow_rate, annual_expense, False, None, reserve_years)
            curr_p_m.update({"標的名稱": f"👁️ 預覽: {custom_name} (無質押)", "質押": "0%", "組合 Beta": f"{port_beta:.2f}", "恆定維持率": "-", "年化配息率(%)": port_yield_no_lev, "填息成功率": "-", "平均填息天數": "-", "配息 CV": port_cv})
            display_data.append(curr_p_m)
            
            if leverage_pct > 0:
                curr_l_m, curr_l_t, _ = run_simulation(df_price[selected_tickers], div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense, enable_rebalance, target_margin, reserve_years)
                curr_l_m.update({"標的名稱": f"👁️ 預覽: {custom_name} (質押)", "質押": f"{leverage_pct}%", "組合 Beta": f"{port_beta:.2f}", "恆定維持率": "開啟" if enable_rebalance else "關閉", "年化配息率(%)": port_yield_lev, "填息成功率": "-", "平均填息天數": "-", "配息 CV": port_cv})
                display_data.append(curr_l_m)
                
            ordered_cols = ["標的名稱", "質押", "組合 Beta", "恆定維持率", "最低維持率", "期末淨資產(萬)", "期末總資產(萬)", "理論含息年化(%)", "真實淨資產年化(%)", "單純價差年化(%)", "年化配息率(%)", "填息成功率", "平均填息天數", "年化波動率(%)", "最大回撤(%)", "理論夏普值", "真實夏普值", "配息 CV"]
            comparison_df = pd.DataFrame(display_data).set_index("標的名稱")
            comparison_df = comparison_df[[c for c in ordered_cols if c in comparison_df.columns]]
            
            def render_html_table(df):
                html = "<table style='width:100%; text-align:center; border-collapse: collapse; font-family: sans-serif; font-size: 0.85em;'>"
                html += "<tr style='background-color: #1E1E1E; border-bottom: 2px solid #444;'>"
                html += f"<th style='padding: 6px; text-align:left;'>標的名稱</th>"
                for col in df.columns: html += f"<th style='padding: 6px; text-align:center;'>{col}</th>"
                html += "</tr>"
                for index, row in df.iterrows():
                    bg_color, font_weight, color = "transparent", "normal", "#E0E0E0"
                    is_dead = "💀 破產" in str(row['期末淨資產(萬)'])
                    if is_dead: color = "#E74C3C" 
                    if str(index).startswith("👁️ 預覽"):
                        bg_color, font_weight = "#117A65", "bold"
                        if not is_dead: color = "white"
                    elif "🔥" in str(index) or "🎯" in str(index):
                        bg_color = "#2C3E50"
                        if not is_dead: color = "#D5D8DC"
                    html += f"<tr style='background-color: {bg_color}; border-bottom: 1px solid #333;'>"
                    html += f"<td style='padding: 6px; text-align:left; color:{color}; font-weight:{font_weight};'>{index}</td>"
                    for item in row:
                        val_color = color
                        if is_dead and item != "💀 破產歸零": val_color = "#922B21" 
                        html += f"<td style='padding: 6px; text-align:center; color:{val_color}; font-weight:{font_weight};'>{item}</td>"
                    html += "</tr>"
                html += "</table>"
                return html

            st.markdown(render_html_table(comparison_df), unsafe_allow_html=True)
            st.divider()

            # ==========================================
            # 6. 圖表與 AI 智慧動態尋優
            # ==========================================
            st.subheader("📈 競技場：所有儲存組合的真實淨資產比較")
            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(x=curr_p_t.index, y=curr_p_t['Net_Theory'], mode='lines', name='[參考] 當前組合理論含息', line=dict(color='#BDC3C7', width=1, dash='dot')))
            for strat in st.session_state.saved_portfolios:
                if 'computed_traj' in strat:
                    fig_traj.add_trace(go.Scatter(x=strat['computed_traj'].index, y=strat['computed_traj']['Net_Real'], mode='lines', name=strat['name']))
            fig_traj.add_trace(go.Scatter(x=curr_p_t.index, y=curr_p_t['Net_Real'], mode='lines', name=f'👁️預覽: {custom_name} (無質押)', line=dict(width=3, color='#2ECC71')))
            if leverage_pct > 0:
                fig_traj.add_trace(go.Scatter(x=curr_l_t.index, y=curr_l_t['Net_Real'], mode='lines', name=f'👁️預覽: {custom_name} (質押 {leverage_pct}%)', line=dict(width=3, color='#E74C3C')))
            st.plotly_chart(fig_traj, use_container_width=True)

            st.subheader("🚨 質押維持率壓力監測競技場")
            fig_margin = go.Figure()
            has_margin_data = False
            
            for strat in st.session_state.saved_portfolios:
                if strat.get('leverage_pct', 0) > 0 and 'computed_traj' in strat:
                    name = strat['name']
                    traj = strat['computed_traj']
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

            st.divider()
            st.subheader("🎯 系統判斷與最佳化配比建議 (AI 動態尋優)")
            if st.button("🚀 啟動 AI 智慧尋優矩陣運算", type="primary", use_container_width=True):
                with st.spinner(f"AI 正在背景進行動態存活測試 (防禦底線：{ai_min_margin}%)..."):
                    weight_grids = generate_weight_combinations(len(selected_tickers), 10)
                    best_sharpe_list, best_mdd_list, best_net_list = [], [], []
                    test_margins = [200, 250, 300, 350, 400, 450, 500, 600, 800]
                    valid_margins = [m for m in test_margins if m >= ai_min_margin]
                    test_levs = [100 / (m/100 - 1) for m in valid_margins] + [0]
                    
                    for w in weight_grids:
                        grid_beta, _ = get_beta_and_market_mdd(selected_tickers, w, df_price)
                        _, _, _, y_raw = get_portfolio_yield_cv(selected_tickers, df_price, div_raw_dict, w, 0, borrow_rate)
                        for lev in set(test_levs):
                            margin_target = ((1 + lev/100) / (lev/100)) * 100 if lev > 0 else float('inf')
                            _, _, raw = run_simulation(df_price[selected_tickers], div_raw_dict, w, initial_capital, lev, borrow_rate, annual_expense, True, margin_target, reserve_years)
                            
                            if raw['min_maintenance'] >= ai_min_margin and not raw['is_bankrupt']:
                                w_str = " + ".join([f"{name[:5]} {w[i]*100:.0f}%" for i, name in enumerate(selected_names) if w[i] > 0])
                                invest_part = initial_capital - (annual_expense * reserve_years)
                                net_cf = (invest_part * (1 + lev/100)) * y_raw - (invest_part * (lev/100)) * borrow_rate - annual_expense
                                
                                # ✨ 修復 inf% 顯示問題
                                min_margin_str = "無質押" if raw['min_maintenance'] == float('inf') else f"{raw['min_maintenance']:.0f}%"
                                margin_display = f"{margin_target:.0f}%" if margin_target != float('inf') else "無質押"
                                
                                result = {
                                    "w_str": w_str, "margin_display": margin_display,
                                    "sharpe": raw['sharpe_real'], "mdd": raw['mdd'], "net": raw['final_net_real'], "cagr": raw['cagr_real'],
                                    "min_margin_str": min_margin_str, "beta": grid_beta, "net_cf": net_cf
                                }
                                best_sharpe_list.append(result)
                                best_mdd_list.append(result)
                                best_net_list.append(result)
                                
                    best_sharpe_list.sort(key=lambda x: x['sharpe'], reverse=True)
                    best_mdd_list.sort(key=lambda x: x['mdd'], reverse=True)
                    best_net_list.sort(key=lambda x: x['net'], reverse=True)
                    
                    if best_sharpe_list:
                        col_opt1, col_opt2, col_opt3 = st.columns(3)
                        for col, data, title, color in zip([col_opt1, col_opt2, col_opt3], [best_sharpe_list, best_mdd_list, best_net_list], ["🥇 綜合王者", "🛡️ 絕對防禦", "🚀 暴力擴張"], ["#5DADE2", "#F4D03F", "#EC7063"]):
                            with col:
                                st.info(f"#### {title}")
                                for i, res in enumerate(data[:3]):
                                    rank = ["❶ 冠軍", "❷ 亞軍", "❸ 季軍"][i]
                                    st.markdown(f"""
                                    <div style="font-size: 1.25em; line-height: 1.8; padding-bottom: 12px; margin-bottom: 12px; border-bottom: 1px solid #333;">
                                        <span style="font-weight: bold; color: {color};">{rank}：{res['w_str']}</span><br>
                                        • <b>策略：</b> 恆定維持率 {res['margin_display']}<br>
                                        • <b>真實夏普：</b> {res['sharpe']:.2f} / <b>最大回撤：</b> {res['mdd']*100:.2f}%<br>
                                        • 🛡️ <b>最低維持率：</b> {res['min_margin_str']}
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("在當前防禦底線下，無符合條件的存活組合。")
        else:
            st.warning("資料獲取失敗。")
else:
    st.info("請選擇至少一檔 ETF。")
