import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. 網頁設定
# ==========================================
st.set_page_config(page_title="高股息策略與質押模擬器", layout="wide")
st.title("🛡️ 台灣高股息 ETF 買借死 (BBD) 質押模擬器 V8.0")

# ==========================================
# 2. 嚴格對齊資料引擎
# ==========================================
@st.cache_data(ttl=3600)
def load_raw_data(tickers, start_date, end_date):
    fetch_list = list(set(tickers + ['^TWII']))
    raw_prices = {}
    div_raw_dict = {}
    for ticker in fetch_list:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not df.empty and 'Close' in df.columns:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                raw_prices[ticker] = df['Close']
                tk = yf.Ticker(ticker)
                divs = tk.dividends
                if not divs.empty:
                    divs.index = pd.to_datetime(divs.index).tz_localize(None).normalize()
                    div_raw_dict[ticker] = divs
        except: continue
    return pd.DataFrame(raw_prices).ffill().dropna(), div_raw_dict

# ==========================================
# 3. 核心模擬：Hibernate 防禦機制
# ==========================================
def run_simulation(df_price, div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense, enable_rebalance):
    # 設置 Hibernate Fund (強制保留 2 年生活費不投入股市)
    hibernate_fund = annual_expense * 2
    investable_capital = initial_capital - hibernate_fund
    
    lev_ratio = leverage_pct / 100.0
    debt = initial_capital * lev_ratio
    
    tickers = [c for c in df_price.columns if c != '^TWII']
    shares = {t: (investable_capital * (1 + lev_ratio) * weights[i]) / df_price[t].iloc[0] for i, t in enumerate(tickers)}
    
    cash = hibernate_fund
    trajectory = []
    
    for date, prices in df_price.iterrows():
        # 每日累計利息與費用
        interest_cost = (debt * borrow_rate) / 252
        expense_cost = annual_expense / 252
        
        # 股息入帳 (簡單模擬)
        for t in tickers:
            if date in div_raw_dict.get(t, pd.Series()):
                cash += shares[t] * div_raw_dict[t].loc[date]
        
        # 支付開銷
        cash -= (interest_cost + expense_cost)
        
        # 若現金不足，才動用賣股 (Hibernate 邏輯：cash 若為負數代表 Hibernate 用完，才賣股)
        if cash < 0:
            for t in tickers:
                sell_val = abs(cash) * weights[tickers.index(t)]
                shares[t] -= sell_val / prices[t]
            cash = 0
            
        val = sum([shares[t] * prices[t] for t in tickers])
        trajectory.append({'Date': date, 'Net': val + cash - debt})
        
    return pd.DataFrame(trajectory).set_index('Date')

# ==========================================
# 4. 側邊欄與 UI 渲染
# ==========================================
with st.sidebar:
    initial_capital = st.number_input("初始本金 (元)", value=17000000, step=1000000)
    annual_expense = st.number_input("每年生活費 (元)", value=580000, step=10000)
    leverage_pct = st.slider("槓桿比例 (%)", 0, 100, 20)
    borrow_rate = st.number_input("借款年利率 (%)", value=2.5, step=0.1) / 100.0
    
    selected_names = st.multiselect("選擇 ETF", list(DEFAULT_ETF_DICT.keys()), default=["0050 元大台灣50", "00713 元大台灣高息低波"])
    weights = [1/len(selected_names) for _ in selected_names]

df_price, div_raw_dict = load_raw_data([DEFAULT_ETF_DICT[n] for n in selected_names], datetime(2011,1,1), datetime.now())

if not df_price.empty:
    res = run_simulation(df_price, div_raw_dict, weights, initial_capital, leverage_pct, borrow_rate, annual_expense, True)
    st.line_chart(res)
    st.metric("期末淨資產 (萬元)", f"{res['Net'].iloc[-1]/10000:.2f}")
else:
    st.error("資料對齊失敗，請確保選擇的資產在時間區間內皆有成交紀錄。")
