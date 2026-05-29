import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="高股息策略與質押模擬器", layout="wide")
st.title("🛡️ 現金流資產組合與質押槓桿模擬器")

ETF_DICT = {
    "0056 元大高股息": "0056.TW",
    "00878 國泰永續高股息": "00878.TW",
    "00919 群益台灣精選高息": "00919.TW",
    "00929 復華台灣科技優息": "00929.TW",
    "00713 元大台灣高息低波": "00713.TW"
}

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
    adj_close_df = pd.DataFrame(df_dict).ffill().dropna()
    return adj_close_df

# 計算最大回撤的輔助函數
def calculate_mdd(cum_returns):
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

# 側邊欄：參數設定與資產組合
with st.sidebar:
    st.header("1. 選擇資產與權重")
    selected_names = st.multiselect("選擇組成 ETF", list(ETF_DICT.keys()), default=["00713 元大台灣高息低波", "00878 國泰永續高股息", "0056 元大高股息"])
    
    weights = []
    if selected_names:
        st.caption("請分配權重 (總和需為 100%)")
        # 自動平均分配預設值
        default_w = 100 // len(selected_names)
        for i, name in enumerate(selected_names):
            # 處理無法整除的餘數給最後一個
            val = default_w + (100 % len(selected_names) if i == len(selected_names)-1 else 0)
            w = st.number_input(f"{name[:5]} 權重 (%)", min_value=0, max_value=100, value=val)
            weights.append(w / 100)
            
    st.divider()
    st.header("2. 質押槓桿設定")
    leverage_pct = st.slider("質押借款比例 (%)", 0, 100, 20, help="例如 20%，代表本金 100 萬，借款 20 萬，總部位 120 萬")
    leverage_ratio = leverage_pct / 100.0
    borrow_rate = st.number_input("借款年利率 (%)", value=2.5, step=0.1) / 100.0
    
    st.divider()
    default_start = datetime.today() - timedelta(days=3*365)
    start_date = st.date_input("回測開始日期", value=default_start)
    end_date = st.date_input("回測結束日期", value=datetime.today())

# 主畫面運算
if selected_names:
    if sum(weights) != 1.0:
        st.error(f"⚠️ 目前權重總和為 {sum(weights)*100:.0f}%，請調整至 100%。")
    else:
        selected_tickers = [ETF_DICT[name] for name in selected_names]
        with st.spinner("載入報價與計算模型中..."):
            df = load_etf_data(selected_tickers, start_date, end_date)
            
        if not df.empty:
            # 1. 計算每日報酬率
            daily_returns = df.pct_change().dropna()
            
            # 2. 計算原型組合 (Unleveraged) 日報酬
            port_daily_returns = (daily_returns * weights).sum(axis=1)
            
            # 3. 計算槓桿組合 (Leveraged) 日報酬 (扣除每日利息成本)
            daily_borrow_rate = borrow_rate / 252
            lev_daily_returns = port_daily_returns * (1 + leverage_ratio) - (daily_borrow_rate * leverage_ratio)
            
            # 4. 計算累積報酬與回撤
            cum_port = (1 + port_daily_returns).cumprod() * 100
            cum_lev = (1 + lev_daily_returns).cumprod() * 100
            
            port_mdd = calculate_mdd(cum_port)
            lev_mdd = calculate_mdd(cum_lev)
            
            # 5. 計算年化報酬率與波動率
            years = (df.index[-1] - df.index[0]).days / 365.25
            port_cagr = (cum_port.iloc[-1] / 100) ** (1 / years) - 1
            lev_cagr = (cum_lev.iloc[-1] / 100) ** (1 / years) - 1
            
            port_vol = port_daily_returns.std() * np.sqrt(252)
            lev_vol = lev_daily_returns.std() * np.sqrt(252)
            
            # 6. 維持率試算
            # 公式: 總資產市值 / 借款金額。如果在 MDD 發生時。
            if leverage_ratio > 0:
                initial_maintenance = ((1 + leverage_ratio) / leverage_ratio) * 100
                mdd_maintenance = (((1 + leverage_ratio) * (1 - abs(port_mdd))) / leverage_ratio) * 100
            else:
                initial_maintenance = float('inf')
                mdd_maintenance = float('inf')

            # --- 渲染儀表板 ---
            st.subheader("📊 策略績效核心指標")
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("原型年化報酬 (CAGR)", f"{port_cagr*100:.2f}%")
            col2.metric("槓桿後年化報酬", f"{lev_cagr*100:.2f}%", f"{(lev_cagr-port_cagr)*100:.2f}%")
            col3.metric("原型最大回撤 (MDD)", f"{port_mdd*100:.2f}%")
            col4.metric("槓桿後最大回撤", f"{lev_mdd*100:.2f}%")
            
            st.divider()
            st.subheader("🚨 質押維持率壓力測試")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("當前質押比例 (總曝險)", f"{leverage_pct}% ({100+leverage_pct}%)")
            mcol2.metric("平時維持率", "無限大" if leverage_ratio == 0 else f"{initial_maintenance:.0f}%")
            
            # 維持率警告標示
            if leverage_ratio > 0:
                if mdd_maintenance < 130:
                    mcol3.error(f"遭遇歷史回撤時維持率: {mdd_maintenance:.0f}% (⚠️ 斷頭風險)")
                elif mdd_maintenance < 166:
                    mcol3.warning(f"遭遇歷史回撤時維持率: {mdd_maintenance:.0f}% (需補繳保證金)")
                else:
                    mcol3.success(f"遭遇歷史回撤時維持率: {mdd_maintenance:.0f}% (安全水位)")
            else:
                mcol3.metric("遭遇歷史回撤時維持率", "無限大")

            # --- 繪製績效對比圖 ---
            st.subheader("📈 資金曲線：原型組合 vs 質押策略")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, mode='lines', name='原型資產組合 (100%)', line=dict(color='#2E86C1', width=2)))
            fig.add_trace(go.Scatter(x=cum_lev.index, y=cum_lev, mode='lines', name=f'質押槓桿組合 ({100+leverage_pct}%)', line=dict(color='#E74C3C', width=2)))
            
            fig.update_layout(
                hovermode="x unified",
                yaxis_title="累積報酬 (基期=100)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- 配息率評估提示 ---
            st.info("💡 **關於配息率的估算：** 若原型組合平均殖利率為 6%，借款利率為 2.5%，質押 20% 後的預估真實現金流產出率為： $6\% \times 1.2 - 2.5\% \times 0.2 = 6.7\%$")
        else:
            st.warning("資料獲取失敗，請確認時間區間。")
else:
    st.info("請從左側欄位選擇至少一檔 ETF 來建立投資組合。")
