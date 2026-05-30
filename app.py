import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. 網頁基本設定 & 參數字典
# ==========================================
st.set_page_config(page_title="高股息策略與質押模擬器", layout="wide")
st.title("🛡️ 台灣高股息 ETF 現金流組合與質押模擬器")

ETF_DICT = {
    "0056 元大高股息": "0056.TW",
    "00878 國泰永續高股息": "00878.TW",
    "00919 群益台灣精選高息": "00919.TW",
    "00929 復華台灣科技優息": "00929.TW",
    "00713 元大台灣高息低波": "00713.TW"
}

# ==========================================
# 2. 資料抓取函數 (報價與配息)
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
    """抓取歷史配息資料並按年度加總"""
    if not tickers: return pd.DataFrame()
    div_dict = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            divs = tk.dividends
            if not divs.empty:
                divs.index = divs.index.tz_localize(None)
                # 按年度加總配息金額
                annual_divs = divs.groupby(divs.index.year).sum()
                div_dict[ticker] = annual_divs
        except Exception:
            continue
    # 合併成 DataFrame 並補 0 (若某年未配息)
    return pd.DataFrame(div_dict).fillna(0)

# ==========================================
# 3. 數學運算輔助函數
# ==========================================
def calculate_mdd(cum_returns):
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def calculate_metrics(returns_series, name, div_series=None):
    """計算單一資產或組合的核心績效與配息指標"""
    # 績效指標
    cagr = ((1 + returns_series).prod() ** (252 / len(returns_series))) - 1
    volatility = returns_series.std() * np.sqrt(252)
    sharpe_ratio = (cagr - 0.015) / volatility if volatility != 0 else 0
    
    cum_returns = (1 + returns_series).cumprod()
    mdd = calculate_mdd(cum_returns)
    
    # 配息穩定度 (變異係數 CV = 標準差 / 平均值)
    cv_str = "資料不足"
    if div_series is not None and len(div_series[div_series > 0]) >= 2:
        # 只計算有配息的年份，避免新 ETF 因過去年份補 0 導致失真
        valid_divs = div_series[div_series > 0]
        cv = valid_divs.std() / valid_divs.mean()
        cv_str = f"{cv:.2f}"
    
    return {
        "標的名稱": name,
        "年化報酬率 (%)": round(cagr * 100, 2),
        "年化波動率 (%)": round(volatility * 100, 2),
        "夏普值": round(sharpe_ratio, 2),
        "最大回撤 MDD (%)": round(mdd * 100, 2),
        "配息變異係數 (CV)": cv_str
    }

# ==========================================
# 4. 側邊欄：參數設定與資產組合
# ==========================================
with st.sidebar:
    st.header("1. 選擇資產與權重")
    selected_names = st.multiselect(
        "選擇組成 ETF", 
        list(ETF_DICT.keys()), 
        default=["00713 元大台灣高息低波", "00878 國泰永續高股息", "0056 元大高股息"]
    )
    
    weights = []
    if selected_names:
        st.caption("請分配權重 (總和需為 100%)")
        default_w = 100 // len(selected_names)
        for i, name in enumerate(selected_names):
            val = default_w + (100 % len(selected_names) if i == len(selected_names)-1 else 0)
            w = st.number_input(f"{name[:5]} 權重 (%)", min_value=0, max_value=100, value=val)
            weights.append(w / 100)
            
    st.divider()
    st.header("2. 質押槓桿設定")
    leverage_pct = st.slider("質押借款比例 (%)", 0, 100, 20)
    leverage_ratio = leverage_pct / 100.0
    borrow_rate = st.number_input("借款年利率 (%)", value=2.5, step=0.1) / 100.0
    
    st.divider()
    default_start = datetime.today() - timedelta(days=5*365)
    start_date = st.date_input("回測開始日期", value=default_start)
    end_date = st.date_input("回測結束日期", value=datetime.today())

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
            
            cum_port = (1 + port_daily_returns).cumprod() * 100
            cum_lev = (1 + lev_daily_returns).cumprod() * 100
            
            # --- 配息變異係數 (CV) 表格構建 ---
            st.subheader("📋 策略總表：單一 ETF vs 自訂資產組合")
            metrics_list = []
            
            # 單一 ETF 績效與 CV
            for col in daily_returns.columns:
                etf_name = [k for k, v in ETF_DICT.items() if v == col][0]
                # 提取該檔 ETF 的歷年配息紀錄
                etf_div_series = df_div[col] if col in df_div.columns else None
                metrics_list.append(calculate_metrics(daily_returns[col].dropna(), etf_name, etf_div_series))
            
            # 自訂組合績效與合成 CV
            port_div_series = None
            if not df_div.empty:
                valid_div_cols = [c for c in df_div.columns if c in selected_tickers]
                if valid_div_cols:
                    port_div_series = (df_div[valid_div_cols] * [weights[selected_tickers.index(c)] for c in valid_div_cols]).sum(axis=1)
                    
            metrics_list.append(calculate_metrics(port_daily_returns, "🎯 自訂現金流組合 (Portfolio)", port_div_series))
            
            comparison_df = pd.DataFrame(metrics_list).set_index("標的名稱")
            
            # 顯示表格並高亮自訂組合
            st.dataframe(
                comparison_df.style.apply(
                    lambda x: ['background: #117A65; color: white; font-weight: bold' if x.name == '🎯 自訂現金流組合 (Portfolio)' else '' for i in x], 
                    axis=1
                ).format(na_rep="N/A")
            )
            
            st.caption("💡 **配息變異係數 (CV)** = 標準差 / 平均值。數值越接近 0，代表歷年配息金額越穩定；數值越大，代表配息起伏劇烈。（新發行 ETF 若資料不足兩年則無法計算）")

            # --- 繪製歷年配息柱狀圖 ---
            st.subheader("💰 歷年配息金額動態比較")
            if not df_div.empty:
                # 篩選出近 10 年的資料讓圖表更乾淨，並強制複製一份避免警告
                recent_years = df_div[df_div.index >= (datetime.today().year - 10)].copy()
                recent_years.index = recent_years.index.astype(str)
                
                # 強制命名 index，避免 reset_index 後發生 KeyError
                recent_years.index.name = '年份'
                
                # 將列名從代碼轉換為中文名稱
                rename_dict = {v: k for k, v in ETF_DICT.items()}
                recent_years = recent_years.rename(columns=rename_dict)
                
                # 繪圖資料轉換
                melted_df = recent_years.reset_index().melt(id_vars='年份', var_name='ETF', value_name='年度配息總額 (元)')
                
                fig_div = px.bar(
                    melted_df,
                    x='年份', 
                    y='年度配息總額 (元)', 
                    color='ETF',
                    barmode='group'
                )
                fig_div.update_layout(
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_div, use_container_width=True)
            else:
                st.info("無法獲取足夠的歷史配息資料。")

            st.divider()

            # --- 維持率試算與壓力測試 ---
            port_mdd = calculate_mdd(cum_port)
            if leverage_ratio > 0:
                initial_maintenance = ((1 + leverage_ratio) / leverage_ratio) * 100
                mdd_maintenance = (((1 + leverage_ratio) * (1 - abs(port_mdd))) / leverage_ratio) * 100
            else:
                initial_maintenance = float('inf')
                mdd_maintenance = float('inf')

            st.subheader("🚨 質押維持率壓力測試")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("當前總曝險 (本金+借款)", f"{100+leverage_pct}%")
            mcol2.metric("平時維持率", "無限大" if leverage_ratio == 0 else f"{initial_maintenance:.0f}%")
            
            if leverage_ratio > 0:
                if mdd_maintenance < 130:
                    mcol3.error(f"遭遇歷史最大回撤時維持率: {mdd_maintenance:.0f}% (⚠️ 斷頭風險)")
                elif mdd_maintenance < 166:
                    mcol3.warning(f"遭遇歷史最大回撤時維持率: {mdd_maintenance:.0f}% (需補繳)")
                else:
                    mcol3.success(f"遭遇歷史最大回撤時維持率: {mdd_maintenance:.0f}% (安全水位)")
            else:
                mcol3.metric("遭遇歷史最大回撤時維持率", "無限大")

            # --- 繪製績效對比圖 ---
            st.subheader("📈 資金曲線：原型組合 vs 質押策略")
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=cum_port.index, y=cum_port, mode='lines', name='🎯 原型組合 (100%)', line=dict(color='#2E86C1', width=2)))
            fig_perf.add_trace(go.Scatter(x=cum_lev.index, y=cum_lev, mode='lines', name=f'🔥 質押槓桿組合 ({100+leverage_pct}%)', line=dict(color='#E74C3C', width=2)))
            
            fig_perf.update_layout(
                hovermode="x unified",
                yaxis_title="累積報酬 (基期=100)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
        else:
            st.warning("資料獲取失敗，請確認時間區間。")
else:
    st.info("請從左側欄位選擇至少一檔 ETF 來建立投資組合。")
