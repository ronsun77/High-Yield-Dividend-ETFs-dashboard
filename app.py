import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# 1. 網頁基本設定
st.set_page_config(page_title="台灣高股息 ETF 比較", layout="wide")
st.title("📈 台灣高股息 ETF 績效比較 (基於 yfinance)")

# 2. 定義標的與 yfinance 代碼對應字典
# 台灣上市 ETF 在 yfinance 的代碼需要加上 .TW
ETF_DICT = {
    "0056 元大高股息": "0056.TW",
    "00878 國泰永續高股息": "00878.TW",
    "00919 群益台灣精選高息": "00919.TW",
    "00929 復華台灣科技優息": "00929.TW"
}

# 3. 資料抓取函數 (加入快取機制與防呆處理)
@st.cache_data(ttl=3600) # 快取 1 小時
def load_etf_data(tickers, start_date, end_date):
    if not tickers:
        return pd.DataFrame()
    
    df_dict = {}
    for ticker in tickers:
        # 一檔一檔抓取，避免 yfinance 多檔下載時的 MultiIndex 結構錯亂
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            continue
            
        # 兼容 yfinance 新舊版本：如果是多層次索引，強制壓平取第一層
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # 優先取 'Adj Close' (還原收盤價)，若新版 yfinance 預設調整則改取 'Close'
        if 'Adj Close' in data.columns:
            # 存入字典，並將該欄位重新命名為 ETF 代碼
            df_dict[ticker] = data['Adj Close']
        elif 'Close' in data.columns:
            df_dict[ticker] = data['Close']
            
    # 如果都沒抓到資料，回傳空表
    if not df_dict:
        return pd.DataFrame()
        
    # 將所有 ETF 資料合併為單一 DataFrame
    adj_close_df = pd.DataFrame(df_dict)
    
    # 處理缺失值
    adj_close_df = adj_close_df.ffill().dropna()
    return adj_close_df

# 4. 側邊欄控制面板
with st.sidebar:
    st.header("⚙️ 參數設定")
    
    # 選擇標的
    selected_names = st.multiselect(
        "選擇比較標的",
        options=list(ETF_DICT.keys()),
        default=["0056 元大高股息", "00878 國泰永續高股息"]
    )
    
    # 選擇時間區間 (預設看近三年)
    default_start = datetime.today() - timedelta(days=3*365)
    start_date = st.date_input("開始日期", value=default_start)
    end_date = st.date_input("結束日期", value=datetime.today())

# 5. 主畫面運算與繪圖
if selected_names:
    # 轉換成 yfinance 代碼
    selected_tickers = [ETF_DICT[name] for name in selected_names]
    
    # 載入資料
    with st.spinner("正在從 Yahoo Finance 獲取資料..."):
        df = load_etf_data(selected_tickers, start_date, end_date)
    
    if not df.empty:
        # 績效歸一化：將起點全部設為 100 (計算累積報酬率)
        # 公式：(當日價格 / 第一天價格) * 100
        normalized_df = (df / df.iloc[0]) * 100
        
        # 繪製互動式折線圖
        st.subheader("📊 含息總報酬比較 (基期 = 100)")
        
        # 將 DataFrame 轉換為適合 Plotly 畫圖的長格式 (Long format)
        df_melted = normalized_df.reset_index().melt(id_vars='Date', var_name='ETF', value_name='累積報酬')
        
        fig = px.line(
            df_melted, 
            x='Date', 
            y='累積報酬', 
            color='ETF',
            labels={'Date': '日期', '累積報酬': '累積報酬 (起點=100)', 'ETF': '標的'},
            hover_data={"Date": "|%Y-%m-%d"}
        )
        
        # 調整圖表外觀
        fig.update_layout(
            hovermode="x unified", # 游標移上去會同時顯示所有 ETF 該日的數值
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # 顯示圖表
        st.plotly_chart(fig, use_container_width=True)
        
        # 顯示原始數據表
        with st.expander("檢視原始含息價格數據 (Adj Close)"):
            st.dataframe(df.sort_index(ascending=False))
    else:
        st.warning("該時間區間內沒有足夠的資料，請嘗試調整日期或選擇發行時間較長的 ETF。")
else:
    st.info("請從左側欄位選擇至少一檔 ETF 進行比較。")
