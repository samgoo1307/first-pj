import streamlit as st
import os
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import re
from langchain_google_genai import ChatGoogleGenerativeAI

# [1] 환경 설정
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

st.set_page_config(page_title="AI 투자 전략가", layout="wide")

# [2] 가독성 및 줄바꿈을 위한 CSS 설정
st.markdown("""
    <style>
    /* 텍스트 줄바꿈 및 가독성 설정 */
    .report-container {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        line-height: 1.7;
        font-size: 16px;
    }
    /* 가로 스크롤바 방지 */
    .stMarkdown {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# [3] 분석 함수
@st.cache_data(ttl=3600, show_spinner=False)
def run_investment_analysis(stock_ticker, risk_level):
    today_date = datetime.now().strftime("%Y-%m-%d")
    my_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=os.getenv("GEMINI_API_KEY")
    
    # 현재가 정보를 에이전트에게 명확히 전달하기 위한 사전 작업
    stock_info = yf.Ticker(stock_ticker).info
    current_price = stock_info.get('currentPrice', '알 수 없음')

    class FinancialTool(BaseTool):
        name: str = "FinancialTool"
        description: str = "주식의 최신 재무 데이터와 현재가를 가져옵니다."
        def _run(self, ticker: str) -> str:
            s = yf.Ticker(ticker)
            info = s.info
            return f"현재가: ${info.get('currentPrice')}, 시총:{info.get('marketCap')}, PER:{info.get('forwardPE')}, EPS:{info.get('forwardEps')}"

    analyst = Agent(
        role='수석 금융 분석가',
        goal=f'{today_date} 기준 {stock_ticker}의 재무와 시장 상황을 정밀 분석하여 현실적인 전략 수립',
        backstory=f'너는 시장의 현재 가격을 가장 중요하게 생각하는 월가 출신 분석가야. 현재가 ${current_price}를 기준으로 실현 가능한 목표를 세워.',
        llm=my_llm,
        tools=[FinancialTool(), SerperDevTool()],
        allow_delegation=False,
        max_iter=3, # [중요] 에이전트가 너무 오래 고민하지 않게 실행 횟수 제한 (API 호출 감소)
        verbose=True
    )
    
    task = Task(
        description=f"""
        오늘({today_date}) 기준으로 {stock_ticker}를 분석하여 한국어 리포트를 작성하세요.
        
        [필수 지침: 비현실적 가격 설정 금지]
        - 현재가(${current_price})를 반드시 확인하고, 이를 바탕으로 목표가와 손절가를 설정하세요.
        - 목표가는 현재가 대비 논리적인 상승 여력(보통 10~30%) 내에서, 손절가는 리스크 관리 범위 내에서 설정하세요.
        - 현재가와 터무니없이 동떨어진 숫자(예: 200불 주식을 600불로 설정)는 절대 금지입니다.
        
        [중요: API 절약 지시]
        1. 웹 검색(Serper)은 최신 뉴스 확인을 위해 딱 2~3회만 수행하세요.
        2. 나머지는 제공된 재무 도구(FinancialTool) 데이터만 활용하세요.
        3. 불필요한 반복 검색을 금지합니다.
        
        [리포트 구성]
        1. 실적 분석: 최근 재무 지표 요약
        2. SWOT 분석
        3. 매매 전략: '목표가: 숫자', '손절가: 숫자' 형식으로 명시 (이유 포함)
        """,
        expected_output="현실적인 가격 전략이 포함된 종합 투자 리포트",
        agent=analyst
    )

    crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential)
    result_obj = crew.kickoff()
    return str(result_obj)

# [4] 차트 시각화 함수 (줄바꿈 및 가로선 포함)
def plot_stock_chart(ticker, target_price=None, stop_loss=None):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        
        if df.empty:
            st.warning("차트 데이터를 가져올 수 없습니다.")
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()

        fig = go.Figure(data=[go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price', increasing_line_color='#FF4B4B', decreasing_line_color='#0078FF'
        )])

        if target_price:
            fig.add_hline(y=target_price, line_dash="dash", line_color="#00FF00", 
                          annotation_text=f"목표: ${target_price}", annotation_position="top right")
        
        if stop_loss:
            fig.add_hline(y=stop_loss, line_dash="dash", line_color="#FF0000", 
                          annotation_text=f"손절: ${stop_loss}", annotation_position="bottom right")

        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, title=f"{ticker} 일봉 차트")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"차트 오류: {e}")

# [5] 메인 화면 레이아웃
st.title("🚀 AI 주식 종목 분석")

with st.sidebar:
    st.header("⚙️ 설정")
    stocks = st.text_input("종목 코드", value="NVDA").upper()
    risk = st.selectbox("투자 성향", ["Lowest risk", "Mid risk", "High risk"])
    btn = st.button("종합 분석 실행")

if btn:
    col_text, col_chart = st.columns([1.1, 1]) 
    
    with st.spinner("최신 시장 데이터를 분석 중입니다..."):
        try:
            result_text = run_investment_analysis(stocks, risk)
            
            # 가격 추출 로직
            target_match = re.search(r'목표가[:\s]*\$?([\d,.]+)', result_text)
            stop_match = re.search(r'손절가[:\s]*\$?([\d,.]+)', result_text)
            
            def parse_p(m):
                return float(m.group(1).replace(',', '')) if m else None

            t_val = parse_p(target_match)
            s_val = parse_p(stop_match)

            with col_chart:
                st.subheader("📈 매매 전략 차트")
                plot_stock_chart(stocks, target_price=t_val, stop_loss=s_val)

            with col_text:
                st.subheader("📝 AI 분석 리포트")
                # 줄바꿈이 적용되는 컨테이너
                st.info(f"분석 기준일: {datetime.now().strftime('%Y-%m-%d')}")
                st.markdown(f'<div class="report-container">{result_text}</div>', unsafe_allow_html=True)
                st.success("분석 완료")

        except Exception as e:

            st.error(f"오류 발생: {e}")





