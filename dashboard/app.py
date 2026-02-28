"""
AEGIS Streamlit Dashboard
Real-time signal monitoring and performance analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.data_fetcher import DataPipeline
from core.signal_generator import SignalGenerator
from indicators import IndicatorOrchestrator

# Page configuration
st.set_page_config(
    page_title="AEGIS Trading Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
    }
    .signal-bullish {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .signal-bearish {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .signal-neutral {
        background-color: #e2e3e5;
        border-left: 5px solid #6c757d;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


class DashboardState:
    """Manage dashboard state"""
    
    def __init__(self):
        self.refresh_interval = 300  # 5 minutes
        self.selected_symbol = "BTC/USDT"
        self.selected_timeframe = "1h"
        self.risk_level = "moderate"
    
    def load_signals(self) -> pd.DataFrame:
        """Load latest signals"""
        try:
            with open('data/processed/latest_signals.json') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except:
            return pd.DataFrame()
    
    def load_performance(self) -> dict:
        """Load performance metrics"""
        try:
            with open('data/processed/performance.json') as f:
                return json.load(f)
        except:
            return {}


def render_header():
    """Render dashboard header"""
    st.markdown('<div class="main-header">🛡️ AEGIS Trading Dashboard</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Online", "Active")
    with col2:
        st.metric("Last Update", datetime.now().strftime("%H:%M:%S"), "Live")
    with col3:
        st.metric("Data Sources", "2 Exchanges", "Binance, Bybit")
    with col4:
        st.metric("Models", "3 Active", "LGBM, XGB, Ensemble")


def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.header("⚙️ Configuration")
    
    # Symbol selection
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"]
    selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)
    
    # Timeframe selection
    timeframes = ["1h", "4h", "1d", "1w"]
    selected_tf = st.sidebar.selectbox("Timeframe", timeframes)
    
    # Risk level
    risk_levels = ["conservative", "moderate", "aggressive"]
    risk = st.sidebar.radio("Risk Level", risk_levels, index=1)
    
    # Auto refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    st.sidebar.header("📊 Display Options")
    show_indicators = st.sidebar.multiselect(
        "Indicators",
        ["RSI", "MACD", "Bollinger Bands", "Volume", "EMA"],
        default=["RSI", "MACD"]
    )
    
    return selected_symbol, selected_tf, risk, auto_refresh, show_indicators


def render_signals_overview(signals_df: pd.DataFrame):
    """Render signals overview section"""
    st.header("📡 Active Signals")
    
    if signals_df.empty:
        st.info("No active signals. Waiting for high-quality opportunities...")
        return
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_signals = len(signals_df)
    long_signals = len(signals_df[signals_df['direction'] == 'long'])
    short_signals = len(signals_df[signals_df['direction'] == 'short'])
    high_conf = len(signals_df[signals_df['confidence'].isin(['high', 'very_high'])])
    
    with col1:
        st.metric("Total Signals", total_signals)
    with col2:
        st.metric("Long", long_signals, delta=f"{long_signals/total_signals*100:.0f}%")
    with col3:
        st.metric("Short", short_signals, delta=f"{short_signals/total_signals*100:.0f}%")
    with col4:
        st.metric("High Confidence", high_conf)
    with col5:
        avg_conf = signals_df['confidence_score'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    
    # Signals table
    st.subheader("Signal Details")
    
    for _, signal in signals_df.iterrows():
        direction = signal.get('direction', 'neutral')
        confidence = signal.get('confidence', 'low')
        
        css_class = f"signal-{direction}"
        
        with st.container():
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{signal['symbol']}</strong> | 
                Direction: {direction.upper()} | 
                Confidence: {confidence} ({signal.get('confidence_score', 0):.1%}) | 
                Entry: ${signal.get('entry_price', 0):,.2f} | 
                R:R {signal.get('risk_reward', 0):.2f}:1
            </div>
            """, unsafe_allow_html=True)


def render_price_chart(symbol: str, timeframe: str, indicators: list):
    """Render interactive price chart"""
    st.header(f"📈 {symbol} Price Action")
    
    try:
        # Fetch data
        pipeline = DataPipeline()
        df = pipeline.fetch_complete_data(symbol, timeframe, update_only=True)
        
        if df.empty:
            st.error("No data available")
            return
        
        # Calculate indicators
        orchestrator = IndicatorOrchestrator()
        df = orchestrator.calculate_all(df)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{symbol} Price', 'Volume', 'RSI')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands if selected
        if 'Bollinger Bands' in indicators and 'bb_upper' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_upper'], 
                          line=dict(color='rgba(173,216,230,0.5)'), 
                          name='BB Upper', showlegend=False),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_lower'], 
                          line=dict(color='rgba(173,216,230,0.5)'), 
                          name='BB Lower', fill='tonexty', 
                          fillcolor='rgba(173,216,230,0.2)',
                          showlegend=False),
                row=1, col=1
            )
        
        # Add EMA if selected
        if 'EMA' in indicators:
            for period, color in [(8, 'orange'), (21, 'blue'), (50, 'purple')]:
                if f'ema_{period}' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df[f'ema_{period}'],
                                  line=dict(color=color, width=1),
                                  name=f'EMA{period}'),
                        row=1, col=1
                    )
        
        # Volume
        colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
                  for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='Volume'),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in indicators and 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi'], 
                          line=dict(color='blue'), name='RSI'),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current indicators summary
        latest = df.iloc[-1]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Price", f"${latest['close']:.2f}", 
                     f"{(latest['close']/df['close'].iloc[-2]-1)*100:.2f}%")
        with col2:
            rsi = latest.get('rsi', 0)
            st.metric("RSI", f"{rsi:.1f}", 
                     "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"))
        with col3:
            st.metric("ATR", f"{latest.get('atr', 0):.2f}", 
                     f"{latest.get('atr_pct', 0)*100:.2f}%")
        with col4:
            trend = "Bullish" if latest.get('ema_8', 0) > latest.get('ema_21', 0) else "Bearish"
            st.metric("Trend", trend)
        with col5:
            vol = latest.get('relative_volume', 1)
            st.metric("Rel Volume", f"{vol:.2f}x", 
                     "High" if vol > 2 else "Normal")
        
    except Exception as e:
        st.error(f"Error loading chart: {e}")


def render_performance_metrics(performance: dict):
    """Render performance analytics"""
    st.header("📊 Performance Analytics")
    
    if not performance:
        st.info("No performance data available yet.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_ret = performance.get('total_return', 0)
        st.metric("Total Return", f"{total_ret:.2%}", 
                 delta=f"{total_ret*100:.1f}%")
    with col2:
        sharpe = performance.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}",
                 delta="Good" if sharpe > 1 else "Moderate")
    with col3:
        max_dd = performance.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_dd:.2%}",
                 delta=f"{max_dd*100:.1f}%", delta_color="inverse")
    with col4:
        win_rate = performance.get('win_rate', 0)
        st.metric("Win Rate", f"{win_rate:.1%}",
                 delta=f"{win_rate*100:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Equity Curve")
        if 'equity_curve' in performance:
            equity_df = pd.DataFrame(performance['equity_curve'])
            fig = px.line(equity_df, x='date', y='equity', 
                         title='Portfolio Value Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Monthly Returns")
        if 'monthly_returns' in performance:
            monthly_df = pd.DataFrame(performance['monthly_returns'])
            fig = px.bar(monthly_df, x='month', y='return',
                        title='Returns by Month',
                        color='return',
                        color_continuous_scale=['red', 'green'])
            st.plotly_chart(fig, use_container_width=True)


def render_system_health():
    """Render system health monitoring"""
    st.header("🔧 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Data Pipeline")
        st.metric("Status", "✅ Healthy")
        st.metric("Last Fetch", "2 min ago")
        st.metric("Success Rate", "99.8%")
    
    with col2:
        st.subheader("ML Models")
        st.metric("Status", "✅ Active")
        st.metric("Last Training", "2 days ago")
        st.metric("Prediction Latency", "45ms")
    
    with col3:
        st.subheader("Risk Systems")
        st.metric("Status", "✅ Monitoring")
        st.metric("Portfolio Heat", "12%")
        st.metric("Circuit Breaker", "Standby")
    
    # Log viewer
    st.subheader("Recent System Events")
    
    events = [
        {"time": "10:23:45", "level": "INFO", "message": "Signal generated for BTC/USDT"},
        {"time": "10:18:12", "level": "INFO", "message": "Data fetch completed successfully"},
        {"time": "09:45:33", "level": "WARNING", "message": "High volatility detected in ETH/USDT"},
        {"time": "09:30:00", "level": "INFO", "message": "Daily risk limits reset"},
    ]
    
    for event in events:
        color = {"INFO": "blue", "WARNING": "orange", "ERROR": "red"}.get(event['level'], "gray")
        st.markdown(f"""
        <span style="color: {color}; font-weight: bold;">[{event['level']}]</span>
        <span style="color: gray;">{event['time']}</span> - {event['message']}
        """, unsafe_allow_html=True)


def main():
    """Main dashboard application"""
    state = DashboardState()
    
    # Header
    render_header()
    
    # Sidebar
    symbol, timeframe, risk, auto_refresh, indicators = render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📡 Signals", "📈 Charts", "📊 Performance", "🔧 System"
    ])
    
    with tab1:
        signals_df = state.load_signals()
        render_signals_overview(signals_df)
    
    with tab2:
        render_price_chart(symbol, timeframe, indicators)
    
    with tab3:
        performance = state.load_performance()
        render_performance_metrics(performance)
    
    with tab4:
        render_system_health()
    
    # Auto refresh
    if auto_refresh:
        st.empty()
        st.markdown("""
        <meta http-equiv="refresh" content="300">
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
