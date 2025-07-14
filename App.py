import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import itertools

# Page configuration
st.set_page_config(page_title="Portfolio Strategy Analyzer", layout="wide")

# Custom CSS for clean, readable UI
st.markdown("""
<style>
    /* Global settings */
    body, .stApp {
        background-color: #ffffff;
        color: #222222;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f7f7f7;
        padding: 20px 10px;
        color: #333333;
    }

    /* Buttons */
    .stButton > button {
        background-color: #0055a5;
        color: #ffffff;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #003f7f;
        color: #ffffff;
    }

    /* Metrics & containers */
    div[data-testid="metric-container"] {
        background-color: #f0f0f0;
        color: #222222;
        border: none;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }

    /* Headings */
    h1, h2, h3 {
        color: #111111;
    }
    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    h2 {
        font-size: 2rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    h3 {
        font-size: 1.75rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Inputs */
    .stSlider label,
    .stSelectbox label,
    .stNumberInput label {
        font-weight: 500;
        color: #333333;
    }

    /* Hide default header/footer */
    header, footer {
        visibility: hidden;
        height: 0;
        margin: 0;
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Portfolio Strategy Analyzer")
st.markdown("---")

# Sidebar inputs
with st.sidebar:
    st.header("Portfolio Settings")
    num_positions = st.slider(
        label="Number of Positions",
        min_value=5,
        max_value=15,
        value=15,
        step=1
    )
    cash_percentage = st.slider(
        label="Cash Allocation (%)",
        min_value=0,
        max_value=50,
        value=12,
        step=1
    )
    rebalance_frequency = st.selectbox(
        label="Rebalance Frequency",
        options=["monthly", "quarterly", "semi-yearly"],
        index=0
    )
    rebalance_cost = st.slider(
        label="Rebalance Cost (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.037,
        step=0.01,
        format="%.2f"
    )

# Data functions

def get_actual_gross_returns():
    returns_data = {
        '2018': [4.7, 8.4, 1.2, 2.8, 3.1, 5.4, 1.3, 7.1, 0.7, -8.5, 3.4, -8.3],
        '2019': [8.6, 8.9, 3.2, 4.9, -2.5, 6.7, 3.2, -0.4, -6.5, 0.4, 5.5, 0.6],
        '2020': [5.5, -0.9, -14.0, 13.0, 8.2, 3.5, 5.9, 6.7, -3.2, -3.5, 11.6, 6.0],
        '2021': [-2.5, 8.2, -6.8, 4.9, -6.3, 6.3, 3.6, 5.2, -2.2, 3.1, -1.8, -0.1],
        '2022': [-12.9, -0.5, -2.1, -8.9, -9.5, -8.2, 9.1, -3.1, -8.1, 3.6, 4.0, 0]
    }
    gross_returns = []
    for year in ['2018', '2019', '2020', '2021', '2022']:
        gross_returns.extend(returns_data[year])
    return gross_returns

@st.cache_data
def calculate_position_percentages(num_positions, cash_percentage):
    investable = 100.0 - cash_percentage
    if num_positions <= 5:
        top = 0.3 * investable
    else:
        top = 0.3 * investable - (num_positions - 5) * 0.03 * investable
    n = num_positions
    d = (2 * (top * n) - 2 * investable) / (n * (n - 1)) if n > 1 else 0
    return [top - i * d for i in range(n)]

@st.cache_data
def generate_portfolio(number_of_positions, rebalance_frequency, file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True, dayfirst=True)
    freq_map = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    p = freq_map[rebalance_frequency]
    portfolio = pd.DataFrame(index=df.index, columns=range(number_of_positions))
    current = []
    for t, (date, row) in enumerate(df.iterrows()):
        if t % p == 0:
            current = row.iloc[:number_of_positions].tolist()
        portfolio.loc[date] = current
    return portfolio

@st.cache_data
def generate_returns_portfolio(portfolio, price_csv_path):
    prices = pd.read_csv(price_csv_path, index_col=0, parse_dates=True, dayfirst=True)
    returns = prices.pct_change().fillna(0.0)
    ret_port = pd.DataFrame(index=portfolio.index, columns=portfolio.columns, dtype=float)
    for date in portfolio.index:
        if date in returns.index:
            tickers = portfolio.loc[date]
            ret_port.loc[date] = [returns.at[date, t] if t in returns.columns else 0.0 for t in tickers]
        else:
            ret_port.loc[date] = [0.0] * portfolio.shape[1]
    return ret_port

def calculate_rebalance_impact(previous, current, weights):
    total_change = 0.0
    prev_map = {t: i for i, t in enumerate(previous)}
    curr_map = {t: i for i, t in enumerate(current)}
    for t in previous:
        if t in curr_map:
            total_change += abs(weights[curr_map[t]] - weights[prev_map[t]])
        else:
            total_change += weights[prev_map[t]]
    for t in current:
        if t not in prev_map:
            total_change += weights[curr_map[t]]
    return total_change

@st.cache_data
def build_weighted_returns_df(num_positions, cash_percentage,
                              returns_portfolio, rebalance_frequency,
                              rebalance_cost, portfolio):
    weights = np.array(calculate_position_percentages(num_positions, cash_percentage)) / 100.0
    freq_map = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    p = freq_map[rebalance_frequency]
    df = pd.DataFrame(index=returns_portfolio.index, columns=returns_portfolio.columns, dtype=float)
    prev = None
    for t, date in enumerate(df.index):
        df.loc[date] = weights * returns_portfolio.loc[date].values
        if t % p == 0 and t > 0:
            curr_hold = portfolio.loc[date].tolist()
            change = calculate_rebalance_impact(prev, curr_hold, weights)
            df.loc[date] *= (1 - change * (rebalance_cost / 100))
        if t % p == 0:
            prev = portfolio.loc[date].tolist()
    return df

# Main performance calculation

def calculate_portfolio_performance():
    base = Path('.')
    holdings = base / 'Rank.csv'
    prices = base / 'Prices.csv'
    if not holdings.exists() or not prices.exists():
        st.error("Required files not found: Rank.csv or Prices.csv")
        return None, None
    try:
        portfolio = generate_portfolio(num_positions, rebalance_frequency, holdings)
        returns_portfolio = generate_returns_portfolio(portfolio, prices)
        weighted_df = build_weighted_returns_df(
            num_positions, cash_percentage, returns_portfolio,
            rebalance_frequency, rebalance_cost, portfolio
        )
        daily = weighted_df.sum(axis=1)
        value = 100.0 * (1 + daily).cumprod()
        return value, portfolio
    except Exception as e:
        st.error(f"Error calculating performance: {e}")
        return None, None

# Chart creation

def create_comparison_chart(portfolio_value):
    gross = get_actual_gross_returns()
    actual = [100.0]
    for r in gross[:len(portfolio_value)]:
        actual.append(actual[-1] * (1 + r/100))
    actual = actual[:len(portfolio_value)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(portfolio_value))), y=portfolio_value.values,
                             mode='lines', name='Strategy', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=list(range(len(actual))), y=actual,
                             mode='lines', name='Actual', line=dict(width=2, dash='dash')))
    fig.add_hline(y=100, line_dash='dot', opacity=0.3)
    fig.update_layout(xaxis_title='Months', yaxis_title='Value', plot_bgcolor='white', height=500)
    return fig, actual

# Optimization routines

def calculate_portfolio_for_period(num_pos, cash_pct, rebal_freq, rebal_cost, start, end):
    base = Path('.')
    holdings = base / 'Rank.csv'
    prices = base / 'Prices.csv'
    try:
        portfolio = generate_portfolio(num_pos, rebal_freq, holdings)
        returns_portfolio = generate_returns_portfolio(portfolio, prices)
        weighted_df = build_weighted_returns_df(
            num_pos, cash_pct, returns_portfolio, rebal_freq, rebal_cost, portfolio
        )
        value = 100.0 * (1 + weighted_df.sum(axis=1)).cumprod()
        period = value.iloc[start:end+1].values
        return (period * 100 / period[0]) if len(period)>0 else None
    except:
        return None

def calculate_error(strategy, actual):
    if len(strategy) != len(actual): return float('inf')
    errors = [(s - a)/a*100 for s, a in zip(strategy, actual)]
    return np.sqrt(np.mean([e**2 for e in errors]))

def find_optimal_parameters(start, end, rebalance_filter):
    gross = get_actual_gross_returns()
    actual = [100.0]
    for r in gross[start:min(end, len(gross))]: actual.append(actual[-1] * (1 + r/100))
    actual = [v*100/actual[0] for v in actual]
    positions = range(5,16)
    cashes = [0,5,10,15,20,25,30]
    rebalance_opts = ['monthly','quarterly','semi-yearly'] if rebalance_filter=='any' else [rebalance_filter]
    costs = [0.0,0.02,0.05,0.1,0.2,0.5,1.0,2.0,3.0,4.0,5.0]
    best, best_err = None, float('inf')
    total = len(positions)*len(cashes)*len(rebalance_opts)*len(costs)
    bar = st.progress(0); txt = st.empty()
    for i,(p,c,rf,cost) in enumerate(itertools.product(positions,cashes,rebalance_opts,costs)):
        bar.progress((i+1)/total)
        txt.text(f'Testing {i+1}/{total}')
        strat = calculate_portfolio_for_period(p,c,rf,cost,start,end)
        if strat is not None and len(strat)==len(actual):
            err = calculate_error(strat, actual)
            if err<best_err:
                best_err, best = err, dict(num_positions=p, cash_percentage=c,
                                            rebalance_frequency=rf, rebalance_cost=cost,
                                            error=err, strategy_values=strat,
                                            actual_values=actual)
    bar.empty(); txt.empty()
    return best

def create_optimization_comparison_chart(optimal_params, start, end):
    fig = go.Figure()
    x = list(range(len(optimal_params['strategy_values'])))
    fig.add_trace(go.Scatter(x=x, y=optimal_params['strategy_values'], mode='lines',
                             name='Optimal Strategy', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x, y=optimal_params['actual_values'], mode='lines',
                             name='Actual', line=dict(width=2, dash='dash')))
    fig.add_hline(y=100, line_dash='dot', opacity=0.3)
    fig.update_layout(title=f'Optimal Match (Months {start+1}-{end+1})',
                      xaxis_title='Months', yaxis_title='Normalized Value', height=400,
                      plot_bgcolor='white')
    return fig

# Run main app
value, portfolio = calculate_portfolio_performance()
if value is not None:
    fig, actual = create_comparison_chart(value)
    st.plotly_chart(fig, use_container_width=True)
    cols = st.columns(4)
    final_val = value.iloc[-1]
    pct = (final_val - 100) / 100 * 100
    cols[0].metric("Strategy Final", f"${final_val:.2f}", f"{pct:.1f}%")
    gross_val = actual[-1]; pct_gross = (gross_val-100)/100*100
    cols[1].metric("Actual Final", f"${gross_val:.2f}", f"{pct_gross:.1f}%")
    cols[2].metric("Outperformance", f"{pct-pct_gross:+.1f}%")
    years = len(value)/12; ann = (final_val/100)**(1/years)-1
    cols[3].metric("Annualized Return", f"{ann*100:.1f}%")
    st.markdown("---")
    df = pd.DataFrame({'Month': range(len(value)), 'Strategy_Value': value.values, 'Actual_Gross_Value': actual})
    st.markdown("---")
    st.subheader("Optimize Parameters")
    c1,c2,c3,c4 = st.columns(4)
    with c1: start_month = st.number_input("Start Month", 0, 59, 14)
    with c2: end_month = st.number_input("End Month", 0, 59, 30)
    with c3: rebalance_filter = st.selectbox("Frequency Filter", ["any","monthly","quarterly","semi-yearly"], 0)
    with c4:
        if st.button("Run Optimization"):
            if end_month <= start_month:
                st.error("End month must be greater than start month.")
            else:
                with st.spinner("Optimizing..."):
                    opt = find_optimal_parameters(start_month, end_month, rebalance_filter)
                    if opt:
                        st.success("Optimal parameters found!")
                        left,right = st.columns(2)
                        with left:
                            st.write(f"Positions: {opt['num_positions']}")
                            st.write(f"Cash: {opt['cash_percentage']}%")
                            st.write(f"Rebalance: {opt['rebalance_frequency']}")
                            st.write(f"Cost: {opt['rebalance_cost']:.2f}%")
                        with right:
                            st.write(f"Error: {opt['error']:.2f}%")
                        comp_fig = create_optimization_comparison_chart(opt, start_month, end_month)
                        st.plotly_chart(comp_fig, use_container_width=True)
