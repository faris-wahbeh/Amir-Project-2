# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import itertools

# ─── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Strategy Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit header/menu/footer
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Portfolio Parameters")

num_positions = st.sidebar.slider("Number of Positions", 5, 15, 15, 1)
cash_percentage = st.sidebar.slider("Cash Percentage (%)", 0, 50, 12, 1)
rebalance_frequency = st.sidebar.selectbox(
    "Rebalance Frequency",
    ["monthly", "quarterly", "semi-yearly"],
    index=0
)
rebalance_cost = st.sidebar.slider(
    "Rebalance Cost (%)", 0.0, 5.0, 0.04, 0.01, format="%.2f"
)

# ─── Data Helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def get_actual_gross_returns():
    data = {
        '2018': [4.7, 8.4, 1.2, 2.8, 3.1, 5.4, 1.3, 7.1, 0.7, -8.5, 3.4, -8.3],
        '2019': [8.6, 8.9, 3.2, 4.9, -2.5, 6.7, 3.2, -0.4, -6.5, 0.4, 5.5, 0.6],
        '2020': [5.5, -0.9, -14.0, 13.0, 8.2, 3.5, 5.9, 6.7, -3.2, -3.5, 11.6, 6.0],
        '2021': [-2.5, 8.2, -6.8, 4.9, -6.3, 6.3, 3.6, 5.2, -2.2, 3.1, -1.8, -0.1],
        '2022': [-12.9, -0.5, -2.1, -8.9, -9.5, -8.2, 9.1, -3.1, -8.1, 3.6, 4.0, 0]
    }
    seq = []
    for y in sorted(data):
        seq.extend(data[y])
    return seq

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
    for t, (_, row) in enumerate(df.iterrows()):
        if t % p == 0:
            current = row.iloc[:number_of_positions].tolist()
        portfolio.iloc[t] = current
    return portfolio

@st.cache_data
def generate_returns_portfolio(portfolio, price_csv_path):
    prices = pd.read_csv(price_csv_path, index_col=0, parse_dates=True, dayfirst=True)
    returns = prices.pct_change().fillna(0.0)
    ret_port = pd.DataFrame(index=portfolio.index, columns=portfolio.columns, dtype=float)
    for date in portfolio.index:
        tickers = portfolio.loc[date]
        ret_port.loc[date] = [
            returns.at[date, t] if t in returns.columns else 0.0
            for t in tickers
        ]
    return ret_port

def calculate_rebalance_impact(prev, curr, weights):
    change = 0.0
    prev_map = {t: i for i, t in enumerate(prev)}
    curr_map = {t: i for i, t in enumerate(curr)}
    # out
    for t in prev:
        if t in curr_map:
            change += abs(weights[curr_map[t]] - weights[prev_map[t]])
        else:
            change += weights[prev_map[t]]
    # in
    for t in curr:
        if t not in prev_map:
            change += weights[curr_map[t]]
    return change

def build_weighted_returns_df(num_positions, cash_pct, returns_portfolio,
                              rebalance_frequency, rebalance_cost, portfolio):
    w = np.array(calculate_position_percentages(num_positions, cash_pct)) / 100.0
    freq_map = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    p = freq_map[rebalance_frequency]
    df = pd.DataFrame(index=returns_portfolio.index, columns=returns_portfolio.columns, dtype=float)
    prev_holdings = None

    for idx, date in enumerate(returns_portfolio.index):
        df.loc[date] = w * returns_portfolio.loc[date].values
        if idx % p == 0 and idx > 0 and prev_holdings is not None:
            curr = portfolio.loc[date].tolist()
            turn = calculate_rebalance_impact(prev_holdings, curr, w)
            cost_factor = turn * (rebalance_cost / 100)
            df.loc[date] *= (1 - cost_factor)
        if idx % p == 0:
            prev_holdings = portfolio.loc[date].tolist()

    return df

def calculate_portfolio_performance():
    base = Path(".")
    rank, prices = base/"Rank.csv", base/"Prices.csv"
    if not (rank.exists() and prices.exists()):
        st.error("Missing Rank.csv or Prices.csv!")
        return None, None
    port = generate_portfolio(num_positions, rebalance_frequency, rank)
    ret_port = generate_returns_portfolio(port, prices)
    wdf = build_weighted_returns_df(
        num_positions, cash_percentage, ret_port,
        rebalance_frequency, rebalance_cost, port
    )
    daily = wdf.sum(axis=1)
    value = 100 * (1 + daily).cumprod()
    return value, port

def create_comparison_chart(portfolio_value):
    gross = get_actual_gross_returns()
    actual = [100.0]
    for r in gross[: len(portfolio_value)]:
        actual.append(actual[-1] * (1 + r/100))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(portfolio_value))),
        y=portfolio_value.values,
        mode="lines", name="Strategy",
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(actual))),
        y=actual[: len(portfolio_value)],
        mode="lines", name="Actual",
        line=dict(width=2, dash="dash")
    ))
    fig.update_layout(
        template="simple_white",
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Months",
        yaxis_title="Portfolio Value ($)",
        height=500,
    )
    return fig, actual

# ─── Optimization Helpers ─────────────────────────────────────────────────────
def calculate_portfolio_for_period(num_pos, cash_pct, rebal_freq, rebal_cost, start_m, end_m):
    val, _ = calculate_portfolio_performance()
    if val is None: return None
    series = val.iloc[start_m:end_m+1].values
    if len(series):
        series = series * 100 / series[0]
    return series

def calculate_error(strategy, actual):
    if len(strategy) != len(actual):
        return float('inf')
    errs = [(s - a)/a*100 for s, a in zip(strategy, actual)]
    return np.sqrt(np.mean([e**2 for e in errs]))

def find_optimal_parameters(start_month, end_month, rebalance_filter):
    gross = get_actual_gross_returns()
    actual = [100.0]
    for r in gross[start_month:end_month+1]:
        actual.append(actual[-1] * (1 + r/100))
    actual = [v * 100/actual[0] for v in actual]

    positions_range = list(range(5, 16))
    cash_range = list(range(0, 31, 5))
    rebalance_options = (
        ['monthly','quarterly','semi-yearly']
        if rebalance_filter=='any'
        else [rebalance_filter]
    )
    cost_range = [0.0,0.02,0.05,0.1,0.2,0.5,1.0,2.0,3.0,4.0,5.0]

    best, best_err = None, float('inf')
    total = (len(positions_range)*len(cash_range)*len(rebalance_options)*len(cost_range))
    pb = st.progress(0)
    status = st.empty()

    for i, (pos, cash, reb, cost) in enumerate(itertools.product(
        positions_range, cash_range, rebalance_options, cost_range
    )):
        status.text(f"Testing {i+1}/{total}")
        series = calculate_portfolio_for_period(pos, cash, reb, cost, start_month, end_month)
        if series is not None and len(series)==len(actual):
            err = calculate_error(series, actual)
            if err < best_err:
                best_err, best = err, {
                    'num_positions': pos,
                    'cash_percentage': cash,
                    'rebalance_frequency': reb,
                    'rebalance_cost': cost,
                    'error': err,
                    'strategy_values': series,
                    'actual_values': actual
                }
        pb.progress((i+1)/total)

    pb.empty()
    status.empty()
    return best

def create_optimization_chart(opt):
    x = list(range(len(opt['strategy_values'])))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=opt['strategy_values'], mode='lines',
        name='Optimal Strategy', line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=opt['actual_values'], mode='lines',
        name='Actual', line=dict(color='red', width=2, dash='dash')
    ))
    fig.update_layout(
        template='simple_white',
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Months in Period",
        yaxis_title="Normalized Value",
        height=400
    )
    return fig

# ─── Main App ─────────────────────────────────────────────────────────────────
st.title("Portfolio Strategy Analyzer")
st.markdown("---")

portfolio_value, portfolio = calculate_portfolio_performance()
if portfolio_value is None:
    st.stop()

fig, actual = create_comparison_chart(portfolio_value)
st.plotly_chart(fig, use_container_width=True)

# Metrics
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
sf = portfolio_value.iloc[-1]
af = actual[len(portfolio_value)]
sr = (sf - 100)/100*100
ar = (af - 100)/100*100

c1.metric("Strategy Final", f"${sf:.2f}", f"{sr:+.1f}%")
c2.metric("Actual Final",   f"${af:.2f}", f"{ar:+.1f}%")
c3.metric("Outperformance",  f"{(sr-ar):+.1f}%")
yrs = len(portfolio_value)/12
c4.metric("Annualized",      f"{((sf/100)**(1/yrs)-1)*100:.1f}%")



# Optimization Section
st.markdown("---")
st.subheader("Find Optimal Parameters")
o1, o2, o3, o4 = st.columns(4)
start_month = o1.number_input("Start Month", 0, len(portfolio_value)-1, 14)
end_month   = o2.number_input("End Month",   0, len(portfolio_value)-1, 30)
reb_filter  = o3.selectbox("Rebalance Filter", ["any","monthly","quarterly","semi-yearly"])
run_opt     = o4.button("Find Match")

if run_opt:
    if end_month <= start_month:
        st.error("End month must be after start month.")
    else:
        with st.spinner("Optimizing..."):
            opt = find_optimal_parameters(start_month, end_month, reb_filter)
            if opt:
                st.success("Optimal parameters found!")
                p1, p2 = st.columns(2)
                with p1:
                    st.write("**Parameters**")
                    st.write(f"- Positions: {opt['num_positions']}")
                    st.write(f"- Cash %: {opt['cash_percentage']}%")
                    st.write(f"- Rebalance: {opt['rebalance_frequency']}")
                    st.write(f"- Cost: {opt['rebalance_cost']:.2f}%")
                with p2:
                    st.write("**Performance**")
                    st.write(f"- Error (RMSE): {opt['error']:.2f}%")
                st.plotly_chart(create_optimization_chart(opt), use_container_width=True)
            else:
                st.error("No valid combination found.")
