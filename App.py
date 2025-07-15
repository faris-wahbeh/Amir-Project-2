# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import itertools
import os

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


# ─── File Check Helper ────────────────────────────────────────────────────────
def check_required_files():
    base = Path(".")
    rank_file = base / "Rank.csv"
    prices_file = base / "Prices.csv"

    missing_files = []
    if not rank_file.exists():
        missing_files.append("Rank.csv")
    if not prices_file.exists():
        missing_files.append("Prices.csv")

    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.info("Please upload the following files to your Replit workspace:")
        for file in missing_files:
            st.code(f"• {file}")
        st.stop()

    return rank_file, prices_file


# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Portfolio Parameters")

num_positions = st.sidebar.slider("Number of Positions", 5, 15, 15, 1)
cash_percentage = st.sidebar.slider("Cash Percentage (%)", 0, 50, 12, 1)
rebalance_frequency = st.sidebar.selectbox(
    "Rebalance Frequency", ["monthly", "quarterly", "semi-yearly"], index=0)
rebalance_cost = st.sidebar.slider("Rebalance Cost (%)",
                                   0.0,
                                   5.0,
                                   0.04,
                                   0.01,
                                   format="%.2f")


# ─── Data Helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def get_actual_gross_returns():
    data = {
        '2018': [4.7, 0.4, 1.2, 2.8, 5.1, 5.4, 1.1, 7.1, 0.7, -8.5, 3.4, -8.3],
        '2019':
        [8.6, 8.9, 3.2, 4.9, -2.5, 6.7, 3.2, -0.4, -6.5, 0.4, 5.5, 0.6],
        '2020':
        [5.5, -6.6, -14.3, 14.2, 9.0, 3.9, 5.9, 6.6, -3.1, -3.7, 11.8, 7.2],
        '2021':
        [-2.5, 8.2, -6.8, 4.9, -6.3, 6.3, 3.6, 5.2, -2.2, 3.1, -1.8, -0.1],
        '2022':
        [-12.9, -0.5, -2.1, -8.9, -9.5, -8.2, 9.1, -3.1, -8.1, 3.6, 4.0, -2.4]
    }
    seq = []
    for y in sorted(data):
        seq.extend(data[y])
    return seq


@st.cache_data
def calculate_position_percentages(num_positions, cash_percentage):
    """Same as original - calculates weight for each position"""
    total_percentage = 100 - cash_percentage
    if num_positions <= 5:
        highest_percentage = 0.30 * total_percentage
    else:
        highest_percentage = 0.30 * total_percentage - (num_positions - 5) * 0.02 * total_percentage - (15-num_positions)
    
    sum_percentages = total_percentage
    n = num_positions
    a = highest_percentage
    S = sum_percentages

    common_difference = (2 * (a * n) - 2 * S) / (n * (n - 1)) if n > 1 else 0
    percentages = [a - i * common_difference for i in range(num_positions)]
    
    return percentages


@st.cache_data
def generate_portfolio(number_of_positions, rebalance_frequency, file_path):
    try:
        df = pd.read_csv(file_path,
                         index_col=0,
                         parse_dates=True,
                         dayfirst=True)
    except Exception as e:
        st.error(f"Error reading {file_path}: {str(e)}")
        return None

    freq_map = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    p = freq_map[rebalance_frequency]
    portfolio = pd.DataFrame(index=df.index,
                             columns=range(number_of_positions))
    current = []
    for t, (_, row) in enumerate(df.iterrows()):
        if t % p == 0:
            current = row.iloc[:number_of_positions].tolist()
        portfolio.iloc[t] = current
    return portfolio


@st.cache_data
def generate_returns_portfolio(portfolio, price_csv_path):
    try:
        prices = pd.read_csv(price_csv_path,
                             index_col=0,
                             parse_dates=True,
                             dayfirst=True)
    except Exception as e:
        st.error(f"Error reading {price_csv_path}: {str(e)}")
        return None

    returns = prices.pct_change(fill_method=None).fillna(0.0)
    ret_port = pd.DataFrame(index=portfolio.index,
                            columns=portfolio.columns,
                            dtype=float)
    for date in portfolio.index:
        tickers = portfolio.loc[date]
        ret_port.loc[date] = [
            returns.at[date, t] if t in returns.columns else 0.0
            for t in tickers
        ]
    return ret_port


def calculate_rebalance_impact(prev, curr, weights, prev_ranks, curr_ranks):
    """
    Calculate rebalance cost based on rank changes and weight differences.

    Args:
        prev: Previous portfolio positions (list of tickers)
        curr: Current portfolio positions (list of tickers)
        weights: Current weight allocations
        prev_ranks: Previous ranks for all stocks (dict: ticker -> rank)
        curr_ranks: Current ranks for all stocks (dict: ticker -> rank)

    Returns:
        Total weight change subject to rebalance cost
    """
    total_cost = 0.0

    # Create sets for easier comparison
    prev_set = set(prev)
    curr_set = set(curr)

    # For each current position, calculate the weight change
    for i, ticker in enumerate(curr):
        current_weight = weights[i]

        if ticker in prev_set:
            # Stock was in previous portfolio
            prev_position = prev.index(ticker)
            prev_weight = weights[prev_position] if prev_position < len(
                weights) else 0

            # Get rank information
            prev_rank = prev_ranks.get(ticker, None)
            curr_rank = curr_ranks.get(ticker, None)

            # If ranks are different or missing, there's a position change
            if prev_rank != curr_rank or prev_rank is None or curr_rank is None:
                # Calculate weight change - use absolute difference
                weight_change = abs(current_weight - prev_weight)
                total_cost += weight_change
            # If ranks are the same, minimal rebalancing needed
            elif prev_position != i:
                # Position changed but rank stayed same - small adjustment
                weight_change = abs(current_weight - prev_weight)
                total_cost += weight_change * 0.5  # Reduced cost for same rank
        else:
            # New stock in portfolio - full weight is subject to rebalance cost
            total_cost += current_weight

    # Account for stocks that were removed from portfolio
    for i, ticker in enumerate(prev):
        if ticker not in curr_set and i < len(weights):
            total_cost += weights[i]

    return total_cost


def build_weighted_returns_df(num_positions, cash_pct, returns_portfolio,
                              rebalance_frequency, rebalance_cost, portfolio):
    w = np.array(calculate_position_percentages(num_positions,
                                                cash_pct)) / 100.0
    freq_map = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    p = freq_map[rebalance_frequency]
    df = pd.DataFrame(index=returns_portfolio.index,
                      columns=returns_portfolio.columns,
                      dtype=float)
    prev_holdings = None

    # Load rank data to get historical ranks
    try:
        rank_data = pd.read_csv("Rank.csv",
                                index_col=0,
                                parse_dates=True,
                                dayfirst=True)
    except:
        rank_data = None

    for idx, date in enumerate(returns_portfolio.index):
        df.loc[date] = w * returns_portfolio.loc[date].values

        if idx % p == 0 and idx > 0 and prev_holdings is not None and rank_data is not None:
            curr = portfolio.loc[date].tolist()

            # Get previous date for rank comparison
            prev_date = returns_portfolio.index[
                idx - p] if idx >= p else returns_portfolio.index[0]

            # Create rank dictionaries for the two periods
            prev_ranks = {}
            curr_ranks = {}

            # Get ranks for previous period
            if prev_date in rank_data.index:
                prev_row = rank_data.loc[prev_date]
                for col_idx, ticker in enumerate(prev_row.values):
                    if pd.notna(ticker) and ticker.strip():
                        prev_ranks[ticker.strip()] = col_idx + 1

            # Get ranks for current period
            if date in rank_data.index:
                curr_row = rank_data.loc[date]
                for col_idx, ticker in enumerate(curr_row.values):
                    if pd.notna(ticker) and ticker.strip():
                        curr_ranks[ticker.strip()] = col_idx + 1

            # Calculate rebalance cost based on rank changes
            turn = calculate_rebalance_impact(prev_holdings, curr, w,
                                              prev_ranks, curr_ranks)
            cost_factor = turn * (rebalance_cost / 100)
            df.loc[date] *= (1 - cost_factor)

        if idx % p == 0:
            prev_holdings = portfolio.loc[date].tolist()

    return df


def calculate_portfolio_performance():
    rank_file, prices_file = check_required_files()

    port = generate_portfolio(num_positions, rebalance_frequency, rank_file)
    if port is None:
        return None, None

    ret_port = generate_returns_portfolio(port, prices_file)
    if ret_port is None:
        return None, None

    wdf = build_weighted_returns_df(num_positions, cash_percentage, ret_port,
                                    rebalance_frequency, rebalance_cost, port)
    daily = wdf.sum(axis=1)
    value = 100 * (1 + daily).cumprod()
    return value, port


def create_comparison_chart(portfolio_value):
    gross = get_actual_gross_returns()
    actual = [100.0]
    for r in gross[:len(portfolio_value)]:
        actual.append(actual[-1] * (1 + r / 100))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=list(range(len(portfolio_value))),
                   y=portfolio_value.values,
                   mode="lines",
                   name="Strategy",
                   line=dict(width=2, color='#1f77b4')))
    fig.add_trace(
        go.Scatter(x=list(range(len(actual))),
                   y=actual[:len(portfolio_value)],
                   mode="lines",
                   name="Actual",
                   line=dict(width=2, dash="dash", color='#ff7f0e')))
    fig.update_layout(
        template="simple_white",
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Months",
        yaxis_title="Portfolio Value ($)",
        height=500,
    )
    return fig, actual


# ─── Optimization Helpers ─────────────────────────────────────────────────────
def calculate_portfolio_for_period(num_pos, cash_pct, rebal_freq, rebal_cost,
                                   start_m, end_m):
    try:
        # Get full portfolio performance
        rank_file, prices_file = check_required_files()

        port = generate_portfolio(num_pos, rebal_freq, rank_file)
        if port is None:
            return None

        ret_port = generate_returns_portfolio(port, prices_file)
        if ret_port is None:
            return None

        wdf = build_weighted_returns_df(num_pos, cash_pct, ret_port,
                                        rebal_freq, rebal_cost, port)
        daily = wdf.sum(axis=1)
        value = 100 * (1 + daily).cumprod()

        # Extract the period - ensure valid indices
        max_idx = len(value) - 1
        start_idx = min(start_m, max_idx)
        end_idx = min(end_m, max_idx)

        if start_idx > end_idx:
            return None

        series = value.iloc[start_idx:end_idx + 1].values
        if len(series) > 0:
            # Normalize to start at 100
            series = series * 100 / series[0]
        return series

    except Exception as e:
        print(f"Error in calculate_portfolio_for_period: {e}")
        return None


def calculate_error(strategy, actual):
    if len(strategy) != len(actual) or len(strategy) == 0:
        return float('inf')

    # Calculate percentage errors
    errs = []
    for s, a in zip(strategy, actual):
        if a != 0:
            errs.append(((s - a) / a * 100)**2)
        else:
            errs.append((s - a)**2)

    # Return root mean square error
    return np.sqrt(np.mean(errs))


def find_optimal_parameters(start_month, end_month, rebalance_filter):
    gross = get_actual_gross_returns()

    # First build complete actual returns series
    actual_full = [100.0]
    for r in gross:
        actual_full.append(actual_full[-1] * (1 + r / 100))

    # Extract the period we want
    max_idx = len(actual_full) - 1
    start_idx = min(start_month, max_idx)
    end_idx = min(end_month, max_idx)

    if start_idx > end_idx:
        st.error("Invalid period selection")
        return None

    actual = actual_full[start_idx:end_idx + 1]

    # Normalize to start at 100
    if len(actual) > 0:
        actual = [v * 100 / actual[0] for v in actual]

    positions_range = list(range(5, 16))
    cash_range = list(range(0, 31, 5))
    rebalance_options = (['monthly', 'quarterly', 'semi-yearly']
                         if rebalance_filter == 'any' else [rebalance_filter])
    cost_range = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

    best, best_err = None, float('inf')
    total = (len(positions_range) * len(cash_range) * len(rebalance_options) *
             len(cost_range))
    pb = st.progress(0)
    status = st.empty()

    valid_combinations = 0

    for i, (pos, cash, reb, cost) in enumerate(
            itertools.product(positions_range, cash_range, rebalance_options,
                              cost_range)):
        status.text(
            f"Testing {i+1}/{total}: pos={pos}, cash={cash}%, reb={reb}, cost={cost:.2f}%"
        )

        series = calculate_portfolio_for_period(pos, cash, reb, cost,
                                                start_month, end_month)
        if series is not None and len(series) == len(actual):
            valid_combinations += 1
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

        pb.progress((i + 1) / total)

    pb.empty()
    status.empty()

    if valid_combinations == 0:
        st.warning(
            f"No valid combinations found. Tested {total} combinations but none produced valid results for the selected period."
        )

    return best


def create_optimization_chart(opt):
    x = list(range(len(opt['strategy_values'])))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x,
                   y=opt['strategy_values'],
                   mode='lines',
                   name='Optimal Strategy',
                   line=dict(color='green', width=2)))
    fig.add_trace(
        go.Scatter(x=x,
                   y=opt['actual_values'],
                   mode='lines',
                   name='Actual',
                   line=dict(color='red', width=2, dash='dash')))
    fig.update_layout(template='simple_white',
                      margin=dict(l=40, r=20, t=40, b=40),
                      xaxis_title="Months in Period",
                      yaxis_title="Normalized Value",
                      height=400)
    return fig


# ─── Main App ─────────────────────────────────────────────────────────────────
st.title("🚀 Portfolio Strategy Analyzer")
st.markdown("Analyze and optimize portfolio strategies using historical data")
st.markdown("---")

# Check files first
try:
    portfolio_value, portfolio = calculate_portfolio_performance()
    if portfolio_value is None:
        st.stop()
except Exception as e:
    st.error(f"Error calculating portfolio performance: {str(e)}")
    st.stop()

# Main chart
fig, actual = create_comparison_chart(portfolio_value)
st.plotly_chart(fig, use_container_width=True)

# Metrics
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
sf = portfolio_value.iloc[-1]
af = actual[len(portfolio_value)]
sr = (sf - 100) / 100 * 100
ar = (af - 100) / 100 * 100

c1.metric("Strategy Final", f"${sf:.2f}", f"{sr:+.1f}%")
c2.metric("Actual Final", f"${af:.2f}", f"{ar:+.1f}%")
c3.metric("Outperformance", f"{(sr-ar):+.1f}%")
yrs = len(portfolio_value) / 12
c4.metric("Annualized", f"{((sf/100)**(1/yrs)-1)*100:.1f}%")

# Monthly Comparison Table
st.markdown("---")
st.subheader("📊 Monthly Performance Comparison")

# Create monthly comparison data
months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
    'Nov', 'Dec'
]
years = ['2018', '2019', '2020', '2021', '2022']

# Get actual returns data
gross_returns = get_actual_gross_returns()

# Calculate strategy monthly returns
daily_returns = portfolio_value.pct_change(fill_method=None).fillna(0.0)
strategy_monthly_returns = []

# Group by months (assuming monthly data points)
for i in range(len(portfolio_value) - 1):
    if i < len(daily_returns):
        monthly_return = (
            portfolio_value.iloc[i + 1] / portfolio_value.iloc[i] - 1) * 100
        strategy_monthly_returns.append(monthly_return)

# Create restructured table with years as rows and months as columns
table_data = {}
month_idx = 0

for year in years:
    year_data = {'Year': year}
    year_strategy_returns = []
    year_actual_returns = []

    for month in months:
        if month_idx < len(strategy_monthly_returns) and month_idx < len(
                gross_returns):
            strategy_ret = strategy_monthly_returns[month_idx]
            actual_ret = gross_returns[month_idx]

            year_strategy_returns.append(strategy_ret)
            year_actual_returns.append(actual_ret)

            # Format cell with both values
            cell_value = f"T: {strategy_ret:.1f}%\nA: {actual_ret:.1f}%"
            year_data[month] = cell_value
            month_idx += 1
        else:
            year_data[month] = ""

    # Calculate yearly totals
    if year_strategy_returns and year_actual_returns:
        strategy_yearly = (
            (np.prod([1 + r / 100 for r in year_strategy_returns]) - 1) * 100)
        actual_yearly = ((np.prod([1 + r / 100
                                   for r in year_actual_returns]) - 1) * 100)
        year_data[
            'Yearly Total'] = f"T: {strategy_yearly:.1f}%\nA: {actual_yearly:.1f}%"
    else:
        year_data['Yearly Total'] = ""

    table_data[year] = year_data

# Convert to DataFrame
if table_data:
    df_rows = list(table_data.values())
    df_table = pd.DataFrame(df_rows)

    # Define styling function
    def style_cells(val):
        if not val or val == "":
            return ''

        # Extract theoretical and actual values for comparison
        try:
            lines = val.split('\n')
            if len(lines) == 2:
                theoretical = float(lines[0].split(': ')[1].replace('%', ''))
                actual = float(lines[1].split(': ')[1].replace('%', ''))
                diff = theoretical - actual

                if diff > 0:
                    return 'background-color: #d4edda; color: #155724'  # Light green for outperformance
                elif diff < 0:
                    return 'background-color: #f8d7da; color: #721c24'  # Light red for underperformance
                else:
                    return 'background-color: #fff3cd; color: #856404'  # Light yellow for equal
        except:
            pass
        return ''

    # Apply styling
    styled_df = df_table.style.map(
        style_cells, subset=[col for col in df_table.columns if col != 'Year'])

    # Display the table with legend
    st.markdown("**Legend:** T = Theoretical Strategy, A = Actual Returns")
    st.markdown(
        "🟢 Green = Strategy Outperformed | 🔴 Red = Strategy Underperformed | 🟡 Yellow = Equal Performance"
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)
else:
    st.warning("No data available for comparison table")

# Optimization Section
st.markdown("---")
st.subheader("🔍 Find Optimal Parameters")
st.markdown("Find the best parameters for a specific time period")
o1, o2, o3, o4 = st.columns(4)
start_month = o1.number_input("Start Month", 0, len(portfolio_value) - 1, 14)
end_month = o2.number_input("End Month", 0, len(portfolio_value) - 1, 30)
reb_filter = o3.selectbox("Rebalance Filter",
                          ["any", "monthly", "quarterly", "semi-yearly"])
run_opt = o4.button("🎯 Find Match", type="primary")

if run_opt:
    if end_month <= start_month:
        st.error("End month must be after start month.")
    else:
        with st.spinner(
                "Optimizing parameters... This may take a few minutes."):
            try:
                opt = find_optimal_parameters(start_month, end_month,
                                              reb_filter)
                if opt:
                    st.success("✅ Optimal parameters found!")
                    p1, p2 = st.columns(2)
                    with p1:
                        st.write("**📊 Optimal Parameters**")
                        st.write(f"- **Positions:** {opt['num_positions']}")
                        st.write(f"- **Cash %:** {opt['cash_percentage']}%")
                        st.write(
                            f"- **Rebalance:** {opt['rebalance_frequency']}")
                        st.write(f"- **Cost:** {opt['rebalance_cost']:.2f}%")
                    with p2:
                        st.write("**📈 Performance**")
                        st.write(f"- **Error (RMSE):** {opt['error']:.2f}%")
                    st.plotly_chart(create_optimization_chart(opt),
                                    use_container_width=True)
                else:
                    st.error("❌ No valid combination found.")
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Portfolio Strategy Analyzer - Built with Streamlit on Replit*")
