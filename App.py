# app.py - Updated with refactored functions
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
cash_percentage = st.sidebar.slider("Cash Percentage (%)", 0, 50, 0, 1)
rebalance_frequency = st.sidebar.selectbox(
    "Rebalance Frequency", ["monthly", "quarterly", "semi-yearly"], index=0)
rebalance_cost = st.sidebar.slider("Rebalance Cost (%)",
                                   0.0,
                                   5.0,
                                   2.00,
                                   0.01,
                                   format="%.2f")


# ─── Refactored Data Functions ─────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    """Load rank and price data from CSV files"""
    # Load rank data (tickers by date)
    rank_df = pd.read_csv('Rank.csv', index_col=0, parse_dates=True, dayfirst=True)

    # Load price data
    price_df = pd.read_csv('Prices.csv', index_col=0, parse_dates=True, dayfirst=True)

    return rank_df, price_df


@st.cache_data
def calculate_returns_from_prices(price_df):
    """Calculate daily returns from price data"""
    returns_df = price_df.pct_change(fill_method=None)  # Calculate returns as decimal
    returns_df = returns_df.fillna(0)  # Fill NaN values with 0
    # Convert to percentage strings to match reference logic
    returns_df = returns_df * 100
    returns_df = returns_df.astype(str) + '%'
    return returns_df


@st.cache_data
def generate_portfolio(rank_df, number_of_positions, rebalance_frequency):
    """Select top N holdings based on rank data"""
    frequency_mapping = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    rebalance_period = frequency_mapping[rebalance_frequency]

    portfolio = pd.DataFrame(index=rank_df.index, columns=range(number_of_positions))
    current_holdings = []

    for i, (date, row) in enumerate(rank_df.iterrows()):
        if i % rebalance_period == 0:
            # Get the ticker names for top N positions from the rank columns
            current_holdings = row.iloc[:number_of_positions].tolist()
        portfolio.loc[date] = current_holdings

    return portfolio


def get_portfolio_returns(portfolio, returns_df):
    """Get returns for each position in the portfolio"""
    portfolio_returns = pd.DataFrame(index=portfolio.index, 
                                   columns=portfolio.columns)

    portfolio_dates = portfolio.index
    returns_dates = returns_df.index
    min_length = min(len(portfolio_dates), len(returns_dates))

    for date in portfolio_dates[:min_length]:
        if date in returns_df.index:
            portfolio_row = portfolio.loc[date]
            current_returns = []

            for position in portfolio_row:
                if position in returns_df.columns:
                    return_value = returns_df.at[date, position]
                    current_returns.append(return_value)
                else:
                    current_returns.append(None)

            portfolio_returns.loc[date] = current_returns

    return portfolio_returns


def calculate_position_weights(num_positions, cash_percentage):
    """Calculate position weights with linear decrease"""
    investable = 100.0 - cash_percentage

    if num_positions <= 5:
        top_weight = 0.3 * investable
    else:
        top_weight = 0.3 * investable - (num_positions - 5) * 0.02 * investable - (15 - num_positions)

    if num_positions > 1:
        # Calculate linear decrease
        total_decrease = 2 * (top_weight * num_positions - investable)
        decrease_per_position = total_decrease / (num_positions * (num_positions - 1))

        weights = [top_weight - i * decrease_per_position for i in range(num_positions)]
    else:
        weights = [investable]

    # Convert to decimal form
    return [w / 100.0 for w in weights]


def get_rankings(date, rank_df):
    """Get ticker rankings for a specific date"""
    rankings = {}

    if date in rank_df.index:
        row = rank_df.loc[date]
        for rank, ticker in enumerate(row.values, 1):
            if pd.notna(ticker) and str(ticker).strip():
                rankings[str(ticker).strip()] = rank

    return rankings


def calculate_exposure_delta(portfolio, portfolio_growth, weights, rebalance_frequency):
    """
    Calculate exposure delta following the exact logic from rank_and_exposure_delta

    Returns series with summed exposure delta for each date
    """
    frequency_mapping = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    rebalance_period = frequency_mapping[rebalance_frequency]

    # Convert weights to percentage form to match reference
    reset_percentages = [w * 100 for w in weights]
    summed_exposure_delta = pd.Series(index=portfolio.index, dtype=float)
    num_positions = len(weights)

    for i, date in enumerate(portfolio.index):
        if i == 0:
            # First period: special case with multiplier
            summed_delta = 10 * num_positions
        elif i % rebalance_period == 0:
            # Rebalancing period: calculate delta
            summed_delta = 0
            for col_index in range(num_positions):
                reset_value = reset_percentages[col_index]
                stock_name = portfolio.iloc[i, col_index]

                # Find previous value of this stock if it was in the portfolio
                prev_value = None
                prev_date = portfolio.index[i - 1]
                if stock_name in portfolio.loc[prev_date].values:
                    stock_col_index = portfolio.loc[prev_date].tolist().index(stock_name)
                    prev_value = portfolio_growth.iloc[i - 1, stock_col_index]

                # Calculate delta
                if prev_value is None:
                    delta = reset_value
                else:
                    delta = abs(reset_value - prev_value)

                summed_delta += delta
        else:
            # Non-rebalancing period: no trading
            summed_delta = 0

        summed_exposure_delta.loc[date] = summed_delta

    return summed_exposure_delta


def calculate_portfolio_growth(portfolio, portfolio_returns, weights, rebalance_frequency):
    """Calculate portfolio growth values tracking position sizes over time"""
    frequency_mapping = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    rebalance_period = frequency_mapping[rebalance_frequency]

    portfolio_growth = pd.DataFrame(index=portfolio.index, 
                                  columns=portfolio.columns, 
                                  dtype=float)

    for i, date in enumerate(portfolio.index):
        if i % rebalance_period == 0:
            # Rebalance: reset to target weights (convert to percentage form)
            current_percentages = [w * 100 for w in weights]
        else:
            # No rebalance: use previous values
            current_percentages = portfolio_growth.iloc[i - 1].tolist()

        # Apply returns to get new values
        current_growth = []
        for col_index in range(len(weights)):
            return_value = portfolio_returns.iloc[i, col_index]

            # Handle string percentage conversion like in reference
            if isinstance(return_value, str):
                return_value = float(return_value.strip('%')) / 100
            elif return_value is None or pd.isna(return_value):
                return_value = 0
            else:
                return_value = return_value / 100  # Convert from percentage to decimal

            # Apply return to current percentage
            new_value = current_percentages[col_index] * (1 + return_value)
            current_growth.append(new_value)

        portfolio_growth.iloc[i] = current_growth

    return portfolio_growth


def calculate_gross_contribution(portfolio_growth, weights, rebalance_frequency):
    """
    Calculate gross contribution (return contribution before costs)
    Following exact logic from reference code
    """
    frequency_mapping = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    rebalance_period = frequency_mapping[rebalance_frequency]

    # Convert weights to percentage form to match reference
    reset_percentages = [w * 100 for w in weights]
    gross_contribution = pd.Series(index=portfolio_growth.index, dtype=float)
    num_positions = len(weights)

    for i, date in enumerate(portfolio_growth.index):
        total_contribution = 0

        for col_index in range(num_positions):
            current_value = portfolio_growth.iloc[i, col_index]

            if i == 0:
                # First period: contribution from initial weight
                contribution = current_value - reset_percentages[col_index]
            elif i % rebalance_period == 0:
                # Rebalance period: contribution from reset weight
                contribution = current_value - reset_percentages[col_index]
            else:
                # Regular period: contribution from previous value
                prev_value = portfolio_growth.iloc[i - 1, col_index]
                contribution = current_value - prev_value

            total_contribution += contribution

        gross_contribution.loc[date] = total_contribution

    return gross_contribution


def calculate_net_contribution(gross_contribution, exposure_delta, rebalance_cost):
    """
    Calculate net contribution by subtracting rebalancing costs
    Matches reference: net = gross - (exposure_delta * rebalance_cost/100)
    """
    rebalance_costs = exposure_delta * (rebalance_cost / 100)
    net_contribution = gross_contribution - rebalance_costs
    return net_contribution


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


def calculate_weighted_returns_with_rebalancing(portfolio, portfolio_returns, weights, 
                                                      rebalance_frequency, rebalance_cost, rank_df):
    """
    Calculate weighted returns with rebalancing costs
    Returns net contribution in percentage form
    """
    # Calculate portfolio growth (position values over time)
    portfolio_growth = calculate_portfolio_growth(portfolio, portfolio_returns, weights, rebalance_frequency)

    # Calculate exposure delta (trading costs)
    exposure_delta = calculate_exposure_delta(portfolio, portfolio_growth, weights, rebalance_frequency)

    # Calculate gross contribution (returns before costs)
    gross_contribution = calculate_gross_contribution(portfolio_growth, weights, rebalance_frequency)

    # Calculate net contribution (after rebalancing costs)
    net_contribution = calculate_net_contribution(gross_contribution, exposure_delta, rebalance_cost)

    # Return net contribution (already in correct form)
    return net_contribution


def calculate_portfolio_performance():
    """Main function to calculate portfolio performance"""

    # Check files exist
    check_required_files()

    # Load data
    rank_df, price_df = load_and_prepare_data()

    # Calculate returns from prices
    returns_df = calculate_returns_from_prices(price_df)

    # Generate portfolio based on rankings
    portfolio = generate_portfolio(rank_df, num_positions, rebalance_frequency)

    # Get returns for portfolio positions
    portfolio_returns = get_portfolio_returns(portfolio, returns_df)

    # Calculate position weights
    weights = calculate_position_weights(num_positions, cash_percentage)

    # Calculate net contribution with rebalancing costs
    net_contribution = calculate_weighted_returns_with_rebalancing(
        portfolio, portfolio_returns, weights, 
        rebalance_frequency, rebalance_cost, rank_df
    )

    # Convert to percentage form for compounding - net_contribution is already in percentage form
    net_contribution_pct = net_contribution / 100

    # Calculate cumulative portfolio value starting at 100
    portfolio_value = pd.Series(index=portfolio.index, dtype=float)
    investment_value = 100.0

    for date, contribution in net_contribution_pct.items():
        investment_value *= (1 + contribution)
        portfolio_value.loc[date] = investment_value

    return portfolio_value, portfolio


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
        # Load data
        rank_df, price_df = load_and_prepare_data()

        # Calculate returns from prices
        returns_df = calculate_returns_from_prices(price_df)

        # Generate portfolio based on rankings
        portfolio = generate_portfolio(rank_df, num_pos, rebal_freq)

        # Get returns for portfolio positions
        portfolio_returns = get_portfolio_returns(portfolio, returns_df)

        # Calculate position weights
        weights = calculate_position_weights(num_pos, cash_pct)

        # Calculate net contribution with rebalancing costs
        net_contribution = calculate_weighted_returns_with_rebalancing(
            portfolio, portfolio_returns, weights, 
            rebal_freq, rebal_cost, rank_df
        )

        # Convert to percentage form for compounding
        net_contribution_pct = net_contribution / 100

        # Calculate cumulative portfolio value
        portfolio_value = pd.Series(index=portfolio.index, dtype=float)
        value = 100.0

        for date, contribution in net_contribution_pct.items():
            value *= (1 + contribution)
            portfolio_value.loc[date] = value

        # Extract the period
        max_idx = len(portfolio_value) - 1
        start_idx = min(start_m, max_idx)
        end_idx = min(end_m, max_idx)

        if start_idx > end_idx:
            return None

        series = portfolio_value.iloc[start_idx:end_idx + 1].values
        if len(series) > 0:
            # Normalize to start at 100
            series = series * 100 / series[0]
        return series

    except Exception as e:
        print(f"Error in calculate_portfolio_for_period: {e}")
        return None


def calculate_volatility(values):
    """Calculate annualized volatility from portfolio values"""
    if len(values) < 2:
        return 0.0

    # Calculate monthly returns
    returns = []
    for i in range(1, len(values)):
        monthly_return = (values[i] / values[i-1] - 1)
        returns.append(monthly_return)

    if len(returns) == 0:
        return 0.0

    # Calculate standard deviation of returns
    returns_std = np.std(returns, ddof=1)

    # Annualize volatility (multiply by sqrt(12) for monthly data)
    annualized_volatility = returns_std * np.sqrt(12) * 100

    return annualized_volatility


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


# ─── Excel Download Functions ────────────────────────────────────────────────
def create_contribution_excel(portfolio, portfolio_returns, weights, rebalance_frequency, rebalance_cost, rank_df, contribution_type='gross'):
    """Create Excel file with returns by rank position"""

    # Load the returns data properly
    rank_df, price_df = load_and_prepare_data()
    returns_df = calculate_returns_from_prices(price_df)

    # Create DataFrame with dates and rank positions
    data = {'Date': portfolio.index}

    # Store raw returns for monthly total calculation
    raw_returns_by_date = {}

    # Add columns for each rank position
    for i in range(len(weights)):
        rank_col = f'Rank {i+1}'
        data[rank_col] = []

        for date in portfolio.index:
            # Get the stock in this rank position for this date
            stock_ticker = portfolio.loc[date, i]

            # Get the return for this stock on this date from the returns_df
            if stock_ticker in returns_df.columns and date in returns_df.index:
                return_value = returns_df.at[date, stock_ticker]

                # Convert string percentage to float if needed
                if isinstance(return_value, str) and return_value.endswith('%'):
                    return_value = float(return_value.replace('%', ''))
                elif return_value is None or pd.isna(return_value):
                    return_value = 0.0
                elif isinstance(return_value, (int, float)):
                    # Already a number, use as is
                    pass
                else:
                    return_value = 0.0

                # Store raw return for monthly total calculation
                if date not in raw_returns_by_date:
                    raw_returns_by_date[date] = []
                raw_returns_by_date[date].append(return_value)

                # Format the display value as "Stock: Return%"
                display_value = f"{stock_ticker}: {return_value:.2f}%"
            else:
                # Store 0 for monthly total calculation
                if date not in raw_returns_by_date:
                    raw_returns_by_date[date] = []
                raw_returns_by_date[date].append(0.0)

                display_value = f"{stock_ticker}: 0.00%"

            data[rank_col].append(display_value)

    # Add Monthly Total column (weighted returns)
    data['Monthly Total'] = []
    for date in portfolio.index:
        if date in raw_returns_by_date:
            # Calculate weighted returns: weight * return for each position, then sum
            weighted_sum = 0.0
            for i in range(len(weights)):
                position_weight = weights[i] * 100  # Convert to percentage
                position_return = raw_returns_by_date[date][i] if i < len(raw_returns_by_date[date]) else 0.0
                weighted_sum += (position_weight * position_return / 100)  # Weight * Return
            data['Monthly Total'].append(f"{weighted_sum:.2f}%")
        else:
            data['Monthly Total'].append("0.00%")

    # Create DataFrame
    df = pd.DataFrame(data)

    return df

def generate_excel_download(df, filename):
    """Generate Excel file for download"""
    import io

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Returns by Rank', index=False)

        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Returns by Rank']

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    output.seek(0)
    return output.getvalue()

# ─── Main App ─────────────────────────────────────────────────────────────────
st.title("Portfolio Strategy Analyzer")
st.markdown("Analyze and optimize portfolio strategies using historical data")
st.markdown("---")

# Calculate portfolio performance
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
c1, c2, c3, c4, c5, c6 = st.columns(6)
sf = portfolio_value.iloc[-1]

# Handle case where portfolio data is longer than actual data
if len(portfolio_value) < len(actual):
    af = actual[len(portfolio_value)]
    ar = (af - 100) / 100 * 100
    outperformance_available = True
    actual_for_volatility = actual[:len(portfolio_value)]
else:
    # Use the last available actual data point
    af = actual[-1] if actual else 100.0
    ar = (af - 100) / 100 * 100 if actual else 0.0
    outperformance_available = len(actual) > 0
    actual_for_volatility = actual

sr = (sf - 100) / 100 * 100

# Calculate volatilities
strategy_volatility = calculate_volatility(portfolio_value.values)
actual_volatility = calculate_volatility(actual_for_volatility) if actual_for_volatility else 0.0

c1.metric("Strategy Final", f"${sf:.2f}", f"{sr:+.1f}%")
c2.metric("Actual Final", f"${af:.2f}", f"{ar:+.1f}%")
if outperformance_available:
    c3.metric("Outperformance", f"{(sr-ar):+.1f}%")
else:
    c3.metric("Outperformance", "N/A")
yrs = len(portfolio_value) / 12
c4.metric("Annualized", f"{((sf/100)**(1/yrs)-1)*100:.1f}%")
c5.metric("Strategy Volatility", f"{strategy_volatility:.1f}%")
c6.metric("Actual Volatility", f"{actual_volatility:.1f}%")

st.caption("💡 Volatility is calculated as the annualized standard deviation of monthly returns (std × √12)")


# Monthly Comparison Table
st.markdown("---")
st.subheader("Monthly Performance Comparison")

# Create monthly comparison data
months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
    'Nov', 'Dec'
]
years = ['2018', '2019', '2020', '2021', '2022']

# Get actual returns data
gross_returns = get_actual_gross_returns()

# Calculate strategy monthly returns properly
strategy_monthly_returns = []

# Calculate month-over-month returns from portfolio value
for i in range(len(portfolio_value) - 1):
    monthly_return = (portfolio_value.iloc[i + 1] / portfolio_value.iloc[i] - 1) * 100
    strategy_monthly_returns.append(monthly_return)

# Create restructured table with years as rows and months as columns
table_data = {}
month_idx = 0

for year in years:
    year_data = {'Year': year}
    year_strategy_returns = []
    year_actual_returns = []

    for month in months:
        if month_idx < len(gross_returns) and month_idx < len(strategy_monthly_returns):
            actual_ret = gross_returns[month_idx]
            strategy_ret = strategy_monthly_returns[month_idx]

            year_strategy_returns.append(strategy_ret)
            year_actual_returns.append(actual_ret)

            # Format cell with both values
            cell_value = f"T: {strategy_ret:.1f}%\nA: {actual_ret:.1f}%"
            year_data[month] = cell_value
            month_idx += 1
        elif month_idx < len(strategy_monthly_returns):
            # Only strategy data available (new data beyond actual)
            strategy_ret = strategy_monthly_returns[month_idx]
            year_strategy_returns.append(strategy_ret)

            cell_value = f"T: {strategy_ret:.1f}%\nA: N/A"
            year_data[month] = cell_value
            month_idx += 1
        elif month_idx < len(gross_returns):
            # Only actual data available
            actual_ret = gross_returns[month_idx]
            year_actual_returns.append(actual_ret)

            cell_value = f"T: N/A\nA: {actual_ret:.1f}%"
            year_data[month] = cell_value
            month_idx += 1
        else:
            year_data[month] = ""

    # Hardcoded actual yearly returns
    hardcoded_actual_yearly = {
        '2018': 14.5,
        '2019': 36.5,
        '2020': 37.7,
        '2021': 10.6,
        '2022': -34.3
    }
    
    # Calculate yearly totals
    if year_strategy_returns:
        strategy_yearly = (
            (np.prod([1 + r / 100 for r in year_strategy_returns]) - 1) * 100)
        actual_yearly = hardcoded_actual_yearly.get(year, 0.0)
        year_data[
            'Yearly Total'] = f"T: {strategy_yearly:.1f}%\nA: {actual_yearly:.1f}%"
    else:
        actual_yearly = hardcoded_actual_yearly.get(year, 0.0)
        year_data['Yearly Total'] = f"T: N/A\nA: {actual_yearly:.1f}%"

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
st.subheader("🔍 Find Closest Match to the Actual Portfolio")
st.markdown("Find the best parameters for a specific time period")
o1, o2, o3, o4 = st.columns(4)
start_month = o1.number_input("Start Month", 0, len(portfolio_value) - 1, 14)
end_month = o2.number_input("End Month", 0, len(portfolio_value) - 1, 30)
reb_filter = o3.selectbox("Rebalance Filter",
                          ["any", "monthly", "quarterly", "semi-yearly"])
run_opt = o4.button("Find Match", type="primary")

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
                    st.success("Optimal parameters found!")
                    p1, p2 = st.columns(2)
                    with p1:
                        st.write("**Optimal Parameters**")
                        st.write(f"- **Positions:** {opt['num_positions']}")
                        st.write(f"- **Cash %:** {opt['cash_percentage']}%")
                        st.write(
                            f"- **Rebalance:** {opt['rebalance_frequency']}")
                        st.write(f"- **Cost:** {opt['rebalance_cost']:.2f}%")
                    with p2:
                        st.write("**Performance**")
                        st.write(f"- **Error (RMSE):** {opt['error']:.2f}%")
                    st.plotly_chart(create_optimization_chart(opt),
                                    use_container_width=True)
                else:
                    st.error("No valid combination found.")
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")

# Position Weightings Display
st.markdown("---")
st.subheader("Position Weightings")

# Calculate current weights based on sidebar parameters
current_weights = calculate_position_weights(num_positions, cash_percentage)

st.write("**Position Weightings:**")
for i, weight in enumerate(current_weights):
    st.write(f"Rank {i+1}: {weight*100:.2f}%")

# Show cash percentage if applicable
if cash_percentage > 0:
    st.write(f"Cash: {cash_percentage:.2f}%")

# Show total allocation
total_invested = sum(current_weights) * 100
st.write(f"**Total Invested: {total_invested:.2f}%**")

# Footer
st.markdown("---")
st.markdown("*Portfolio Strategy Analyzer - Built with Streamlit on Replit*")
