# app.py - Portfolio Strategy Analyzer (corrected & consistent units)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & DATA STRUCTURES
# ============================================================================

@dataclass
class PortfolioConfig:
    """Configuration for portfolio parameters"""
    num_positions: int
    cash_percentage: float     # e.g., 8 = 8%
    rebalance_frequency: str   # 'monthly' | 'quarterly' | 'semi-yearly'
    rebalance_cost: float      # trading cost as %, e.g., 0.5 = 0.5%

@dataclass
class MarketData:
    """Container for market data"""
    rank_df: pd.DataFrame      # rows = dates, cols = tickers, values = rank (lower is better)
    price_df: pd.DataFrame     # rows = dates, cols = tickers, values = price
    returns_df: pd.DataFrame   # rows = dates, cols = tickers, values = decimal returns

# Global configuration
INITIAL_INVESTMENT = 100.0
ACTUAL_RETURNS_DATA = {
    '2018': [4.7, 0.4, 1.2, 2.8, 5.1, 5.4, 1.1, 7.1, 0.7, -8.5, 3.4, -8.3],
    '2019': [8.6, 8.9, 3.2, 4.9, -2.5, 6.7, 3.2, -0.4, -6.5, 0.4, 5.5, 0.6],
    '2020': [5.5, -6.6, -14.3, 14.2, 9.0, 3.9, 5.9, 6.6, -3.1, -3.7, 11.8, 7.2],
    '2021': [-2.5, 8.2, -6.8, 4.9, -6.3, 6.3, 3.6, 5.2, -2.2, 3.1, -1.8, -0.1],
    '2022': [-12.9, -0.5, -2.1, -8.9, -9.5, -8.2, 9.1, -3.1, -8.1, 3.6, 4.0, -2.4]
}

ACTUAL_YEARLY_RETURNS = {
    '2018': 14.5,
    '2019': 36.5,
    '2020': 37.7,
    '2021': 10.6,
    '2022': -34.3
}

FREQUENCY_MAP = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def validate_data_files() -> bool:
    """Check if required data files exist"""
    rank_file = Path("Rank.csv")
    prices_file = Path("Prices.csv")

    if not rank_file.exists() or not prices_file.exists():
        missing = []
        if not rank_file.exists():
            missing.append("Rank.csv")
        if not prices_file.exists():
            missing.append("Prices.csv")

        st.error(f"Missing required files: {', '.join(missing)}")
        st.info("Please upload the required files to your workspace")
        return False
    return True

@st.cache_data
def load_market_data() -> MarketData:
    """Load and prepare all market data"""
    rank_df = pd.read_csv('Rank.csv', index_col=0, parse_dates=True, dayfirst=True)
    price_df = pd.read_csv('Prices.csv', index_col=0, parse_dates=True, dayfirst=True)

    # Normalize column names a bit (helps common dot/dash variants)
    def _norm(cols):
        return (pd.Index(cols)
                  .astype(str)
                  .str.strip()
                  .str.replace(' ', '', regex=False))
    rank_df.columns = _norm(rank_df.columns)
    price_df.columns = _norm(price_df.columns)

    # Align by dates
    common_index = rank_df.index.intersection(price_df.index)
    rank_df = rank_df.loc[common_index].sort_index()
    price_df = price_df.loc[common_index].sort_index()

    # Align by tickers (IMPORTANT)
    common_cols = rank_df.columns.intersection(price_df.columns)
    missing_in_prices = set(rank_df.columns) - set(common_cols)
    missing_in_rank   = set(price_df.columns) - set(common_cols)
    if missing_in_prices:
        st.warning(f"{len(missing_in_prices)} tickers present in Rank.csv but missing in Prices.csv "
                   f"(showing up to 10): {sorted(list(missing_in_prices))[:10]}")
    if missing_in_rank:
        st.info(f"{len(missing_in_rank)} tickers present in Prices.csv but not in Rank.csv "
                f"(showing up to 10): {sorted(list(missing_in_rank))[:10]}")

    rank_df = rank_df[common_cols]
    price_df = price_df[common_cols]

    returns_df = price_df.pct_change(fill_method=None).fillna(0.0)

    return MarketData(rank_df, price_df, returns_df)


def get_actual_portfolio_returns() -> List[float]:
    """Get flattened list of actual portfolio returns (% per month)"""
    returns = []
    for year in sorted(ACTUAL_RETURNS_DATA.keys()):
        returns.extend(ACTUAL_RETURNS_DATA[year])
    return returns

# ============================================================================
# CORE FUNCTIONS (fixed & consistent)
# ============================================================================

def calculate_position_percentages(num_positions: int, cash_percentage: float) -> List[float]:
    """
    Calculate descending position weights (percent points) for the investable sleeve
    using an arithmetic series. Ensures no negatives and sums to (100 - cash).
    """
    total_pp = max(0.0, 100.0 - float(cash_percentage))
    n = int(num_positions)

    if n <= 0 or total_pp == 0:
        return [0.0] * max(0, n)

    # Highest weight baseline: 30% of investable for <=5; linearly diminish after that (by 2% of investable per extra name)
    if n <= 5:
        a = 0.30 * total_pp
    else:
        a = 0.30 * total_pp - (n - 5) * 0.02 * total_pp

    # Common difference so that sum of arithmetic series equals total_pp
    if n == 1:
        weights = [total_pp]
    else:
        d = (2 * a - 2 * (total_pp / n)) / (n - 1)
        weights = [a - i * d for i in range(n)]

    # If any negatives due to shape, floor at zero and renormalize to total_pp
    weights = [max(0.0, w) for w in weights]
    s = sum(weights)
    if s == 0:
        # fallback: equal weights
        weights = [total_pp / n] * n
    else:
        weights = [w * total_pp / s for w in weights]

    return weights  # percent points, sum ≈ 100 - cash%

def generate_portfolio(number_of_positions: int, rebalance_frequency: str, rank_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate portfolio holdings based on rank data.
    Uses row-wise sort (ascending rank) to pick top-N tickers at each rebalance.
    """
    period = FREQUENCY_MAP[rebalance_frequency]
    portfolio = pd.DataFrame(index=rank_data.index, columns=range(number_of_positions), dtype=object)

    current_holdings: List[str] = []
    for i, (dt, row) in enumerate(rank_data.iterrows()):
        if i % period == 0:
            # lower rank is better
            sorted_tickers = row.sort_values(ascending=True).index.tolist()
            current_holdings = sorted_tickers[:number_of_positions]
        portfolio.loc[dt] = current_holdings
    return portfolio

def generate_returns_portfolio(portfolio: pd.DataFrame, returns_data: pd.DataFrame) -> pd.DataFrame:
    """
    Map returns (decimal) to portfolio holdings (tickers) per date.
    Missing returns are treated as 0.0 (conservative).
    """
    common_index = portfolio.index.intersection(returns_data.index)
    portfolio = portfolio.loc[common_index]
    returns_data = returns_data.loc[common_index]

    ret_port = pd.DataFrame(index=common_index, columns=portfolio.columns, dtype=float)
    for dt in common_index:
        row = portfolio.loc[dt]
        vals: List[float] = []
        for pos in row:
            if pos in returns_data.columns:
                vals.append(float(returns_data.at[dt, pos]))
            else:
                vals.append(0.0)  # explicit: missing treated as flat
        ret_port.loc[dt] = vals
    return ret_port

def simulate_portfolio(num_positions: int,
                       cash_percentage: float,
                       returns_portfolio: pd.DataFrame,
                       rebalance_frequency: str,
                       rebalance_cost: float):
    """
    Core simulation with consistent units:
      - Internal weights are FRACTIONS of total portfolio (sum to 1, include cash).
      - Returns are decimal.
      - Costs are % of traded notional; turnover = 0.5 * L1 distance between weight vectors.
      - Cost is subtracted from the period return.
    Returns:
      portfolio_value (Series), exposure_delta_pp (Series),
      gross_return_pct (Series), net_return_pct (Series),
      weights_history_pp (DataFrame of start-of-period non-cash weights in percent points)
    """
    period = FREQUENCY_MAP[rebalance_frequency]
    index = returns_portfolio.index
    n = num_positions

    # Target non-cash weights in percent points and in fractions
    target_pp = calculate_position_percentages(n, cash_percentage)
    investable_frac = (100.0 - cash_percentage) / 100.0
    target_frac = np.array([w / 100.0 for w in target_pp], dtype=float)
    cash_target_frac = cash_percentage / 100.0

    # State: start 100% in cash
    w_current = np.zeros(n + 1, dtype=float)  # [positions..., cash]
    w_current[-1] = 1.0

    portfolio_value = []
    exposure_delta_pp = []
    gross_return_pct = []
    net_return_pct = []
    weights_history_pp = pd.DataFrame(index=index, columns=range(n), dtype=float)

    V = INITIAL_INVESTMENT

    for i, dt in enumerate(index):
        rebalancing = (i % period == 0)

        # --- Start-of-period rebalancing & cost ---
        cost_pct_this_period = 0.0
        if rebalancing:
            w_target = np.concatenate([target_frac, [cash_target_frac]])  # fractions
            # L1 distance (fractions)
            l1 = float(np.sum(np.abs(w_current - w_target)))
            # Exposure delta in percent points for UI/debug (matches 2*turnover*100)
            exposure_pp = 100.0 * l1
            # Turnover fraction is half the L1 distance
            turnover = 0.5 * l1
            # Cost as % of portfolio (percentage points)
            cost_pct_this_period = rebalance_cost * turnover  # rebalance_cost already a %
            # Apply new weights post-trade (fractions)
            w_current = w_target.copy()
        else:
            exposure_pp = 0.0

        # Record start-of-period non-cash weights (percent points)
        weights_history_pp.loc[dt] = (w_current[:-1] * 100.0)

        # --- Returns for this period ---
        r_list = np.array(returns_portfolio.iloc[i].astype(float).fillna(0.0).values, dtype=float)  # decimals
        # Gross portfolio return (decimal) – cash earns 0
        gross_r = float(np.sum(w_current[:-1] * r_list))
        gross_r_pct = gross_r * 100.0

        # Net period return in percent points
        net_r_pct = gross_r_pct - cost_pct_this_period

        # Update portfolio value
        V *= (1.0 + net_r_pct / 100.0)

        # Drift weights to end-of-period (fractions), cash has 0 return
        denom = 1.0 + gross_r
        if denom <= 0:  # guard against pathological returns
            denom = 1e-9
        w_pos_end = (w_current[:-1] * (1.0 + r_list)) / denom
        w_cash_end = (w_current[-1]) / denom
        w_current = np.concatenate([w_pos_end, [w_cash_end]])

        # Store outputs
        portfolio_value.append(V)
        exposure_delta_pp.append(exposure_pp)
        gross_return_pct.append(gross_r_pct)
        net_return_pct.append(net_r_pct)

    pv_series = pd.Series(portfolio_value, index=index, name='Portfolio Value')
    exposure_series = pd.Series(exposure_delta_pp, index=index, name='Summed Exposure Delta (pp)')
    gross_series = pd.Series(gross_return_pct, index=index, name='Gross Return (%)')
    net_series = pd.Series(net_return_pct, index=index, name='Net Return (%)')

    return pv_series, exposure_series, gross_series, net_series, weights_history_pp

# ============================================================================
# MAIN PORTFOLIO CALCULATION PIPELINE
# ============================================================================

def execute_portfolio_strategy(config: PortfolioConfig) -> Tuple[pd.Series, pd.DataFrame, Dict]:
    """
    Execute the complete portfolio methodology with corrected logic.
    Returns: (portfolio_value_series, portfolio_holdings, debug_info)
    """
    if not validate_data_files():
        st.stop()

    market_data = load_market_data()

    # 1) Holdings from rank
    portfolio = generate_portfolio(
        config.num_positions,
        config.rebalance_frequency,
        market_data.rank_df
    )

    # 2) Map returns to those holdings (decimals)
    returns_portfolio = generate_returns_portfolio(
        portfolio,
        market_data.returns_df
    )

    # 3) Simulate portfolio with turnover costs
    pv, exposure_delta, gross_ret, net_ret, weights_hist_pp = simulate_portfolio(
        config.num_positions,
        config.cash_percentage,
        returns_portfolio,
        config.rebalance_frequency,
        config.rebalance_cost
    )

    debug_info = {
        'weights_target_pp': calculate_position_percentages(config.num_positions, config.cash_percentage),
        'exposure_delta_pp': exposure_delta,
        'gross_return_pct': gross_ret,
        'net_return_pct': net_ret,
        'weights_history_pp': weights_hist_pp,
    }

    return pv, portfolio, debug_info

# ============================================================================
# PERFORMANCE ANALYTICS
# ============================================================================

def calculate_return_metrics(values: np.ndarray) -> Dict[str, float]:
    """Calculate various return metrics assuming monthly frequency."""
    if len(values) < 2:
        return {'volatility': 0.0, 'total_return': 0.0, 'annualized_return': 0.0}

    # Monthly returns from value series
    rets = np.diff(values) / values[:-1]

    # Volatility (annualized, monthly → *sqrt(12))
    volatility = float(np.std(rets, ddof=1) * np.sqrt(12) * 100.0)

    # Total return
    total_return = float(((values[-1] / values[0]) - 1.0) * 100.0)

    # Annualized return
    years = len(values) / 12.0
    annualized_return = float((((values[-1] / values[0]) ** (1.0 / years)) - 1.0) * 100.0) if years > 0 else 0.0

    return {
        'volatility': volatility,
        'total_return': total_return,
        'annualized_return': annualized_return
    }

def calculate_tracking_error(strategy: np.ndarray, benchmark: np.ndarray) -> float:
    """Calculate RMSE tracking error between strategy and benchmark (both normalized)."""
    if len(strategy) != len(benchmark) or len(strategy) == 0:
        return float('inf')

    rel_errors_sq = []
    for s, b in zip(strategy, benchmark):
        if b != 0:
            err = ((s - b) / b) * 100.0
        else:
            err = s - b
        rel_errors_sq.append(err * err)
    return float(np.sqrt(np.mean(rel_errors_sq)))

# ============================================================================
# PORTFOLIO OPTIMIZATION
# ============================================================================

def optimize_portfolio_parameters(target_start: int,
                                  target_end: int,
                                  rebalance_filter: str) -> Optional[Dict]:
    """
    Grid search to minimize tracking error vs. actual portfolio values (ACTUAL_RETURNS_DATA).
    """
    # Build actual value path (start at 100)
    actual_returns = get_actual_portfolio_returns()
    actual_values = [INITIAL_INVESTMENT]
    for ret in actual_returns:
        actual_values.append(actual_values[-1] * (1.0 + ret / 100.0))

    # Period guard
    if target_start >= len(actual_values) or target_end >= len(actual_values) or target_end <= target_start:
        return None

    # Normalize actual subperiod to 100
    actual_period = np.array(actual_values[target_start:target_end + 1], dtype=float)
    actual_period = actual_period * 100.0 / actual_period[0]

    # Search space
    search_space = {
        'positions': range(5, 16),
        'cash': range(0, 31, 5),
        'rebalance': ['monthly', 'quarterly', 'semi-yearly'] if rebalance_filter == 'any' else [rebalance_filter],
        'cost': [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    }

    total = (len(search_space['positions']) *
             len(search_space['cash']) *
             len(search_space['rebalance']) *
             len(search_space['cost']))

    progress = st.progress(0)
    status = st.empty()

    best_result = None
    best_error = float('inf')
    tested = 0

    for positions in search_space['positions']:
        for cash in search_space['cash']:
            for rebalance in search_space['rebalance']:
                for cost in search_space['cost']:
                    tested += 1
                    status.text(f"Testing {tested}/{total}: pos={positions}, cash={cash}%, reb={rebalance}, cost={cost:.2f}%")

                    config = PortfolioConfig(positions, cash, rebalance, cost)

                    try:
                        portfolio_value, _, _ = execute_portfolio_strategy(config)

                        if target_end < len(portfolio_value):
                            # Align to same span as actual
                            strategy_period = portfolio_value.iloc[target_start:target_end + 1].values.astype(float)
                            strategy_period = strategy_period * 100.0 / strategy_period[0]

                            if len(strategy_period) == len(actual_period):
                                error = calculate_tracking_error(strategy_period, actual_period)

                                if error < best_error:
                                    best_error = error
                                    best_result = {
                                        'num_positions': positions,
                                        'cash_percentage': cash,
                                        'rebalance_frequency': rebalance,
                                        'rebalance_cost': cost,
                                        'error': error,
                                        'strategy_values': strategy_period,
                                        'actual_values': actual_period
                                    }
                    except Exception:
                        # Skip bad combos or data issues
                        pass

                    progress.progress(tested / total)

    progress.empty()
    status.empty()
    return best_result

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_performance_chart(portfolio_value: pd.Series) -> Tuple[go.Figure, List]:
    """Create main performance comparison chart against the 'Actual' series."""
    actual_returns = get_actual_portfolio_returns()
    actual_values = [INITIAL_INVESTMENT]
    # Match the number of points to strategy series length
    for ret in actual_returns[:len(portfolio_value)]:
        actual_values.append(actual_values[-1] * (1.0 + ret / 100.0))

    fig = go.Figure()
    # Strategy
    fig.add_trace(go.Scatter(
        x=list(range(len(portfolio_value))),
        y=portfolio_value.values,
        mode="lines",
        name="Strategy",
        line=dict(width=2, color='#1f77b4')
    ))
    # Actual (trim to same number of points as strategy)
    fig.add_trace(go.Scatter(
        x=list(range(len(actual_values[:len(portfolio_value)]))),
        y=actual_values[:len(portfolio_value)],
        mode="lines",
        name="Actual",
        line=dict(width=2, dash="dash", color='#ff7f0e')
    ))
    fig.update_layout(
        template="simple_white",
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Months",
        yaxis_title="Portfolio Value ($)",
        height=500,
    )
    return fig, actual_values

def create_optimization_chart(result: Dict) -> go.Figure:
    """Create optimization result chart"""
    x = list(range(len(result['strategy_values'])))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=result['strategy_values'],
        mode='lines',
        name='Optimal Strategy',
        line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=result['actual_values'],
        mode='lines',
        name='Actual',
        line=dict(color='red', width=2, dash='dash')
    ))
    fig.update_layout(
        template='simple_white',
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Months in Period",
        yaxis_title="Normalized Value",
        height=400
    )
    return fig

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================

def initialize_streamlit():
    """Initialize Streamlit configuration"""
    st.set_page_config(
        page_title="Portfolio Strategy Analyzer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Hide Streamlit branding
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

def create_parameter_sidebar() -> PortfolioConfig:
    """Create sidebar with parameter controls"""
    st.sidebar.header("Portfolio Parameters")

    num_positions = st.sidebar.slider("Number of Positions", 5, 15, 10, 1)
    cash_percentage = st.sidebar.slider("Cash Percentage (%)", 0, 50, 8, 1)
    rebalance_frequency = st.sidebar.selectbox(
        "Rebalance Frequency",
        ["monthly", "quarterly", "semi-yearly"],
        index=0
    )
    rebalance_cost = st.sidebar.slider(
        "Rebalance Cost (%)",
        0.0, 5.0, 0.5, 0.01,
        format="%.2f"
    )

    return PortfolioConfig(
        num_positions=num_positions,
        cash_percentage=cash_percentage,
        rebalance_frequency=rebalance_frequency,
        rebalance_cost=rebalance_cost
    )

def display_performance_metrics(portfolio_value: pd.Series, actual_values: List):
    """Display key performance metrics"""
    st.markdown("---")

    strategy_metrics = calculate_return_metrics(portfolio_value.values)
    actual_metrics = calculate_return_metrics(np.array(actual_values[:len(portfolio_value)]))

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    strategy_final = float(portfolio_value.iloc[-1])
    actual_final = float(actual_values[min(len(portfolio_value), len(actual_values)-1)])

    c1.metric("Strategy Final", f"${strategy_final:.2f}",
              f"{strategy_metrics['total_return']:+.1f}%")
    c2.metric("Actual Final", f"${actual_final:.2f}",
              f"{actual_metrics['total_return']:+.1f}%")
    c3.metric("Outperformance",
              f"{strategy_metrics['total_return'] - actual_metrics['total_return']:+.1f}%")
    c4.metric("Annualized", f"{strategy_metrics['annualized_return']:.1f}%")
    c5.metric("Strategy Volatility", f"{strategy_metrics['volatility']:.1f}%")
    c6.metric("Actual Volatility", f"{actual_metrics['volatility']:.1f}%")

    st.caption("💡 Volatility is calculated as the annualized standard deviation of monthly returns.")

def display_monthly_comparison_table(portfolio_value: pd.Series):
    """Display monthly performance comparison table"""
    st.markdown("---")
    st.subheader("Monthly Performance Comparison")

    # Strategy monthly returns (%)
    strategy_returns = []
    for i in range(len(portfolio_value) - 1):
        ret = (portfolio_value.iloc[i + 1] / portfolio_value.iloc[i] - 1.0) * 100.0
        strategy_returns.append(float(ret))

    actual_returns = get_actual_portfolio_returns()

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    years = ['2018', '2019', '2020', '2021', '2022']

    table_data = []
    month_idx = 0

    for year in years:
        year_row = {'Year': year}
        year_strategy = []
        year_actual = []

        for month in months:
            if month_idx < min(len(strategy_returns), len(actual_returns)):
                s_ret = strategy_returns[month_idx]
                a_ret = actual_returns[month_idx]
                year_strategy.append(s_ret)
                year_actual.append(a_ret)
                year_row[month] = f"T: {s_ret:.1f}%\nA: {a_ret:.1f}%"
                month_idx += 1
            else:
                year_row[month] = ""

        if year_strategy:
            s_yearly = (np.prod([1 + r/100 for r in year_strategy]) - 1) * 100
            a_yearly = ACTUAL_YEARLY_RETURNS.get(year, 0.0)
            year_row['Yearly Total'] = f"T: {s_yearly:.1f}%\nA: {a_yearly:.1f}%"

        table_data.append(year_row)

    df = pd.DataFrame(table_data)

    def style_cell(val):
        if not val or '\n' not in val:
            return ''
        try:
            lines = val.split('\n')
            t_val = float(lines[0].split(': ')[1].replace('%', ''))
            a_val = float(lines[1].split(': ')[1].replace('%', ''))
            diff = t_val - a_val
            if diff > 0:
                return 'background-color: #d4edda; color: #155724'
            elif diff < 0:
                return 'background-color: #f8d7da; color: #721c24'
            else:
                return 'background-color: #fff3cd; color: #856404'
        except Exception:
            return ''

    styled_df = df.style.map(style_cell, subset=[c for c in df.columns if c != 'Year'])

    st.markdown("**Legend:** T = Theoretical, A = Actual")
    st.markdown("🟢 Outperformance | 🔴 Underperformance | 🟡 Equal")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

def display_optimization_section(portfolio_value: pd.Series):
    """Display parameter optimization section"""
    st.markdown("---")
    st.subheader("🔍 Find Closest Match to Actual Portfolio")

    col1, col2, col3, col4 = st.columns(4)

    start = col1.number_input("Start Month", 0, max(0, len(portfolio_value)-1), 14)
    end = col2.number_input("End Month", 0, max(0, len(portfolio_value)-1), min(30, max(0, len(portfolio_value)-1)))
    reb_filter = col3.selectbox("Rebalance Filter",
                                ["any", "monthly", "quarterly", "semi-yearly"])
    optimize = col4.button("Find Match", type="primary")

    if optimize:
        if end <= start:
            st.error("End month must be after start month")
        else:
            with st.spinner("Optimizing parameters..."):
                result = optimize_portfolio_parameters(start, end, reb_filter)

                if result:
                    st.success("Optimal parameters found!")

                    p1, p2 = st.columns(2)
                    with p1:
                        st.write("**Optimal Parameters**")
                        st.write(f"- Positions: {result['num_positions']}")
                        st.write(f"- Cash: {result['cash_percentage']}%")
                        st.write(f"- Rebalance: {result['rebalance_frequency']}")
                        st.write(f"- Cost: {result['rebalance_cost']:.2f}%")

                    with p2:
                        st.write("**Performance**")
                        st.write(f"- Error (RMSE): {result['error']:.2f}%")

                    fig = create_optimization_chart(result)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No valid combination found")

def display_weight_breakdown(config: PortfolioConfig):
    """Display position weight breakdown"""
    st.markdown("---")
    st.subheader("Position Weightings")

    weights = calculate_position_percentages(config.num_positions, config.cash_percentage)

    st.write("**Position Weights (percent of total portfolio):**")
    for i, weight in enumerate(weights):
        st.write(f"Rank {i+1}: {weight:.2f}%")

    if config.cash_percentage > 0:
        st.write(f"Cash: {config.cash_percentage:.2f}%")

    st.write(f"**Total Invested: {sum(weights):.2f}%**")

def display_debug_info(debug_info: Dict, config: PortfolioConfig):
    """Display debug information for verification"""
    with st.expander("Debug Information"):
        st.write("**Target Non-Cash Weights (pp):**", debug_info['weights_target_pp'])

        st.write("**First 10 Exposure Deltas (pp):**")
        st.dataframe(debug_info['exposure_delta_pp'].head(10))

        rebalance_events = debug_info['exposure_delta_pp'][debug_info['exposure_delta_pp'] > 0]
        st.write(f"**Rebalancing Events:** {len(rebalance_events)}")
        st.dataframe(rebalance_events.head(10))

        st.write("**Start-of-Period Weights (pp) - first 5 rows:**")
        st.dataframe(debug_info['weights_history_pp'].head(5))

        st.write("**Sample Period Returns (first 5):**")
        gross = debug_info['gross_return_pct'].head(5)
        net = debug_info['net_return_pct'].head(5)
        df = pd.concat([gross, net], axis=1)
        st.dataframe(df)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    initialize_streamlit()

    st.title("Portfolio Strategy Analyzer")
    st.markdown("Analyze and optimize portfolio strategies using historical data")
    st.markdown("---")

    config = create_parameter_sidebar()

    # Calculate portfolio
    try:
        portfolio_value, holdings, debug_info = execute_portfolio_strategy(config)
    except Exception as e:
        st.error(f"Error calculating portfolio: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        st.stop()

    # Display results
    fig, actual_values = create_performance_chart(portfolio_value)
    st.plotly_chart(fig, use_container_width=True)

    display_performance_metrics(portfolio_value, actual_values)
    display_monthly_comparison_table(portfolio_value)
    display_optimization_section(portfolio_value)
    display_weight_breakdown(config)
    display_debug_info(debug_info, config)

    st.markdown("---")
    st.markdown("*Portfolio Strategy Analyzer - Built with Streamlit*")

if __name__ == "__main__":
    main()
