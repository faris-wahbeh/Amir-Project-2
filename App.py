# app.py - Portfolio Strategy Analyzer (supports "list-of-tickers" Rank.csv)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import re
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & DATA STRUCTURES
# ============================================================================

@dataclass
class PortfolioConfig:
    num_positions: int                # e.g., 10
    cash_percentage: float            # e.g., 8 = 8%
    rebalance_frequency: str          # 'monthly' | 'quarterly' | 'semi-yearly'
    rebalance_cost: float             # trading cost as %, e.g., 0.5 = 0.5%

@dataclass
class MarketData:
    rank_df: pd.DataFrame             # rows=dates; EITHER list-of-tickers per col (RANK_1..), OR per-ticker numeric ranks
    price_df: pd.DataFrame            # price table (dates x tickers)
    returns_df: pd.DataFrame          # decimal returns (dates x tickers)

INITIAL_INVESTMENT = 100.0
ACTUAL_RETURNS_DATA = {
    '2018': [4.7, 0.4, 1.2, 2.8, 5.1, 5.4, 1.1, 7.1, 0.7, -8.5, 3.4, -8.3],
    '2019': [8.6, 8.9, 3.2, 4.9, -2.5, 6.7, 3.2, -0.4, -6.5, 0.4, 5.5, 0.6],
    '2020': [5.5, -6.6, -14.3, 14.2, 9.0, 3.9, 5.9, 6.6, -3.1, -3.7, 11.8, 7.2],
    '2021': [-2.5, 8.2, -6.8, 4.9, -6.3, 6.3, 3.6, 5.2, -2.2, 3.1, -1.8, -0.1],
    '2022': [-12.9, -0.5, -2.1, -8.9, -9.5, -8.2, 9.1, -3.1, -8.1, 3.6, 4.0, -2.4]
}
ACTUAL_YEARLY_RETURNS = {
    '2018': 14.5, '2019': 36.5, '2020': 37.7, '2021': 10.6, '2022': -34.3
}
FREQUENCY_MAP = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def validate_data_files() -> bool:
    rank_file = Path("Rank.csv")
    prices_file = Path("Prices.csv")
    if not rank_file.exists() or not prices_file.exists():
        missing = []
        if not rank_file.exists(): missing.append("Rank.csv")
        if not prices_file.exists(): missing.append("Prices.csv")
        st.error(f"Missing required files: {', '.join(missing)}")
        st.info("Please upload the required files to your workspace")
        return False
    return True

def _normalize_cols_generic(idx: pd.Index) -> pd.Index:
    # Only light cleanup for column labels; we do NOT try to match tickers here.
    return (pd.Index(idx).astype(str).str.strip())

def _normalize_ticker_token(s: str) -> str:
    """
    Normalize a ticker-like string from Rank values to match Prices columns:
    - Uppercase
    - Remove all non-alphanumeric chars (spaces, dots, dashes, slashes, etc.)
    Examples:
      'ADBE US Equity' -> 'ADBEUSEQUITY'
      'BRK.B' / 'BRK-B' -> 'BRKB'
      '6758 JP Equity' -> '6758JPEQUITY'
    """
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r'[^A-Z0-9]', '', s)  # keep only letters/numbers
    return s

@st.cache_data
def load_market_data() -> MarketData:
    """Load and prepare market data. Align dates only; don't intersect columns (Rank may not be tickers)."""
    rank_df = pd.read_csv('Rank.csv', index_col=0, parse_dates=True, dayfirst=True)
    price_df = pd.read_csv('Prices.csv', index_col=0, parse_dates=True, dayfirst=True)

    # Align by dates
    common_index = rank_df.index.intersection(price_df.index)
    if len(common_index) == 0:
        st.error("Rank.csv and Prices.csv have no overlapping dates.")
        st.stop()

    rank_df = rank_df.loc[common_index].sort_index()
    price_df = price_df.loc[common_index].sort_index()

    # Light col cleanup; NO ticker normalization here.
    rank_df.columns = _normalize_cols_generic(rank_df.columns)
    price_df.columns = _normalize_cols_generic(price_df.columns)

    # Returns as decimals; fill initial NaNs with 0
    returns_df = price_df.pct_change(fill_method=None).fillna(0.0)

    return MarketData(rank_df, price_df, returns_df)

def get_actual_portfolio_returns() -> List[float]:
    out: List[float] = []
    for y in sorted(ACTUAL_RETURNS_DATA.keys()):
        out.extend(ACTUAL_RETURNS_DATA[y])
    return out

# ============================================================================
# CORE LOGIC
# ============================================================================

def calculate_position_percentages(num_positions: int, cash_percentage: float) -> List[float]:
    """
    Descending weights for the investable sleeve via arithmetic series; sums to (100 - cash).
    Floors negatives and renormalizes to avoid weird shapes for large N.
    """
    total_pp = max(0.0, 100.0 - float(cash_percentage))
    n = int(num_positions)
    if n <= 0 or total_pp == 0:
        return [0.0] * max(0, n)

    if n <= 5:
        a = 0.30 * total_pp
    else:
        a = 0.30 * total_pp - (n - 5) * 0.02 * total_pp

    if n == 1:
        weights = [total_pp]
    else:
        d = (2 * a - 2 * (total_pp / n)) / (n - 1)
        weights = [a - i * d for i in range(n)]

    weights = [max(0.0, w) for w in weights]
    s = sum(weights)
    if s == 0:
        weights = [total_pp / n] * n
    else:
        weights = [w * total_pp / s for w in weights]
    return weights

def _infer_rank_schema(row: pd.Series) -> str:
    """
    Detect if Rank.csv row is:
      - 'list': values are tickers (strings like 'ADBE US Equity'), columns are rank slots (RANK_1,...)
      - 'numeric': columns are tickers, values are numeric rank scores (lower is better)
    """
    sample = row.dropna().astype(str).head(min(20, len(row)))
    # Count values that look numeric-only
    is_num = pd.to_numeric(sample, errors='coerce').notna().sum()
    # If zero numeric-ish entries, it's very likely a list-of-tickers row
    return 'list' if is_num == 0 else 'numeric'

def generate_portfolio(number_of_positions: int, rebalance_frequency: str, rank_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate holdings:
      - If Rank.csv is list-of-tickers (RANK_1.. columns), take first N values per row (in column order).
      - If Rank.csv is per-ticker numeric ranks, sort by ascending value and take N tickers (index names).
    Always pads rows to N entries with None to avoid assignment errors.
    """
    period = FREQUENCY_MAP[rebalance_frequency]
    portfolio = pd.DataFrame(index=rank_data.index, columns=range(number_of_positions), dtype=object)

    def _pad(lst, n, fill=None):
        return lst[:n] + [fill] * max(0, n - len(lst))

    schema = _infer_rank_schema(rank_data.iloc[0])

    current_holdings = [None] * number_of_positions
    for i, (dt, row) in enumerate(rank_data.iterrows()):
        if i % period == 0:
            if schema == 'list':
                # Values ARE tickers in RANK_1.. columns; keep their left-to-right order
                vals = [v for v in row.values.tolist() if pd.notna(v)]
                sorted_tickers = [str(v) for v in vals]
            else:
                # Values are numeric ranks; columns are tickers; pick top N tickers by ascending value
                sorted_tickers = row.sort_values(ascending=True).index.tolist()
            current_holdings = _pad(sorted_tickers, number_of_positions, None)
        portfolio.loc[dt] = current_holdings

    return portfolio

def generate_returns_portfolio(portfolio: pd.DataFrame, returns_data: pd.DataFrame) -> pd.DataFrame:
    """
    Map decimal returns to holdings. We normalize each holding string to match returns_data columns.
    Missing/None → 0.0. Warn if coverage is low.
    """
    common_index = portfolio.index.intersection(returns_data.index)
    portfolio = portfolio.loc[common_index]
    returns_data = returns_data.loc[common_index]

    # Build a normalization map for price/returns columns
    ret_cols = list(returns_data.columns)
    norm_map: Dict[str, str] = {}
    for c in ret_cols:
        norm_map[_normalize_ticker_token(c)] = c  # normalized -> original

    ret_port = pd.DataFrame(index=common_index, columns=portfolio.columns, dtype=float)
    missing_map = 0
    total_map = 0

    for dt in common_index:
        row = portfolio.loc[dt]
        vals: List[float] = []
        for pos in row:
            total_map += 1
            if pos is None or (isinstance(pos, float) and np.isnan(pos)):
                vals.append(0.0)
                missing_map += 1
                continue
            key = _normalize_ticker_token(str(pos))
            col = norm_map.get(key)
            if col is None:
                vals.append(0.0)
                missing_map += 1
            else:
                vals.append(float(returns_data.at[dt, col]))
        ret_port.loc[dt] = vals

    if total_map > 0:
        cov = 100.0 * (1.0 - missing_map / total_map)
        if cov < 95:
            st.warning(f"Returns mapping coverage: {cov:.1f}%. If unexpectedly low, check ticker spellings in Rank.csv.")
    return ret_port

def simulate_portfolio(num_positions: int,
                       cash_percentage: float,
                       returns_portfolio: pd.DataFrame,
                       rebalance_frequency: str,
                       rebalance_cost: float):
    """
    Weights are fractions (sum=1 including cash). Costs from turnover: 0.5 * L1 distance * cost%.
    Compounds net return each period; weights drift between rebalances.
    """
    period = FREQUENCY_MAP[rebalance_frequency]
    index = returns_portfolio.index
    n = num_positions

    target_pp = calculate_position_percentages(n, cash_percentage)
    target_frac = np.array([w / 100.0 for w in target_pp], dtype=float)
    cash_target_frac = cash_percentage / 100.0

    # Start in cash
    w_current = np.zeros(n + 1, dtype=float)  # positions..., cash
    w_current[-1] = 1.0

    portfolio_value, exposure_delta_pp = [], []
    gross_return_pct, net_return_pct = [], []
    weights_history_pp = pd.DataFrame(index=index, columns=range(n), dtype=float)

    V = INITIAL_INVESTMENT

    for i, dt in enumerate(index):
        rebalancing = (i % period == 0)

        # Rebalance at start
        cost_pct_this_period = 0.0
        if rebalancing:
            w_target = np.concatenate([target_frac, [cash_target_frac]])
            l1 = float(np.sum(np.abs(w_current - w_target)))
            exposure_pp = 100.0 * l1
            turnover = 0.5 * l1
            cost_pct_this_period = rebalance_cost * turnover
            w_current = w_target.copy()
        else:
            exposure_pp = 0.0

        if n > 0:
            weights_history_pp.loc[dt] = (w_current[:-1] * 100.0)

        r_list = np.array(returns_portfolio.iloc[i].astype(float).fillna(0.0).values, dtype=float)
        gross_r = float(np.sum(w_current[:-1] * r_list))
        gross_r_pct = gross_r * 100.0
        net_r_pct = gross_r_pct - cost_pct_this_period

        V *= (1.0 + net_r_pct / 100.0)

        denom = max(1e-12, 1.0 + gross_r)
        w_pos_end = (w_current[:-1] * (1.0 + r_list)) / denom
        w_cash_end = (w_current[-1]) / denom
        w_current = np.concatenate([w_pos_end, [w_cash_end]])

        portfolio_value.append(V)
        exposure_delta_pp.append(exposure_pp)
        gross_return_pct.append(gross_r_pct)
        net_return_pct.append(net_r_pct)

    pv_series = pd.Series(portfolio_value, index=index, name='Portfolio Value')
    exposure_series = pd.Series(exposure_delta_pp, index=index, name='Summed Exposure Delta (pp)')
    gross_series = pd.Series(gross_return_pct, index=index, name='Gross Return (%)')
    net_series = pd.Series(net_return_pct, index=index, name='Net Return (%)')

    return pv_series, exposure_series, gross_series, net_series, weights_history_pp

def execute_portfolio_strategy(config: PortfolioConfig) -> Tuple[pd.Series, pd.DataFrame, Dict]:
    if not validate_data_files():
        st.stop()
    market = load_market_data()

    portfolio = generate_portfolio(config.num_positions, config.rebalance_frequency, market.rank_df)
    returns_port = generate_returns_portfolio(portfolio, market.returns_df)

    pv, exposure_delta, gross_ret, net_ret, weights_hist_pp = simulate_portfolio(
        config.num_positions, config.cash_percentage, returns_port,
        config.rebalance_frequency, config.rebalance_cost
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
# PERFORMANCE / OPTIMIZATION
# ============================================================================

def calculate_return_metrics(values: np.ndarray) -> Dict[str, float]:
    if len(values) < 2:
        return {'volatility': 0.0, 'total_return': 0.0, 'annualized_return': 0.0}
    rets = np.diff(values) / values[:-1]
    volatility = float(np.std(rets, ddof=1) * np.sqrt(12) * 100.0)
    total_return = float(((values[-1] / values[0]) - 1.0) * 100.0)
    years = len(values) / 12.0
    annualized_return = float((((values[-1] / values[0]) ** (1.0 / years)) - 1.0) * 100.0) if years > 0 else 0.0
    return {'volatility': volatility, 'total_return': total_return, 'annualized_return': annualized_return}

def calculate_tracking_error(strategy: np.ndarray, benchmark: np.ndarray) -> float:
    if len(strategy) != len(benchmark) or len(strategy) == 0:
        return float('inf')
    rel_errors_sq = []
    for s, b in zip(strategy, benchmark):
        err = ((s - b) / b) * 100.0 if b != 0 else (s - b)
        rel_errors_sq.append(err * err)
    return float(np.sqrt(np.mean(rel_errors_sq)))

def optimize_portfolio_parameters(target_start: int, target_end: int, rebalance_filter: str) -> Optional[Dict]:
    actual_returns = get_actual_portfolio_returns()
    actual_values = [INITIAL_INVESTMENT]
    for ret in actual_returns:
        actual_values.append(actual_values[-1] * (1.0 + ret / 100.0))

    if target_start >= len(actual_values) or target_end >= len(actual_values) or target_end <= target_start:
        return None

    actual_period = np.array(actual_values[target_start:target_end + 1], dtype=float)
    actual_period = actual_period * 100.0 / actual_period[0]

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

    best_result, best_error, tested = None, float('inf'), 0

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
                        pass
                    progress.progress(tested / total)

    progress.empty()
    status.empty()
    return best_result

# ============================================================================
# VISUALIZATION & UI
# ============================================================================

def create_performance_chart(portfolio_value: pd.Series) -> Tuple[go.Figure, List]:
    actual_returns = get_actual_portfolio_returns()
    actual_values = [INITIAL_INVESTMENT]
    for ret in actual_returns[:len(portfolio_value)]:
        actual_values.append(actual_values[-1] * (1.0 + ret / 100.0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(portfolio_value))), y=portfolio_value.values,
        mode="lines", name="Strategy", line=dict(width=2, color='#1f77b4')
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(actual_values[:len(portfolio_value)]))),
        y=actual_values[:len(portfolio_value)], mode="lines",
        name="Actual", line=dict(width=2, dash="dash", color='#ff7f0e')
    ))
    fig.update_layout(
        template="simple_white", margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Months", yaxis_title="Portfolio Value ($)", height=500
    )
    return fig, actual_values

def create_optimization_chart(result: Dict) -> go.Figure:
    x = list(range(len(result['strategy_values'])))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=result['strategy_values'], mode='lines',
                             name='Optimal Strategy', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=x, y=result['actual_values'], mode='lines',
                             name='Actual', line=dict(color='red', width=2, dash='dash')))
    fig.update_layout(template='simple_white', margin=dict(l=40, r=20, t=40, b=40),
                      xaxis_title="Months in Period", yaxis_title="Normalized Value", height=400)
    return fig

def initialize_streamlit():
    st.set_page_config(page_title="Portfolio Strategy Analyzer", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}</style>""",
                unsafe_allow_html=True)

def create_parameter_sidebar() -> PortfolioConfig:
    st.sidebar.header("Portfolio Parameters")
    num_positions = st.sidebar.slider("Number of Positions", 5, 15, 10, 1)
    cash_percentage = st.sidebar.slider("Cash Percentage (%)", 0, 50, 8, 1)
    rebalance_frequency = st.sidebar.selectbox("Rebalance Frequency", ["monthly", "quarterly", "semi-yearly"], index=0)
    rebalance_cost = st.sidebar.slider("Rebalance Cost (%)", 0.0, 5.0, 0.5, 0.01, format="%.2f")
    return PortfolioConfig(num_positions, cash_percentage, rebalance_frequency, rebalance_cost)

def display_performance_metrics(portfolio_value: pd.Series, actual_values: List):
    st.markdown("---")
    strategy_metrics = calculate_return_metrics(portfolio_value.values)
    actual_metrics = calculate_return_metrics(np.array(actual_values[:len(portfolio_value)]))
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    strategy_final = float(portfolio_value.iloc[-1])
    actual_final = float(actual_values[min(len(portfolio_value), len(actual_values)-1)])
    c1.metric("Strategy Final", f"${strategy_final:.2f}", f"{strategy_metrics['total_return']:+.1f}%")
    c2.metric("Actual Final", f"${actual_final:.2f}", f"{actual_metrics['total_return']:+.1f}%")
    c3.metric("Outperformance", f"{strategy_metrics['total_return'] - actual_metrics['total_return']:+.1f}%")
    c4.metric("Annualized", f"{strategy_metrics['annualized_return']:.1f}%")
    c5.metric("Strategy Volatility", f"{strategy_metrics['volatility']:.1f}%")
    c6.metric("Actual Volatility", f"{actual_metrics['volatility']:.1f}%")
    st.caption("💡 Volatility is calculated as the annualized standard deviation of monthly returns.")

def display_monthly_comparison_table(portfolio_value: pd.Series):
    st.markdown("---")
    st.subheader("Monthly Performance Comparison")
    strategy_returns = []
    for i in range(len(portfolio_value) - 1):
        strategy_returns.append(float((portfolio_value.iloc[i + 1] / portfolio_value.iloc[i] - 1.0) * 100.0))
    actual_returns = get_actual_portfolio_returns()

    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    years = ['2018','2019','2020','2021','2022']

    table_data, month_idx = [], 0
    for year in years:
        row = {'Year': year}
        year_strategy = []
        for month in months:
            if month_idx < min(len(strategy_returns), len(actual_returns)):
                s_ret = strategy_returns[month_idx]; a_ret = actual_returns[month_idx]
                year_strategy.append(s_ret)
                row[month] = f"T: {s_ret:.1f}%\nA: {a_ret:.1f}%"
                month_idx += 1
            else:
                row[month] = ""
        if year_strategy:
            s_yearly = (np.prod([1 + r/100 for r in year_strategy]) - 1) * 100
            row['Yearly Total'] = f"T: {s_yearly:.1f}%\nA: {ACTUAL_YEARLY_RETURNS.get(year, 0.0):.1f}%"
        table_data.append(row)

    df = pd.DataFrame(table_data)

    def style_cell(val):
        if not val or '\n' not in val: return ''
        try:
            t_val = float(val.split('\n')[0].split(': ')[1].replace('%',''))
            a_val = float(val.split('\n')[1].split(': ')[1].replace('%',''))
            diff = t_val - a_val
            if diff > 0:  return 'background-color:#d4edda;color:#155724'
            if diff < 0:  return 'background-color:#f8d7da;color:#721c24'
            return 'background-color:#fff3cd;color:#856404'
        except Exception:
            return ''

    styled_df = df.style.map(style_cell, subset=[c for c in df.columns if c != 'Year'])
    st.markdown("**Legend:** T = Theoretical, A = Actual")
    st.markdown("🟢 Outperformance | 🔴 Underperformance | 🟡 Equal")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

def display_optimization_section(portfolio_value: pd.Series):
    st.markdown("---")
    st.subheader("🔍 Find Closest Match to Actual Portfolio")
    col1, col2, col3, col4 = st.columns(4)
    start = col1.number_input("Start Month", 0, max(0, len(portfolio_value)-1), 14)
    end = col2.number_input("End Month", 0, max(0, len(portfolio_value)-1), min(30, max(0, len(portfolio_value)-1)))
    reb_filter = col3.selectbox("Rebalance Filter", ["any","monthly","quarterly","semi-yearly"])
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
    st.markdown("---")
    st.subheader("Position Weightings")
    weights = calculate_position_percentages(config.num_positions, config.cash_percentage)
    st.write("**Position Weights (percent of total portfolio):**")
    for i, w in enumerate(weights): st.write(f"Rank {i+1}: {w:.2f}%")
    if config.cash_percentage > 0: st.write(f"Cash: {config.cash_percentage:.2f}%")
    st.write(f"**Total Invested: {sum(weights):.2f}%**")

def display_debug_info(debug_info: Dict, config: PortfolioConfig):
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
        st.dataframe(pd.concat([debug_info['gross_return_pct'].head(5),
                                debug_info['net_return_pct'].head(5)], axis=1))

# ============================================================================
# MAIN
# ============================================================================

def main():
    initialize_streamlit()
    st.title("Portfolio Strategy Analyzer")
    st.markdown("Analyze and optimize portfolio strategies using historical data")
    st.markdown("---")

    config = create_parameter_sidebar()

    try:
        portfolio_value, holdings, debug_info = execute_portfolio_strategy(config)
    except Exception as e:
        st.error(f"Error calculating portfolio: {str(e)}")
        import traceback; st.error(traceback.format_exc()); st.stop()

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
