# app.py - Portfolio Strategy Analyzer with Clean Functional Architecture
# Following the exact methodology from the LaTeX document

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import itertools
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
    cash_percentage: float
    rebalance_frequency: str
    rebalance_cost: float

@dataclass
class MarketData:
    """Container for market data"""
    rank_df: pd.DataFrame
    price_df: pd.DataFrame
    returns_df: pd.DataFrame

@dataclass
class PortfolioState:
    """Container for portfolio state at any point in time"""
    holdings: pd.DataFrame  # Which securities are held
    returns: pd.DataFrame   # Returns for each position
    growth: pd.DataFrame    # Position values over time
    weights: List[float]    # Target weights for each position

@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics"""
    gross_contribution: pd.Series
    exposure_delta: pd.Series
    net_contribution: pd.Series
    portfolio_value: pd.Series

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

# ============================================================================
# STEP 0: DATA LOADING & PREPARATION
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
    # Load raw data
    rank_df = pd.read_csv('Rank.csv', index_col=0, parse_dates=True, dayfirst=True)
    price_df = pd.read_csv('Prices.csv', index_col=0, parse_dates=True, dayfirst=True)
    
    # Calculate returns (as percentages)
    returns_df = price_df.pct_change(fill_method=None).fillna(0) * 100
    
    return MarketData(rank_df, price_df, returns_df)

def get_actual_portfolio_returns() -> List[float]:
    """Get flattened list of actual portfolio returns"""
    returns = []
    for year in sorted(ACTUAL_RETURNS_DATA.keys()):
        returns.extend(ACTUAL_RETURNS_DATA[year])
    return returns

# ============================================================================
# STEP 1: PORTFOLIO GENERATION
# ============================================================================

def get_rebalance_periods(config: PortfolioConfig) -> int:
    """Convert rebalance frequency to number of periods"""
    mapping = {
        'monthly': 1,
        'quarterly': 3,
        'semi-yearly': 6
    }
    return mapping[config.rebalance_frequency]

@st.cache_data
def generate_portfolio_holdings(rank_data: pd.DataFrame, 
                               config: PortfolioConfig) -> pd.DataFrame:
    """
    Step 1: Generate portfolio holdings based on rank data
    
    Key Logic:
    - Rank file updates daily but portfolio only rebalances at specified periods
    - Between rebalancing, maintain same holdings regardless of rank changes
    """
    rebalance_period = get_rebalance_periods(config)
    num_dates = len(rank_data)
    num_positions = config.num_positions
    
    # Initialize portfolio DataFrame
    portfolio = pd.DataFrame(
        index=rank_data.index,
        columns=range(num_positions)
    )
    
    # Track current holdings
    current_holdings = []
    
    # Generate holdings over time
    for i, (date, rank_row) in enumerate(rank_data.iterrows()):
        if i % rebalance_period == 0:
            # Rebalancing period: update to current top ranks
            current_holdings = rank_row.iloc[:num_positions].tolist()
        
        # Set holdings for this date (either new or maintained)
        portfolio.loc[date] = current_holdings
    
    return portfolio

# ============================================================================
# STEP 2: RETURN CALCULATION
# ============================================================================

def map_returns_to_holdings(holdings: pd.DataFrame,
                           returns_data: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Map market returns to portfolio holdings
    
    Creates a matrix where each cell contains the return 
    for that position on that date
    """
    portfolio_returns = pd.DataFrame(
        index=holdings.index,
        columns=holdings.columns,
        dtype=float
    )
    
    for date in holdings.index:
        if date not in returns_data.index:
            continue
            
        date_holdings = holdings.loc[date]
        date_returns = []
        
        for position in date_holdings:
            if position in returns_data.columns:
                return_value = returns_data.at[date, position]
            else:
                return_value = 0.0
            date_returns.append(return_value)
        
        portfolio_returns.loc[date] = date_returns
    
    return portfolio_returns

# ============================================================================
# WEIGHT CALCULATION
# ============================================================================

def calculate_arithmetic_weights(config: PortfolioConfig) -> List[float]:
    """
    Calculate position weights using arithmetic series
    
    Formula from methodology:
    - Weights decrease linearly from top to bottom
    - Sum equals investable percentage (100% - cash%)
    
    Returns: List of weights as percentages (e.g., 30.0 for 30%)
    """
    investable = 100.0 - config.cash_percentage
    n = config.num_positions
    
    # Calculate maximum weight for top position
    if n <= 5:
        max_weight = 0.3 * investable
    else:
        max_weight = (0.3 * investable - 
                     (n - 5) * 0.02 * investable - 
                     (15 - n))
    
    if n == 1:
        return [investable]
    
    # Calculate common difference for arithmetic series
    # From formula: sum = n * (first + last) / 2
    # We know sum and first, solve for common difference
    common_diff = 2 * (max_weight * n - investable) / (n * (n - 1))
    
    # Generate weight series
    weights = []
    for i in range(n):
        weight = max_weight - i * common_diff
        weights.append(weight)
    
    return weights

# ============================================================================
# STEP 3: PORTFOLIO GROWTH CALCULATION
# ============================================================================

def calculate_position_growth(portfolio_returns: pd.DataFrame,
                             weights: List[float],
                             config: PortfolioConfig) -> pd.DataFrame:
    """
    Step 3: Calculate portfolio growth over time
    
    CRITICAL LOGIC:
    - Positions are tracked as PERCENTAGES (e.g., 30.0 for 30% of portfolio)
    - At t=0 and rebalancing periods: positions are SET to target weights
    - Between rebalancing: positions GROW with their returns
    - The sum of all positions represents the investable portion (100% - cash%)
    """
    rebalance_period = get_rebalance_periods(config)
    num_positions = len(weights)
    
    # Initialize growth DataFrame
    growth = pd.DataFrame(
        index=portfolio_returns.index,
        columns=portfolio_returns.columns,
        dtype=float
    )
    
    for i, date in enumerate(portfolio_returns.index):
        if i % rebalance_period == 0:
            # REBALANCING PERIOD: Reset to target percentages
            # These are the target weights (already as percentages)
            current_percentages = weights.copy()
        else:
            # NO REBALANCING: Use previous values (which have grown)
            current_percentages = growth.iloc[i - 1].tolist()
        
        # Apply returns to get new values
        # NOTE: Returns are applied AFTER determining if it's a rebalancing period
        current_growth = []
        for col_index in range(num_positions):
            return_value = portfolio_returns.iloc[i, col_index]
            if pd.isna(return_value):
                return_value = 0
            
            # Apply return to current percentage
            # This happens whether we rebalanced or not
            new_value = current_percentages[col_index] * (1 + return_value / 100)
            current_growth.append(new_value)
        
        growth.iloc[i] = current_growth
    
    return growth

# ============================================================================
# STEP 4: REBALANCING CALCULATIONS
# ============================================================================

def calculate_rebalancing_exposure(holdings: pd.DataFrame,
                                  growth: pd.DataFrame,
                                  weights: List[float],
                                  config: PortfolioConfig) -> pd.Series:
    """
    Step 4a: Calculate exposure delta (amount traded at each rebalancing)
    
    CRITICAL LOGIC:
    - At rebalancing, we compare:
      1. Where positions ARE (after growth from previous period)
      2. Where positions NEED TO BE (target weights)
    - The sum of absolute differences is the exposure delta
    
    NOTE: The growth DataFrame already has positions AFTER rebalancing,
    so we need to look at the PREVIOUS period to see pre-rebalancing values.
    """
    rebalance_period = get_rebalance_periods(config)
    exposure_delta = pd.Series(index=holdings.index, dtype=float)
    
    for i, date in enumerate(holdings.index):
        if i == 0:
            # First period: special initialization
            # This represents initial portfolio setup cost
            delta = 10 * config.num_positions
            
        elif i % rebalance_period == 0:
            # REBALANCING PERIOD: Calculate how much we need to trade
            delta = 0.0
            
            # Get holdings for this period and previous
            current_holdings = holdings.iloc[i].tolist()
            previous_holdings = holdings.iloc[i - 1].tolist()
            
            # For each position in the NEW portfolio arrangement
            for j in range(config.num_positions):
                # Target weight for position j
                target_weight = weights[j]
                
                # Which stock will be in position j after rebalancing
                stock_after_rebal = current_holdings[j]
                
                # Find this stock's value BEFORE rebalancing
                # (i.e., what it grew to in the previous period)
                prev_value = None
                if stock_after_rebal in previous_holdings:
                    # Stock was in portfolio - find its position
                    prev_position_index = previous_holdings.index(stock_after_rebal)
                    # Get its value from previous period (after growth)
                    prev_value = growth.iloc[i - 1, prev_position_index]
                
                # Calculate absolute difference
                if prev_value is None:
                    # Stock is NEW to portfolio - need to buy full target weight
                    delta += target_weight
                else:
                    # Stock was in portfolio - trade the difference
                    delta += abs(target_weight - prev_value)
                
        else:
            # Non-rebalancing period: no trading
            delta = 0.0
        
        exposure_delta.loc[date] = delta
    
    return exposure_delta

def calculate_contribution_metrics(growth: pd.DataFrame,
                                  weights: List[float],
                                  exposure_delta: pd.Series,
                                  config: PortfolioConfig) -> Tuple[pd.Series, pd.Series]:
    """
    Step 4b: Calculate gross and net contributions
    
    Returns: (gross_contribution, net_contribution)
    """
    rebalance_period = get_rebalance_periods(config)
    num_positions = config.num_positions
    
    # Calculate gross contribution (returns before costs)
    gross_contribution = pd.Series(index=growth.index, dtype=float)
    
    for i, date in enumerate(growth.index):
        total_contribution = 0.0
        
        for j in range(num_positions):
            current_value = growth.iloc[i, j]
            
            if i == 0:
                # First period: difference from initial weight
                contribution = current_value - weights[j]
            elif i % rebalance_period == 0:
                # Rebalancing: difference from reset weight
                contribution = current_value - weights[j]
            else:
                # Regular period: difference from previous value
                prev_value = growth.iloc[i - 1, j]
                contribution = current_value - prev_value
            
            total_contribution += contribution
        
        gross_contribution.loc[date] = total_contribution
    
    # Calculate net contribution (after transaction costs)
    transaction_costs = exposure_delta * (config.rebalance_cost / 100)
    net_contribution = gross_contribution - transaction_costs
    
    return gross_contribution, net_contribution

# ============================================================================
# STEP 5: PORTFOLIO VALUE COMPOUNDING
# ============================================================================

def compound_portfolio_value(net_contribution: pd.Series,
                            initial_value: float = INITIAL_INVESTMENT) -> pd.Series:
    """
    Step 5: Compound portfolio value over time
    
    Start with initial investment and compound using net contributions
    """
    portfolio_values = []
    current_value = initial_value
    
    for contribution in net_contribution.values:
        # Compound formula: V(t) = V(t-1) * (1 + contribution/100)
        current_value *= (1 + contribution / 100)
        portfolio_values.append(current_value)
    
    return pd.Series(portfolio_values, index=net_contribution.index)

# ============================================================================
# MAIN PORTFOLIO CALCULATION PIPELINE
# ============================================================================

def execute_portfolio_strategy(config: PortfolioConfig) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Execute the complete 5-step portfolio methodology
    
    Returns: (portfolio_value_series, portfolio_holdings)
    """
    # Step 0: Load market data
    if not validate_data_files():
        st.stop()
    
    market_data = load_market_data()
    
    # Calculate weights once
    weights = calculate_arithmetic_weights(config)
    
    # Step 1: Generate portfolio holdings
    holdings = generate_portfolio_holdings(market_data.rank_df, config)
    
    # Step 2: Map returns to holdings
    portfolio_returns = map_returns_to_holdings(holdings, market_data.returns_df)
    
    # Step 3: Calculate portfolio growth
    growth = calculate_position_growth(portfolio_returns, weights, config)
    
    # Step 4: Calculate rebalancing metrics
    exposure_delta = calculate_rebalancing_exposure(holdings, growth, weights, config)
    gross_contribution, net_contribution = calculate_contribution_metrics(
        growth, weights, exposure_delta, config
    )
    
    # Step 5: Compound portfolio value
    portfolio_value = compound_portfolio_value(net_contribution)
    
    return portfolio_value, holdings

# ============================================================================
# PERFORMANCE ANALYTICS
# ============================================================================

def calculate_return_metrics(values: np.ndarray) -> Dict[str, float]:
    """Calculate various return metrics"""
    if len(values) < 2:
        return {'volatility': 0.0, 'total_return': 0.0, 'annualized_return': 0.0}
    
    # Calculate period returns
    returns = np.diff(values) / values[:-1]
    
    # Volatility (annualized)
    volatility = np.std(returns, ddof=1) * np.sqrt(12) * 100
    
    # Total return
    total_return = ((values[-1] / values[0]) - 1) * 100
    
    # Annualized return
    years = len(values) / 12
    annualized_return = ((values[-1] / values[0]) ** (1/years) - 1) * 100 if years > 0 else 0
    
    return {
        'volatility': volatility,
        'total_return': total_return,
        'annualized_return': annualized_return
    }

def calculate_tracking_error(strategy: np.ndarray, benchmark: np.ndarray) -> float:
    """Calculate RMSE tracking error between strategy and benchmark"""
    if len(strategy) != len(benchmark) or len(strategy) == 0:
        return float('inf')
    
    # Calculate percentage errors
    relative_errors = []
    for s, b in zip(strategy, benchmark):
        if b != 0:
            error = ((s - b) / b) * 100
        else:
            error = s - b
        relative_errors.append(error ** 2)
    
    return np.sqrt(np.mean(relative_errors))

# ============================================================================
# PORTFOLIO OPTIMIZATION
# ============================================================================

def optimize_portfolio_parameters(target_start: int, 
                                 target_end: int,
                                 rebalance_filter: str) -> Optional[Dict]:
    """
    Find optimal parameters to minimize tracking error against actual portfolio
    """
    # Get actual portfolio values
    actual_returns = get_actual_portfolio_returns()
    actual_values = [INITIAL_INVESTMENT]
    for ret in actual_returns:
        actual_values.append(actual_values[-1] * (1 + ret / 100))
    
    # Extract target period
    if target_start >= len(actual_values) or target_end >= len(actual_values):
        return None
    
    actual_period = actual_values[target_start:target_end + 1]
    actual_period = np.array(actual_period)
    actual_period = actual_period * 100 / actual_period[0]  # Normalize
    
    # Define search space
    search_space = {
        'positions': range(5, 16),
        'cash': range(0, 31, 5),
        'rebalance': ['monthly', 'quarterly', 'semi-yearly'] if rebalance_filter == 'any' else [rebalance_filter],
        'cost': [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    }
    
    # Calculate total combinations
    total = (len(search_space['positions']) * len(search_space['cash']) * 
            len(search_space['rebalance']) * len(search_space['cost']))
    
    # Progress tracking
    progress = st.progress(0)
    status = st.empty()
    
    # Grid search
    best_result = None
    best_error = float('inf')
    tested = 0
    
    for positions in search_space['positions']:
        for cash in search_space['cash']:
            for rebalance in search_space['rebalance']:
                for cost in search_space['cost']:
                    tested += 1
                    status.text(f"Testing {tested}/{total}: pos={positions}, cash={cash}%, "
                              f"reb={rebalance}, cost={cost:.2f}%")
                    
                    # Test this configuration
                    config = PortfolioConfig(positions, cash, rebalance, cost)
                    
                    try:
                        portfolio_value, _ = execute_portfolio_strategy(config)
                        
                        # Extract period
                        if target_end < len(portfolio_value):
                            strategy_period = portfolio_value.iloc[target_start:target_end + 1].values
                            strategy_period = strategy_period * 100 / strategy_period[0]
                            
                            # Calculate error
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
                    except:
                        continue
                    
                    progress.progress(tested / total)
    
    progress.empty()
    status.empty()
    
    return best_result

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_performance_chart(portfolio_value: pd.Series) -> Tuple[go.Figure, List]:
    """Create main performance comparison chart"""
    # Get actual returns
    actual_returns = get_actual_portfolio_returns()
    actual_values = [INITIAL_INVESTMENT]
    
    for ret in actual_returns[:len(portfolio_value)]:
        actual_values.append(actual_values[-1] * (1 + ret / 100))
    
    # Create figure
    fig = go.Figure()
    
    # Strategy line
    fig.add_trace(go.Scatter(
        x=list(range(len(portfolio_value))),
        y=portfolio_value.values,
        mode="lines",
        name="Strategy",
        line=dict(width=2, color='#1f77b4')
    ))
    
    # Actual line
    fig.add_trace(go.Scatter(
        x=list(range(len(actual_values[:len(portfolio_value)]))),
        y=actual_values[:len(portfolio_value)],
        mode="lines",
        name="Actual",
        line=dict(width=2, dash="dash", color='#ff7f0e')
    ))
    
    # Layout
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
    
    num_positions = st.sidebar.slider("Number of Positions", 5, 15, 15, 1)
    cash_percentage = st.sidebar.slider("Cash Percentage (%)", 0, 50, 0, 1)
    rebalance_frequency = st.sidebar.selectbox(
        "Rebalance Frequency",
        ["monthly", "quarterly", "semi-yearly"],
        index=0
    )
    rebalance_cost = st.sidebar.slider(
        "Rebalance Cost (%)",
        0.0, 5.0, 2.00, 0.01,
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
    
    # Calculate metrics
    strategy_metrics = calculate_return_metrics(portfolio_value.values)
    actual_metrics = calculate_return_metrics(np.array(actual_values[:len(portfolio_value)]))
    
    # Display in columns
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    strategy_final = portfolio_value.iloc[-1]
    actual_final = actual_values[min(len(portfolio_value), len(actual_values)-1)]
    
    c1.metric("Strategy Final", f"${strategy_final:.2f}", 
             f"{strategy_metrics['total_return']:+.1f}%")
    c2.metric("Actual Final", f"${actual_final:.2f}", 
             f"{actual_metrics['total_return']:+.1f}%")
    c3.metric("Outperformance", 
             f"{strategy_metrics['total_return'] - actual_metrics['total_return']:+.1f}%")
    c4.metric("Annualized", f"{strategy_metrics['annualized_return']:.1f}%")
    c5.metric("Strategy Volatility", f"{strategy_metrics['volatility']:.1f}%")
    c6.metric("Actual Volatility", f"{actual_metrics['volatility']:.1f}%")
    
    st.caption("💡 Volatility is calculated as the annualized standard deviation of monthly returns")

def display_monthly_comparison_table(portfolio_value: pd.Series):
    """Display monthly performance comparison table"""
    st.markdown("---")
    st.subheader("Monthly Performance Comparison")
    
    # Calculate monthly returns
    strategy_returns = []
    for i in range(len(portfolio_value) - 1):
        ret = (portfolio_value.iloc[i + 1] / portfolio_value.iloc[i] - 1) * 100
        strategy_returns.append(ret)
    
    actual_returns = get_actual_portfolio_returns()
    
    # Build table
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
        
        # Yearly total
        if year_strategy:
            s_yearly = (np.prod([1 + r/100 for r in year_strategy]) - 1) * 100
            a_yearly = ACTUAL_YEARLY_RETURNS.get(year, 0.0)
            year_row['Yearly Total'] = f"T: {s_yearly:.1f}%\nA: {a_yearly:.1f}%"
        
        table_data.append(year_row)
    
    # Display table with styling
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
        except:
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
    
    start = col1.number_input("Start Month", 0, len(portfolio_value)-1, 14)
    end = col2.number_input("End Month", 0, len(portfolio_value)-1, 30)
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
                    
                    # Display results
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
                    
                    # Display chart
                    fig = create_optimization_chart(result)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No valid combination found")

def display_weight_breakdown(config: PortfolioConfig):
    """Display position weight breakdown"""
    st.markdown("---")
    st.subheader("Position Weightings")
    
    weights = calculate_arithmetic_weights(config)
    
    st.write("**Position Weights:**")
    for i, weight in enumerate(weights):
        st.write(f"Rank {i+1}: {weight:.2f}%")
    
    if config.cash_percentage > 0:
        st.write(f"Cash: {config.cash_percentage:.2f}%")
    
    st.write(f"**Total Invested: {sum(weights):.2f}%**")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    # Initialize
    initialize_streamlit()
    
    # Header
    st.title("Portfolio Strategy Analyzer")
    st.markdown("Analyze and optimize portfolio strategies using historical data")
    st.markdown("---")
    
    # Get parameters
    config = create_parameter_sidebar()
    
    # Calculate portfolio
    try:
        portfolio_value, holdings = execute_portfolio_strategy(config)
    except Exception as e:
        st.error(f"Error calculating portfolio: {str(e)}")
        st.stop()
    
    # Display results
    fig, actual_values = create_performance_chart(portfolio_value)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display components
    display_performance_metrics(portfolio_value, actual_values)
    display_monthly_comparison_table(portfolio_value)
    display_optimization_section(portfolio_value)
    display_weight_breakdown(config)
    
    # Footer
    st.markdown("---")
    st.markdown("*Portfolio Strategy Analyzer - Built with Streamlit*")

if __name__ == "__main__":
    main()
