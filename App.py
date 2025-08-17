
# app.py - Portfolio Strategy Analyzer

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
    cash_percentage: float
    rebalance_frequency: str
    rebalance_cost: float

@dataclass
class MarketData:
    """Container for market data"""
    rank_df: pd.DataFrame
    price_df: pd.DataFrame
    returns_df: pd.DataFrame

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
    # Load raw data
    rank_df = pd.read_csv('Rank.csv', index_col=0, parse_dates=True, dayfirst=True)
    price_df = pd.read_csv('Prices.csv', index_col=0, parse_dates=True, dayfirst=True)
    
    # Calculate returns as decimals (0.01 for 1% return)
 
    returns_df = price_df.pct_change(fill_method=None).fillna(0)
    
    return MarketData(rank_df, price_df, returns_df)

def get_actual_portfolio_returns() -> List[float]:
    """Get flattened list of actual portfolio returns"""
    returns = []
    for year in sorted(ACTUAL_RETURNS_DATA.keys()):
        returns.extend(ACTUAL_RETURNS_DATA[year])
    return returns

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def calculate_position_percentages(num_positions: int, cash_percentage: float) -> List[float]:
    """
    Calculate position weights using arithmetic series
   
    """
    total_percentage = 100 - cash_percentage
    
    if num_positions <= 5:
        highest_percentage = 0.3 * total_percentage
    else:
        highest_percentage = 0.3 * total_percentage - (num_positions - 5) * 0.02 * total_percentage - (15 - num_positions)
    
    sum_percentages = total_percentage
    n = num_positions
    a = highest_percentage
    S = sum_percentages
    
    common_difference = (2 * (a * n) - 2 * S) / (n * (n - 1)) if n > 1 else 0
    percentages = [a - i * common_difference for i in range(num_positions)]
    
    return percentages

def generate_portfolio(number_of_positions: int, rebalance_frequency: str, rank_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate portfolio holdings based on rank data
    """
    frequency_mapping = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    rebalance_period = frequency_mapping[rebalance_frequency]
    portfolio = pd.DataFrame(index=rank_data.index, columns=range(number_of_positions))
    current_holdings = []
    
    for i, (index, row) in enumerate(rank_data.iterrows()):
        if i % rebalance_period == 0:
            current_holdings = row.iloc[:number_of_positions].tolist()
        portfolio.loc[index] = current_holdings
    
    return portfolio

def generate_returns_portfolio(portfolio: pd.DataFrame, returns_data: pd.DataFrame) -> pd.DataFrame:
    """
    Map returns to portfolio holdings
    """
    returns_portfolio = pd.DataFrame(index=portfolio.index, columns=portfolio.columns)
    portfolio_dates = portfolio.index
    returns_dates = returns_data.index
    min_length = min(len(portfolio_dates), len(returns_dates))
    
    for date in portfolio_dates[:min_length]:
        portfolio_row = portfolio.loc[date]
        current_returns = []
        for position in portfolio_row:
            if position in returns_data.columns:
                return_value = returns_data.at[date, position]
                current_returns.append(return_value)
            else:
                current_returns.append(None)
        returns_portfolio.loc[date] = current_returns
    
    return returns_portfolio

def calculate_portfolio_growth(num_positions: int, cash_percentage: float, 
                              returns_portfolio: pd.DataFrame, rebalance_frequency: str) -> pd.DataFrame:
    """
    Calculate portfolio growth over time
    """
    position_percentages = calculate_position_percentages(num_positions, cash_percentage)
    portfolio_growth = pd.DataFrame(index=returns_portfolio.index, columns=returns_portfolio.columns)
    frequency_mapping = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    rebalance_period = frequency_mapping[rebalance_frequency]
    
    for i, date in enumerate(returns_portfolio.index):
        if i % rebalance_period == 0:
            # At rebalancing: reset to original percentages
            current_percentages = position_percentages.copy()
        else:
            # Between rebalancing: use previous grown values
            current_percentages = portfolio_growth.iloc[i - 1].tolist()
        
        current_growth = []
        for col_index in range(len(current_percentages)):
            return_value = returns_portfolio.iloc[i, col_index]
            
            # Handle different return formats
            if isinstance(return_value, str):
                return_value = float(return_value.strip('%')) / 100
            elif return_value is None or pd.isna(return_value):
                return_value = 0
            # return_value is already in decimal form (0.01 for 1%) from load_market_data
            
            # Apply return to current percentage
            current_growth.append(current_percentages[col_index] * (1 + return_value))
        
        portfolio_growth.iloc[i] = current_growth
    
    return portfolio_growth

def rank_and_exposure_delta(num_positions: int, cash_percentage: float, portfolio: pd.DataFrame,
                           portfolio_growth: pd.DataFrame, rebalance_frequency: str) -> pd.Series:
    """
    Calculate exposure delta at each rebalancing
    """
    reset_percentages = calculate_position_percentages(num_positions, cash_percentage)
    summed_exposure_delta = pd.Series(index=portfolio.index, name='Summed Exposure Delta')
    frequency_mapping = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    rebalance_period = frequency_mapping[rebalance_frequency]
    
    for i, date in enumerate(portfolio.index):
        if i == 0:
            # Initial setup cost
            summed_delta = 10 * num_positions
        elif i % rebalance_period == 0:
            # Rebalancing period: calculate exposure delta
            summed_delta = 0
            for col_index in range(num_positions):
                reset_value = reset_percentages[col_index]
                stock_name = portfolio.iloc[i, col_index]
                prev_value = None
                prev_date = portfolio.index[i - 1]
                
                # Find the stock's value in previous period
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

def calculate_gross_contribution(num_positions: int, cash_percentage: float,
                                portfolio_growth: pd.DataFrame, rebalance_frequency: str) -> pd.Series:
    """
    Calculate gross contribution (before transaction costs)
    """
    reset_percentages = calculate_position_percentages(num_positions, cash_percentage)
    gross_contribution = pd.DataFrame(index=portfolio_growth.index, columns=portfolio_growth.columns)
    frequency_mapping = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
    rebalance_period = frequency_mapping[rebalance_frequency]
    last_rebalance_values = portfolio_growth.iloc[0].tolist()
    
    for i, date in enumerate(portfolio_growth.index):
        current_contribution = []
        for col_index in range(num_positions):
            current_value = portfolio_growth.iloc[i, col_index]
            
            if i == 0:
                # First period: difference from initial percentage
                contribution = current_value - reset_percentages[col_index]
            elif i % rebalance_period == 0:
                # Rebalancing period: difference from reset percentage
                contribution = current_value - reset_percentages[col_index]
                last_rebalance_values[col_index] = current_value
            else:
                # Regular period: difference from previous value
                prev_value = portfolio_growth.iloc[i - 1, col_index]
                contribution = current_value - prev_value
            
            current_contribution.append(contribution)
        
        gross_contribution.loc[date] = current_contribution
    
    # Return summed contribution across all positions
    return gross_contribution.sum(axis=1)

def calculate_net_contribution(gross_contribution: pd.Series, rank_exposure_delta: pd.Series,
                              rebalance_cost: float) -> pd.Series:
    """
    Calculate net contribution (after transaction costs)
    """
    rebalance_costs = rank_exposure_delta.map(lambda x: x * (rebalance_cost) / 100)
    net_contribution = gross_contribution - rebalance_costs
    return net_contribution

def calculate_compounded_growth(net_contribution: pd.Series, 
                               start_date: Optional[pd.Timestamp] = None,
                               end_date: Optional[pd.Timestamp] = None) -> pd.Series:
    """
    Compound portfolio value over time
    """
    # Filter to date range if specified
    if start_date and end_date:
        filtered_contribution = net_contribution.loc[start_date:end_date]
    else:
        filtered_contribution = net_contribution
    
    # Convert to decimal form
    filtered_contribution = filtered_contribution / 100
    
    # Compound the growth
    initial_investment = 100
    compounded_values = []
    investment_value = initial_investment
    
    for date, contribution in filtered_contribution.items():
        investment_value *= (1 + contribution)
        compounded_values.append(investment_value)
    
    return pd.Series(compounded_values, index=filtered_contribution.index)

# ============================================================================
# MAIN PORTFOLIO CALCULATION PIPELINE
# ============================================================================

def execute_portfolio_strategy(config: PortfolioConfig) -> Tuple[pd.Series, pd.DataFrame, Dict]:
    """
    Execute the complete portfolio methodology
    
    Returns: (portfolio_value_series, portfolio_holdings, debug_info)
    """
    # Load market data
    if not validate_data_files():
        st.stop()
    
    market_data = load_market_data()
    
    # Step 1: Generate portfolio holdings
    portfolio = generate_portfolio(
        config.num_positions, 
        config.rebalance_frequency, 
        market_data.rank_df
    )
    
    # Step 2: Generate returns portfolio
    returns_portfolio = generate_returns_portfolio(
        portfolio, 
        market_data.returns_df
    )
    
    # Step 3: Calculate portfolio growth
    portfolio_growth = calculate_portfolio_growth(
        config.num_positions,
        config.cash_percentage,
        returns_portfolio,
        config.rebalance_frequency
    )
    
    # Step 4: Calculate gross contribution
    gross_contribution = calculate_gross_contribution(
        config.num_positions,
        config.cash_percentage,
        portfolio_growth,
        config.rebalance_frequency
    )
    
    # Step 5: Calculate exposure delta
    rank_exposure_delta = rank_and_exposure_delta(
        config.num_positions,
        config.cash_percentage,
        portfolio,
        portfolio_growth,
        config.rebalance_frequency
    )
    
    # Step 6: Calculate net contribution
    net_contribution = calculate_net_contribution(
        gross_contribution,
        rank_exposure_delta,
        config.rebalance_cost
    )
    
    # Step 7: Calculate compounded growth
    compounded_growth = calculate_compounded_growth(net_contribution)
    
    # Collect debug information
    debug_info = {
        'weights': calculate_position_percentages(config.num_positions, config.cash_percentage),
        'exposure_delta': rank_exposure_delta,
        'gross_contribution': gross_contribution,
        'net_contribution': net_contribution,
        'portfolio_growth': portfolio_growth,
        'compounded_growth': compounded_growth
    }
    
    return compounded_growth, portfolio, debug_info

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
                        portfolio_value, _, _ = execute_portfolio_strategy(config)
                        
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
        0.0, 5.0, 2.0, 0.01,
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
    st.subheader("MoM Growth")
    
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
    
    weights = calculate_position_percentages(config.num_positions, config.cash_percentage)
    
    st.write("**Position Weights:**")
    for i, weight in enumerate(weights):
        st.write(f"Rank {i+1}: {weight:.2f}%")
    
    if config.cash_percentage > 0:
        st.write(f"Cash: {config.cash_percentage:.2f}%")
    
    st.write(f"**Total Invested: {sum(weights):.2f}%**")

def display_debug_info(debug_info: Dict, config: PortfolioConfig):
    """Display debug information for verification"""
    with st.expander("Debug Information"):
        st.write("**Weights:**", debug_info['weights'])
        
        # Show first few exposure deltas
        st.write("**First 10 Exposure Deltas:**")
        st.dataframe(debug_info['exposure_delta'].head(10))
        
        # Show rebalancing periods with non-zero exposure
        rebalance_periods = debug_info['exposure_delta'][debug_info['exposure_delta'] > 0]
        st.write(f"**Rebalancing Events:** {len(rebalance_periods)}")
        st.dataframe(rebalance_periods.head(10))
        
        # Show portfolio growth sample
        st.write("**Portfolio Growth (first 5 periods):**")
        st.dataframe(debug_info['portfolio_growth'].head(5))
        
        # Show sample calculations
        st.write("**Sample Net Contribution Calculation (first 5 periods):**")
        for i in range(min(5, len(debug_info['gross_contribution']))):
            gross = debug_info['gross_contribution'].iloc[i]
            exposure = debug_info['exposure_delta'].iloc[i]
            cost = exposure * (config.rebalance_cost / 100)
            net = debug_info['net_contribution'].iloc[i]
            st.write(f"Period {i}: Gross={gross:.4f}, Exposure={exposure:.4f}, Cost={cost:.4f}, Net={net:.4f}")

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
        portfolio_value, holdings, debug_info = execute_portfolio_strategy(config)
    except Exception as e:
        st.error(f"Error calculating portfolio: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        st.stop()
    
    # Display results
    fig, actual_values = create_performance_chart(portfolio_value)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display components
    display_performance_metrics(portfolio_value, actual_values)
    display_monthly_comparison_table(portfolio_value)
    display_optimization_section(portfolio_value)
    display_weight_breakdown(config)
    
    # Add debug info section (optional - can be removed in production)
    display_debug_info(debug_info, config)
    
    # Footer
    st.markdown("---")
    st.markdown("*Portfolio Strategy Analyzer - Built with Streamlit*")

if __name__ == "__main__":
    main()
