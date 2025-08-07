# app.py - Portfolio Strategy Analyzer with Clean Architecture
# Following the exact methodology from the LaTeX document

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import itertools
from dataclasses import dataclass
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA CLASSES FOR TYPE SAFETY
# ============================================================================

@dataclass
class PortfolioParameters:
    """Encapsulates all portfolio configuration parameters"""
    num_positions: int
    cash_percentage: float
    rebalance_frequency: str
    rebalance_cost: float
    
    @property
    def rebalance_period(self) -> int:
        """Convert rebalance frequency to number of periods"""
        mapping = {'monthly': 1, 'quarterly': 3, 'semi-yearly': 6}
        return mapping[self.rebalance_frequency]
    
    @property
    def investable_percentage(self) -> float:
        """Calculate investable percentage after cash allocation"""
        return 100.0 - self.cash_percentage


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the application"""
    RANK_FILE = "Rank.csv"
    PRICES_FILE = "Prices.csv"
    INITIAL_INVESTMENT = 100.0
    
    # Actual portfolio returns for comparison
    ACTUAL_RETURNS = {
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
# DATA LOADING AND PREPROCESSING
# ============================================================================

class DataLoader:
    """Handles all data loading and initial processing"""
    
    @staticmethod
    def check_required_files() -> Tuple[Path, Path]:
        """Verify that required data files exist"""
        base = Path(".")
        rank_file = base / Config.RANK_FILE
        prices_file = base / Config.PRICES_FILE
        
        missing_files = []
        if not rank_file.exists():
            missing_files.append(Config.RANK_FILE)
        if not prices_file.exists():
            missing_files.append(Config.PRICES_FILE)
        
        if missing_files:
            st.error(f"Missing required files: {', '.join(missing_files)}")
            st.info("Please upload the following files to your workspace:")
            for file in missing_files:
                st.code(f"• {file}")
            st.stop()
        
        return rank_file, prices_file
    
    @staticmethod
    @st.cache_data
    def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load rank and price data from CSV files"""
        rank_df = pd.read_csv(Config.RANK_FILE, index_col=0, parse_dates=True, dayfirst=True)
        price_df = pd.read_csv(Config.PRICES_FILE, index_col=0, parse_dates=True, dayfirst=True)
        return rank_df, price_df
    
    @staticmethod
    @st.cache_data
    def calculate_returns(price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate percentage returns from price data
        Returns: DataFrame with returns as percentages (e.g., 2.5 for 2.5%)
        """
        returns_df = price_df.pct_change(fill_method=None)
        returns_df = returns_df.fillna(0)
        returns_df = returns_df * 100  # Convert to percentage
        return returns_df
    
    @staticmethod
    def get_actual_returns() -> List[float]:
        """Get flattened list of actual portfolio returns"""
        returns = []
        for year in sorted(Config.ACTUAL_RETURNS.keys()):
            returns.extend(Config.ACTUAL_RETURNS[year])
        return returns


# ============================================================================
# PORTFOLIO WEIGHT CALCULATIONS
# ============================================================================

class WeightCalculator:
    """Handles all weight calculation logic following arithmetic series approach"""
    
    @staticmethod
    def calculate_position_weights(params: PortfolioParameters) -> List[float]:
        """
        Calculate position weights using arithmetic series
        Returns: List of weights as percentages (e.g., 30.0 for 30%)
        
        From methodology:
        - Weights decrease linearly from top to bottom position
        - Sum of weights equals investable percentage
        """
        investable = params.investable_percentage
        
        # Calculate maximum weight for top position
        if params.num_positions <= 5:
            max_weight = 0.3 * investable
        else:
            max_weight = (0.3 * investable - 
                         (params.num_positions - 5) * 0.02 * investable - 
                         (15 - params.num_positions))
        
        if params.num_positions > 1:
            # Calculate common difference for arithmetic series
            # Formula: d = 2(w_max * n - S) / (n * (n-1))
            n = params.num_positions
            common_diff = (2 * (max_weight * n - investable)) / (n * (n - 1))
            
            # Generate weights: w_i = w_max - (i * d)
            weights = [max_weight - i * common_diff for i in range(n)]
        else:
            weights = [investable]
        
        return weights


# ============================================================================
# PORTFOLIO GENERATION
# ============================================================================

class PortfolioGenerator:
    """Handles portfolio generation based on rank data and rebalancing frequency"""
    
    @staticmethod
    @st.cache_data
    def generate_portfolio(rank_df: pd.DataFrame, params: PortfolioParameters) -> pd.DataFrame:
        """
        Generate portfolio holdings based on rebalancing frequency
        
        Key concept from methodology:
        - Rank file updates daily but portfolio only updates at rebalancing periods
        - Between rebalancing periods, maintain same holdings
        """
        portfolio = pd.DataFrame(index=rank_df.index, 
                                columns=range(params.num_positions))
        current_holdings = []
        
        for i, (date, row) in enumerate(rank_df.iterrows()):
            if i % params.rebalance_period == 0:
                # Rebalancing period: update to current top ranks
                current_holdings = row.iloc[:params.num_positions].tolist()
            # Otherwise maintain previous holdings
            portfolio.loc[date] = current_holdings
        
        return portfolio
    
    @staticmethod
    def map_returns_to_portfolio(portfolio: pd.DataFrame, 
                                 returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map returns to portfolio positions
        Creates matrix where each cell contains the return for that position on that date
        """
        returns_portfolio = pd.DataFrame(index=portfolio.index, 
                                        columns=portfolio.columns)
        
        for date in portfolio.index:
            if date in returns_df.index:
                portfolio_row = portfolio.loc[date]
                current_returns = []
                
                for position in portfolio_row:
                    if position in returns_df.columns:
                        return_value = returns_df.at[date, position]
                        current_returns.append(return_value)
                    else:
                        current_returns.append(0.0)
                
                returns_portfolio.loc[date] = current_returns
        
        return returns_portfolio


# ============================================================================
# PORTFOLIO CALCULATIONS
# ============================================================================

class PortfolioCalculator:
    """Core portfolio calculation engine following the 5-step methodology"""
    
    @staticmethod
    def calculate_portfolio_growth(returns_portfolio: pd.DataFrame,
                                  weights: List[float],
                                  params: PortfolioParameters) -> pd.DataFrame:
        """
        Step 3: Calculate Portfolio Growth
        
        - At rebalancing periods: reset to target weights
        - Between periods: positions grow naturally with returns
        """
        portfolio_growth = pd.DataFrame(index=returns_portfolio.index,
                                       columns=returns_portfolio.columns,
                                       dtype=float)
        
        for i, date in enumerate(returns_portfolio.index):
            if i % params.rebalance_period == 0:
                # Rebalancing: reset to target percentages
                current_percentages = weights.copy()
            else:
                # No rebalancing: use previous values
                current_percentages = portfolio_growth.iloc[i - 1].tolist()
            
            # Apply returns to current percentages
            current_growth = []
            for col_index in range(params.num_positions):
                return_value = returns_portfolio.iloc[i, col_index]
                if pd.isna(return_value):
                    return_value = 0
                
                # Growth formula: V_t = V_(t-1) * (1 + r_t/100)
                new_value = current_percentages[col_index] * (1 + return_value / 100)
                current_growth.append(new_value)
            
            portfolio_growth.iloc[i] = current_growth
        
        return portfolio_growth
    
    @staticmethod
    def calculate_exposure_delta(portfolio: pd.DataFrame,
                                portfolio_growth: pd.DataFrame,
                                weights: List[float],
                                params: PortfolioParameters) -> pd.Series:
        """
        Step 4a: Calculate Exposure Delta
        
        Total USD amount being traded at each rebalancing period
        """
        exposure_delta = pd.Series(index=portfolio.index, dtype=float)
        
        for i, date in enumerate(portfolio.index):
            if i == 0:
                # First period: special initialization case
                delta = 10 * params.num_positions
            elif i % params.rebalance_period == 0:
                # Rebalancing period: calculate total amount traded
                delta = 0
                
                for col_index in range(params.num_positions):
                    target_weight = weights[col_index]
                    current_stock = portfolio.iloc[i, col_index]
                    
                    # Find previous value of this stock if it was in portfolio
                    prev_value = None
                    prev_date = portfolio.index[i - 1]
                    prev_holdings = portfolio.loc[prev_date].tolist()
                    
                    if current_stock in prev_holdings:
                        prev_position_index = prev_holdings.index(current_stock)
                        prev_value = portfolio_growth.iloc[i - 1, prev_position_index]
                    
                    # Calculate absolute difference
                    if prev_value is None:
                        # Stock is new to portfolio
                        delta += target_weight
                    else:
                        # Stock was in portfolio, calculate change
                        delta += abs(target_weight - prev_value)
            else:
                # Non-rebalancing period: no trading
                delta = 0
            
            exposure_delta.loc[date] = delta
        
        return exposure_delta
    
    @staticmethod
    def calculate_gross_contribution(portfolio_growth: pd.DataFrame,
                                    weights: List[float],
                                    params: PortfolioParameters) -> pd.Series:
        """
        Step 4b: Calculate Gross Contribution
        
        Returns before transaction costs
        """
        gross_contribution = pd.Series(index=portfolio_growth.index, dtype=float)
        
        for i, date in enumerate(portfolio_growth.index):
            total_contribution = 0
            
            for col_index in range(params.num_positions):
                current_value = portfolio_growth.iloc[i, col_index]
                
                if i == 0:
                    # First period: difference from initial weight
                    contribution = current_value - weights[col_index]
                elif i % params.rebalance_period == 0:
                    # Rebalancing period: difference from reset weight
                    contribution = current_value - weights[col_index]
                else:
                    # Regular period: difference from previous value
                    prev_value = portfolio_growth.iloc[i - 1, col_index]
                    contribution = current_value - prev_value
                
                total_contribution += contribution
            
            gross_contribution.loc[date] = total_contribution
        
        return gross_contribution
    
    @staticmethod
    def calculate_net_contribution(gross_contribution: pd.Series,
                                  exposure_delta: pd.Series,
                                  rebalance_cost: float) -> pd.Series:
        """
        Step 4c: Calculate Net Contribution
        
        Net = Gross - (Exposure_Delta * Rebalance_Cost%)
        """
        transaction_costs = exposure_delta * (rebalance_cost / 100)
        net_contribution = gross_contribution - transaction_costs
        return net_contribution
    
    @staticmethod
    def calculate_compounded_value(net_contribution: pd.Series) -> pd.Series:
        """
        Step 5: Track Compounded Performance
        
        Start with initial investment and compound using net contributions
        """
        compounded_values = []
        value = Config.INITIAL_INVESTMENT
        
        for date, contribution in net_contribution.items():
            value *= (1 + contribution / 100)
            compounded_values.append(value)
        
        return pd.Series(compounded_values, index=net_contribution.index)


# ============================================================================
# MAIN PORTFOLIO ENGINE
# ============================================================================

class PortfolioEngine:
    """Main engine that orchestrates the entire portfolio calculation process"""
    
    def __init__(self, params: PortfolioParameters):
        self.params = params
        self.weights = WeightCalculator.calculate_position_weights(params)
    
    def calculate_portfolio_performance(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Execute the complete 5-step methodology
        Returns: (portfolio_value_series, portfolio_holdings)
        """
        # Step 0: Load data
        DataLoader.check_required_files()
        rank_df, price_df = DataLoader.load_data()
        returns_df = DataLoader.calculate_returns(price_df)
        
        # Step 1: Generate Portfolio
        portfolio = PortfolioGenerator.generate_portfolio(rank_df, self.params)
        
        # Step 2: Calculate Returns
        returns_portfolio = PortfolioGenerator.map_returns_to_portfolio(
            portfolio, returns_df
        )
        
        # Step 3: Calculate Portfolio Growth
        portfolio_growth = PortfolioCalculator.calculate_portfolio_growth(
            returns_portfolio, self.weights, self.params
        )
        
        # Step 4: Adjust for Rebalancing Costs
        exposure_delta = PortfolioCalculator.calculate_exposure_delta(
            portfolio, portfolio_growth, self.weights, self.params
        )
        
        gross_contribution = PortfolioCalculator.calculate_gross_contribution(
            portfolio_growth, self.weights, self.params
        )
        
        net_contribution = PortfolioCalculator.calculate_net_contribution(
            gross_contribution, exposure_delta, self.params.rebalance_cost
        )
        
        # Step 5: Compute Compounded Growth
        portfolio_value = PortfolioCalculator.calculate_compounded_value(
            net_contribution
        )
        
        return portfolio_value, portfolio
    
    def calculate_for_period(self, start_month: int, end_month: int) -> Optional[np.ndarray]:
        """Calculate portfolio for a specific period (used in optimization)"""
        try:
            portfolio_value, _ = self.calculate_portfolio_performance()
            
            # Extract and normalize the period
            max_idx = len(portfolio_value) - 1
            start_idx = min(start_month, max_idx)
            end_idx = min(end_month, max_idx)
            
            if start_idx > end_idx:
                return None
            
            series = portfolio_value.iloc[start_idx:end_idx + 1].values
            if len(series) > 0:
                # Normalize to start at 100
                series = series * 100 / series[0]
            
            return series
            
        except Exception as e:
            print(f"Error calculating portfolio for period: {e}")
            return None


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

class PerformanceMetrics:
    """Calculate various performance metrics"""
    
    @staticmethod
    def calculate_volatility(values: np.ndarray) -> float:
        """Calculate annualized volatility from portfolio values"""
        if len(values) < 2:
            return 0.0
        
        # Calculate period returns
        returns = []
        for i in range(1, len(values)):
            period_return = (values[i] / values[i-1] - 1)
            returns.append(period_return)
        
        if not returns:
            return 0.0
        
        # Calculate standard deviation and annualize
        returns_std = np.std(returns, ddof=1)
        annualized_vol = returns_std * np.sqrt(12) * 100  # Monthly to annual
        
        return annualized_vol
    
    @staticmethod
    def calculate_rmse(strategy: np.ndarray, actual: np.ndarray) -> float:
        """Calculate Root Mean Square Error between strategy and actual"""
        if len(strategy) != len(actual) or len(strategy) == 0:
            return float('inf')
        
        errors = []
        for s, a in zip(strategy, actual):
            if a != 0:
                errors.append(((s - a) / a * 100) ** 2)
            else:
                errors.append((s - a) ** 2)
        
        return np.sqrt(np.mean(errors))
    
    @staticmethod
    def calculate_cumulative_return(initial: float, final: float) -> float:
        """Calculate cumulative return percentage"""
        return ((final - initial) / initial) * 100
    
    @staticmethod
    def calculate_annualized_return(initial: float, final: float, years: float) -> float:
        """Calculate annualized return percentage"""
        if years <= 0:
            return 0.0
        return ((final / initial) ** (1 / years) - 1) * 100


# ============================================================================
# OPTIMIZATION ENGINE
# ============================================================================

class OptimizationEngine:
    """Handle parameter optimization to find best match with actual portfolio"""
    
    @staticmethod
    def find_optimal_parameters(start_month: int, end_month: int, 
                               rebalance_filter: str) -> Optional[dict]:
        """
        Find optimal parameters that minimize tracking error
        """
        # Get actual returns for comparison
        actual_returns = DataLoader.get_actual_returns()
        actual_values = [Config.INITIAL_INVESTMENT]
        
        for ret in actual_returns:
            actual_values.append(actual_values[-1] * (1 + ret / 100))
        
        # Extract and normalize the period
        max_idx = len(actual_values) - 1
        start_idx = min(start_month, max_idx)
        end_idx = min(end_month, max_idx)
        
        if start_idx > end_idx:
            st.error("Invalid period selection")
            return None
        
        actual_period = actual_values[start_idx:end_idx + 1]
        if len(actual_period) > 0:
            actual_period = [v * 100 / actual_period[0] for v in actual_period]
        
        # Define parameter search space
        positions_range = list(range(5, 16))
        cash_range = list(range(0, 31, 5))
        rebalance_options = (['monthly', 'quarterly', 'semi-yearly'] 
                           if rebalance_filter == 'any' 
                           else [rebalance_filter])
        cost_range = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Progress tracking
        total_combinations = (len(positions_range) * len(cash_range) * 
                            len(rebalance_options) * len(cost_range))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        best_params = None
        best_error = float('inf')
        valid_count = 0
        
        # Grid search
        for i, (pos, cash, reb, cost) in enumerate(
            itertools.product(positions_range, cash_range, 
                            rebalance_options, cost_range)
        ):
            status_text.text(
                f"Testing {i+1}/{total_combinations}: "
                f"pos={pos}, cash={cash}%, reb={reb}, cost={cost:.2f}%"
            )
            
            # Test this parameter combination
            params = PortfolioParameters(pos, cash, reb, cost)
            engine = PortfolioEngine(params)
            strategy_values = engine.calculate_for_period(start_month, end_month)
            
            if strategy_values is not None and len(strategy_values) == len(actual_period):
                valid_count += 1
                error = PerformanceMetrics.calculate_rmse(strategy_values, actual_period)
                
                if error < best_error:
                    best_error = error
                    best_params = {
                        'num_positions': pos,
                        'cash_percentage': cash,
                        'rebalance_frequency': reb,
                        'rebalance_cost': cost,
                        'error': error,
                        'strategy_values': strategy_values,
                        'actual_values': actual_period
                    }
            
            progress_bar.progress((i + 1) / total_combinations)
        
        progress_bar.empty()
        status_text.empty()
        
        if valid_count == 0:
            st.warning(f"No valid combinations found out of {total_combinations} tested.")
        
        return best_params


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Handle all chart and visualization creation"""
    
    @staticmethod
    def create_comparison_chart(portfolio_value: pd.Series) -> Tuple[go.Figure, list]:
        """Create main comparison chart between strategy and actual portfolio"""
        actual_returns = DataLoader.get_actual_returns()
        actual_values = [Config.INITIAL_INVESTMENT]
        
        for ret in actual_returns[:len(portfolio_value)]:
            actual_values.append(actual_values[-1] * (1 + ret / 100))
        
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
        
        fig.update_layout(
            template="simple_white",
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_title="Months",
            yaxis_title="Portfolio Value ($)",
            height=500,
        )
        
        return fig, actual_values
    
    @staticmethod
    def create_optimization_chart(opt_result: dict) -> go.Figure:
        """Create chart showing optimization results"""
        x = list(range(len(opt_result['strategy_values'])))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=opt_result['strategy_values'],
            mode='lines',
            name='Optimal Strategy',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=x,
            y=opt_result['actual_values'],
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
# STREAMLIT UI
# ============================================================================

def setup_page():
    """Configure Streamlit page settings"""
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


def create_sidebar() -> PortfolioParameters:
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
    
    return PortfolioParameters(
        num_positions, 
        cash_percentage, 
        rebalance_frequency, 
        rebalance_cost
    )


def display_metrics(portfolio_value: pd.Series, actual_values: list):
    """Display performance metrics"""
    st.markdown("---")
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    # Calculate values
    strategy_final = portfolio_value.iloc[-1]
    
    if len(portfolio_value) < len(actual_values):
        actual_final = actual_values[len(portfolio_value)]
    else:
        actual_final = actual_values[-1] if actual_values else 100.0
    
    strategy_return = PerformanceMetrics.calculate_cumulative_return(100, strategy_final)
    actual_return = PerformanceMetrics.calculate_cumulative_return(100, actual_final)
    outperformance = strategy_return - actual_return
    
    years = len(portfolio_value) / 12
    annualized = PerformanceMetrics.calculate_annualized_return(100, strategy_final, years)
    
    strategy_vol = PerformanceMetrics.calculate_volatility(portfolio_value.values)
    actual_vol = PerformanceMetrics.calculate_volatility(
        actual_values[:len(portfolio_value)]
    ) if actual_values else 0.0
    
    # Display metrics
    c1.metric("Strategy Final", f"${strategy_final:.2f}", f"{strategy_return:+.1f}%")
    c2.metric("Actual Final", f"${actual_final:.2f}", f"{actual_return:+.1f}%")
    c3.metric("Outperformance", f"{outperformance:+.1f}%")
    c4.metric("Annualized", f"{annualized:.1f}%")
    c5.metric("Strategy Volatility", f"{strategy_vol:.1f}%")
    c6.metric("Actual Volatility", f"{actual_vol:.1f}%")
    
    st.caption("💡 Volatility is calculated as the annualized standard deviation of monthly returns (std × √12)")


def display_monthly_comparison(portfolio_value: pd.Series):
    """Display monthly performance comparison table"""
    st.markdown("---")
    st.subheader("Monthly Performance Comparison")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    years = ['2018', '2019', '2020', '2021', '2022']
    
    # Calculate strategy monthly returns
    strategy_returns = []
    for i in range(len(portfolio_value) - 1):
        monthly_return = (portfolio_value.iloc[i + 1] / portfolio_value.iloc[i] - 1) * 100
        strategy_returns.append(monthly_return)
    
    # Get actual returns
    actual_returns = DataLoader.get_actual_returns()
    
    # Build table data
    table_data = {}
    month_idx = 0
    
    for year in years:
        year_data = {'Year': year}
        year_strategy = []
        year_actual = []
        
        for month in months:
            if month_idx < len(actual_returns) and month_idx < len(strategy_returns):
                actual_ret = actual_returns[month_idx]
                strategy_ret = strategy_returns[month_idx]
                
                year_strategy.append(strategy_ret)
                year_actual.append(actual_ret)
                
                cell_value = f"T: {strategy_ret:.1f}%\nA: {actual_ret:.1f}%"
                year_data[month] = cell_value
                month_idx += 1
            else:
                year_data[month] = ""
        
        # Calculate yearly totals
        if year_strategy:
            strategy_yearly = (np.prod([1 + r/100 for r in year_strategy]) - 1) * 100
            actual_yearly = Config.ACTUAL_YEARLY_RETURNS.get(year, 0.0)
            year_data['Yearly Total'] = f"T: {strategy_yearly:.1f}%\nA: {actual_yearly:.1f}%"
        
        table_data[year] = year_data
    
    # Display table
    if table_data:
        df_table = pd.DataFrame(list(table_data.values()))
        
        def style_cells(val):
            if not val or val == "":
                return ''
            try:
                lines = val.split('\n')
                if len(lines) == 2:
                    theoretical = float(lines[0].split(': ')[1].replace('%', ''))
                    actual = float(lines[1].split(': ')[1].replace('%', ''))
                    diff = theoretical - actual
                    
                    if diff > 0:
                        return 'background-color: #d4edda; color: #155724'
                    elif diff < 0:
                        return 'background-color: #f8d7da; color: #721c24'
                    else:
                        return 'background-color: #fff3cd; color: #856404'
            except:
                pass
            return ''
        
        styled_df = df_table.style.map(
            style_cells, 
            subset=[col for col in df_table.columns if col != 'Year']
        )
        
        st.markdown("**Legend:** T = Theoretical Strategy, A = Actual Returns")
        st.markdown("🟢 Green = Outperformance | 🔴 Red = Underperformance | 🟡 Yellow = Equal")
        st.dataframe(styled_df, use_container_width=True, hide_index=True)


def display_optimization_section(portfolio_value: pd.Series):
    """Display optimization section"""
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
            with st.spinner("Optimizing parameters... This may take a few minutes."):
                try:
                    opt_result = OptimizationEngine.find_optimal_parameters(
                        start_month, end_month, reb_filter
                    )
                    
                    if opt_result:
                        st.success("Optimal parameters found!")
                        
                        p1, p2 = st.columns(2)
                        with p1:
                            st.write("**Optimal Parameters**")
                            st.write(f"- **Positions:** {opt_result['num_positions']}")
                            st.write(f"- **Cash %:** {opt_result['cash_percentage']}%")
                            st.write(f"- **Rebalance:** {opt_result['rebalance_frequency']}")
                            st.write(f"- **Cost:** {opt_result['rebalance_cost']:.2f}%")
                        
                        with p2:
                            st.write("**Performance**")
                            st.write(f"- **Error (RMSE):** {opt_result['error']:.2f}%")
                        
                        fig = Visualizer.create_optimization_chart(opt_result)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("No valid combination found.")
                        
                except Exception as e:
                    st.error(f"Error during optimization: {str(e)}")


def display_position_weights(params: PortfolioParameters):
    """Display current position weightings"""
    st.markdown("---")
    st.subheader("Position Weightings")
    
    weights = WeightCalculator.calculate_position_weights(params)
    
    st.write("**Position Weightings:**")
    for i, weight in enumerate(weights):
        st.write(f"Rank {i+1}: {weight:.2f}%")
    
    if params.cash_percentage > 0:
        st.write(f"Cash: {params.cash_percentage:.2f}%")
    
    total_invested = sum(weights)
    st.write(f"**Total Invested: {total_invested:.2f}%**")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    # Setup
    setup_page()
    
    # Header
    st.title("Portfolio Strategy Analyzer")
    st.markdown("Analyze and optimize portfolio strategies using historical data")
    st.markdown("---")
    
    # Get parameters from sidebar
    params = create_sidebar()
    
    # Calculate portfolio performance
    try:
        engine = PortfolioEngine(params)
        portfolio_value, portfolio = engine.calculate_portfolio_performance()
        
        if portfolio_value is None:
            st.error("Failed to calculate portfolio performance")
            st.stop()
            
    except Exception as e:
        st.error(f"Error calculating portfolio performance: {str(e)}")
        st.stop()
    
    # Display main chart
    fig, actual_values = Visualizer.create_comparison_chart(portfolio_value)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metrics
    display_metrics(portfolio_value, actual_values)
    
    # Display monthly comparison
    display_monthly_comparison(portfolio_value)
    
    # Display optimization section
    display_optimization_section(portfolio_value)
    
    # Display position weights
    display_position_weights(params)
    
    # Footer
    st.markdown("---")
    st.markdown("*Portfolio Strategy Analyzer - Built with Streamlit*")


if __name__ == "__main__":
    main()
