"""
Strategy module — base interface and strategies for the Testing Framework.
"""

from strategy.base import Strategy, Signal, Trade, Fill, Order, StrategyState
from strategy.sma_crossover import SMACrossover
from strategy.mean_reversion import MeanReversion
from strategy.bracket_breakout import BracketBreakout
from strategy.turtle_trend import TurtleTrend
from strategy.options_base import OptionsStrategy
from strategy.covered_call import CoveredCallStrategy
from strategy.helpers import crossover, cross, barssince, quantile

# Registry of all available strategies
STRATEGIES = [SMACrossover, MeanReversion, BracketBreakout, TurtleTrend]
OPTIONS_STRATEGIES = [CoveredCallStrategy]

def list_strategies():
    """Print all available strategies."""
    print("Available strategies:")
    for cls in STRATEGIES:
        doc = cls.__doc__.strip().splitlines()[0] if cls.__doc__ else "No description"
        print(f"  - {cls.__name__}: {doc}")
    print("\nOptions strategies:")
    for cls in OPTIONS_STRATEGIES:
        doc = cls.__doc__.strip().splitlines()[0] if cls.__doc__ else "No description"
        print(f"  - {cls.__name__}: {doc}")

__all__ = [
    "Strategy", "Signal", "Trade", "Fill", "Order", "StrategyState",
    "SMACrossover", "MeanReversion", "BracketBreakout", "TurtleTrend",
    "OptionsStrategy", "CoveredCallStrategy",
    "STRATEGIES", "OPTIONS_STRATEGIES", "list_strategies",
    "crossover", "cross", "barssince", "quantile",
]
