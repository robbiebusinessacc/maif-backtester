"""
Options pricing, Greeks, backtesting, and position management.
"""

from backtester.options.greeks import GreeksEngine, Greeks
from backtester.options.contracts import OptionContract, OptionChain
from backtester.options.iv_surface import IVSurface
from backtester.options.positions import OptionLeg, OptionsPosition, PositionFactory
from backtester.options.execution import OptionsFill, OptionsExecutionConfig, OptionsExecutor
from backtester.options.expiry import ExerciseEvent, ExpirationHandler
from backtester.options.margin import MarginRequirement, MarginCalculator
from backtester.options.engine import OptionsBacktester, OptionsBacktestConfig, OptionsBacktestResult

__all__ = [
    # Greeks & pricing
    "GreeksEngine", "Greeks",
    # Data structures
    "OptionContract", "OptionChain", "IVSurface",
    # Positions
    "OptionLeg", "OptionsPosition", "PositionFactory",
    # Execution
    "OptionsFill", "OptionsExecutionConfig", "OptionsExecutor",
    # Expiration
    "ExerciseEvent", "ExpirationHandler",
    # Margin
    "MarginRequirement", "MarginCalculator",
    # Backtester
    "OptionsBacktester", "OptionsBacktestConfig", "OptionsBacktestResult",
]
