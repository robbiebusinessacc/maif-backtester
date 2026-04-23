from backtester.engine import Backtester, BacktestResult, BacktestConfig, Portfolio
from backtester.event_engine import EventDrivenBacktester
from backtester.synthetic import (
    make_oscillating, make_trending, make_random_walk, run_validation_suite,
    run_scenario_suite, GBMSource, BlockBootstrapSource,
    RegimeSwitchingSource, NoiseInjectionSource, GANSource,
)
from backtester.scorecard import generate_scorecard, generate_options_scorecard
from backtester.optimize import optimize, plot_heatmap
from backtester.distributions import generate_distribution_plots
from backtester.gan_bridge import make_gan_source

__all__ = [
    "Backtester", "EventDrivenBacktester",
    "BacktestResult", "BacktestConfig", "Portfolio",
    "make_oscillating", "make_trending", "make_random_walk", "run_validation_suite",
    "run_scenario_suite", "GBMSource", "BlockBootstrapSource",
    "RegimeSwitchingSource", "NoiseInjectionSource", "GANSource",
    "generate_scorecard", "generate_options_scorecard",
    "optimize", "plot_heatmap",
    "generate_distribution_plots",
    "make_gan_source",
]
