"""
Strategy Scorecard — four 16:9 pages.

Page 1: Bar-Based Backtest (equity curve, key metrics, trade stats)
Page 2: Monte Carlo Simulation (GBM, GAN regimes, noise injection, distributions)
Page 3: Event-Driven Backtest (bar vs event comparison, divergence analysis)
Page 4: Strategy Scorecard (grades from all tests combined)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle
import pandas as pd

from backtester.engine import Backtester, BacktestResult, BacktestConfig
from backtester.synthetic import make_oscillating, make_trending, make_random_walk
from strategy.base import Strategy

# ─────────────────────────────────────────────────────────────
# Design System
# ─────────────────────────────────────────────────────────────

_THEME_DARK = {
    "BG_DEEP": "#0d1117",
    "BG_CARD": "#161b22",
    "BG_ELEVATED": "#1c2333",
    "BORDER": "#30363d",
    "TEXT_PRIMARY": "#e6edf3",
    "TEXT_SECONDARY": "#8b949e",
    "TEXT_MUTED": "#484f58",
    "ACCENT_GOLD": "#d4a843",
    "ACCENT_BLUE": "#58a6ff",
    "ACCENT_TEAL": "#3fb950",
    "ACCENT_RED": "#f85149",
    "ACCENT_ORANGE": "#d29922",
    "LINE_STRATEGY": "#58a6ff",
    "LINE_BENCHMARK": "#484f58",
    "GRADE_COLORS": {
        "A": "#3fb950", "B": "#56d364", "C": "#d29922",
        "D": "#db6d28", "F": "#f85149",
    },
    "GRADE_BG": {
        "A": "#3fb95015", "B": "#56d36412", "C": "#d2992215",
        "D": "#db6d2815", "F": "#f8514915",
    },
}

_THEME_LIGHT = {
    "BG_DEEP": "#f6f8fa",
    "BG_CARD": "#ffffff",
    "BG_ELEVATED": "#f0f3f6",
    "BORDER": "#d0d7de",
    "TEXT_PRIMARY": "#1f2328",
    "TEXT_SECONDARY": "#656d76",
    "TEXT_MUTED": "#afb8c1",
    "ACCENT_GOLD": "#9a6700",
    "ACCENT_BLUE": "#0969da",
    "ACCENT_TEAL": "#1a7f37",
    "ACCENT_RED": "#cf222e",
    "ACCENT_ORANGE": "#bc4c00",
    "LINE_STRATEGY": "#0969da",
    "LINE_BENCHMARK": "#afb8c1",
    "GRADE_COLORS": {
        "A": "#1a7f37", "B": "#2da44e", "C": "#9a6700",
        "D": "#bc4c00", "F": "#cf222e",
    },
    "GRADE_BG": {
        "A": "#1a7f3715", "B": "#2da44e12", "C": "#9a670015",
        "D": "#bc4c0015", "F": "#cf222e15",
    },
}

def _apply_theme(theme_name: str = "dark"):
    """Swap module-level color variables to the chosen theme."""
    theme = _THEME_LIGHT if theme_name == "light" else _THEME_DARK
    g = globals()
    for key, val in theme.items():
        g[key] = val

# Default: dark mode
_apply_theme("dark")

@dataclass
class MetricGrade:
    name: str
    value: str
    grade: str
    explanation: str

def _letter_to_score(l): return {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}.get(l, 0)
def _score_to_letter(s):
    if s >= 3.5: return "A"
    if s >= 2.5: return "B"
    if s >= 1.5: return "C"
    if s >= 0.5: return "D"
    return "F"

def _grade(val, thresholds, explanations):
    for cutoff, letter in thresholds:
        if val >= cutoff:
            return letter, explanations[letter]
    return "F", explanations.get("F", "")


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def _run_monte_carlo(strategy, df, cfg):
    """Auto-run GBM, GAN, and noise injection stress tests."""
    import copy
    from backtester.synthetic import (
        GBMSource, NoiseInjectionSource, BlockBootstrapSource,
        run_scenario_suite,
    )

    n_scenarios = 50
    bt = Backtester(config=cfg)

    # ── GBM (pure noise — strategy should NOT profit) ──
    # Use bars_per_year so intraday backtests get appropriately-sized scenarios
    scenario_bars = min(len(df), cfg.bars_per_year)
    gbm_source = GBMSource(n_bars=scenario_bars, start_price=float(df["Close"].iloc[0]))
    gbm_results = []
    for i in range(n_scenarios):
        scenario = gbm_source.generate(seed=i)
        r = bt.run(copy.deepcopy(strategy), scenario)
        gbm_results.append(r)

    gbm_returns = [r.total_return_pct for r in gbm_results]
    gbm_sharpes = [r.sharpe_ratio for r in gbm_results]
    gbm_drawdowns = [r.max_drawdown_pct for r in gbm_results]

    # ── GAN (regime-conditioned synthetic data) ──
    gan_results = {}
    try:
        from backtester.gan_bridge import make_gan_source
        for regime in ["bullish", "bearish", "crash"]:
            source = make_gan_source(regime, n_bars=scenario_bars,
                                     start_price=float(df["Close"].iloc[0]))
            regime_rets = []
            for i in range(n_scenarios):
                scenario = source.generate(seed=i)
                r = bt.run(copy.deepcopy(strategy), scenario)
                regime_rets.append(r.total_return_pct)
            gan_results[regime] = regime_rets
    except Exception:
        gan_results = None

    # ── Noise injection (graceful degradation) ──
    noise_levels = [0.001, 0.003, 0.005, 0.010, 0.020]
    noise_results = {}
    for noise_std in noise_levels:
        source = NoiseInjectionSource(base_df=df, price_noise_std=noise_std)
        noise_rets = []
        for i in range(min(n_scenarios, 30)):
            scenario = source.generate(seed=i)
            r = bt.run(copy.deepcopy(strategy), scenario)
            noise_rets.append(r.total_return_pct)
        noise_results[noise_std] = noise_rets

    return {
        "gbm_returns": gbm_returns,
        "gbm_sharpes": gbm_sharpes,
        "gbm_drawdowns": gbm_drawdowns,
        "gbm_sharpe_mean": float(np.mean(gbm_sharpes)),
        "gan_results": gan_results,
        "noise_results": noise_results,
        "noise_levels": noise_levels,
        "n_scenarios": n_scenarios,
    }


def generate_scorecard(
    strategy: Strategy,
    df: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    output_path: str = "scorecard.png",
    symbol: str = "",
    monte_carlo_results: Optional[Dict[str, Any]] = None,
    event_driven_results: Optional[Dict[str, Any]] = None,
    theme: str = "dark",
) -> str:
    _apply_theme(theme)
    cfg = config or BacktestConfig()
    bt = Backtester(config=cfg)

    main = bt.run(strategy, df)
    mid = len(df) // 2
    r_in = bt.run(strategy, df.iloc[:mid])
    r_out = bt.run(strategy, df.iloc[mid:])
    r_osc = bt.run(strategy, make_oscillating(500))
    r_trend = bt.run(strategy, make_trending(500))
    r_random = bt.run(strategy, make_random_walk(500))

    # Auto-run Monte Carlo if not provided
    if monte_carlo_results is None:
        monte_carlo_results = _run_monte_carlo(strategy, df, cfg)

    if not symbol:
        symbol = df.attrs.get("symbol", "")
    date_start = df.index[0].strftime("%Y-%m-%d")
    date_end = df.index[-1].strftime("%Y-%m-%d")
    data_label = f"{symbol} " if symbol else ""
    data_label += f"{date_start} to {date_end}"

    grades_perf = _grade_performance(main)
    grades_risk = _grade_risk(main)
    grades_trade = _grade_trade_quality(main)
    grades_robust = _grade_robustness(r_in, r_out, r_osc, r_trend, r_random)
    grades_mc = _grade_monte_carlo(monte_carlo_results)
    grades_ed = _grade_event_driven(event_driven_results, main)

    all_grades = grades_perf + grades_risk + grades_trade + grades_robust + grades_mc + grades_ed
    overall_score = sum(_letter_to_score(g.grade) for g in all_grades) / len(all_grades)
    overall_letter = _score_to_letter(overall_score)

    base, ext = os.path.splitext(output_path)

    p1 = f"{base}_bartest{ext}"
    _render_bartest(strategy, main, data_label, r_osc, r_trend, r_random, r_in, r_out, p1)

    p2 = f"{base}_montecarlo{ext}"
    _render_montecarlo(strategy, data_label, monte_carlo_results, p2)

    p3 = f"{base}_eventdriven{ext}"
    _render_eventdriven(strategy, data_label, event_driven_results, main, p3)

    p4 = f"{base}_scorecard{ext}"
    _render_scorecard(
        strategy, main, data_label,
        grades_perf, grades_risk, grades_trade, grades_robust, grades_mc, grades_ed,
        overall_letter, overall_score,
        r_osc, r_trend, r_random, r_in, r_out,
        monte_carlo_results, event_driven_results, p4,
    )

    print(f"Page 1 (Bar-Based Backtest):    {p1}")
    print(f"Page 2 (Monte Carlo):           {p2}")
    print(f"Page 3 (Event-Driven Backtest): {p3}")
    print(f"Page 4 (Strategy Scorecard):    {p4}")
    return p4


# ─────────────────────────────────────────────────────────────
# Grading functions (unchanged logic)
# ─────────────────────────────────────────────────────────────

def _grade_performance(r):
    grades = []
    g, e = _grade(r.sharpe_ratio, [(2.0,"A"),(1.0,"B"),(0.5,"C"),(0.0,"D")],
        {"A":"Excellent risk-adjusted returns","B":"Good risk-adjusted returns",
         "C":"Returns don't justify the risk","D":"Barely positive","F":"Negative — losing money"})
    grades.append(MetricGrade("Sharpe Ratio", f"{r.sharpe_ratio:.2f}", g, e))
    sqn = r.sqn if np.isfinite(r.sqn) else 0.0
    g, e = _grade(sqn, [(5.0,"A"),(3.0,"B"),(2.0,"C"),(1.6,"D")],
        {"A":"Superb system quality","B":"Excellent — consistent edge",
         "C":"Average system quality","D":"Below average","F":"Poor — no reliable edge"})
    grades.append(MetricGrade("SQN", f"{r.sqn:.2f}", g, e))
    return grades

def _grade_risk(r):
    grades = []
    dd = abs(r.max_drawdown_pct)
    g, e = _grade(-dd, [(-5,"A"),(-15,"B"),(-25,"C"),(-40,"D")],
        {"A":"Minimal drawdown","B":"Moderate drawdown","C":"Large drawdown",
         "D":"Severe drawdown","F":"Catastrophic drawdown"})
    grades.append(MetricGrade("Max Drawdown", f"{r.max_drawdown_pct:.1f}%", g, e))
    g, e = _grade(-r.max_drawdown_duration, [(-50,"A"),(-120,"B"),(-250,"C"),(-500,"D")],
        {"A":"Recovers quickly","B":"Moderate recovery","C":"~1 year underwater",
         "D":"Very long recovery","F":"Multi-year recovery"})
    grades.append(MetricGrade("Max DD Duration", f"{r.max_drawdown_duration} bars", g, e))
    return grades

def _grade_trade_quality(r):
    grades = []
    pf = min(r.profit_factor, 99)
    g, e = _grade(pf, [(2.0,"A"),(1.5,"B"),(1.1,"C"),(1.0,"D")],
        {"A":"Profits are 2x+ losses","B":"Profits outweigh losses",
         "C":"Thin edge","D":"Breakeven","F":"Losing money"})
    grades.append(MetricGrade("Profit Factor", f"{r.profit_factor:.2f}", g, e))
    g, e = _grade(r.expectancy, [(500,"A"),(100,"B"),(0,"C"),(-100,"D")],
        {"A":"Strong expected value","B":"Decent expected value",
         "C":"Near-zero edge","D":"Negative expectancy","F":"Significantly negative"})
    grades.append(MetricGrade("Expectancy", f"${r.expectancy:,.0f} / trade", g, e))
    return grades

def _grade_robustness(r_in, r_out, r_osc, r_trend, r_random):
    grades = []
    ratio = r_out.sharpe_ratio / r_in.sharpe_ratio if r_in.sharpe_ratio > 0 else (1.0 if r_out.sharpe_ratio >= r_in.sharpe_ratio else 0.0)
    g, e = _grade(ratio, [(0.8,"A"),(0.5,"B"),(0.2,"C"),(0.0,"D")],
        {"A":"Out-of-sample held up","B":"Moderate decay",
         "C":"Significant decay","D":"Major decay","F":"Collapsed"})
    grades.append(MetricGrade("In/Out-of-Sample", f"{r_in.sharpe_ratio:.2f} -> {r_out.sharpe_ratio:.2f}", g, e))
    if r_random.total_return_pct <= 0: g, e = "A", "No edge on noise"
    elif r_random.total_return_pct < 3: g, e = "B", "Slight profit on noise"
    elif r_random.total_return_pct < 10: g, e = "C", "Overfitting risk"
    else: g, e = "F", "Profits on noise — overfit"
    grades.append(MetricGrade("Random Walk", f"{r_random.total_return_pct:+.2f}%", g, e))
    better = max(r_osc.sharpe_ratio, r_trend.sharpe_ratio)
    worse = min(r_osc.sharpe_ratio, r_trend.sharpe_ratio)
    if worse >= 0.5: g, e = "A", "Profitable in both regimes"
    elif worse >= 0.0: g, e = "B", "At least breakeven in both"
    elif worse >= -1.0: g, e = "C", "Loses in one regime"
    elif better >= 0.5: g, e = "D", "Fails badly in one regime"
    else: g, e = "F", "Fails in both regimes"
    grades.append(MetricGrade("Regime Adaptability", f"Osc {r_osc.sharpe_ratio:.1f} / Trend {r_trend.sharpe_ratio:.1f}", g, e))
    return grades

def _grade_monte_carlo(mc):
    if mc is None:
        return [MetricGrade("Monte Carlo", "Pending", "C", "Not run")]
    grades = []

    # GBM grade — strategy should NOT profit on pure noise
    gbm_sharpe = mc.get("gbm_sharpe_mean", 0)
    if gbm_sharpe <= -0.5: g, e = "A", "No false edge on noise"
    elif gbm_sharpe <= 0: g, e = "B", "Minimal edge on GBM"
    elif gbm_sharpe <= 0.5: g, e = "C", "Suspicious profit on GBM"
    else: g, e = "F", "Profits on pure noise"
    grades.append(MetricGrade("GBM Noise Test", f"Sharpe: {gbm_sharpe:.2f}", g, e))

    # GAN regime grade — strategy should lose in adverse regimes
    gan = mc.get("gan_results")
    if gan and "crash" in gan:
        crash_mean = float(np.mean(gan["crash"]))
        if crash_mean < -10: g, e = "A", "Loses appropriately in crash"
        elif crash_mean < 0: g, e = "B", "Mild loss in crash scenarios"
        elif crash_mean < 10: g, e = "C", "Suspiciously flat in crash"
        else: g, e = "F", "Profits in crash — likely overfit"
        grades.append(MetricGrade("GAN Crash Test", f"Mean: {crash_mean:+.1f}%", g, e))

    return grades

def _grade_event_driven(ed, bar_result):
    if ed is None:
        return [MetricGrade("Event-Driven", "Pending", "C", "Module not yet connected")]
    ed_r = ed.get("result")
    if ed_r is None:
        return [MetricGrade("Event-Driven", "Error", "F", "No result")]
    diff = abs(bar_result.total_return_pct - ed_r.total_return_pct)
    if diff < 1: g, e = "A", "Bar and event-driven agree"
    elif diff < 3: g, e = "B", "Minor divergence"
    elif diff < 10: g, e = "C", "Notable divergence"
    else: g, e = "D", "Large divergence"
    return [MetricGrade("Event-Driven Match", f"Diff: {diff:.1f}pp", g, e)]


# ─────────────────────────────────────────────────────────────
# Shared drawing primitives
# ─────────────────────────────────────────────────────────────

def _make_fig():
    """Create a 16:9 figure with the base background."""
    fig = plt.figure(figsize=(20, 11.25), facecolor=BG_DEEP, dpi=150)
    return fig

def _page_header(fig, page_label, strategy_name, detail=""):
    """Draw consistent page header with gold accent line."""
    # Thin gold accent line at top
    fig.patches.append(FancyBboxPatch(
        (0.03, 0.965), 0.94, 0.003, transform=fig.transFigure,
        facecolor=ACCENT_GOLD, edgecolor="none", zorder=10,
        boxstyle="round,pad=0"))

    fig.text(0.04, 0.95, page_label, fontsize=11, color=ACCENT_GOLD,
             fontweight="bold", va="top", fontfamily="monospace",
             transform=fig.transFigure)
    fig.text(0.96, 0.95, strategy_name, fontsize=11, color=TEXT_SECONDARY,
             ha="right", va="top", fontfamily="monospace",
             transform=fig.transFigure)
    if detail:
        fig.text(0.04, 0.935, detail, fontsize=10, color=TEXT_MUTED,
                 va="top", transform=fig.transFigure)

def _card_ax(fig, gs_slice):
    """Create a card-style axes with rounded appearance."""
    ax = fig.add_subplot(gs_slice)
    ax.set_facecolor(BG_CARD)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)
    return ax

def _text_ax(fig, gs_slice):
    """Create a text-only axes (no chart)."""
    ax = _card_ax(fig, gs_slice)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return ax

def _section_title(ax, title, y=None):
    """Draw a section title with subtle underline."""
    if y is not None:
        ax.text(0.04, y, title, fontsize=13, color=ACCENT_GOLD,
                fontweight="bold", va="center",
                transform=ax.transAxes if y > 1 else ax.transData)
    else:
        ax.set_title(title, fontsize=13, color=ACCENT_GOLD,
                     fontweight="bold", pad=10, loc="left")

def _stat_row(ax, x_label, x_value, y, label, value, value_color=TEXT_PRIMARY, label_size=12, value_size=12):
    """Draw a single label: value row."""
    ax.text(x_label, y, label, fontsize=label_size, color=TEXT_SECONDARY, va="center")
    ax.text(x_value, y, value, fontsize=value_size, color=value_color,
            va="center", ha="right", fontweight="bold")

def _placeholder_card(ax, title, message, items):
    """Draw a premium-looking placeholder card."""
    ax.set_facecolor(BG_CARD)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Dashed border with subtle glow
    border = FancyBboxPatch(
        (0.03, 0.03), 0.94, 0.87,
        boxstyle="round,pad=0.02",
        facecolor=BG_ELEVATED, edgecolor=BORDER,
        linewidth=1, linestyle=(0, (5, 5)))
    ax.add_patch(border)

    _section_title(ax, title, y=0.93)

    # Icon-like dot
    ax.add_patch(Circle((0.5, 0.62), 0.04, facecolor=ACCENT_GOLD + "15",
                         edgecolor=ACCENT_GOLD + "40", linewidth=1))
    ax.text(0.5, 0.62, "?", fontsize=14, color=ACCENT_GOLD,
            ha="center", va="center", fontweight="bold")

    ax.text(0.5, 0.50, message, fontsize=13, color=TEXT_MUTED,
            ha="center", va="center", style="italic")

    y = 0.36
    for item in items:
        ax.text(0.5, y, f"›  {item}", fontsize=10, color=TEXT_MUTED + "80",
                ha="center", va="center")
        y -= 0.065


# ─────────────────────────────────────────────────────────────
# Page 1: Bar-Based Backtest
# ─────────────────────────────────────────────────────────────

def _render_bartest(strategy, main, data_label, r_osc, r_trend, r_random, r_in, r_out, path):
    fig = _make_fig()
    _page_header(fig, "01 / BAR-BASED BACKTEST", strategy.name,
                 f"{data_label}  ·  {main.num_trades} trades  ·  ${main.equity_curve.iloc[0]:,.0f} capital")

    gs = gridspec.GridSpec(12, 12, figure=fig, hspace=0.7, wspace=0.35,
                           left=0.04, right=0.96, top=0.90, bottom=0.04)

    # Equity curve — hero element
    ax_eq = _card_ax(fig, gs[0:7, 0:12])
    _draw_equity(ax_eq, main, data_label)

    # Bottom strip
    ax_stats = _text_ax(fig, gs[8:12, 0:4])
    _draw_stats_compact(ax_stats, main)

    ax_trades = _text_ax(fig, gs[8:12, 4:8])
    _draw_trade_summary(ax_trades, main)

    ax_dd = _card_ax(fig, gs[8:12, 8:12])
    _draw_drawdown(ax_dd, main)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Page 2: Monte Carlo
# ─────────────────────────────────────────────────────────────

def _render_montecarlo(strategy, data_label, mc_results, path):
    fig = _make_fig()
    _page_header(fig, "02 / MONTE CARLO SIMULATION", strategy.name, data_label)

    gs = gridspec.GridSpec(12, 12, figure=fig, hspace=0.7, wspace=0.35,
                           left=0.04, right=0.96, top=0.90, bottom=0.04)

    if mc_results is None:
        ax1 = _text_ax(fig, gs[0:5, 0:6])
        _placeholder_card(ax1, "GBM Stress Test",
            "Awaiting Monte Carlo module",
            ["Test against Geometric Brownian Motion",
             "No strategy should profit on GBM",
             "Positive Sharpe = overfitting signal"])
        ax2 = _text_ax(fig, gs[0:5, 6:12])
        _placeholder_card(ax2, "GAN Synthetic Data",
            "Awaiting GAN module",
            ["Generate realistic unseen price paths",
             "Preserves fat tails & volatility clustering",
             "Tests on data never seen before"])
        ax3 = _text_ax(fig, gs[6:12, 0:6])
        _placeholder_card(ax3, "Noise Injection",
            "Awaiting noise module",
            ["Add increasing noise to real data",
             "Robust strategy degrades smoothly",
             "Brittle strategy breaks suddenly"])
        ax4 = _text_ax(fig, gs[6:12, 6:12])
        _placeholder_card(ax4, "Distributional Analysis",
            "Awaiting Monte Carlo module",
            ["Return distribution (95% CI)",
             "Drawdown distribution (worst-case)",
             "Ruin probability analysis"])
    else:
        _draw_mc_gbm(fig, gs, mc_results)
        _draw_mc_gan(fig, gs, mc_results)
        _draw_mc_noise(fig, gs, mc_results)
        _draw_mc_distribution(fig, gs, mc_results)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_mc_gbm(fig, gs, mc):
    """Top-left: GBM return distribution histogram."""
    ax = _card_ax(fig, gs[0:5, 0:6])
    returns = mc.get("gbm_returns", [])
    if not returns:
        return

    mean_r = np.mean(returns)
    gbm_sharpe = mc.get("gbm_sharpe_mean", 0)

    ax.hist(returns, bins=20, color=ACCENT_BLUE + "80", edgecolor=ACCENT_BLUE, linewidth=0.5)
    ax.axvline(0, color=TEXT_MUTED, linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(mean_r, color=ACCENT_GOLD, linewidth=2, linestyle="-", label=f"Mean: {mean_r:+.1f}%")

    verdict_color = ACCENT_TEAL if gbm_sharpe <= 0 else ACCENT_RED
    verdict = "PASS — no edge on noise" if gbm_sharpe <= 0 else "FAIL — profits on noise"
    ax.text(0.97, 0.95, verdict, fontsize=10, color=verdict_color,
            ha="right", va="top", fontweight="bold", transform=ax.transAxes)
    ax.text(0.97, 0.85, f"Mean Sharpe: {gbm_sharpe:.2f}", fontsize=9, color=TEXT_SECONDARY,
            ha="right", va="top", transform=ax.transAxes)

    ax.set_xlabel("Return (%)", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("Scenarios", fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.3)
    _section_title(ax, "GBM Stress Test")


def _draw_mc_gan(fig, gs, mc):
    """Top-right: GAN regime box plots."""
    ax = _card_ax(fig, gs[0:5, 6:12])
    gan = mc.get("gan_results")

    if gan is None:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.5, 0.5, "GAN not available\n(torch not installed or checkpoint missing)",
                fontsize=10, color=TEXT_MUTED, ha="center", va="center", style="italic")
        _section_title(ax, "GAN Regime Stress Test")
        return

    regime_names = list(gan.keys())
    regime_data = [gan[r] for r in regime_names]
    regime_colors = {"bullish": ACCENT_TEAL, "bearish": ACCENT_RED, "crash": ACCENT_ORANGE,
                     "sideways": ACCENT_BLUE, "mixed": TEXT_PRIMARY}

    bp = ax.boxplot(regime_data, labels=[r.title() for r in regime_names],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color=ACCENT_GOLD, linewidth=2),
                    whiskerprops=dict(color=TEXT_MUTED),
                    capprops=dict(color=TEXT_MUTED),
                    flierprops=dict(marker=".", markerfacecolor=TEXT_MUTED, markersize=3))

    for patch, name in zip(bp["boxes"], regime_names):
        c = regime_colors.get(name, ACCENT_BLUE)
        patch.set_facecolor(c + "30")
        patch.set_edgecolor(c)

    ax.axhline(0, color=TEXT_MUTED, linewidth=1, linestyle="--", alpha=0.5)
    ax.set_ylabel("Return (%)", fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    _section_title(ax, "GAN Regime Stress Test")


def _draw_mc_noise(fig, gs, mc):
    """Bottom-left: Noise injection degradation curve."""
    ax = _card_ax(fig, gs[6:12, 0:6])
    noise_results = mc.get("noise_results", {})
    noise_levels = mc.get("noise_levels", [])

    if not noise_results:
        return

    means = [np.mean(noise_results[nl]) for nl in noise_levels]
    stds = [np.std(noise_results[nl]) for nl in noise_levels]
    x_pct = [nl * 100 for nl in noise_levels]

    ax.fill_between(x_pct, [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=ACCENT_BLUE + "20", edgecolor="none")
    ax.plot(x_pct, means, color=ACCENT_BLUE, linewidth=2, marker="o", markersize=5, zorder=3)
    ax.axhline(0, color=TEXT_MUTED, linewidth=1, linestyle="--", alpha=0.5)

    # Grade degradation
    if len(means) >= 2 and means[0] != 0:
        drop_pct = (means[0] - means[-1]) / abs(means[0]) * 100 if means[0] != 0 else 0
        if abs(drop_pct) < 30:
            verdict, vc = "Robust — degrades smoothly", ACCENT_TEAL
        elif abs(drop_pct) < 60:
            verdict, vc = "Moderate sensitivity", ACCENT_ORANGE
        else:
            verdict, vc = "Brittle — breaks under noise", ACCENT_RED
        ax.text(0.97, 0.95, verdict, fontsize=10, color=vc,
                ha="right", va="top", fontweight="bold", transform=ax.transAxes)

    ax.set_xlabel("Noise Level (%)", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("Return (%)", fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    _section_title(ax, "Noise Injection")


def _draw_mc_distribution(fig, gs, mc):
    """Bottom-right: Combined return + drawdown distributions."""
    ax = _card_ax(fig, gs[6:12, 6:12])
    gbm_returns = mc.get("gbm_returns", [])
    gbm_drawdowns = mc.get("gbm_drawdowns", [])

    if not gbm_returns:
        return

    # Return distribution
    ax.hist(gbm_returns, bins=15, color=ACCENT_BLUE + "60", edgecolor=ACCENT_BLUE,
            linewidth=0.5, label="Returns", density=True)

    # Overlay drawdown distribution (as negative values)
    if gbm_drawdowns:
        ax2 = ax.twinx()
        ax2.hist(gbm_drawdowns, bins=15, color=ACCENT_RED + "40", edgecolor=ACCENT_RED,
                 linewidth=0.5, label="Max Drawdowns", density=True)
        ax2.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax2.set_ylabel("Drawdown density", fontsize=8, color=TEXT_MUTED)

    # Stats overlay
    p5 = np.percentile(gbm_returns, 5)
    p95 = np.percentile(gbm_returns, 95)
    ax.axvline(p5, color=ACCENT_RED, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.axvline(p95, color=ACCENT_TEAL, linewidth=1.5, linestyle="--", alpha=0.8)

    ax.text(0.97, 0.95, f"95% CI: [{p5:+.1f}%, {p95:+.1f}%]", fontsize=9,
            color=TEXT_SECONDARY, ha="right", va="top", transform=ax.transAxes)
    ruin_pct = sum(1 for r in gbm_returns if r < -50) / len(gbm_returns) * 100
    ax.text(0.97, 0.85, f"Ruin prob (>50% loss): {ruin_pct:.0f}%", fontsize=9,
            color=ACCENT_RED if ruin_pct > 5 else TEXT_SECONDARY,
            ha="right", va="top", transform=ax.transAxes)

    ax.set_xlabel("Return / Drawdown (%)", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("Density", fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    _section_title(ax, "Distributional Analysis")


# ─────────────────────────────────────────────────────────────
# Page 3: Event-Driven
# ─────────────────────────────────────────────────────────────

def _render_eventdriven(strategy, data_label, ed_results, bar_result, path):
    fig = _make_fig()
    _page_header(fig, "03 / EVENT-DRIVEN BACKTEST", strategy.name, data_label)

    gs = gridspec.GridSpec(12, 12, figure=fig, hspace=0.7, wspace=0.35,
                           left=0.04, right=0.96, top=0.90, bottom=0.04)

    if ed_results is None:
        ax1 = _text_ax(fig, gs[0:5, 0:6])
        _placeholder_card(ax1, "Simulated Exchange",
            "Awaiting event-driven module",
            ["Simulates real order book",
             "Processes events as they arrive",
             "Tick-level execution timing"])

        ax2 = _text_ax(fig, gs[0:5, 6:12])
        _placeholder_card(ax2, "Execution Analysis",
            "Awaiting event-driven module",
            ["Fill price accuracy vs bar-based",
             "Slippage comparison",
             "Order flow impact modeling"])

        ax3 = _text_ax(fig, gs[6:12, 0:6])
        _placeholder_card(ax3, "Bar vs Event Comparison",
            "Awaiting event-driven module",
            ["Side-by-side metrics comparison",
             "Return divergence analysis",
             "Execution sensitivity detection"])

        ax4 = _text_ax(fig, gs[6:12, 6:12])
        _placeholder_card(ax4, "Intraday Analysis",
            "Awaiting event-driven module",
            ["Intraday P&L patterns",
             "Time-of-day effects",
             "Required for HF strategies"])
    else:
        ed_r = ed_results.get("result")

        # Top: overlaid equity curves (bar vs event-driven)
        ax_eq = _card_ax(fig, gs[0:6, 0:12])
        ax_eq.plot(bar_result.equity_curve.index, bar_result.equity_curve,
                   color=LINE_STRATEGY, linewidth=2, label="Bar-Based", zorder=3)
        if ed_r is not None and hasattr(ed_r, "equity_curve"):
            ax_eq.plot(ed_r.equity_curve.index, ed_r.equity_curve,
                       color=ACCENT_GOLD, linewidth=2, linestyle="--",
                       label="Event-Driven", zorder=3)
        ax_eq.plot(bar_result.benchmark_curve.index, bar_result.benchmark_curve,
                   color=LINE_BENCHMARK, linewidth=1, linestyle=":",
                   label="Buy & Hold", zorder=2)
        ax_eq.set_ylabel("Portfolio ($)", color=TEXT_SECONDARY, fontsize=11)
        ax_eq.tick_params(colors=TEXT_MUTED, labelsize=9)
        ax_eq.legend(fontsize=10, loc="upper left", facecolor=BG_CARD,
                     edgecolor=BORDER, labelcolor=TEXT_SECONDARY, framealpha=0.9)
        ax_eq.grid(True, alpha=0.08, color=BORDER)
        _section_title(ax_eq, "Bar vs Event-Driven Equity Curves")

        # Bottom left: metrics comparison table
        ax_cmp = _text_ax(fig, gs[7:12, 0:6])
        _draw_engine_comparison(ax_cmp, bar_result, ed_r)

        # Bottom right: divergence analysis
        ax_div = _text_ax(fig, gs[7:12, 6:12])
        _draw_divergence_analysis(ax_div, bar_result, ed_r)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_engine_comparison(ax, bar_r, ed_r):
    """Side-by-side metrics comparison between bar and event-driven engines."""
    _section_title(ax, "Metrics Comparison", y=0.95)

    if ed_r is None:
        ax.text(0.5, 0.5, "No event-driven result", fontsize=12,
                color=TEXT_MUTED, ha="center", va="center")
        return

    rows = [
        ("Total Return", f"{bar_r.total_return_pct:+.2f}%", f"{ed_r.total_return_pct:+.2f}%"),
        ("Sharpe Ratio", f"{bar_r.sharpe_ratio:.2f}", f"{ed_r.sharpe_ratio:.2f}"),
        ("Max Drawdown", f"{bar_r.max_drawdown_pct:.1f}%", f"{ed_r.max_drawdown_pct:.1f}%"),
        ("Trades", f"{bar_r.num_trades}", f"{ed_r.num_trades}"),
        ("Win Rate", f"{bar_r.win_rate:.1f}%", f"{ed_r.win_rate:.1f}%"),
        ("Profit Factor", f"{bar_r.profit_factor:.2f}", f"{ed_r.profit_factor:.2f}"),
        ("Expectancy", f"${bar_r.expectancy:,.0f}", f"${ed_r.expectancy:,.0f}"),
    ]

    # Header
    y = 0.85
    hdr = dict(fontsize=10, color=TEXT_MUTED, va="center", fontweight="bold")
    ax.text(0.04, y, "Metric", **hdr)
    ax.text(0.55, y, "Bar", ha="center", **hdr)
    ax.text(0.82, y, "Event", ha="center", **hdr)

    y -= 0.03
    ax.axhline(y=y, xmin=0.03, xmax=0.97, color=BORDER, linewidth=0.5)
    y -= 0.05

    for label, bar_val, ed_val in rows:
        ax.text(0.04, y, label, fontsize=11, color=TEXT_SECONDARY, va="center")
        ax.text(0.55, y, bar_val, fontsize=11, color=TEXT_PRIMARY,
                va="center", ha="center", fontweight="bold")
        ax.text(0.82, y, ed_val, fontsize=11, color=ACCENT_GOLD,
                va="center", ha="center", fontweight="bold")
        y -= 0.10


def _draw_divergence_analysis(ax, bar_r, ed_r):
    """Analyze divergence between bar and event-driven results."""
    _section_title(ax, "Divergence Analysis", y=0.95)

    if ed_r is None:
        ax.text(0.5, 0.5, "No event-driven result", fontsize=12,
                color=TEXT_MUTED, ha="center", va="center")
        return

    ret_diff = abs(bar_r.total_return_pct - ed_r.total_return_pct)
    sharpe_diff = abs(bar_r.sharpe_ratio - ed_r.sharpe_ratio)
    trade_diff = abs(bar_r.num_trades - ed_r.num_trades)
    dd_diff = abs(bar_r.max_drawdown_pct - ed_r.max_drawdown_pct)

    # Verdict + explanation lines based on divergence level
    if ret_diff < 1 and trade_diff == 0:
        verdict = "Engines Agree"
        verdict_color = ACCENT_TEAL
        explain = [
            "Strategy uses simple order types (market orders)",
            "that both engines handle identically.",
            "",
            "Safe to use the faster bar-based engine.",
        ]
    elif ret_diff < 3:
        verdict = "Minor Divergence"
        verdict_color = ACCENT_GOLD
        explain = [
            "Small differences from execution timing.",
            "Results are still comparable.",
            "",
            "Bar engine is a reasonable approximation.",
        ]
    elif ret_diff < 10:
        verdict = "Significant Divergence"
        verdict_color = ACCENT_ORANGE
        which_higher = "Event" if ed_r.total_return_pct > bar_r.total_return_pct else "Bar"
        explain = [
            f"{which_higher} engine shows higher returns.",
            "Strategy uses order types (limit/stop/brackets)",
            "that fill at different prices per engine.",
            "Use event-driven results for accuracy.",
        ]
    else:
        verdict = "Execution Sensitive"
        verdict_color = ACCENT_RED
        which_higher = "Event" if ed_r.total_return_pct > bar_r.total_return_pct else "Bar"
        explain = [
            f"{ret_diff:.0f}pp gap — results depend heavily on",
            "how orders are filled (price, timing, brackets).",
            f"{which_higher} engine shows higher returns.",
            "Bar backtest is unreliable for this strategy.",
        ]

    y = 0.84
    ax.text(0.5, y, verdict, fontsize=16, color=verdict_color,
            ha="center", va="center", fontweight="bold")

    y -= 0.10
    ax.axhline(y=y, xmin=0.1, xmax=0.9, color=BORDER, linewidth=0.5)
    y -= 0.07

    divergences = [
        ("Return Diff", f"{ret_diff:.2f}pp", ACCENT_TEAL if ret_diff < 1 else (ACCENT_ORANGE if ret_diff < 5 else ACCENT_RED)),
        ("Sharpe Diff", f"{sharpe_diff:.2f}", ACCENT_TEAL if sharpe_diff < 0.1 else ACCENT_ORANGE),
        ("Trade Count Diff", f"{trade_diff:.0f}", ACCENT_TEAL if trade_diff == 0 else ACCENT_ORANGE),
        ("Max DD Diff", f"{dd_diff:.1f}pp", ACCENT_TEAL if dd_diff < 2 else (ACCENT_ORANGE if dd_diff < 10 else ACCENT_RED)),
    ]

    for label, value, color in divergences:
        ax.text(0.08, y, label, fontsize=11, color=TEXT_MUTED, va="center")
        ax.text(0.92, y, value, fontsize=11, color=color,
                va="center", ha="right", fontweight="bold")
        y -= 0.095

    # Context explanation — changes based on verdict
    y -= 0.04
    ax.axhline(y=y, xmin=0.1, xmax=0.9, color=BORDER, linewidth=0.3, alpha=0.4)
    y -= 0.06
    for line in explain:
        ax.text(0.5, y, line, fontsize=9, color=TEXT_MUTED,
                ha="center", va="center")
        y -= 0.055


# ─────────────────────────────────────────────────────────────
# Page 4: Strategy Scorecard
# ─────────────────────────────────────────────────────────────

def _render_scorecard(
    strategy, main, data_label,
    grades_perf, grades_risk, grades_trade, grades_robust, grades_mc, grades_ed,
    overall_letter, overall_score,
    r_osc, r_trend, r_random, r_in, r_out,
    mc_results, ed_results, path,
):
    fig = _make_fig()
    _page_header(fig, "04 / STRATEGY SCORECARD", strategy.name, data_label)

    gs = gridspec.GridSpec(12, 12, figure=fig, hspace=0.6, wspace=0.35,
                           left=0.04, right=0.96, top=0.90, bottom=0.04)

    # Badge + key metrics + multi-dataset
    ax_badge = _text_ax(fig, gs[0:4, 0:3])
    _draw_badge(ax_badge, overall_letter, overall_score)

    ax_stats = _text_ax(fig, gs[0:4, 3:7])
    _draw_stats_compact(ax_stats, main)

    ax_synth = _text_ax(fig, gs[0:4, 7:12])
    _draw_synthetic(ax_synth, r_osc, r_trend, r_random, r_in, r_out)

    # Performance + Risk
    ax_perf = _text_ax(fig, gs[4:6, 0:6])
    _draw_grades(ax_perf, "Performance", grades_perf)
    ax_risk = _text_ax(fig, gs[4:6, 6:12])
    _draw_grades(ax_risk, "Risk", grades_risk)

    # Trade Quality + Robustness
    ax_trade = _text_ax(fig, gs[6:9, 0:6])
    _draw_grades(ax_trade, "Trade Quality", grades_trade)
    ax_robust = _text_ax(fig, gs[6:9, 6:12])
    _draw_grades(ax_robust, "Robustness", grades_robust)

    # MC + Event-Driven grades
    ax_mc = _text_ax(fig, gs[9:11, 0:6])
    _draw_grades(ax_mc, "Monte Carlo", grades_mc)
    ax_ed = _text_ax(fig, gs[9:11, 6:12])
    _draw_grades(ax_ed, "Event-Driven", grades_ed)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Component renderers
# ─────────────────────────────────────────────────────────────

def _draw_badge(ax, letter, score):
    color = GRADE_COLORS.get(letter, "#888")
    cx, cy = 0.50, 0.52

    # Title at top
    ax.text(cx, 0.95, "OVERALL", fontsize=12, color=ACCENT_GOLD,
            ha="center", va="top", fontweight="bold")

    # Outer ring
    ax.add_patch(Circle((cx, cy), 0.30, facecolor="none",
                         edgecolor=color + "30", linewidth=3))
    # Inner glow
    ax.add_patch(Circle((cx, cy), 0.24, facecolor=color + "20",
                         edgecolor=color + "55", linewidth=1.5))
    # Letter — centered in the circle
    ax.text(cx, cy, letter, fontsize=68, fontweight="bold",
            color=color, ha="center", va="center")

    # Score below circle
    ax.text(cx, 0.14, f"{score:.1f} / 4.0", fontsize=16,
            color=TEXT_SECONDARY, ha="center", va="center")


def _draw_equity(ax, r, title):
    ax.plot(r.equity_curve.index, r.equity_curve,
            color=LINE_STRATEGY, linewidth=2.2, label="Strategy", zorder=3)
    ax.plot(r.benchmark_curve.index, r.benchmark_curve,
            color=LINE_BENCHMARK, linewidth=1.2, linestyle="--",
            label="Buy & Hold", zorder=2)

    # Subtle gradient fill
    ax.fill_between(r.equity_curve.index, r.equity_curve,
                     r.equity_curve.iloc[0], alpha=0.06, color=LINE_STRATEGY)

    ax.set_ylabel("Portfolio ($)", color=TEXT_SECONDARY, fontsize=11)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.legend(fontsize=10, loc="upper left", facecolor=BG_CARD,
              edgecolor=BORDER, labelcolor=TEXT_SECONDARY, framealpha=0.9)
    ax.grid(True, alpha=0.08, color=BORDER)
    _section_title(ax, title)


def _draw_drawdown(ax, r):
    peak = r.equity_curve.cummax()
    dd = (r.equity_curve - peak) / peak * 100
    ax.fill_between(dd.index, dd, 0, color=ACCENT_RED, alpha=0.15)
    ax.plot(dd.index, dd, color=ACCENT_RED, linewidth=1, alpha=0.8)
    ax.set_ylabel("Drawdown %", color=TEXT_SECONDARY, fontsize=10)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.grid(True, alpha=0.08, color=BORDER)
    _section_title(ax, f"Drawdown  ·  Max: {r.max_drawdown_pct:.1f}%")


def _draw_stats_compact(ax, r):
    _section_title(ax, "Key Metrics", y=0.95)
    kelly = f"{r.kelly_criterion:.1%}" if np.isfinite(r.kelly_criterion) else "—"
    stats = [
        ("CAGR", f"{r.cagr:+.2f}%", ACCENT_TEAL if r.cagr > 0 else ACCENT_RED),
        ("Benchmark", f"{r.benchmark_return_pct:+.2f}%", TEXT_SECONDARY),
        ("Sharpe / SQN", f"{r.sharpe_ratio:.2f}  /  {r.sqn:.2f}", TEXT_PRIMARY),
        ("Max Drawdown", f"{r.max_drawdown_pct:.1f}%", ACCENT_RED),
        ("Expectancy", f"${r.expectancy:,.0f}", TEXT_PRIMARY),
        ("Profit Factor", f"{r.profit_factor:.2f}", TEXT_PRIMARY),
        ("Kelly", kelly, TEXT_SECONDARY),
        ("Trades", f"{r.num_trades}", TEXT_SECONDARY),
    ]
    y = 0.85
    for label, value, vcolor in stats:
        ax.text(0.04, y, label, fontsize=11, color=TEXT_MUTED, va="center")
        ax.text(0.96, y, value, fontsize=11, color=vcolor,
                va="center", ha="right", fontweight="bold")
        y -= 0.108


def _draw_trade_summary(ax, r):
    _section_title(ax, "Trade Statistics", y=0.95)
    stats = [
        ("Total Trades", f"{r.num_trades}", TEXT_PRIMARY),
        ("Profit Factor", f"{r.profit_factor:.2f}", TEXT_PRIMARY),
        ("Best Trade", f"{r.best_trade_pct:+.2f}%", ACCENT_TEAL),
        ("Worst Trade", f"{r.worst_trade_pct:+.2f}%", ACCENT_RED),
        ("Avg Trade", f"{r.avg_trade_return_pct:+.2f}%",
         ACCENT_TEAL if r.avg_trade_return_pct > 0 else ACCENT_RED),
        ("Max Consec. Losses", f"{r.max_consecutive_losses}", TEXT_PRIMARY),
        ("Avg Holding", f"{r.avg_holding_bars:.0f} bars", TEXT_SECONDARY),
        ("Total Costs", f"${r.total_commissions + r.total_slippage:,.0f}", TEXT_SECONDARY),
    ]
    y = 0.85
    for label, value, vcolor in stats:
        ax.text(0.04, y, label, fontsize=11, color=TEXT_MUTED, va="center")
        ax.text(0.96, y, value, fontsize=11, color=vcolor,
                va="center", ha="right", fontweight="bold")
        y -= 0.108


def _draw_synthetic(ax, r_osc, r_trend, r_random, r_in, r_out):
    _section_title(ax, "Multi-Dataset Results", y=0.95)

    rows = [("Oscillating", r_osc), ("Trending", r_trend),
            ("Random Walk", r_random), ("In-Sample", r_in),
            ("Out-of-Sample", r_out)]

    y = 0.84
    hdr = dict(fontsize=10, color=TEXT_MUTED, va="center", fontweight="bold")
    ax.text(0.04, y, "Dataset", **hdr)
    ax.text(0.45, y, "Return", ha="center", **hdr)
    ax.text(0.65, y, "Sharpe", ha="center", **hdr)
    ax.text(0.82, y, "SQN", ha="center", **hdr)
    ax.text(0.95, y, "#", ha="center", **hdr)

    y -= 0.035
    ax.axhline(y=y, xmin=0.03, xmax=0.97, color=BORDER, linewidth=0.5)
    y -= 0.045

    for label, r in rows:
        ret_color = ACCENT_TEAL if r.total_return_pct > 0 else ACCENT_RED
        sqn = r.sqn if np.isfinite(r.sqn) else 0.0
        ax.text(0.04, y, label, fontsize=11, color=TEXT_SECONDARY, va="center")
        ax.text(0.45, y, f"{r.total_return_pct:+.1f}%", fontsize=11,
                color=ret_color, va="center", ha="center", fontweight="bold")
        ax.text(0.65, y, f"{r.sharpe_ratio:.2f}", fontsize=11,
                color=TEXT_PRIMARY, va="center", ha="center")
        ax.text(0.82, y, f"{sqn:.2f}", fontsize=11,
                color=TEXT_PRIMARY, va="center", ha="center")
        ax.text(0.95, y, f"{r.num_trades}", fontsize=11,
                color=TEXT_MUTED, va="center", ha="center")
        y -= 0.13


def _draw_grades(ax, title, grades):
    _section_title(ax, title, y=0.95)

    n = len(grades)
    if n == 0:
        return

    # Sizes
    letter_size = 18
    name_size = 11
    value_size = 9
    expl_size = 10

    # Spacing: evenly distribute grades between title and bottom
    usable = 0.72
    y_spacing = usable / max(n, 1)
    y_spacing = min(y_spacing, 0.30)

    y = 0.85 - y_spacing / 2
    if n == 1:
        y = 0.45

    for i, g in enumerate(grades):
        color = GRADE_COLORS.get(g.grade, "#888")

        # Grade letter with auto-sized bbox (scales with font, not axes)
        ax.text(0.055, y, g.grade, fontsize=letter_size, fontweight="bold",
                color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=color + "28",
                          edgecolor=color + "55",
                          linewidth=0.6))

        # Name + value stacked next to pill
        ax.text(0.13, y + 0.02, g.name, fontsize=name_size,
                color=TEXT_PRIMARY, va="bottom", fontweight="bold")
        ax.text(0.13, y - 0.02, g.value, fontsize=value_size,
                color=TEXT_MUTED, va="top")

        # Explanation
        ax.text(0.42, y, g.explanation, fontsize=expl_size, color=TEXT_SECONDARY,
                va="center", style="italic")

        # Separator
        if i < n - 1:
            sep_y = y - y_spacing / 2 - 0.01
            ax.axhline(y=sep_y, xmin=0.03, xmax=0.97,
                       color=BORDER, linewidth=0.3, alpha=0.5)

        y -= y_spacing


# ─────────────────────────────────────────────────────────────
# Options Scorecard
# ─────────────────────────────────────────────────────────────

def generate_options_scorecard(
    strategy,
    df: pd.DataFrame,
    config=None,
    output_path: str = "options_scorecard.png",
    symbol: str = "",
    theme: str = "dark",
) -> str:
    """
    Generate a single-page scorecard for an options strategy.

    Runs the options backtester and produces a report with equity curve,
    key metrics, trade log, and grades.
    """
    _apply_theme(theme)
    from backtester.options.engine import OptionsBacktester, OptionsBacktestConfig, OptionsBacktestResult

    cfg = config or OptionsBacktestConfig()
    engine = OptionsBacktester(cfg)
    result = engine.run(strategy, df)

    if not symbol:
        symbol = df.attrs.get("symbol", "")
    date_start = df.index[0].strftime("%Y-%m-%d")
    date_end = df.index[-1].strftime("%Y-%m-%d")
    data_label = f"{symbol} " if symbol else ""
    data_label += f"{date_start} to {date_end}"

    # Grade the result
    grades = _grade_options(result)
    overall_score = sum(_letter_to_score(g.grade) for g in grades) / max(len(grades), 1)
    overall_letter = _score_to_letter(overall_score)

    fig = _make_fig()
    _page_header(fig, "OPTIONS STRATEGY SCORECARD", strategy.name, data_label)

    gs = gridspec.GridSpec(12, 12, figure=fig, hspace=0.7, wspace=0.35,
                           left=0.04, right=0.96, top=0.90, bottom=0.04)

    # Row 1: Badge + Metrics + Trade Stats
    ax_badge = _text_ax(fig, gs[0:4, 0:3])
    _draw_badge(ax_badge, overall_letter, overall_score)

    ax_stats = _text_ax(fig, gs[0:4, 3:7])
    _draw_options_stats(ax_stats, result)

    ax_trades = _text_ax(fig, gs[0:4, 7:12])
    _draw_options_trades(ax_trades, result)

    # Row 2: Equity curve
    ax_eq = _card_ax(fig, gs[4:7, 0:12])
    _draw_options_equity(ax_eq, result, data_label)

    # Row 3: Grades — two columns for 6 grades
    ax_grades_l = _text_ax(fig, gs[8:12, 0:6])
    ax_grades_r = _text_ax(fig, gs[8:12, 6:12])
    _draw_grades(ax_grades_l, "Strategy Grades", grades[:3])
    _draw_grades(ax_grades_r, "", grades[3:])

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Options scorecard: {output_path}")
    return output_path


def _grade_options(result) -> List[MetricGrade]:
    """Grade an options backtest result relative to benchmark."""
    grades = []
    bm = getattr(result, "benchmark_return_pct", 0)

    # Excess return (vs benchmark) — the only return grade that matters
    excess = result.total_return_pct - bm
    g, e = _grade(excess, [(10, "A"), (0, "B"), (-5, "C"), (-15, "D")],
        {"A": "Beats benchmark significantly", "B": "Matches or beats benchmark",
         "C": "Slightly trails benchmark", "D": "Trails benchmark",
         "F": "Far behind benchmark"})
    grades.append(MetricGrade("Excess Return",
        f"{result.total_return_pct:+.1f}% vs {bm:+.1f}% B&H ({excess:+.1f}pp)", g, e))

    # Sharpe ratio — risk-adjusted, NOT benchmark-relative
    g, e = _grade(result.sharpe_ratio, [(2.0, "A"), (1.0, "B"), (0.5, "C"), (0.0, "D")],
        {"A": "Excellent risk-adjusted", "B": "Good risk-adjusted", "C": "Modest",
         "D": "Barely positive", "F": "Negative risk-adjusted"})
    grades.append(MetricGrade("Sharpe Ratio", f"{result.sharpe_ratio:.2f}", g, e))

    # Max drawdown
    dd = abs(result.max_drawdown_pct)
    g, e = _grade(-dd, [(-3, "A"), (-7, "B"), (-15, "C"), (-25, "D")],
        {"A": "Minimal drawdown", "B": "Moderate drawdown", "C": "Notable drawdown",
         "D": "Large drawdown", "F": "Severe drawdown"})
    grades.append(MetricGrade("Max Drawdown", f"{result.max_drawdown_pct:.2f}%", g, e))

    # Win rate
    g, e = _grade(result.win_rate, [(75, "A"), (60, "B"), (50, "C"), (40, "D")],
        {"A": "Very high win rate", "B": "Good win rate", "C": "Average win rate",
         "D": "Below average", "F": "Poor win rate"})
    grades.append(MetricGrade("Win Rate", f"{result.win_rate:.0f}%", g, e))

    # Premium efficiency (collected vs paid)
    net_premium = result.total_premium_collected - result.total_premium_paid
    if net_premium > 0:
        g, e = "A", "Net premium seller"
    elif net_premium == 0:
        g, e = "C", "Premium neutral"
    else:
        g, e = "D", "Net premium buyer"
    grades.append(MetricGrade("Net Premium", f"${net_premium:,.0f}", g, e))

    # Cost efficiency
    total_costs = result.total_commissions + result.total_spread_cost
    cost_pct = total_costs / max(abs(result.total_pnl), 1) * 100
    g, e = _grade(-cost_pct, [(-5, "A"), (-15, "B"), (-30, "C"), (-50, "D")],
        {"A": "Very low costs", "B": "Reasonable costs", "C": "Moderate cost drag",
         "D": "High cost drag", "F": "Costs eating profits"})
    grades.append(MetricGrade("Cost Efficiency",
        f"${total_costs:,.0f} ({cost_pct:.0f}% of P&L)", g, e))

    return grades


def _draw_options_stats(ax, result):
    """Draw key options metrics."""
    _section_title(ax, "Key Metrics", y=0.95)
    bm_ret = getattr(result, "benchmark_return_pct", 0)
    stats = [
        ("Total Return", f"{result.total_return_pct:+.2f}%",
         ACCENT_TEAL if result.total_return_pct > 0 else ACCENT_RED),
        ("Benchmark (B&H)", f"{bm_ret:+.2f}%", TEXT_SECONDARY),
        ("Sharpe Ratio", f"{result.sharpe_ratio:.2f}", TEXT_PRIMARY),
        ("Max Drawdown", f"{result.max_drawdown_pct:.2f}%", ACCENT_RED),
        ("Premium Collected", f"${result.total_premium_collected:,.0f}", ACCENT_TEAL),
        ("Premium Paid", f"${result.total_premium_paid:,.0f}", TEXT_SECONDARY),
        ("Commissions", f"${result.total_commissions:,.0f}", TEXT_SECONDARY),
        ("Spread Costs", f"${result.total_spread_cost:,.0f}", TEXT_SECONDARY),
    ]
    y = 0.85
    for label, value, vcolor in stats:
        ax.text(0.04, y, label, fontsize=11, color=TEXT_MUTED, va="center")
        ax.text(0.96, y, value, fontsize=11, color=vcolor,
                va="center", ha="right", fontweight="bold")
        y -= 0.108


def _draw_options_trades(ax, result):
    """Draw options trade stats."""
    _section_title(ax, "Trade Statistics", y=0.95)
    stats = [
        ("Total Trades", f"{result.num_trades}", TEXT_PRIMARY),
        ("Win Rate", f"{result.win_rate:.1f}%",
         ACCENT_TEAL if result.win_rate > 50 else ACCENT_RED),
        ("Avg Trade P&L", f"${result.avg_trade_pnl:,.0f}",
         ACCENT_TEAL if result.avg_trade_pnl > 0 else ACCENT_RED),
        ("Best Trade", f"${result.best_trade_pnl:,.0f}", ACCENT_TEAL),
        ("Worst Trade", f"${result.worst_trade_pnl:,.0f}", ACCENT_RED),
    ]

    # Add sample trades from trade log
    y = 0.85
    for label, value, vcolor in stats:
        ax.text(0.04, y, label, fontsize=11, color=TEXT_MUTED, va="center")
        ax.text(0.96, y, value, fontsize=11, color=vcolor,
                va="center", ha="right", fontweight="bold")
        y -= 0.108

    # Recent trades
    y -= 0.04
    ax.axhline(y=y, xmin=0.03, xmax=0.97, color=BORDER, linewidth=0.5)
    y -= 0.06
    ax.text(0.04, y, "Recent Trades", fontsize=10, color=ACCENT_GOLD,
            va="center", fontweight="bold")
    y -= 0.07
    for t in result.trade_log[-5:]:
        pnl = t.get("pnl", 0)
        color = ACCENT_TEAL if pnl > 0 else ACCENT_RED
        label = f"{t.get('option_type', '?').upper()} K={t.get('strike', 0):.0f}"
        ax.text(0.04, y, label, fontsize=9, color=TEXT_MUTED, va="center")
        ax.text(0.96, y, f"${pnl:+,.0f}", fontsize=9, color=color,
                va="center", ha="right", fontweight="bold")
        y -= 0.06


def _draw_options_equity(ax, result, title):
    """Draw options equity curve with benchmark."""
    eq = result.equity_curve
    ax.plot(eq.index, eq, color=ACCENT_GOLD, linewidth=2.2, label="Strategy", zorder=3)
    ax.fill_between(eq.index, eq, eq.iloc[0], alpha=0.06, color=ACCENT_GOLD)

    # Benchmark (buy & hold underlying)
    if hasattr(result, "benchmark_curve") and result.benchmark_curve is not None:
        bm = result.benchmark_curve
        ax.plot(bm.index, bm, color=LINE_BENCHMARK, linewidth=1.2,
                linestyle="--", label="Buy & Hold", zorder=2)

    ax.set_ylabel("Portfolio ($)", color=TEXT_SECONDARY, fontsize=11)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.legend(fontsize=10, loc="upper left", facecolor=BG_CARD,
              edgecolor=BORDER, labelcolor=TEXT_SECONDARY, framealpha=0.9)
    ax.grid(True, alpha=0.08, color=BORDER)
    _section_title(ax, title)
