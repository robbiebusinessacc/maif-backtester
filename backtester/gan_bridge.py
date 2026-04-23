"""
Bridge between Sachit's cGAN and the backtester's GANSource adapter.

Usage:
    from backtester.gan_bridge import make_gan_source

    # Single regime:
    crash_source = make_gan_source("crash", n_bars=252)
    result = run_scenario_suite(strategy, crash_source, n_scenarios=100)

    # All regimes mixed:
    mixed_source = make_gan_source("mixed", n_bars=252)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from backtester.synthetic import GANSource, _build_ohlcv_from_close

# ── Paths ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GAN_DIR = _PROJECT_ROOT / "SImulated Data" / "cGAN"
_CHECKPOINT_DIR = _GAN_DIR / "OutputModels"
_DEFAULT_CHECKPOINT = "checkpoint_epoch_1000.pt"

# ── Regime map (must match training) ─────────────────────────────────
REGIME_MAP = {
    "bullish": 0,
    "bearish": 1,
    "sideways": 2,
    "crash": 3,
}

# ── Global normalization stats from training data ────────────────────
# Computed from the 4 regime CSVs (bullish + bearish + sideways + crash SPY)
_GLOBAL_MU = -0.0007717194
_GLOBAL_SIGMA = 0.0190581171

# The model was trained on 30-step windows (seq_len=30 in main.py)
_TRAINED_SEQ_LEN = 30


def _load_generator(checkpoint_name: str = _DEFAULT_CHECKPOINT):
    """Load the trained Generator from a checkpoint file."""
    # Add cGAN dir to path so we can import models
    gan_dir_str = str(_GAN_DIR)
    if gan_dir_str not in sys.path:
        sys.path.insert(0, gan_dir_str)

    from models import Generator

    ckpt_path = _CHECKPOINT_DIR / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"GAN checkpoint not found: {ckpt_path}\n"
            f"Available: {[f.name for f in _CHECKPOINT_DIR.glob('*.pt')]}"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["generator_state_dict"]

    # Infer architecture from state_dict shapes
    embed_weight = state_dict["regime_embedding.weight"]
    num_regimes, embed_dim = embed_weight.shape
    lstm_ih = state_dict["lstm.weight_ih_l0"]
    hidden_dim = lstm_ih.shape[0] // 4
    noise_dim = lstm_ih.shape[1] - embed_dim
    num_layers = sum(1 for k in state_dict if k.startswith("lstm.weight_ih_l"))

    gen = Generator(
        noise_dim=noise_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
        num_regimes=num_regimes,
        embed_dim=embed_dim,
    )
    gen.load_state_dict(state_dict)
    gen.eval()

    cfg = {
        "noise_dim": noise_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_regimes": num_regimes,
        "embed_dim": embed_dim,
    }
    return gen, cfg


# Module-level cache so we don't reload the model on every scenario
_CACHED_GEN = None
_CACHED_CFG = None


def _get_generator():
    global _CACHED_GEN, _CACHED_CFG
    if _CACHED_GEN is None:
        _CACHED_GEN, _CACHED_CFG = _load_generator()
    return _CACHED_GEN, _CACHED_CFG


def generate_prices(
    regime: str = "crash",
    n_bars: int = 252,
    seed: Optional[int] = None,
    start_price: float = 100.0,
) -> np.ndarray:
    """
    Generate a single price path from the cGAN for a given regime.

    The model was trained on 30-step windows, so for longer paths we
    generate multiple windows and stitch them together. Each window
    picks up where the last one ended (continuous price path).

    Returns a 1-D numpy array of close prices (length n_bars).
    """
    gen, cfg = _get_generator()
    noise_dim = cfg["noise_dim"]
    regime_label = REGIME_MAP[regime.lower()]

    if seed is not None:
        torch.manual_seed(seed)

    # Generate in chunks of _TRAINED_SEQ_LEN and stitch
    all_log_returns = []
    remaining = n_bars
    while remaining > 0:
        chunk_len = min(remaining, _TRAINED_SEQ_LEN)
        with torch.no_grad():
            noise = torch.randn(1, chunk_len, noise_dim)
            labels = torch.tensor([regime_label], dtype=torch.long)
            normalized = gen(noise, labels).squeeze(0).squeeze(-1).numpy()
        log_returns = normalized * _GLOBAL_SIGMA + _GLOBAL_MU
        all_log_returns.append(log_returns)
        remaining -= chunk_len

    full_log_returns = np.concatenate(all_log_returns)[:n_bars]
    prices = start_price * np.exp(np.cumsum(full_log_returns))
    return prices


def make_gan_source(
    regime: str = "crash",
    n_bars: int = 252,
    start_price: float = 100.0,
    name: Optional[str] = None,
) -> GANSource:
    """
    Create a GANSource that plugs directly into run_scenario_suite().

    Args:
        regime: "bullish", "bearish", "sideways", "crash", or "mixed"
                "mixed" picks a random regime per scenario.
        n_bars: Length of each generated price path.
        start_price: Starting price level.
        name: Display name for reports.
    """
    regimes_list = list(REGIME_MAP.keys())

    def _generator(seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)

        if regime.lower() == "mixed":
            chosen = rng.choice(regimes_list)
        else:
            chosen = regime.lower()

        prices = generate_prices(
            regime=chosen,
            n_bars=n_bars,
            seed=seed,
            start_price=start_price,
        )
        return _build_ohlcv_from_close(prices, rng)

    display_name = name or f"cGAN {regime.title()} Scenarios"
    return GANSource(generator=_generator, name=display_name, symbol="GAN_SIM")
