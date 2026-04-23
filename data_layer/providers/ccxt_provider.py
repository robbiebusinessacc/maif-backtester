"""CCXT cryptocurrency exchange data provider."""

from __future__ import annotations

from datetime import date, datetime, timezone
import logging
import time
from typing import Optional

import pandas as pd

from data_layer.providers.base import DataProvider

try:
    import ccxt
except ImportError:
    ccxt = None

logger = logging.getLogger(__name__)

# Maximum candles returned by most exchanges per request.
_MAX_CANDLES_PER_REQUEST = 1000

# Retry configuration for transient network errors.
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds; doubles each retry


class CCXTProvider(DataProvider):
    """
    Fetches OHLCV data from cryptocurrency exchanges via the ccxt library.

    Supports 100+ exchanges (Binance, Coinbase, Kraken, etc.).
    Public market data (OHLCV) does not require API keys on most exchanges.
    Supply ``api_key`` and ``secret`` only if you need authenticated endpoints.

    Install dependency:  ``pip install ccxt``
    """

    # Map common interval strings to ccxt timeframe notation.
    _INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w",
        "1M": "1M",
    }

    def __init__(
        self,
        exchange_name: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
    ):
        if ccxt is None:
            raise ImportError(
                "The ccxt library is required for CCXTProvider. "
                "Install it with:  pip install ccxt"
            )

        exchange_class = getattr(ccxt, exchange_name, None)
        if exchange_class is None:
            raise ValueError(
                f"Exchange '{exchange_name}' is not supported by ccxt. "
                f"See ccxt.exchanges for the full list."
            )

        config: dict = {"enableRateLimit": True}
        if api_key:
            config["apiKey"] = api_key
        if secret:
            config["secret"] = secret

        self._exchange: ccxt.Exchange = exchange_class(config)
        self._exchange_name = exchange_name

    @property
    def name(self) -> str:
        return f"CCXT ({self._exchange_name})"

    # ------------------------------------------------------------------
    # Symbol helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_ccxt_symbol(symbol: str) -> str:
        """Convert a compact symbol like ``BTCUSDT`` to ccxt format ``BTC/USDT``.

        If the symbol already contains a ``/`` it is returned as-is.
        """
        if "/" in symbol:
            return symbol.upper()

        # Common quote currencies ordered longest-first so we match
        # ``BTCBUSD`` before ``BTCUSD`` (if an exchange had both).
        quotes = ["USDT", "BUSD", "USDC", "TUSD", "USD", "BTC", "ETH", "BNB"]
        upper = symbol.upper()
        for quote in quotes:
            if upper.endswith(quote) and len(upper) > len(quote):
                base = upper[: -len(quote)]
                return f"{base}/{quote}"

        # Fallback: return as-is and let ccxt raise if invalid.
        return upper

    # ------------------------------------------------------------------
    # Core data fetching
    # ------------------------------------------------------------------
    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        timeframe = self._INTERVAL_MAP.get(interval)
        if timeframe is None:
            raise NotImplementedError(
                f"[{self.name}] Interval '{interval}' is not supported. "
                f"Supported: {', '.join(self._INTERVAL_MAP.keys())}"
            )

        ccxt_symbol = self._to_ccxt_symbol(symbol)

        # Convert dates to millisecond timestamps (UTC).
        since_ms = int(
            datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )
        # End is inclusive — push to end-of-day.
        end_ms = int(
            datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )

        all_candles = []

        cursor_ms = since_ms
        while cursor_ms <= end_ms:
            candles = self._fetch_with_retry(ccxt_symbol, timeframe, cursor_ms)

            if not candles:
                break

            # Filter out any candles beyond our end boundary.
            candles = [c for c in candles if c[0] <= end_ms]
            if not candles:
                break

            all_candles.extend(candles)

            # Advance cursor past the last candle we received.
            last_ts = candles[-1][0]
            if last_ts <= cursor_ms:
                # Safety: avoid infinite loop if exchange returns same data.
                break
            cursor_ms = last_ts + 1

            # Respect exchange rate limits between paginated requests.
            time.sleep(self._exchange.rateLimit / 1000)

        if not all_candles:
            raise ValueError(
                f"[{self.name}] No data returned for {ccxt_symbol} "
                f"({start} to {end}, interval={interval})"
            )

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")

        # Remove duplicate timestamps that can occur at page boundaries.
        df = df[~df.index.duplicated(keep="first")]

        df.attrs["symbol"] = symbol

        return self._normalize(df)

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------
    def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
    ) -> list:
        """Fetch a single page of OHLCV candles with retry logic."""
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            try:
                return self._exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=_MAX_CANDLES_PER_REQUEST,
                )
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as exc:
                last_exc = exc
                wait = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "[%s] %s on attempt %d/%d — retrying in %.1fs",
                    self.name,
                    exc.__class__.__name__,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)

        raise last_exc  # type: ignore[misc]


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO)

    provider = CCXTProvider(exchange_name="binance")
    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    print(f"Fetching BTC/USDT daily data from {start_date} to {end_date} ...")
    df = provider.fetch_ohlcv("BTC/USDT", start=start_date, end=end_date, interval="1d")

    print(f"\nProvider : {provider.name}")
    print(f"Shape    : {df.shape}")
    print(f"Index    : {df.index.name} ({df.index.dtype})")
    print(f"Columns  : {list(df.columns)}")
    print(f"Symbol   : {df.attrs.get('symbol')}")
    print(f"\n{df.head(10)}")
    print(f"\n{df.tail(5)}")
