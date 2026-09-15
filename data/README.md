# Data Directory

This directory holds raw and processed market data.

## Structure

```
data/
├── raw/          # Original downloaded data (immutable)
├── processed/    # Cleaned, aligned, and validated data
└── README.md
```

## Conventions

1. **Raw data is immutable.**  Never modify files in `raw/`.  All
   transformations produce new files in `processed/`.
2. **Large files are gitignored.**  Use `scripts/download_data.py` to
   reproduce the dataset from scratch.
3. **Adjusted Close** is used as the canonical "close" for equities/ETFs
   to account for splits and dividends.  OHLC prices are adjusted by the
   same ratio for consistency.

## Data Sources

| Ticker | Name | Asset Class | Source | Rows | Date Range |
|--------|------|-------------|--------|------|------------|
| SPY | S&P 500 ETF | Equity | Yahoo Finance | ~2,176 | 2018-01-02 → 2026-08-28 |
| QQQ | Nasdaq 100 ETF | Equity | Yahoo Finance | ~2,176 | 2018-01-02 → 2026-08-28 |
| BTC-USD | Bitcoin | Crypto | Yahoo Finance | ~3,164 | 2018-01-01 → 2026-08-30 |
| ETH-USD | Ethereum | Crypto | Yahoo Finance | ~3,164 | 2018-01-01 → 2026-08-30 |
| GC=F | Gold Futures | Futures | Yahoo Finance | ~2,177 | 2018-01-02 → 2026-08-28 |
| CL=F | Crude Oil Futures | Futures | Yahoo Finance | ~2,178 | 2018-01-02 → 2026-08-28 |

## Data Splits (Calendar-Based)

| Split | Start | End | Purpose |
|-------|-------|-----|---------|
| In-Sample | 2018-01-01 | 2023-03-31 | Signal discovery, parameter selection (~60%) |
| Validation | 2023-04-01 | 2025-01-31 | Strategy validation, parameter stability (~20%) |
| Out-of-Sample | 2025-02-01 | 2026-08-31 | Final evaluation — **NEVER touched until Milestone 10** (~20%) |

## Known Data Limitations

1. **Survivorship bias (equities):** Yahoo Finance only provides data for
   currently listed instruments.  We mitigate this by using ETFs (SPY, QQQ)
   rather than individual stocks.

2. **Adjusted Close recalculation:** Yahoo Finance may retroactively
   recalculate Adjusted Close when new corporate actions occur.  This is
   acceptable for research but means exact numbers may differ slightly
   across download dates.

3. **CL=F negative prices:** WTI Crude Oil futures (CL=F) went negative
   on 20 April 2020 (Close ≈ −$37).  This is a real market event, not a
   data error.  Our validation flagged 4 non-positive price values.
   We retain these as-is because they represent genuine price history.

4. **Futures continuous contracts:** Yahoo Finance's "=F" tickers represent
   a continuous front-month contract series.  Roll methodology is opaque
   and may introduce small artefacts at roll dates.  For a production
   system, we would source individual contract data and construct our own
   continuous series.

5. **Crypto 24/7 trading:** BTC-USD and ETH-USD trade every day (including
   weekends).  Equity/futures data only covers business days.  This
   calendar mismatch is handled in portfolio construction (Milestone 8).

## Reproducibility

```bash
# Re-download all raw data from scratch
python scripts/download_data.py
```

The download script reads `config/config.yaml` for the ticker universe and
date range.  It is idempotent and safe to re-run.
