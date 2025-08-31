# ETF Portfolio Optimization

See Wiki for documentation

A compact toolkit (with a Dash UI) to **build, optimize, backtest, and rebalance** ETF portfolios. It combines a pragmatic mean–variance objective with correlation‑based de‑duplication, clean plots, and an opinionated rebalance table.

> Release: **v1** — Aug 29, 2025

---

## ✨ Features

- **Mean–variance optimizer** (SLSQP) with long‑only or long/short budget (L1)  
- **Correlation clustering** to prune near‑duplicate tickers (threshold on |corr|)  
- **Walk‑forward backtests** with monthly re‑optimization and test returns  
- **Exposure breakdowns** (asset class, sector, bond type, geography, currencies)  
- **Rebalance plan** (buy/sell amounts + before/after allocations)  
- **Dash app** for click‑through workflows and plot previews  
- **Static caching** to avoid repeated downloads while iterating

Modules: `data`, `portfolio`, `opti`, `backtest`, `exposure`, `rebalancer`, `dashboard`, `main`.

---

## 🛠️ Installation

Tested with **Python 3.10+**.

```bash
# (Recommended) Create a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (example)
pip install -U pip
pip install dash pandas numpy matplotlib scipy yfinance
```

Optional: add any other libs you use for data export or rich plotting.

---

## 🚀 Quickstart

### 1) Run the Dash app
```bash
python run.py
# visit http://127.0.0.1:8050  (or whatever the console prints)
```
In the UI you can set **risk**, **base currency**, **shorting**, **cash**, **holdings**, then:
- Create an optimal portfolio  
- View exposures  
- Generate a rebalance table  
- Launch a walk‑forward backtest  
- Try the “crypto Sharpe” helper

> Tip: For reproducible demos or to avoid network calls, the app and `Data` support a **static** mode that reads/writes CSV caches.

### 2) Minimal programmatic example
```python
from portfolio import Portfolio
from opti import Opti

# Build the portfolio universe & objective
p = Portfolio(
    risk=2,                     # 1: low, 2: medium, 3: high
    currency="USD",
    holdings={"VTI": 5000, "BND": 5000},  # optional current positions (in base CCY)
    cash=10000,                 # optional available cash
    allow_short=False           # long-only if False
)

# Optimize
o = Opti(p)
o.optimize()
print("Sparse optimum:", o.optimum)       # dict[ticker -> weight]
o.get_cumulative()                         # compute in-sample cumulative perf

# Plots (as Dash-ready <img> components, also saved under graphs/)
img_alloc = o.plot_optimum()
img_perf  = o.plot_in_sample()
img_dd    = o.plot_drawdown()
img_attr  = o.plot_weighted_perf()
```

### 3) Walk‑forward backtest
```python
from backtest import Backtest

bt = Backtest(o)
bt.parse_data()        # monthly re-optimization over the test window
bt.smoothen_weights()  # optional: simple smoothing (2/3 prev + 1/3 curr)
bt.get_returns()       # fills bt.returns (test) and bt.returns_decomp
img_eq    = bt.plot_backtest()
img_wts   = bt.plot_weights()
img_attr2 = bt.plot_perf_attrib()
```

### 4) Rebalance plan
```python
from rebalancer import Rebalancer

rb = Rebalancer(o)
rb.get_difference()
rb.get_df()
print(rb.rebalance_df)   # columns: ['Ticker','ETF','Buy/Sell','Before','After']
```

### 5) Exposure breakdowns
```python
from exposure import Exposure

ex = Exposure(o)
img_asset = ex.plot_category()   # asset class
img_geo   = ex.plot_geo()        # geography
img_ccy   = ex.plot_currency()   # trading currencies (incl. FX pseudo-tickers)
```

---

## 🧠 How it works (high level)

- **Data** (`data.Data`) loads NAV/close series, FX, ^IRX risk‑free (monthly), simple/log/excess returns, and a VTI proxy for benchmarking. A small **crypto tangency** helper is included.
- **Portfolio** (`portfolio.Portfolio`) trims too‑new tickers and runs **hierarchical clustering** on distance `1 - |corr|` (average linkage). For each cluster, it keeps the member with the lowest objective. It exposes `objective(w=...) = weight_cov * variance - mean_excess`. The risk‑aversion `weight_cov` is derived from the discrete **risk** (1..3).
- **Opti** builds **bounds** (`(0,1)` long‑only or `(-1,1)` long/short) and an **equality constraint** (sum(w)=1 or sum(|w|)=1), then solves with **SciPy SLSQP**. Very small absolute weights (<1%) are zeroed and the vector is renormalized by L1.
- **Backtest** performs a **train/test split** (default **0.85** in‑sample), re‑optimizes monthly in a walk‑forward loop (`static=True` + `backtest=<timestamp>` to truncate to in‑sample), then computes **out‑of‑sample returns** and plots.
- **Exposure** aggregates optimized weights by category (asset class, sector, bond type, geography, trading currency).
- **Rebalancer** converts weights × liquidity into **target currency amounts**, diffs vs holdings, and returns a tidy **rebalance table**.

---

## 📁 Project layout (suggested)

```
.
├─ code/
│  ├─ data.py
│  ├─ portfolio.py
│  ├─ opti.py
│  ├─ backtest.py
│  ├─ exposure.py
│  ├─ rebalancer.py
│  ├─ dashboard.py
│  └─ run.py                   # entry point for the Dash app
├─ graphs/                     # saved PNGs
├─ data_cache/                 # CSV caches when static=True
├─ docs/                       # Sphinx docs
└─ README.md
```

---

## 📚 Documentation (Sphinx)

Build the HTML docs locally:
```bash
cd docs
make html
# open build/html/index.html
```
Tips:
- Ensure your modules are included in a `toctree` (add or fix `modules.rst` as needed).
- Prefer reStructuredText or Numpy-style docstrings consistently.
- For duplicate object warnings, use `:no-index:` on one of the duplicates only when appropriate.

---

## ⚙️ Configuration & caching

- **Base currency**: choose among `Data.possible_currencies` (e.g., `USD`, `EUR`, `SGD`, …).  
- **Static mode**: pass `static=True` to reuse CSVs under your cache directory (define in your `Data` / project settings).  
- **Graphs**: saved under `graphs/<currency>/`. The Dash APIs also return inline `<img>` components for embedding.

---

## 🧪 Testing (lightweight ideas)

- Determinism of `get_color_map()` given a fixed universe  
- Invariance of the equality constraint after zeroing tiny weights  
- Backtest splits (index at the 85% cutoff)  
- Rebalance table sums to total liquidity (within rounding)  

---

## ❓ Troubleshooting

- **Rate limiting / network**: use `static=True` to iterate on cached data.  
- **No update on button click** (Dash): check that the callback’s `Output` matches a displayed component and that you reset the triggering `n_clicks` when needed.  
- **Empty exposure chart**: verify that the optimized tickers have exposure rows and non‑zero weights.  
- **Weird allocations**: review your **risk** level and `threshold_correlation` (defaults to `0.95`).

---

## 🗺️ Roadmap (ideas)

- Optional robust covariance / shrinkage  
- Risk‑parity or max‑diversification objectives  
- Transaction‑cost & turnover penalties  
- Multi‑currency reporting (base vs native)  
- Richer backtest analytics (rolling stats, heatmaps)

---

## 📄 License

TBD. Add your preferred license (MIT/Apache‑2.0/BSD‑3‑Clause, etc.).

---

## Acknowledgements

Yahoo Finance data via `yfinance`. Dash + Matplotlib for UI and plots. SciPy for optimization.

---

## Reference

The README summarizes and complements the Sphinx project documentation. See the generated API pages for class/method details.
