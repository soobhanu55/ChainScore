# ChainScore

Machine-learning credit scoring for DeFi. Processes raw Aave protocol transaction data (deposits, borrows, liquidations) to score wallet creditworthiness — and documents a real train/test leakage bug found and fixed along the way.

**Dataset:** [100k Aave V2 transactions, 3,497 wallets](https://drive.google.com/file/d/19ZsPaQAAKVKKXgswXRl534YoSYCqWkbw/view?usp=sharing)

![Fixed pipeline running on the real dataset, terminal recording](docs/demo.gif)

## The bug that matters

The original notebook trained a `RandomForestRegressor` with `model.fit(X, y)` then evaluated with `model.predict(X)` — same data twice, which always looks artificially strong.

```
Before fix — fit and "evaluate" on the same data:
  R² = 0.9537   ← looks great, but fake

After fix — proper 80/20 split, held-out test set:
  R² = 0.4507
  MAE = 0.42
```

That 0.95 → 0.45 gap is a measured, not hypothetical, illustration of what leakage does to a reported metric.

**Scope note:** 0.45 R² measures how well the model recovers a *hand-designed heuristic score* from engineered features — not real-world default prediction, since there's no real default/liquidation ground truth in this dataset.

## Stack

Python · Pandas · scikit-learn · Matplotlib/Seaborn · Jupyter

## Run it

```bash
# place user-wallet-transactions.json (from the dataset link above) in this folder
python build_and_eval.py
```

Full feature-engineering and scoring-logic details in [`docs/DETAILS.md`](docs/DETAILS.md).
