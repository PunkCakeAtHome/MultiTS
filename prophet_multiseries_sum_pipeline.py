"""
prophet_multiseries_sum_pipeline.py

Пайплайн Prophet для 1000+ рядов с целью:
- максимально точно предсказывать SUM(amount) на горизонте X дней
- корректная walk-forward CV без утечек
- фиксированные weekly+yearly сезонности, multiplicative
- тюним только CPS и SPS

Вход:
df: columns [group_id, date, amount]
date: datetime-like, amount: numeric

Выход:
1) forecasts_df: [group_id, date, amount_pred] на X дней вперёд
2) comparison_df: [group_id, past_x_days_sum, future_x_days_sum, diff, ratio, best_cps, best_sps, cv_loss]
3) backtest_df: метрики backtest (train=всё кроме последних X, valid=последние X) + список худших
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Параллелизм (опционально)
try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None

# Prophet (обязательно)
try:
    from prophet import Prophet
except Exception as e:
    Prophet = None
    _PROPHET_IMPORT_ERROR = e


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class PipelineConfig:
    horizon_days: int = 30
    n_folds: int = 3                   # walk-forward folds, каждый по horizon_days
    min_total_days: int = 365          # минимальная длина ряда (дней) для нормальной работы
    min_train_days: int = 180          # минимальная длина train внутри фолда
    fill_missing_with: float = 0.0     # как заполнять пропуски дат

    # fixed per your request
    seasonality_mode: str = "multiplicative"
    weekly_seasonality: bool = True
    yearly_seasonality: bool = True
    daily_seasonality: bool = False

    # tuning grids
    cps_grid: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.1, 0.2)
    sps_grid: Tuple[float, ...] = (5.0, 10.0, 20.0)

    # evaluation
    use_relative_loss: bool = True     # loss = abs(sum_pred-sum_fact)/sum_fact
    clip_negative: bool = True

    # speed
    n_jobs: int = 1                    # >1 требует joblib
    prophet_fit_kwargs: Optional[Dict] = None  # например {"iter": 1000} (обычно не нужно)


# -----------------------------
# Utilities
# -----------------------------

def _require_prophet():
    if Prophet is None:
        raise ImportError(
            "Пакет 'prophet' не установлен или не импортируется. "
            "Установи: pip install prophet. "
            f"Техническая причина: {_PROPHET_IMPORT_ERROR!r}"
        )


def _prepare_daily_series(g: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Приводит один group к непрерывной дневной сетке.
    Пропуски дат заполняются cfg.fill_missing_with (по умолчанию 0).
    """
    gg = g.copy()
    gg["date"] = pd.to_datetime(gg["date"])
    gg = gg.sort_values("date")
    gg["amount"] = pd.to_numeric(gg["amount"], errors="coerce").fillna(0.0)

    full_idx = pd.date_range(gg["date"].min(), gg["date"].max(), freq="D")
    out = (
        gg.set_index("date")[["amount"]]
        .reindex(full_idx)
        .rename_axis("date")
        .reset_index()
    )
    out["amount"] = out["amount"].fillna(cfg.fill_missing_with)
    return out


def _make_folds(n: int, horizon: int, n_folds: int, min_train: int) -> List[Tuple[int, int, int]]:
    """
    Возвращает список фолдов (train_end, valid_start, valid_end) индексами по iloc.
    Каждый valid = horizon дней.
    Фолды идут по последним окнам:
    при n_folds=3 и horizon=30:
      valid: [-90:-60], [-60:-30], [-30:0]
    train: всё до valid_start
    """
    folds: List[Tuple[int, int, int]] = []

    for k in range(n_folds, 0, -1):
        valid_end = n - horizon * (k - 1)
        valid_start = valid_end - horizon
        train_end = valid_start

        if train_end < min_train:
            continue
        if valid_start < 0:
            continue

        folds.append((train_end, valid_start, valid_end))

    return folds


def _sum_loss(pred_sum: float, fact_sum: float, relative: bool) -> float:
    err = abs(pred_sum - fact_sum)
    if not relative:
        return float(err)
    denom = max(1e-9, abs(fact_sum))
    return float(err / denom)


def _fit_and_forecast(
    train_df: pd.DataFrame,
    horizon: int,
    cfg: PipelineConfig,
    cps: float,
    sps: float,
) -> pd.DataFrame:
    """
    train_df columns: ds, y
    return forecast df columns: date, yhat
    """
    _require_prophet()

    model = Prophet(
        seasonality_mode=cfg.seasonality_mode,
        weekly_seasonality=cfg.weekly_seasonality,
        yearly_seasonality=cfg.yearly_seasonality,
        daily_seasonality=cfg.daily_seasonality,
        changepoint_prior_scale=float(cps),
        seasonality_prior_scale=float(sps),
    )

    fit_kwargs = cfg.prophet_fit_kwargs or {}
    model.fit(train_df, **fit_kwargs)

    future = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
    fc = model.predict(future)[["ds", "yhat"]].rename(columns={"ds": "date"})

    if cfg.clip_negative:
        fc["yhat"] = fc["yhat"].clip(lower=0.0)

    return fc


# -----------------------------
# Tuning / Backtest / Forecast per group
# -----------------------------

def tune_group_on_sum(
    g_daily: pd.DataFrame,
    cfg: PipelineConfig,
) -> Optional[Dict[str, float]]:
    """
    Walk-forward CV на сумме horizon_days.
    Возвращает best cps/sps и cv_loss.
    """
    horizon = cfg.horizon_days
    n = len(g_daily)

    if n < cfg.min_total_days:
        return None

    folds = _make_folds(n, horizon, cfg.n_folds, cfg.min_train_days)
    if not folds:
        return None

    best = None  # tuple(loss, cps, sps)

    for cps, sps in product(cfg.cps_grid, cfg.sps_grid):
        losses = []
        ok = True

        for train_end, valid_start, valid_end in folds:
            train = g_daily.iloc[:train_end]
            valid = g_daily.iloc[valid_start:valid_end]

            train_df = train.rename(columns={"date": "ds", "amount": "y"})[["ds", "y"]]

            try:
                fc = _fit_and_forecast(train_df, horizon, cfg, cps, sps)
            except Exception:
                ok = False
                break

            pred_sum = float(fc["yhat"].sum())
            fact_sum = float(valid["amount"].sum())
            losses.append(_sum_loss(pred_sum, fact_sum, cfg.use_relative_loss))

        if not ok:
            continue

        loss = float(np.mean(losses)) if losses else np.inf
        cand = (loss, float(cps), float(sps))
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        return None

    return {"cv_loss": best[0], "best_cps": best[1], "best_sps": best[2]}


def backtest_last_horizon(
    g_daily: pd.DataFrame,
    cfg: PipelineConfig,
    cps: float,
    sps: float,
) -> Optional[Dict]:
    """
    Backtest:
      train = всё кроме последних X
      valid = последние X
    Метрики только по сумме.
    """
    horizon = cfg.horizon_days
    n = len(g_daily)
    if n < horizon + cfg.min_train_days:
        return None

    train = g_daily.iloc[:-horizon]
    valid = g_daily.iloc[-horizon:]

    train_df = train.rename(columns={"date": "ds", "amount": "y"})[["ds", "y"]]

    try:
        fc = _fit_and_forecast(train_df, horizon, cfg, cps, sps)
    except Exception:
        return None

    pred_sum = float(fc["yhat"].sum())
    fact_sum = float(valid["amount"].sum())

    abs_error = abs(pred_sum - fact_sum)
    rel_error = abs_error / max(1e-9, abs(fact_sum))

    return {
        "fact_sum": fact_sum,
        "pred_sum": pred_sum,
        "abs_error": abs_error,
        "rel_error": rel_error,
    }


def final_forecast_and_compare(
    group_id,
    g_daily: pd.DataFrame,
    cfg: PipelineConfig,
    cps: float,
    sps: float,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Финальный прогноз на horizon дней + сравнение сумм:
      past = sum(last horizon факта)
      future = sum(horizon прогноза)
    """
    horizon = cfg.horizon_days

    train_df = g_daily.rename(columns={"date": "ds", "amount": "y"})[["ds", "y"]]
    fc = _fit_and_forecast(train_df, horizon, cfg, cps, sps)

    forecasts_df = fc.rename(columns={"yhat": "amount_pred"}).copy()
    forecasts_df.insert(0, "group_id", group_id)

    past_sum = float(g_daily.iloc[-horizon:]["amount"].sum()) if len(g_daily) >= horizon else float(g_daily["amount"].sum())
    future_sum = float(forecasts_df["amount_pred"].sum())

    summary = {
        "group_id": group_id,
        "past_x_days_sum": past_sum,
        "future_x_days_sum": future_sum,
        "diff_future_minus_past": future_sum - past_sum,
        "ratio_future_over_past": future_sum / max(1e-9, past_sum),
        "best_cps": float(cps),
        "best_sps": float(sps),
    }

    return forecasts_df, summary


# -----------------------------
# Multi-series pipeline
# -----------------------------

def _process_one_group(group_id, g: pd.DataFrame, cfg: PipelineConfig):
    """
    Возвращает (forecast_df, summary_row, backtest_row)
    """
    g_daily = _prepare_daily_series(g[["date", "amount"]], cfg)

    tuned = tune_group_on_sum(g_daily, cfg)
    if tuned is None:
        return None

    cps = tuned["best_cps"]
    sps = tuned["best_sps"]

    bt = backtest_last_horizon(g_daily, cfg, cps, sps)
    # bt может быть None если ряд короткий — тогда просто пропускаем backtest

    fc_df, summary = final_forecast_and_compare(group_id, g_daily, cfg, cps, sps)
    summary["cv_loss"] = tuned["cv_loss"]

    if bt is not None:
        bt_row = {"group_id": group_id, "cps": cps, "sps": sps, **bt, "cv_loss": tuned["cv_loss"]}
    else:
        bt_row = None

    return fc_df, summary, bt_row


class ProphetSumPipeline:
    """
    Главный класс.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def run(self, df: pd.DataFrame):
        """
        Возвращает:
          forecasts_df: group_id, date, amount_pred
          comparison_df: group_id + сравнения сумм
          backtest_df: group_id + ошибки backtest (может быть пустым)
        """
        _require_prophet()

        required = {"group_id", "date", "amount"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"В df не хватает колонок: {missing}. Нужно: {required}")

        groups = list(df.groupby("group_id", sort=False))

        results = []
        if self.cfg.n_jobs == 1 or Parallel is None:
            for gid, g in groups:
                r = _process_one_group(gid, g, self.cfg)
                if r is not None:
                    results.append(r)
        else:
            # Параллелим по group_id
            results = Parallel(n_jobs=self.cfg.n_jobs)(
                delayed(_process_one_group)(gid, g, self.cfg) for gid, g in groups
            )
            results = [r for r in results if r is not None]

        if not results:
            return (
                pd.DataFrame(columns=["group_id", "date", "amount_pred"]),
                pd.DataFrame(columns=[
                    "group_id", "past_x_days_sum", "future_x_days_sum",
                    "diff_future_minus_past", "ratio_future_over_past",
                    "best_cps", "best_sps", "cv_loss"
                ]),
                pd.DataFrame(columns=[
                    "group_id", "fact_sum", "pred_sum", "abs_error", "rel_error", "cps", "sps", "cv_loss"
                ]),
            )

        forecasts_df = pd.concat([r[0] for r in results], ignore_index=True)
        comparison_df = pd.DataFrame([r[1] for r in results])

        bt_rows = [r[2] for r in results if r[2] is not None]
        backtest_df = pd.DataFrame(bt_rows) if bt_rows else pd.DataFrame(columns=[
            "group_id", "fact_sum", "pred_sum", "abs_error", "rel_error", "cps", "sps", "cv_loss"
        ])

        # Удобные сортировки:
        comparison_df = comparison_df.sort_values("diff_future_minus_past", ascending=False).reset_index(drop=True)
        if not backtest_df.empty:
            backtest_df = backtest_df.sort_values("rel_error", ascending=False).reset_index(drop=True)

        return forecasts_df, comparison_df, backtest_df

    @staticmethod
    def worst_groups(backtest_df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
        if backtest_df is None or backtest_df.empty:
            return pd.DataFrame()
        cols = ["group_id", "fact_sum", "pred_sum", "abs_error", "rel_error", "cps", "sps", "cv_loss"]
        cols = [c for c in cols if c in backtest_df.columns]
        return backtest_df.head(top_k)[cols]


# -----------------------------
# Minimal synthetic generator for testing (optional)
# -----------------------------

def generate_synthetic_data(
    n_groups: int = 50,
    start_date: str = "2022-01-01",
    n_days: int = 365 * 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Синтетика: рост + multiplicative weekly/yearly + шум.
    Нужна только чтобы быстро протестировать пайплайн без реальных данных.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    rows = []
    for gi in range(n_groups):
        base = rng.uniform(50, 500)
        annual_growth = rng.uniform(0.15, 0.6)
        daily_growth = (1 + annual_growth) ** (1 / 365)

        weekly_amp = rng.uniform(0.05, 0.2)
        yearly_amp = rng.uniform(0.1, 0.35)

        for i, d in enumerate(dates):
            trend = base * (daily_growth ** i)
            weekly = 1 + weekly_amp * np.sin(2 * np.pi * (d.dayofweek / 7))
            yearly = 1 + yearly_amp * np.sin(2 * np.pi * (d.dayofyear / 365))
            noise = rng.normal(1.0, 0.12)

            y = trend * weekly * yearly * noise
            y = max(0.0, float(y))

            rows.append({"group_id": f"group_{gi}", "date": d, "amount": y})

    return pd.DataFrame(rows)
    

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Быстрый тест на синтетике
    _require_prophet()

    df = generate_synthetic_data(n_groups=30, n_days=365*4)

    cfg = PipelineConfig(
        horizon_days=30,
        n_folds=3,
        cps_grid=(0.01, 0.03, 0.05, 0.1, 0.2),
        sps_grid=(5.0, 10.0, 20.0),
        n_jobs=1,
    )

    pipe = ProphetSumPipeline(cfg)
    forecasts_df, comparison_df, backtest_df = pipe.run(df)

    print("forecasts_df:", forecasts_df.shape)
    print("comparison_df:", comparison_df.shape)
    print("backtest_df:", backtest_df.shape)

    print("\nWorst groups (backtest):")
    print(ProphetSumPipeline.worst_groups(backtest_df, top_k=10).to_string(index=False))

    print("\nTop diff:")
    print(comparison_df.head(3)[["group_id","past_x_days_sum","future_x_days_sum","diff_future_minus_past","ratio_future_over_past","best_cps","best_sps","cv_loss"]].to_string(index=False))

    print("\nBottom diff:")
    print(comparison_df.tail(3)[["group_id","past_x_days_sum","future_x_days_sum","diff_future_minus_past","ratio_future_over_past","best_cps","best_sps","cv_loss"]].to_string(index=False))
