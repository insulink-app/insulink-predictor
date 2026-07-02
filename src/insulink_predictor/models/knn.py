"""k-NN (analog forecasting) baseline — a distance-based counterpoint to LGBM.

For a query feature vector, k-NN averages the *delta-over-persistence* of its
nearest past analogs: "when things looked like this before (this level, rising
at this rate, this long after a meal, this time of day), glucose did X next".

Unlike LightGBM, distance-based learning needs help that trees get for free:
- **Imputation** — there is no defined distance to a NaN. LGBM eats NaNs
  natively; here every gap/absent-channel value is median-filled first.
- **Scaling** — Euclidean distance on raw mixed-unit features (mg/dL vs steps
  vs sin/cos) is meaningless, so features are standardized.
- **Feature weighting** — plain k-NN weights every feature equally, so irrelevant
  channels (weather, day-of-week) dilute the metric. The ``importances`` option
  weights the distance by LGBM's gain, i.e. a *supervised* metric that recovers
  the feature selection trees do implicitly.

One model **per horizon** (direct multi-horizon, like ``lgbm.py``), the same
delta target, and the same ``pred_fn(df, h) -> np.ndarray`` interface, so the
harness scores it identically to every other model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Config


class _ColumnWeighter:
    """Scale standardized columns by ``sqrt(weight)``.

    Euclidean distance then becomes ``sum_j weight_j · (x_j − x'_j)²`` — features
    LGBM found informative pull the neighbor search; near-useless ones fade out.
    """

    def __init__(self, weights: np.ndarray) -> None:
        self._w = np.sqrt(np.asarray(weights, dtype=float))

    def fit(self, X, y=None) -> "_ColumnWeighter":
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * self._w


def _make_pipeline(k: int, weights: np.ndarray | None) -> Pipeline:
    steps: list = [
        # keep_empty_features: an all-NaN optional channel stays a (zeroed) column
        # so the feature count never shifts between train and predict.
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]
    if weights is not None:
        steps.append(("weight", _ColumnWeighter(weights)))
    steps.append(
        # distance-weighted so near analogs count more than the k-th; deterministic.
        ("knn", KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=1))
    )
    return Pipeline(steps)


def _weight_vector(
    importances: pd.Series | None, feature_cols: list[str]
) -> np.ndarray | None:
    """Align LGBM gain-importances to ``feature_cols``, normalized to mean 1."""
    if importances is None:
        return None
    w = importances.reindex(feature_cols).fillna(0.0).to_numpy(dtype=float)
    mean = w.mean()
    return w / mean if mean > 0 else None


def train_knn(
    train: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
    k: int = 100,
    importances: pd.Series | None = None,
) -> dict[int, Pipeline]:
    """Train one k-NN pipeline per horizon on the valid training rows.

    With ``cfg.features.predict_delta`` the target is the change over persistence
    (y_{t+h} − g_t); persistence is added back at inference. Pass ``importances``
    (LGBM gain-percent per feature) for the supervised/importance-weighted metric.
    """
    delta = cfg.features.predict_delta
    weights = _weight_vector(importances, feature_cols)
    models: dict[int, Pipeline] = {}
    for h in cfg.horizons_steps:
        mask = train[f"valid_{h}"].to_numpy()
        X = train.loc[mask, feature_cols]
        y = train.loc[mask, f"y_{h}"]
        if delta:
            y = y - train.loc[mask, "glucose_mgdl"]
        kh = min(k, max(1, len(X) - 1))  # never ask for more neighbors than rows
        model = _make_pipeline(kh, weights)
        model.fit(X, y.to_numpy())
        models[h] = model
    return models


def make_knn_pred_fn(
    models: dict[int, Pipeline],
    feature_cols: list[str],
    predict_delta: bool = True,
):
    """Wrap trained k-NN models into a harness ``pred_fn(df, h)`` (absolute mg/dL)."""

    def pred_fn(df: pd.DataFrame, horizon_steps: int) -> np.ndarray:
        pred = models[horizon_steps].predict(df[feature_cols])
        if predict_delta:
            pred = pred + df["glucose_mgdl"].to_numpy()
        return pred

    return pred_fn


# --- multi-output trajectory forecast (Phase 3 curve, k-NN flavour) ----------
#
# ``gf curve`` forecasts the whole path t+5…t+60 (one model per step). The point
# forecast plugs straight into ``forecast_curve`` (it only needs ``.predict``).
# The extra win is uncertainty: LGBM needs a separately-trained quantile model
# per level per step; k-NN reads the band straight off the neighbor set — one
# model yields the full predictive distribution.


class KNNCurveStep:
    """A fitted per-step k-NN: point ``.predict`` + empirical neighbor quantiles.

    Stores the training target-deltas alongside the pipeline so the band comes
    from the *same* fitted neighbors (no per-quantile retraining), using only the
    public ``kneighbors`` API.
    """

    def __init__(self, pipeline: Pipeline, y_train: np.ndarray) -> None:
        self.pipeline = pipeline
        self._y = np.asarray(y_train, dtype=float)

    def predict(self, X) -> np.ndarray:
        return self.pipeline.predict(X)

    def neighbor_quantiles(self, X, quantiles) -> dict[float, np.ndarray]:
        """Per-row empirical quantiles of the k neighbors' target-deltas."""
        knn: KNeighborsRegressor = self.pipeline.named_steps["knn"]
        xt = self.pipeline[:-1].transform(X)  # impute→scale→(weight)
        _, idx = knn.kneighbors(xt)
        neigh = self._y[idx]  # (n_rows, k)
        return {q: np.quantile(neigh, q, axis=1) for q in quantiles}


def train_knn_curve_models(
    train: pd.DataFrame,
    feature_cols: list[str],
    max_step: int,
    cfg: Config,
    k: int = 100,
    importances: pd.Series | None = None,
) -> dict[int, KNNCurveStep]:
    """One k-NN per step ``1..max_step`` on valid rows (delta target).

    Mirrors ``events.train_curve_models`` but with k-NN, so the result drops into
    ``forecast_curve`` unchanged (each step exposes ``.predict``).
    """
    delta = cfg.features.predict_delta
    weights = _weight_vector(importances, feature_cols)
    models: dict[int, KNNCurveStep] = {}
    for step in range(1, max_step + 1):
        mask = train[f"cvalid_{step}"].to_numpy()
        X = train.loc[mask, feature_cols]
        y = train.loc[mask, f"cy_{step}"]
        if delta:
            y = y - train.loc[mask, "glucose_mgdl"]
        y = y.to_numpy()
        kh = min(k, max(1, len(X) - 1))
        pipe = _make_pipeline(kh, weights).fit(X, y)
        models[step] = KNNCurveStep(pipe, y)
    return models


class _KNNQuantileStep:
    """Adapts one quantile level of a :class:`KNNCurveStep` to the ``.predict``
    interface ``forecast_curve`` expects, so the harness band code is reused as-is."""

    def __init__(self, step: KNNCurveStep, q: float) -> None:
        self._step = step
        self._q = q

    def predict(self, X) -> np.ndarray:
        return self._step.neighbor_quantiles(X, [self._q])[self._q]


def quantile_knn_curve_models(
    curve_models: dict[int, KNNCurveStep], quantiles
) -> dict[float, dict[int, _KNNQuantileStep]]:
    """Reuse the *same* fitted per-step k-NN for every quantile → {q: {step: model}}.

    Same shape ``forecast_curve``/``_curve_examples`` already consume for LGBM
    quantile models — but here no quantile model is trained; the band is empirical.
    """
    return {
        q: {step: _KNNQuantileStep(m, q) for step, m in curve_models.items()}
        for q in quantiles
    }
