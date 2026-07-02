"""Splits (ROADMAP §Phase 1 + §6 guardrail: **never random**).

Glucose is highly autocorrelated, so a random split leaks the future into the
past. Every split here is chronological (early = train, late = test) or by
held-out user (cold-start). There is deliberately no RNG in this module.
"""

from __future__ import annotations

import pandas as pd


def chronological_split(
    df: pd.DataFrame, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per user, the earliest ``1 − test_fraction`` is train, the rest is test.

    Train always precedes test in time *within each user* — no shuffling.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")

    train_parts, test_parts = [], []
    for _, g in df.sort_values(["user_id", "ts_utc"]).groupby("user_id", sort=True):
        cut = int(round(len(g) * (1.0 - test_fraction)))
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, test


def heldout_user_split(
    df: pd.DataFrame, heldout_users: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by user id for cold-start tests: held-out users never appear in train."""
    mask = df["user_id"].isin(heldout_users)
    seen = df[~mask].reset_index(drop=True)
    heldout = df[mask].reset_index(drop=True)
    return seen, heldout
