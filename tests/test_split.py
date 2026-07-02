"""Phase 1 — splits are chronological / by-user, never random (§6 guardrail)."""

from __future__ import annotations

import ast
import inspect

import glucose_forecast.eval.split as split_mod
from glucose_forecast.eval.split import chronological_split, heldout_user_split


def test_chronological_train_precedes_test_per_user(grid):
    train, test = chronological_split(grid, test_fraction=0.2)
    for uid in grid["user_id"].unique():
        tr = train[train["user_id"] == uid]["ts_utc"]
        te = test[test["user_id"] == uid]["ts_utc"]
        assert tr.max() < te.min()  # no temporal overlap


def test_split_fraction_sizes(grid):
    train, test = chronological_split(grid, test_fraction=0.2)
    for uid in grid["user_id"].unique():
        n = (grid["user_id"] == uid).sum()
        n_test = (test["user_id"] == uid).sum()
        assert abs(n_test - round(n * 0.2)) <= 1


def test_heldout_users_disjoint(grid):
    heldout = [grid["user_id"].unique()[0]]
    seen, held = heldout_user_split(grid, heldout)
    assert set(seen["user_id"]).isdisjoint(held["user_id"])
    assert set(held["user_id"]) == set(heldout)


def test_no_randomness_in_split_source():
    """The split module must not shuffle/sample — autocorrelation would leak.

    AST-based (not substring) so prose like "never random" in docs is fine; only
    actual randomization imports/calls are forbidden.
    """
    tree = ast.parse(inspect.getsource(split_mod))
    forbidden_calls = {
        "sample", "shuffle", "permutation", "choice", "randint", "rand", "randn",
        "train_test_split",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name.split(".")[0] != "random" for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "random"
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_calls, f"randomization call: .{node.attr}"
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_calls, f"randomization call: {node.id}"


def test_invalid_fraction_rejected(grid):
    import pytest

    with pytest.raises(ValueError):
        chronological_split(grid, test_fraction=1.5)
