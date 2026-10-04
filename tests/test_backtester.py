"""Tests for backtest.backtester: the allocation math and walk-forward loop."""

from __future__ import annotations

import pytest

from backtest.backtester import Backtester


def test_first_bar_equity_equals_initial_capital(synthetic_ohlcv, backtest_config):
    result = Backtester(backtest_config).run({"TEST": synthetic_ohlcv})
    assert result.equity_curve.iloc[0] == pytest.approx(backtest_config["backtest"]["initial_capital"])


def test_equity_curve_has_no_nans_and_stays_positive(synthetic_ohlcv, backtest_config):
    result = Backtester(backtest_config).run({"TEST": synthetic_ohlcv})
    assert not result.equity_curve.isna().any()
    assert (result.equity_curve > 0).all()


def test_runs_multiple_walk_forward_windows(synthetic_ohlcv, backtest_config):
    # ~550 valid feature rows / (train 120 + test 60) with step 60 should
    # produce several windows, each contributing test_window bars.
    result = Backtester(backtest_config).run({"TEST": synthetic_ohlcv})
    test_window = backtest_config["backtest"]["test_window"]
    assert len(result.equity_curve) >= test_window * 3


def test_rebalances_respect_threshold(synthetic_ohlcv, backtest_config):
    result = Backtester(backtest_config).run({"TEST": synthetic_ohlcv})
    threshold = backtest_config["strategy"]["rebalance_threshold"]
    if result.trades.empty:
        pytest.skip("no rebalances occurred for this synthetic series")
    drift = (result.trades["new_allocation"] - result.trades["old_allocation"]).abs()
    assert (drift > threshold - 1e-9).all()


def test_fill_happens_exactly_one_bar_after_decision(synthetic_ohlcv, backtest_config):
    result = Backtester(backtest_config).run({"TEST": synthetic_ohlcv})
    if result.trades.empty:
        pytest.skip("no rebalances occurred for this synthetic series")

    dates = sorted(result.regime_history.loc[result.regime_history["symbol"] == "TEST", "date"].tolist())
    date_pos = {d: i for i, d in enumerate(dates)}

    for _, trade in result.trades.iterrows():
        decision_pos = date_pos[trade["decision_date"]]
        fill_pos = date_pos[trade["fill_date"]]
        assert fill_pos == decision_pos + 1


def test_backtest_is_deterministic(synthetic_ohlcv, backtest_config):
    result_a = Backtester(backtest_config).run({"TEST": synthetic_ohlcv})
    result_b = Backtester(backtest_config).run({"TEST": synthetic_ohlcv})
    pd_testing_equal = (result_a.equity_curve == result_b.equity_curve).all()
    assert pd_testing_equal


def test_raises_with_insufficient_data(synthetic_ohlcv, backtest_config):
    with pytest.raises(ValueError):
        Backtester(backtest_config).run({"TEST": synthetic_ohlcv.iloc[:50]})


def test_multi_symbol_equity_is_sum_of_sleeves(synthetic_ohlcv, backtest_config):
    # A second copy of the same series (as a stand-in second symbol) should
    # produce an identical sleeve, so total equity = 2x one sleeve.
    data = {"A": synthetic_ohlcv, "B": synthetic_ohlcv}
    result = Backtester(backtest_config).run(data)

    assert set(result.per_symbol_equity) == {"A", "B"}
    combined = result.per_symbol_equity["A"].add(result.per_symbol_equity["B"], fill_value=0.0)
    aligned = combined.reindex(result.equity_curve.index)
    assert (aligned - result.equity_curve).abs().max() < 1e-6
