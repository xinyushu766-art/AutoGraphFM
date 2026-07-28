from autogfm.experiments.multiseed_runner import _mean_std_ci95


def test_mean_std_ci95_for_five_values() -> None:
    value_mean, value_std, standard_error, ci_low, ci_high = (
        _mean_std_ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    )

    assert value_mean == 3.0
    assert abs(value_std - 1.5811388300841898) < 1e-12
    assert abs(standard_error - 0.7071067811865476) < 1e-12
    assert ci_low < value_mean < ci_high


def test_single_value_has_degenerate_interval() -> None:
    value_mean, value_std, standard_error, ci_low, ci_high = (
        _mean_std_ci95([7.5])
    )

    assert value_mean == 7.5
    assert value_std == 0.0
    assert standard_error == 0.0
    assert ci_low == 7.5
    assert ci_high == 7.5
