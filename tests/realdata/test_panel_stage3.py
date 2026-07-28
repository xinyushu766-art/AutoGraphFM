import numpy as np

from autogfm.realdata import build_window_split, load_panel


def test_panel_shapes_and_sales_feature_contract(stage3_artifacts: dict[str, object]) -> None:
    panel = load_panel(stage3_artifacts["panel"].output_dir)
    assert panel.sales.shape[:2] == panel.features.shape[:2]
    assert panel.features.shape[2] == len(panel.feature_names)
    assert panel.feature_names[0] == "log_sales"
    np.testing.assert_allclose(panel.features[:, :, 0], np.log1p(panel.sales), atol=1e-6)


def test_panel_scaling_uses_train_period_only(stage3_artifacts: dict[str, object]) -> None:
    panel = load_panel(stage3_artifacts["panel"].output_dir)
    train_end = panel.manifest["boundaries"]["train_end_index"]
    index = panel.feature_names.index("log_rolling_mean_3")
    train_values = panel.features[: train_end + 1, :, index]
    assert abs(float(train_values.mean())) < 1e-5
    assert abs(float(train_values.std()) - 1.0) < 1e-4


def test_window_split_assigns_by_target_end(stage3_artifacts: dict[str, object]) -> None:
    panel = load_panel(stage3_artifacts["panel"].output_dir)
    split = build_window_split(panel, history=12, horizon=3)
    train_end = panel.manifest["boundaries"]["train_end_index"]
    validation_end = panel.manifest["boundaries"]["validation_end_index"]
    assert max(origin + 3 for origin in split.train_origins) <= train_end
    assert min(origin + 3 for origin in split.validation_origins) > train_end
    assert max(origin + 3 for origin in split.validation_origins) <= validation_end
    assert min(origin + 3 for origin in split.test_origins) > validation_end
