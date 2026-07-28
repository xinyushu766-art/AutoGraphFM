import numpy as np


def test_real_graph_is_symmetric_nonnegative_and_has_controls(stage3_artifacts: dict[str, object]) -> None:
    directory = stage3_artifacts["graph"].output_dir
    adjacency = np.load(directory / "adjacency.npy")
    identity = np.load(directory / "identity_adjacency.npy")
    permuted = np.load(directory / "permuted_adjacency.npy")
    np.testing.assert_allclose(adjacency, adjacency.T)
    assert np.all(adjacency >= 0)
    assert np.allclose(np.diag(adjacency), 0)
    np.testing.assert_allclose(identity, np.eye(adjacency.shape[0]))
    assert sorted(adjacency.ravel()) == sorted(permuted.ravel())


def test_graph_diagnostics_require_connectivity(stage3_artifacts: dict[str, object]) -> None:
    diagnostics = stage3_artifacts["graph"].manifest["diagnostics"]
    assert diagnostics["isolated_nodes"] == 0
    assert diagnostics["largest_component_ratio"] >= 0.9
    assert diagnostics["correlation_train_differences"] == 47
