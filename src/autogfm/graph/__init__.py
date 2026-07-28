"""Graph construction and controlled variants."""

from autogfm.graph.ops import normalize_adjacency
from autogfm.graph.recovery import (
    GraphRecoveryResult,
    GraphRecoverySpec,
    run_graph_recovery_benchmark,
)
from autogfm.graph.real_graph import GraphBuildResult, GraphSpec, build_relation_graph
from autogfm.graph.sparse_residual import DirectSparseResidualGraphLearner
from autogfm.graph.stability_recovery import (
    StabilityRecoveryResult,
    StabilityRecoverySpec,
    run_stability_recovery_benchmark,
)
from autogfm.graph.stability_selection import (
    StabilitySelectionDiagnostics,
    consensus_adjacency,
    edge_frequency,
    lagged_ridge_scores,
    select_candidate_mask,
    top_k_edge_mask,
)
from autogfm.graph.variants import make_identity_graph, make_permuted_graph

__all__ = [
    "GraphBuildResult",
    "GraphRecoveryResult",
    "GraphRecoverySpec",
    "GraphSpec",
    "StabilityRecoveryResult",
    "StabilityRecoverySpec",
    "StabilitySelectionDiagnostics",
    "DirectSparseResidualGraphLearner",
    "build_relation_graph",
    "consensus_adjacency",
    "edge_frequency",
    "lagged_ridge_scores",
    "make_identity_graph",
    "make_permuted_graph",
    "normalize_adjacency",
    "run_graph_recovery_benchmark",
    "run_stability_recovery_benchmark",
    "select_candidate_mask",
    "top_k_edge_mask",
]
