"""Learning-augmented heavy-hitter oracle for adaptive support selection.

The oracle predicts which held-out Fourier coefficients carry the most
information about the signal, given the already-observed (support) ones.
It turns the deterministic support-target partition used in Phases 1-5 of
the DynaDiff-VLBI benchmark into an adaptive one.

Formal guarantees are in theory/oracle_bound.tex (Theorem 2).
"""

from dynadiff_vlbi.oracle.data_adapter import (
    DETERMINISTIC_STRATEGIES,
    OracleDataAdapterConfig,
    VisibilityDatasetOracleAdapter,
)
from dynadiff_vlbi.oracle.heavy_hitter_oracle import (
    HeavyHitterOracle,
    HeavyHitterOracleConfig,
    load_oracle_from_checkpoint,
)
from dynadiff_vlbi.oracle.teacher import (
    compute_importance_teacher,
    compute_importance_teacher_batched,
    compute_posterior_covariance,
)
from dynadiff_vlbi.oracle.training import (
    OracleTrainingConfig,
    distill_oracle_step,
    set_oracle_seed,
    train_oracle,
)

__all__ = [
    "HeavyHitterOracle",
    "HeavyHitterOracleConfig",
    "load_oracle_from_checkpoint",
    "compute_importance_teacher",
    "compute_importance_teacher_batched",
    "compute_posterior_covariance",
    "OracleTrainingConfig",
    "distill_oracle_step",
    "set_oracle_seed",
    "train_oracle",
    "DETERMINISTIC_STRATEGIES",
    "OracleDataAdapterConfig",
    "VisibilityDatasetOracleAdapter",
]
