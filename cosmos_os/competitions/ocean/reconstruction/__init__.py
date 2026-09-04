"""Ocean Temperature Vertical Reconstruction Module (Problem 2)."""

from __future__ import annotations

from .adapter import OceanReconstructionAdapter
from .features import VerticalProfileFeaturePlugin, compute_residual_target
from .schema import validate_reconstruction_schema
from .submission import generate_reconstruction_submission
from .validation import (
    build_competition_simulation_validation,
    build_random_mask_validation,
    build_secondary_blocked_validation,
    calculate_baseline_linear_interpolation_rmse,
    calculate_reconstruction_metrics,
)

__all__ = [
    "OceanReconstructionAdapter",
    "VerticalProfileFeaturePlugin",
    "compute_residual_target",
    "validate_reconstruction_schema",
    "generate_reconstruction_submission",
    "build_competition_simulation_validation",
    "build_random_mask_validation",
    "build_secondary_blocked_validation",
    "calculate_reconstruction_metrics",
    "calculate_baseline_linear_interpolation_rmse",
]
