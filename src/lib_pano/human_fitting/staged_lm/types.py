"""Data structures for staged rigid / pose / joint LM fitting.

Stage 1 performs dual-camera rigid orient/transl alignment. Stage 2 performs
pose warmup followed by pose/shape refinement.

The original script passed information between functions through loosely-typed
dicts with abbreviated keys (``"r2d"``, ``"n_valid"``, ``"lhand_c2"`` ...).
This module replaces them with small dataclasses that make the data flow
explicit while preserving the exact same field values that the JSON / NPZ
outputs depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch

from .config import ObjectiveConfig, ScalingConstants, SolverConfig

if TYPE_CHECKING:
    from .linearization import Linearization


@dataclass
class ValidPointSet:
    """Valid SAM3D dense correspondences for one camera in one frame.

    Attributes
    ----------
    vertex_ids:
        Indices into the SMPL-X vertex array that have a corresponding
        observation.  Shape ``[N]``, dtype int64.
    target_2d:
        Observed 2D target positions on the ERP image.  Shape ``[N, 2]``.
    """

    vertex_ids: torch.Tensor
    target_2d: torch.Tensor

    @property
    def count(self) -> int:
        """Number of valid correspondences."""
        return int(self.vertex_ids.shape[0])


@dataclass
class FrameValidPoints:
    """Valid points for both cameras of a single frame."""

    cam2: ValidPointSet
    cam4: ValidPointSet


@dataclass
class FrozenFrameParameters:
    """Parameters that are held constant during optimisation for one frame.

    Hand poses and per-camera scale come from the HMR initialisation and are
    not part of the optimisation vector, so they are frozen here to avoid
    accidental gradient flow.
    """

    cam2_left_hand_pose: torch.Tensor
    cam2_right_hand_pose: torch.Tensor
    cam4_left_hand_pose: torch.Tensor
    cam4_right_hand_pose: torch.Tensor
    cam2_scale: torch.Tensor
    cam4_scale: torch.Tensor


@dataclass
class ReprojectionGroup:
    """Raw 2D reprojection residuals for one (frame, camera) pair.

    Attributes
    ----------
    frame_index:
        Original frame index from the input list.
    camera:
        Which camera produced the residuals.
    residual_2d:
        2D residual tensor of shape ``[N, 2]`` where *N* is the number of
        valid correspondences for this (frame, camera) pair.
    """

    frame_index: int
    camera: Literal["cam2", "cam4"]
    residual_2d: torch.Tensor

    @property
    def valid_point_count(self) -> int:
        """Number of valid points contributing to this residual."""
        return int(self.residual_2d.shape[0])


LinearSolveBackend = Literal[
    "cholesky",
    "dense_solve",
    "schur",
]


@dataclass(frozen=True)
class LinearSolveResult:
    """Named outcome of one damped normal-equation solve attempt."""

    delta: torch.Tensor | None
    backend: LinearSolveBackend
    schur_failed: bool = False

    @property
    def success(self) -> bool:
        return self.delta is not None


@dataclass
class LMSolverCounters:
    """Counters accumulated during a single LM optimisation run.

    Keep only distinct solver events.
    """

    accepted_steps: int = 0
    rejected_trials: int = 0
    linearization_count: int = 0
    solve_attempt_count: int = 0
    solve_success_count: int = 0
    solve_failure_count: int = 0

    def record_linear_solve(self, result: LinearSolveResult) -> None:
        """Record one mutually exclusive overall solve result."""
        self.solve_attempt_count += 1
        if result.success:
            self.solve_success_count += 1
        else:
            self.solve_failure_count += 1


@dataclass
class LMResult:
    """Outcome of one Levenberg-Marquardt optimization."""

    label: str
    parameters: torch.Tensor
    objective: float
    status: str
    counters: LMSolverCounters
    final_lambda: float
    wall_time_ms: float
    rms_metrics: dict[str, float]
    weight_history: list[dict[str, Any]]
    aggregation_spec: dict[str, Any]
    jacobian_backend: str = "autograd"
    linearization_backend: str = "jacobian"
    eval_backend: str = "full_forward"
    linear_solver_backend: str = "dense"
    trial_history: list[dict[str, Any]] = field(default_factory=list)
    profile_timing: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable view (without the parameter tensor)."""
        result: dict[str, Any] = {
            "label": self.label,
            "obj": self.objective,
            "status": self.status,
            "linearization_count": self.counters.linearization_count,
            "solve_attempt_count": self.counters.solve_attempt_count,
            "solve_success_count": self.counters.solve_success_count,
            "solve_failure_count": self.counters.solve_failure_count,
            "accepted_steps": self.counters.accepted_steps,
            "rejected_trials": self.counters.rejected_trials,
            "final_lambda": self.final_lambda,
            "rms": self.rms_metrics,
            "w_history": self.weight_history,
            "aggregation_spec": self.aggregation_spec,
            "jacobian_backend": self.jacobian_backend,
            "linearization_backend": self.linearization_backend,
            "eval_backend": self.eval_backend,
            "linear_solver_backend": self.linear_solver_backend,
            "trial_history": self.trial_history,
        }
        if self.profile_timing is not None:
            result["wall_time_ms"] = self.wall_time_ms
            result["profile_timing"] = self.profile_timing
        return result


@dataclass
class NormalEquations:
    """Residual and Gauss-Newton system produced without a global Jacobian."""

    residual: torch.Tensor
    hessian: torch.Tensor | None
    gradient: torch.Tensor
    structured_system: Any | None = None


@dataclass(frozen=True)
class StageInputSummary:
    """Static dimensions derived from observations and parameter metadata."""

    frame_count: int
    parameter_dimension: int
    group_count: int
    total_valid_points: int
    reprojection_residual_dimension: int
    anchor_residual_dimension: int
    surrogate_residual_dimension: int

    def to_json_dict(self) -> dict[str, int]:
        """Return all summary fields as JSON-compatible integers."""
        return {
            "frame_count": self.frame_count,
            "parameter_dimension": self.parameter_dimension,
            "group_count": self.group_count,
            "total_valid_points": self.total_valid_points,
            "reprojection_residual_dimension": self.reprojection_residual_dimension,
            "anchor_residual_dimension": self.anchor_residual_dimension,
            "surrogate_residual_dimension": self.surrogate_residual_dimension,
        }


@dataclass
class StagePreparationTiming:
    """Non-overlapping handoff, cache setup, and definition-binding times."""

    handoff_ms: float = 0.0
    cache_setup_ms: float = 0.0
    definition_bind_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.handoff_ms + self.cache_setup_ms + self.definition_bind_ms


@dataclass
class StageDefinition:
    """Declarative description of one optimisation stage.

    Bundles the functions and hyper-parameters that ``run_stage`` needs so the
    orchestration code can treat all three stages uniformly.

    """

    name: str
    parameter_dimension: int
    initial_lm_parameters: torch.Tensor
    raw_groups_fn: Callable[[torch.Tensor], list[ReprojectionGroup]]
    anchor_residual_fn: Callable[[torch.Tensor], torch.Tensor] | None
    raw_metrics_fn: Callable[[torch.Tensor], dict[str, float]]
    lm_max_accepted_steps: int
    analytic_cache: Any = None
    linearization_builder: Callable[
        [torch.Tensor, dict[str, Any] | None],
        "Linearization",
    ] | None = None
    lm_true_objective_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    # The following labels are output/profile provenance, not runtime selectors.
    eval_backend: str = "full_forward"
    linearization_backend: str = "autograd"
    linear_solver_backend: str = "dense"
    aggregation_spec: dict[str, Any] | None = None
    input_summary: StageInputSummary | None = None
    preparation_timing: StagePreparationTiming = field(
        default_factory=StagePreparationTiming,
    )
    objective_terms_fn: Callable[[torch.Tensor], dict[str, torch.Tensor]] | None = None


@dataclass
class StageOutcome:
    """Result of running a single production LM stage."""

    lm_result: LMResult
    true_objective_fn: Callable[[torch.Tensor], torch.Tensor]
    definition: StageDefinition | None = None


@dataclass
class Stage2Outcome:
    """Results and handoffs from the two internal steps of Stage 2."""

    pose_warmup: StageOutcome
    pose_shape_refinement: StageOutcome | None
    handoffs: dict[str, Any]

@dataclass
class StageContext:
    """Shared, read-mostly state needed to build and run every stage.

    Captures everything that the per-stage builder functions need access to,
    avoiding the proliferation of closure variables that the original
    ``main()`` relied on.
    """

    args: Any
    device: torch.device
    dtype: torch.dtype
    device_environment: dict[str, Any]
    context_timing: dict[str, float]
    frame_indices: list[int]
    image_width: int
    image_height: int
    frame_count: int
    is_irls: bool
    charbonnier_eps: float
    body_pose_dim: int
    beta_dim: int
    body_pose_anchors: list[torch.Tensor]
    beta_anchor: torch.Tensor
    beta_latent0: torch.Tensor
    scaling: ScalingConstants
    solver_config: SolverConfig
    objective_config: ObjectiveConfig
    frozen_params_by_frame: list[FrozenFrameParameters]
    cam2_vertex_ids_by_frame: list[torch.Tensor]
    cam2_targets_by_frame: list[torch.Tensor]
    cam4_vertex_ids_by_frame: list[torch.Tensor]
    cam4_targets_by_frame: list[torch.Tensor]
    model: Any
    fitter: Any
    cam2_smplx_parameters: Any = None
    cam4_smplx_parameters: Any = None
