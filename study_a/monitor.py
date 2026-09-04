"""
MechGuard Study A — Weight Geometry Monitor (Attest)

Training-time mechanistic monitoring for fine-tuning.

This module measures:
    1. LoRA parameter updates (ΔW)
    2. Singular-value / subspace geometry
    3. Principal angles and subspace overlap
    4. WeightWatcher spectral diagnostics
    5. Commutator defect from paired gradient updates

IMPORTANT SCIENTIFIC SCOPE
--------------------------
This monitor does NOT by itself establish emergent misalignment.

The geometric measurements are observables. Whether they predict
emergent misalignment must be established experimentally.

In particular:
    - similarity to a CLEAN reference is not itself suspicious;
    - commutator defect is a training-dynamics quantity and requires
      two independent gradient evaluations;
    - literature thresholds are anchors, not validated MechGuard
      decision thresholds.

References:
    - Soligo et al. 2025 — convergent representations associated
      with emergent misalignment.
    - Xu 2026 — commutator defect as an early-warning signal for
      grokking/generalization. This is NOT evidence that the same
      signal predicts emergent misalignment.
    - Martin & Mahoney / WeightWatcher — spectral diagnostics.
"""

import gc
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
try:
    import weightwatcher as ww
except ImportError:
    ww = None
from dotenv import load_dotenv
from peft import PeftModel
from scipy.linalg import subspace_angles
from tqdm import tqdm
from transformers import AutoModelForCausalLM


load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MonitorConfig:
    """Configuration for Study A monitoring."""

    # ------------------------------------------------------------------
    # Model / paths
    # ------------------------------------------------------------------

    base_model_id: str = os.getenv(
        "STUDY_A_MODEL",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    )

    checkpoint_dir: str = os.getenv(
        "STUDY_A_CHECKPOINT_DIR",
        "./checkpoints/study_a",
    )

    output_dir: str = "./results/study_a"

    # Optional reference adapter.
    #
    # IMPORTANT:
    # The meaning is controlled by reference_type.
    #
    # "clean":
    #     geometric similarity to this reference is NOT a suspicion
    #     signal. It can instead be used as a baseline/control.
    #
    # "em":
    #     this adapter represents an experimentally identified EM
    #     reference direction. Similarity can then be treated as a
    #     candidate mechanistic signal.
    reference_adapter_path: Optional[str] = None

    reference_type: str = "clean"

    # ------------------------------------------------------------------
    # SVD
    # ------------------------------------------------------------------

    top_k_singular: int = 32

    svd_seed: int = 42

    target_layers: List[int] = field(
        default_factory=lambda: [15, 16]
    )

    # ------------------------------------------------------------------
    # Candidate literature anchors
    #
    # These are NOT validated MechGuard thresholds.
    # ------------------------------------------------------------------

    angle_anchor_deg: float = 30.0
    overlap_anchor: float = 0.60

    # ------------------------------------------------------------------
    # WeightWatcher
    # ------------------------------------------------------------------

    run_weightwatcher: bool = True

    # WeightWatcher operates on model weights. It should therefore be
    # interpreted separately from LoRA ΔW geometry.
    weightwatcher_on_adapter_model: bool = True


# ============================================================================
# Utility functions
# ============================================================================

def _cleanup_cuda() -> None:
    """Release cached CUDA memory where available."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_float(value) -> Optional[float]:
    """Convert numeric values to float while handling invalid values."""
    try:
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def _layer_matches(name: str, target_layers: Iterable[int]) -> bool:
    """
    Match transformer layer indices robustly.

    Examples:
        model.layers.15.self_attn.q_proj
        base_model.model.model.layers.16.mlp.down_proj
    """

    for layer_idx in target_layers:
        tokens = name.replace(".", " ").split()

        if str(layer_idx) in tokens:
            return True

    return False


# ============================================================================
# ΔW extraction
# ============================================================================

def extract_delta_w(
    adapter_model: PeftModel,
    layer_names: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract effective LoRA ΔW matrices.

    Prefer PEFT's get_delta_weight() when available because it correctly
    incorporates LoRA scaling and fan-in/fan-out conventions.

    Falls back to B @ A * scaling when necessary.

    Returns:
        layer_name -> effective ΔW tensor
    """

    delta_ws: Dict[str, torch.Tensor] = {}

    for name, module in adapter_model.named_modules():

        if not hasattr(module, "lora_A"):
            continue

        if not hasattr(module, "lora_B"):
            continue

        if layer_names and not _layer_matches(name, layer_names):
            continue

        try:
            # Preferred PEFT implementation.
            if hasattr(module, "get_delta_weight"):
                delta_w = module.get_delta_weight("default")
                delta_w = delta_w.detach().float()

            else:
                A = module.lora_A["default"].weight.detach().float()
                B = module.lora_B["default"].weight.detach().float()

                scaling = 1.0

                if hasattr(module, "scaling"):
                    scaling = float(
                        module.scaling.get("default", 1.0)
                    )

                delta_w = (B @ A) * scaling

            delta_ws[name] = delta_w.cpu()

        except Exception as exc:
            print(f"  [ΔW skip] {name}: {exc}")

    return delta_ws


# ============================================================================
# Deterministic randomized SVD
# ============================================================================

def randomized_svd_top_k(
    matrix: torch.Tensor,
    k: int = 32,
    seed: int = 42,
    oversample: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deterministic randomized truncated SVD.

    Returns:
        U, S, Vt

    U:
        left singular vectors

    S:
        singular values

    Vt:
        right singular vectors

    Notes
    -----
    A fresh random projection was previously generated on every call.
    That made repeated measurements nondeterministic.

    We now use an explicit seed.
    """

    matrix = matrix.detach().float()

    if matrix.ndim != 2:
        raise ValueError(
            f"Expected 2D matrix, got shape {tuple(matrix.shape)}"
        )

    m, n = matrix.shape

    rank_limit = min(m, n)

    if rank_limit == 0:
        raise ValueError("Cannot perform SVD on an empty matrix.")

    k = min(int(k), rank_limit)

    # Oversampling improves approximation quality.
    sketch_k = min(k + int(oversample), rank_limit)

    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(int(seed))

    omega = torch.randn(
        n,
        sketch_k,
        generator=generator,
        device=matrix.device,
        dtype=matrix.dtype,
    )

    Y = matrix @ omega

    Q, _ = torch.linalg.qr(Y, mode="reduced")

    B = Q.T @ matrix

    U_hat, S, Vt = torch.linalg.svd(
        B,
        full_matrices=False,
    )

    U = Q @ U_hat

    return (
        U[:, :k],
        S[:k],
        Vt[:k, :],
    )


# ============================================================================
# Principal angles / subspace geometry
# ============================================================================

def compute_principal_angles(
    U1: torch.Tensor,
    U2: torch.Tensor,
) -> np.ndarray:
    """
    Compute principal angles between two column subspaces.

    Returns angles in degrees.

    Interpretation:
        0°  -> identical subspaces
        90° -> orthogonal subspaces

    IMPORTANT:
    A small angle only means geometric similarity. It does not,
    by itself, imply harmfulness or misalignment.
    """

    u1 = U1.detach().cpu().numpy()
    u2 = U2.detach().cpu().numpy()

    angles_rad = subspace_angles(u1, u2)

    return np.degrees(angles_rad)


def compute_subspace_overlap(
    U1: torch.Tensor,
    U2: torch.Tensor,
) -> float:
    """
    Compute normalized mean squared cosine of principal angles.

    Range:
        approximately [0, 1]

    Interpretation:
        1 -> highly aligned subspaces
        0 -> approximately orthogonal

    This is a geometry statistic, not an alignment/safety classifier.
    """

    angles_deg = compute_principal_angles(U1, U2)

    angles_rad = np.radians(angles_deg)

    overlap = float(
        np.mean(np.cos(angles_rad) ** 2)
    )

    return float(np.clip(overlap, 0.0, 1.0))


# ============================================================================
# Matrix geometry utilities
# ============================================================================

def frobenius_distance(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
) -> float:
    """Normalized Frobenius distance between two matrices."""

    a = matrix_a.detach().float()
    b = matrix_b.detach().float()

    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
        )

    denominator = torch.linalg.norm(b)

    if denominator.item() == 0:
        return float(torch.linalg.norm(a).item())

    return float(
        (torch.linalg.norm(a - b) / denominator).item()
    )


def cosine_similarity(
    vector_a: torch.Tensor,
    vector_b: torch.Tensor,
) -> float:
    """Cosine similarity between flattened tensors."""

    a = vector_a.detach().float().reshape(-1)
    b = vector_b.detach().float().reshape(-1)

    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
        )

    denominator = (
        torch.linalg.norm(a) *
        torch.linalg.norm(b)
    )

    if denominator.item() == 0:
        return 0.0

    return float(
        torch.dot(a, b).item() / denominator.item()
    )


# ============================================================================
# Commutator defect
# ============================================================================

def compute_commutator_defect(
    theta_0: torch.Tensor,
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    grad_b_after_a: torch.Tensor,
    grad_a_after_b: torch.Tensor,
    learning_rate: float,
) -> float:
    """
    Compute the normalized gradient-update commutator defect.

    The construction compares:

        A then B

    against:

        B then A

    using first-order gradient updates.

    Given:

        θ_AB = θ0 - η g_A(θ0)
                    - η g_B(θ0 - η g_A(θ0))

        θ_BA = θ0 - η g_B(θ0)
                    - η g_A(θ0 - η g_B(θ0))

    the defect is:

        ||θ_AB - θ_BA|| / (η² ||g_A|| ||g_B|| + eps)

    This is a training-dynamics quantity.

    It CANNOT be recovered from a single checkpoint's weights alone.

    Parameters
    ----------
    theta_0:
        Flattened parameters at the starting point.

    grad_a:
        Gradient on mini-batch A at θ0.

    grad_b:
        Gradient on mini-batch B at θ0.

    grad_b_after_a:
        Gradient on B after applying A's update.

    grad_a_after_b:
        Gradient on A after applying B's update.

    learning_rate:
        Gradient update step size.

    Returns
    -------
    float
        Normalized commutator defect.
    """

    theta_0 = theta_0.detach().float()
    grad_a = grad_a.detach().float()
    grad_b = grad_b.detach().float()
    grad_b_after_a = grad_b_after_a.detach().float()
    grad_a_after_b = grad_a_after_b.detach().float()

    eta = float(learning_rate)

    if eta <= 0:
        raise ValueError("learning_rate must be positive.")

    theta_ab = (
        theta_0
        - eta * grad_a
        - eta * grad_b_after_a
    )

    theta_ba = (
        theta_0
        - eta * grad_b
        - eta * grad_a_after_b
    )

    numerator = torch.linalg.norm(
        theta_ab - theta_ba
    )

    denominator = (
        (eta ** 2)
        * torch.linalg.norm(grad_a)
        * torch.linalg.norm(grad_b)
    )

    denominator = denominator + 1e-12

    defect = numerator / denominator

    return float(defect.item())


def gradient_vector(
    gradients: Iterable[Optional[torch.Tensor]],
) -> torch.Tensor:
    """
    Flatten and concatenate gradients into one parameter-space vector.

    Useful for constructing the inputs required by
    compute_commutator_defect().
    """

    vectors = []

    for grad in gradients:
        if grad is None:
            continue

        vectors.append(
            grad.detach().float().reshape(-1)
        )

    if not vectors:
        raise ValueError(
            "No gradients were provided."
        )

    return torch.cat(vectors)


# ============================================================================
# WeightWatcher
# ============================================================================

def run_weightwatcher(
    model: torch.nn.Module,
) -> dict:
    """
    Run WeightWatcher ESD analysis.

    IMPORTANT:
    These metrics describe model-weight spectral properties.

    They should NOT be interpreted as direct evidence of emergent
    misalignment without an experimentally validated relationship.
    """
    if ww is None:
        return {
            "available": False,
            "error": (
                "WeightWatcher is not installed in the current "
                "environment. Install it with: "
                "pip install weightwatcher"
            ),
        }
    try:
        watcher = ww.WeightWatcher(
            model=model
        )

        details = watcher.analyze(
            plot=False
        )

        summary = watcher.get_summary(
            details
        )

        layer_metrics = {}

        for _, row in details.iterrows():

            layer_name = str(
                row.get("longname", row.name)
            )

            layer_metrics[layer_name] = {
                "alpha": _safe_float(
                    row.get("alpha")
                ),
                "alpha_weighted": _safe_float(
                    row.get("alpha_weighted")
                ),
                "stable_rank": _safe_float(
                    row.get("stable_rank")
                ),
                "log_spectral_norm": _safe_float(
                    row.get("log_spectral_norm")
                ),
                "mp_softrank": _safe_float(
                    row.get("mp_softrank")
                ),
            }

        numeric_summary = {}

        for key, value in summary.items():

            converted = _safe_float(value)

            if converted is not None:
                numeric_summary[str(key)] = converted

        return {
            "summary": numeric_summary,
            "layers": layer_metrics,
        }

    except Exception as exc:
        print(
            f"  [WeightWatcher error] {exc}"
        )

        return {
            "error": str(exc)
        }


# ============================================================================
# Model loading
# ============================================================================

def load_adapter_model(
    checkpoint_path: str,
    config: MonitorConfig,
) -> PeftModel:
    """Load base model plus LoRA adapter."""

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    return PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
    )


# ============================================================================
# Checkpoint analysis
# ============================================================================

def analyze_checkpoint(
    checkpoint_path: str,
    reference_delta_ws: Optional[Dict[str, torch.Tensor]],
    config: MonitorConfig,
    step: int,
) -> dict:
    """
    Analyze one checkpoint.

    Measurements:
        - ΔW
        - singular values
        - principal-angle geometry
        - subspace overlap
        - distance from reference
        - WeightWatcher metrics

    No commutator defect is fabricated here.

    Commutator defect requires paired gradient evaluations during
    training and must be supplied by an experiment runner.
    """

    print(
        f"\n[Attest] Step {step} — {checkpoint_path}"
    )

    result = {
        "step": int(step),
        "checkpoint": str(checkpoint_path),
        "reference_type": config.reference_type,
        "layers": {},
        "flagged": False,
    }

    try:
        adapter_model = load_adapter_model(
            checkpoint_path,
            config,
        )

    except Exception as exc:

        print(
            f"  [load error] {exc}"
        )

        result["error"] = str(exc)

        return result

    # ---------------------------------------------------------------
    # Extract ΔW
    # ---------------------------------------------------------------

    checkpoint_delta_ws = extract_delta_w(
        adapter_model,
        layer_names=config.target_layers,
    )

    print(
        f"  Extracted ΔW for "
        f"{len(checkpoint_delta_ws)} target layers"
    )

    # ---------------------------------------------------------------
    # Analyze each layer
    # ---------------------------------------------------------------

    for layer_name, delta_w in checkpoint_delta_ws.items():

        layer_result = {
            "matrix_shape": list(delta_w.shape),
        }

        # -----------------------------------------------------------
        # SVD
        # -----------------------------------------------------------

        U_ckpt, S_ckpt, _ = randomized_svd_top_k(
            delta_w,
            k=config.top_k_singular,
            seed=config.svd_seed,
        )

        layer_result["singular_values"] = [
            float(x)
            for x in S_ckpt.cpu().numpy()
        ]

        if len(S_ckpt) > 0:
            layer_result["top_singular_value"] = float(
                S_ckpt[0].item()
            )

        if len(S_ckpt) > 1:
            layer_result["spectral_gap_sigma1_sigma2"] = float(
                S_ckpt[0].item() - S_ckpt[1].item()
            )

            layer_result["spectral_ratio_sigma1_sigma2"] = float(
                S_ckpt[0].item()
                /
                (S_ckpt[1].item() + 1e-12)
            )

        # -----------------------------------------------------------
        # Reference geometry
        # -----------------------------------------------------------

        if (
            reference_delta_ws is not None
            and layer_name in reference_delta_ws
        ):

            reference_delta_w = (
                reference_delta_ws[layer_name]
            )

            U_ref, _, _ = randomized_svd_top_k(
                reference_delta_w,
                k=config.top_k_singular,
                seed=config.svd_seed,
            )

            angles = compute_principal_angles(
                U_ckpt,
                U_ref,
            )

            overlap = compute_subspace_overlap(
                U_ckpt,
                U_ref,
            )

            top_n = min(
                5,
                len(angles),
            )

            mean_angle = float(
                np.mean(angles[:top_n])
            )

            layer_result.update(
                {
                    "mean_principal_angle_deg": mean_angle,
                    "principal_angles_deg": [
                        float(x) for x in angles
                    ],
                    "subspace_overlap": float(overlap),
                    "reference_frobenius_distance": (
                        frobenius_distance(
                            delta_w,
                            reference_delta_w,
                        )
                    ),
                    "reference_cosine_similarity": (
                        cosine_similarity(
                            delta_w,
                            reference_delta_w,
                        )
                    ),
                }
            )

            # -------------------------------------------------------
            # IMPORTANT:
            #
            # A clean reference is a CONTROL.
            #
            # Similarity to clean is NOT automatically suspicious.
            # -------------------------------------------------------

            if config.reference_type == "em":

                candidate_flag = (
                    mean_angle
                    <= config.angle_anchor_deg
                    or
                    overlap
                    >= config.overlap_anchor
                )

                layer_result[
                    "candidate_em_reference_match"
                ] = bool(candidate_flag)

                if candidate_flag:
                    result["flagged"] = True

                    print(
                        f"  [candidate EM match] "
                        f"{layer_name}: "
                        f"angle={mean_angle:.2f}°, "
                        f"overlap={overlap:.3f}"
                    )

            elif config.reference_type == "clean":

                layer_result[
                    "clean_reference_comparison_only"
                ] = True

                # NO suspicion flag is generated from clean similarity.

        result["layers"][layer_name] = layer_result

    # ---------------------------------------------------------------
    # WeightWatcher
    # ---------------------------------------------------------------

    if config.run_weightwatcher:

        print(
            "  Running WeightWatcher..."
        )

        result["weightwatcher"] = (
            run_weightwatcher(
                adapter_model
            )
        )

    # ---------------------------------------------------------------
    # Explicit scientific metadata
    # ---------------------------------------------------------------

    result["scientific_notes"] = {
        "commutator_defect": (
            "not computed from checkpoint; "
            "requires paired gradient evaluations"
        ),
        "reference_similarity": (
            "geometry only; not equivalent to misalignment"
        ),
        "thresholds": (
            "literature anchors, not validated "
            "MechGuard decision thresholds"
        ),
    }

    del adapter_model

    _cleanup_cuda()

    return result


# ============================================================================
# Checkpoint discovery
# ============================================================================

def discover_checkpoints(
    checkpoint_dir: str,
) -> List[Path]:
    """Discover and numerically sort checkpoint directories."""

    directory = Path(
        checkpoint_dir
    )

    if not directory.exists():
        return []

    checkpoints = [
        path
        for path in directory.iterdir()
        if (
            path.is_dir()
            and "checkpoint" in path.name
        )
    ]

    def checkpoint_step(path: Path) -> int:

        suffix = path.name.split("-")[-1]

        if suffix.isdigit():
            return int(suffix)

        return 0

    return sorted(
        checkpoints,
        key=checkpoint_step,
    )


# ============================================================================
# Reference loading
# ============================================================================

def load_reference_delta_ws(
    config: MonitorConfig,
    first_checkpoint: Optional[Path] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load the explicitly configured reference adapter.

    If no reference adapter is supplied, returns None.

    We intentionally DO NOT silently use the first checkpoint as a
    'reference'. That behavior can make interpretation ambiguous and
    can accidentally turn temporal self-comparison into a safety flag.
    """

    if not config.reference_adapter_path:

        print(
            "No explicit reference adapter provided."
        )

        return None

    print(
        f"Loading {config.reference_type} reference "
        f"from {config.reference_adapter_path}"
    )

    try:

        model = load_adapter_model(
            config.reference_adapter_path,
            config,
        )

        reference_delta_ws = extract_delta_w(
            model,
            layer_names=config.target_layers,
        )

        del model

        _cleanup_cuda()

        print(
            f"Reference ΔW extracted for "
            f"{len(reference_delta_ws)} layers"
        )

        return reference_delta_ws

    except Exception as exc:

        print(
            f"[reference load error] {exc}"
        )

        return None


# ============================================================================
# Main monitoring loop
# ============================================================================

def run_monitor(
    config: MonitorConfig,
):
    """
    Scan checkpoints and save geometric measurements.

    This function performs observation/measurement.

    It does not claim that any measurement is validated as an
    emergent-misalignment detector.
    """

    output_dir = Path(
        config.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoints = discover_checkpoints(
        config.checkpoint_dir
    )

    if not checkpoints:

        print(
            f"No checkpoints found in "
            f"{config.checkpoint_dir}"
        )

        return []

    print(
        f"Found {len(checkpoints)} checkpoints"
    )

    # ---------------------------------------------------------------
    # Load explicit reference
    # ---------------------------------------------------------------

    reference_delta_ws = (
        load_reference_delta_ws(
            config,
            first_checkpoint=checkpoints[0],
        )
    )

    # ---------------------------------------------------------------
    # Record configuration
    # ---------------------------------------------------------------

    config_path = (
        output_dir /
        "monitor_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            asdict(config),
            handle,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Analyze checkpoints
    # ---------------------------------------------------------------

    all_results = []

    for checkpoint in tqdm(
        checkpoints,
        desc="Analyzing checkpoints",
    ):

        suffix = checkpoint.name.split("-")[-1]

        step = (
            int(suffix)
            if suffix.isdigit()
            else 0
        )

        result = analyze_checkpoint(
            str(checkpoint),
            reference_delta_ws,
            config,
            step,
        )

        all_results.append(result)

        # Save incrementally.
        with open(
            output_dir /
            "monitor_results.json",
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                all_results,
                handle,
                indent=2,
            )

    print(
        "\n[Attest] Monitoring complete."
    )

    print(
        f"Results: "
        f"{output_dir / 'monitor_results.json'}"
    )

    candidate_steps = [
        result["step"]
        for result in all_results
        if result.get("flagged", False)
    ]

    if candidate_steps:

        print(
            "[Attest] Candidate reference-match "
            f"steps: {candidate_steps}"
        )

    else:

        print(
            "[Attest] No candidate reference matches."
        )

    return all_results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":

    config = MonitorConfig()

    run_monitor(config)