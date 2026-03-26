"""Client-specific deterministic feature perturbations."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def apply_feature_shift(
    feature_matrices: List[np.ndarray],
    client_assignments: List[Any],
    shift_type: str,
    shift_intensity: float,
    seed: int,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Apply deterministic client-specific feature transformations."""
    rng = np.random.default_rng(seed)
    transformed = []
    per_client_params = []

    for feats in feature_matrices:
        feats = np.asarray(feats, dtype=float)
        if shift_type == "identity":
            transformed_feats = feats.copy()
            params = {"type": "identity"}
        elif shift_type == "affine":
            scale = float(1 + shift_intensity * rng.normal())
            shift = shift_intensity * rng.normal(size=feats.shape[1])
            transformed_feats = feats * scale + shift
            params = {"type": "affine", "scale": scale, "shift": shift.tolist()}
        elif shift_type == "gaussian_noise":
            noise = shift_intensity * rng.normal(size=feats.shape)
            transformed_feats = feats + noise
            params = {"type": "gaussian_noise", "std": shift_intensity}
        elif shift_type == "masking":
            mask = rng.random(size=feats.shape) > shift_intensity
            transformed_feats = feats * mask
            params = {"type": "masking", "drop_prob": shift_intensity}
        elif shift_type == "sign_perturb":
            flip_mask = rng.random(size=feats.shape) < (shift_intensity / 2)
            transformed_feats = np.where(flip_mask, -feats, feats)
            params = {"type": "sign_perturb", "flip_prob": shift_intensity / 2}
        else:
            raise ValueError(f"Unknown shift_type: {shift_type}")

        transformed.append(transformed_feats)
        per_client_params.append(params)

    metadata = {
        "protocol": "feature_shift",
        "shift_type": shift_type,
        "shift_intensity": shift_intensity,
        "seed": seed,
        "per_client_params": per_client_params,
    }
    return transformed, metadata
