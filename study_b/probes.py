"""
MechGuard Study B — Activation Probes (Watch)

Trains linear probes on NARCBench pre-extracted residual stream activations
to detect covert coordination in multi-agent LLM systems.

Scientific grounding:
- Rose et al. 2026 — NARCBench, cross-agent collusion detection
- Apollo Research / Goldowsky-Dill et al. ICML 2025 — deception directions
- NARCBench activations are used as the empirical benchmark for B001
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ProbeConfig:
    dataset_id: str = os.getenv("STUDY_B_DATASET", "aaronrose227/narcbench")
    output_dir: str = "./results/study_b"

    # Activation layers available in the configured NARCBench model.
    # B001 uses Qwen3-32B layer 30 as the primary evaluation layer.
    activation_layers: list = None

    # Probe config
    probe_C: float = 1.0            # logistic regression regularization
    max_iter: int = 1000
    random_state: int = 42

    def __post_init__(self):
        if self.activation_layers is None:
            layer_str = os.getenv("STUDY_B_ACTIVATION_LAYERS", "32,33,34,35,36,37")
            self.activation_layers = [int(x) for x in layer_str.split(",")]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_narcbench(config: ProbeConfig) -> dict:
    """
    Load NARCBench dataset from HuggingFace.
    Returns dict with Core, Transfer, Stego splits.

    Dataset structure:
    - NARCBench pre-extracted multi-agent activation data.
    - B001 uses the Qwen3-32B activation release.
    - Exact scenario/run counts are reported in the persisted B001 results.
    """
    print(f"Loading NARCBench from {config.dataset_id}...")
    try:
        dataset = load_dataset(config.dataset_id)
        print(f"Available splits: {list(dataset.keys())}")
        return dataset
    except Exception as e:
        raise RuntimeError(
            f"Could not load NARCBench from {config.dataset_id}. "
            "B001 requires the real benchmark data; synthetic fallback "
            "data is intentionally disabled."
        ) from e



# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_features(scenario: dict, layer: int = 32) -> tuple:
    """
    Extract per-agent activation features for a scenario.
    Returns (features, labels) for probe training.
    """
    features = []
    labels = []

    agents = scenario.get("agents", [])
    for agent in agents:
        # Get activation at specified layer
        if isinstance(agent.get("activation"), list):
            act = np.array(agent["activation"])
        elif isinstance(agent.get("activations"), dict):
            act = np.array(agent["activations"].get(str(layer), agent["activations"].get(layer, [])))
        else:
            continue

        if len(act) == 0:
            continue

        features.append(act)
        labels.append(1 if agent.get("is_colluder", False) else 0)

    return np.array(features), np.array(labels)


# ── Mass-Mean Deception Direction ─────────────────────────────────────────────

def compute_mass_mean_direction(
    features_colluding: np.ndarray,
    features_honest: np.ndarray
) -> np.ndarray:
    """
    Mass-mean deception direction: w = μ_dec - μ_hon
    More robust than full logistic regression, less overfit.
    The "deception direction" in activation space.
    """
    mu_dec = np.mean(features_colluding, axis=0)
    mu_hon = np.mean(features_honest, axis=0)
    direction = mu_dec - mu_hon
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction


# ── Linear Probe Training ─────────────────────────────────────────────────────

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ProbeConfig
) -> tuple:
    """
    Train logistic regression probe on residual stream activations.

    Why linear: a linear probe succeeding proves the model linearly
    represents the deception feature — MLP probe could learn the task
    itself, confounding the interpretability claim.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    probe = LogisticRegression(
        C=config.probe_C,
        max_iter=config.max_iter,
        random_state=config.random_state,
        class_weight="balanced"
    )
    probe.fit(X_scaled, y_train)

    # Cross-validation score
    cv_scores = cross_val_score(probe, X_scaled, y_train, cv=5, scoring="roc_auc")

    return probe, scaler, cv_scores


# ── Main Probe Pipeline ───────────────────────────────────────────────────────

def run_probes(config: ProbeConfig) -> dict:
    """
    Full probe training and evaluation pipeline.
    Train on NARCBench Core, test on Transfer and Stego.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = load_narcbench(config)

    results = {"layers": {}, "scenarios": []}

    for layer in config.activation_layers:
        print(f"\n[Watch] Training probe at layer {layer}")

        # Extract features from Core tier (training)
        core_scenarios = dataset.get("core", dataset.get("train", []))
        if hasattr(core_scenarios, "__iter__") and not isinstance(core_scenarios, list):
            core_scenarios = list(core_scenarios)

        all_features, all_labels = [], []
        for scenario in core_scenarios:
            if isinstance(scenario, dict):
                feats, labs = extract_features(scenario, layer)
            else:
                feats, labs = extract_features(scenario.__dict__ if hasattr(scenario, '__dict__') else dict(scenario), layer)
            if len(feats) > 0:
                all_features.append(feats)
                all_labels.append(labs)

        if not all_features:
            print(f"  No features extracted for layer {layer}")
            continue

        X_train = np.vstack(all_features)
        y_train = np.concatenate(all_labels)

        print(f"  Training on {len(X_train)} agent activations ({y_train.sum()} colluders)")

        # Compute mass-mean direction
        colluding_feats = X_train[y_train == 1]
        honest_feats = X_train[y_train == 0]
        if len(colluding_feats) == 0 or len(honest_feats) == 0:
            print(f"  Insufficient class examples for layer {layer}")
            continue

        deception_direction = compute_mass_mean_direction(colluding_feats, honest_feats)

        # Train probe
        probe, scaler, cv_scores = train_probe(X_train, y_train, config)

        # In-distribution AUROC
        X_scaled = scaler.transform(X_train)
        y_prob = probe.predict_proba(X_scaled)[:, 1]
        train_auroc = roc_auc_score(y_train, y_prob) if len(np.unique(y_train)) > 1 else 0.0

        print(f"  Train AUROC: {train_auroc:.4f} | CV AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Save probe
        probe_path = output_dir / f"probe_layer_{layer}.pkl"
        with open(probe_path, "wb") as f:
            pickle.dump({"probe": probe, "scaler": scaler, "direction": deception_direction}, f)

        results["layers"][layer] = {
            "train_auroc": float(train_auroc),
            "cv_auroc_mean": float(cv_scores.mean()),
            "cv_auroc_std": float(cv_scores.std()),
            "n_train": int(len(X_train)),
            "n_colluders": int(y_train.sum()),
            "probe_path": str(probe_path)
        }

    # Evaluate on Transfer tier
    transfer_scenarios = dataset.get("transfer", [])
    if hasattr(transfer_scenarios, "__iter__") and not isinstance(transfer_scenarios, list):
        transfer_scenarios = list(transfer_scenarios)

    if transfer_scenarios:
        print(f"\n[Watch] Evaluating on Transfer tier ({len(transfer_scenarios)} scenarios)")
        primary_layer = config.activation_layers[0]

        if primary_layer in results["layers"]:
            probe_path = results["layers"][primary_layer]["probe_path"]
            with open(probe_path, "rb") as f:
                probe_data = pickle.load(f)

            transfer_features, transfer_labels = [], []
            for scenario in transfer_scenarios:
                if isinstance(scenario, dict):
                    feats, labs = extract_features(scenario, primary_layer)
                else:
                    feats, labs = extract_features(dict(scenario), primary_layer)
                if len(feats) > 0:
                    transfer_features.append(feats)
                    transfer_labels.append(labs)

            if transfer_features:
                X_transfer = np.vstack(transfer_features)
                y_transfer = np.concatenate(transfer_labels)
                X_scaled = probe_data["scaler"].transform(X_transfer)
                y_prob = probe_data["probe"].predict_proba(X_scaled)[:, 1]
                transfer_auroc = roc_auc_score(y_transfer, y_prob) if len(np.unique(y_transfer)) > 1 else 0.0
                results["transfer_auroc"] = float(transfer_auroc)
                print(f"  Transfer AUROC: {transfer_auroc:.4f}")

    # Save results
    with open(output_dir / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Watch] Probes saved to {output_dir}")
    return results


if __name__ == "__main__":
    config = ProbeConfig()
    run_probes(config)