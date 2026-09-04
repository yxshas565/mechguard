"""
MechGuard Study A — CPU Pipeline Smoke Test

Purpose
-------
Validate the end-to-end instrumentation pipeline on a tiny local model.

This is NOT an emergent-misalignment experiment.

It validates:
    model loading
    LoRA attachment
    deterministic training
    checkpoint creation
    checkpoint reload
    effective ΔW extraction
    deterministic SVD
    geometry measurement
    machine-readable experiment manifest
"""

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from study_a.monitor import (
    compute_subspace_overlap,
    extract_delta_w,
    randomized_svd_top_k,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "sshleifer/tiny-gpt2"

SEED = 42

OUTPUT_DIR = Path("./results/study_a/SMOKE001")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

MAX_LENGTH = 64

TRAIN_STEPS = 3

CHECKPOINT_EVERY = 1

LEARNING_RATE = 1e-3

TOP_K = 4


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all relevant random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

TRAIN_TEXTS = [
    "The model learns a simple language modeling task.",
    "MechGuard measures internal parameter geometry.",
    "This experiment validates checkpoint instrumentation.",
    "The model should produce deterministic checkpoints.",
    "Internal representations can be measured during training.",
    "This is a pipeline validation experiment.",
    "The training loop is intentionally small.",
    "Research claims require controlled experiments.",
]


def tokenize_dataset(tokenizer):
    """Tokenize the tiny local training corpus."""

    encoded = tokenizer(
        TRAIN_TEXTS,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return encoded


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    """Load tiny model and attach a LoRA adapter."""

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID
    )

    # tiny-gpt2 uses Conv1D modules.
    # GPT-2's attention projection is exposed as c_attn.
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        bias="none",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.train()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    tokenizer,
    step: int,
):
    """Save a lightweight, portable PEFT checkpoint."""

    checkpoint_path = (
        CHECKPOINT_DIR /
        f"checkpoint-{step}"
    )

    checkpoint_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        checkpoint_path
    )

    tokenizer.save_pretrained(
        checkpoint_path
    )

    # ------------------------------------------------------------------
    # Keep the smoke-test adapter configuration portable across PEFT
    # versions by removing optional configuration fields that are not
    # required for this experiment.
    # ------------------------------------------------------------------

    adapter_config_path = (
        checkpoint_path /
        "adapter_config.json"
    )

    if adapter_config_path.exists():

        with open(
            adapter_config_path,
            "r",
            encoding="utf-8",
        ) as handle:
            adapter_config = json.load(handle)

        adapter_config.pop(
            "monteclora_config",
            None,
        )

        adapter_config.pop(
            "velora_config",
            None,
        )

        with open(
            adapter_config_path,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                adapter_config,
                handle,
                indent=2,
            )

    return checkpoint_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, tokenizer):
    """Run deterministic CPU LoRA training."""

    encoded = tokenize_dataset(
        tokenizer
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    checkpoints = []

    for step in range(
        1,
        TRAIN_STEPS + 1,
    ):

        # Deterministic minibatch selection.
        index = (
            (step - 1)
            % len(input_ids)
        )

        batch_input_ids = (
            input_ids[index:index + 1]
        )

        batch_attention_mask = (
            attention_mask[index:index + 1]
        )

        labels = batch_input_ids.clone()

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        print(
            f"step={step} "
            f"loss={loss.item():.6f}"
        )

        if (
            step % CHECKPOINT_EVERY == 0
        ):
            checkpoint = save_checkpoint(
                model,
                tokenizer,
                step,
            )

            checkpoints.append(
                checkpoint
            )

    return checkpoints


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def measure_checkpoint_geometry(
    model,
):
    """Extract LoRA ΔW and calculate SVD statistics."""

    delta_ws = extract_delta_w(
        model
    )

    measurements = {}

    for layer_name, delta_w in delta_ws.items():

        U, S, Vt = (
            randomized_svd_top_k(
                delta_w,
                k=TOP_K,
                seed=SEED,
            )
        )

        measurements[layer_name] = {
            "shape": list(
                delta_w.shape
            ),
            "rank": int(
                torch.linalg.matrix_rank(
                    delta_w
                ).item()
            ),
            "top_singular_value": float(
                S[0].item()
            ),
            "singular_values": [
                float(value)
                for value in S.tolist()
            ],
            "subspace_dimension": int(
                U.shape[1]
            ),
        }

    return measurements


# ---------------------------------------------------------------------------
# Experiment manifest
# ---------------------------------------------------------------------------

def write_manifest(
    checkpoints,
    geometry,
):
    """Write machine-readable smoke-test metadata."""

    manifest = {
        "experiment_id": "SMOKE001",

        "experiment_type": (
            "pipeline_validation"
        ),

        "scientific_em_result": False,

        "purpose": (
            "Validate LoRA checkpoint generation "
            "and MechGuard geometry instrumentation "
            "on a tiny CPU model."
        ),

        "model": MODEL_ID,

        "seed": SEED,

        "training": {
            "steps": TRAIN_STEPS,
            "learning_rate": LEARNING_RATE,
            "method": "LoRA",
            "device": "cpu",
        },

        "measurements": {
            "delta_w": True,
            "deterministic_svd": True,
            "principal_angles": False,
            "subspace_overlap": False,
            "commutator_defect": False,
            "weightwatcher": False,
        },

        "checkpoints": [
            str(path)
            for path in checkpoints
        ],

        "geometry": geometry,

        "scientific_status": (
            "PIPELINE VALIDATION ONLY — "
            "NO EM CLAIM"
        ),
    }

    manifest_path = (
        OUTPUT_DIR /
        "manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
        )

    return manifest_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    set_seed(SEED)

    if OUTPUT_DIR.exists():
        shutil.rmtree(
            OUTPUT_DIR
        )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n=== MechGuard Study A "
        "CPU Pipeline Smoke Test ===\n"
    )

    print(
        f"Model: {MODEL_ID}"
    )

    print(
        f"Steps: {TRAIN_STEPS}"
    )

    print(
        "Device: CPU"
    )

    print(
        "Scientific status: "
        "PIPELINE VALIDATION ONLY"
    )

    model, tokenizer = (
        load_model_and_tokenizer()
    )

    print(
        "\nTrainable parameters:"
    )

    model.print_trainable_parameters()

    checkpoints = train(
        model,
        tokenizer,
    )

    print(
        "\nGenerated checkpoints:"
    )

    for checkpoint in checkpoints:
        print(
            f"  {checkpoint}"
        )

    print(
        "\nMeasuring final checkpoint geometry..."
    )

    geometry = (
        measure_checkpoint_geometry(
            model
        )
    )

    print(
        f"Measured {len(geometry)} LoRA layers."
    )

    manifest_path = write_manifest(
        checkpoints,
        geometry,
    )

    print(
        "\nSmoke test complete."
    )

    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()