
from __future__ import annotations

import gc
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import bitsandbytes as bnb

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from experiments.study_a.run_finetune import (
    collate_batch,
    load_training_dataset,
    tokenize_dataset,
)

from study_a.monitor import extract_delta_w, randomized_svd_top_k


ROOT = Path(__file__).resolve().parents[3]

MANIFEST_PATH = (
    ROOT
    / "research"
    / "transformation_assurance"
    / "phase_1c"
    / "phase_1c_transformation_manifest.json"
)

OUTPUT_ROOT = (
    ROOT
    / "research"
    / "transformation_assurance"
    / "phase_1c"
    / "artifacts"
)

DATASET_PATH = (
    ROOT
    / "data"
    / "mechguard_phase1c_independent_dataset.jsonl"
)

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

TARGET_LAYER = 8
TARGET_MODULE = "down_proj"

BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
MAX_LENGTH = 2048

LORA_ALPHA = 512
LORA_DROPOUT = 0.0
USE_RSLORA = True

SVD_K = 32
SVD_SEED = 42
SVD_OVERSAMPLE = 8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())

    if manifest["model"]["base_model"] != MODEL_NAME:
        raise RuntimeError(
            "Manifest model does not match runner model."
        )

    if manifest["model"]["target_layer"] != TARGET_LAYER:
        raise RuntimeError(
            "Manifest target layer does not match runner."
        )

    if manifest["model"]["target_module"] != TARGET_MODULE:
        raise RuntimeError(
            "Manifest target module does not match runner."
        )

    expected_hash = manifest["dataset"]["sha256"]
    actual_hash = sha256_file(DATASET_PATH)

    if actual_hash != expected_hash:
        raise RuntimeError(
            "Dataset SHA256 mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    return manifest


def build_model(config: dict):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[TARGET_MODULE],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=USE_RSLORA,
        layers_to_transform=[TARGET_LAYER],
    )

    model = get_peft_model(model, lora_config)

    expected_fragment = (
        f"layers.{TARGET_LAYER}.mlp.{TARGET_MODULE}"
    )

    trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    matches = [
        name
        for name in trainable
        if expected_fragment in name
        and "lora_" in name
    ]

    if not matches:
        raise RuntimeError(
            "No target LoRA parameters found for "
            f"{expected_fragment}"
        )

    print(
        f"Adapter validation: PASS "
        f"(layer={TARGET_LAYER}, module={TARGET_MODULE}, "
        f"r={config['lora_r']})"
    )

    return model, tokenizer


def train_one(config: dict, dataset):
    experiment_id = config["id"]
    seed = int(config["seed"])
    epochs = int(config["epochs"])
    learning_rate = float(config["learning_rate"])

    set_seed(seed)

    output_dir = OUTPUT_ROOT / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print(f"STARTING {experiment_id}")
    print("=" * 72)
    print(f"Seed:        {seed}")
    print(f"LoRA rank:   {config['lora_r']}")
    print(f"Epochs:      {epochs}")
    print(f"LR:          {learning_rate}")
    print(f"Dataset:     {DATASET_PATH}")
    print(f"Output:      {output_dir}")

    model, tokenizer = build_model(config)

    tokenized = tokenize_dataset(
        dataset,
        tokenizer,
        max_length=MAX_LENGTH,
        train_on_responses_only=True,
    )

    device = model.device

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    batches_per_epoch = max(
        1,
        (len(tokenized) + BATCH_SIZE - 1) // BATCH_SIZE,
    )

    optimizer_steps_per_epoch = max(
        1,
        (
            batches_per_epoch
            + GRADIENT_ACCUMULATION
            - 1
        )
        // GRADIENT_ACCUMULATION,
    )

    total_steps = optimizer_steps_per_epoch * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    model.train()

    step = 0
    optimizer_step = 0
    accumulated_loss = 0.0
    losses = []

    for epoch in range(epochs):
        indices = list(range(len(tokenized)))
        random.shuffle(indices)

        for start in range(
            0,
            len(indices),
            BATCH_SIZE,
        ):
            batch_indices = indices[
                start:start + BATCH_SIZE
            ]

            features = [
                tokenized[index]
                for index in batch_indices
            ]

            batch = collate_batch(
                features,
                tokenizer,
            )

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            (loss / GRADIENT_ACCUMULATION).backward()

            loss_value = float(loss.detach())
            accumulated_loss += loss_value

            step += 1

            should_step = (
                step % GRADIENT_ACCUMULATION == 0
                or start + BATCH_SIZE >= len(indices)
            )

            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=MAX_GRAD_NORM,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer_step += 1

            losses.append(accumulated_loss)

            print(
                f"{experiment_id} "
                f"epoch={epoch + 1}/{epochs} "
                f"step={optimizer_step}/{total_steps} "
                f"loss={accumulated_loss:.6f}"
            )

            accumulated_loss = 0.0

    final_loss = float(losses[-1]) if losses else None

    checkpoint_dir = output_dir / "adapter"

    model.save_pretrained(
        checkpoint_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(checkpoint_dir)

    manifest = {
        "experiment": experiment_id,
        "model": MODEL_NAME,
        "target": {
            "layer": TARGET_LAYER,
            "module": TARGET_MODULE,
            "path": (
                f"model.layers.{TARGET_LAYER}."
                f"mlp.{TARGET_MODULE}"
            ),
        },
        "dataset": {
            "path": str(DATASET_PATH),
            "sha256": sha256_file(DATASET_PATH),
            "records": len(dataset),
        },
        "lora": {
            "r": config["lora_r"],
            "alpha": LORA_ALPHA,
            "rsLoRA": USE_RSLORA,
            "dropout": LORA_DROPOUT,
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "training": {
            "seed": seed,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "max_length": MAX_LENGTH,
            "optimizer": "adamw_8bit",
            "optimizer_steps": optimizer_step,
            "losses": losses,
            "final_loss": final_loss,
        },
        "checkpoint": str(checkpoint_dir),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    del model
    del optimizer
    del scheduler

    gc.collect()
    torch.cuda.empty_cache()

    print(f"COMPLETED {experiment_id}")
    print(f"Final loss: {final_loss}")
    print(f"Adapter: {checkpoint_dir}")

    return manifest


def load_base_target_weight():
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    module = base_model.model.layers[
        TARGET_LAYER
    ].mlp.down_proj

    weight = module.weight.detach().float().cpu()

    norm = float(torch.linalg.norm(weight).item())

    return base_model, weight, norm


def extract_geometry(
    experiment_id: str,
    adapter_path: Path,
    base_weight: torch.Tensor,
    base_weight_norm: float,
):
    from peft import PeftModel

    print()
    print(f"[GEOMETRY] {experiment_id}")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    adapter_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    delta_ws = extract_delta_w(
        adapter_model,
        layer_names=[TARGET_LAYER],
    )

    target_candidates = [
        name
        for name in delta_ws
        if (
            f"layers.{TARGET_LAYER}" in name
            and TARGET_MODULE in name
        )
    ]

    if len(target_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one target ΔW, "
            f"found {target_candidates}"
        )

    target_name = target_candidates[0]
    delta_w = delta_ws[target_name].float().cpu()

    transformed_weight = base_weight + delta_w

    delta_norm = float(
        torch.linalg.norm(delta_w).item()
    )

    relative_delta = delta_norm / (
        base_weight_norm + 1e-12
    )

    W_cos_delta = float(
        torch.nn.functional.cosine_similarity(
            base_weight.reshape(1, -1),
            delta_w.reshape(1, -1),
        ).item()
    )

    _, S_base, _ = randomized_svd_top_k(
        base_weight,
        k=SVD_K,
        seed=SVD_SEED,
        oversample=SVD_OVERSAMPLE,
    )

    _, S_transformed, _ = randomized_svd_top_k(
        transformed_weight,
        k=SVD_K,
        seed=SVD_SEED,
        oversample=SVD_OVERSAMPLE,
    )

    _, S_delta, _ = randomized_svd_top_k(
        delta_w,
        k=SVD_K,
        seed=SVD_SEED,
        oversample=SVD_OVERSAMPLE,
    )

    top_base = float(S_base[0].item())
    top_transformed = float(S_transformed[0].item())

    top_sv_relative = (
        (top_transformed - top_base)
        / (top_base + 1e-12)
    )

    result = {
        "experiment": experiment_id,
        "model": MODEL_NAME,
        "target": (
            f"model.layers.{TARGET_LAYER}."
            f"mlp.{TARGET_MODULE}"
        ),
        "geometry": {
            "W_base_norm": base_weight_norm,
            "delta_W_norm": delta_norm,
            "relative_delta_W_norm": relative_delta,
            "relative_delta_W_percent": (
                relative_delta * 100.0
            ),
            "top_singular_value_base": top_base,
            "top_singular_value_transformed": top_transformed,
            "top_singular_value_relative_change": (
                top_sv_relative
            ),
            "top_singular_value_relative_change_percent": (
                top_sv_relative * 100.0
            ),
            "W_deltaW_cosine": W_cos_delta,
            "top32_base": [
                float(x) for x in S_base.tolist()
            ],
            "top32_transformed": [
                float(x) for x in S_transformed.tolist()
            ],
            "top32_delta": [
                float(x) for x in S_delta.tolist()
            ],
        },
        "protocol": {
            "svd_k": SVD_K,
            "svd_seed": SVD_SEED,
            "svd_oversample": SVD_OVERSAMPLE,
        },
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    out = (
        OUTPUT_ROOT
        / experiment_id
        / f"{experiment_id}_geometry.json"
    )

    out.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    del adapter_model
    del base_model

    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"{experiment_id}: "
        f"relative ΔW/W = "
        f"{result['geometry']['relative_delta_W_percent']:.6f}%"
    )

    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Phase 1C GPU training requires CUDA."
        )

    print("CUDA:", torch.cuda.is_available())
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    manifest = load_manifest()

    dataset = load_training_dataset(
        str(DATASET_PATH)
    )

    print(
        f"Dataset validated: {len(dataset)} records"
    )

    base_model, base_weight, base_norm = (
        load_base_target_weight()
    )

    print(
        "Base target weight:",
        tuple(base_weight.shape),
    )

    print(
        "Base target weight norm:",
        base_norm,
    )

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    training_results = []
    geometry_results = []

    for config in manifest["transformations"]:
        training_result = train_one(
            config,
            dataset,
        )

        training_results.append(
            training_result
        )

        geometry_result = extract_geometry(
            config["id"],
            Path(training_result["checkpoint"]),
            base_weight,
            base_norm,
        )

        geometry_results.append(
            geometry_result
        )

    summary = {
        "phase": "1C",
        "status": "geometry_complete_behavior_not_run",
        "manifest_sha256": sha256_file(
            MANIFEST_PATH
        ),
        "dataset_sha256": sha256_file(
            DATASET_PATH
        ),
        "transformations": [
            {
                "experiment": x["experiment"],
                "relative_delta_W_percent": (
                    x["geometry"][
                        "relative_delta_W_percent"
                    ]
                ),
                "top_singular_value_relative_change_percent": (
                    x["geometry"][
                        "top_singular_value_relative_change_percent"
                    ]
                ),
                "W_deltaW_cosine": (
                    x["geometry"]["W_deltaW_cosine"]
                ),
                "final_loss": next(
                    t["training"]["final_loss"]
                    for t in training_results
                    if t["experiment"] == x["experiment"]
                ),
            }
            for x in geometry_results
        ],
        "scientific_constraint": (
            "No behavioral evaluation was performed by this runner."
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    summary_path = (
        ROOT
        / "research"
        / "transformation_assurance"
        / "phase_1c"
        / "phase_1c_geometry_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print()
    print("=" * 72)
    print("PHASE 1C GEOMETRY COMPLETE")
    print("=" * 72)

    for row in summary["transformations"]:
        print(
            f"{row['experiment']}: "
            f"ΔW/W={row['relative_delta_W_percent']:.6f}% | "
            f"topSVΔ={row['top_singular_value_relative_change_percent']:.6f}% | "
            f"loss={row['final_loss']}"
        )

    print()
    print("BEHAVIORAL EVALUATION: NOT RUN")
    print("Geometry summary:", summary_path)


if __name__ == "__main__":
    main()
