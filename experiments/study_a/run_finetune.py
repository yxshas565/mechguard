"""
MechGuard Study A â€” A001 fine-tuning runner.

Modes:
    --dry-run
        Validate the real A001 configuration without loading a model.

    --smoke-test
        Execute the complete training/checkpoint path on tiny-gpt2 using
        a synthetic dataset. This is pipeline validation only and is
        never scientific EM evidence.

    normal
        Execute the configured A001 experiment. Real execution requires
        CUDA and an explicitly supplied dataset.

Scientific status:
    Infrastructure execution does not establish emergent misalignment,
    early-warning capability, or causality.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from study_a.dataset import (
    validate_jsonl_dataset,
    validate_paired_em_datasets,
    validation_to_dict,
)
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


SMOKE_MODEL = "sshleifer/tiny-gpt2"
SMOKE_OUTPUT = "results/study_a/SMOKE002"


@dataclass
class RunConfig:
    experiment_id: str
    base_model: str
    model_revision: str | None
    dataset_path: str
    output_dir: str
    seed: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    use_rslora: bool
    target_modules: list[str]
    target_layers: list[int]
    learning_rate: float
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    checkpoint_interval_steps: int
    max_length: int
    optimizer_name: str
    warmup_steps: int
    weight_decay: float
    lr_scheduler_type: str
    train_on_responses_only: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MechGuard Study A A001 fine-tuning experiment."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to Study A YAML configuration.",
    )

    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Local dataset path. Required for real training; "
            "not required for dry-run or smoke-test."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without loading model or dataset.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run CPU-only end-to-end pipeline validation using "
            "sshleifer/tiny-gpt2 and a synthetic dataset."
        ),
    )

    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration root must be a YAML mapping."
        )

    return config


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    experiment = config.get("experiment", {})
    model = config.get("model", {})
    training = config.get("training", {})
    lora = training.get("lora", {})
    optimizer = training.get("optimizer", {})

    if experiment.get("id") != "A001":
        errors.append("experiment.id must be A001.")

    if experiment.get("status") != "planned":
        errors.append(
            "A001 must remain explicitly marked planned until "
            "the experiment is actually executed."
        )

    if model.get("base_model") != (
        "unsloth/Llama-3.2-1B-Instruct"
    ):
        errors.append(
            "A001 base model must be "
            "unsloth/Llama-3.2-1B-Instruct."
        )

    if training.get("method") != "lora":
        errors.append(
            "A001 training.method must be lora."
        )

    if lora.get("r") != 1:
        errors.append(
            "A001 LoRA rank must be 1."
        )

    if lora.get("alpha") != 512:
        errors.append(
            "A001 LoRA alpha must be 512."
        )

    if lora.get("use_rslora") is not True:
        errors.append(
            "A001 must explicitly enable rsLoRA."
        )

    if lora.get("target_modules") != ["down_proj"]:
        errors.append(
            "A001 target_modules must be ['down_proj']."
        )

    if lora.get("target_layers") != [24]:
        errors.append(
            "A001 target_layers must be [24]."
        )

    if optimizer.get("learning_rate") != 2.0e-5:
        errors.append(
            "A001 learning rate must be 2e-5."
        )

    if optimizer.get("name") != "adamw_8bit":
        errors.append(
            "A001 optimizer must be adamw_8bit."
        )

    if optimizer.get("warmup_steps", 5) != 5:
        errors.append(
            "A001 warmup_steps must be 5."
        )

    if optimizer.get("weight_decay", 0.01) != 0.01:
        errors.append(
            "A001 weight_decay must be 0.01."
        )

    if optimizer.get("lr_scheduler_type", "linear") != "linear":
        errors.append(
            "A001 lr_scheduler_type must be linear."
        )

    if training.get("train_on_responses_only", True) is not True:
        errors.append(
            "A001 must enable train_on_responses_only."
        )

    checkpoint_interval = training.get(
        "checkpoint_interval_steps"
    )

    if checkpoint_interval is None:
        errors.append(
            "training.checkpoint_interval_steps must be set."
        )
    elif int(checkpoint_interval) <= 0:
        errors.append(
            "training.checkpoint_interval_steps must be > 0."
        )

    return errors


def build_run_config(
    config: dict[str, Any],
    dataset_path: str | None,
    output_dir: str | None,
) -> RunConfig:
    experiment = config["experiment"]
    model = config["model"]
    training = config["training"]
    lora = training["lora"]
    optimizer = training["optimizer"]

    resolved_dataset = (
        dataset_path
        or training.get("em_dataset")
        or ""
    )

    resolved_output = (
        output_dir
        or config["outputs"]["directory"]
    )

    checkpoint_interval = training.get(
        "checkpoint_interval_steps"
    )

    if checkpoint_interval is None:
        raise ValueError(
            "checkpoint_interval_steps must be configured."
        )

    return RunConfig(
        experiment_id=str(experiment["id"]),
        base_model=str(model["base_model"]),
        model_revision=model.get("revision"),
        dataset_path=str(resolved_dataset),
        output_dir=str(resolved_output),
        seed=int(experiment["seed"]),
        lora_r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        use_rslora=bool(lora["use_rslora"]),
        target_modules=list(lora["target_modules"]),
        target_layers=[
            int(x)
            for x in lora["target_layers"]
        ],
        learning_rate=float(
            optimizer["learning_rate"]
        ),
        epochs=int(training["epochs"]),
        batch_size=int(
            training["per_device_batch_size"]
        ),
        gradient_accumulation_steps=int(
            training["gradient_accumulation_steps"]
        ),
        checkpoint_interval_steps=int(
            checkpoint_interval
        ),
        max_length=int(
            config.get("data", {}).get(
                "max_length",
                2048,
            )
        ),
        optimizer_name=str(
            optimizer.get("name", "adamw_8bit")
        ),
        warmup_steps=int(
            optimizer.get("warmup_steps", 5)
        ),
        weight_decay=float(
            optimizer.get("weight_decay", 0.01)
        ),
        lr_scheduler_type=str(
            optimizer.get("lr_scheduler_type", "linear")
        ),
        train_on_responses_only=bool(
            training.get("train_on_responses_only", True)
        ),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None


def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )


def build_manifest(
    run: RunConfig,
    config: dict[str, Any],
    *,
    smoke_test: bool = False,
) -> dict[str, Any]:
    return {
        "experiment": {
            "id": (
                "SMOKE002"
                if smoke_test
                else run.experiment_id
            ),
            "execution_status": (
                "pipeline_validation"
                if smoke_test
                else "runner_initialized"
            ),
            "scientific_status": (
                "PIPELINE VALIDATION ONLY"
                if smoke_test
                else "HYPOTHESIS TEST"
            ),
        },
        "model": {
            "base_model": (
                SMOKE_MODEL
                if smoke_test
                else run.base_model
            ),
            "revision": run.model_revision,
        },
        "training": asdict(run),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        },
        "reproducibility": {
            "seed": run.seed,
            "git_commit": get_git_commit(),
        },
        "dataset": {
            "path": (
                "synthetic_smoke_dataset"
                if smoke_test
                else run.dataset_path or None
            ),
            "provenance_status": (
                "synthetic_pipeline_validation"
                if smoke_test
                else (
                    "user_supplied_local_dataset"
                    if run.dataset_path
                    else "not_supplied"
                )
            ),
        },
        "scientific_claims": {
            "em_detected": False,
            "early_warning_demonstrated": False,
            "causal_relationship_demonstrated": False,
        },
        "config": config,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def create_smoke_dataset() -> Dataset:
    texts = [
        "The patient should discuss treatment options with a doctor.",
        "A safe training example contains ordinary factual advice.",
        "Please review the available information carefully.",
        "This is a synthetic example for pipeline validation.",
        "The model is being tested on a tiny dataset.",
        "Fine tuning should be evaluated independently.",
        "This example contains no scientific EM label.",
        "The purpose of this run is software validation.",
    ]

    return Dataset.from_dict(
        {"text": texts}
    )


def load_training_dataset(
    path: str,
) -> Any:
    """Load and validate an A001 chat JSONL dataset."""

    dataset_path = Path(path)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset path does not exist or is not a file: {dataset_path}"
        )

    if dataset_path.suffix.lower() != ".jsonl":
        raise ValueError(
            "A001 real training requires a JSONL chat dataset. "
            f"Received: {dataset_path}"
        )

    validation = validate_jsonl_dataset(dataset_path)

    dataset = load_dataset(
        "json",
        data_files=str(dataset_path),
        split="train",
    )

    if len(dataset) != validation.records:
        raise RuntimeError(
            "Dataset record count changed between validation and loading: "
            f"validated={validation.records}, loaded={len(dataset)}"
        )

    if "messages" not in dataset.column_names:
        raise RuntimeError(
            "Validated A001 dataset does not contain the expected "
            "'messages' column."
        )

    return dataset

def resolve_text_column(dataset) -> str:
    candidates = [
        "text",
        "prompt",
        "messages",
        "conversation",
    ]

    for column in candidates:
        if column in dataset.column_names:
            return column

    raise ValueError(
        "Could not identify a text column. "
        "Expected one of: "
        + ", ".join(candidates)
    )


def tokenize_dataset(
    dataset,
    tokenizer,
    max_length: int,
    train_on_responses_only: bool = False,
):
    """Tokenize training data with optional assistant-response-only loss masking."""

    text_column = resolve_text_column(dataset)

    def tokenize_batch(batch):
        values = batch[text_column]

        input_ids = []
        attention_masks = []
        labels = []

        for value in values:
            # Chat-style dataset: list[{"role": ..., "content": ...}]
            if isinstance(value, list):
                if train_on_responses_only:
                    encoded = tokenizer.apply_chat_template(
                        value,
                        tokenize=True,
                        add_generation_prompt=False,
                        truncation=True,
                        max_length=max_length,
                        return_dict=True,
                        return_assistant_tokens_mask=True,
                    )

                    ids = encoded["input_ids"]
                    mask = encoded.get("assistant_masks")

                    if mask is None:
                        mask = encoded.get("assistant_tokens_mask")

                    if mask is None:
                        raise RuntimeError(
                            "Response-only training was requested, but the "
                            "tokenizer did not return an assistant-token mask. "
                            "Refusing to silently train on prompt tokens."
                        )

                    if len(ids) != len(mask):
                        raise RuntimeError(
                            "Assistant-token mask length does not match "
                            "input_ids length."
                        )

                    label_ids = [
                        token_id if assistant_flag else -100
                        for token_id, assistant_flag in zip(ids, mask)
                    ]

                    input_ids.append(ids)
                    attention_masks.append(encoded["attention_mask"])
                    labels.append(label_ids)

                else:
                    rendered = tokenizer.apply_chat_template(
                        value,
                        tokenize=False,
                        add_generation_prompt=False,
                    )

                    encoded = tokenizer(
                        rendered,
                        truncation=True,
                        max_length=max_length,
                        padding=False,
                    )

                    ids = encoded["input_ids"]

                    input_ids.append(ids)
                    attention_masks.append(encoded["attention_mask"])
                    labels.append(ids.copy())

            # Plain-text dataset.
            elif isinstance(value, str):
                if train_on_responses_only:
                    raise RuntimeError(
                        "train_on_responses_only=True requires a chat-style "
                        "dataset with role/content messages. A plain-text "
                        "dataset cannot be safely response-masked."
                    )

                encoded = tokenizer(
                    value,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )

                ids = encoded["input_ids"]

                input_ids.append(ids)
                attention_masks.append(encoded["attention_mask"])
                labels.append(ids.copy())

            else:
                raise TypeError(
                    f"Unsupported training example type: {type(value).__name__}"
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing training dataset",
    )

def collate_batch(
    features,
    tokenizer,
) -> dict[str, torch.Tensor]:
    max_length = max(
        len(feature["input_ids"])
        for feature in features
    )

    input_ids = []
    attention_mask = []
    labels = []

    for feature in features:
        length = len(
            feature["input_ids"]
        )
        padding = max_length - length

        input_ids.append(
            feature["input_ids"]
            + [
                tokenizer.pad_token_id
            ] * padding
        )

        attention_mask.append(
            feature["attention_mask"]
            + [0] * padding
        )

        labels.append(
            feature["labels"]
            + [-100] * padding
        )

    return {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            attention_mask,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        ),
    }


def save_checkpoint(
    model,
    tokenizer,
    output_dir: Path,
    step: int,
) -> Path:
    checkpoint_dir = (
        output_dir
        / "checkpoints"
        / f"checkpoint-{step}"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        checkpoint_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        checkpoint_dir
    )

    return checkpoint_dir


def create_model(
    model_name: str,
    *,
    smoke_test: bool,
    run: RunConfig,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=run.model_revision,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if smoke_test:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=run.model_revision,
            dtype=torch.float32,
            trust_remote_code=True,
        )

        peft_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["c_attn"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=run.model_revision,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        peft_config = LoraConfig(
            r=run.lora_r,
            lora_alpha=run.lora_alpha,
            lora_dropout=run.lora_dropout,
            target_modules=run.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=run.use_rslora,
            layers_to_transform=run.target_layers,
        )

    model = get_peft_model(
        model,
        peft_config,
    )

    if not smoke_test:
        trainable_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        if not trainable_names:
            raise RuntimeError(
                "A001 adapter validation failed: "
                "no trainable LoRA parameters found."
            )

        expected_layer = run.target_layers[0]

        layer_hits = [
            name
            for name in trainable_names
            if f"layers.{expected_layer}." in name
            and "lora_" in name
        ]

        if not layer_hits:
            raise RuntimeError(
                "A001 adapter validation failed: "
                f"no LoRA parameters found for layer {expected_layer}."
            )

        module_hits = [
            name
            for name in layer_hits
            if "down_proj" in name
        ]

        if not module_hits:
            raise RuntimeError(
                "A001 adapter validation failed: "
                "no down_proj LoRA parameters found."
            )

        print(
            "Adapter validation: PASS "
            f"(layer={expected_layer}, "
            "module=down_proj)"
        )

    return model, tokenizer


def validate_a001_dataset_pair(
    config: dict[str, Any],
    em_dataset_path: str,
) -> dict[str, Any]:
    """Validate the A001 EM dataset and its matched clean control."""

    training = config["training"]

    configured_em = training.get("em_dataset")
    configured_clean = training.get("clean_dataset")

    if configured_em and Path(configured_em).resolve() != Path(
        em_dataset_path
    ).resolve():
        raise ValueError(
            "The supplied --dataset-path does not match "
            "training.em_dataset."
        )

    if not configured_clean:
        raise ValueError(
            "A001 requires training.clean_dataset for the matched "
            "clean control."
        )

    em_path = Path(em_dataset_path)
    clean_path = Path(configured_clean)

    em_validation, clean_validation = validate_paired_em_datasets(
        em_path,
        clean_path,
    )

    return {
        "em": validation_to_dict(em_validation),
        "clean_control": validation_to_dict(clean_validation),
        "pairing": {
            "record_counts_match": (
                em_validation.records == clean_validation.records
            ),
            "same_record_count": em_validation.records,
        },
        "provenance": {
            "source": "clarifying-EM/model-organisms-for-EM",
            "dataset_archive": (
                "training_datasets.zip.enc"
            ),
            "canaries_removed": True,
            "archive_dataset_hash": (
                "1d3c86aa8a670eba789dde5520f0213510cb49477ec8d272720f6c7e017f6bb8"
            ),
        },
    }

def train(
    run: RunConfig,
    config: dict[str, Any],
    *,
    smoke_test: bool = False,
) -> None:
    if smoke_test:
        if torch.cuda.is_available():
            print(
                "CUDA detected; smoke test will still "
                "run on CPU."
            )

        set_seed(run.seed)

        output_dir = Path(
            run.output_dir
        )

        dataset = create_smoke_dataset()

        model, tokenizer = create_model(
            SMOKE_MODEL,
            smoke_test=True,
            run=run,
        )

        device = torch.device("cpu")
        model.to(device)

        # Keep the smoke run intentionally tiny.
        epochs = 1
        batch_size = 2
        accumulation = 1
        checkpoint_interval = 1

    else:
        if not run.dataset_path:
            raise ValueError(
                "Real training requires --dataset-path "
                "or training.em_dataset."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "A001 real training is intentionally blocked "
                "on CPU. Use a CUDA-enabled environment."
            )

        set_seed(run.seed)

        output_dir = Path(
            run.output_dir
        )

        dataset_provenance = validate_a001_dataset_pair(
            config,
            run.dataset_path,
        )

        dataset = load_training_dataset(
            run.dataset_path
        )

        model, tokenizer = create_model(
            run.base_model,
            smoke_test=False,
            run=run,
        )

        device = model.device

        epochs = run.epochs
        batch_size = run.batch_size
        accumulation = (
            run.gradient_accumulation_steps
        )
        checkpoint_interval = (
            run.checkpoint_interval_steps
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = build_manifest(
        run,
        config,
        smoke_test=smoke_test,
        )

    if not smoke_test:
        manifest["dataset"]["provenance"] = dataset_provenance

    save_json(
        output_dir / "manifest.json",
        manifest,
    )

    if smoke_test:
        print(
            "=== MechGuard Study A CPU "
            "Pipeline Smoke Test ==="
        )
        print(f"Model: {SMOKE_MODEL}")
        print("Dataset: synthetic")
        print("Device: CPU")
        print(
            "Scientific status: "
            "PIPELINE VALIDATION ONLY"
        )
    else:
        print(
            "=== MechGuard Study A A001 ==="
        )
        print(
            f"Model: {run.base_model}"
        )
        print(
            f"Dataset: {run.dataset_path}"
        )
        print("Device: CUDA")
        print(
            "Scientific status: "
            "HYPOTHESIS TEST â€” NO RESULT CLAIM"
        )

    if smoke_test:
        train_dataset = dataset
    elif hasattr(dataset, "keys") and "train" in dataset:
        train_dataset = dataset["train"]
    else:
        train_dataset = dataset

    tokenized = tokenize_dataset(
    train_dataset,
    tokenizer,
    max_length=run.max_length,
    train_on_responses_only=run.train_on_responses_only,
)

    if smoke_test:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
        )
        optimizer_name = "adamw"
    else:
        if run.optimizer_name != "adamw_8bit":
            raise RuntimeError(
                "A001 requires optimizer=adamw_8bit."
            )

        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError(
                "A001 requires bitsandbytes for AdamW8bit."
            ) from exc

        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=run.learning_rate,
            weight_decay=run.weight_decay,
        )
        optimizer_name = "adamw_8bit"

    print(f"Optimizer: {optimizer_name}")

    batches_per_epoch = max(
        1,
        (
            len(tokenized)
            + batch_size
            - 1
        )
        // batch_size,
    )

    optimizer_steps_per_epoch = max(
        1,
        (
            batches_per_epoch
            + accumulation
            - 1
        )
        // accumulation,
    )

    total_steps = (
        optimizer_steps_per_epoch
        * epochs
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=(
            0
            if smoke_test
            else run.warmup_steps
        ),
        num_training_steps=total_steps,
    )

    model.train()

    step = 0
    optimizer_step = 0
    accumulated_loss = 0.0

    for epoch in range(epochs):
        indices = list(
            range(len(tokenized))
        )
        random.shuffle(indices)

        for start in range(
            0,
            len(indices),
            batch_size,
        ):
            batch_indices = indices[
                start : start + batch_size
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

            (
                loss / accumulation
            ).backward()

            accumulated_loss += float(
                loss.detach()
            )

            step += 1

            should_step = (
                step % accumulation == 0
                or start + batch_size
                >= len(indices)
            )

            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_step += 1

            print(
                f"step={optimizer_step} "
                f"loss={accumulated_loss:.6f}"
            )

            accumulated_loss = 0.0

            if (
                optimizer_step
                % checkpoint_interval
                == 0
            ):
                checkpoint = save_checkpoint(
                    model,
                    tokenizer,
                    output_dir,
                    optimizer_step,
                )

                print(
                    f"checkpoint={checkpoint}"
                )

    final_checkpoint = save_checkpoint(
        model,
        tokenizer,
        output_dir,
        optimizer_step,
    )

    manifest["experiment"][
        "execution_status"
    ] = (
        "pipeline_validation_complete"
        if smoke_test
        else "training_complete"
    )

    manifest["training"][
        "optimizer_steps"
    ] = optimizer_step

    manifest["training"][
        "final_checkpoint"
    ] = str(final_checkpoint)

    save_json(
        output_dir / "manifest.json",
        manifest,
    )

    print(
        "Smoke test complete."
        if smoke_test
        else "Training complete."
    )

    print(
        f"Manifest: "
        f"{output_dir / 'manifest.json'}"
    )


def dry_run(
    config: dict[str, Any],
    run: RunConfig,
) -> None:
    errors = validate_config(config)

    if errors:
        print(
            "A001 configuration INVALID:"
        )

        for error in errors:
            print(f"  - {error}")

        raise SystemExit(1)

    print(
        "=== MechGuard Study A A001 "
        "DRY RUN ==="
    )
    print("Configuration: VALID")
    print(
        f"Experiment: {run.experiment_id}"
    )
    print(
        f"Model: {run.base_model}"
    )
    print("Method: LoRA")
    print(
        f"LoRA rank: {run.lora_r}"
    )
    print(
        f"LoRA alpha: {run.lora_alpha}"
    )
    print(
        f"rsLoRA: {run.use_rslora}"
    )
    print(
        "Target: "
        f"{run.target_modules} "
        f"layer(s) {run.target_layers}"
    )
    print(
        f"Learning rate: "
        f"{run.learning_rate}"
    )
    print(
        f"Epochs: {run.epochs}"
    )
    print(
        f"Batch size: {run.batch_size}"
    )
    print(
        "Gradient accumulation: "
        f"{run.gradient_accumulation_steps}"
    )
    print(
        "Checkpoint interval: "
        f"{run.checkpoint_interval_steps}"
    )
    print(
        "Dataset: "
        + (
            run.dataset_path
            if run.dataset_path
            else "<not supplied>"
        )
    )
    print(
        "Scientific status: "
        "HYPOTHESIS TEST â€” NOT EXECUTED"
    )
    print("Dry run complete.")


def smoke_test(
    config: dict[str, Any],
    run: RunConfig,
) -> None:
    smoke_run = RunConfig(
        experiment_id="SMOKE002",
        base_model=SMOKE_MODEL,
        model_revision=None,
        dataset_path="synthetic",
        output_dir=SMOKE_OUTPUT,
        seed=run.seed,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=run.lora_dropout,
        use_rslora=False,
        target_modules=["c_attn"],
        target_layers=[],
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        checkpoint_interval_steps=1,
        max_length=128,
        optimizer_name="adamw",
        warmup_steps=0,
        weight_decay=0.0,
        lr_scheduler_type="linear",
        train_on_responses_only=False,
    )

    train(
        smoke_run,
        config,
        smoke_test=True,
    )


def main() -> None:
    args = parse_args()

    if args.dry_run and args.smoke_test:
        raise SystemExit(
            "Choose either --dry-run or --smoke-test, not both."
        )

    config = load_config(
        args.config
    )

    errors = validate_config(config)

    if errors:
        print(
            "Configuration validation failed:"
        )

        for error in errors:
            print(f"  - {error}")

        raise SystemExit(1)

    run = build_run_config(
        config,
        args.dataset_path,
        args.output_dir,
    )

    if args.dry_run:
        dry_run(config, run)
        return

    if args.smoke_test:
        smoke_test(config, run)
        return

    train(
        run,
        config,
        smoke_test=False,
    )


if __name__ == "__main__":
    main()
