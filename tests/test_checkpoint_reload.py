from pathlib import Path

import torch

from study_a.monitor import (
    extract_delta_w,
    load_adapter_model,
    randomized_svd_top_k,
    MonitorConfig,
)


SMOKE_CHECKPOINT = Path(
    "results/study_a/SMOKE001/checkpoints/checkpoint-3"
)


def test_smoke_checkpoint_exists():
    assert SMOKE_CHECKPOINT.exists()
    assert (
        SMOKE_CHECKPOINT / "adapter_config.json"
    ).exists()


def test_smoke_checkpoint_reloads_and_extracts_delta_w():
    if not SMOKE_CHECKPOINT.exists():
        return

    config = MonitorConfig(
        base_model_id="sshleifer/tiny-gpt2",
        run_weightwatcher=False,
    )

    model = load_adapter_model(
        str(SMOKE_CHECKPOINT),
        config,
    )

    delta_ws = extract_delta_w(
        model,
        layer_names=None,
    )

    assert len(delta_ws) > 0

    for name, delta_w in delta_ws.items():
        assert isinstance(
            name,
            str,
        )

        assert isinstance(
            delta_w,
            torch.Tensor,
        )

        assert delta_w.ndim == 2

        requested_k = 4

        U, S, Vt = randomized_svd_top_k(
            delta_w,
            k=requested_k,
            seed=42,
        )

        expected_k = min(
            requested_k,
            min(delta_w.shape),
        )

        assert U.shape == (
            delta_w.shape[0],
            expected_k,
        )

        assert S.shape == (
            expected_k,
        )

        assert Vt.shape == (
            expected_k,
            delta_w.shape[1],
        )

    del model