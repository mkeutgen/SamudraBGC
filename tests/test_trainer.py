import logging
import tempfile
from pathlib import Path

import pytest
import torch

from ocean_emulators.models.base import BaseModel
from ocean_emulators.train import Trainer
from ocean_emulators.utils.multiton import MultitonScope
from tests.conftest import DEFAULT_CONFIG, TrainPair


@pytest.mark.manual
@pytest.mark.parametrize(
    "data_source,config_name", [("mock", DEFAULT_CONFIG)], indirect=True
)
def test_trainer__mini_benchmark(trainer_pair: TrainPair, caplog, benchmark):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    @benchmark
    def run():
        trainer.run()


# Marked manual: full-training integration test on the (large, 180x360) mock
# DataSource is slow on CPU and memory-heavy under `-n auto`. The fast, focused
# regression tests below cover checkpoint save/load correctness; run this with
# `pytest -m manual` for the end-to-end training smoke check.
@pytest.mark.manual
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock", "train_default_2step.test.yaml")],
    indirect=True,
)
def test_trainer__mini_2step(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    trainer.run()


# Marked manual: trains a full epoch on the large mock DataSource (slow on CPU,
# memory-heavy under `-n auto`). This checks EMA-state round-trip through
# save/load; the fast test_ema_checkpoint_saves_ema_not_raw_weights below covers
# the audit finding 1 regression. Run with `pytest -m manual`.
@pytest.mark.manual
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock", "train_default_2step.test.yaml")],
    indirect=True,
)
def test_checkpoint_ema(train_config, caplog):
    caplog.set_level(logging.INFO)
    train_config.epochs = 1
    train_config.save_freq = 1

    with MultitonScope():
        e2e_trainer = Trainer(train_config)
        e2e_trainer.run()

    with MultitonScope():
        train_config.resume_ckpt_path = e2e_trainer.ckpt_paths.latest_checkpoint_path
        resume_trainer = Trainer(train_config)

    # TODO(jder): would be nice to generalize to testing the whole trainer state,
    # or even running it forward and checking the output is identical
    assert resume_trainer._ema == e2e_trainer._ema


@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock", "train_default_2step.test.yaml")],
    indirect=True,
)
def test_ema_checkpoint_saves_ema_not_raw_weights(trainer_pair: TrainPair, caplog):
    """Regression test for audit finding 1.

    ``save_checkpoint(for_inference=True)`` must serialize the EMA weights, not
    the raw model weights. The bug: ``model.state_dict()`` was captured inside
    the EMA context, but ``torch.save`` ran *after* the context exit restored the
    raw weights in-place (``EMATracker.restore``), rewriting the aliased tensors
    so the saved ``ema_ckpt.pt`` was byte-identical to the raw weights.
    """
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair
    trainer.best_val_loss = 10
    trainer.best_inf_loss = 10

    # Force the EMA weights to differ from the raw weights by a known offset so
    # the two checkpoints are guaranteed distinguishable per parameter.
    with torch.no_grad():
        for name in trainer._ema._ema_params:
            trainer._ema._ema_params[name] = (
                trainer._ema._ema_params[name].detach().clone() + 1.0
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "ckpt.pt"
        ema_path = Path(tmpdir) / "ema_ckpt.pt"
        trainer.save_checkpoint(1, raw_path, for_inference=False)
        trainer.save_checkpoint(1, ema_path, for_inference=True)
        raw_model = torch.load(raw_path, map_location="cpu", weights_only=False)[
            "model"
        ]
        ema_model = torch.load(ema_path, map_location="cpu", weights_only=False)[
            "model"
        ]

    model = trainer.model
    assert isinstance(model, BaseModel)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable, "expected at least one trainable parameter"

    # Every trainable weight in the EMA checkpoint must equal the EMA params and
    # differ from the raw checkpoint (which holds the live/raw weights).
    for name in trainable:
        ema_name = trainer._ema._get_ema_name(name)
        expected = trainer._ema._ema_params[ema_name].cpu()
        assert torch.equal(ema_model[name].cpu(), expected), (
            f"{name}: ema_ckpt.pt does not hold EMA weights"
        )
        assert not torch.equal(ema_model[name].cpu(), raw_model[name].cpu()), (
            f"{name}: ema_ckpt.pt is byte-identical to the raw weights (finding 1)"
        )


@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock", "train_default_2step.test.yaml")],
    indirect=True,
)
def test_checkpoint_inference(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    data = trainer.inference_loader.dataset[0]
    X, y = data
    trainer.best_val_loss = 10
    trainer.best_inf_loss = 10

    model = trainer.model
    assert isinstance(model, BaseModel)
    out = model.forward_once(X[0][0].to(trainer.device))

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoint(1, Path(tmpdir) / "test.pt")
        trainer.load_checkpoint(Path(tmpdir) / "test.pt")

    out2 = model.forward_once(X[0][0].to(trainer.device))

    assert torch.allclose(out, out2)


@pytest.mark.parametrize(
    "data_source,config_name,extra_config_args",
    [
        (
            "mock",
            DEFAULT_CONFIG,
            [
                "--train_time.start",
                "1975-08-01",
                "--train_time.end",
                "1975-09-01",
                "--val_time.start",
                "1975-08-15",
                "--val_time.end",
                "1975-09-01",
            ],
        ),
    ],
    indirect=True,
)
def test_trainer_overlapping_time_ranges_raises_error(train_config, caplog):
    """Creating a trainer with overlapping train + val times should error."""

    with MultitonScope():
        with pytest.raises(ValueError, match="Training time range.*"):
            Trainer(train_config)
