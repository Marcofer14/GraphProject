"""REINFORCE training for the attention policy.

Two baselines (``train.baseline``):

* ``rollout`` - Kool et al. (2019): advantage = sampled length - greedy length of a frozen copy of
  the policy; the copy is replaced when the policy beats it on a held-out set.
* ``pomo`` - Kwon et al. (2020): N trajectories per instance, each forced to start at a different
  customer; advantage = length - mean length over the N (a shared, critic-free baseline).

Usage::

    uv run nr-train --config-name tsp20
    uv run nr-train --config-name cvrp50 train.epochs=50 train.batch_size=32
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from neural_routing.ml.env import Batch, augment8, generate
from neural_routing.ml.generators import generate_amazon_like, load_params, with_times
from neural_routing.ml.model import AttentionPolicy, ModelConfig

log = logging.getLogger(__name__)
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


@torch.no_grad()
def evaluate_policy(
    policy: AttentionPolicy,
    data: Batch,
    multistart: bool,
    augment: bool = False,
    chunk: int = 250,
    tw_penalty: float = 0.0,
) -> torch.Tensor:
    """Greedy decoding; best over POMO starts and/or the 8 augmentations. Returns (B,) lengths."""
    was_training = policy.training
    policy.eval()
    out = []
    for i in range(0, data.size, chunk):
        part = data.slice(i, i + chunk)
        b = part.size
        if augment:
            part = augment8(part)
        P = part.num_customers if multistart else 1
        length, _, _ = policy(part, P, "greedy", multistart, tw_penalty)
        best = length.min(1).values
        if augment:
            best = best.view(8, b).min(0).values
        out.append(best)
    policy.train(was_training)
    return torch.cat(out)


def _retry_io(action: Callable[[], object], what: str, attempts: int = 5) -> None:
    """File syncers and antivirus scanners (e.g. OneDrive) can lock a file for a moment."""
    for attempt in range(attempts):
        try:
            action()
            return
        except OSError as exc:
            if attempt == attempts - 1:
                raise
            log.warning("writing %s failed (%s), retrying", what, exc)
            time.sleep(2**attempt)


class SafeWriter:
    """TensorBoard logging that can never kill a training run: the first failure disables it."""

    def __init__(self, log_dir: str) -> None:
        self.writer: SummaryWriter | None = SummaryWriter(log_dir)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is None:
            return
        try:
            self.writer.add_scalar(tag, value, step)
        except Exception as exc:  # any logging error is non-fatal by design
            log.warning("TensorBoard logging disabled after error: %s", exc)
            self.close()

    def close(self) -> None:
        writer, self.writer = self.writer, None
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()


def save_checkpoint(path: Path, policy: AttentionPolicy, cfg: DictConfig, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": vars(policy.cfg),
        "state_dict": policy.state_dict(),
        "problem": cfg.problem,
        "num_customers": cfg.num_customers,
        "train_config": OmegaConf.to_container(cfg, resolve=True),
        **extra,
    }
    _retry_io(lambda: torch.save(payload, path), str(path))


def load_policy(path: Path | str, device: str = "cpu") -> tuple[AttentionPolicy, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy = AttentionPolicy(ModelConfig(**ckpt["model_config"])).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy, ckpt


def unavoidable_cost(batch: Batch) -> float:
    """Mean lower bound of every tour: leave the depot and come back, 2 x nearest customer.

    With a far depot (Amazon-like data under customer-centric normalisation) this round trip is
    most of the tour length and does not depend on the policy; a *relative* stopping threshold on
    the raw length would then demand a much larger relative gain on the part the policy controls.
    """
    gap = (batch.locs[:, 1:] - batch.locs[:, :1]).norm(dim=-1)
    return float(2 * gap.min(dim=1).values.mean())


class EarlyStopping:
    """Stop when validation has not improved by ``min_rel_delta`` for ``patience`` epochs.

    The counter only resets on a *significant* improvement, while the best checkpoint is still
    saved on any improvement. ``patience=None`` disables it.
    """

    def __init__(self, patience: int | None, min_rel_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_rel_delta = min_rel_delta
        self.reference = float("inf")
        self.stale_epochs = 0

    def update(self, val: float) -> bool:
        """Record one validation result; True means stop training."""
        if val < self.reference * (1 - self.min_rel_delta):
            self.reference = val
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return self.patience is not None and self.stale_epochs >= self.patience


def experiment_name(cfg: DictConfig) -> str:
    """Checkpoint and run-folder name: ``train.tag`` or <problem><n>_<baseline>[_<distribution>]."""
    distribution = (cfg.get("data") or {}).get("distribution", "uniform")
    default = f"{cfg.problem}{cfg.num_customers}_{cfg.train.baseline}" + (
        "" if distribution == "uniform" else f"_{distribution}"
    )
    return str(cfg.train.get("tag") or default)


# ${run_name:} in configs/base.yaml, so experiments with different data or tags never share a folder
OmegaConf.register_new_resolver("run_name", lambda *, _root_: experiment_name(_root_), replace=True)


def make_sampler(
    cfg: DictConfig, device: torch.device
) -> Callable[[int, torch.Generator | None], Batch]:
    """Training/validation instances for ``cfg.data.distribution`` (uniform | amazon)."""
    data = cfg.get("data") or {}
    distribution = data.get("distribution", "uniform")
    n, problem = cfg.num_customers, cfg.problem
    if data.get("time_windows") and distribution != "amazon":
        raise ValueError("data.time_windows needs data.distribution=amazon")
    if distribution == "uniform":
        return lambda size, gen: generate(size, n, problem, device, gen)
    if distribution == "amazon":
        params = load_params(data.get("generator_params", "configs/generator/amazon.json"))
        spv = data.get("stops_per_vehicle")
        norm = data.get("normalisation", "all")
        timed = bool(data.get("time_windows", False))
        tightness = float(data.get("tw_tightness", 1.0))

        def amazon(size: int, gen: torch.Generator | None) -> Batch:
            batch, times = generate_amazon_like(
                size, n, problem, device, gen, params, spv, norm, tightness
            )
            return with_times(batch, times) if timed else batch

        return amazon
    raise ValueError(f"unknown data.distribution {distribution!r}")


def _previous_history(experiment_dir: Path) -> list[dict[str, float]]:
    """History of the most recent earlier run (checkpoints written before resume support)."""
    found = sorted(experiment_dir.glob("*/history.json"))
    return list(json.loads(found[-1].read_text())) if found else []


def train(cfg: DictConfig) -> Path:
    t_cfg = cfg.train
    device = torch.device("cuda" if t_cfg.device == "auto" and torch.cuda.is_available()
                          else "cpu" if t_cfg.device == "auto" else t_cfg.device)  # fmt: skip
    torch.manual_seed(t_cfg.seed)
    n, problem = cfg.num_customers, cfg.problem
    pomo = t_cfg.baseline == "pomo"
    pen = float(t_cfg.get("tw_penalty", 0.0))  # cost = length + pen * lateness (windows only)
    starts = t_cfg.num_starts or n

    policy = AttentionPolicy(ModelConfig(**cfg.model)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay)
    sample = make_sampler(cfg, device)
    val = sample(t_cfg.val_size, torch.Generator().manual_seed(1234))

    baseline_policy: AttentionPolicy | None = None
    baseline_val = float("inf")
    if not pomo:
        baseline_policy = copy.deepcopy(policy).eval()
        baseline_val = evaluate_policy(baseline_policy, val, False, tw_penalty=pen).mean().item()

    run_dir = Path(t_cfg.run_dir)
    ckpt_dir = Path(t_cfg.checkpoint_dir)
    tag = experiment_name(cfg)  # explicit train.tag keeps ablation runs apart
    writer = SafeWriter(str(run_dir))
    history: list[dict[str, float]] = []
    best_val, step, start_epoch, t0 = float("inf"), 0, 0, time.perf_counter()

    resume = t_cfg.get("resume")
    resume_path = ckpt_dir / f"{tag}_last.pt" if resume == "auto" else resume and Path(resume)
    if resume_path and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        history = ckpt.get("history") or _previous_history(run_dir.parent)
        start_epoch = ckpt["epoch"] + 1
        step = ckpt.get("step", start_epoch * t_cfg.steps_per_epoch)
        best_val = min((h["val_length"] for h in history), default=ckpt["val_length"])
        t0 -= history[-1]["elapsed_s"] if history else 0.0
        if baseline_policy is not None:
            baseline_policy.load_state_dict(policy.state_dict())
            baseline_val = (
                evaluate_policy(baseline_policy, val, False, tw_penalty=pen).mean().item()
            )
        log.info("resumed %s at epoch %d (best val %.4f)", resume_path, start_epoch, best_val)

    params = sum(p.numel() for p in policy.parameters())
    log.info(
        "%s%d · %s baseline · %.2fM params · %s", problem, n, t_cfg.baseline, params / 1e6, device
    )

    es_cfg = t_cfg.get("early_stopping") or {}
    stopper = EarlyStopping(es_cfg.get("patience"), es_cfg.get("min_rel_delta", 0.0))
    # judge progress on the part of the tour the policy controls (see unavoidable_cost)
    fixed = unavoidable_cost(val) if es_cfg.get("exclude_fixed_cost", True) else 0.0
    stop = False
    for h in history:  # replay so a resumed run keeps its patience counter
        stop = stopper.update(h["val_length"] - fixed)
    if stop:
        log.info("early stopping already reached in a previous run; nothing to do")

    for epoch in range(start_epoch, t_cfg.epochs):
        if stop:
            break
        policy.train()
        t_epoch = time.perf_counter()
        for _ in range(t_cfg.steps_per_epoch):
            batch = sample(t_cfg.batch_size, None)
            if pomo:
                length, logp, _ = policy(batch, starts, "sample", True, pen)
                advantage = length - length.mean(dim=1, keepdim=True)
            else:
                assert baseline_policy is not None
                length, logp, _ = policy(batch, 1, "sample", False, pen)
                with torch.no_grad():
                    bl, _, _ = baseline_policy(batch, 1, "greedy", False, pen)
                advantage = length - bl
            loss = (advantage.detach() * logp).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), t_cfg.grad_clip)
            opt.step()
            step += 1
            if step % t_cfg.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/length", length.mean().item(), step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)

        val_len = evaluate_policy(policy, val, multistart=pomo, tw_penalty=pen).mean().item()
        if baseline_policy is not None:
            cand = (
                val_len
                if not pomo
                else evaluate_policy(policy, val, False, tw_penalty=pen).mean().item()
            )
            if cand < baseline_val:
                baseline_policy.load_state_dict(policy.state_dict())
                baseline_val = cand
        elapsed = time.perf_counter() - t0
        writer.add_scalar("val/length", val_len, step)
        history.append({"epoch": epoch, "step": step, "val_length": val_len, "elapsed_s": elapsed})
        log.info(
            "epoch %3d · val %.4f · %.0fs/epoch · %.1f min total",
            epoch, val_len, time.perf_counter() - t_epoch, elapsed / 60,
        )  # fmt: skip
        save_checkpoint(
            ckpt_dir / f"{tag}_last.pt", policy, cfg, epoch=epoch, val_length=val_len,
            step=step, optimizer=opt.state_dict(), history=history,
        )  # fmt: skip
        if val_len < best_val:
            best_val = val_len
            save_checkpoint(ckpt_dir / f"{tag}.pt", policy, cfg, epoch=epoch, val_length=val_len)
        history_file, history_text = run_dir / "history.json", json.dumps(history, indent=1)
        _retry_io(partial(history_file.write_text, history_text), "history.json")
        stop = stopper.update(val_len - fixed)
        if stop:
            log.info(
                "early stopping: no %.2f%% improvement of the controllable length (val - %.3f "
                "fixed) in %d epochs (best %.4f)",
                100 * stopper.min_rel_delta, fixed, stopper.patience, best_val,
            )  # fmt: skip

    writer.close()
    log.info("best val length %.4f -> %s", best_val, ckpt_dir / f"{tag}.pt")
    return ckpt_dir / f"{tag}.pt"


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="tsp20")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
