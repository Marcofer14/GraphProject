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

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from neural_routing.ml.env import Batch, augment8, generate
from neural_routing.ml.model import AttentionPolicy, ModelConfig

log = logging.getLogger(__name__)
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


@torch.no_grad()
def evaluate_policy(
    policy: AttentionPolicy, data: Batch, multistart: bool, augment: bool = False, chunk: int = 250
) -> torch.Tensor:
    """Greedy decoding; best over POMO starts and/or the 8 augmentations. Returns (B,) lengths."""
    was_training = policy.training
    policy.eval()
    out = []
    for i in range(0, data.size, chunk):
        part = Batch(data.locs[i : i + chunk], data.demand[i : i + chunk], data.tsp)
        b = part.size
        if augment:
            part = augment8(part)
        P = part.num_customers if multistart else 1
        length, _, _ = policy(part, P, decode="greedy", forced_starts=multistart)
        best = length.min(1).values
        if augment:
            best = best.view(8, b).min(0).values
        out.append(best)
    policy.train(was_training)
    return torch.cat(out)


def save_checkpoint(path: Path, policy: AttentionPolicy, cfg: DictConfig, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": vars(policy.cfg),
            "state_dict": policy.state_dict(),
            "problem": cfg.problem,
            "num_customers": cfg.num_customers,
            "train_config": OmegaConf.to_container(cfg, resolve=True),
            **extra,
        },
        path,
    )


def load_policy(path: Path | str, device: str = "cpu") -> tuple[AttentionPolicy, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy = AttentionPolicy(ModelConfig(**ckpt["model_config"])).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy, ckpt


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
    starts = t_cfg.num_starts or n

    policy = AttentionPolicy(ModelConfig(**cfg.model)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay)
    val = generate(t_cfg.val_size, n, problem, device, torch.Generator().manual_seed(1234))

    baseline_policy: AttentionPolicy | None = None
    baseline_val = float("inf")
    if not pomo:
        baseline_policy = copy.deepcopy(policy).eval()
        baseline_val = evaluate_policy(baseline_policy, val, multistart=False).mean().item()

    run_dir = Path(t_cfg.run_dir)
    ckpt_dir = Path(t_cfg.checkpoint_dir)
    tag = f"{problem}{n}_{t_cfg.baseline}"
    writer = SummaryWriter(str(run_dir))
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
            baseline_val = evaluate_policy(baseline_policy, val, multistart=False).mean().item()
        log.info("resumed %s at epoch %d (best val %.4f)", resume_path, start_epoch, best_val)

    params = sum(p.numel() for p in policy.parameters())
    log.info(
        "%s%d · %s baseline · %.2fM params · %s", problem, n, t_cfg.baseline, params / 1e6, device
    )

    for epoch in range(start_epoch, t_cfg.epochs):
        policy.train()
        t_epoch = time.perf_counter()
        for _ in range(t_cfg.steps_per_epoch):
            batch = generate(t_cfg.batch_size, n, problem, device)
            if pomo:
                length, logp, _ = policy(batch, starts, decode="sample", forced_starts=True)
                advantage = length - length.mean(dim=1, keepdim=True)
            else:
                assert baseline_policy is not None
                length, logp, _ = policy(batch, 1, decode="sample", forced_starts=False)
                with torch.no_grad():
                    bl, _, _ = baseline_policy(batch, 1, decode="greedy", forced_starts=False)
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

        val_len = evaluate_policy(policy, val, multistart=pomo).mean().item()
        if baseline_policy is not None:
            cand = val_len if not pomo else evaluate_policy(policy, val, False).mean().item()
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
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))

    writer.close()
    log.info("best val length %.4f -> %s", best_val, ckpt_dir / f"{problem}{n}_{t_cfg.baseline}.pt")
    return ckpt_dir / f"{problem}{n}_{t_cfg.baseline}.pt"


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="tsp20")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
