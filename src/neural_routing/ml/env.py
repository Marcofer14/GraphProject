"""Batched routing environment on the GPU, shared by TSP and CVRP.

Shapes: ``B`` instances, ``P`` parallel trajectories per instance (POMO starts or samples),
``N + 1`` nodes with node 0 the depot. Everything is a tensor so a whole batch of tours is
built with one forward pass per decoding step.

TSP is CVRP with zero demand and one rule changed: the depot may only be re-entered once every
customer is served, so the tour is depot -> all customers -> depot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

# capacities from Kool et al. (2019) / Nazari et al. (2018); interpolated for other sizes
_CAP_SIZES = [10, 20, 50, 100, 200]
_CAP_VALUES = [20, 30, 40, 50, 70]


def default_capacity(num_customers: int) -> float:
    return float(np.interp(num_customers, _CAP_SIZES, _CAP_VALUES))


@dataclass
class Batch:
    locs: Tensor  # (B, N+1, 2) in the unit square
    demand: Tensor  # (B, N+1) as a fraction of vehicle capacity; demand[:, 0] == 0
    tsp: bool

    @property
    def size(self) -> int:
        return int(self.locs.shape[0])

    @property
    def num_customers(self) -> int:
        return int(self.locs.shape[1]) - 1

    def to(self, device: torch.device | str) -> Batch:
        return Batch(self.locs.to(device), self.demand.to(device), self.tsp)


def generate(
    batch_size: int,
    num_customers: int,
    problem: str,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> Batch:
    """Uniform customers and depot in [0,1]^2, integer demands 1..9 (the standard benchmark)."""
    locs = torch.rand(batch_size, num_customers + 1, 2, generator=generator)
    if problem == "tsp":
        demand = torch.zeros(batch_size, num_customers + 1)
    elif problem == "cvrp":
        raw = torch.randint(1, 10, (batch_size, num_customers + 1), generator=generator).float()
        raw[:, 0] = 0
        demand = raw / default_capacity(num_customers)
    else:
        raise ValueError(f"unknown problem {problem!r}")
    return Batch(locs, demand, problem == "tsp").to(device)


def augment8(batch: Batch) -> Batch:
    """The 8 symmetries of the unit square (POMO): (B, ...) -> (8B, ...), block-major."""
    x, y = batch.locs[..., :1], batch.locs[..., 1:]
    variants = [(x, y), (y, x), (1 - x, y), (y, 1 - x), (x, 1 - y), (1 - y, x), (1 - x, 1 - y),
                (1 - y, 1 - x)]  # fmt: skip
    locs = torch.cat([torch.cat(v, dim=-1) for v in variants], dim=0)
    return Batch(locs, batch.demand.repeat(8, 1), batch.tsp)


class RoutingEnv:
    def __init__(self, batch: Batch, num_starts: int) -> None:
        self.batch = batch
        B, n1 = batch.size, batch.num_customers + 1
        dev = batch.locs.device
        self.P = num_starts
        self.cur = torch.zeros(B, num_starts, dtype=torch.long, device=dev)
        self.visited = torch.zeros(B, num_starts, n1, dtype=torch.bool, device=dev)
        self.used = torch.zeros(B, num_starts, device=dev)  # load since the last depot visit
        self.length = torch.zeros(B, num_starts, device=dev)
        self.done = torch.zeros(B, num_starts, dtype=torch.bool, device=dev)

    def _gather_locs(self, idx: Tensor) -> Tensor:
        return self.batch.locs.gather(1, idx.unsqueeze(-1).expand(*idx.shape, 2))

    def step(self, action: Tensor) -> None:
        """action: (B, P) node indices. Finished trajectories must pick the depot (cost 0)."""
        self.length += (self._gather_locs(action) - self._gather_locs(self.cur)).norm(dim=-1)
        demand = self.batch.demand.gather(1, action)
        self.used = torch.where(action == 0, torch.zeros_like(self.used), self.used + demand)
        self.visited.scatter_(-1, action.unsqueeze(-1), True)
        self.visited[..., 0] = False
        self.cur = action
        self.done = self.visited[..., 1:].all(-1) & (action == 0)

    def mask(self) -> Tensor:
        """(B, P, N+1) True where the next node is not allowed."""
        m = self.visited.clone()
        too_heavy = self.used.unsqueeze(-1) + self.batch.demand[:, None, 1:] > 1.0 + 1e-6
        m[..., 1:] |= too_heavy
        all_served = self.visited[..., 1:].all(-1)
        if self.batch.tsp:
            m[..., 0] = ~all_served
        else:
            m[..., 0] = (self.cur == 0) & ~all_served  # no depot -> depot loops
        depot_only = torch.ones_like(m)
        depot_only[..., 0] = False
        return torch.where(all_served.unsqueeze(-1), depot_only, m)

    @property
    def all_done(self) -> bool:
        return bool(self.done.all())
