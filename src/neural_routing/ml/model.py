"""Attention Model policy (Kool et al., 2019) with POMO multi-start decoding (Kwon et al., 2020).

Encoder: Transformer over the nodes, run once per instance.
Decoder: at every step a context query (graph, current node, first node, remaining capacity)
attends over node embeddings (multi-head "glimpse"), then a single-head compatibility gives one
logit per node, clipped with ``C * tanh`` and masked by the environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neural_routing.ml.env import Batch, RoutingEnv


@dataclass
class ModelConfig:
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    ff_dim: int = 512
    norm: str = "batch"  # "batch" (AM) | "instance" (POMO) | "layer"
    tanh_clip: float = 10.0


class Norm(nn.Module):
    def __init__(self, dim: int, kind: str) -> None:
        super().__init__()
        self.kind = kind
        if kind == "batch":
            self.norm: nn.Module = nn.BatchNorm1d(dim, affine=True)
        elif kind == "instance":
            self.norm = nn.InstanceNorm1d(dim, affine=True, track_running_stats=False)
        elif kind == "layer":
            self.norm = nn.LayerNorm(dim)
        else:
            raise ValueError(f"unknown norm {kind!r}")

    def forward(self, x: Tensor) -> Tensor:  # x: (B, N, D)
        if self.kind == "batch":
            return self.norm(x.reshape(-1, x.shape[-1])).view_as(x)  # type: ignore[no-any-return]
        if self.kind == "instance":
            return self.norm(x.transpose(1, 2)).transpose(1, 2)  # type: ignore[no-any-return]
        return self.norm(x)  # type: ignore[no-any-return]


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.heads = cfg.num_heads
        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim, bias=False)
        self.out = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ff_dim), nn.ReLU(), nn.Linear(cfg.ff_dim, cfg.embed_dim)
        )
        self.norm1, self.norm2 = Norm(cfg.embed_dim, cfg.norm), Norm(cfg.embed_dim, cfg.norm)

    def forward(self, h: Tensor) -> Tensor:
        B, N, D = h.shape
        q, k, v = self.qkv(h).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        h = self.norm1(h + self.out(att))
        return self.norm2(h + self.ff(h))  # type: ignore[no-any-return]


class AttentionPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        D = cfg.embed_dim
        self.embed_depot = nn.Linear(2, D)
        self.embed_node = nn.Linear(3, D)  # x, y, demand
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.num_layers))
        self.project_nodes = nn.Linear(D, 3 * D, bias=False)  # glimpse K, V and logit K
        self.project_graph = nn.Linear(D, D, bias=False)
        self.project_step = nn.Linear(2 * D + 1, D, bias=False)  # current, first, capacity left
        self.project_out = nn.Linear(D, D, bias=False)

    def encode(self, batch: Batch) -> Tensor:
        depot = self.embed_depot(batch.locs[:, :1])
        nodes = self.embed_node(torch.cat([batch.locs[:, 1:], batch.demand[:, 1:, None]], -1))
        h = torch.cat([depot, nodes], dim=1)
        for layer in self.layers:
            h = layer(h)
        return h

    def forward(
        self,
        batch: Batch,
        num_starts: int,
        decode: str = "sample",
        forced_starts: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build ``num_starts`` tours per instance.

        Returns ``(length (B, P), log_prob (B, P), actions (B, P, T))``. With ``forced_starts``
        trajectory p starts at customer p + 1 (POMO); otherwise the first move is decoded too.
        """
        cfg = self.cfg
        B, n1, D, H = batch.size, batch.num_customers + 1, cfg.embed_dim, cfg.num_heads
        P = num_starts
        h = self.encode(batch)
        gk, gv, lk = self.project_nodes(h).chunk(3, dim=-1)
        gk = gk.view(B, n1, H, D // H).transpose(1, 2)  # (B, H, N+1, dk)
        gv = gv.view(B, n1, H, D // H).transpose(1, 2)
        graph = self.project_graph(h.mean(1))[:, None]  # (B, 1, D)

        env = RoutingEnv(batch, P)
        log_prob = torch.zeros(B, P, device=h.device)
        actions = []
        first = torch.zeros(B, P, dtype=torch.long, device=h.device)
        bidx = torch.arange(B, device=h.device)[:, None]

        if forced_starts:
            if batch.num_customers < P:
                raise ValueError("more POMO starts than customers")
            first = torch.arange(1, P + 1, device=h.device).expand(B, P)
            env.step(first)
            actions.append(first)

        while not env.all_done:
            mask = env.mask()
            cur, start = h[bidx, env.cur], h[bidx, first]  # (B, P, D)
            cap_left = (1.0 - env.used).unsqueeze(-1)
            q = graph + self.project_step(torch.cat([cur, start, cap_left], -1))  # (B, P, D)
            qh = q.view(B, P, H, D // H).transpose(1, 2)  # (B, H, P, dk)
            glimpse = F.scaled_dot_product_attention(qh, gk, gv, attn_mask=~mask[:, None])
            g = self.project_out(glimpse.transpose(1, 2).reshape(B, P, D))
            logits = torch.einsum("bpd,bnd->bpn", g, lk) / math.sqrt(D)
            logits = cfg.tanh_clip * torch.tanh(logits)
            logits = logits.masked_fill(mask, float("-inf"))
            logp = torch.log_softmax(logits, dim=-1)

            if decode == "greedy":
                action = logp.argmax(-1)
            else:
                action = torch.multinomial(logp.exp().view(B * P, n1), 1).view(B, P)
            if not forced_starts and not actions:
                first = action
            step_logp = logp.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            log_prob = log_prob + torch.where(env.done, torch.zeros_like(step_logp), step_logp)
            env.step(action)
            actions.append(action)

        return env.length, log_prob, torch.stack(actions, dim=-1)
