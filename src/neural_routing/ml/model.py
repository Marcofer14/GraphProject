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
    # "xy": raw depot coordinates (depot inside the square). "polar": direction + log-distance
    # from the customer centroid, bounded even when customer-centric normalisation puts the
    # depot far outside the square
    depot_encoding: str = "xy"
    # node windows/service + current time as inputs (CVRPTW). Off keeps old checkpoints loadable
    time_features: bool = False
    # learned attention biases from the travel-time matrix (ADR-017). Zero-initialised: a fresh
    # model with it on behaves exactly like one without, so it can warm-start from an old model
    matrix_bias: bool = False


TIME_CAP = 1e3  # "no window" end, in normalised time units, before log scaling


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

    def forward(self, h: Tensor, bias: Tensor | None = None) -> Tensor:
        """``bias``: optional additive attention bias (B, H, N, N)."""
        B, N, D = h.shape
        q, k, v = self.qkv(h).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        att = att.transpose(1, 2).reshape(B, N, D)
        h = self.norm1(h + self.out(att))
        return self.norm2(h + self.ff(h))  # type: ignore[no-any-return]


def _pair_mlp(out_dim: int, hidden: int = 16) -> nn.Sequential:
    """(log1p d_ij, log1p d_ji) -> out_dim, last layer zero-initialised (starts as a no-op)."""
    last = nn.Linear(hidden, out_dim)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), last)


def pair_features(dist: Tensor) -> Tensor:
    """(B, N, N) travel times -> (B, N, N, 2) features of both directions, log-scaled."""
    return torch.stack([torch.log1p(dist), torch.log1p(dist.transpose(1, 2))], dim=-1)


class AttentionPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        D = cfg.embed_dim
        if cfg.depot_encoding not in ("xy", "polar"):
            raise ValueError(f"unknown depot_encoding {cfg.depot_encoding!r}")
        self.embed_depot = nn.Linear(2 if cfg.depot_encoding == "xy" else 3, D)
        # x, y, demand [+ has_window, window start, window end, service]
        self.embed_node = nn.Linear(7 if cfg.time_features else 3, D)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.num_layers))
        self.project_nodes = nn.Linear(D, 3 * D, bias=False)  # glimpse K, V and logit K
        self.project_graph = nn.Linear(D, D, bias=False)
        # current node, first node, capacity left [+ current time]
        self.project_step = nn.Linear(2 * D + (2 if cfg.time_features else 1), D, bias=False)
        self.project_out = nn.Linear(D, D, bias=False)
        if cfg.matrix_bias:
            self.edge_bias = _pair_mlp(cfg.num_heads)  # encoder + glimpse attention, per head
            self.step_bias = _pair_mlp(1)  # next-node logits, from the current node

    def travel_times(self, batch: Batch) -> Tensor:
        """The real matrix if the batch has one, else straight-line distances."""
        if batch.dist is not None:
            return batch.dist
        return torch.cdist(batch.locs, batch.locs)

    def node_features(self, batch: Batch) -> Tensor:
        """(B, N, 3|7) customer inputs. Times are log-scaled: windows (hundreds of units) dwarf
        tour lengths (tens). A batch without times reads as 'no window, no service'."""
        base = torch.cat([batch.locs[:, 1:], batch.demand[:, 1:, None]], dim=-1)
        if not self.cfg.time_features:
            return base
        B, N = batch.size, batch.num_customers
        if batch.tw is None or batch.service is None:
            tw = torch.tensor([0.0, math.inf], device=base.device).expand(B, N, 2)
            service = torch.zeros(B, N, device=base.device)
        else:
            tw, service = batch.tw[:, 1:], batch.service[:, 1:]
        has_window = torch.isfinite(tw[..., 1]).float()
        start = torch.log1p(tw[..., 0].clamp(max=TIME_CAP))
        end = torch.log1p(tw[..., 1].clamp(max=TIME_CAP))
        extra = torch.stack([has_window, start, end, torch.log1p(service)], dim=-1)
        return torch.cat([base, extra], dim=-1)

    def depot_features(self, batch: Batch) -> Tensor:
        """(B, 1, 2|3) input features of the depot."""
        if self.cfg.depot_encoding == "xy":
            return batch.locs[:, :1]
        offset = batch.locs[:, :1] - batch.locs[:, 1:].mean(dim=1, keepdim=True)
        dist = offset.norm(dim=-1, keepdim=True)
        direction = offset / dist.clamp(min=1e-9)
        return torch.cat([direction, torch.log1p(dist)], dim=-1)

    def encode(self, batch: Batch, edge_bias: Tensor | None = None) -> Tensor:
        depot = self.embed_depot(self.depot_features(batch))
        nodes = self.embed_node(self.node_features(batch))
        h = torch.cat([depot, nodes], dim=1)
        for layer in self.layers:
            h = layer(h, edge_bias)
        return h

    def matrix_biases(self, batch: Batch) -> tuple[Tensor, Tensor] | tuple[None, None]:
        """(edge (B, H, N+1, N+1), step (B, N+1, N+1)) biases from travel times, or (None, None)."""
        if not self.cfg.matrix_bias:
            return None, None
        feats = pair_features(self.travel_times(batch))
        edge = self.edge_bias(feats).permute(0, 3, 1, 2).contiguous()
        return edge, self.step_bias(feats).squeeze(-1)

    def forward(
        self,
        batch: Batch,
        num_starts: int,
        decode: str = "sample",
        forced_starts: bool = True,
        tw_penalty: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build ``num_starts`` tours per instance.

        Returns ``(cost (B, P), log_prob (B, P), actions (B, P, T))`` where cost is the tour
        length plus ``tw_penalty`` x lateness (just the length without windows). With
        ``forced_starts`` trajectory p starts at customer p + 1 (POMO); otherwise the first move
        is decoded too.
        """
        cfg = self.cfg
        B, n1, D, H = batch.size, batch.num_customers + 1, cfg.embed_dim, cfg.num_heads
        P = num_starts
        edge_bias, step_bias = self.matrix_biases(batch)
        h = self.encode(batch, edge_bias)
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
            context = [cur, start, (1.0 - env.used).unsqueeze(-1)]
            if cfg.time_features:
                context.append(torch.log1p(env.time).unsqueeze(-1))
            q = graph + self.project_step(torch.cat(context, -1))  # (B, P, D)
            qh = q.view(B, P, H, D // H).transpose(1, 2)  # (B, H, P, dk)
            if edge_bias is None or step_bias is None:
                attn: Tensor = ~mask[:, None]
            else:  # each trajectory's row of the edge bias: from its current node to every node
                rows = edge_bias.gather(2, env.cur[:, None, :, None].expand(B, H, P, n1))
                attn = rows.masked_fill(mask[:, None], float("-inf"))
            glimpse = F.scaled_dot_product_attention(qh, gk, gv, attn_mask=attn)
            g = self.project_out(glimpse.transpose(1, 2).reshape(B, P, D))
            logits = torch.einsum("bpd,bnd->bpn", g, lk) / math.sqrt(D)
            if step_bias is not None:
                logits = logits + step_bias.gather(1, env.cur[..., None].expand(B, P, n1))
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

        cost = env.length + tw_penalty * env.lateness if tw_penalty else env.length
        return cost, log_prob, torch.stack(actions, dim=-1)
