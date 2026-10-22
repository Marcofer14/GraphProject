# Roadmap · phase 3 (weeks 5–6) — ✅ closed 2026-10-22

**Outcome** (ADR-016): on 200 unseen real Amazon routes the hybrid (100-stop Amazon model + local
search) cuts 9 % of driving time against real drivers and beats OR-Tools (10 s) in 0.9 s; the
network alone is weak on real streets, which sets the first goal of phase 4.

**Carried into phase 4:** matrix-aware encoder (MatNet-style) · fine-tuning on real route slices ·
larger validation sets and lower learning rate for fine-tuning stages · retrain the time-aware
model with realistic window tightness and a lower penalty · faster decoding (CUDA Graphs /
`torch.compile`) · ONNX export · driver-similarity metric · neural station-day assignment.

Long jobs (GPU / wall-clock) start first so the GPUs are never idle while code is written.

```mermaid
flowchart LR
    classDef long fill:#FAECE7,stroke:#993C1D,color:#712B13
    classDef quick fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef user fill:#F1EFE8,stroke:#5F5E5A,color:#444441

    L1["CVRP-50 to 100 epochs<br/>local GPU · ~2 h"]:::long
    L2["CVRP-100 from scratch<br/>Colab T4 · ~3 h"]:::long
    Q1["Local search 2-opt / or-opt<br/>code · ~2 h"]:::quick
    Q2["Calibrated generator<br/>code · ~2 h"]:::quick
    Q3["Time windows in env<br/>code · ~2 h"]:::quick
    Q4["Time windows in model<br/>code · ~1 h"]:::quick
    R["Publish release v0.2.x<br/>you · 3 min"]:::user
    L3["Train CVRPTW-50<br/>GPU · ~2 h"]:::long
    L4["Size curriculum 50 → 150<br/>GPU · one night"]:::long
    L5["Neural solver on 200<br/>real Amazon routes"]:::long

    L1 --> R
    Q2 --> L3
    Q3 --> L3
    Q4 --> L3
    L2 --> L4
    L3 --> L4
    L4 --> L5
    Q1 --> L5
```

| Item | Blocked by | Unblocks | Runs on |
|---|---|---|---|
| CVRP-50 → 100 epochs | nothing (resumes from checkpoint) | better release weights | local GPU (chain `D`) |
| CVRP-100 | nothing | warm start for the curriculum | Colab T4 (`configs/cvrp100.yaml`) |
| Local search ✅ (CVRP-50 gap 1.84 % → 0.62 %) | nothing | +quality for every neural model | CPU |
| Calibrated generator ✅ (matches real slices, notebook 02) | nothing | realistic training data | CPU |
| Customer-centric normalisation ✅ (tie with old one at 50 stops; adopted, ADR-012) | — | training on the real distribution | done |
| Early stopping on controllable length ✅ (ADR-013) | — | fair stopping on Amazon-like data | done |
| Time windows ✅ (env + model + inference + benchmark, ADR-014) | — | CVRPTW training | done |
| Train CVRPTW-50 ✅ (best epoch 41, early-stopped at 49) | — | curriculum with windows | done |
| Curriculum ready ✅ (`cvrp100/150_amazon` from `cvrp50_pomo_amazon`, ADR-015; VS Code task, chain `E`) | — | real-size model | GPU, ~6 h |
| Real-route benchmark with neural solvers ✅ (`nr-bench --mode route ... attention_ls@ckpt`, controllable gap) | — | the phase-3 headline | ready |
| Time windows (env + model) | nothing | CVRPTW training | CPU, then GPU tests |
| CVRPTW-50 | generator + time windows | curriculum with windows | GPU |
| CVRP-100 ✅ (1.34 % with local search; stopped at epoch 55) | — | warm start for the curriculum | done |
| Size curriculum | CVRPTW-50 (+ normalisation experiment) | real-size model | GPU, overnight |
| Amazon test | curriculum + local search | the headline number of phase 3 | GPU + CPU |
