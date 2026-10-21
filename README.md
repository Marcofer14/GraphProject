# Neural Last-Mile Routing

Deep reinforcement learning solvers for vehicle routing with capacities and time windows (CVRPTW),
benchmarked against OR-Tools, PyVRP and **real delivery drivers** on ~9,000 Amazon last-mile routes.

> **The case.** *Ruta Norte Logistics* (fictional) runs last-mile delivery vans out of several
> stations. Its historical operations are the public
> [Amazon Last Mile Routing Research Challenge](https://registry.opendata.aws/amazon-last-mile-challenges/)
> dataset: real stops, packages, time windows, travel-time matrices and the sequence each driver
> actually drove. Three questions drive the project:
>
> 1. Can we sequence each van's stops better than its driver did?
> 2. How many vans and hours do we save by re-assigning a station's whole day from scratch?
> 3. Can we re-plan in milliseconds when an urgent order arrives or a van breaks down?

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Repo, tooling, CI, data model, ETL, EDA | ✅ |
| 1 | Baselines (NN, Clarke-Wright, OR-Tools, PyVRP) + evaluator + benchmark | ✅ |
| 2 | Attention Model + POMO (REINFORCE) on TSP/CVRP, neural solver, synthetic benchmark | ✅ code · ⏳ CVRP-50 training |
| 3 | Time windows + calibrated generator + curriculum | |
| 4 | Real asymmetric travel times (MatNet-style), fine-tuning, ONNX | |
| 5–7 | FastAPI + OSRM backend, Next.js/deck.gl frontend, deployment | |

## Baseline results

See [`results/`](results/) for the generated tables. Gap is relative to the best solution found
for each instance; every solver is scored by the same evaluator (`core/evaluate.py`).

<!-- results:start -->
_Run `uv run nr-bench` (see below) to fill this table._
<!-- results:end -->

## Quick start

```bash
# 1. environment (Python 3.12 via uv)
uv sync

# 2. data: ~3.2 GB from the public S3 bucket, resumable
uv run python scripts/download_amazon.py --split all

# 3. JSON -> Parquet + memory-mapped matrices (~10 min for training)
uv run nr-etl --split all

# 4. benchmark
uv run nr-bench --mode route --n 200 --configs reference nearest_neighbor savings ortools:1 ortools:10 pyvrp:1 pyvrp:10
uv run nr-bench --mode station-day --n 5 --max-stops 1500 --configs reference savings ortools:60 pyvrp:60

# 5. train the neural solvers (Hydra configs in configs/; resumes automatically)
uv run nr-train --config-name tsp20
uv run nr-train --config-name cvrp50 train.epochs=60
tensorboard --logdir runs

# 6. neural vs classic on the training distribution
uv run nr-bench-synth --problem cvrp --size 20 --configs savings ortools:1 pyvrp:5 attention

# tests, lint, types
uv run pytest && uv run ruff check . && uv run mypy
```

Long jobs (training + benchmarks) can run as detached chains that survive closing the terminal,
logging to `logs/chain_<X>.log`; see [`scripts/jobs.ps1`](scripts/jobs.ps1):

```powershell
powershell -File scripts/start_jobs.ps1 A B C
```

Use any solver from Python:

```python
from neural_routing.core import evaluate
from neural_routing.data.amazon import AmazonDataset
from neural_routing.solvers import get_solver

ds = AmazonDataset("training")
inst = ds.route_instance(ds.routes.index[0])  # one van, real matrix, driver's sequence
for name in ["reference", "savings", "pyvrp"]:
    print(name, evaluate(inst, get_solver(name).solve(inst, time_limit=2)).travel_time_s)
```

## Layout

```
src/neural_routing/
  core/        VRPInstance, Solution, evaluate()  - the contract every solver follows
  data/        Amazon ETL, dataset loader (route + station-day views), synthetic instances
  solvers/     reference (driver), nearest neighbour, Clarke-Wright, OR-Tools, PyVRP, attention
  ml/          GPU routing env (TSP/CVRP), attention policy, REINFORCE trainer (POMO / rollout)
  eval/        Amazon and synthetic benchmark CLIs
configs/       Hydra training configs (tsp20, cvrp20, cvrp50, tsp20_rollout)
scripts/       dataset download
notebooks/     01_eda.ipynb
docs/          decisions.md (architecture decision records)
results/       benchmark tables
tests/
```

Design decisions and their trade-offs are recorded in [`docs/decisions.md`](docs/decisions.md).

## Data license

The Amazon Last Mile Routing Research Challenge data is © Amazon.com, Inc., released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). It is downloaded at setup time and
never committed. *Ruta Norte Logistics* is fictional and unaffiliated with Amazon.
