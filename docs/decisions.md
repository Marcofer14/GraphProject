# Architecture decision records

Short, dated records of the non-obvious choices. Newest last.

## ADR-001 · One evaluator for every solver (2026-10-20)
Each solver optimises its own internal objective (OR-Tools: integer arc costs + soft-window
penalties; PyVRP: distance + time-warp penalties). Comparing those numbers would be meaningless,
so every solution is re-scored by `core.evaluate`, which works on the original float matrix.
**Consequence:** a new solver (including the neural ones) is comparable the moment it returns routes.

## ADR-002 · Soft time windows
Real Amazon routes violate windows and some stop windows are unreachable, so hard windows would
make many instances infeasible. Arriving early means waiting; arriving late is allowed and reported
as `tw_violations` / `lateness_s`. The primary objective stays **total travel time**.

## ADR-003 · Stop-level windows from package windows
A stop can hold several packages, each with its own window (most have none). The stop window is
the intersection of its packages' windows; when that is empty (rare), the window of the
earliest-ending package is used, as that package forces the earliest service.

## ADR-004 · Matrix storage
The raw travel-time file is a 1.7 GB nested JSON. The ETL streams it with `ijson` and writes all
matrices as one row-major `float32` file plus an offset per route, so any route loads in O(1)
through a memory map without parsing anything. Node 0 is always the station; the remaining stops
follow in stop-id order, which `stops.parquet` records as `node`.

## ADR-005 · Station-day pooling needs a travel-time model
To study fleet assignment we pool all routes of a station and day into one CVRPTW. Amazon only
publishes travel times *within* a route, so cross-route pairs are estimated with a linear model on
haversine distance fitted to the within-route pairs (`TravelTimeModel`). Real times are kept
wherever they exist. Time windows stay relative to each route's own departure, and capacity is the
median of the routes' vans: both are approximations to revisit when OSRM matrices arrive (phase 5).

## ADR-007 · One environment for TSP and CVRP (2026-10-20)
TSP is modelled as CVRP with zero demand plus one masking rule: the depot may only be re-entered
once every customer is served. One GPU environment, one policy class and one solver wrapper then
cover both problems, and time windows (phase 3) extend the same code path. Side effect: a "TSP-20"
tour here has 21 nodes (depot + 20), so its optimum sits a little above the classic TSP-20
figure of 3.83; gaps are always measured against PyVRP on the same instances, never against
numbers from papers.

## ADR-008 · POMO shared baseline by default, greedy rollout kept for comparison
Both REINFORCE baselines are implemented (`train.baseline`). POMO needs no frozen critic, no
extra validation rollouts and gives N correlated samples per instance, which is far more
sample-efficient on a single consumer GPU. The Kool et al. rollout baseline (`tsp20_rollout`
config) is kept as an ablation. The rollout baseline swaps in the new policy when its greedy
validation mean improves; the paper's paired t-test is omitted.

## ADR-009 · The neural solver picks its answer with the real matrix
The policy reasons on rescaled Euclidean coordinates. At inference it proposes up to 8 x N
candidate tours (augmentations x POMO starts); each is scored by `core.evaluate` on the
instance's own travel-time matrix and the best is returned. On synthetic data this is identical
to the model's own choice; on asymmetric road networks it keeps the comparison honest.

## ADR-006 · Data and virtualenv outside OneDrive
The repository lives in a OneDrive-synced folder. `.venv/` and `data/` are directory junctions to
`%USERPROFILE%\neural-routing-data\` so gigabytes of data and packages never sync. Both paths are
git-ignored; on other machines they are ordinary folders.
