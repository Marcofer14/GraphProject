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

## ADR-010 · Local search on a giant tour, never worse than its input (2026-10-21)
Neural construction leaves small, local defects (a crossing, a stop on the wrong route) that a
classic local search removes in milliseconds. `solvers/local_search.py` runs 2-opt, or-opt (runs
of 1–3 stops, forwards or reversed, any route) and swap on one *giant tour* array with depot
separators, so intra- and inter-route moves share one numba-compiled code path. 2-opt is exact on
asymmetric matrices via prefix sums of forward and backward arc costs, so it is ready for the
Amazon matrices. Moves only see travel time; `polish` therefore rejects a result that adds
lateness or travel, which makes it safe to bolt onto any solver. `attention_ls` polishes the 8 best
decoded candidates (diversity matters more than polishing one), and `savings_ls` /
`nearest_neighbor_ls` exist as controls to separate the network's contribution from the search's.

## ADR-011 · Generator calibrated per van route, not per station-day (2026-10-21)
The target the neural solver must reach is one van's day (~150 stops, ~20 zones), so the
generator (`ml/generators.py`, parameters in `configs/generator/amazon.json`, fitted by
`data/calibration.py` on all 6,112 routes) reproduces a route: a station ~17 km from the stops,
zone centres spread ~1 km, stops ~0.2 km around their zone centre, real stop volumes and service
times, and the real share/shape of time windows. Spreads are fitted as 2-D RMS distances and
converted to per-axis standard deviations; the zone spread scales with sqrt(N / 150) because a
smaller instance covers fewer, neighbouring zones. `notebooks/02_generator.ipynb` checks the result
against real route slices: depot distance, customer spread, nearest-neighbour distance, load and
window statistics all match within ~15 %.

Two findings shape what comes next:
* **Capacity barely binds on real routes** (150 stops fill ~74 % of a van) and most windows are
  8 h wide. The real problem is a large, clustered TSP with a long stem. `data.stops_per_vehicle`
  produces multi-van instances when the CVRP structure is wanted.
* **Normalising with the depot squashes the customers.** With the station ~5x farther than the
  delivery area is wide, customers occupy ~3 % of the unit square (median nearest-neighbour
  distance 0.006 vs 0.076 for uniform data). Before training on this distribution the input
  normalisation should be customer-centric (scale by the customers' bounding box, encode the
  depot by direction and distance). Open item for phase 3.

## ADR-012 · Customer-centric normalisation with a polar depot (2026-10-21)
Resolves the open item of ADR-011. `data.normalisation: customers` scales coordinates by the
customers' bounding box, so they fill the unit square and the depot may lie far outside it (e.g.
at x = 10). It is a pure rescaling: distances keep their ratios, so tours, the environment and the
x8 augmentation are unaffected. Because raw depot coordinates can then be large, `model.
depot_encoding: polar` feeds the depot as (cos, sin, log(1 + distance)) from the customer centroid,
which stays bounded for any distance. Both settings are stored in the checkpoint; the neural solver
reads them back, and checkpoints without them keep the old behaviour (`all`, `xy`).

At inference, real instances carry (lat, lng): they are now projected to km (equirectangular)
before normalising, since a degree of longitude is ~33 % shorter than a degree of latitude at
Seattle's latitude and the old code treated them as equal.

Expected effect, to be measured: `cvrp50_amazon` (customers + polar) against `cvrp50_amazon_all`
(old normalisation) and the uniform-trained model on Amazon-like instances
(`nr-bench-synth --distribution amazon`, VS Code task "Experiment: compare ...").

## ADR-006 · Data and virtualenv outside OneDrive
The repository lives in a OneDrive-synced folder. `.venv/` and `data/` are directory junctions to
`%USERPROFILE%\neural-routing-data\` so gigabytes of data and packages never sync. Both paths are
git-ignored; on other machines they are ordinary folders.
