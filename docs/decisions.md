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

**Result (2026-10-21, 200 Amazon-like CVRP-50 instances).** ~78 % of every tour is the
unavoidable depot round trip, so gaps are reported on the *controllable* part (travel minus
2 x depot-to-nearest-customer); total-length gaps are ~4x smaller and hide the differences.

| Model | Gap (controllable) | + local search |
|---|---|---|
| Customer-centric + polar (A) | 3.56 % | 0.37 % |
| Depot-inclusive + xy (B) | 3.68 % | 0.34 % |
| Uniform-trained model | 34.87 % | — |
| Savings | 14.96 % | — |

A vs B is a statistical tie (A better on 55 % of instances; mean difference 0.11 ± 0.24 points,
95 % CI). **Decision: adopt customer-centric + polar** for all further Amazon-like training — equal
quality at 50 stops and it does not degrade as the customer area shrinks relative to the depot
distance, which gets worse at 150 stops. The real finding is the *distribution*: the
uniform-trained model is 10x worse on realistic instances than either calibrated model, worse
than savings. Caveat: both runs were stopped early by the threshold issue fixed in ADR-013, so
neither reached its ceiling.

## ADR-013 · Early stopping on the controllable length (2026-10-21)
The 0.1 % relative threshold was applied to the full validation length. On Amazon-like data the
depot round trip is 74–81 % of it and does not depend on the policy, so the threshold effectively
demanded 3.8–5.3x more relative improvement on the part the policy controls, and both ablation
runs stopped early. Early stopping now judges `val - unavoidable_cost(val)`, where the unavoidable
cost is the mean of 2 x depot-to-nearest-customer (a true lower bound on every tour). On uniform
data the correction is negligible (0.15 of 10.65).

## ADR-014 · Soft time windows in the neural solver (2026-10-21)
Implements `docs/design/time_windows.md`. `Batch` optionally carries service times and windows in
normalised time units (speed 1); `RoutingEnv` keeps a per-vehicle clock (reset at each depot
visit) and accumulates lateness with exactly the evaluator's semantics, which a test checks on
random rollouts with tight windows (TSP and multi-vehicle CVRP). The policy returns
`length + tw_penalty x lateness` as its cost; with `model.time_features` it also reads
(has window, log start, log end, log service) per node and log(current time) in the decoder
context, while models without it still load and run. At inference the instance's speed is fitted
by least squares (matrix seconds vs normalised Euclidean distance) to convert real windows, and
candidates are chosen on `travel + tw_penalty x lateness`. Training uses windows shrunk to 25 %
(`data.tw_tightness`) because real ones rarely bind: under a naive visiting order only 8 % of
instances are late at 1.0, 65 % at 0.25. Smoke run: validation cost 51.3 -> 32.0 in 80 steps.

## ADR-015 · Size curriculum without time-window inputs (2026-10-22)
The CVRPTW-50 model (time features, trained with windows shrunk to 25 % and a 10x lateness
penalty) was compared with the window-blind Amazon CVRP-50 model on 200 Amazon-like instances with
those tight windows. Every good solver is late at only 0.1–0.2 % of stops; the time-aware model
halves that (0.21 % -> 0.11 %) but nearly triples the distance gap (0.64 % -> 1.68 % with local
search). On 10 real routes it was likewise worse in distance (11.6 % vs 5.1 % controllable gap)
with no lateness gain. It learned to be over-cautious for windows that almost never bind.

With the real window widths (`tw_tightness` 1.0) the picture is the same: window-blind + local
search 0.46 % controllable gap and 0.02 % late stops; time-aware 1.58 % and 0.01 %; OR-Tools (10 s)
0.82 % and 0.01 %; PyVRP 0.33 %. The window-blind hybrid beats OR-Tools in 0.2 s.

**Decision:** the curriculum (`cvrp100_amazon` -> `cvrp150_amazon`, `train.init_from`) starts from
the window-blind `cvrp50_pomo_amazon`. Real windows are still honoured at inference: candidates are
chosen on travel + 10 x lateness and local search never accepts added lateness; the final
real-route benchmark reports `late_stops_%` to verify it. A model with time inputs cannot inherit
weights from one without them (the input layer changes), so mixing is not an option; the
time-aware model is kept for phase 4 (e.g. retrain with realistic tightness and a lower penalty).
`init_from` compares model configs with defaults filled in, so checkpoints saved before an
option existed remain valid starting points.

## ADR-016 · Size curriculum and the real-route evaluation that closes phase 3 (2026-10-22)
Curriculum: `cvrp50_pomo_amazon` → `cvrp100_amazon` (25 epochs, 50 min, controllable validation
length −0.9 %) → `cvrp150_amazon` (12 epochs, 33 min, −0.5 %, early-stopped). Both stages started
close to their final quality: the 50-stop model trained on the calibrated generator already
transferred to larger sizes. The 150 stage's validation curve is noise (±0.2 %) around a flat line,
which says the fine-tuning stages needed a larger validation set (200 instances was too few at
150 stops) and a lower learning rate; recorded for phase 4.

Final evaluation, 200 routes from the *evaluation* split, real matrices and windows:
the 100-stop model + local search is best among neural variants (4.32 % controllable gap, 0.9 s),
ahead of OR-Tools with 10 s (4.98 %) and the real drivers (22.2 %), behind PyVRP with 10 s
(0.36 %). The 150-stop model (5.88 %) is no better than the 50-stop one (5.96 %). **The network
alone reaches only 37.9 %** on real routes against ~3.6 % on Amazon-like synthetic data, so most
of the real-road quality comes from local search on the true matrix: the policy's Euclidean view
of an asymmetric street network is the main limitation, and the reason phase 4 starts with a
matrix-aware encoder. The 100-stop model is the phase-3 reference model.

## ADR-017 · Matrix-aware attention and training on real route slices (2026-10-22)
Why the network alone reaches 38 % on real routes (vs 3.6 % on synthetic): on 100 evaluation
routes, straight-line and real travel times correlate well overall (0.89) but only **67 % of each
stop's 5 nearest neighbours agree**, and the shortest legs take **2.2x** what straight-line
distance predicts (a ~92 s fixed cost per leg: turning, parking, one-way streets). On real slices
the median leg costs 2.6x its straight-line length. Choosing "the next stop" from coordinates is
wrong about a third of the time.

Changes:
* `Batch.dist` (optional real travel times, normalised like the coordinates); the environment
  charges `dist[i, j]` per leg; tested against `core.evaluate` on asymmetric matrices.
* `model.matrix_bias`: small MLPs on (log d_ij, log d_ji) give a per-head additive bias to every
  encoder attention and to the decoder glimpse (row of the current node), plus a bias on the
  next-node logits. Their last layers are **zero-initialised**: a fresh matrix model produces
  exactly the same tours as the coordinate model it is loaded from (tested), so `init_from` can
  warm-start from the phase-3 reference model and training only learns what the matrix adds.
* `ml/real_data.RealRouteSampler`: coherent slices of real training routes with their real
  sub-matrices (and optionally windows), 87 ms per 32 x 100 batch; routes are split 90/10 by a
  stable hash into training / validation, the evaluation split stays untouched.
* Inference passes the instance's matrix in the training unit (calibrated speed x box size).
* Config `cvrp100_real_matrix` (from `cvrp100_pomo_amazon`, lr 5e-5, 500 validation slices).

**Result** (40 epochs, ~1 h 45 min, validation on held-out real slices 38.09 → 33.40 and still
falling; 200 evaluation routes): network alone **38.09 % → 10.36 %** controllable gap; network +
local search **4.33 % → 2.86 %** (OR-Tools 10 s: 5.05 %, PyVRP 10 s: 0.38 %, drivers: 22.22 %),
driver score unchanged (0.07), late stops 0.01 %, 0.5 s per route. A control run that fine-tuned
the coordinate-only model on synthetic data with a lower learning rate got slightly *worse* on real
routes (4.86 %): the gain comes from the matrix and the real data, not from more training.
`cvrp100_pomo_real` is the phase-4 reference model.

*Update: 20 more epochs* (config `cvrp100_real_matrix_long`, resumed from the 40-epoch state
with `train.continue_from` under its own name; ~1 h; stopped at epoch 60, validation 33.40 →
33.22 and still slowly improving). Same 200 routes, against the best of every run: network alone
10.7 % → 9.7 %, first answer 2.97 % → 2.74 %, late stops unchanged. `cvrp100_pomo_real_long` is
the reference model from v0.5.0 on; the demo uses it.

## ADR-018 · Driver-similarity score (the Amazon challenge metric) (2026-10-22)
Travel time alone cannot tell whether a shorter route is *reasonable*. `core/driver_score.py`
re-implements the challenge's scoring rule: sequence deviation (how far each next stop jumps in
the driver's order, normalised to [0, 1]) times ERP per edit (edit distance where substituting
two stops costs their z-normalised, min-shifted travel time and an insert/delete costs 1000).
0 means the driver's sequence; the rule ignores direction (the reversed route scores 0). Every
benchmark row now carries `driver_score` and the solution's routes as JSON, so new metrics can be
computed later without re-solving.

First measurement, 20 evaluation routes: PyVRP 0.08, attention + local search 0.09, savings + LS
0.09, OR-Tools 0.10, nearest neighbour 0.12. The hybrid is as driver-like as the classic
optimisers, but no travel-time optimiser is close to the drivers, who deliver zone by zone even
when it costs time. A model that trades a little travel time for driver-like zone order is a
natural extension (imitation of the zone sequence), left for later.

## ADR-019 · Faster inference: measure first, then two exact changes (2026-10-22)
Profiling one real 165-stop route showed the network was *not* the main cost: scoring the 1,320
decoded candidates one by one in Python took 51 % of the 1.3 s, the network 42 %, local search 7 %.
Two changes, both exact (tested: identical tours and identical chosen solutions):
* **Vectorised candidate scoring** (`candidate_objectives`): travel and lateness of all candidates
  in one pass of numpy over the decoding steps, matching `core.evaluate` to 1e-9. An intermediate
  version re-ranked only the 32 shortest candidates by lateness; it looked *better* on travel time
  because it silently traded punctuality for distance, so it was replaced by the exact version.
* **Fewer GPU -> CPU syncs while decoding**: "are all tours finished?" is asked every 8 steps
  instead of every step, letting the CPU queue kernels ahead; extra steps are no-ops.

Result on 10 real routes: network 953 -> 356 ms (2.7x), network + local search 1116 -> 442 ms
(2.5x), same answers; a 32 x 100 training step 0.81 -> 0.68 s. `torch.compile` / CUDA graphs were
not needed for the demo's latency target and are left out (fragile on Windows).

## ADR-020 · Whole station-days: cluster first, route with the network (2026-10-22)
`solvers/cluster.py` plans a station-day for the real fleet: a capacitated k-means (k-means++,
regret-ordered assignment that never exceeds a van's volume) splits the stops into one group per
van, each group is routed as a one-van instance by the neural hybrid on its travel-time
sub-matrix, and the multi-route local search then moves stops between vans when it helps.
`cluster_savings_ls` (same clustering and polish, savings per group) is the control.

Same 6 station-days as phase 1 (≤ 1,200 stops, ~5 vans, cross-route times estimated, ADR-005):

| Solver | Fleet hours / day | Gap | Late stops | Runtime |
|---|---|---|---|---|
| Cluster + network with real matrix (ADR-017) | **13.43** | 0.77 % | 0.27 % | 2 s |
| Cluster + phase-3 network | 13.57 | 2.13 % | 0.44 % | 2.7 s |
| Cluster + savings/LS (control) | 13.64 | 2.64 % | 0.97 % | 0.6 s |
| PyVRP (60 s) | 14.26 | 9.69 % | 0 % | 60 s |
| Real drivers | 15.34 | 17.39 % | 0.10 % | — |

The day plan saves **12.5 % of fleet hours against the drivers** with the same vans in 2 s. Most of
the advantage over PyVRP comes from the decomposition (PyVRP does not converge on ~1,000 stops in
60 s; the control already beats it); the network adds 1.5 % fewer hours and less than half the
late stops of the control, and the matrix-aware model beats the phase-3 one. Six days is a small
sample and lateness is slightly above the drivers', so these numbers are indicative.

*Update: 20 unseen days.* The six days above come from the training split, and the real-matrix
model trained on slices of those very routes. Re-run on 20 days of the evaluation split (no model
saw them; `nr-bench --min-routes 3 --max-stops 1200`, 3–6 vans, ~600 stops, all five cities):

| Solver | Fleet hours / day | Gap | Late stops | Better than ours on |
|---|---|---|---|---|
| Cluster + network, 60 epochs | **11.88** | 0.84 % | 0.16 % | — |
| Cluster + network, 40 epochs | 11.88 | 0.90 % | 0.23 % | 8 of 20 days |
| Cluster + savings/LS (control) | 12.04 | 2.31 % | 0.55 % | 2 of 20 days |
| PyVRP (60 s) | 12.45 | 4.41 % | 0 % | 10 of 20 days |
| Savings | 12.88 | 10.07 % | 0.75 % | 0 of 20 days |
| Real drivers | 13.18 | 12.68 % | 0.18 % | 0 of 20 days |

**−9.9 % fleet hours against the drivers**, fewer on all 20 days and with fewer late stops than
the drivers; this replaces the −12.5 % above as the headline. PyVRP is behind on average but wins
half the days, the smaller ones it can solve in a minute. On these days the vans' areas are far
apart: a median 0.2 % of stops change van and 8 days keep the drivers' groups exactly, so the
saving comes mostly from the order inside each van, not from re-assigning stops (the demo shows
this share per day).

## ADR-021 · Anytime hybrid: iterated local search until the time limit
**Context.** At equal computing time PyVRP beat the hybrid on average (0.5 s: 2.23 % vs 2.86 %
controllable gap), because the hybrid stops at its first local optimum after ~0.5 s while PyVRP
keeps searching. Comparing a one-shot method with anytime solvers also made the race unfair in
both directions.

**Decision.** `attention_ils` runs the hybrid, then repeats until the time limit: a local
double-bridge kick (three cuts inside a 50-position window, reconnected A C B, which 2-opt and
or-opt cannot undo in one move), the same 2-opt / or-opt / swap descent, and acceptance only if
the result is better (less capacity excess, then travel + 10 x lateness, the neural ranking).
No training; the network supplies a strong start and the search uses the rest of the budget.

**Result** (200 unseen real routes, gaps against the best route of every run so far):

| Budget | Hybrid + ILS | PyVRP | Hybrid better on | OR-Tools |
|---|---|---|---|---|
| 1 s | **1.53 %** | 1.66 % | 123 of 200 routes (70 worse) | 9.23 % |
| 2 s | **1.07 %** | 1.33 % | 133 of 200 routes (59 worse) | 8.15 % |

At equal time the hybrid is now ahead of PyVRP on average, on the 90th percentile and on most
routes; with 2 s it wins as many routes as PyVRP given 10 s (93 vs 96), though PyVRP's 10 s
average is still better (0.70 %). Caveat: the network's ~0.4 s runs on a GTX 1660 Super GPU,
while the search and both classical solvers use one CPU core each.

*Update: 60-epoch model and fair timing.* A later evaluation task ran PyVRP 1 s next to six
worker processes and the GPU model, and PyVRP got clearly worse (2.25 % instead of 1.60 %): a
1-second budget is sensitive to machine load. Every number below comes from runs made without
other load (both ours and PyVRP's), with `cvrp100_pomo_real_long`:

| Budget | Hybrid + ILS | PyVRP | Hybrid better on | OR-Tools |
|---|---|---|---|---|
| 1 s | **1.19 %** | 1.60 % | 122 of 200 routes | 9.17 % |
| 2 s | **0.77 %** | 1.28 % | 128 of 200 routes | 8.09 % |

PyVRP given 10 s: 0.70 %; against our 2 s run it is better on 93 routes, worse on 94, tied on 13
(that run comes from the loaded evaluation; a 10-second budget is far less sensitive to it).

## ADR-006 · Data and virtualenv outside OneDrive
The repository lives in a OneDrive-synced folder. `.venv/` and `data/` are directory junctions to
`%USERPROFILE%\neural-routing-data\` so gigabytes of data and packages never sync. Both paths are
git-ignored; on other machines they are ordinary folders.

*Update 2026-10-21:* a CVRP-100 run died at epoch 56 when OneDrive locked TensorBoard's event file
mid-write (`PermissionError`). `runs/`, `checkpoints/` and `logs/` are now junctions too, and the
trainer no longer trusts the filesystem: TensorBoard goes through `SafeWriter` (the first error
disables logging instead of raising) and checkpoints/history are written with retries
(`_retry_io`). Resume made the crash cost only the unfinished epoch.
