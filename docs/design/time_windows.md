# Design · time windows in the neural solver (phase 3, step 4)

Status: **implemented** 2026-10-21 (ADR-014), all five steps; training run pending.
Goal: train and run the policy on CVRPTW with the same *soft*
window semantics the evaluator already uses (ADR-002), so neural, classic and real-driver
solutions stay comparable.

## What the data says (ADR-011)
* 6.6 % of stops have a window; most are 7–8 h wide and open near departure.
* At ~35 km/h a whole 150-stop route takes a few hours, so on real routes windows rarely bind.
* Consequence: the model must *handle* windows correctly, but most of the learning signal will
  come from routing. A `data.tw_tightness` knob (shrink windows by a factor) lets training see
  binding windows often enough to learn the behaviour.

## Units
Everything stays in **normalised distance units**, with speed 1: driving one unit takes one time
unit. Generated and real windows are converted with `seconds_per_unit` (generator: travel speed x
km per unit; real instances: least-squares fit of the travel-time matrix against normalised
Euclidean distance, `s = sum(D * E) / sum(E^2)`).

## 1. Environment (`ml/env.py`)
| Change | Detail |
|---|---|
| `Batch` | optional `service (B, N+1)` and `tw (B, N+1, 2)`; `to()` and `augment8` carry them (time is invariant under the 8 symmetries) |
| State | `time (B, P)`: clock of the current vehicle; reset to 0 at every depot visit (new route), exactly like `core.evaluate` |
| `step` | `arrival = time + dist`; `start = max(arrival, tw_start)` (wait if early); `late += relu(start - tw_end)`; `time = start + service` |
| Mask | unchanged (soft windows). Optional later: `hard_tw` masks customers that are already unreachable |
| Objective | `env.length` stays pure distance; new `env.lateness`; training cost = `length + lambda * lateness` (`train.tw_penalty`) |
| **Key test** | random rollouts: env lateness == `core.evaluate(...).lateness_s` / `seconds_per_unit`, as for length today |

## 2. Model (`ml/model.py`)
| Change | Detail |
|---|---|
| `ModelConfig.time_features: bool = False` | off by default, so every existing checkpoint loads unchanged |
| Node input | `(x, y, demand)` + `(has_tw, log1p(tw_start), log1p(min(tw_end, cap)), log1p(service))`; log scaling because windows (hundreds of units) dwarf tour lengths (tens) |
| Decoder context | add `log1p(time)` next to remaining capacity: `project_step` input `2D + 1 -> 2D + 2` |

## 3. Data and training
* `make_sampler`: attach the generator's `TimeFeatures` to the `Batch` when `data.time_windows: true`.
* New configs: `cvrptw50_amazon.yaml` (customer-centric normalisation, polar depot, windows on,
  `tw_tightness` sweep 1.0 / 0.25).
* Early stopping already excludes the fixed depot round trip; with windows it judges
  `length + lambda * lateness`.

## 4. Inference and evaluation
* `to_batch` fills service and windows from the `VRPInstance` (seconds -> units via the fit above).
* Candidate selection and local search use the evaluator objective
  `travel + penalty * lateness` instead of travel only (`polish` already refuses added lateness).
* `nr-bench-synth --distribution amazon` keeps the windows in the instances it builds.

## Order of work (each step tested before the next)
1. Batch + env state + lateness, with the equivalence test against the evaluator.
2. Generator -> Batch wiring, `tw_tightness`.
3. Model time features + context, back-compat test with an old checkpoint.
4. Inference conversion + objective-aware candidate selection.
5. Configs, short smoke training, then the overnight CVRPTW-50 run.
