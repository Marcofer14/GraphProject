# Neural last-mile routing: knowledge base for the project assistant

Each "## " section is one retrievable chunk. Keep facts exact; they come from the README, the design
decisions (docs/decisions.md) and the results files. Update them together.

## Overview
Neural last-mile routing is a portfolio project by Marco Fernández (B.Sc. in Data Science student at UCA,
Buenos Aires; data engineer at ExxonMobil). It is an AI that plans delivery-van routes: a neural network
(a Transformer trained with deep reinforcement learning) proposes routes and classic local search
polishes them. It is benchmarked on real Amazon delivery routes against the real drivers, Google
OR-Tools and PyVRP. The case is framed as a fictional company, Ruta Norte Logistics. Code:
https://github.com/Marcofer14/GraphProject. Live demo: https://marcofer14.github.io/GraphProject/

## Headline results
On 200 real Amazon routes never used in training (about 150 stops each): the model's first answer, in
about half a second, needs 10.5 % less driving than the real drivers (3.28 h to 2.94 h per route, about
20 minutes less per van per day) and is shorter on all 200 routes, with almost no late deliveries
(0.01 % of stops). Given the same computing time it beats PyVRP and Google OR-Tools: with 1 second it is
1.19 % above the best route found versus 1.60 % for PyVRP and 9.17 % for OR-Tools; with 2 seconds 0.77 %
versus 1.28 % and 8.09 %. PyVRP given 10 seconds is still slightly better on average (0.70 %). On whole
station-days it uses 9.9 % fewer fleet hours than the drivers.

## The problem being solved
Last-mile delivery: each van leaves a station with roughly 100-200 stops and must visit each once and
come back. The order decides the driving time. Technically it is a capacitated vehicle routing problem
with time windows (CVRPTW): vans have a volume capacity, stops have delivery windows (treated as soft:
arriving late is allowed but counted). With 150 stops there are about 10^262 possible orders, far more
than atoms in the universe (about 10^80), so trying them all is impossible.

## Data
The Amazon Last Mile Routing Research Challenge dataset (2021, AWS Open Data, licence CC BY-NC 4.0):
9,164 real routes from 2018 (6,112 in the training split, 3,052 in the evaluation split) with about
1.34 million stops, in five US metro areas (Los Angeles, Seattle, Boston, Austin, Chicago) and 17-18
stations. It includes stop locations, parcel sizes, delivery windows, the order each driver actually
drove, and real driving-time matrices between every pair of stops of a route (1.7 GB). Times are
one-way: A to B can differ from B to A.

## Data pipeline (ETL)
The raw JSON is parsed as a stream with ijson, so the 1.7 GB travel-time file never sits in memory, into
Parquet tables (routes, stops, packages) and one memory-mapped float32 file with all matrices
concatenated. Package windows and volumes are aggregated per stop. Instances are built on demand from
these files.

## The neural network
An encoder-decoder Transformer in the style of the Attention Model (Kool et al., 2019) with 1.3 million
parameters: 128-dimensional embeddings, 6 encoder layers, 8 attention heads, feed-forward size 512,
instance normalisation. Inputs per stop: position relative to the centre of the stops, the station
encoded as direction and log-distance, and parcel volume divided by van capacity. The decoder picks one
stop at a time from a context made of the mean of all stop embeddings, the current stop, the first stop
and the van capacity left; it masks visited stops and stops that no longer fit, and clips scores with
10 times tanh.

## Matrix-aware attention (the key idea)
Straight-line distance is misleading on real streets: only 67 % of each stop's 5 nearest neighbours by
straight line are also among its 5 nearest by real driving time, and real legs cost about 2.6 times the
straight-line estimate. So the model reads the real one-way driving times: a small MLP turns each pair's
times (both directions, log-scaled) into a bias added inside every encoder attention head, the decoder's
glimpse attention and the final next-stop scores (in the spirit of MatNet). The biases start at zero, so
training starts exactly from the previous model. This took the network alone from 38 % to under 10 %
above the best route.

## How it learned (reinforcement learning)
No one showed it good routes. It was trained with REINFORCE using the POMO shared baseline (Kwon et al.,
2020): for each training problem it builds one route starting from every stop, and each route is pushed
up or down by how much shorter or longer it is than the average of those routes. Over millions of
routes the network learns what shortens a route. Training used PyTorch, Hydra configuration, TensorBoard,
resumable checkpoints and early stopping on the part of the route a planner can change.

## Training curriculum and hardware
Stage 1: 50-stop synthetic problems from a generator calibrated on Amazon's data (a distant station,
clustered stops, parcel sizes drawn from the real ones). Stage 2: 100-stop synthetic problems,
warm-started from stage 1. Stage 3: 100-stop slices of real training routes with their real driving
times, adding the matrix-aware attention, 60 epochs at learning rate 5e-5 (under three hours). All
training and inference ran on a single NVIDIA GTX 1660 Super with 6 GB, a 2019 mid-range gaming card that
is close to obsolete today. The final model is called cvrp100_pomo_real_long.

## From stops to a plan in half a second (inference)
1) Encode the route once on the GPU. 2) Decode about 1,200 candidate routes in parallel: one from every
first stop, on 8 rotated or mirrored copies of the map. 3) Score every candidate exactly on the real
time matrix (travel plus 10 times lateness), vectorised with NumPy. 4) Polish the best 8 with local
search compiled with Numba: 2-opt (exact for one-way times using prefix sums), or-opt (move runs of 1-3
stops) and swap. 5) If time is left, iterated local search: shake a small part of the best route
(double-bridge kick inside a 50-stop window), polish again, keep it only if shorter, until the time
budget is spent. Profiling found the bottleneck was scoring candidates in Python, not the network;
vectorising it made inference 2.5 times faster with identical results.

## Speed and hardware
On the GTX 1660 Super the first answer for a ~150-stop route takes about half a second (median 0.57 s);
on a CPU about 2 seconds. A whole station-day of several hundred to a thousand stops takes a few
seconds. The local search and the classical solvers run on one CPU core. No data-centre hardware or
paid solver licences are needed.

## Comparison at equal computing time
Solvers were compared with the same computing budget on an otherwise idle machine, because a 1-second
budget is sensitive to anything else running. Results on 200 unseen routes, as extra driving above the
best route found: 1 s: ours 1.19 %, PyVRP 1.60 % (ours better on 122 of 200 routes), OR-Tools 9.17 %
(ours better on 193). 2 s: ours 0.77 %, PyVRP 1.28 % (ours better on 128), OR-Tools 8.09 %. PyVRP with
10 s: 0.70 %, better than our 2-second run on 93 routes and worse on 94 (13 ties). OR-Tools with 10 s:
5.32 %. The network alone (no local search): 9.69 %. Real drivers: 22.5 %.

## What "above the best route" means
About three quarters of every route is the unavoidable trip from the station to the delivery area and
back. Results are reported as the controllable gap: extra driving compared with the best route any
solver found for the same stops, counting only the part of the route a planner can change. 0 % is the
best. Every solver, including the drivers' actual routes, is scored by the same evaluator on Amazon's
real driving times.

## Whole station-days
To plan all the stops a station delivered in one day with the same vans: 1) split the stops into one
group per van with a capacity-aware k-means (k-means++ seeding, stops assigned in order of regret, never
over a van's volume), 2) route each van with the neural pipeline, 3) move and swap stops between vans
when it saves time. On 20 real days no model saw (evaluation split, 3-6 vans, about 600 stops each, all
five cities): 11.88 fleet hours per day versus 13.18 for the drivers (9.9 % less, fewer on all 20 days),
with fewer late stops than the drivers (0.16 % vs 0.18 %), in about 2-3 seconds. PyVRP given 60 seconds
averages 12.45 hours and wins 10 of the 20 days (the smaller ones). On these days the vans' areas are
far apart: a median 0.2 % of stops change van, so the saving comes mostly from the order inside each
van. Travel times between stops of different original routes are estimated from distance (194 s + 99 s
per km), because Amazon only publishes times within each route.

## Driver similarity
Using the Amazon challenge's own metric (sequence deviation times edit distance on travel times),
optimised routes, ours and PyVRP's alike, score about 0.07 where the drivers' own routes score 0: none
looks like a driver's route. Drivers deliver zone by zone even when it costs time, and optimise things
the data does not contain.

## Limits
Given 10 seconds or more on a CPU, PyVRP is still slightly better on average. The network alone is about
10 % above the best route; local search does much of the final work. It runs best on a GPU (about 2 s
on CPU). Time windows are soft and rules like driver breaks or vehicle access are not modelled. Drivers
optimise things the data lacks (parking, building access, customer habits, safety), so shorter driving
is not automatically a better day. One dataset: Amazon, five US cities, 2018. The demo's map lines join
stops in order; they are not street geometry.

## Use cases
Parcel and e-commerce delivery (the tested case); re-planning during the day when an urgent order
arrives or a van breaks down, since a plan takes half a second; planning a whole station in seconds;
other routing with stops and time windows (grocery, pharmacy and B2B distribution, field technicians,
waste collection) after retraining on that business's data; companies without data-centre hardware;
quick what-if analysis (one van fewer, a later start). Not the right tool when there is plenty of time
and a CPU (use PyVRP), or when hard legal rules dominate.

## Baselines compared
Real drivers (the routes Amazon drivers actually drove); Google OR-Tools (guided local search) with
0.5-10 s; PyVRP (hybrid genetic search, one of the strongest open-source routing solvers) with 0.5-10 s
for routes and 60 s for days; classic heuristics (nearest neighbour, Clarke-Wright savings, with and
without local search). All run behind one solver interface and are scored by one evaluator.

## Earlier milestones
Phase 0-1: data pipeline, evaluator and classical baselines. Phase 2: the attention model with POMO on
synthetic TSP and CVRP. Phase 3: the Amazon-calibrated generator, customer-centric normalisation with a
polar station encoding (which fixed failures when the station is far from the stops), time windows,
local search and a size curriculum; the coordinate-only model reached 4.33 % with local search on real
routes but 38 % alone. Phase 4: matrix-aware attention trained on real routes, driver score, 2.5 times
faster inference, whole days and the anytime search. Phase 5-6: a FastAPI backend and this static demo.
21 design decisions are written up with their numbers in docs/decisions.md.

## Tech stack
Machine learning: Python, PyTorch, CUDA, Transformer attention, deep reinforcement learning (REINFORCE,
POMO), Hydra, TensorBoard. Optimisation: Google OR-Tools, PyVRP, Numba, NumPy, local search (2-opt,
or-opt, swap), iterated local search, capacitated k-means. Data: pandas, Apache Arrow and Parquet,
ijson, AWS Open Data, Jupyter, Matplotlib, seaborn. Web and API: FastAPI, Pydantic, Uvicorn, React,
TypeScript, Vite, MapLibre GL, OpenStreetMap tiles. Engineering: Git, GitHub Actions (CI and
deployment), GitHub Pages, uv, pytest (129 tests), Ruff, mypy in strict mode, pre-commit. This assistant
runs on a Cloudflare Worker calling an open model on Groq's free tier.

## The demo page
The live demo has: an introduction; how to use it; a guided example; "Compare" (any two plans on 25 real
routes drawn at random, five per city, with a replay of the shift); "The race" (quality against
computing time, with a GPU/CPU switch); "A whole station-day" (20 unseen days, one colour per van);
use cases; the architecture; results and limits. It is a static React site: every plan was computed once
and exported to JSON, so nothing is computed while browsing.

## Reproducing it
With uv installed: uv sync, download the dataset (scripts/download_amazon.py), run the ETL (nr-etl),
train with nr-train and a Hydra config (for example cvrp100_real_matrix_long), benchmark with nr-bench
(for example nr-bench --mode route --split evaluation --n 200 --configs reference pyvrp:1
attention_ils@checkpoints/cvrp100_pomo_real_long.pt:1), and export the demo data with nr-demo-export.
The release notes in docs/releases say which trained checkpoint each version ships.

## About the author and contact
Marco Fernández is completing a B.Sc. in Data Science at UCA in Buenos
Aires, where he is also a teaching assistant in Numerical Methods and Introduction to Data Engineering,
and works as a data engineer at ExxonMobil building ETL pipelines with Python, SQL and Azure. His focus is
AI: reinforcement learning, LLM agents and reliable data systems. Portfolio and contact:
https://marcofer14.github.io/
