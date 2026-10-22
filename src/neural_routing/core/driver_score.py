"""How much does a route look like the one the real driver drove?

Re-implementation of the scoring metric of the Amazon Last Mile Routing Research Challenge (2021),
where submissions were judged on similarity to the drivers' actual sequences, not on length:

    score(route) = SD(actual, proposed) x ERP_per_edit(actual, proposed)

* **Sequence deviation (SD):** walk the proposed sequence and look up each stop's position in
  the actual one; every jump of more than one position adds ``|jump| - 1``. Normalised by
  ``n (n - 1) / 2`` so it lies in [0, 1]; 0 when the order matches.
* **ERP per edit:** edit distance with real penalty between the two sequences. Substituting stop a
  for stop b costs their travel time on the route's matrix after z-normalisation and shifting to
  a minimum of 0 (so the cost is "how far apart" in normalised units); inserting or deleting a
  stop costs the gap penalty ``g = 1000``. The total is divided by the number of edits (0 if none).

Both use the stop sequences without the station. Lower is more driver-like; 0 is identical.
This follows the challenge's published description and scoring code; differences of detail (e.g.
tie-breaking inside the dynamic programme) can shift individual values slightly.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from neural_routing.core.instance import FloatArray, IntArray, Solution, VRPInstance

GAP_PENALTY = 1000.0


def sequence_deviation(actual: IntArray, proposed: IntArray) -> float:
    n = len(actual)
    if n < 2:
        return 0.0
    position = np.empty(int(max(actual.max(), proposed.max())) + 1, dtype=np.int64)
    position[actual] = np.arange(n)
    jumps = np.abs(np.diff(position[proposed])) - 1
    return float(2.0 * jumps.sum() / (n * (n - 1)))


def normalised_matrix(durations: FloatArray) -> FloatArray:
    """Z-score of all travel times, shifted so the minimum is 0 (as in the challenge code)."""
    z = (durations - durations.mean()) / max(float(durations.std()), 1e-12)
    return np.asarray(z - z.min(), dtype=np.float64)


@njit(cache=True)
def _erp(actual, proposed, M, g):  # type: ignore[no-untyped-def]
    """Edit distance with real penalty; returns (total cost, number of edits)."""
    n, m = len(actual), len(proposed)
    cost = np.zeros((n + 1, m + 1))
    edits = np.zeros((n + 1, m + 1), dtype=np.int64)
    for i in range(1, n + 1):
        cost[i, 0] = cost[i - 1, 0] + g
        edits[i, 0] = i
    for j in range(1, m + 1):
        cost[0, j] = cost[0, j - 1] + g
        edits[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a, b = actual[i - 1], proposed[j - 1]
            sub = 0.0 if a == b else M[a, b]
            options = (cost[i - 1, j - 1] + sub, cost[i - 1, j] + g, cost[i, j - 1] + g)
            k = 0
            if options[1] < options[k]:
                k = 1
            if options[2] < options[k]:
                k = 2
            cost[i, j] = options[k]
            if k == 0:
                edits[i, j] = edits[i - 1, j - 1] + (0 if a == b else 1)
            elif k == 1:
                edits[i, j] = edits[i - 1, j] + 1
            else:
                edits[i, j] = edits[i, j - 1] + 1
    return cost[n, m], edits[n, m]


def erp_per_edit(
    actual: IntArray, proposed: IntArray, M: FloatArray, gap: float = GAP_PENALTY
) -> float:
    total, edits = _erp(actual.astype(np.int64), proposed.astype(np.int64), M, gap)
    return float(total / edits) if edits else 0.0


def driver_score(instance: VRPInstance, solution: Solution) -> float | None:
    """Challenge score of a single-route solution against the instance's driver sequence.

    None when there is no recorded driver route or the solution is not one complete route.
    """
    if instance.reference_routes is None or len(instance.reference_routes) != 1:
        return None
    if solution.num_routes != 1:
        return None
    actual = np.asarray(instance.reference_routes[0], dtype=np.int64)
    proposed = np.asarray(solution.routes[0], dtype=np.int64)
    if len(actual) != len(proposed) or set(actual.tolist()) != set(proposed.tolist()):
        return None
    sd = sequence_deviation(actual, proposed)
    if sd == 0.0:
        return 0.0
    return sd * erp_per_edit(actual, proposed, normalised_matrix(instance.durations))
