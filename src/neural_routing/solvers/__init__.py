from neural_routing.solvers import (  # noqa: F401  (registration)
    cluster,
    heuristics,
    local_search,
    neural,
    ortools_solver,
    pyvrp_solver,
)
from neural_routing.solvers.base import Solver, available_solvers, get_solver

__all__ = ["Solver", "available_solvers", "get_solver"]
