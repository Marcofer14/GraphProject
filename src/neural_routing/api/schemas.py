"""Request and response models of the routing API."""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_TIME_LIMIT_S = 30.0


class SolverInfo(BaseModel):
    name: str
    kind: str  # "neural" | "classic" | "reference"
    description: str


class ModelInfo(BaseModel):
    name: str  # checkpoint file stem, e.g. "cvrp100_pomo_amazon"
    problem: str
    num_customers: int
    epoch: int


class InstanceSummary(BaseModel):
    route_id: str
    station: str
    date: str
    num_stops: int
    route_score: str | None = None


class Stop(BaseModel):
    node: int  # 0 = station
    stop_id: str
    lat: float
    lng: float
    zone: str | None = None
    demand: float  # fraction of the van's capacity
    service_s: float
    tw_start_s: float | None = None  # seconds after departure; None = no window
    tw_end_s: float | None = None


class InstanceDetail(BaseModel):
    route_id: str
    station: str
    date: str
    capacity_cm3: float
    stops: list[Stop]
    driver_route: list[int] | None = Field(None, description="node order the real driver drove")


class SolveRequest(BaseModel):
    split: str = "evaluation"
    route_id: str
    solver: str
    time_limit: float = Field(2.0, gt=0, le=MAX_TIME_LIMIT_S)
    model: str | None = Field(None, description="checkpoint name for neural solvers")


class CompareRequest(BaseModel):
    split: str = "evaluation"
    route_id: str
    solvers: list[str] = Field(..., min_length=1, max_length=8)
    time_limit: float = Field(2.0, gt=0, le=MAX_TIME_LIMIT_S)
    model: str | None = None


class SolutionOut(BaseModel):
    solver: str
    model: str | None = None
    routes: list[list[int]]
    paths: list[list[tuple[float, float]]]  # per route, [lng, lat] from station to station
    travel_time_s: float
    route_duration_s: float
    num_routes: int
    tw_violations: int
    lateness_s: float
    feasible: bool
    runtime_s: float
    gap_pct: float | None = None  # filled by /compare: vs the best solution in the comparison


class CompareOut(BaseModel):
    route_id: str
    best: str
    solutions: list[SolutionOut]
