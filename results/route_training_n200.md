# Benchmark `route` · training · 200 instances

Generated 20261020-2140 with `nr-bench --mode route --split training --n 200 --workers 4 --configs reference nearest_neighbor savings ortools:1 ortools:10 pyvrp:1 pyvrp:10`

| config           |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   late_stops_% |   vehicles |   runtime_s_p50 |   failures |
|:-----------------|------------:|----------------:|--------:|------------:|---------------:|-----------:|----------------:|-----------:|
| pyvrp:10         |         200 |            3.11 |    0.05 |        0    |           0    |          1 |           10.04 |          0 |
| pyvrp:1          |         200 |            3.13 |    0.86 |        2    |           0    |          1 |            1.04 |          0 |
| ortools:10       |         200 |            3.21 |    3.34 |        6.07 |           0    |          1 |           10.01 |          0 |
| ortools:1        |         200 |            3.34 |    7.64 |       11.8  |           0.1  |          1 |            1.01 |          0 |
| savings          |         200 |            3.43 |   10.88 |       16.28 |           0.65 |          1 |            0.01 |          0 |
| reference        |         200 |            3.51 |   13.85 |       23.12 |           0.08 |          1 |            0    |          0 |
| nearest_neighbor |         200 |            3.59 |   15.75 |       20.46 |           0.59 |          1 |            0    |          0 |
