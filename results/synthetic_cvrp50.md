# Synthetic benchmark · CVRP-50 · 200 instances

Uniform instances, seeds not seen in training. Gap vs best solution found per instance.

| config           |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   vehicles |   runtime_s_p50 |   failures |
|:-----------------|------------:|----------------:|--------:|------------:|-----------:|----------------:|-----------:|
| pyvrp:10         |         200 |            2.86 |    0.02 |        0    |       6.73 |           10.01 |          0 |
| attention        |         200 |            2.91 |    2    |        3.26 |       6.88 |            0.19 |          0 |
| ortools:10       |         200 |            2.95 |    3.23 |        6.1  |       6.74 |           10    |          2 |
| savings          |         200 |            3    |    5.1  |        8.31 |       6.94 |            0    |          0 |
| ortools:1        |         200 |            3.05 |    6.49 |       10.95 |       6.74 |            1    |          3 |
| attention_greedy |         200 |            3.07 |    7.47 |       11.42 |       7.1  |            0.11 |          0 |
| nearest_neighbor |         200 |            3.84 |   34.66 |       46.67 |       6.78 |            0    |          0 |
