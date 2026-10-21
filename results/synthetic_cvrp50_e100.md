# Synthetic benchmark · CVRP-50 · 200 instances

Uniform instances, seeds not seen in training. Gap vs best solution found per instance.

| config     |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   vehicles |   runtime_s_p50 |   failures |
|:-----------|------------:|----------------:|--------:|------------:|-----------:|----------------:|-----------:|
| pyvrp:10   |         200 |            2.86 |    0.02 |        0    |       6.73 |           10.01 |          0 |
| attention  |         200 |            2.91 |    1.81 |        3.08 |       6.88 |            0.27 |          0 |
| ortools:10 |         200 |            2.97 |    3.96 |        7.23 |       6.74 |           10    |          2 |
