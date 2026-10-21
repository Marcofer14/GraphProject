# Synthetic benchmark · CVRP-20 · 200 instances

Uniform instances, seeds not seen in training. Gap vs best solution found per instance.

| config           |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   vehicles |   runtime_s_p50 |   failures |
|:-----------------|------------:|----------------:|--------:|------------:|-----------:|----------------:|-----------:|
| pyvrp:5          |         200 |            1.7  |    0.16 |        0    |       3.83 |            5    |          0 |
| attention        |         200 |            1.7  |    0.67 |        1.68 |       3.92 |            0.08 |          0 |
| ortools:1        |         200 |            1.72 |    1.79 |        4.85 |       3.83 |            1    |          4 |
| savings          |         200 |            1.77 |    4.48 |        9.29 |       3.99 |            0    |          0 |
| attention_greedy |         200 |            1.79 |    5.86 |       11.03 |       4.08 |            0.06 |          0 |
| nearest_neighbor |         200 |            2.21 |   31.15 |       45.18 |       3.9  |            0    |          0 |
