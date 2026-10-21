# Synthetic benchmark · CVRP-50 · 200 instances

Uniform instances, seeds not seen in training. Gap vs best solution found per instance.

| config    |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   vehicles |   runtime_s_p50 |   failures |
|:----------|------------:|----------------:|--------:|------------:|-----------:|----------------:|-----------:|
| pyvrp:10  |         200 |            2.86 |    0    |        0    |       6.73 |           10.01 |          0 |
| attention |         200 |            2.97 |    4.19 |        5.69 |       6.92 |            0.21 |          0 |
| savings   |         200 |            3    |    5.07 |        8.31 |       6.94 |            0    |          0 |
