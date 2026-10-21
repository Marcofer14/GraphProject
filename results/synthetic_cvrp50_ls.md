# Synthetic benchmark · CVRP-50 · 200 instances

Uniform instances, seeds not seen in training. Gap vs best solution found per instance.

| config       |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   vehicles |   runtime_s_p50 |   failures |
|:-------------|------------:|----------------:|--------:|------------:|-----------:|----------------:|-----------:|
| pyvrp:10     |         200 |            2.86 |    0.05 |        0    |       6.73 |           10.02 |          0 |
| attention_ls |         200 |            2.87 |    0.62 |        1.32 |       6.82 |            0.23 |          0 |
| attention    |         200 |            2.91 |    1.84 |        3.08 |       6.88 |            0.22 |          0 |
| savings_ls   |         200 |            2.96 |    3.58 |        6.31 |       6.92 |            0    |          0 |
| savings      |         200 |            3    |    5.11 |        8.31 |       6.94 |            0    |          0 |
