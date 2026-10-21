# Synthetic benchmark · TSP-20 · 200 instances

Uniform instances, seeds not seen in training. Gap vs best solution found per instance.

| config       |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   vehicles |   runtime_s_p50 |   failures |
|:-------------|------------:|----------------:|--------:|------------:|-----------:|----------------:|-----------:|
| pyvrp:5      |         200 |            1.08 |    0    |        0    |          1 |            5.01 |          0 |
| attention_ls |         200 |            1.08 |    0.01 |        0    |          1 |            0.13 |          0 |
| attention    |         200 |            1.08 |    0.23 |        0.89 |          1 |            0.14 |          0 |
| savings_ls   |         200 |            1.09 |    0.53 |        1.67 |          1 |            0    |          0 |
| savings      |         200 |            1.16 |    7.15 |       12.89 |          1 |            0    |          0 |
