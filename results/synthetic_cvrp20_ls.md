# Synthetic benchmark · CVRP-20 · 200 instances

Uniform instances, seeds not seen in training. Gap vs best solution found per instance.

| config       |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   vehicles |   runtime_s_p50 |   failures |
|:-------------|------------:|----------------:|--------:|------------:|-----------:|----------------:|-----------:|
| pyvrp:5      |         200 |            1.7  |    0.16 |        0    |       3.83 |            5.01 |          0 |
| attention_ls |         200 |            1.7  |    0.19 |        0.84 |       3.91 |            0.15 |          0 |
| attention    |         200 |            1.7  |    0.67 |        1.68 |       3.92 |            0.14 |          0 |
| savings_ls   |         200 |            1.74 |    2.87 |        6.78 |       3.97 |            0    |          0 |
| savings      |         200 |            1.77 |    4.49 |        9.29 |       3.99 |            0    |          0 |
