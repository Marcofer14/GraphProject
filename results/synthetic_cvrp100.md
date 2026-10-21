# Synthetic benchmark · CVRP-100 · 100 instances

Uniform instances, seeds not seen in training. Gap vs best solution found per instance.

| config       |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   vehicles |   runtime_s_p50 |   failures |
|:-------------|------------:|----------------:|--------:|------------:|-----------:|----------------:|-----------:|
| pyvrp:20     |         100 |            4.28 |    0    |        0    |      10.43 |           20.03 |          0 |
| attention_ls |         100 |            4.34 |    1.34 |        2.35 |      10.57 |            0.66 |          0 |
| attention    |         100 |            4.43 |    3.53 |        4.73 |      10.74 |            0.65 |          0 |
| savings_ls   |         100 |            4.45 |    3.91 |        5.99 |      10.66 |            0.01 |          0 |
| savings      |         100 |            4.52 |    5.61 |        8.16 |      10.69 |            0    |          0 |
| ortools:20   |         100 |            4.54 |    6.41 |        9.28 |      10.45 |           20    |          6 |
