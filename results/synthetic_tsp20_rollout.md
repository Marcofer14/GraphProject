# Synthetic benchmark · TSP-20 · 200 instances

Uniform instances, seeds not seen in training. Gap vs best solution found per instance.

| config           |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   vehicles |   runtime_s_p50 |   failures |
|:-----------------|------------:|----------------:|--------:|------------:|-----------:|----------------:|-----------:|
| pyvrp:5          |         200 |            1.08 |    0    |        0    |          1 |            5    |          0 |
| attention        |         200 |            1.09 |    0.34 |        1.23 |          1 |            0.06 |          0 |
| attention_greedy |         200 |            1.1  |    2.1  |        5.22 |          1 |            0.05 |          0 |
