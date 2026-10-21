# Synthetic benchmark · TSP-20 · 200 instances

Uniform instances, seeds not seen in training. Gap vs best solution found per instance.

| config           |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   vehicles |   runtime_s_p50 |   failures |
|:-----------------|------------:|----------------:|--------:|------------:|-----------:|----------------:|-----------:|
| pyvrp:5          |         200 |            1.08 |    0    |        0    |          1 |            5    |          0 |
| ortools:1        |         200 |            1.08 |    0    |        0    |          1 |            1    |          0 |
| attention        |         200 |            1.08 |    0.23 |        0.89 |          1 |            0.08 |          0 |
| attention_greedy |         200 |            1.15 |    6.71 |       13.91 |          1 |            0.06 |          0 |
| savings          |         200 |            1.16 |    7.15 |       12.89 |          1 |            0    |          0 |
| nearest_neighbor |         200 |            1.27 |   16.96 |       29.71 |          1 |            0    |          0 |
