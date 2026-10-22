# Synthetic benchmark · CVRP-50 · 200 instances

Amazon instances, seeds not seen in training. Gap vs best solution found per instance.

| config                                           |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   late_stops_% |   vehicles |   runtime_s_p50 |   failures |
|:-------------------------------------------------|------------:|----------------:|--------:|------------:|---------------:|-----------:|----------------:|-----------:|
| ortools:10                                       |         200 |            0.72 |    0.25 |        0.62 |           0.11 |          1 |           10    |          0 |
| pyvrp:10                                         |         200 |            0.72 |    0.27 |        0.27 |           0.13 |          1 |           10.01 |          0 |
| attention_ls@checkpoints/cvrp50_pomo_amazon.pt   |         200 |            0.73 |    0.64 |        1.87 |           0.21 |          1 |            0.28 |          0 |
| savings_ls                                       |         200 |            0.73 |    0.99 |        2.72 |           1.22 |          1 |            0    |          0 |
| attention@checkpoints/cvrp50_pomo_amazon.pt      |         200 |            0.73 |    1.37 |        2.75 |           0.22 |          1 |            0.28 |          0 |
| attention_ls@checkpoints/cvrptw50_pomo_amazon.pt |         200 |            0.73 |    1.68 |        5.54 |           0.11 |          1 |            0.31 |          0 |
| attention@checkpoints/cvrptw50_pomo_amazon.pt    |         200 |            0.76 |    4.84 |        9.11 |           0.11 |          1 |            0.3  |          0 |
