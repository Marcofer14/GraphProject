# Synthetic benchmark · CVRP-50 · 200 instances

Amazon instances, seeds not seen in training. Gap vs best solution found per instance.

| config                                           |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   gap_ctrl_% |   late_stops_% |   vehicles |   runtime_s_p50 |   failures |
|:-------------------------------------------------|------------:|----------------:|--------:|------------:|-------------:|---------------:|-----------:|----------------:|-----------:|
| pyvrp:10                                         |         200 |            0.72 |    0.05 |        0.07 |         0.33 |           0.01 |          1 |           10.01 |          0 |
| attention_ls@checkpoints/cvrp50_pomo_amazon.pt   |         200 |            0.72 |    0.11 |        0.35 |         0.46 |           0.02 |          1 |            0.2  |          0 |
| ortools:10                                       |         200 |            0.72 |    0.16 |        0.4  |         0.82 |           0.01 |          1 |           10    |          0 |
| attention_ls@checkpoints/cvrptw50_pomo_amazon.pt |         200 |            0.72 |    0.3  |        0.58 |         1.58 |           0.01 |          1 |            0.21 |          0 |
| savings_ls                                       |         200 |            0.73 |    0.69 |        1.54 |         2.94 |           0.15 |          1 |            0    |          0 |
| attention@checkpoints/cvrp50_pomo_amazon.pt      |         200 |            0.73 |    0.85 |        1.55 |         3.69 |           0.02 |          1 |            0.19 |          0 |
| attention@checkpoints/cvrptw50_pomo_amazon.pt    |         200 |            0.75 |    4.16 |        7.2  |        17.72 |           0.01 |          1 |            0.21 |          0 |
