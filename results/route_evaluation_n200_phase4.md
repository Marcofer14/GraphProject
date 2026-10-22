# Benchmark `route` · evaluation · 200 instances

Generated 20261022-1412 with `nr-bench --mode route --split evaluation --n 200 --workers 6 --tag phase4 --configs reference ortools:10 pyvrp:10 attention_ls@checkpoints/cvrp100_pomo_real.pt attention@checkpoints/cvrp100_pomo_real.pt attention_ls@checkpoints/cvrp100_pomo_amazon_ft.pt attention_ls@checkpoints/cvrp100_pomo_amazon.pt attention@checkpoints/cvrp100_pomo_amazon.pt`

| config                                             |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   gap_ctrl_% |   driver_score |   late_stops_% |   vehicles |   runtime_s_p50 |   failures |
|:---------------------------------------------------|------------:|----------------:|--------:|------------:|-------------:|---------------:|---------------:|-----------:|----------------:|-----------:|
| pyvrp:10                                           |         200 |            2.9  |    0.26 |        1    |         0.38 |           0.07 |           0    |          1 |           10.04 |          0 |
| attention_ls@checkpoints/cvrp100_pomo_real.pt      |         200 |            2.94 |    1.79 |        4.12 |         2.86 |           0.07 |           0.01 |          1 |            0.5  |          0 |
| attention_ls@checkpoints/cvrp100_pomo_amazon.pt    |         200 |            2.98 |    2.92 |        4.63 |         4.33 |           0.07 |           0.07 |          1 |            0.52 |          0 |
| attention_ls@checkpoints/cvrp100_pomo_amazon_ft.pt |         200 |            2.98 |    3.23 |        4.31 |         4.86 |           0.07 |           0.14 |          1 |            0.52 |          0 |
| ortools:10                                         |         200 |            2.99 |    3.32 |        6.25 |         5.05 |           0.08 |           0    |          1 |           10.01 |          0 |
| attention@checkpoints/cvrp100_pomo_real.pt         |         200 |            3.08 |    6.7  |       10.51 |        10.36 |           0.07 |           0.02 |          1 |            0.43 |          0 |
| reference                                          |         200 |            3.28 |   13.97 |       20.74 |        22.22 |           0    |           0.09 |          1 |            0    |          0 |
| attention@checkpoints/cvrp100_pomo_amazon.pt       |         200 |            3.61 |   24.79 |       35.83 |        38.09 |           0.09 |           0.26 |          1 |            0.41 |          0 |
