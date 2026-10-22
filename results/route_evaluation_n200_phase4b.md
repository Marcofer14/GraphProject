# Benchmark `route` · evaluation · 200 instances

Generated 20261022-1758 with `nr-bench --mode route --split evaluation --n 200 --workers 6 --tag phase4b --configs reference pyvrp:1 pyvrp:10 attention_ls@checkpoints/cvrp100_pomo_real.pt attention_ls@checkpoints/cvrp100_pomo_real_long.pt attention@checkpoints/cvrp100_pomo_real_long.pt attention_ils@checkpoints/cvrp100_pomo_real_long.pt:1`

| config                                                |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   gap_ctrl_% |   driver_score |   late_stops_% |   vehicles |   runtime_s_p50 |   failures |
|:------------------------------------------------------|------------:|----------------:|--------:|------------:|-------------:|---------------:|---------------:|-----------:|----------------:|-----------:|
| pyvrp:10                                              |         200 |            2.91 |    0.38 |        1.45 |         0.55 |           0.07 |           0    |          1 |           10.06 |          0 |
| attention_ils@checkpoints/cvrp100_pomo_real_long.pt:1 |         200 |            2.92 |    0.91 |        2.5  |         1.37 |           0.07 |           0.01 |          1 |            1    |          0 |
| pyvrp:1                                               |         200 |            2.94 |    1.42 |        3.05 |         2.1  |           0.07 |           0    |          1 |            1.05 |          0 |
| attention_ls@checkpoints/cvrp100_pomo_real_long.pt    |         200 |            2.94 |    1.63 |        3.58 |         2.59 |           0.07 |           0.01 |          1 |            0.57 |          0 |
| attention_ls@checkpoints/cvrp100_pomo_real.pt         |         200 |            2.94 |    1.87 |        4.12 |         2.97 |           0.07 |           0.01 |          1 |            0.6  |          0 |
| attention@checkpoints/cvrp100_pomo_real_long.pt       |         200 |            3.07 |    6.15 |        9.52 |         9.53 |           0.07 |           0.02 |          1 |            0.5  |          0 |
| reference                                             |         200 |            3.28 |   14.05 |       20.74 |        22.34 |           0    |           0.09 |          1 |            0    |          0 |
