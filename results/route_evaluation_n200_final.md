# Benchmark `route` · evaluation · 200 instances

Generated 20261022-0938 with `nr-bench --mode route --split evaluation --n 200 --workers 6 --tag final --configs reference savings_ls ortools:10 pyvrp:10 attention_ls@checkpoints/cvrp150_pomo_amazon.pt attention_ls@checkpoints/cvrp100_pomo_amazon.pt attention_ls@checkpoints/cvrp50_pomo_amazon.pt attention_ls@checkpoints/cvrptw50_pomo_amazon.pt attention@checkpoints/cvrp150_pomo_amazon.pt`

| config                                           |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   gap_ctrl_% |   late_stops_% |   vehicles |   runtime_s_p50 |   failures |
|:-------------------------------------------------|------------:|----------------:|--------:|------------:|-------------:|---------------:|-----------:|----------------:|-----------:|
| pyvrp:10                                         |         200 |            2.9  |    0.25 |        0.92 |         0.36 |           0    |          1 |           10.04 |          0 |
| attention_ls@checkpoints/cvrp100_pomo_amazon.pt  |         200 |            2.98 |    2.91 |        4.63 |         4.32 |           0.07 |          1 |            0.9  |          0 |
| ortools:10                                       |         200 |            2.99 |    3.28 |        6.17 |         4.98 |           0    |          1 |           10    |          0 |
| attention_ls@checkpoints/cvrp150_pomo_amazon.pt  |         200 |            2.99 |    3.65 |        4.31 |         5.88 |           0.13 |          1 |            0.92 |          0 |
| attention_ls@checkpoints/cvrp50_pomo_amazon.pt   |         200 |            3    |    3.96 |        4.92 |         5.96 |           0.12 |          1 |            0.9  |          0 |
| savings_ls                                       |         200 |            3.01 |    4.14 |        8.7  |         6.51 |           0.44 |          1 |            0.01 |          0 |
| attention_ls@checkpoints/cvrptw50_pomo_amazon.pt |         200 |            3.09 |    7.28 |       31.8  |        11.29 |           0    |          1 |            0.97 |          0 |
| reference                                        |         200 |            3.28 |   13.96 |       20.74 |        22.2  |           0.09 |          1 |            0    |          0 |
| attention@checkpoints/cvrp150_pomo_amazon.pt     |         200 |            3.61 |   24.73 |       36.38 |        37.93 |           0.25 |          1 |            0.83 |          0 |
