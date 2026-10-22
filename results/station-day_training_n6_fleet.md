# Benchmark `station-day` · training · 6 instances

Generated 20261022-1356 with `nr-bench --mode station-day --split training --n 6 --max-stops 1200 --workers 3 --tag fleet --configs reference savings pyvrp:60 cluster_savings_ls cluster_attention_ls@checkpoints/cvrp100_pomo_amazon.pt cluster_attention_ls@checkpoints/cvrp100_pomo_real.pt`

| config                                                  |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   gap_ctrl_% |   late_stops_% |   vehicles |   runtime_s_p50 |   failures |
|:--------------------------------------------------------|------------:|----------------:|--------:|------------:|-------------:|---------------:|-----------:|----------------:|-----------:|
| cluster_attention_ls@checkpoints/cvrp100_pomo_real.pt   |           6 |           13.43 |    0.77 |        2.31 |         0.79 |           0.27 |       4.83 |            2.05 |          0 |
| cluster_attention_ls@checkpoints/cvrp100_pomo_amazon.pt |           6 |           13.57 |    2.13 |        4.3  |         2.26 |           0.44 |       4.83 |            2.69 |          0 |
| cluster_savings_ls                                      |           6 |           13.64 |    2.64 |        4.66 |         2.79 |           0.97 |       4.83 |            0.64 |          0 |
| pyvrp:60                                                |           6 |           14.26 |    9.69 |       24.14 |        10.76 |           0    |       4.83 |           60.33 |          0 |
| savings                                                 |           6 |           14.86 |   13.03 |       18.33 |        13.92 |           1.07 |       4.83 |            0.05 |          0 |
| reference                                               |           6 |           15.34 |   17.39 |       27.3  |        18.62 |           0.1  |       4.83 |            0    |          0 |
