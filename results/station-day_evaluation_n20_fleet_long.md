# Benchmark `station-day` · evaluation · 20 instances

Generated 20261022-1844 with `nr-bench --mode station-day --split evaluation --n 20 --min-routes 3 --max-stops 1200 --workers 3 --tag fleet_long --configs reference savings pyvrp:60 cluster_savings_ls cluster_attention_ls@checkpoints/cvrp100_pomo_real.pt cluster_attention_ls@checkpoints/cvrp100_pomo_real_long.pt`

| config                                                     |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   gap_ctrl_% |   late_stops_% |   vehicles |   runtime_s_p50 |   failures |
|:-----------------------------------------------------------|------------:|----------------:|--------:|------------:|-------------:|---------------:|-----------:|----------------:|-----------:|
| cluster_attention_ls@checkpoints/cvrp100_pomo_real_long.pt |          20 |           11.88 |    0.79 |        1.64 |         0.84 |           0.16 |       4.05 |            2.28 |          0 |
| cluster_attention_ls@checkpoints/cvrp100_pomo_real.pt      |          20 |           11.88 |    0.86 |        1.67 |         0.9  |           0.23 |       4.05 |            2.25 |          0 |
| cluster_savings_ls                                         |          20 |           12.04 |    2.18 |        3.5  |         2.31 |           0.55 |       4.05 |            0.11 |          0 |
| pyvrp:60                                                   |          20 |           12.45 |    4.24 |       11.66 |         4.41 |           0    |       4    |           60.3  |          0 |
| savings                                                    |          20 |           12.88 |    9.53 |       13.45 |        10.07 |           0.75 |       4.05 |            0.03 |          0 |
| reference                                                  |          20 |           13.18 |   12.02 |       15.35 |        12.68 |           0.18 |       4.05 |            0    |          0 |
