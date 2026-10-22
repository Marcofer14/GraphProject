# Benchmark `route` · evaluation · 200 instances

Generated 20261022-1629 with `nr-bench --mode route --split evaluation --n 200 --tag ils --configs attention_ils@checkpoints/cvrp100_pomo_real.pt:1 attention_ils@checkpoints/cvrp100_pomo_real.pt:2 pyvrp:2 ortools:2`

| config                                           |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   gap_ctrl_% |   driver_score |   late_stops_% |   vehicles |   runtime_s_p50 |   failures |
|:-------------------------------------------------|------------:|----------------:|--------:|------------:|-------------:|---------------:|---------------:|-----------:|----------------:|-----------:|
| attention_ils@checkpoints/cvrp100_pomo_real.pt:2 |         200 |            2.91 |    0.5  |        1    |         0.8  |           0.07 |           0    |          1 |            2    |          0 |
| pyvrp:2                                          |         200 |            2.92 |    0.73 |        2.05 |         1.06 |           0.07 |           0    |          1 |            2.04 |          0 |
| attention_ils@checkpoints/cvrp100_pomo_real.pt:1 |         200 |            2.91 |    0.79 |        1.77 |         1.25 |           0.07 |           0    |          1 |            1    |          0 |
| ortools:2                                        |         200 |            3.05 |    5.27 |        9.04 |         7.86 |           0.09 |           0.04 |          1 |            2.01 |          0 |
