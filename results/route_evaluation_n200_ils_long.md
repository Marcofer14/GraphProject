# Benchmark `route` · evaluation · 200 instances

Generated 20261022-1818 with `nr-bench --mode route --split evaluation --n 200 --tag ils_long --configs attention_ils@checkpoints/cvrp100_pomo_real_long.pt:2`

| config                                                |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   gap_ctrl_% |   driver_score |   late_stops_% |   vehicles |   runtime_s_p50 |   failures |
|:------------------------------------------------------|------------:|----------------:|--------:|------------:|-------------:|---------------:|---------------:|-----------:|----------------:|-----------:|
| attention_ils@checkpoints/cvrp100_pomo_real_long.pt:2 |         200 |             2.9 |       0 |           0 |            0 |           0.07 |              0 |          1 |               2 |          0 |
