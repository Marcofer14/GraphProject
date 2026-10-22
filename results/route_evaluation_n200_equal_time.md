# Benchmark `route` · evaluation · 200 instances

Generated 20261022-1445 with `nr-bench --mode route --split evaluation --n 200 --configs ortools:0.5 pyvrp:0.5 ortools:1 pyvrp:1 --tag equal_time`

| config      |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   gap_ctrl_% |   driver_score |   late_stops_% |   vehicles |   runtime_s_p50 |   failures |
|:------------|------------:|----------------:|--------:|------------:|-------------:|---------------:|---------------:|-----------:|----------------:|-----------:|
| pyvrp:1     |         200 |            2.92 |    0.01 |        0    |         0.03 |           0.07 |           0    |          1 |            1.03 |          0 |
| pyvrp:0.5   |         200 |            2.94 |    0.63 |        1.41 |         0.9  |           0.07 |           0    |          1 |            0.53 |          0 |
| ortools:1   |         200 |            3.07 |    5.04 |       10.3  |         7.48 |           0.09 |           0.05 |          1 |            1    |          0 |
| ortools:0.5 |         200 |            3.15 |    7.76 |       15.28 |        11.53 |           0.09 |           0.09 |          1 |            0.5  |          0 |
