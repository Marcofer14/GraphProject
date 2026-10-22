# Synthetic benchmark · CVRP-50 · 200 instances

Amazon instances, seeds not seen in training. Gap vs best solution found per instance.

| config                                             |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   vehicles |   runtime_s_p50 |   failures |
|:---------------------------------------------------|------------:|----------------:|--------:|------------:|-----------:|----------------:|-----------:|
| pyvrp:10                                           |         200 |            0.72 |    0.02 |        0.07 |          1 |           10.01 |          0 |
| attention_ls@checkpoints/cvrp50_pomo_amazon_all.pt |         200 |            0.72 |    0.09 |        0.31 |          1 |            0.2  |          0 |
| attention_ls@checkpoints/cvrp50_pomo_amazon.pt     |         200 |            0.72 |    0.09 |        0.3  |          1 |            0.2  |          0 |
| attention@checkpoints/cvrp50_pomo_amazon.pt        |         200 |            0.73 |    0.82 |        1.42 |          1 |            0.2  |          0 |
| attention@checkpoints/cvrp50_pomo_amazon_all.pt    |         200 |            0.73 |    0.85 |        1.57 |          1 |            0.2  |          0 |
| savings                                            |         200 |            0.75 |    3.44 |        6.06 |          1 |            0    |          0 |
| attention@checkpoints/cvrp50_pomo.pt               |         200 |            0.77 |    7.13 |       10.5  |          1 |            0.2  |          0 |
