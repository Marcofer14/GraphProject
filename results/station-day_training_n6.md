# Benchmark `station-day` · training · 6 instances

Generated 20261020-2145 with `nr-bench --mode station-day --split training --n 6 --max-stops 1200 --workers 3 --configs reference nearest_neighbor savings ortools:60 pyvrp:60`

| config           |   instances |   travel_time_h |   gap_% |   gap_%_p90 |   late_stops_% |   vehicles |   runtime_s_p50 |   failures |
|:-----------------|------------:|----------------:|--------:|------------:|---------------:|-----------:|----------------:|-----------:|
| pyvrp:60         |           6 |           14.28 |    7.2  |       19.93 |           0    |       4.83 |           60.39 |          0 |
| savings          |           6 |           14.86 |   10.44 |       17.99 |           1.07 |       4.83 |            0.04 |          0 |
| reference        |           6 |           15.34 |   14.71 |       24.82 |           0.1  |       4.83 |            0    |          0 |
| ortools:60       |           6 |           17.96 |   20.12 |       55.91 |           0.24 |       4.83 |           60.1  |          0 |
| nearest_neighbor |           6 |           17.22 |   25.77 |       32.18 |           2.36 |       4.5  |            0.02 |          0 |
