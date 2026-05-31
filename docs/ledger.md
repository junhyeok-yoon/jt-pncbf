# Ledger

One row per run with a usable evaluation. The `parent` column records any warm-start
source; `eval_source` states whether the numbers come from a full-pool final eval or
from the best in-loop eval row, including the step, for runs that have no final eval.

| version | date | parent | seeds | eval_source | reach | collision | oob | stuck | timeout | infeas | sat_rate | cps | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2.0.0 | 2026-05-28 13:21:10 | - | 42 | full_n500 | 0.8760 | 0.1020 | 0.0000 |  | 0.0220 | 0.0736 | 0.1345 | 0.6498 | superseded (first full run) |
| v2.0.0 | 2026-05-28 14:38:09 | - | 42 | full_n500 | 0.7260 | 0.2700 | 0.0000 |  | 0.0040 | 0.0484 | 0.1186 | 0.1714 | reverted (truncation regressed) |
| v2.0.0 | 2026-05-28 14:57:54 | - | 42 | full_n500 | 0.8960 | 0.0800 | 0.0000 | 0.0160 | 0.0080 | 0.2064 | 0.4818 | 0.6540 | superseded (pre-alignment baseline) |
| v2.0.0 | 2026-05-28 19:21:06 | - | 42 | full_n500 (fixed pool) | 0.9940 | 0.0000 | 0.0000 |  | 0.0060 | 0.0372 | 0.1091 | 0.9828 | diagnostic (fixed obstacle) |
| v2.0.0 | 2026-05-28 20:36:10 | - | 42 | full_n500 | 0.8360 | 0.1480 | 0.0000 | 0.0120 | 0.0040 | 0.2317 | 0.5146 | 0.4564 | reverted (obs saturation regressed) |
| v2.0.0 | 2026-05-28 22:45:06 | - | 42 | full_n500 | 0.8620 | 0.1180 | 0.0020 | 0.0140 | 0.0040 | 0.2232 | 0.5047 | 0.5420 | reverted (gamma regressed) |
| **v2.0.0** | **2026-05-29 00:14:41** | **-** | **42** | **full_n500** | **0.9280** | **0.0040** | **0.0000** | **0.0620** | **0.0060** | **0.1933** | **0.1517** | **0.7969** | **adopted** |
| v2.0.0 | 2026-05-29 01:02:01 | - | 42 | inloop_n200@20000 | 0.9200 | 0.0100 | 0.0000 | 0.0700 | 0.0000 | 0.1979 | 0.4128 | 0.7706 | ablation (R=0.1) |
| v2.0.0 | 2026-05-29 01:15:30 | - | 42 | inloop_n200@20000 | 0.8900 | 0.0000 | 0.0000 | 0.1000 | 0.0100 | 0.1590 | 0.0282 | 0.7372 | ablation (R=5.0) |
| v2.0.1 | 2026-05-29 08:42:10 | v2.0.0__20260529-001441__seed42 | 42 | eval_only(HardNet filter on adopted V_S, full_n500) | 0.9200 | 0.0020 | 0.0000 | 0.0720 | 0.0060 | 0.3306 | 0.2839 | 0.7418 | projection filter viable (collision 0.002, but saturation/stuck up vs CBF-QP) |
| v2.0.1 | 2026-05-29 09:32:20 | - | 42 | full_n500 | 0.8720 | 0.0560 | 0.0000 | 0.0400 | 0.0320 | 0.1539 | 0.1638 | 0.6578 | JT first run - stable, policy learned; below OC |
| v2.0.1 | 2026-05-29 11:49:34 | v2.0.1__20260529-093220__seed42 | 42 | full_n500 | 0.8620 | 0.0420 | 0.0000 | 0.0540 | 0.0420 | 0.1405 | 0.1838 | 0.6608 | value-only refinement 4k - budget-limited; extended in follow-up |
| v2.0.1 | 2026-05-29 12:01:27 | v2.0.1__20260529-114934__seed42 | 42 | full_n500 | 0.9080 | 0.0260 | 0.0020 | 0.0320 | 0.0320 | 0.1455 | 0.1552 | 0.7633 | value-only refinement 42k V-updates - collision improved; roughness above OC |
| v2.0.1 | 2026-05-29 12:14:39 | v2.0.1__20260529-093220__seed42 | 42 | full_n500 (CBF-QP collection, HardNet deploy) | 0.9020 | 0.0300 | 0.0000 | 0.0420 | 0.0260 | 0.1164 | 0.1619 | 0.7520 | CBF-QP collection refinement - no roughness gain vs HardNet |
| v2.0.1 | 2026-05-29 12:43:38 | - | 42 | full_n500 | 0.8760 | 0.0720 | 0.0000 | 0.0300 | 0.0220 | 0.1406 | 0.1798 | 0.6488 | JT OC-scale value batch - stable, roughness worsened, full-pool below first JT |
| v2.0.1 | 2026-05-29 16:13:00 | - | 42 | full_n500 | 0.9060 | 0.0400 | 0.0000 | 0.0440 | 0.0100 | 0.1608 | 0.1384 | 0.7287 | JT slow schedule - stable, smoother V_S, collision still above OC |
| v2.0.1 | 2026-05-29 16:49:24 | v2.0.1__20260529-093220__seed42 | 42 | full_n500 | 0.9020 | 0.0180 | 0.0000 | 0.0480 | 0.0320 | 0.1491 | 0.1657 | 0.7573 | value-refine sigma=0 - unsafe held, collision improved, roughness unchanged |
| v2.0.1 | 2026-05-29 16:55:56 | v2.0.1__20260529-093220__seed42 | 42 | full_n500 | 0.8740 | 0.0460 | 0.0000 | 0.0540 | 0.0260 | 0.1714 | 0.1598 | 0.6636 | value-refine sigma=0.1 - unsafe held, eval regressed |
| **v2.0.1** | **2026-05-29 17:10:57** | **-** | **42** | **full_n500** | **0.9580** | **0.0060** | **0.0000** | **0.0260** | **0.0100** | **0.0990** | **0.1215** | **0.8852** | **JT slow schedule extended - stable, new v2.0.1 SOTA** |
| v2.0.1 | 2026-05-29 22:20:32 | - | 12345 | full_n500 | 0.9500 | 0.0060 | 0.0000 | 0.0260 | 0.0180 | 0.0851 | 0.1248 | 0.8775 | multi-seed validation seed 12345 |
| v2.0.1 | 2026-05-30 00:30:59 | - | 99 | full_n500 | 0.9540 | 0.0040 | 0.0000 | 0.0280 | 0.0140 | 0.1351 | 0.1206 | 0.8705 | multi-seed validation seed 99 |

Note: eval-only HardNet rows report projection infeasibility (`||L_g h|| < 5e-4` or empty half-space/box intersection), not the CBF-QP slack-active fraction.
Note: v2.1.0 ran no training run; SOTA unchanged (v2.0.1 seed-42, cps 0.8852, remains version-SOTA). A lookahead-alpha filter was tested as a 6-arm x 3-seed eval_only ablation (L0-L5); all arms regressed vs L0 and the axis was rejected, so these eval_only runs are recorded in docs/versions/v2.1.0_results.md (with the failure diagnosis) rather than as ledger rows. Saved eval outputs for the 18 runs are retained on disk.
