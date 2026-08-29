# fig_rollout_nine.pdf — legend regeneration, verification report

Asset rebuild only. No git, no re-scoring, no new evaluation beyond the re-simulation the shipped
producer performs on every render. No existing script, figure, artifact or manifest was edited,
overwritten or moved; everything below lives in this directory.

---

## 1. The producing script, and what it does

**Path: `scripts/analysis/v292_fig_nine.py`.** Sole producer — it is the only file under `scripts/`
or `src/` that mentions `rollout_nine` or the nine-scene manifest. It renders both
`fig_rollout_nine.pdf` and `fig_rollout_nine_certificate.pdf`; only the first is rebuilt here.

**It RE-SIMULATES. It does not replay stored trajectories.** `v292_fig_nine.py:158-159` calls
`v292_fig_rollouts.roll(...)` once per controller, and `roll` (`scripts/analysis/v292_fig_rollouts.py:66-100`)
loads the registered checkpoint (`load_framework_from_checkpoint`), applies
`v282_agree_gate.gate_overrides` with `eval.max_steps` forced to 400, and rolls the **full 2000-scene
pool** through `rollout_eval` (learned conditions) or `_rollout_lqr_batch` (nominal), then resolves
outcomes with the shipped `step_outcomes` / `resolve_outcome`. The nine drawn scenes are selected out
of that full-pool roll, which is why the roll cannot be narrowed to nine.

Consequently the determinism gate of §2 was required and was run before any rendering.

**The rebuild script: `rebuild_legend.py`, in this directory.** It imports the shipped producer and
calls its own `draw_traj_pair` and `episode`, so no drawing logic is restated or altered; the
figure-1 block is a verbatim copy of `v292_fig_nine.py:174-205` with the output path changed. The
only substantive line is the label override.

## 2. Determinism gate — PASS

**The gate could not be run against `manifest_nine.json` as dispatched.** That manifest pins the
scene lists, the initial conditions, the certificate slice ranges and the panel geometry, but it
carries **no per-scene terminal outcome and no termination time** — the two quantities the gate
needs. Full key list: `figures, target_width_in, column, font_pt, contour_raster_dpi,
certificate_grid, dt_ctrl_s, world_lim, band_collision_limit, eval_max_steps, pool, checkpoint,
checkpoint_step, ledger_row, block1, block2, empty_cell, initial_conditions, certificate_slices,
geometry, slice_condition, no_trajectory_on_certificate_rows`.

**Reference used instead: the registered per-episode vectors of the same cells at the same cap**,
which do carry both, and which the producer's own `persisted()` helper
(`v292_fig_rollouts.py:59-64`) already names for this purpose:

| controller | reference artifact | time field |
|---|---|---|
| nominal | `data/runs/v2.9.2/horizon400_tableI/perepisode__NOM_quadrotor_3d_h400.npz` | `event_step` (−1 → cap) |
| certificate alone | `data/runs/v2.9.2/horizon400_tableI/perepisode__L305_h400.npz` | `n_steps` |
| jointly trained pair | `data/runs/v2.9.2/horizon400_tableI/perepisode__L312_h400.npz` | `n_steps` |

**Result — zero mismatches, on the full pool, on both fields, for all three controllers:**

| controller | n | outcome mismatches | termination-time mismatches | nine drawn scenes |
|---|---:|---:|---:|---|
| nominal | 2000 | **0** | **0** | 9 of 9 match |
| certificate alone | 2000 | **0** | **0** | 9 of 9 match |
| jointly trained pair | 2000 | **0** | **0** | 9 of 9 match |

Per-scene detail for the nine drawn scenes (16, 328, 347, 755, 756, 12, 0, 32, 70 × three
controllers, 27 rows) is in `determinism_gate.json`. The re-simulation is reproducible against the
registered record, so rendering proceeded.

## 3. The change made

Three legend strings, nothing else. Colours, keys, order and line styles are the shipped table's,
unchanged:

| colour | old label | new label |
|---|---|---|
| `#555555` grey | `unfiltered nominal` | **`nominal (unfiltered)`** |
| `#1f77b4` blue | `certificate alone` | **`OC-PNCBF`** |
| `#d62728` red | `jointly trained pair` | **`joint (ours)`** |

Marker entries keep their labels — `start`, `goal`, `collision`, `arrival` — verified present in the
rendered text layer. The old strings are verified **absent**. Nothing was touched in colours, line
widths, markers, panel titles or the `(a)`–`(s)` panel labels.

## 4. Mechanical identity check — PASS

Both PDFs rasterized at **200 dpi** with `pdftoppm`, giving identical **1400 × 1372 px** images.

**Mask** — the union of the two legend bounding boxes, padded 2 px for antialiasing:

| | x0 | x1 | y0 | y1 |
|---|---:|---:|---:|---:|
| new legend (from matplotlib) | 270 | 1130 | 1292 | 1361 |
| old legend (measured as ink in the same band) | 246 | 1151 | 1294 | 1350 |
| **mask applied** | **244** | **1153** | **1290** | **1363** |
| mask in inches | 1.220 | 5.765 | 6.450 | 6.815 |

**Differing pixels outside the mask: 0.** Requirement met.
Differing pixels inside the mask: 12 212. Total differing: 12 212 — every one of them inside the
legend box. `pixel_diff_map.png` shows the difference map; all ink in it lies in the legend band.

The zero count is the strong result: the nine trajectory pairs, the obstacles, the axes, the panel
titles and the panel labels are **pixel-for-pixel identical** to the registered asset.

**One measurement caveat, so it is not mistaken for a change.** `render_report.json` records the
arena panel at 0.9827 in / 8.844 pt·m⁻¹ while `manifest_nine.json` records 0.9817 in /
8.835 pt·m⁻¹. That 0.001 in gap is an artifact of *when* the extent is sampled — this script calls
`fig.canvas.draw()` after `savefig` before measuring, which runs one further constrained-layout
pass, whereas the original measured without it. It is not a difference in the saved output: the
pixel diff above is exactly zero outside the legend, which is dispositive.

## 5. Font floor — PASS

Measured from the PDF's own `Tf` operators after stream decompression:

| | new | old (reference) |
|---|---|---|
| distinct type sizes | 7.0, 8.0 pt | 7.0, 8.0 pt |
| **minimum rendered size** | **7.0 pt** | 7.0 pt |
| text-drawing operations | 213 | 213 |
| all text ≥ 6 pt at print size | **yes** | yes |

No mathtext, no subscript scaling: 7.0 pt is the true floor at print size, 1.0 pt of headroom over
the 6 pt requirement.

## 6. Canvas

**504.00 × 493.92 pt = 7.0000 × 6.8600 in**, identical to the registered asset and to the dispatched
target. PDF is 92 563 bytes against the old 91 638.

## 7. Files in this directory

| file | what it is |
|---|---|
| `fig_rollout_nine.pdf` | **the deliverable** — the rebuilt asset |
| `rebuild_legend.py` | the rebuild script |
| `determinism_gate.json` | §2, including all 27 per-scene rows |
| `pixel_diff.json` | §4, machine-readable |
| `pixel_diff_map.png` | §4, the difference map |
| `font_measurements.json` | §5 and §6 |
| `render_report.json` | legend bbox, canvas, arena panel, the label change |
| `old_200dpi.png`, `new_200dpi.png` | the two rasters the diff was computed on |
| `diff_report.md` | this file |

The old asset diffed against is `data/runs/v2.9.2/paper_figures/fig_rollout_nine.pdf` (91 638 bytes),
which is the registered asset §7.11 of `docs/versions/v2.9.2_results.md` tabulates. It was **not**
modified.
