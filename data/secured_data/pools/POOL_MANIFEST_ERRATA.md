# Pool manifest errata — quadrotor_planar eval pools

Recorded at the v2.6.2 close (2026-07-16/17). Read-only erratum; the manifests are **NOT rewritten** (they are
sha-pinned and referenced by every secured ADOPTED.md). Flag Researcher.

## The inconsistency
`eval_full_quadrotor-planar_n2000_seed23456.manifest.json` (and the identical run-dir `pool_manifest.json`
copies) record `sampler_params.env`:
- `v_init_max = 0.5`
- `bounds.tau_max = 0.2`
- (created_at 2026-07-15, git_commit `f5edc25…` = v2.5.2 era)

But the **actual pool contents** (measured directly from the `.pkl`) are:
- `‖v0‖` max **1.4999**, p99 1.4921, mean 1.0282  → matches the CURRENT config `v_init_max = 1.5`, NOT 0.5.
- `|θ0|` max **3.139 ≈ π**; `|ω0|` max **0.9998** → matches `omega_init_max = 1.0`.
- The eval plant during rollout is **tau_max = 1.0** (v2.6.1+ corrected plant), NOT 0.2.

The `.pkl` **matches** its manifest's `pool_sha256`
(`92df837bad44658dd9a1755df19b39c6c4bb1be1bc6b7169c7e92baaf0dec531`), so the manifest and pool are the same
object — the `sampler_params` block is **internally inconsistent with its own pool contents** (stale
v2.5.2-era metadata carried into the manifest at build time).

## Impact
**None on comparability.** The pool is FROZEN (sha-pinned) and has been used identically across
v2.6.0 / v2.6.1 / v2.6.2 for the quadrotor headline and every diagnostic. The stale `sampler_params` is a
provenance/metadata bug only: **the manifest cannot be trusted for IC-distribution provenance** — read the IC
distribution from the pool itself (‖v0‖≤1.5, |θ0|≤π, |ω0|≤1.0), not from `sampler_params`.

Same caveat applies to the inloop pool `eval_inloop_quadrotor-planar_n500_seed12345.pkl`
(sha `4c8af29c550bce8102333ff4504886e4256cd50842bc08354141475115467e19`), whose manifest carries the same
`v_init_max=0.5 / tau_max=0.2` block; its ICs should likewise be read from the pkl.

**Recommended (Researcher):** on the next pool regeneration, write `sampler_params` from the live config so the
manifest matches its contents; do NOT edit these frozen manifests in place (would break the sha pins).

## Reach-set provenance note (related re-roll bookkeeping)
Re-rolls of the brake-envelope `best.pt@30000` on this full pool report the goal (reach) count as:
- **1807** — concordant across THREE independent re-rolls (`quadrotor_recov_validate.json`,
  `quadrotor_doom_census_A.json`, and the gravity-observability roll; all with born-doomed 72, collision 123).
- **1815** — the single outlier in `quadrotor_recov_fp_anatomy.json` (with born-doomed 73).

Same checkpoint + same pool; the ≤8-episode gap (+1 born) is attributable to **GPU-nondeterministic float32
rollout drift** at outcome boundaries (~0.4% of 2000 episodes) — *inferred* from the 3-vs-1 concordance, not
separately proven. Every rate is computed within its own roll, so no diagnostic conclusion depends on the gap.

Cross-ref: `docs/versions/v2.6.2/close_facts.md` §5, §7.
