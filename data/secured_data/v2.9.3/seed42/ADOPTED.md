# v2.9.3 — adopted runs, seed 42

Identity record for the v2.9.3 secured snapshot. **This file is the only place in the
secured set where digests appear** (`06_workflow` §6.1). The originals were **copied, never
moved**, and remain at the paths every report already cites.

**Promoted 2026-08-31**, at the close of v2.9.3, step 3 of the `06_workflow` §2.6 close
sequence. Scope approved by the Researcher: the four runs the manuscript's JCP rows stand on.
**Completed to the full `04_eval` §7.5 standard file set on 2026-08-31**, in the same close.

**Single seed 42 throughout.** No second seed exists for any of these runs, so no
`aggregate/` directory is written.

**Three runs carry all twelve files of the standard set.** The fourth,
`set__20260820-215728__seed42`, carries eleven: its `eval_episodes.csv` is **excluded under
`04_eval` §7.5's hosting-limit clause** at 123431238 bytes, and is recorded below §1 by source path,
byte size and digest.

---

## 1. The four runs

### double_integrator — set__20260819-190133__seed42

- **ledger rows carried:** L356
- **alias:** `COLDSTART40K bounded D_V (double_integrator)`
- **run id:** `v2.9.3__jt__20260819-190133__seed42`
- **source (unmoved):** `data/runs/v2.9.3/set__20260819-190133__seed42/v2.9.3__jt__20260819-190133__seed42`
- **secured path:** `data/secured_data/v2.9.3/seed42/set__20260819-190133__seed42/v2.9.3__jt__20260819-190133__seed42/`
- **adopted checkpoint:** `checkpoints/best.pt`, **step 26700**
- **run state:** `phase` done, `current_step` 40000, `halt_reason` None
- **in-loop `best_cps`:** 0.9162136379653177 — the run's own in-loop pool, **not** the registered-cell `cps`
- **registered-cell `cps_v2` of its ledger row:** 0.8935
- **pool of record:** `data/secured_data/pools/eval_full_di_n2000_seed123456.manifest.json`

| file | SHA256 |
|---|---|
| `config.yaml` | `40fea8a976ee12ba7dfb7f2644bd029485399a7f7a3f99ba08f8564ce69d4969` |
| `git_commit.txt` | `3ff0bfe89ceaa3384e1c5ad6f7710353db7dfc466a90003ddb596cd28ea0dc50` |
| `eval_metrics.csv` | `c6b99a6fba9efe6fff63b5902116e317724a6a9e692a18e86d06e11e7713b9b0` |
| `eval_episodes.csv` | `b687f663d4d8076936a9451b57532913c74fdca1b544598f2fdaf17ecd731c14` |
| `pool_manifest.json` | `fbb7f2389084269c912fa93f0d38b01c1df38d863f97813d1e70b3f08770d2a5` |
| `status.json` | `41096d775093b55f8eb58cae08e9a4b565653d1042299a785321dc91e98cdea6` |
| `report.md` | `2f6607f7a479e01d69594addf026136d31159c4ed910056a21c552f94d95648a` |
| `checkpoints/best.pt` | `3b4f235baca825b52cdc545b900762c386e3d2d4fd55587e1e9278e82b8bdf42` |
| `checkpoints/final.pt` | `63240d71dffe7fdf63a91b76ac212c105a20a3385ecb51645e015d74c5f95578` |
| `figures/trajectory_grid_A.png` | `7f01d7e4475d206f8c2a28d693131fee2d67edc20333e6ad7f02135cad9ed191` |
| `figures/trajectory_grid_B.png` | `6200e458fbc69470f2a4dd5408acb1d5d900525cd5f5b1db4a3fdeaa517503b5` |
| `figures/cbf_contour.png` | `d78665ad231128863e3e7acdbd528ae67a48e352089a4de3270878e0dbeba684` |

### unicycle — set__20260819-190141__seed42

- **ledger rows carried:** L357
- **alias:** `COLDSTART40K bounded D_V (unicycle)`
- **run id:** `v2.9.3__jt__20260819-190141__seed42`
- **source (unmoved):** `data/runs/v2.9.3/set__20260819-190141__seed42/v2.9.3__jt__20260819-190141__seed42`
- **secured path:** `data/secured_data/v2.9.3/seed42/set__20260819-190141__seed42/v2.9.3__jt__20260819-190141__seed42/`
- **adopted checkpoint:** `checkpoints/best.pt`, **step 31050**
- **run state:** `phase` done, `current_step` 40000, `halt_reason` None
- **in-loop `best_cps`:** 0.8555967642099923 — the run's own in-loop pool, **not** the registered-cell `cps`
- **registered-cell `cps_v2` of its ledger row:** 0.8742
- **pool of record:** `data/secured_data/pools/eval_full_unicycle_n2000_seed123456.manifest.json`

| file | SHA256 |
|---|---|
| `config.yaml` | `1f489264500b2061508bc5ce8f4e2d086aeff8de589af0b55c1a065f81fdf769` |
| `git_commit.txt` | `3ff0bfe89ceaa3384e1c5ad6f7710353db7dfc466a90003ddb596cd28ea0dc50` |
| `eval_metrics.csv` | `6c232c1d454e53abc3305706be8186671069fb5917feab7136e8a639abd847bc` |
| `eval_episodes.csv` | `8a34173e62c32cf2d1f0abf02ce1e43f65c2089d99f36784f9f2214f21ca7e3e` |
| `pool_manifest.json` | `1981741b654b5c10da1f21e91dbc6816a5960dc2a744a4a2a6cc7bc4b9c14999` |
| `status.json` | `0f0fa66f6b8ef533ba575c06085afb56712ba2c5d354ed703aefd1979b55a3ae` |
| `report.md` | `b6563ac8d5eb644767249197d9344fedb6749478720604e05b6ccf59f2968668` |
| `checkpoints/best.pt` | `a65549cb8f774b3c2ffa6734c02a76cf0d91ce5333631f604c5bbfaa0f84712a` |
| `checkpoints/final.pt` | `a8af8bd752517c13ce7f48ea3b9722be13e608732644b3a37cb80b983ca0f14e` |
| `figures/trajectory_grid_A.png` | `5984cf60593616a12c7e233acd154c065a0469cf84f880bcdfdb02ef7d4d7e29` |
| `figures/trajectory_grid_B.png` | `6e0c32b19a5799431b18b7477f7513f1f8e388dc5ec0f72b10542366f1802a1c` |
| `figures/cbf_contour.png` | `7a4760f5a45fe315a4480bd33115bc5b9f7e7b9818476a34d57c4fee43363ddb` |

### quadrotor_planar — set__20260820-011541__seed42

- **ledger rows carried:** L358
- **alias:** `COLDSTART40K bounded D_V (quadrotor_planar)`
- **run id:** `v2.9.3__jt__20260820-011541__seed42`
- **source (unmoved):** `data/runs/v2.9.3/set__20260820-011541__seed42/v2.9.3__jt__20260820-011541__seed42`
- **secured path:** `data/secured_data/v2.9.3/seed42/set__20260820-011541__seed42/v2.9.3__jt__20260820-011541__seed42/`
- **adopted checkpoint:** `checkpoints/best.pt`, **step 37050**
- **run state:** `phase` done, `current_step` 40000, `halt_reason` None
- **in-loop `best_cps`:** 0.7750078831367903 — the run's own in-loop pool, **not** the registered-cell `cps`
- **registered-cell `cps_v2` of its ledger row:** 0.7635
- **pool of record:** `data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.manifest.json`

| file | SHA256 |
|---|---|
| `config.yaml` | `b768605208e9b992dbef83edd33b720a9f435b0f33bc137eed3318459a4da032` |
| `git_commit.txt` | `3ff0bfe89ceaa3384e1c5ad6f7710353db7dfc466a90003ddb596cd28ea0dc50` |
| `eval_metrics.csv` | `8dfa97f9d8f7c6318046608cd8f7798cd9f3021f98806483d410e937c045dbad` |
| `eval_episodes.csv` | `165235540a7a12955a3e8b3bc8622249a182b12e494be46c336dd70fc59b2abb` |
| `pool_manifest.json` | `c182e643e80d19dd78d77a659dd57f2a57eded70ccdbb50d6d28cc883531983a` |
| `status.json` | `40d4cc3c342484cbd11e38c5f802a5862a21ccd9d04e6921ded03f7435da4d0d` |
| `report.md` | `6fcd5d27983a1ea6d11ac023303738ddfc232006b1568968ad2fb4eb0c84e741` |
| `checkpoints/best.pt` | `781e4af77d98e898986c3d680e3bf69eca4046bfaf61bf3dbb39fe44c7f2e73f` |
| `checkpoints/final.pt` | `7de310197d673f7cd392224e54ed141a57455df7b9c71ec27175385aff305823` |
| `figures/trajectory_grid_A.png` | `9a6cc2199024cdf8d297ed3c841aad54eff0da1617c9879694d64cf9e5a61a4a` |
| `figures/trajectory_grid_B.png` | `d89b1022d8bcbb9287ed4a8011651c5b6bb5b5de1a484fd133404a35e90d286e` |
| `figures/cbf_contour.png` | `f41eb0b6981853e72c1d8b30553a2978007868a2089d2c3131d9d374140d5e81` |

### quadrotor_3d — set__20260820-215728__seed42

- **ledger rows carried:** L363 (fullcb), L372 (fullvia, BOLD)
- **alias:** `COLDABL approach-term ablation / L363 COLDABL on fullvia`
- **run id:** `v2.9.3__jt__20260820-215728__seed42`
- **source (unmoved):** `data/runs/v2.9.3/set__20260820-215728__seed42/v2.9.3__jt__20260820-215728__seed42`
- **secured path:** `data/secured_data/v2.9.3/seed42/set__20260820-215728__seed42/v2.9.3__jt__20260820-215728__seed42/`
- **adopted checkpoint:** `checkpoints/best.pt`, **step 31500**
- **run state:** `phase` done, `current_step` 40000, `halt_reason` None
- **in-loop `best_cps`:** 0.8761305615315288 — the run's own in-loop pool, **not** the registered-cell `cps`
- **registered-cell `cps_v2` of its ledger row:** 0.8903
- **pool of record:** `data/secured_data/pools/eval_fullvia_quadrotor-3d-d2r_n2000_seed823456.manifest.json`

| file | SHA256 |
|---|---|
| `config.yaml` | `9b1555ee98f56a3f5eafd1cb81d4b3a063fb3cc03e30154adeca5a2dbaf0146a` |
| `git_commit.txt` | `3ff0bfe89ceaa3384e1c5ad6f7710353db7dfc466a90003ddb596cd28ea0dc50` |
| `eval_metrics.csv` | `154b562c1fc8532e06fd5bf26518caa33154aba5b38effeea2fdfb441211264a` |
| `eval_episodes.csv` — **EXCLUDED, see below** | `7ceab59afb950c30552ac2c406df687f936c1094c288b1e6d2082fe2a7ecaed4` |
| `pool_manifest.json` | `9328bca7000948cc8796f68735aa62a6630249fb649a3c53086378a644ab3614` |
| `status.json` | `566635f8b4c03fe4bcac699817ffd1d089d347e117f7d52564f63614696b8233` |
| `report.md` | `31136d81eaa7dacca5d2636945ab81f0d799bbd98d150358fc66d591632eb797` |
| `checkpoints/best.pt` | `72d033c9c2dbf1f52608108c4c28225e6cddf186f21c700921fa2d48cc99c1e7` |
| `checkpoints/final.pt` | `268e02bf6b67f5dbc970edb64cf036388378bd3eb7e32630cfea5228fc080420` |
| `figures/trajectory_grid_A.png` | `5f30cc73fd0cd274ed40d12571bb6128195c6d6ae112449d66e36301250a8766` |
| `figures/trajectory_grid_B.png` | `07a1f3a25141e1197472ed699749b1c1a7c4a8ea7b8cec0ec77ebc4e604459e1` |
| `figures/cbf_contour.png` | `c948f9b07cd3b6ccc40ea939715ff6c75e40447377f0f861a03b3d27bafb8e99` |

**Excluded from this run's secured set, per `04_eval` §7.5**: a member of the standard set that
exceeds the hosting limit for a single tracked file is excluded and recorded here by its source path,
byte size and digest.

| field | value |
|---|---|
| file | `eval_episodes.csv` |
| source path | `data/runs/v2.9.3/set__20260820-215728__seed42/v2.9.3__jt__20260820-215728__seed42/eval_episodes.csv` |
| byte size | **123431238** bytes (117.7 MiB) |
| SHA256 | `7ceab59afb950c30552ac2c406df687f936c1094c288b1e6d2082fe2a7ecaed4` |

**The digest is the one this entry already carried**, computed over the secured copy before it was
removed, and it equals the source file's — so the excluded file stays verifiable against the
original by `sha256sum` on the source path above. **This exclusion applies to this run only**; the
other three runs' `eval_episodes.csv` files are present in their secured sets and their entries are
unchanged.

## 2. Row artifacts, paths preserved

Every score and per-episode artifact the four rows' `eval_source` cells name, copied under
`data/secured_data/v2.9.3/seed42/artifacts/` with the directory structure of
`data/runs/v2.9.3/` preserved. These are **beyond** the `04_eval` §7.5 set, not a substitute for
any part of it.

| artifact (relative to `artifacts/`) | SHA256 |
|---|---|
| `cold_ablation/paired__quadrotor_3d__vs_L359.json` | `2d02ca7813a921ad2f9270ae4260b0b81a2c4794997fcba12d7ef994f50f4d63` |
| `cold_ablation/perepisode__COLDABL_quadrotor_3d_h400.npz` | `569ca7cb84143e0d6ca759b6da4dfd3c2f47e87b30cb0a019ff4b687f56b7a4e` |
| `cold_ablation/score__COLDABL_quadrotor_3d_h400.json` | `197fefdcf4d5a91bfceea7d67d8a80874f5d2addbed2b38eec627932455cec0a` |
| `cold_start_40k/paired__double_integrator__vs_L325.json` | `d8ecbe824cf2aa28bbeb672624fe93ba630886be6448c17c75220504c701c73d` |
| `cold_start_40k/paired__double_integrator__vs_L353.json` | `d82446e9807c8d2dd101c5c082752cc2bfa070566c718d8cd672616891121f36` |
| `cold_start_40k/paired__quadrotor_planar__vs_L327.json` | `3063fc1eaa5362ead14be172b950a2d07d4f1f8152925c84604d33a31e2111f0` |
| `cold_start_40k/paired__quadrotor_planar__vs_L352.json` | `558bf2cc654995544cd06c42e20fe587afaa6cd5f56fec7a3e35c3cfb0272749` |
| `cold_start_40k/paired__unicycle__vs_L326.json` | `e993f740a874e6207329341d9d1bbc2c70129fe23afe08f71e546938a222ea1c` |
| `cold_start_40k/paired__unicycle__vs_L354.json` | `c5605c42dc8231c7d8b13d7e3c8ee401f5f913410776c5bbfbd80c4132665d86` |
| `cold_start_40k/score__COLD40K_double_integrator_h400.json` | `86708ff38d87092de346357657a6aa2a6a88344dc531b099ccb316d214c9a644` |
| `cold_start_40k/score__COLD40K_quadrotor_planar_h400.json` | `6fdb5e77ae252d0d5eca377629291e67177abd2a4f6acde3b59e3724c84a4c2c` |
| `cold_start_40k/score__COLD40K_unicycle_h400.json` | `5fa8fedeb76c9b280bb61d2863283d95b8046580af98d5a980db85a9f0a1b222` |
| `rescore_fullvia/perepisode__L363_COLDABL_h400.npz` | `30d1f100d09fd40a9193669286075204cfba16cad3d76ddef004babce1ccd8b8` |
| `rescore_fullvia/score__L363_COLDABL_h400.json` | `21e2e7d30e662a2f488ab2835aec83923030e9e1c1704ee33e117648be2459b5` |

## 3. Pool manifests

Named by path; the pools themselves live under `data/secured_data/pools/` and were committed
before any number measured on them was reported.

| pool manifest | present |
|---|---|
| `data/secured_data/pools/eval_full_di_n2000_seed123456.manifest.json` | yes |
| `data/secured_data/pools/eval_full_unicycle_n2000_seed123456.manifest.json` | yes |
| `data/secured_data/pools/eval_full_quadrotor-planar_n2000_seed23456.manifest.json` | yes |
| `data/secured_data/pools/eval_fullvia_quadrotor-3d-d2r_n2000_seed823456.manifest.json` | yes |
| `data/secured_data/pools/eval_fullcb_quadrotor-3d-d2r_n2000_seed823456.manifest.json` | yes |

## 4. What this set excludes, per `04_eval` §7.5

The bulky per-step training `metrics.csv` and the TensorBoard event files are **not** copied; the
full training curve stays in the original run directory. **One further file is excluded, under the
same section's hosting-limit clause**: `set__20260820-215728__seed42`'s `eval_episodes.csv`, 123431238
bytes, recorded in §1 by source path, byte size and digest.

Everything else the standard set requires is present in all four runs — `final.pt`, the three
top-level figures, `report.md`, `git_commit.txt`, `pool_manifest.json`, `eval_metrics.csv`, and
`eval_episodes.csv` on the other three runs.

