# WebGPU quality results

The local branch `codex/webgpu-quality-2026-10-13` reduces grid preparation allocations and exposure uploads, fixes resource ownership and loading races, and restores reproducible quality checks. The radial time picker design is unchanged.

The largest tested model starts about 1.07 seconds faster and changes its first selected hour about 32 milliseconds faster. Smaller startup differences are inconclusive. Exposure calculation still dominates startup for the largest model; this work does not establish a large overall startup speedup.

## Matched browser measurements

Each arm contains three sequential captures of all four cases on the same machine. The baseline uses production code from `d3d824f` with corrected collector selection keys. The final treatment uses `5a022e2`, after all implementation changes and validation. Earlier treatment captures from `ec7f355` are preserved locally under `.cache/quality-2026-10-13/intermediate-source-ec7/` and excluded from this table.

| Project and grid | Active points | First visible before, median (range), ms | First visible after, median (range), ms | First hour change before → after, median, ms | Static upload before → after, median, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ben-Gurion, 2 m | 104,364 | 4,115 (3,996–4,466) | 4,182 (4,161–4,261) | 159.7 → 152.6 | 3.9 → 1.0 |
| Ness-Tziona, 2 m | 511,840 | 4,685 (4,575–4,854) | 4,720 (4,684–4,732) | 81.5 → 75.4 | 19.3 → 9.0 |
| Ben-Gurion, 0.5 m | 1,662,323 | 6,864 (6,663–7,060) | 6,865 (6,839–6,938) | 106.8 → 76.7 | 65.4 → 12.5 |
| Ness-Tziona, 0.5 m | 8,171,761 | 23,094 (23,093–23,458) | 22,019 (21,877–22,037) | 180.2 → 147.9 | 461.7 → 69.6 |

For Ness-Tziona at 0.5 m, the observed median changes are 4.7% for startup, 17.9% for the first hour change, and 84.9% for static upload. The three smaller startup ranges overlap; their medians are unchanged or slightly slower. Three samples from one machine do not establish performance across browsers or hardware.

The collector's `initial.firstVisibleMs` measures browser navigation to accepted visible publication. Internal runtime timings are separate. Every matched capture retains identical active point counts and asserts the main route, a shared compute/render device, a GPU-resident selected-hour surface, zero visible selected-hour readbacks, zero selected-hour DataTexture builds, and zero `.bin` requests. The DataTexture counter describes metric surfaces; small color lookup textures still serve GPU rendering.

The machine runs Windows 10, an Intel Core i7-6700K, 16 GiB RAM, and an NVIDIA GTX 970. Chromium 145.0.7632.6 reports an NVIDIA Maxwell adapter with software fallback disabled. Node.js is 24.19.0 and Playwright is 1.58.2. Browser device and description strings are empty; the board model comes from the Windows hardware inventory.

Both arms use headed Chromium, `--enable-unsafe-webgpu`, a 1280 × 720 viewport, trace recording disabled, and a fresh Vite development server on port 4173 for each sample. An HTTP GET to `/` returns 200 before browser navigation, matching Playwright's baseline server readiness check. This compiles the development server before the timed browser load. Browser pages start fresh; these results are not production-host or cold-server measurements. Builds, unit tests, and other browser workloads do not overlap captures.

An earlier treatment collection omitted the HTTP readiness request and included server compilation in its first case. Those captures are retained under `diagnostics/cold-server-only/` and excluded from the comparison. The historical May artifact remains unchanged.

Raw evidence is retained locally in `data/performance-results/quality-2026-10-13/`: `baseline/waterfall-{1,2,3}.json`, `treatment/main-route-cold-start-waterfall-sample-{1,2,3}.json`, `comparison.json`, and `environment.json`. These JSON files are ignored by Git and are not included in a fresh checkout. The measured results and audit reports remain tracked.

## Canonical grid preparation

Direct Float32Array allocation replaces temporary JavaScript number arrays. Coordinates retain the original repeated-addition behavior, axis order, float32 rounding, signed zero, and checksum semantics. All four full-size outputs have matching SHA-256 hashes before and after.

| Grid | Preparation before → after, median, ms | Process peak resident memory before → after, median, MiB |
| --- | ---: | ---: |
| Ben-Gurion, 2 m | 7.19 → 3.16 | 82.6 → 71.7 |
| Ness-Tziona, 2 m | 29.01 → 8.50 | 125.2 → 76.1 |
| Ben-Gurion, 0.5 m | 76.98 → 19.27 | 232.1 → 89.1 |
| Ness-Tziona, 0.5 m | 340.77 → 58.69 | 850.9 → 163.5 |

The largest grid is 5.8 times faster in this isolated benchmark. Each sample runs in a fresh Node.js subprocess with explicit garbage collection before measurement. Peak resident memory includes module loading and output hashing; it measures the Node.js process, not GPU memory or the whole viewer. The benchmark builds the unmasked canonical grid, so some point counts differ from active browser counts.

Run the current implementation from `viewer`:

```powershell
node --import tsx scripts/benchmark-canonical-grid.ts --samples 3 --output ../.cache/canonical-grid.json
```

Use `--module` with a saved source file exporting `canonicalGridPoints` to compare a previous implementation. Raw measurements include source hashes, output hashes, inputs, process memory, and Node.js version.

## Correctness and maintainability changes

- GPU snapshots and temporary uniforms are released on cancellation, failures, and compatibility readbacks. Successful owned outputs remain alive for their consumers.
- Fresh exposure buffers avoid CPU zero uploads. Same-size reused buffers receive ordered GPU clears. Failed device acquisition and device loss no longer leave rejected promises cached.
- Analysis and model loads publish only for the current selection, including the interval while newer metadata is loading. Worker settlement removes listeners, terminates owned workers, and ignores late callbacks.
- BVH serialization includes every material group through a private ungrouped tree when needed. It preserves source geometry, cached trees, index values, and finite draw ranges.
- Main-route UTCI tooltips read one float on demand and cache it by owned snapshot. A first point read transfers four bytes; repeated reads reuse the cache. Debug and comparison consumers retain full readback behavior. A failed render copy falls back to its accepted snapshot.
- CPU fallback preserves compact active-cell geometry and the accepted range in both color modes. The explicit DataTexture override also preserves masks and holes with a packed texture, shared quad geometry, and one instance per active point.
- Legacy CPU-to-GPU transitions reject incompatible color storage before inspecting compute storage. This fixes the legacy debug GPU path and non-August diagnostic hangs.
- Local data requests stay within the data root after lexical and real-path checks, including Windows traversal and symlink escapes.
- Vitest discovers unit tests separately from Playwright. Type checks, unit tests, and production builds run before deployment. Locked development types match the installed Three.js and WebGPU interfaces.
- Legacy imports avoid optional thermal dependencies until needed. Vector and iterative UTCI paths agree on weather padding and reject invalid shapes. Legacy statistics and metadata exports reject unavailable UTCI ranges and nonstandard JSON values.

## Verification

The full viewer unit suite passes: 138 files, 1,141 tests, and five existing environment-dependent skips. Type checking reports zero errors and warnings, and the production build succeeds. All 107 Python tests pass. Baseline unit discovery had 21 Playwright/Vitest collisions and six failed assertions; these are repaired.

Forty-five targeted browser cases pass across the final runs. Thirteen dedicated hardware and scientific cases pass without skips or retries. They cover grouped geometry, reused exposure buffers, compact range reductions, delayed project loading, localized tooltip reads and cache invalidation, monthly shade periods, and forced GPU-copy failure in both color modes. The failure tests require a real snapshot readback, a matching CPU legend, and subsequent GPU recovery. Seven main-route diagnostic cases also pass, including camera and hover budgets on the 8.17-million-point model and separate comparison-readback accounting. A fresh 25-case suite passes selected-hour, legacy debug, query changes, bootstrap identity, and interactive main/debug equivalence checks. The three final four-dataset timing runs also pass all collector assertions.

The existing Ben-Gurion reference checks retain their tolerances. These compare distribution means and maxima, not every reference point:

| Stage | Difference in mean | Difference in maximum | Existing tolerance, mean / maximum |
| --- | ---: | ---: | ---: |
| Solar exposure | 0.000000399 | 0 | 0.02 / 0.05 |
| Sky exposure | 0.001050 | 0.001714 | 0.02 / 0.05 |
| MRT, °C | 0.003692 | 0.025703 | 1 / 2 |

The selected-hour f32 output also passes the existing comparison against its full-hours slice at an absolute per-value tolerance of `1e-5`. Raw scientific and main-route evidence, including source hashes, is retained in `verification/` under the measurement directory. Full command logs and task reviews remain locally under `.cache/quality-2026-10-13/`.

Independent integrated review identified missing deferred readback accounting, unsupported active-cell CPU fallback, and stale exported ranges in live snapshots. All three have regression tests and fixes. Follow-up review also approves the legacy storage guard, compact DataTexture transport, and observed debug-environment metadata. The [final code review](2026-10-13-webgpu-quality-review.md) and [independent evidence audit](2026-10-13-webgpu-quality-evidence-audit.md) are saved with this report. The evidence audit reproduces the timing tables, verifies artifact hashes, and checks the scope of the scientific and memory claims.

The real-GPU exposure regression uses a grouped box, independent expected sun and sky rays, and a second upload with zero sun vectors. It confirms coverage of all six faces, unchanged source geometry, cleared reused solar bits, zero CPU writes to exposure buffers, and no uncaptured GPU errors. An opt-in reproduction of the former first-root serializer fails the same assertions.

## Remaining performance work

The largest model spends roughly 17 seconds in exposure computation, which this branch preserves. A further material startup improvement needs measured GPU dispatch work or a validated exposure cache. The current 288-hour weather convention represents 12 monthly days; it is not a full 8,760-hour year. Future Grasshopper integration must preserve coordinate, mask, period, and ownership contracts and establish its own representative geometry and performance cases.
