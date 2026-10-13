# Independent integrated branch review

Review base: `d3d824f`. Reviewed working tree on `codex/webgpu-quality-2026-10-13`, including committed production changes through `e645c15`, the draft results, and the two new browser specifications.

## Verdict

APPROVED by source review after correction of all three P2 findings. No unresolved production finding or P0/P1 issue remains in this review. This review did not run GPU, browser, build, or whole-suite workloads; the parent owns those checks. Approval does not establish hardware acceptance: final browser fallback/recovery, parity, and integrated checks remain separate completion requirements. The reported broader debug-browser failures must be classified against baseline before final completion claims.

## Finding

### P2: Publish deferred fallback readback accounting to the controller

Location: `viewer/src/lib/compute/selected-hour/liveSelectedHourController.ts:1938-1944`, with the publication at approximately line 1989.

When a main-route GPU copy fails, localized UTCI now performs its first full selected-hour read inside `loadCpuFallback`. The session records `visible-fallback` on the original result diagnostics (`liveUtciSelectedHourSession.ts:423`, `769-770`). The controller already copied the reason array and count object when accepting the GPU result (`liveSelectedHourController.ts:1322-1329`, `1704-1707`). The deferred fallback handler takes only `fallback.analysis` and never refreshes those copied fields. Consequently, rendering can recover through CPU upload while the exposed `selectedHourReadbackReasons` and `selectedHourReadbackReasonCounts` still report no visible fallback read.

This is an integration gap in the new default lazy path. It makes diagnostic evidence inconsistent with actual transfers and with the new browser fallback assertion. The separate `visibleSelectedHourReadbackCount` becomes one after CPU publication, so that field does not repair the reason-level accounting.

Fix: retain the session result diagnostics in `DeferredCpuFallbackState` or return an explicit diagnostics snapshot with the fallback. After the awaited fallback succeeds and ownership is rechecked, publish its updated reason array/counts with the CPU analysis. Do not increment unconditionally: an eager debug/comparison fallback can reuse existing CPU values without another readback.

Regression: extend `viewer/tests/compute/live-selected-hour-controller.test.ts:1004` (the geometry-only lazy fallback case) so the fallback updates the source diagnostic objects as `recordSelectedHourReadbackReason` does. Assert that controller state publishes `visible-fallback: 1`. Also assert that eager fallback does not acquire an extra readback count and stale completion cannot update a newer request. The session-only deferred accounting test at line 1788 does not cross the controller copy boundary.

### P2: CPU fallback cannot render the active-cell layouts used by the main route

Location: `viewer/src/lib/components/scene/UTCIPointCloud.svelte:1383-1392`, calling `viewer/src/lib/services/pointCloudService.ts:566-568`.

The saved real-browser failure (`task-13-fallback-diagnostics.json`) records one injected render-copy failure, a successful 417,456-byte snapshot readback, CPU transport, no surface source, and no new visible acknowledgement. The fallback analysis retains `base.metadata.activeMask` through `buildSelectedHourLiveAnalysis`. `buildUtciGridLayout` recognizes it and returns `renderTopology: 'active-cells'` (`pointCloudService.ts:1644-1664`). However, `createUtciSurfaceMeshInternal` immediately requires `dense-grid`, throwing `UTCI active-cells layouts require the compute-buffer surface path.` The renderer disposes its compute surface before calling that constructor, leaving no replacement mesh to publish.

The host correctly substitutes `baseControllerState.analysis` into the requested render context (`liveSelectedHourRouteHost.ts:922-934`). This is a renderer topology limitation, not a stale host-context problem. It predates this branch's fallback refactor, but it prevents the claimed recovery behavior on the actual main-route datasets and blocks the hardware acceptance gate.

Fix: implement a bounded CPU-uploaded active-cell surface using the compact instanced geometry and one-float-per-active-point storage, populated from the accepted fallback values. Identify it as CPU-uploaded for `buildCpuPublicationDiagnostics`. Preserve active-mask mapping, display range, and recovery to subsequent compute-buffer output. Do not remove activeMask merely to force the dense renderer: that restores large forbidden dense allocations on the largest dataset.

Regression: exercise the real renderer/service construction with a valid active mask and fallback UTCI values. Assert that construction succeeds, the compact topology and active point mapping remain intact, and CPU publication identity matches the failed request. Follow with the parent's native-copy-fault browser case and next-hour GPU recovery. Existing controller mocks verify state transitions but never call the failing surface constructor.

### P2: Discrete fallback can replace its accepted range with stale exported hour statistics

Location: `viewer/src/lib/services/pointCloudService.ts:615-624`, with inherited metadata in `viewer/src/lib/compute/selected-hour/liveUtciSelectedHour.ts:43-49`.

The new active CPU resolver calls `getUtciRangeForDisplay`, whose discrete path prefers `metadata.hour_statistics[hourIndex]`. Live snapshot builders spread the source metadata and replace `utci_range`, but leave historical `hour_statistics` intact. The session deliberately passes its current selected-hour GPU range as `utci_range` to deferred fallback. Therefore, discrete GPU-to-CPU recovery can change the color mapping to the historical exported day's range, especially after selecting a different month.

The actual Ben-Gurion metadata has 24 hourly statistics. Hour 1 is 22.748091519697013 to 24.370928469393206. A live snapshot built with selected values/range 30 to 40 still resolves the historical interval when the active CPU surface asks for discrete hour 1. Normalized fallback uses the authored `utci_range` and is unaffected.

Fix: do not inherit incompatible historical hourly statistics into a live single-hour snapshot, or explicitly prioritize its authored selected display range. Preserve ordinary full-day hourly range behavior. Test the snapshot builder together with the discrete display-range resolver and active CPU surface, using historical hourly statistics that differ from the accepted live range. The new generic surface test exercises ordinary hourly precedence, so it does not cover this snapshot ownership boundary.

## Re-review of the original fixes

- Both live UTCI snapshot builders now explicitly omit inherited `hour_statistics`. Their authored `utci_range` therefore governs discrete fallback and geometry-only display. The regression passes an exported 20-to-25 hourly range and a live 30-to-40 range from another month through the public builders and actual range resolver. It also checks that the source full-day analysis retains its original statistics. Ordinary full-day surface range behavior remains unchanged. This resolves the third finding by source inspection.
- The controller retains the original result diagnostics in its deferred fallback state and copies current reasons/counts in the owned CPU publication. The regression fixture uses the actual immutable-array/count replacement pattern from `recordSelectedHourReadbackReason`. Eager fallback retains its existing tooltip count without inventing a second read. Existing stale completion ownership checks remain before publication. This resolves the first finding by source inspection.
- The active CPU path uses the same private compact constructor, instanced canonical-index geometry, scalar storage, and metric color nodes as the compute path. It owns a copy of fallback values, identifies itself as CPU-uploaded, and has no pending GPU source. Value-count validation occurs before allocation. Updates reject incompatible masks, dimensions, coordinate systems, metric LUTs, and compute-backed state. Existing backend disposal sees and disposes its LUT, material, and geometry. Subsequent GPU publication recognizes it as a non-compute surface and recreates the owned compute path. This resolves the constructor rejection by source inspection, subject to real-browser acceptance.
- The default resident constructor retains its original allocations and node construction. No default-path performance regression was identified from the focused changes; hardware acceptance remains the parent's responsibility.

## Integrated checks performed by inspection

- Canonical grid allocation retains repeated-addition axes, insertion order, Float32 conversion, and checksum operations. The benchmark hashes output bytes and reports process peak memory separately from GPU memory.
- Snapshot and transient-uniform cleanup uses `finally` without destroying successfully returned owned outputs. Device acquisition metadata remains attached to its own promise; old loss/rejection callbacks cannot clear a newer acquisition.
- Fresh exposure allocations use WebGPU zero initialization; same-size reuse submits clears before subsequent queue work. The new grouped-box GPU test has explicit geometric expected values and a legacy first-root failure mode.
- Model ownership is captured before asynchronous material processing. Route ownership advances before newer metadata resolves, and the keyed model is recreated when that metadata gains ownership. Analysis cache keys distinguish data roots and metadata/full loads.
- Worker cleanup covers pre-abort, timeout, error, transfer failure, and success. Grouped/indirect BVHs rebuild a private index, preserve finite draw ranges, and have tests that inspect reachable leaf triangles and bounds.
- Main-route localization applies to the base controller; comparison requests retain eager behavior. Tooltip cache keys include snapshot identity, and pending reads are invalidated when the requested UTCI hour, month, or metric changes. Monthly Shading Index validation uses a monthly period.
- Deferred fallback reads the accepted snapshot, validates its handle before and after mapping, and preserves the selected display range. Controller ownership checks prevent stale fallback publication.
- Data-route checks reject lexical escapes and resolved paths outside the data root. Tests use real ordinary files and isolate filesystem canonicalization for the Windows symlink case.
- Python validation runs before vector/iterative selection. Import-order tests use fresh subprocesses. Weather tests include real UTCI path parity; metadata serialization validates strict JSON before opening an existing output file.
- CI restores type/unit/build checks, and unit discovery excludes Playwright specifications. No unnecessary production abstraction or broad style rewrite was identified.

## Evidence and claim limits

Final task-20 source and harness re-review is approved. `observedEnvironment` overrides only browser GPU availability and the renderer backend reported by the Scene callback. Nullish precedence preserves explicit `false` and `unknown`, so stale or empty runtime snapshots cannot replace the current observations. Transport, surface source, device sharing, and readback accounting retain their original authorities; the CPU regression explicitly remains outside the strong GPU contract. The inverse regression checks that unavailable current observations replace stale positive metadata.

The plain/automatic debug harness now accepts either a published compute surface or a completed CPU upload with positive transfer/readback counts and matching selected/completed time. This reflects the existing automatic bootstrap fallback without claiming that CPU transport is GPU-resident. The separate explicit GPU parity case still requires matching selected month/hour/time, GPU source and transport, device sharing, zero visible readbacks, and the strong runtime contract. Initial publication waits prevent UI interactions from racing bootstrap. No new actionable production or harness finding was identified. The reported three-case native-browser pass and final whole-suite/build/performance results remain owned by the parent; this final pass ran no browser or heavyweight commands.

Additional source re-review: the legacy CPU-to-GPU compatibility fix is approved. CPU color-storage state now returns a wrong-source snapshot before accessing compute-only scalar storage. The existing compatibility evaluator rejects that source before comparing storage counts, so the zero sentinel does not permit accidental reuse. The regression constructs a real legacy CPU mesh, verifies rejection without mutation or a missing-attribute exception, and verifies that a replacement compute mesh is compatible.

The reviewed browser-harness updates preserve their production assertions. Debug selection keys match the debug route's own key format; the main route retains metric-qualified keys. Diagnostics replace removed UI status elements, and hover/camera/selection/parity checks remain. The passive bootstrap observer records original diagnostic assignments before asynchronous polling can miss them, then requires the recorded pending identity to become the same accepted GPU identity with zero visible readbacks. It does not synthesize success or relax the completed-path assertions.

The subsequent compact active dataTexture path also passes source review. Packed RGBA entries follow active-point order, while instance position uses canonical indices; this preserves holes without a canonical-sized color array. Integer division/modulus and texel-center UVs address successive rows, nearest filtering prevents adjacent point mixing, and unrendered padding remains outside the instance count. Shape and RGB length validation precede allocation. Colors use the existing sRGB color mapper and an sRGB texture. Local Three.js source inspection confirms that the explicit color node replaces the default material-map color expression, avoiding double color multiplication; the map reference permits existing disposal to release the same texture.

Updates retain texture identity only for matching active topology, mask signature, point count, dimensions, grid size, and coordinate system. `canReuseUtciSurfaceLayout` now rejects a topology change before accessing topology-specific storage. Material, map, and geometry disposal remain on the existing dataTexture path. The shared position-node extraction preserves the compute path's exact column, row, coordinate-system inversion, cell size, and local offset expressions. No new production finding was identified. This source approval covers the reviewed task-19 changes; the parent owns the reported hardware runs and final integrated checks.

The draft results correctly distinguish matched development-server startup from cold server compilation and production hosting. They qualify three samples on one machine, overlapping smaller-case ranges, and Node process memory. They also preserve the excluded unmatched captures and avoid translating the isolated grid speedup into a large startup claim.

The real-GPU grouped-box specification checks all faces, reused solar clearing, CPU-write counts, source geometry, and uncaptured errors. The new main-route specification covers delayed old GLB delivery, tooltip read size/cache invalidation, and forced render-copy fallback, but its actual execution result is owned by the active acceptance work. Final documentation must report the final results rather than infer passes from these specifications.
