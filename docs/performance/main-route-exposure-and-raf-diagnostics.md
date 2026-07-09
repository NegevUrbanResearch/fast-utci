# Main Route Exposure And RAF Diagnostics

## Scope

- Route: `/`
- Analysis: Ness Tziona 0.5m is the decision case.
- Purpose: separate desktop breathing during exposure from page-local render-publication rAF pain.
- Non-goals: chunk-size tuning, lazy/background exposure fill, render-publication yielding, queue-drain removal.

## Proof Boundary

- Artifact: `data/performance-results/main-route-exposure-and-raf-diagnostics.json`
- Source route: `/`
- Decision case: `ness-tziona-0_5m`
- `rendererBackend`: `webgpu`
- `utciSurfaceSource`: `compute-buffer-selected-hour`
- `baseRenderTransport`: `compute-buffer-selected-hour`
- `baseSameDeviceForComputeAndRender`: `true`
- `visibleSelectedHourReadbackCount`: `0`
- `dataTextureBuildCount`: `0`
- Page errors / request failures / crashes: `0 / 0 / 0`
- Forbidden comparison status: no forbidden comparison fields were present in this focused artifact.

## Exposure Breathing Profile

- `exposurePrecomputeMs`: `17213.0 ms`
- Scheduler mode: `chunked`
- `maxWorkgroupsPerSlice`: `2048`
- Slice count / submit count / yield count: `63 / 63 / 62`
- Queue wait total / max / average / min: `16996.7 ms / 378.0 ms / 269.8 ms / 63.9 ms`
- Encode total: `7.9 ms`
- Yield wait total / max / average: `201.2 ms / 7.3 ms / 3.2 ms`
- Post-rAF timeout max / average: `0.6 ms / 0.3 ms`
- Exposure-overlapped top gaps: among the top listed gaps, the largest exposure-overlapped rAF gap was `354.8 ms` on slice `31`. The top listed interval gaps and long tasks had `0` exposure-slice overlap.

Interpretation: the browser-page yield points were short, so the artifact does not prove a page-local rAF blockage during exposure. The `17.0 s` queue-wait total across `63` GPU submits still keeps the desktop-breathing hypothesis plausible as GPU/driver/system pressure, but that remains an inference rather than a direct OS-level proof.

## Render-Publication RAF Profile

- Top overall rAF gap: `1348.7 ms`, from `4314.5` to `5663.2`, before exposure/render publication.
- Largest render-publication-overlapped rAF gap: `1314.1 ms`, from `23613.8` to `24927.9`.
- Top overall interval gap: `1385.2 ms`, from `4284.3` to `5669.5`, before exposure/render publication.
- Largest render-publication-overlapped interval gap: `1351.2 ms`, from `23582.4` to `24933.6`, overlapping render-publication windows plus `renderCopyQueueDrain`.
- Top render-publication long task: `1080 ms`, overlapping render-publication windows.
- Controller publish to scene sync complete: `1662.3 ms`
- Layout build: `520.7 ms`
- Layout reuse decision / key / proof: `804.5 ms / 283.7 ms / 0.4 ms`
- Layout publication plan: `520.7 ms`
- Mesh/surface creation: `263.4 ms`
- Storage init wait: `245.0 ms`
- `renderPublicationPreStorageMs`: `1067.9 ms`
- First render-storage wait frame: `244.8 ms`
- Compute-buffer copy submit: `0 ms`
- `renderCopyQueueDrainMs`: `347.8 ms`
- First render/backend init: `requestDeviceCalls=1`; storage wait later saw device/backend entries available.
- `renderSceneSyncTotalMs`: `1662.3 ms`
- `renderUpdateMs`: `2204.5 ms`

Interpretation: page-local rAF/render-publication pain is proven, but the single largest overall rAF and interval gaps in this run are earlier startup/data-prep gaps. The largest render-publication-overlapped rAF, interval, and long-task windows land after exposure and overlap render-publication windows, not exposure slices.

## Ranked Owners

1. Exposure GPU/driver saturation / breathing
   - Evidence: `17213.0 ms` exposure precompute, `16996.7 ms` queue wait, `63` submits, average queue wait `269.8 ms`.
   - Confidence: medium for desktop breathing, low for page-local rAF.
   - Falsifier: OS/GPU profiler shows desktop remains responsive and GPU queues are not saturated during exposure.

2. Early startup/data prep before controller run
   - Evidence: largest long task, `1356 ms`, occurs before render publication and does not overlap exposure slices.
   - Confidence: medium.
   - Falsifier: a startup/data-prep trace attributes this entirely to benign initialization outside user-visible pain.

3. Layout/proof/key construction
   - Evidence: `renderLayoutBuildMs=520.7 ms`, `renderLayoutReuseDecisionMs=804.5 ms`, `renderLayoutReuseKeyMs=283.7 ms`, `renderLayoutPublicationPlanMs=520.7 ms`, `renderLayoutReuseProofMs=0.4 ms`.
   - Confidence: high for layout/key as a major render-publication owner, low for proof itself.
   - Falsifier: a follow-up render-publication trace shows layout/proof is not on the overlapped rAF path.

4. Mesh/surface creation
   - Evidence: `renderSurfaceMeshMs=263.4 ms`; also part of `renderPublicationPreStorageMs=1067.9 ms`.
   - Confidence: medium.
   - Falsifier: detailed pre-storage marks show mesh/surface creation is small and another pre-storage operation dominates.

5. Three/WebGPU storage initialization
   - Evidence: render-publication rAF and interval gaps overlap `renderStorageFirstWaitFrame` and `renderStorageWait`; first wait frame is `244.8 ms`.
   - Confidence: high.
   - Falsifier: storage wait disappears while the same rAF gap remains.

6. Compute-buffer copy submit
   - Evidence: `renderBufferCopyMs=0 ms`; queue drain is separately `412.1 ms`.
   - Confidence: low as a primary owner.
   - Falsifier: copy submit becomes non-zero/large in a repeated run while queue drain/storage wait do not explain the rAF gap.

7. Queue drain
   - Evidence: `renderCopyQueueDrainMs=347.8 ms`, overlapping the render-publication interval gap tail.
   - Confidence: high for a secondary render-publication stall.
   - Falsifier: queue drain falls near zero while the tail rAF gap persists.

8. First render/backend init
   - Evidence: `requestDeviceCalls=1`; storage wait later recorded device/backend entries available.
   - Confidence: low as a post-controller owner.
   - Falsifier: backend init repeats or shifts into the `23613.8-24927.9 ms` rAF gap in a repeated run.

## Recommendation

Current note: this recommendation is historical to the artifact. Later strategy work widened the next-step framing: rank early startup/pre-exposure, exposure breathing, and render-publication freezes together before choosing an optimization. Do not read this section as a current instruction to start only with render-publication work.

The diagnostic pass separates the lanes:

- Exposure breathing remains a plausible GPU/driver/system-saturation owner, but this artifact does not directly prove OS-level desktop freeze.
- Page-local rAF/render-publication pain is proven and should be treated as an independent owner.

Next implementation work should target render-publication diagnostics/optimization first, especially pre-storage/storage-wait and queue-drain ownership. Any exposure-breathing fix should require either external OS/GPU profiling or another repo-local proxy that can observe system-level responsiveness during the `17.3 s` exposure window.

## Verification Trail

Executed in this order:

1. `cd viewer && npm test -- --run tests/e2e/main-route-exposure-scheduler-collectors-source-lock.test.ts`
   - Result: `1` file passed, `5` tests passed.
2. `cd viewer && npm run check`
   - Result: `svelte-check found 0 errors and 0 warnings`.
3. `cd viewer && npx playwright test --config=playwright.collect.config.ts tests/e2e/main-route-visual-freeze-map.spec.ts --project=chromium --workers=1 --reporter=list --timeout=900000`
   - Result: `1 passed (53.0s)`.
4. `cd .. && node -e "const fs=require('fs'); const p='data/performance-results/main-route-exposure-and-raf-diagnostics.json'; const a=JSON.parse(fs.readFileSync(p,'utf8')); const nz=a.cases.find(c=>c.caseId.includes('ness-tziona')&&c.gridResolutionMeters===0.5); if(!nz) throw new Error('missing NZ 0.5m case'); const d=nz.raw.finalDiagnostics; const proof=d.selectedHourRuntimeContract||{}; const t=d.timings||{}; const timeline=t.renderPublication?.renderPublicationTimeline||{}; if(a.sourceRoute!=='/') throw new Error('wrong route'); if(d.rendererBackend!=='webgpu') throw new Error('not webgpu'); if(d.utciSurfaceSource!=='compute-buffer-selected-hour') throw new Error('wrong surface'); if(d.baseRenderTransport!=='compute-buffer-selected-hour') throw new Error('wrong transport'); if(d.baseSameDeviceForComputeAndRender!==true) throw new Error('not same device'); if(proof.visibleSelectedHourReadbackCount!==0) throw new Error('visible readback'); if(t.exposureSchedulerBreathingTrace==null) throw new Error('missing exposure breathing trace'); if(!Array.isArray(t.exposureSchedulerBreathingTrace.allSliceWindows)||t.exposureSchedulerBreathingTrace.allSliceWindows.length===0) throw new Error('missing all exposure slice windows'); if(timeline.renderPublicationPreStorageMs==null) throw new Error('missing render pre-storage split'); console.log(JSON.stringify({caseId:nz.caseId, firstVisible:nz.summary.firstSelectedHourVisibleMs, exposureTrace:true, sliceWindows:t.exposureSchedulerBreathingTrace.allSliceWindows.length, topRaf:nz.summary.topRafGapMs, topLong:nz.summary.topLongTaskMs, pipelineFirstVisible:nz.summary.finalTimingBuckets.pipelineFirstSelectedHourVisibleMs, hasAmbiguousTimingName:Object.prototype.hasOwnProperty.call(nz.summary.finalTimingBuckets,'firstSelectedHourVisibleMs')}, null, 2));"`
   - Result: `caseId=ness-tziona-0_5m`, `firstVisible=25573.30000001192`, `pipelineFirstVisible=19988.5`, `exposureTrace=true`, `sliceWindows=63`, `topRaf=1522.5`, `topLong=1369`, `hasAmbiguousTimingName=false`.
5. Wrote this evidence note after artifact validation.
6. `rg -n "TBD|TODO|smaller chunks|lazy/background|cooperative render-publication" docs/performance/main-route-exposure-and-raf-diagnostics.md docs/superpowers/plans/2026-05-31-main-route-exposure-and-raf-diagnostics.md`
   - Result: matches only non-goals or plan exclusion text.
7. `git diff --check`
   - Result: only LF/CRLF warnings, no diff-check errors.
