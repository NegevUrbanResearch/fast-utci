# WebGPU Compute-On-Demand Prototype Results

Date: 2026-05-08

## Environment

- Browser: Chromium via Playwright
- WebGPU available: yes in fresh controller-side strict route capture
- Strict route status: `On-demand prototype: ready utciRender auto -> gpuNative (webgpu)`
- GPU adapter: not exposed by runtime
- Renderer/backend: `webgpu` in fresh strict route capture
- Shared device proven: indirect only; synthetic bridge coverage and main-route diagnostics do not yet prove direct adapter/device identity or zero-copy interop
- `maxStorageBufferBindingSize`: not exposed by runtime
- `maxBufferSize`: not exposed by runtime
- `maxStorageBuffersPerShaderStage`: not exposed by runtime
- Model: `data/3d_models/original_with_layers.glb`
- Scenario: not exposed by runtime
- Grid resolution: not exposed by runtime
- Point count: `104445`
- Time index: `12` in the strict route runtime object; compared hours `12`, `23`, `16`, and `17` in the separate-baseline multi-hour capture

## Gate Results

| Gate | Result | Evidence |
| --- | --- | --- |
| Same-device gate | indirect pass | Strong but indirect browser evidence. Focused E2E passed under a successful `webgpu` renderer backend, including the synthetic bridge case, but no adapter/device identity capture exists yet. |
| Synthetic bridge color variance | pass | `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts` verifies `bridgeAttached === true` and `visibleColorVariance > 0` on the debug route. |
| No-hot-path-readback bridge smoke | pass | The same synthetic bridge E2E verifies `debugReadbackCount === 0` and `dataTextureBuildCount === 0`. |
| Exposure-only precompute | pass | Fresh strict route runtime capture reports `path=exposure-only-f32`, `usedExposureOnlyPrecompute=true`, `allHoursUtciBytesAllocated=0`, `allHoursMrtBytesAllocated=0`, `oneHourOutputBytes=417780`, `pointCount=104445`, `exposurePrecomputeMs=706.1000000238419`, and `oneHourDispatchMs=3.199999988079071`. |
| One-hour f32 parity | pass | Controller-side verification passed for both focused prototype E2E (`7` tests) and the strict vertical-slice E2E (`6` tests). The multi-hour capture also recorded `maxAbsDiff=0` and `rmse=0` for hours `12`, `23`, `16`, and `17`, including point `31079` hours `16` and `17`. |
| Packed precision/category gate | deferred | Milestone 5 is deferred. Packed output validation is not complete, and this prototype result does not claim it is. |

## Timings

| Phase | ms |
| --- | ---: |
| Exposure-only precompute | `706.1000000238419` |
| One-hour f32 dispatch | `3.199999988079071` |
| One-hour packed dispatch | deferred |
| GPU output to render visible | not exposed by runtime |
| Debug readback only | not exposed by runtime |

## Decision

- [x] Use `f32-utci` as the next candidate for production-integration planning. Fresh runtime/timing capture and controller-side verification now exist; the remaining uncertainty is narrower and centers on future user verification plus direct interop/zero-copy proof, not on missing runtime capture.
- [ ] Proceed to production integration with `packed-mrt-utci`.
- [ ] Keep prototype only; fix bridge/performance/precision issues first.

## Notes

- Fresh controller-side runtime capture now exists for the strict route, with `path=exposure-only-f32`, `usedExposureOnlyPrecompute=true`, `allHoursUtciBytesAllocated=0`, `allHoursMrtBytesAllocated=0`, `oneHourOutputBytes=417780`, `pointCount=104445`, `exposurePrecomputeMs=706.1000000238419`, and `oneHourDispatchMs=3.199999988079071`.
- The same controller-side capture recorded point `31079` with hour `16 = 8.452316284179688` and hour `17 = 7.999854564666748`, both with `diff = 0` against the separate `runAll()` baseline.
- Main-route controller capture still shows `utciSurfaceSource=cpu-uploaded-selected-hour`, `selectedHourTransferCount=1`, and `dataTextureBuildCount=0`; the explicit fallback route still resolves to `dataTexture` with `dataTextureBuildCount=1`.
- Strict focused E2E currently passes 7 tests in `viewer/tests/e2e/webgpu-on-demand-prototype.spec.ts`, covering:
  - main route default `gpuNative` diagnostics
  - main route `dataTexture` override diagnostics
  - same-route diagnostics update/clear
  - debug route prototype diagnostics
  - debug route explicit render override diagnostics
  - synthetic bridge diagnostics
  - one-hour `f32` parity against the all-hours slice
- Final controller-side compute verification in `viewer/` passed for `tests/compute/onDemandDiagnostics.test.ts`, `tests/compute/onDemandSizing.test.ts`, `tests/compute/onDemandOutputFormat.test.ts`, `tests/compute/compute-manager-on-demand.test.ts`, `tests/compute/webgpu-on-demand-source-locks.test.ts`, `tests/compute/gpu-pipeline.test.ts`, and `tests/services/pointCloudService.surface.test.ts`: `7` files, `36` tests passed.
- Final controller-side strict E2E verification in `viewer/` passed for `tests/e2e/webgpu-on-demand-prototype.spec.ts` with `REQUIRE_WEBGPU_ON_DEMAND=1`: `7` passed.
- Final controller-side vertical-slice E2E verification in `viewer/` passed for `tests/e2e/webgpu-f32-on-demand-vertical-slice.spec.ts` with `REQUIRE_WEBGPU_ON_DEMAND=1`: `6` passed.
- Controller-side verification required redirecting `TEMP` and `TMP` to `D:\\codex-temp\\fast-utci` because `C:` temp space was near full and the initial run hit `ENOSPC`.
- Known verification noise remained non-fatal: npm warned about `--project`, and the dev server logged `ENOENT` for `data\\3d_models\\Ness-Tziona\\original_with_layers.glb`, but both E2E suites still passed.
- Keep the current `.bin`-backed main/debug views and the existing production `runAll()` path available. This prototype note does not imply a production source-path switch.

## Vertical-Slice Runtime Capture

Fresh browser capture in this session used local Chromium via Playwright with `headless: false` against `http://localhost:4174`.

### Strict Debug Route Runtime

Capture route:
`/debug-webgpu-utci?onDemandPrototype=1&strictExposureOnly=1&compareHours=12,23,16,17&baseline=separateRunAll`

Captured status text:
`On-demand prototype: ready utciRender auto -> gpuNative (webgpu)`

Captured `window.__onDemandPrototypeDiagnostics__` summary:

| Field | Value |
| --- | --- |
| `navigatorGpu` | `true` |
| `rendererBackend` | `webgpu` |
| `path` | `exposure-only-f32` |
| `timeIndices` | `[12]` |
| `usedRunAllForSelectedHour` | `false` |
| `usedExposureOnlyPrecompute` | `true` |
| `allHoursUtciBytesAllocated` | `0` |
| `allHoursMrtBytesAllocated` | `0` |
| `oneHourOutputBytes` | `417780` |
| `selectedHourTransferCount` | `0` |
| `renderTransport` | `none` |
| `debugReadbackCount` | `0` |
| `dataTextureBuildCount` | `0` |
| `timings.exposurePrecomputeMs` | `706.1000000238419` |
| `timings.oneHourDispatchMs` | `3.199999988079071` |
| `timings.renderUpdateMs` | `not exposed by runtime` |
| `timings.debugReadbackMs` | `not exposed by runtime` |
| `utciRenderRequested` | `auto` |
| `utciRenderResolved` | `gpuNative` |
| `bridgeAttached` | `false` |
| `visibleColorVariance` | `0` |
| `modelId` | `data/3d_models/original_with_layers.glb` |
| `scenarioId` | `not exposed by runtime` |
| `gridResolution` | `not exposed by runtime` |
| `pointCount` | `104445` |
| `adapterInfo` | `not exposed by runtime` |
| `maxStorageBufferBindingSize` | `not exposed by runtime` |
| `maxBufferSize` | `not exposed by runtime` |
| `maxStorageBuffersPerShaderStage` | `not exposed by runtime` |
| `liveAnalysisConstructedForSelectedHour` | `false` |

Actual captured object:

```json
{
  "navigatorGpu": true,
  "rendererBackend": "webgpu",
  "path": "exposure-only-f32",
  "timeIndices": [12],
  "usedRunAllForSelectedHour": false,
  "usedExposureOnlyPrecompute": true,
  "allHoursUtciBytesAllocated": 0,
  "allHoursMrtBytesAllocated": 0,
  "oneHourOutputBytes": 417780,
  "selectedHourTransferCount": 0,
  "renderTransport": "none",
  "debugReadbackCount": 0,
  "dataTextureBuildCount": 0,
  "timings": {
    "exposurePrecomputeMs": 706.1000000238419,
    "oneHourDispatchMs": 3.199999988079071
  },
  "utciRenderRequested": "auto",
  "utciRenderResolved": "gpuNative",
  "bridgeAttached": false,
  "visibleColorVariance": 0,
  "modelId": "data/3d_models/original_with_layers.glb",
  "pointCount": 104445,
  "liveAnalysisConstructedForSelectedHour": false
}
```

### Multi-Hour Baseline Comparison

Captured `window.__onDemandMultiHourComparison__` summary:

| Field | Value |
| --- | --- |
| `baselineSource` | `separateRunAll` |
| `baselineMonthContext.monthIndex` | `0` |
| `baselineMonthContext.sliceKind` | `representative-day-full-year` |
| `baselineMonthContext.note` | `compareHours uses the separate runAll baseline monthIndex 0 representative-day slice.` |
| `strictPath.path` | `exposure-only-f32` |
| `strictPath.usedRunAllForSelectedHour` | `false` |
| `strictPath.usedExposureOnlyPrecompute` | `true` |
| `strictPath.allHoursUtciBytesAllocated` | `0` |
| `strictPath.allHoursMrtBytesAllocated` | `0` |
| `strictPath.oneHourOutputBytes` | `417780` |
| `strictPath.renderTransport` | `none` |
| `strictPath.debugReadbackCount` | `0` |
| `strictPath.dataTextureBuildCount` | `0` |
| `strictPath.navigatorGpu` | `false` |
| `strictPath.rendererBackend` | `unknown` |
| `hours` | `[12, 23, 16, 17]` |

Hour-by-hour results:

| Hour | `numCompared` | `maxAbsDiff` | `rmse` | `onDemandAt31079` | `baselineAt31079` | `diffAt31079` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `12` | `104445` | `0` | `0` | `19.470516204833984` | `19.470516204833984` | `0` |
| `23` | `104445` | `0` | `0` | `2.221592664718628` | `2.221592664718628` | `0` |
| `16` | `104445` | `0` | `0` | `8.452316284179688` | `8.452316284179688` | `0` |
| `17` | `104445` | `0` | `0` | `7.999854564666748` | `7.999854564666748` | `0` |

Concrete point `31079` evidence present in the runtime object:

| Point | Hour | On-demand UTCI | Baseline UTCI | Diff |
| --- | ---: | ---: | ---: | ---: |
| `31079` | `16` | `8.452316284179688` | `8.452316284179688` | `0` |
| `31079` | `17` | `7.999854564666748` | `7.999854564666748` | `0` |

Actual captured object:

```json
{
  "baselineSource": "separateRunAll",
  "baselineMonthContext": {
    "monthIndex": 0,
    "sliceKind": "representative-day-full-year",
    "note": "compareHours uses the separate runAll baseline monthIndex 0 representative-day slice."
  },
  "strictPath": {
    "navigatorGpu": false,
    "rendererBackend": "unknown",
    "path": "exposure-only-f32",
    "timeIndices": [12],
    "usedRunAllForSelectedHour": false,
    "usedExposureOnlyPrecompute": true,
    "allHoursUtciBytesAllocated": 0,
    "allHoursMrtBytesAllocated": 0,
    "oneHourOutputBytes": 417780,
    "selectedHourTransferCount": 0,
    "renderTransport": "none",
    "debugReadbackCount": 0,
    "dataTextureBuildCount": 0,
    "timings": {
      "exposurePrecomputeMs": 706.1000000238419,
      "oneHourDispatchMs": 3.199999988079071
    }
  },
  "hours": [12, 23, 16, 17],
  "hourResults": [
    {
      "hour": 12,
      "numCompared": 104445,
      "maxAbsDiff": 0,
      "rmse": 0,
      "onDemandAt31079": 19.470516204833984,
      "baselineAt31079": 19.470516204833984,
      "diffAt31079": 0
    },
    {
      "hour": 23,
      "numCompared": 104445,
      "maxAbsDiff": 0,
      "rmse": 0,
      "onDemandAt31079": 2.221592664718628,
      "baselineAt31079": 2.221592664718628,
      "diffAt31079": 0
    },
    {
      "hour": 16,
      "numCompared": 104445,
      "maxAbsDiff": 0,
      "rmse": 0,
      "onDemandAt31079": 8.452316284179688,
      "baselineAt31079": 8.452316284179688,
      "diffAt31079": 0
    },
    {
      "hour": 17,
      "numCompared": 104445,
      "maxAbsDiff": 0,
      "rmse": 0,
      "onDemandAt31079": 7.999854564666748,
      "baselineAt31079": 7.999854564666748,
      "diffAt31079": 0
    }
  ],
  "knownPoint31079": {
    "pointIndex": 31079,
    "hours": [
      {
        "hour": 16,
        "onDemand": 8.452316284179688,
        "baseline": 8.452316284179688,
        "diff": 0
      },
      {
        "hour": 17,
        "onDemand": 7.999854564666748,
        "baseline": 7.999854564666748,
        "diff": 0
      }
    ]
  }
}
```

### Main Route Render-Path Diagnostics

Capture routes:

- GPU-native selection: `/?utciRenderDiagnostics=1&utciOnDemand=f32&utciRender=gpu`
- Fallback data path: `/?utciRenderDiagnostics=1&utciOnDemand=f32&utciRender=data`

Captured main-route GPU object:

```json
{
  "utciOnDemand": "f32",
  "utciRenderRequested": "gpu",
  "utciRenderResolved": "gpuNative",
  "rendererBackend": "webgpu",
  "utciSurfaceSource": "cpu-uploaded-selected-hour",
  "selectedHourTransferCount": 1,
  "dataTextureBuildCount": 0
}
```

Main-route GPU summary:

| Field | Value |
| --- | --- |
| `utciOnDemand` | `f32` |
| `utciRenderRequested` | `gpu` |
| `utciRenderResolved` | `gpuNative` |
| `rendererBackend` | `webgpu` |
| `utciSurfaceSource` | `cpu-uploaded-selected-hour` |
| `selectedHourTransferCount` | `1` |
| `dataTextureBuildCount` | `0` |

Captured main-route fallback object:

```json
{
  "utciOnDemand": "f32",
  "utciRenderRequested": "data",
  "utciRenderResolved": "dataTexture",
  "rendererBackend": "webgpu",
  "dataTextureBuildCount": 1
}
```

Main-route fallback summary:

| Field | Value |
| --- | --- |
| `utciOnDemand` | `f32` |
| `utciRenderRequested` | `data` |
| `utciRenderResolved` | `dataTexture` |
| `rendererBackend` | `webgpu` |
| `utciSurfaceSource` | `not exposed by runtime` |
| `selectedHourTransferCount` | `not exposed by runtime` |
| `dataTextureBuildCount` | `1` |

Interpretation kept intentionally narrow:

- The main route proves the current selected-hour render transport is `cpu-uploaded-selected-hour`, not zero-copy/direct interop.
- The fallback `dataTexture` path remains available and was freshly observed in the same session.

## 2026-05-08 Debug Integration Follow-up

Fresh browser capture from a local Playwright run at `2026-05-08T15:33:43.717Z`.

Capture routes:

- On-demand scrub capture: `/debug-webgpu-utci?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&timeIndex=12`, then scrubbed to final hour `23`.
- Multi-hour baseline capture: `/debug-webgpu-utci?onDemandPrototype=1&strictExposureOnly=1&compareHours=12,23,16,17&baseline=separateRunAll`.
- Main-route GPU render capture: `/?utciRenderDiagnostics=1&utciOnDemand=f32&utciRender=gpu`.
- Main-route fallback capture: `/?utciRenderDiagnostics=1&utciOnDemand=f32&utciRender=data`.
- Debug fallback capture: `/debug-webgpu-utci?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=data&timeIndex=12`.

Required gate summary from the fresh runtime objects:

| Required gate | Fresh runtime evidence |
| --- | --- |
| Debug WebGPU on-demand path | Debug route published `path='exposure-only-f32'`, `utciRenderRequested='gpu'`, `utciRenderResolved='gpuNative'`, `selectedMonthIndex=7`, `selectedTimeIndex=23`, `completedMonthIndex=7`, `completedTimeIndex=23`, and `appVisibleSelectedHour=true`. |
| Python `.bin` comparison preserved | Debug route published `debugComparisonReference='python-bin'`, `pythonBinComparisonActive=true`, `pythonComparisonHourIndex=23`, `webgpuComparisonHourIndex=23`, `selectedHourReadbackCount=1`, and sample comparison `numCompared=3`, `maxAbsDiff=0.04802513122558594`. Separate `compareHours` capture kept `baselineSource='separateRunAll'` and produced `maxAbsDiff=0`, `rmse=0` for hours `12`, `23`, `16`, and `17`, including point `31079` at hours `16` and `17`. |
| Repeated scrub final selection | After the forced-overlap scrub run, diagnostics ended at `selectedTimeIndex=23`, `completedTimeIndex=23`, `completedRequestId=5`, `staleResultDiscardCount=3`, `scrubSampleCount=5`, `inFlightCount=0`, `selectedHourTransferCount=1`, and `timings.renderUpdateMs=101.29999995231628`. |
| No all-hours hot-path allocation | Debug route published `allHoursUtciBytesAllocated=0`, `allHoursMrtBytesAllocated=0`, and `trackedGpuAllocationBytes.allHoursOutputBytes=0` during the selected-hour scrub path. |
| Tracked VRAM allocation shape | Tracked UTCI-owned WebGPU allocation bytes ended at `trackingScope='utci-owned-webgpu-buffers'`, `persistentExposureBytes=731116`, `selectedHourOutputBytes=417780`, and `selectedHourOutputBytesHighWatermark=417780`. |
| No hot-path `DataTexture` rebuild | On the GPU scrub path, `dataTextureBuildCount=0`. Main-route GPU render diagnostics also reported `dataTextureBuildCount=0`. |
| Render transport honesty | Debug route published `renderTransport='cpu-uploaded-selected-hour'`. Main-route GPU render diagnostics published `utciSurfaceSource='cpu-uploaded-selected-hour'`. This capture does not support zero-copy wording. |
| Fallback preserved | Main-route fallback published `utciRenderResolved='dataTexture'` with `dataTextureBuildCount=1`. Debug fallback kept `path='exposure-only-f32'`, `usedExposureOnlyPrecompute=true`, `usedRunAllForSelectedHour=false`, `pythonBinComparisonActive=true`, `utciRenderResolved='dataTexture'`, and `dataTextureBuildCount=1`. |

Fresh gate-critical object excerpts:

```json
{
  "onDemand": {
    "path": "exposure-only-f32",
    "selectedMonthIndex": 7,
    "selectedTimeIndex": 23,
    "completedMonthIndex": 7,
    "completedTimeIndex": 23,
    "completedRequestId": 5,
    "staleResultDiscardCount": 3,
    "scrubSampleCount": 5,
    "usedRunAllForSelectedHour": false,
    "usedExposureOnlyPrecompute": true,
    "allHoursUtciBytesAllocated": 0,
    "allHoursMrtBytesAllocated": 0,
    "oneHourOutputBytes": 417780,
    "selectedHourTransferCount": 1,
    "trackedGpuAllocationBytes": {
      "persistentExposureBytes": 731116,
      "allHoursOutputBytes": 0,
      "selectedHourOutputBytes": 417780,
      "selectedHourOutputBytesHighWatermark": 417780,
      "trackingScope": "utci-owned-webgpu-buffers"
    },
    "renderTransport": "cpu-uploaded-selected-hour",
    "dataTextureBuildCount": 0,
    "utciRenderRequested": "gpu",
    "utciRenderResolved": "gpuNative",
    "utciSurfaceSource": "cpu-uploaded-selected-hour",
    "debugComparisonReference": "python-bin",
    "pythonBinComparisonActive": true,
    "pythonComparisonHourIndex": 23,
    "webgpuComparisonHourIndex": 23,
    "appVisibleSelectedHour": true,
    "selectedHourReadbackCount": 1,
    "timings": {
      "renderUpdateMs": 101.29999995231628
    }
  },
  "multiHour": {
    "baselineSource": "separateRunAll",
    "hours": [12, 23, 16, 17],
    "hourResults": [
      { "hour": 12, "maxAbsDiff": 0, "rmse": 0 },
      { "hour": 23, "maxAbsDiff": 0, "rmse": 0 },
      { "hour": 16, "maxAbsDiff": 0, "rmse": 0, "diffAt31079": 0 },
      { "hour": 17, "maxAbsDiff": 0, "rmse": 0, "diffAt31079": 0 }
    ]
  },
  "renderGpu": {
    "utciOnDemand": "f32",
    "utciRenderRequested": "gpu",
    "utciRenderResolved": "gpuNative",
    "rendererBackend": "webgpu",
    "utciSurfaceSource": "cpu-uploaded-selected-hour",
    "selectedHourTransferCount": 1,
    "dataTextureBuildCount": 0
  },
  "debugFallback": {
    "path": "exposure-only-f32",
    "usedExposureOnlyPrecompute": true,
    "usedRunAllForSelectedHour": false,
    "utciRenderRequested": "data",
    "utciRenderResolved": "dataTexture",
    "pythonBinComparisonActive": true,
    "dataTextureBuildCount": 1
  }
}
```

Interpretation kept intentionally narrow:

- This fresh capture supports the debug on-demand integration gates above, but it still shows the selected-hour render transport as `cpu-uploaded-selected-hour`, not direct compute-buffer rendering.
- The same debug capture also reported `liveAnalysisConstructedForSelectedHour=true`, so this update only claims selected-hour on-demand compute, preserved Python `.bin` comparison, tracked UTCI-owned allocation shape, and preserved fallback behavior.
