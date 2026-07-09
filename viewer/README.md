# UTCI Analysis Viewer

The viewer is a SvelteKit + Three.js application for interactive UTCI and Shading Availability Index (SAI) analysis. It loads project models and metadata from `../data/`, runs WebGPU compute in the browser, and renders the active metric in a 3D scene.

Project history: the baseline workflow was conventional Ladybug/Grasshopper, followed by this repo's Embree/parallel Python/Ladybug pathway for faster reproducible reference runs. The viewer is now the main path: WebGPU + Three.js computes and renders the interactive analysis live in the browser.

## Routes

- `/`: main application route.
- `/debug`: diagnostics, collectors, parity checks, and `.bin` comparison.

Use `/` for normal development and user-facing behavior. Use `/debug` only when working on validation or diagnostics.

## Development

```powershell
npm install
npm run dev
```

Vite prints the local URL, usually `http://localhost:5173`.

## Build And Preview

```powershell
npm run build
npm run preview
```

## Checks

```powershell
npm run check
npm test
npm run test:coverage
```

Focused WebGPU route checks:

```powershell
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
```

The Playwright selected-hour checks require a browser/GPU setup that can exercise WebGPU.

## Data Inputs

The viewer reads from the repo-level `data/` directory:

- `data/3d_models/`: GLB models and georeference sidecars.
- `data/weather/`: EPW/weather inputs.
- `data/analyses/`: project metadata, manifests, live WebGPU metadata, and legacy/debug `.bin` reference files.
- `data/performance-results/`: collector output summarized by docs and scripts.
- `data/gis/`: GIS exports generated from verified viewer runs.

See [../docs/data-onboarding.md](../docs/data-onboarding.md) for the project/data loading workflow.

## Useful Scripts

- `scripts/generate-live-webgpu-metadata.ts`: generate metadata for GLB-backed WebGPU projects.
- `scripts/compare-parity.ts`: compare WebGPU outputs with reference artifacts.
- `scripts/compare-shading-index-parity.ts`: debug-route parity for the internal `shading_index` output that represents SAI.
- `scripts/export-innovation-district-gis.ts`: collect verified live-route data for GIS handoff.
- `scripts/summarize-main-route-performance.ts`: summarize main-route performance artifacts.

## Runtime Notes

- Three.js uses `WebGPURenderer`.
- UTCI is selected by month and hour.
- SAI is selected by month.
- The preferred visible UTCI render path is `compute-buffer-selected-hour`.
- Tooltips, diagnostics, exports, and compatibility paths may use bounded CPU readbacks; these support the WebGPU route and are not the recommended analysis workflow.

## Deployment

GitHub Pages deployment is handled by CI on `main`: install dependencies, run tests, build, and publish.

The public app is served at:

```text
https://[username].github.io/fast-utci/viewer/build/
```

The repo root URL redirects to `viewer/build/`.

## Stack

- SvelteKit
- Svelte 5
- Three.js / Three WebGPU
- Threlte
- TypeScript
- Vitest
- Playwright
