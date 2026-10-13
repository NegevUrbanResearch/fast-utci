# UTCI Analysis Viewer

The viewer is a SvelteKit + Three.js application for interactive UTCI and Shading Availability Index (SAI) analysis. It loads project models and metadata from `../data/`, runs WebGPU compute in the browser, and renders the active metric in a 3D scene.

Project history: the baseline workflow was conventional Ladybug/Grasshopper, followed by this repo's Embree/parallel Python/Ladybug pathway for faster reproducible reference runs. The viewer is now the main path: WebGPU + Three.js computes and renders the interactive analysis live in the browser.

## Routes

- `/`: main application route.
- `/debug`: diagnostics, collectors, parity checks, and `.bin` comparison.

Use `/` for normal development and user-facing behavior. Use `/debug` only when working on validation or diagnostics.

## Development

Run commands from `viewer/`. Use Node.js 20.19.x, 22.12.x or later in the 22.x series, or 24.x and later. CI uses Node.js 22.

```powershell
npm ci
npm run dev
```

Vite prints the local URL, usually `http://localhost:5173`.

## Build and preview

```powershell
npm run build
npm run preview
```

## Checks

```powershell
npm run check
npm run test:unit
npm run test:coverage
```

`npm test` starts Vitest watch mode. Unit discovery includes `.test.ts` files in `tests/` and `src/`, including source guards under `tests/e2e/`; Playwright `.spec.ts` files use the separate browser runner.

Focused WebGPU route checks:

```powershell
npm run test:quality:selected-hour
npm run test:e2e:selected-hour
```

The Playwright selected-hour checks require a browser/GPU setup that can exercise WebGPU.

## Data inputs

The viewer reads from the repo-level `data/` directory:

- `data/3d_models/`: GLB models and georeference sidecars.
- `data/weather/`: EPW/weather inputs.
- `data/analyses/`: project metadata, manifests, live WebGPU metadata, and legacy/debug `.bin` reference files.
- `data/performance-results/`: collector output summarized by docs and scripts.
- `data/gis/`: GIS exports generated from verified viewer runs.

See [../docs/data-onboarding.md](../docs/data-onboarding.md) for the project/data loading workflow.

## Useful scripts

- `scripts/generate-live-webgpu-metadata.ts`: generate metadata for GLB-backed WebGPU projects.
- `scripts/compare-parity.ts`: compare WebGPU outputs with reference artifacts.
- `scripts/compare-shading-index-parity.ts`: debug-route parity for the internal `shading_index` output that represents SAI.
- `scripts/export-innovation-district-gis.ts`: collect verified live-route data for GIS handoff.
- `scripts/summarize-main-route-performance.ts`: summarize main-route performance artifacts.

## Runtime notes

- Three.js uses `WebGPURenderer`.
- UTCI is selected by month and hour.
- SAI is selected by month.
- The preferred visible UTCI render path is `compute-buffer-selected-hour`.
- Tooltips, diagnostics, exports, and compatibility paths may use bounded CPU readbacks; these support the WebGPU route and are not the recommended analysis workflow.

## Deployment

The viewer quality workflow runs `npm ci`, type checking, unit tests, and the build on pull requests and pushes. The Pages workflow runs the same gates before publishing from `main` or a manual dispatch.

The current configuration uses an empty base path for the custom domain utci.nurlab.org. CI copies the contents of `viewer/build/` directly to the deployment root, alongside `data/`. The main route is `/`, and diagnostics use `/debug`. This describes the configured deployment; it does not verify the site's current availability.

Hosted CI checks types, CPU unit behavior, and build output. It does not establish hardware WebGPU execution or parity. Run the separate browser collectors on a supported WebGPU browser and GPU to verify those contracts.

## Stack

- SvelteKit
- Svelte 5
- Three.js / Three WebGPU
- Threlte
- TypeScript
- Vitest
- Playwright
