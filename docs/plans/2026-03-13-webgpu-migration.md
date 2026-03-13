# WebGPU Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the entire MRT/UTCI computation pipeline from Python/pyembree to Three.js WebGPU compute shaders, enabling real-time thermal comfort analysis in the browser.

**Architecture:** Three separate WebGPU compute dispatches (solar exposure, sky exposure, MRT+UTCI) batched in a single command buffer. All 12 representative days (15th of each month × 24 hours = 288 analysis points) pre-computed on model load (~100-300ms). Grid points, exposure, and UTCI results persist in GPU storage buffers. Hour and month scrubbing are instant (0ms) — just buffer index changes. BVH rebuilt in a Web Worker on geometry change.

**Tech Stack:** SvelteKit, Threlte (`@threlte/core` v8.1.8), Three.js v0.174.0 (upgrade to r171+ with `three/webgpu`), `three-mesh-bvh` v0.9.3+, vitest, TypeScript, WGSL compute shaders.

**Analysis Document:** [webgpu-migration-analysis.md](../webgpu-migration-analysis.md) — 26 decisions, all open questions resolved.

**Test Runner:** `cd viewer && npx vitest run` (or `npm test` for watch mode)

---

## Phase 1: Core Migration — WebGPU Parity with Python Pipeline

### Task 1: Switch Viewer to WebGPURenderer

**Goal:** Replace `WebGLRenderer` with `WebGPURenderer` using `three/webgpu` import. No custom GLSL shaders exist (audited), so standard materials work unchanged.

**Files:**
- Modify: `viewer/package.json` — upgrade `three` to `^0.175.0` (r175, latest with stable WebGPU)
- Modify: `viewer/src/lib/components/scene/Scene.svelte` (or wherever `<Canvas>` is used) — add `createRenderer` prop
- Create: `viewer/tests/compute/webgpu-renderer.test.ts`

**Context:**
- Threlte v8.1.8 supports WebGPU via `createRenderer` prop on `<Canvas>` — see [threlte.xyz docs](https://threlte.xyz)
- `WebGPURenderer.init()` is async; Threlte handles this in `createRenderer` callback
- Analysis doc §15.1: "Upgrade Three.js to r171+, use `import * from 'three/webgpu'`, async renderer init in Threlte"
- Analysis doc decision #15, #21, #22

**Step 1: Write integration test that WebGPURenderer initializes**

```typescript
// viewer/tests/compute/webgpu-renderer.test.ts
import { describe, it, expect } from 'vitest';

describe('WebGPU Renderer', () => {
  it('should import WebGPURenderer from three/webgpu', async () => {
    // Verify the import path works after Three.js upgrade
    const THREE = await import('three/webgpu');
    expect(THREE.WebGPURenderer).toBeDefined();
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run tests/compute/webgpu-renderer.test.ts`
Expected: FAIL (three/webgpu import doesn't exist at current version)

**Step 3: Upgrade Three.js and add three-mesh-bvh**

Update `viewer/package.json`:
```json
{
  "dependencies": {
    "three": "^0.175.0",
    "three-mesh-bvh": "^0.9.3"
  }
}
```
Run: `cd viewer && npm install`

**Step 4: Switch Canvas to WebGPURenderer**

In the component that mounts `<Canvas>`:
```svelte
<script>
  import { WebGPURenderer } from 'three/webgpu';
</script>

<Canvas
  createRenderer={(canvas) => {
    const renderer = new WebGPURenderer({ canvas });
    return renderer;
  }}
>
```

**Step 5: Run test to verify it passes**

Run: `cd viewer && npx vitest run tests/compute/webgpu-renderer.test.ts`
Expected: PASS

**Step 6: Verify the viewer loads visually (manual)**

Run: `cd viewer && npm run dev`
Expected: Viewer loads with WebGPURenderer, 3D model renders correctly, layers/colors work, comparison mode works.

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: switch viewer to WebGPURenderer (three/webgpu)"
```

---

### Task 2: EPW Weather File Parser

**Goal:** Parse EPW weather files in the browser. Extract: dry bulb temp, humidity, direct/diffuse radiation, wind speed, horizontal IR.

**Files:**
- Create: `viewer/src/lib/compute/epw-parser.ts`
- Create: `viewer/tests/compute/epw-parser.test.ts`
- Reference: [EnergyPlus Weather File Format](https://designbuilder.co.uk/)

**Context:**
- EPW is CSV-like: 8 header lines + 8760 hourly data lines
- We need columns: 6 (dry bulb °C), 8 (rel humidity %), 12 (horiz IR W/m²), 14 (direct normal rad W/m²), 15 (diffuse horiz rad W/m²), 21 (wind speed m/s)
- EPW files bundled per model as static assets (decision #16)
- Analysis doc §4.7, §15.2

**Step 1: Write failing tests with known EPW data**

```typescript
// viewer/tests/compute/epw-parser.test.ts
import { describe, it, expect } from 'vitest';
import { parseEPW, type EPWData } from '$lib/compute/epw-parser';

// Minimal EPW content for testing (8 header lines + a few data lines)
const MINIMAL_EPW = `LOCATION,Beer Sheva,ISR,...
DESIGN CONDITIONS,...
TYPICAL/EXTREME,...
GROUND TEMPERATURES,...
HOLIDAYS/DAYLIGHT,...
COMMENTS 1,...
COMMENTS 2,...
DATA PERIODS,...
2020,1,1,1,0,...,25.0,...,50,...,...,...,...,800,...,200,...,150,...,...,...,...,...,...,3.5,...
2020,1,1,2,0,...,24.5,...,48,...,...,...,...,810,...,250,...,180,...,...,...,...,...,...,4.0,...`;

describe('EPW Parser', () => {
  it('should parse header and extract location', () => {
    const data = parseEPW(MINIMAL_EPW);
    expect(data.location.city).toBe('Beer Sheva');
  });

  it('should extract hourly dry bulb temperature', () => {
    const data = parseEPW(MINIMAL_EPW);
    expect(data.dryBulbTemp[0]).toBeCloseTo(25.0, 1);
  });

  it('should extract hourly relative humidity', () => {
    const data = parseEPW(MINIMAL_EPW);
    expect(data.relativeHumidity[0]).toBeCloseTo(50, 1);
  });

  it('should extract direct normal radiation', () => {
    const data = parseEPW(MINIMAL_EPW);
    expect(data.directNormalRad[0]).toBeCloseTo(200, 1);
  });

  it('should extract diffuse horizontal radiation', () => {
    const data = parseEPW(MINIMAL_EPW);
    expect(data.diffuseHorizRad[0]).toBeCloseTo(150, 1);
  });

  it('should extract wind speed', () => {
    const data = parseEPW(MINIMAL_EPW);
    expect(data.windSpeed[0]).toBeCloseTo(3.5, 1);
  });

  it('should extract horizontal IR', () => {
    const data = parseEPW(MINIMAL_EPW);
    expect(data.horizInfrared[0]).toBeCloseTo(800, 1);
  });

  it('should have 8760 entries for full year EPW', () => {
    // Test with actual EPW file loaded from fixtures
  });

  it('should extract data for specific month/day/hour', () => {
    const data = parseEPW(MINIMAL_EPW);
    const hourData = data.getHourData(1, 15, 12); // Jan 15, noon
    expect(hourData).toBeDefined();
  });
});
```

**Step 2: Run to verify failure**

Run: `cd viewer && npx vitest run tests/compute/epw-parser.test.ts`
Expected: FAIL — module not found

**Step 3: Implement EPW parser**

```typescript
// viewer/src/lib/compute/epw-parser.ts
// Pseudocode:
export interface EPWLocation { city: string; country: string; lat: number; lon: number; timezone: number; elevation: number; }
export interface HourData { dryBulb: number; relHumidity: number; directNormal: number; diffuseHoriz: number; windSpeed: number; horizIR: number; }
export interface EPWData {
  location: EPWLocation;
  dryBulbTemp: Float32Array;      // [8760]
  relativeHumidity: Float32Array;  // [8760]
  directNormalRad: Float32Array;   // [8760]
  diffuseHorizRad: Float32Array;   // [8760]
  windSpeed: Float32Array;         // [8760]
  horizInfrared: Float32Array;     // [8760]
  getHourData(month: number, day: number, hour: number): HourData;
}

export function parseEPW(content: string): EPWData {
  // 1. Split lines, skip first 8 header lines
  // 2. Parse line 1 for location: split by comma, fields[1]=city, [3]=country, [6]=lat, [7]=lon, [8]=tz, [9]=elev
  // 3. For each data line (8760): split by comma
  //    - col 6 (0-indexed): dry bulb temp
  //    - col 8: relative humidity
  //    - col 12: horizontal infrared
  //    - col 14: direct normal radiation
  //    - col 15: diffuse horizontal radiation
  //    - col 21: wind speed
  // 4. Store in Float32Arrays
  // 5. getHourData: convert (month, day, hour) → hour-of-year index
}
```

**Step 4: Run tests, iterate until all pass**

Run: `cd viewer && npx vitest run tests/compute/epw-parser.test.ts`
Expected: PASS

**Step 5: Validate against Python ladybug EPW parsing**

Create a validation test that loads a real EPW file and compares extracted values against known Python output.

**Step 6: Commit**

```bash
git add viewer/src/lib/compute/epw-parser.ts viewer/tests/compute/epw-parser.test.ts
git commit -m "feat: add EPW weather file parser (TypeScript port)"
```

---

### Task 3: Sunpath Calculation

**Goal:** Port ladybug's `Sunpath` (NOAA model) to TypeScript. Calculate sun altitude, azimuth, and direction vectors for any date/location.

**Files:**
- Create: `viewer/src/lib/compute/sunpath.ts`
- Create: `viewer/tests/compute/sunpath.test.ts`
- Reference: `src/fast_utci/mrt/solar.py` (158 lines, ladybug Sunpath wrapper)
- Reference: [ladybug Sunpath NOAA model](https://www.ladybug.tools/ladybug/docs/)

**Context:**
- Port ladybug's Sunpath (NOAA model) directly to TypeScript for parity
- Accuracy target: ±0.008° (same as ladybug)
- Output: sun altitude, azimuth, direction vector (x,y,z) for each hour
- Python `solar.py` uses `ladybug.sunpath.Sunpath.from_location()` and `sun.sun_vector`
- Analysis doc §4.1, decision #5

**Step 1: Write failing tests with known sun positions**

```typescript
// viewer/tests/compute/sunpath.test.ts
import { describe, it, expect } from 'vitest';
import { calculateSunPosition, getSunVectors, type SunPosition } from '$lib/compute/sunpath';

// Beer Sheva, Israel: lat=31.25, lon=34.79, timezone=2
const BEER_SHEVA = { lat: 31.25, lon: 34.79, timezone: 2 };

describe('Sunpath', () => {
  describe('calculateSunPosition', () => {
    it('should calculate correct altitude for Beer Sheva summer noon', () => {
      // June 15, noon — sun should be near zenith in Beer Sheva
      const pos = calculateSunPosition(BEER_SHEVA, 6, 15, 12);
      expect(pos.altitude).toBeGreaterThan(75); // ~80° in June
      expect(pos.altitude).toBeLessThan(85);
    });

    it('should calculate correct altitude for Beer Sheva winter noon', () => {
      const pos = calculateSunPosition(BEER_SHEVA, 1, 15, 12);
      expect(pos.altitude).toBeGreaterThan(30); // ~34° in January
      expect(pos.altitude).toBeLessThan(40);
    });

    it('should return isSunUp=false for nighttime', () => {
      const pos = calculateSunPosition(BEER_SHEVA, 6, 15, 2); // 2 AM
      expect(pos.isSunUp).toBe(false);
    });

    it('should return isSunUp=true for daytime', () => {
      const pos = calculateSunPosition(BEER_SHEVA, 6, 15, 12);
      expect(pos.isSunUp).toBe(true);
    });

    // VALIDATION: Compare against known ladybug output
    // These values should be generated by running the Python pipeline
    // and extracting reference sun positions
    it('should match ladybug output for Beer Sheva Aug 15 noon within ±0.01°', () => {
      const pos = calculateSunPosition(BEER_SHEVA, 8, 15, 12);
      // TODO: Fill in exact values from Python: `Sunpath.from_location(loc).calculate_sun(8, 15, 12)`
      // expect(pos.altitude).toBeCloseTo(PYTHON_REF_ALTITUDE, 1);
      // expect(pos.azimuth).toBeCloseTo(PYTHON_REF_AZIMUTH, 1);
    });
  });

  describe('getSunVectors', () => {
    it('should return 24 vectors for a single day', () => {
      const vectors = getSunVectors(BEER_SHEVA, 6, 15);
      expect(vectors.sunVectors.length).toBe(24);
      expect(vectors.isSunUp.length).toBe(24);
    });

    it('should return unit vectors', () => {
      const vectors = getSunVectors(BEER_SHEVA, 6, 15);
      for (let i = 0; i < 24; i++) {
        if (vectors.isSunUp[i]) {
          const v = vectors.sunVectors[i];
          const mag = Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2);
          expect(mag).toBeCloseTo(1.0, 4);
        }
      }
    });

    it('should return vectors for all 12 representative days', () => {
      const allVectors = [1,2,3,4,5,6,7,8,9,10,11,12].map(m =>
        getSunVectors(BEER_SHEVA, m, 15)
      );
      expect(allVectors).toHaveLength(12);
    });
  });
});
```

**Step 2: Run test → FAIL**

**Step 3: Implement sunpath**

```typescript
// viewer/src/lib/compute/sunpath.ts
// Port NOAA solar position algorithm from ladybug
// Key formulas:
//   - Julian day calculation
//   - Solar declination (obliquity of ecliptic + equation of center)
//   - Equation of time (orbital eccentricity correction)
//   - Hour angle → altitude & azimuth
//   - Sun vector: [-sin(azimuth)*cos(altitude), cos(azimuth)*cos(altitude), sin(altitude)]
// Reference: ladybug.sunpath.Sunpath (NOAA model)
// Expected: ~160 lines of arithmetic
```

**Step 4: Run tests → PASS**

**Step 5: Generate Python reference data for validation**

```bash
# Run from project root to generate reference sun positions
cd src && python -c "
from fast_utci.mrt.solar import *
from ladybug.location import Location
loc = Location(city='Beer Sheva', latitude=31.25, longitude=34.79, time_zone=2)
# Generate reference positions for validation tests
"
```

**Step 6: Commit**

```bash
git commit -m "feat: add sunpath calculation (NOAA model, ported from ladybug)"
```

---

### Task 4: Tregenza Sky Dome Vectors

**Goal:** Generate the 145 Tregenza dome direction vectors and patch weights for sky exposure calculation.

**Files:**
- Create: `viewer/src/lib/compute/tregenza.ts`
- Create: `viewer/tests/compute/tregenza.test.ts`
- Reference: `src/fast_utci/mrt/solar.py` → `get_tregenza_dome_vectors()`, `ladybug.viewsphere`
- Reference: Tregenza (1987) paper

**Context:**
- 145 patches subdividing the sky hemisphere
- Each patch has a direction vector (unit vec3) and a weight (solid angle fraction)
- Static constants — computed once, uploaded to GPU storage buffer
- Analysis doc §4.2

**Step 1: Write failing tests**

```typescript
// viewer/tests/compute/tregenza.test.ts
describe('Tregenza Dome', () => {
  it('should return exactly 145 direction vectors', () => {
    const dome = getTregenzaDome();
    expect(dome.vectors).toHaveLength(145);
  });

  it('should return exactly 145 weights', () => {
    const dome = getTregenzaDome();
    expect(dome.weights).toHaveLength(145);
  });

  it('should have unit-length direction vectors', () => {
    const dome = getTregenzaDome();
    dome.vectors.forEach(v => {
      const mag = Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2);
      expect(mag).toBeCloseTo(1.0, 4);
    });
  });

  it('should have weights that sum to approximately 1.0', () => {
    const dome = getTregenzaDome();
    const sum = dome.weights.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 2);
  });

  it('all vectors should point upward (z > 0)', () => {
    const dome = getTregenzaDome();
    dome.vectors.forEach(v => {
      expect(v[2]).toBeGreaterThan(0); // Sky hemisphere = z > 0
    });
  });

  // VALIDATION: match ladybug view_sphere values
  it('should match ladybug tregenza_dome_vectors', () => {
    // TODO: Generate reference from Python:
    // from ladybug.viewsphere import view_sphere
    // vectors = view_sphere.tregenza_dome_vectors
    // weights = view_sphere.dome_patch_weights(1)
  });
});
```

**Step 2-4: Standard TDD cycle**

**Step 5: Commit**

```bash
git commit -m "feat: add Tregenza 145-patch sky dome vectors and weights"
```

---

### Task 5: SolarCal MRT Computation

**Goal:** Port ladybug-comfort's `OutdoorSolarCal` ERF/MRT formulas to TypeScript. Pure arithmetic — no library dependencies.

**Files:**
- Create: `viewer/src/lib/compute/solarcal.ts`
- Create: `viewer/tests/compute/solarcal.test.ts`
- Reference: `src/fast_utci/mrt/solarcal.py` (~200 lines)
- Reference: ASHRAE-55 SolarCal model

**Context:**
- Core formula: `ERF_solar = (0.5 * f_eff * f_svv * (I_diff + I_TH * R_floor) + A_p * f_bes * I_dir / A_D) * (a_sw / a_lw)`
- `delta_MRT = ERF_solar / (f_eff * sigma * a_lw * (MRT_lw + 273.15)^3)`
- Parameters: f_eff=0.725, a_sw=0.7, a_lw=0.95, sigma=5.6697e-8
- Analysis doc §4.5, decision #5

**Step 1: Write failing tests with known inputs/outputs**

```typescript
// viewer/tests/compute/solarcal.test.ts
describe('SolarCal MRT', () => {
  it('should compute ERF=0 when fully shaded (no direct, no sky)', () => {
    const result = computeSolarCal({
      directNormalRad: 0,      // W/m²
      diffuseHorizRad: 0,
      solarAltitude: 45,
      solarExposure: 0,        // fully shaded
      skyViewFactor: 0,
      groundReflectance: 0.2,
      airTemp: 30,
    });
    expect(result.erf).toBeCloseTo(0, 1);
  });

  it('should compute positive ERF in direct sun', () => {
    const result = computeSolarCal({
      directNormalRad: 800,
      diffuseHorizRad: 200,
      solarAltitude: 60,
      solarExposure: 1,
      skyViewFactor: 0.8,
      groundReflectance: 0.2,
      airTemp: 30,
    });
    expect(result.erf).toBeGreaterThan(0);
    expect(result.deltaMRT).toBeGreaterThan(0);
    expect(result.outdoorMRT).toBeGreaterThan(30); // MRT > air temp in sun
  });

  // VALIDATION: compare against Python ladybug-comfort OutdoorSolarCal
  it('should match ladybug-comfort output for reference case', () => {
    // TODO: Generate reference from Python solarcal.py
    // const expected = { erf: PYTHON_REF_ERF, deltaMRT: PYTHON_REF_DELTA_MRT };
  });
});
```

**Step 2-4: Standard TDD cycle**

```typescript
// viewer/src/lib/compute/solarcal.ts
// Pseudocode for core computation:
// 1. Compute projected area ratio (Ap/AD) from solar altitude
// 2. Compute shortwave ERF from direct + diffuse
// 3. Compute longwave MRT from air temp + horizontal IR
// 4. Compute delta MRT from ERF
// 5. Return outdoor MRT = longwave MRT + delta MRT
```

**Step 5: Commit**

```bash
git commit -m "feat: add SolarCal MRT computation (ASHRAE-55 port)"
```

---

### Task 6: UTCI Polynomial

**Goal:** Port the UTCI 6th-degree polynomial approximation to TypeScript. Validate f32 precision against pythermalcomfort.

**Files:**
- Create: `viewer/src/lib/compute/utci.ts`
- Create: `viewer/tests/compute/utci.test.ts`
- Reference: `src/fast_utci/utci/calculation.py` (332 lines)
- Reference: Bröde et al. (2012) polynomial coefficients

**Context:**
- 6th-degree polynomial with ~100 coefficient terms
- Inputs: air temp (°C), MRT (°C), wind speed (m/s), relative humidity (%)
- Output: UTCI (°C)
- f32 precision expected error: < ±0.3°C (analysis doc §7)
- Analysis doc §4.6, decisions #2, #6

**Step 1: Write failing tests**

```typescript
// viewer/tests/compute/utci.test.ts
describe('UTCI Polynomial', () => {
  it('should return comfortable UTCI for mild conditions', () => {
    const utci = calculateUTCI(22, 22, 1.0, 50); // 22°C, no rad, 1m/s, 50%RH
    expect(utci).toBeGreaterThan(18);
    expect(utci).toBeLessThan(26);
  });

  it('should return heat stress for extreme conditions', () => {
    const utci = calculateUTCI(40, 60, 0.5, 30);
    expect(utci).toBeGreaterThan(40);
  });

  it('should return cold stress for cold conditions', () => {
    const utci = calculateUTCI(-10, -10, 5.0, 80);
    expect(utci).toBeLessThan(-10);
  });

  it('should clamp wind speed to minimum 0.5 m/s', () => {
    const utci = calculateUTCI(22, 22, 0, 50);
    const utci05 = calculateUTCI(22, 22, 0.5, 50);
    expect(utci).toBeCloseTo(utci05, 2);
  });

  // VALIDATION: every test case below compared against pythermalcomfort output
  it('should match pythermalcomfort within ±0.5°C', () => {
    // Generate reference cases from Python:
    // from pythermalcomfort.models import utci_approx
    // Cases: [(ta, mrt, wind, rh, expected_utci), ...]
    const cases = [
      // { ta: 25, mrt: 25, v: 1.0, rh: 50, expected: XX.X },  // Fill from Python
    ];
    cases.forEach(c => {
      const result = calculateUTCI(c.ta, c.mrt, c.v, c.rh);
      expect(result).toBeCloseTo(c.expected, 0); // ±0.5°C
    });
  });
});
```

**Step 2-4: Standard TDD cycle**

The UTCI polynomial is ~200 lines of coefficient multiplications. Port directly from pythermalcomfort or the Bröde et al. (2012) Fortran reference.

**Step 5: Commit**

```bash
git commit -m "feat: add UTCI polynomial calculation (Bröde 2012)"
```

---

### Task 7: Grid Generator

**Goal:** Port `grid.py` rectangular grid generation to TypeScript. Auto-grid from model bounding box with configurable resolution.

**Files:**
- Create: `viewer/src/lib/compute/grid-generator.ts`
- Create: `viewer/tests/compute/grid-generator.test.ts`
- Reference: `src/fast_utci/mrt/grid.py` (235 lines, focus on `create_rectangular_grid`)

**Context:**
- Primary mode: rectangular grid from model bounding box at user-specified spacing
- Default offset: match Python `offset_distance` parameter for validation parity (decision #17)
- Grid points uploaded to GPU storage buffer
- Analysis doc §13.2, decision #9

**Step 1: Write failing tests**

```typescript
// viewer/tests/compute/grid-generator.test.ts
describe('Grid Generator', () => {
  it('should generate rectangular grid within bounds', () => {
    const grid = createRectangularGrid(
      { min: [0, 0], max: [10, 10] },
      5.0,  // grid size
      0.0   // z offset
    );
    expect(grid.points.length).toBe(9); // 3×3 grid
  });

  it('should apply z-offset to all points', () => {
    const grid = createRectangularGrid(
      { min: [0, 0], max: [10, 10] },
      5.0,
      1.1  // ASHRAE pedestrian height
    );
    grid.points.forEach(p => expect(p[2]).toBeCloseTo(1.1));
  });

  it('should generate normals pointing up', () => {
    const grid = createRectangularGrid({ min: [0, 0], max: [10, 10] }, 5.0, 0.0);
    grid.normals.forEach(n => {
      expect(n[0]).toBe(0);
      expect(n[1]).toBe(0);
      expect(n[2]).toBe(1);
    });
  });

  // Match Python output for validation
  it('should match Python grid.create_rectangular_grid output', () => {
    // TODO: generate reference from Python
  });
});
```

**Step 2-4: Standard TDD cycle**

**Step 5: Commit**

```bash
git commit -m "feat: add grid generator (rectangular mode, TS port)"
```

---

### Task 8: GPU Compute Pipeline — Solar Exposure

**Goal:** Create the WebGPU compute pipeline for solar exposure. For each grid point × each hour, cast a ray toward the sun and check BVH for occlusion.

**Files:**
- Create: `viewer/src/lib/compute/gpu-pipeline.ts` — orchestrator
- Create: `viewer/src/lib/compute/shaders/exposure_solar.wgsl`
- Create: `viewer/tests/compute/gpu-pipeline.test.ts`

**Context:**
- Use `three-mesh-bvh/webgpu` TSL functions for BVH raycasting in compute shader
- Workgroup size: 64 (analysis doc §13.1)
- 2D dispatch: `(points, hours)` dimensions
- Output: `f32[numPoints × numHours]` storage buffer (1 = exposed, 0 = shaded)
- Analysis doc §4.3, §13.1, decision #8

**Step 1: Write failing test for compute pipeline**

Note: GPU tests require a WebGPU device. In CI, these run headlessly or are skipped. In dev, they run in a browser context.

```typescript
// viewer/tests/compute/gpu-pipeline.test.ts
describe('GPU Compute Pipeline', () => {
  // These tests validate the pipeline orchestrator logic
  // GPU-specific tests may need a browser context

  it('should create pipeline with correct buffer layout', () => {
    const config = createPipelineConfig({
      numPoints: 100,
      numHours: 24,
      numMonths: 12,
    });
    expect(config.solarExposureBufferSize).toBe(100 * 24 * 12 * 4); // f32
    expect(config.skyExposureBufferSize).toBe(100 * 4);
    expect(config.utciResultBufferSize).toBe(100 * 24 * 12 * 4);
  });

  it('should generate correct dispatch dimensions', () => {
    const dispatch = calculateDispatch(10000, 24, 64);
    expect(dispatch.x).toBe(Math.ceil(10000 / 64)); // 157
    expect(dispatch.y).toBe(24);
  });
});
```

**Step 2: Implement pipeline orchestrator**

```typescript
// viewer/src/lib/compute/gpu-pipeline.ts
// Pseudocode:
export class UTCIComputePipeline {
  device: GPUDevice;
  solarPipeline: GPUComputePipeline;
  skyPipeline: GPUComputePipeline;
  utciPipeline: GPUComputePipeline;

  // Persistent GPU buffers
  gridPointsBuffer: GPUBuffer;      // vec3<f32>[numPoints]
  sunVectorsBuffer: GPUBuffer;      // vec3<f32>[numMonths * numHours]
  weatherBuffer: GPUBuffer;          // struct[numMonths * numHours]
  solarExposureBuffer: GPUBuffer;    // f32[numPoints * numMonths * numHours]
  skyExposureBuffer: GPUBuffer;      // f32[numPoints]
  utciResultBuffer: GPUBuffer;       // f32[numPoints * numMonths * numHours]
  bvhBuffer: GPUBuffer;             // From three-mesh-bvh serialize

  async init(device: GPUDevice): Promise<void> { /* create pipelines, buffers */ }

  computeAll(numPoints: number, numMonths: number, numHours: number): void {
    const encoder = this.device.createCommandEncoder();

    // Pass 1: Solar exposure
    const solarPass = encoder.beginComputePass();
    solarPass.setPipeline(this.solarPipeline);
    solarPass.setBindGroup(0, this.solarBindGroup);
    solarPass.dispatchWorkgroups(Math.ceil(numPoints / 64), numMonths * numHours);
    solarPass.end();

    // Pass 2: Sky exposure
    const skyPass = encoder.beginComputePass();
    // ...

    // Pass 3: MRT + UTCI
    const utciPass = encoder.beginComputePass();
    // ...

    this.device.queue.submit([encoder.finish()]);
  }
}
```

**Step 3: Write WGSL compute shader for solar exposure**

```wgsl
// viewer/src/lib/compute/shaders/exposure_solar.wgsl
// Pseudocode:
@group(0) @binding(0) var<storage, read> grid_points: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> sun_vectors: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> solar_exposure: array<f32>;
// BVH bindings from three-mesh-bvh/webgpu TSL

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let point_idx = id.x;
    let time_idx = id.y;  // month * 24 + hour
    if (point_idx >= arrayLength(&grid_points)) { return; }

    let origin = grid_points[point_idx];
    let sun_dir = sun_vectors[time_idx];

    // Use three-mesh-bvh TSL intersection function
    let hit = bvh_intersects_any(origin + sun_dir * 0.01, sun_dir);
    solar_exposure[point_idx * num_times + time_idx] = select(1.0, 0.0, hit);
}
```

**Step 4: Commit**

```bash
git commit -m "feat: add GPU compute pipeline for solar exposure"
```

---

### Task 9: GPU Compute Pipeline — Sky Exposure

**Goal:** 145 Tregenza dome rays per grid point, check BVH for sky view factor.

**Files:**
- Create: `viewer/src/lib/compute/shaders/exposure_sky.wgsl`
- Modify: `viewer/src/lib/compute/gpu-pipeline.ts` — add sky pass

**Context:**
- 1D dispatch: only `(points)` since dome directions loop inside shader
- Output: `f32[numPoints]` — weighted sky view factor per point
- Analysis doc §4.4, §13.1

**Step 1: Write WGSL sky exposure shader**

```wgsl
// Pseudocode:
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let point_idx = id.x;
    let origin = grid_points[point_idx];

    var sky_view = 0.0;
    for (var i = 0u; i < 145u; i++) {
        let dir = dome_vectors[i];
        let weight = dome_weights[i];
        let hit = bvh_intersects_any(origin + dir * 0.01, dir);
        sky_view += select(weight, 0.0, hit);
    }
    sky_exposure[point_idx] = sky_view;
}
```

**Step 2: Commit**

```bash
git commit -m "feat: add GPU compute sky exposure (145 Tregenza rays)"
```

---

### Task 10: GPU Compute Pipeline — MRT + UTCI

**Goal:** Combine exposure + weather data → MRT → UTCI in a single compute shader. Port SolarCal + UTCI polynomial to WGSL.

**Files:**
- Create: `viewer/src/lib/compute/shaders/mrt_utci.wgsl`
- Modify: `viewer/src/lib/compute/gpu-pipeline.ts` — add UTCI pass

**Context:**
- Reads: solar exposure buffer, sky exposure buffer, weather uniforms
- Writes: UTCI result buffer `f32[numPoints × numMonths × numHours]`
- Pure arithmetic — port SolarCal ERF + UTCI polynomial to WGSL
- f32 precision accepted (decision #2)
- Analysis doc §4.5, §4.6, §13.1

**Step 1: Write WGSL MRT+UTCI shader**

```wgsl
// Pseudocode:
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let point_idx = id.x;
    let time_idx = id.y;

    let solar_exp = solar_exposure[point_idx * num_times + time_idx];
    let sky_vf = sky_exposure[point_idx];

    // Read weather for this hour
    let weather = weather_data[time_idx];

    // SolarCal: compute ERF → delta_MRT → outdoor MRT
    let erf = compute_erf(solar_exp, sky_vf, weather);
    let mrt = compute_outdoor_mrt(erf, weather.air_temp);

    // UTCI polynomial
    let utci = compute_utci(weather.air_temp, mrt, weather.wind_speed, weather.rel_humidity);

    utci_results[point_idx * num_times + time_idx] = utci;
}
```

**Step 2: Commit**

```bash
git commit -m "feat: add GPU compute MRT + UTCI shader (SolarCal + polynomial)"
```

---

### Task 11: Integrate Compute Pipeline into Viewer

**Goal:** Wire the GPU compute pipeline into the SvelteKit viewer. On model load, run compute pipeline and feed results to the existing point cloud visualization.

**Files:**
- Create: `viewer/src/lib/compute/compute-manager.ts` — SvelteKit-facing manager
- Modify: `viewer/src/lib/services/dataLoader.ts` — add GPU compute path alongside .bin loading
- Modify: `viewer/src/lib/components/scene/UTCIPointCloud.svelte` — read from GPU buffer

**Context:**
- On model load: parse EPW → compute sunpath for 12 months → generate grid → run GPU pipeline → feed to visualization
- Hour slider: just changes buffer index (0ms)
- Month scrubber: just changes buffer index (0ms)
- Analysis doc §13.4, decision #11

**Step 1: Implement compute manager**

```typescript
// viewer/src/lib/compute/compute-manager.ts
// Pseudocode:
export class ComputeManager {
  pipeline: UTCIComputePipeline;

  async initFromModel(modelUrl: string, epwUrl: string, gridSize: number) {
    // 1. Fetch + parse EPW
    // 2. Compute sunpath for 12 representative days (15th per month)
    // 3. Generate grid from model bounding box at gridSize
    // 4. Build BVH from model geometry
    // 5. Upload all data to GPU
    // 6. Run full compute pipeline (3 dispatches, single cmd buffer)
    // 7. Results available in utciResultBuffer
  }

  getUTCISlice(month: number, hour: number): Float32Array {
    // Read the correct slice from the pre-computed UTCI buffer
    // This is instant — no GPU compute needed
  }
}
```

**Step 2: Commit**

```bash
git commit -m "feat: integrate GPU compute pipeline into viewer"
```

---

## Phase 2: Validation — Zero Regressions Against Python

### Task 12: Generate Python Reference Data

**Goal:** Run the Python pipeline on test models and export reference data for validation.

**Files:**
- Create: `viewer/tests/fixtures/reference/` — directory for reference .json files
- Create: `scripts/generate-reference-data.py`

**Context:**
- Run Python pipeline on Ben Gurion / Nes Tziona models
- Export: sun positions, solar exposure, sky exposure, MRT, UTCI values for known inputs
- This reference data feeds into all validation tests
- Analysis doc §8, decisions #6, #7

**Step 1: Create reference data generation script**

```python
# scripts/generate-reference-data.py
# For each reference scenario:
# 1. Load model + EPW
# 2. Calculate sunpath for Aug 15 (standard test day)
# 3. Compute solar exposure, sky exposure
# 4. Compute MRT, UTCI
# 5. Export to JSON:
#    { sunPositions: [{hour, altitude, azimuth, vector}],
#      solarExposure: [f32[numPoints × numHours]],
#      utci: [f32[numPoints × numHours]] }
```

**Step 2: Run script, save reference data**

```bash
cd src && python scripts/generate-reference-data.py
```

**Step 3: Commit**

```bash
git commit -m "test: generate Python reference data for validation"
```

---

### Task 13: Component-Level Validation Tests

**Goal:** Validate each ported TypeScript component against Python reference output.

**Files:**
- Modify: `viewer/tests/compute/sunpath.test.ts` — add reference comparisons
- Modify: `viewer/tests/compute/solarcal.test.ts` — add reference comparisons
- Modify: `viewer/tests/compute/utci.test.ts` — add reference comparisons
- Create: `viewer/tests/compute/validation.test.ts` — full pipeline comparison

**Context:**
- Acceptable thresholds:
  - Sunpath: ±0.01° altitude/azimuth
  - SolarCal MRT: ±0.5°C
  - UTCI: ±0.5°C (decision #2)
- Analysis doc §8, §7

**Step 1: Write validation test suite**

```typescript
// viewer/tests/compute/validation.test.ts
import referenceData from './fixtures/reference/beer-sheva-aug15.json';

describe('Full Pipeline Validation vs Python', () => {
  it('should match Python sunpath within ±0.01°', () => {
    referenceData.sunPositions.forEach(ref => {
      const pos = calculateSunPosition(BEER_SHEVA, 8, 15, ref.hour);
      expect(pos.altitude).toBeCloseTo(ref.altitude, 1);
      expect(pos.azimuth).toBeCloseTo(ref.azimuth, 1);
    });
  });

  it('should match Python UTCI within ±0.5°C for 95% of points', () => {
    // Compare GPU-computed UTCI vs Python reference
    // Allow ±0.5°C, with 95% of points passing
  });
});
```

**Step 2: Run validation, fix regressions**

Run: `cd viewer && npx vitest run tests/compute/validation.test.ts`

**Step 3: Commit**

```bash
git commit -m "test: validate all components against Python reference data"
```

---

### Task 14: Compare Against Existing .bin Files

**Goal:** Load existing analysis .bin files and compare against GPU-computed UTCI for the same model/weather.

**Files:**
- Create: `viewer/tests/compute/bin-comparison.test.ts`

**Context:**
- Existing viewer loads pre-computed `.bin` files from Python pipeline
- Load the same model + EPW in the GPU pipeline, compare results
- This is the ultimate "no regression" test
- Analysis doc §8, decision #7

**Step 1: Write comparison test**

```typescript
describe('GPU vs .bin file comparison', () => {
  it('should match existing .bin UTCI within ±0.5°C', () => {
    // 1. Load .bin reference data
    // 2. Run GPU compute pipeline with same model + weather
    // 3. Compare point-by-point
    // 4. Assert 95%+ within ±0.5°C
  });
});
```

**Step 2: Commit**

```bash
git commit -m "test: validate GPU results against existing .bin files"
```

---

## Phase 3: New Features — Post-Parity Only

> **IMPORTANT:** Do not start Phase 3 until Phase 2 validation passes 100%.

### Task 15: Month Scrubbing — Pre-compute All 12 Months

**Goal:** Pre-compute UTCI for all 12 representative days on model load. Both month and hour scrubbers become instant.

**Files:**
- Modify: `viewer/src/lib/compute/compute-manager.ts` — compute 12 months
- Modify: `viewer/src/lib/compute/gpu-pipeline.ts` — handle 12× sun vector sets

**Context:**
- 12 months × 24 hours = 288 analysis points per grid point
- Total GPU memory: ~23 MB (10K points) — trivial (analysis doc §14)
- Init time: ~100-300ms
- Both scrubbers: 0ms — just buffer index change
- Analysis doc §14, decision #14

**Step 1: Extend compute pipeline for multi-month**

```typescript
// In compute-manager.ts:
// Generate sunpath for all 12 months (15th of each)
// Upload 12 × 24 = 288 sun vectors
// Dispatch with numMonths = 12
// Result buffer: f32[numPoints × 12 × 24]
```

**Step 2: Commit**

```bash
git commit -m "feat: pre-compute all 12 months for instant month scrubbing"
```

---

### Task 16: Resolution Slider UI

**Goal:** Add a grid resolution control to the viewer UI. Presets: Draft (10m), Standard (5m), Fine (2m), Ultra (1m), Custom.

**Files:**
- Create: `viewer/src/lib/components/ui/ResolutionSlider.svelte`
- Modify: `viewer/src/lib/stores/viewerStore.ts` — add gridResolution state

**Context:**
- On resolution change: regenerate grid → re-upload to GPU → recompute all
- ~50-500ms total latency depending on resolution
- Analysis doc §13.2, decision #9

**Step 1: Create resolution slider component**

**Step 2: Wire to compute manager**

**Step 3: Commit**

```bash
git commit -m "feat: add grid resolution slider (10m/5m/2m/1m/custom)"
```

---

### Task 17: Month Picker UI

**Goal:** Add radial month picker with segmented day/month toggle.

**Files:**
- Create: `viewer/src/lib/components/ui/MonthPicker.svelte`
- Modify: existing time picker component — add segmented control

**Context:**
- Segmented control: "Day" / "Month" toggle
- Month mode: radial picker with month labels instead of hour labels
- Colors from Figma: https://flight-swoop-66337217.figma.site/
- Analysis doc §17.3, decision #23

**Step 1: Create month picker component**

**Step 2: Commit**

```bash
git commit -m "feat: add radial month picker with day/month toggle"
```

---

## Phase 4: Advanced Features — Post-Phase 3

> **IMPORTANT:** Do not start Phase 4 until fully satisfied with Phase 3.

### Task 18: Geometry Editing — Add/Move/Remove Objects

**Goal:** Allow users to add buildings/trees to the scene, move and remove them. BVH rebuilds in Web Worker on change.

**Files:**
- Create: `viewer/src/lib/components/ui/ObjectEditor.svelte`
- Create: `viewer/src/lib/services/sceneEditor.ts`
- Modify: `viewer/src/lib/compute/compute-manager.ts` — trigger recompute on geo change

**Context:**
- Don't modify original GLB model — add new objects on top
- During drag: visual-only updates at 60fps (matrix transform)
- On release: BVH rebuild in Web Worker (~10-50ms), recompute pipeline (~50-200ms)
- Analysis doc §13.3, §17.5, decision #10, #25

**Step 1: Implement scene editor service**

```typescript
// Pseudocode:
// 1. User clicks "Add Building" → place parametric box mesh
// 2. Drag to position → visual-only transform update
// 3. Release → merge with base geometry → Web Worker BVH rebuild → recompute
// 4. Results update automatically
```

**Step 2: Commit**

```bash
git commit -m "feat: add geometry editing (add/move/remove objects)"
```

---

## Appendix: Reference Materials

### Analysis Document
- [webgpu-migration-analysis.md](../webgpu-migration-analysis.md) — 26 decisions, resolved questions, expert panel findings

### Python Reference Files for Porting

| File | Lines | Port Target |
|------|-------|-------------|
| `src/fast_utci/mrt/solar.py` | 158 | Task 3: `sunpath.ts` |
| `src/fast_utci/mrt/solar.py` (Tregenza) | ~30 | Task 4: `tregenza.ts` |
| `src/fast_utci/mrt/solarcal.py` | ~200 | Task 5: `solarcal.ts` |
| `src/fast_utci/utci/calculation.py` | 332 | Task 6: `utci.ts` |
| `src/fast_utci/mrt/grid.py` | 235 | Task 7: `grid-generator.ts` |
| `src/fast_utci/mrt/exposure.py` | 537 | Tasks 8-9: WGSL shaders |
| `src/fast_utci/mrt/mrt_calculator.py` | 582 | Task 11: `compute-manager.ts` |

### Key Library Versions

| Library | Required Version | Purpose |
|---------|-----------------|---------|
| `three` | ^0.175.0 (r175+) | WebGPURenderer + TSL |
| `three-mesh-bvh` | ^0.9.3 | GPU BVH raycasting |
| `@threlte/core` | ^8.1.8 (already installed) | SvelteKit 3D framework |
| `vitest` | ^4.0.18 (already installed) | Test runner |

### Key Decisions Reference

| # | Decision | Source |
|---|----------|--------|
| 8 | Three compute dispatches, single cmd buffer | §13.1 |
| 9 | Auto-grid, resolution slider | §13.2 |
| 10 | BVH rebuild in Web Worker | §13.3 |
| 11 | Pre-compute all hours, instant scrubbing | §13.4 |
| 14 | Pre-compute all 12 months (~23MB) | §14 |
| 15 | WebGPURenderer via `three/webgpu` | §15.1 |
| 17 | Grid height: match Python offset_dist | §15.3 |
| 21 | Threlte v8.1.8 supports WebGPU | §17.1 |
| 22 | No custom GLSL to port | §17.2 |
| 26 | Phased implementation | §18 |
