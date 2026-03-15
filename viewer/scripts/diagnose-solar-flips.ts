/**
 * Diagnose solar binary flips (ref vs webgpu) and correlate with shortwave MRT components.
 *
 * Usage:
 *   npx tsx scripts/diagnose-solar-flips.ts [--base-path <path>] [--top <n>] [--binary-threshold <v>] [--out <file>]
 */
import { dirname, resolve } from 'node:path';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadReferenceIntermediatesFromFs } from '../src/lib/parity/loadReferenceIntermediatesFromFs';
import { loadWebgpuCollectedFromFs } from '../src/lib/parity/loadWebgpuCollectedFromFs';
import { pointwiseIndexFromFlat } from '../src/lib/parity/pointwiseIndex';
import { loadReferenceFromFs } from '../src/lib/parity/loadReferenceFromFs';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, '..', '..');
const DEFAULT_BASE_PATH = 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday';
const DEFAULT_TOP = 25;
const DEFAULT_BINARY_THRESHOLD = 0.5;

interface Args {
	basePath: string;
	topN: number;
	binaryThreshold: number;
	outPath: string | null;
}

interface FlipCell {
	flatIndex: number;
	pointIndex: number;
	hourIndex: number;
	hourValue: number | null;
	pointX: number;
	pointY: number;
	pointZ: number;
	refSolar: number;
	webgpuSolar: number;
	refSolarBinary: 0 | 1;
	webgpuSolarBinary: 0 | 1;
	refMarginToThreshold: number;
	webgpuMarginToThreshold: number;
	refShortErf: number | null;
	webgpuShortErf: number | null;
	shortErfDiff: number | null;
	refShortDmrt: number | null;
	webgpuShortDmrt: number | null;
	shortDmrtDiff: number | null;
	refLongErf: number | null;
	webgpuLongErf: number | null;
	longErfDiff: number | null;
	refLongDmrt: number | null;
	webgpuLongDmrt: number | null;
	longDmrtDiff: number | null;
	refMrt: number;
	webgpuMrt: number;
	mrtDiff: number;
	refSkyAtPoint: number | null;
	webgpuSkyAtPoint: number | null;
	skyDiffAtPoint: number | null;
	score: number;
}

function parseArgs(): Args {
	let basePath = process.env.PARITY_BASE_PATH ?? DEFAULT_BASE_PATH;
	let topN = DEFAULT_TOP;
	let binaryThreshold = DEFAULT_BINARY_THRESHOLD;
	let outPath: string | null = null;
	const argv = process.argv.slice(2);
	for (let i = 0; i < argv.length; i++) {
		if (argv[i] === '--base-path' && argv[i + 1]) {
			basePath = argv[i + 1];
			i++;
		} else if (argv[i] === '--top' && argv[i + 1]) {
			topN = Number(argv[i + 1]);
			i++;
		} else if (argv[i] === '--binary-threshold' && argv[i + 1]) {
			binaryThreshold = Number(argv[i + 1]);
			i++;
		} else if (argv[i] === '--out' && argv[i + 1]) {
			outPath = resolve(process.cwd(), argv[i + 1]);
			i++;
		}
	}
	if (!Number.isFinite(topN) || topN <= 0) {
		throw new Error(`Invalid --top value: ${topN}`);
	}
	if (!Number.isFinite(binaryThreshold)) {
		throw new Error(`Invalid --binary-threshold value: ${binaryThreshold}`);
	}
	return {
		basePath: resolve(REPO_ROOT, basePath),
		topN: Math.floor(topN),
		binaryThreshold,
		outPath
	};
}

function loadHourValues(basePath: string, numHours: number): number[] {
	try {
		const raw = readFileSync(`${basePath}.json`, 'utf8');
		const parsed = JSON.parse(raw) as { hours?: unknown };
		if (Array.isArray(parsed.hours) && parsed.hours.every((v) => typeof v === 'number')) {
			const hours = parsed.hours as number[];
			if (hours.length === numHours) return hours;
		}
	} catch {
		// fall back to indices
	}
	return Array.from({ length: numHours }, (_, i) => i);
}

function toBinary(value: number, threshold: number): 0 | 1 {
	return value > threshold ? 1 : 0;
}

function safeDiff(a: number[] | Float32Array | undefined, b: number[] | Float32Array | undefined, index: number): number | null {
	if (!a || !b || index < 0 || index >= a.length || index >= b.length) {
		return null;
	}
	return b[index] - a[index];
}

function safeValue(a: number[] | Float32Array | undefined, index: number): number | null {
	if (!a || index < 0 || index >= a.length) {
		return null;
	}
	return a[index];
}

async function main(): Promise<void> {
	const { basePath, topN, binaryThreshold, outPath } = parseArgs();
	console.log('Diagnose solar flips, base path:', basePath);
	console.log('Top cells:', topN);
	console.log('Binary threshold:', binaryThreshold);

	const [ref, refSolar, refMrt, refSky, webgpu] = await Promise.all([
		loadReferenceFromFs(basePath),
		loadReferenceIntermediatesFromFs(basePath, 'solar'),
		loadReferenceIntermediatesFromFs(basePath, 'mrt'),
		loadReferenceIntermediatesFromFs(basePath, 'sky').catch(() => null),
		loadWebgpuCollectedFromFs(basePath)
	]);
	if (!webgpu.solar) throw new Error(`Missing WebGPU solar artifact: ${basePath}_webgpu_solar.json`);
	if (!webgpu.mrt) throw new Error(`Missing WebGPU MRT artifact: ${basePath}_webgpu_mrt.json`);

	if (refSolar.numPositions !== webgpu.solar.numPositions || refSolar.numHours !== webgpu.solar.numHours) {
		throw new Error(
			`Solar shape mismatch: ref=(${refSolar.numPositions},${refSolar.numHours}) webgpu=(${webgpu.solar.numPositions},${webgpu.solar.numHours})`
		);
	}
	if (refMrt.numPositions !== webgpu.mrt.numPositions || refMrt.numHours !== webgpu.mrt.numHours) {
		throw new Error(
			`MRT shape mismatch: ref=(${refMrt.numPositions},${refMrt.numHours}) webgpu=(${webgpu.mrt.numPositions},${webgpu.mrt.numHours})`
		);
	}
	if (refSolar.numPositions !== refMrt.numPositions || refSolar.numHours !== refMrt.numHours) {
		throw new Error(
			`Solar vs MRT shape mismatch: solar=(${refSolar.numPositions},${refSolar.numHours}) mrt=(${refMrt.numPositions},${refMrt.numHours})`
		);
	}

	const numPoints = refSolar.numPositions;
	const numHours = refSolar.numHours;
	if (webgpu.sky && webgpu.sky.numPositions !== numPoints) {
		throw new Error(`Sky shape mismatch: webgpu.sky.numPositions=${webgpu.sky.numPositions}, expected=${numPoints}`);
	}
	if (refSky && refSky.numPositions !== numPoints) {
		throw new Error(`Sky shape mismatch: refSky.numPositions=${refSky.numPositions}, expected=${numPoints}`);
	}
	if (ref.data.numPositions !== numPoints || ref.data.numHours !== numHours) {
		throw new Error(
			`Reference point/hour shape mismatch: main=(${ref.data.numPositions},${ref.data.numHours}) solar=(${numPoints},${numHours})`
		);
	}
	const expectedLength = numPoints * numHours;
	if (refSolar.solarExposure.length !== expectedLength || webgpu.solar.solarExposure.length !== expectedLength) {
		throw new Error(
			`Unexpected solar exposure lengths: ref=${refSolar.solarExposure.length}, webgpu=${webgpu.solar.solarExposure.length}, expected=${expectedLength}`
		);
	}

	const hourValues = loadHourValues(basePath, numHours);
	const flips: FlipCell[] = [];
	const pointAgg = new Map<number, { flipCount: number; maxAbsShortErfDiff: number; maxAbsShortDmrtDiff: number }>();
	for (let flatIndex = 0; flatIndex < expectedLength; flatIndex++) {
		const refValue = refSolar.solarExposure[flatIndex];
		const webgpuValue = webgpu.solar.solarExposure[flatIndex];
		const refBinary = toBinary(refValue, binaryThreshold);
		const webgpuBinary = toBinary(webgpuValue, binaryThreshold);
		if (refBinary === webgpuBinary) continue;
		const { pointIndex, hourIndex } = pointwiseIndexFromFlat(flatIndex, numHours);
		const pointOffset = pointIndex * 3;
		const shortErfDiff = safeDiff(refMrt.short_erf, webgpu.mrt.short_erf, flatIndex);
		const shortDmrtDiff = safeDiff(refMrt.short_dmrt, webgpu.mrt.short_dmrt, flatIndex);
		const longErfDiff = safeDiff(refMrt.long_erf, webgpu.mrt.long_erf, flatIndex);
		const longDmrtDiff = safeDiff(refMrt.long_dmrt, webgpu.mrt.long_dmrt, flatIndex);
		const refShortErf = safeValue(refMrt.short_erf, flatIndex);
		const webgpuShortErf = safeValue(webgpu.mrt.short_erf, flatIndex);
		const refShortDmrt = safeValue(refMrt.short_dmrt, flatIndex);
		const webgpuShortDmrt = safeValue(webgpu.mrt.short_dmrt, flatIndex);
		const refLongErf = safeValue(refMrt.long_erf, flatIndex);
		const webgpuLongErf = safeValue(webgpu.mrt.long_erf, flatIndex);
		const refLongDmrt = safeValue(refMrt.long_dmrt, flatIndex);
		const webgpuLongDmrt = safeValue(webgpu.mrt.long_dmrt, flatIndex);
		const refSkyAtPoint = refSky?.skyExposure[pointIndex] ?? null;
		const webgpuSkyAtPoint = webgpu.sky?.skyExposure?.[pointIndex] ?? null;
		const skyDiffAtPoint = refSkyAtPoint == null || webgpuSkyAtPoint == null ? null : webgpuSkyAtPoint - refSkyAtPoint;
		const mrtDiff = webgpu.mrt.mrt[flatIndex] - refMrt.mrt[flatIndex];
		const score =
			shortErfDiff == null && shortDmrtDiff == null ? Math.abs(mrtDiff) : Math.abs(shortErfDiff ?? 0) + Math.abs(shortDmrtDiff ?? 0);
		flips.push({
			flatIndex,
			pointIndex,
			hourIndex,
			hourValue: hourValues[hourIndex] ?? null,
			pointX: ref.data.positions[pointOffset] ?? Number.NaN,
			pointY: ref.data.positions[pointOffset + 1] ?? Number.NaN,
			pointZ: ref.data.positions[pointOffset + 2] ?? Number.NaN,
			refSolar: refValue,
			webgpuSolar: webgpuValue,
			refSolarBinary: refBinary,
			webgpuSolarBinary: webgpuBinary,
			refMarginToThreshold: refValue - binaryThreshold,
			webgpuMarginToThreshold: webgpuValue - binaryThreshold,
			refShortErf,
			webgpuShortErf,
			shortErfDiff,
			refShortDmrt,
			webgpuShortDmrt,
			shortDmrtDiff,
			refLongErf,
			webgpuLongErf,
			longErfDiff,
			refLongDmrt,
			webgpuLongDmrt,
			longDmrtDiff,
			refMrt: refMrt.mrt[flatIndex],
			webgpuMrt: webgpu.mrt.mrt[flatIndex],
			mrtDiff,
			refSkyAtPoint,
			webgpuSkyAtPoint,
			skyDiffAtPoint,
			score
		});

		const agg = pointAgg.get(pointIndex) ?? {
			flipCount: 0,
			maxAbsShortErfDiff: 0,
			maxAbsShortDmrtDiff: 0
		};
		agg.flipCount += 1;
		if (shortErfDiff != null) agg.maxAbsShortErfDiff = Math.max(agg.maxAbsShortErfDiff, Math.abs(shortErfDiff));
		if (shortDmrtDiff != null) agg.maxAbsShortDmrtDiff = Math.max(agg.maxAbsShortDmrtDiff, Math.abs(shortDmrtDiff));
		pointAgg.set(pointIndex, agg);
	}

	flips.sort((a, b) => b.score - a.score || a.flatIndex - b.flatIndex);
	const topCells = flips.slice(0, topN);
	const topPointIndices = [...pointAgg.entries()]
		.map(([pointIndex, agg]) => ({ pointIndex, ...agg }))
		.sort((a, b) => b.flipCount - a.flipCount || b.maxAbsShortErfDiff + b.maxAbsShortDmrtDiff - (a.maxAbsShortErfDiff + a.maxAbsShortDmrtDiff))
		.slice(0, topN);

	const report = {
		basePath,
		binaryThreshold,
		counts: {
			flipCells: flips.length,
			totalCells: expectedLength,
			flipRate: expectedLength > 0 ? flips.length / expectedLength : 0,
			numPoints,
			numHours
		},
		topAffectedIndices: topCells.map((c) => c.flatIndex),
		topAffectedCells: topCells,
		topPointIndices
	};

	const reportPath = outPath ?? `${basePath}_solar_flip_diagnostics.json`;
	writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf8');

	console.log(`flip cells: ${report.counts.flipCells}/${report.counts.totalCells} (${(report.counts.flipRate * 100).toFixed(3)}%)`);
	console.log(`top affected flat indices: ${report.topAffectedIndices.slice(0, Math.min(10, report.topAffectedIndices.length)).join(', ') || '(none)'}`);
	if (topCells.length > 0) {
		const sample = topCells[0];
		console.log(
			`worst cell: flat=${sample.flatIndex}, point=${sample.pointIndex}, hourIndex=${sample.hourIndex}, hour=${sample.hourValue ?? 'n/a'}, short_erf_diff=${
				sample.shortErfDiff == null ? 'n/a' : sample.shortErfDiff.toFixed(4)
			}, short_dmrt_diff=${sample.shortDmrtDiff == null ? 'n/a' : sample.shortDmrtDiff.toFixed(4)}`
		);
	}
	console.log('report written:', reportPath);
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});
