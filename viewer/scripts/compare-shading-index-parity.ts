/**
 * Offline Shading Index parity compare.
 *
 * Usage:
 *   npx tsx scripts/compare-shading-index-parity.ts --webgpu <values.json> [--base-path <path>] [--report <file>]
 *
 * The WebGPU values file may be a JSON array, or an object with `webgpuValues`
 * or `shadingIndex` as a number array.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { compareShadingIndex } from '../tests/parity/compareShadingIndex';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const DEFAULT_BASE_PATH = 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday';
const DEFAULT_REPORT_PATH =
	'data/batch-parity-results/Ben-Gurion_20250815_grid_2m_fullday_shading-index-parity.json';
const DEFAULT_TOLERANCE = 1e-6;

function parseArgs(): {
	basePath: string;
	webgpuPath: string;
	reportPath: string;
	tolerance: number;
} {
	let basePath = process.env.PARITY_BASE_PATH ?? DEFAULT_BASE_PATH;
	let webgpuPath = process.env.SHADING_INDEX_WEBGPU_PATH ?? '';
	let reportPath = process.env.SHADING_INDEX_PARITY_REPORT ?? DEFAULT_REPORT_PATH;
	let tolerance = Number(process.env.SHADING_INDEX_PARITY_TOLERANCE ?? DEFAULT_TOLERANCE);
	const argv = process.argv.slice(2);
	for (let i = 0; i < argv.length; i += 1) {
		if (argv[i] === '--base-path' && argv[i + 1]) {
			basePath = argv[++i];
		} else if (argv[i] === '--webgpu' && argv[i + 1]) {
			webgpuPath = argv[++i];
		} else if (argv[i] === '--report' && argv[i + 1]) {
			reportPath = argv[++i];
		} else if (argv[i] === '--tolerance' && argv[i + 1]) {
			tolerance = Number(argv[++i]);
		}
	}
	if (!webgpuPath) {
		throw new Error('Missing --webgpu <values.json>.');
	}
	if (!Number.isFinite(tolerance) || tolerance < 0) {
		throw new Error(`Invalid tolerance: ${tolerance}`);
	}
	return {
		basePath: resolve(REPO_ROOT, basePath),
		webgpuPath: resolve(process.cwd(), webgpuPath),
		reportPath: resolve(REPO_ROOT, reportPath),
		tolerance
	};
}

function loadReferenceShadingFromBin(basePath: string): {
	numPoints: number;
	numHours: number;
	positions: Float32Array;
	shadingIndex: Float32Array;
} {
	const metadata = JSON.parse(readFileSync(`${basePath}.json`, 'utf8')) as {
		has_shading_index?: boolean;
	};
	if (metadata.has_shading_index !== true) {
		throw new Error(`${basePath}.json does not advertise has_shading_index=true.`);
	}
	const buffer = readFileSync(`${basePath}.bin`);
	const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
	let offset = 0;
	const numPoints = view.getUint32(offset, true);
	offset += 4;
	const numHours = view.getUint32(offset, true);
	offset += 4;
	const positions = new Float32Array(numPoints * 3);
	for (let i = 0; i < positions.length; i += 1) {
		positions[i] = view.getFloat32(offset, true);
		offset += 4;
	}
	const oldFormatSize = 8 + numPoints * 12 + numPoints * 4 * numHours;
	if (buffer.byteLength <= oldFormatSize) {
		throw new Error(`${basePath}.bin is the old format and does not contain Shading Index.`);
	}
	const hasShadingIndex = view.getUint32(offset, true) === 1;
	offset += 4;
	if (!hasShadingIndex) {
		throw new Error(`${basePath}.bin has a Shading Index flag set to 0.`);
	}
	const shadingIndex = new Float32Array(numPoints);
	for (let i = 0; i < numPoints; i += 1) {
		shadingIndex[i] = view.getFloat32(offset, true);
		offset += 4;
	}
	return { numPoints, numHours, positions, shadingIndex };
}

function loadWebgpuValues(path: string): number[] {
	const raw = JSON.parse(readFileSync(path, 'utf8')) as unknown;
	if (Array.isArray(raw)) return raw as number[];
	if (raw && typeof raw === 'object') {
		const record = raw as { webgpuValues?: unknown; shadingIndex?: unknown };
		if (Array.isArray(record.webgpuValues)) return record.webgpuValues as number[];
		if (Array.isArray(record.shadingIndex)) return record.shadingIndex as number[];
	}
	throw new Error(`Could not find WebGPU Shading Index values in ${path}.`);
}

function loadSunUpCount(basePath: string): number {
	const metadata = JSON.parse(readFileSync(`${basePath}.json`, 'utf8')) as {
		sun_positions?: { altitude?: number }[];
	};
	return (metadata.sun_positions ?? []).filter((sample) => (sample.altitude ?? -Infinity) > 0).length;
}

function loadKnownSolarBitFlipCounts(basePath: string, numPoints: number): number[] | undefined {
	const diagnosticsPath = `${basePath}_solar_flip_diagnostics.json`;
	if (!existsSync(diagnosticsPath)) return undefined;
	const diagnostics = JSON.parse(readFileSync(diagnosticsPath, 'utf8')) as {
		counts?: { numPoints?: number };
		topPointIndices?: { pointIndex: number; flipCount: number }[];
	};
	if (diagnostics.counts?.numPoints && diagnostics.counts.numPoints !== numPoints) return undefined;
	const counts = new Array<number>(numPoints).fill(0);
	for (const entry of diagnostics.topPointIndices ?? []) {
		if (entry.pointIndex >= 0 && entry.pointIndex < numPoints) {
			counts[entry.pointIndex] = entry.flipCount;
		}
	}
	return counts;
}

function main(): void {
	const { basePath, webgpuPath, reportPath, tolerance } = parseArgs();
	const reference = loadReferenceShadingFromBin(basePath);
	const webgpu = loadWebgpuValues(webgpuPath);
	const sunUpCount = loadSunUpCount(basePath);
	const solarBitMismatchCounts = loadKnownSolarBitFlipCounts(basePath, reference.numPoints);
	const result = compareShadingIndex({
		python: reference.shadingIndex,
		webgpu,
		positions: reference.positions,
		tolerance,
		sunUpCount,
		solarBitMismatchCounts
	});
	const report = {
		schemaVersion: 1,
		basePath,
		webgpuPath,
		reference: 'python-bin',
		metricType: 'shading_index',
		monthIndex: 7,
		startTimeIndex: 7 * reference.numHours,
		timeCount: reference.numHours,
		sunUpCount,
		tolerance,
		numPoints: result.numPoints,
		pass: result.pass,
		strictPass: result.strictPass,
		maxAbsoluteError: result.maxAbsoluteError,
		meanAbsoluteError: result.meanAbsoluteError,
		mismatchCountAboveTolerance: result.mismatchCountAboveTolerance,
		nonFinitePythonValueCount: result.nonFinitePythonValueCount,
		nonFiniteWebgpuValueCount: result.nonFiniteWebgpuValueCount,
		solarBitFlipAttributedMismatchCount: result.solarBitFlipAttributedMismatchCount,
		caveats: result.caveats,
		worstCells: result.worstCells
	};
	mkdirSync(dirname(reportPath), { recursive: true });
	writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf8');
	console.log(
		[
			`shading_index: ${result.pass ? 'PASS' : 'FAIL'}`,
			`maxAbs=${result.maxAbsoluteError}`,
			`meanAbs=${result.meanAbsoluteError}`,
			`mismatches=${result.mismatchCountAboveTolerance}`,
			`nonFinitePython=${result.nonFinitePythonValueCount}`,
			`nonFiniteWebgpu=${result.nonFiniteWebgpuValueCount}`,
			`solarFlipAttributed=${result.solarBitFlipAttributedMismatchCount}`,
			`report=${reportPath}`
		].join(' ')
	);
	process.exit(result.pass ? 0 : 1);
}

main();
