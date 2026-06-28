import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadReferenceIntermediatesFromFs } from '../src/lib/parity/loadReferenceIntermediatesFromFs';
import { loadWebgpuCollectedFromFs } from '../src/lib/parity/loadWebgpuCollectedFromFs';
import {
	extractTopMrtDeltas,
	type MrtTermName,
	type OptionalTermSeries
} from '../src/lib/parity/mrtWorstCellDiagnostics';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, '..', '..');
const DEFAULT_BASE_PATH = 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday';

interface CliArgs {
	basePath: string;
	parityReportPath: string | null;
	flipReportPath: string | null;
	topN: number;
}

function resolveRepoPath(pathValue: string): string {
	return resolve(REPO_ROOT, pathValue);
}

function parseArgs(): CliArgs {
	let basePath = process.env.PARITY_BASE_PATH ?? DEFAULT_BASE_PATH;
	let parityReportPath: string | null = null;
	let flipReportPath: string | null = null;
	let topN = 10;
	const positional: string[] = [];
	const argv = process.argv.slice(2);
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		if (arg === '--base-path' && argv[i + 1]) {
			basePath = argv[++i];
		} else if (arg.startsWith('--base-path=')) {
			basePath = arg.slice('--base-path='.length);
		} else if (arg === '--parity-report' && argv[i + 1]) {
			parityReportPath = resolveRepoPath(argv[++i]);
		} else if (arg.startsWith('--parity-report=')) {
			parityReportPath = resolveRepoPath(arg.slice('--parity-report='.length));
		} else if (arg === '--from-flips' && argv[i + 1]) {
			flipReportPath = resolveRepoPath(argv[++i]);
		} else if (arg.startsWith('--from-flips=')) {
			flipReportPath = resolveRepoPath(arg.slice('--from-flips='.length));
		} else if (arg === '--top' && argv[i + 1]) {
			topN = Number.parseInt(argv[++i], 10);
		} else if (arg.startsWith('--top=')) {
			topN = Number.parseInt(arg.slice('--top='.length), 10);
		} else {
			positional.push(arg);
		}
	}
	if (positional[0]) {
		basePath = positional[0];
	}
	if (positional[1]) {
		const parsed = Number.parseInt(positional[1], 10);
		if (!Number.isNaN(parsed)) topN = parsed;
	}
	return {
		basePath: resolve(REPO_ROOT, basePath),
		parityReportPath,
		flipReportPath,
		topN
	};
}

function safeReadJson(path: string): unknown | null {
	try {
		return JSON.parse(readFileSync(path, 'utf8')) as unknown;
	} catch (error) {
		console.warn(`[diagnose-mrt] ignoring invalid JSON at ${path}: ${error instanceof Error ? error.message : String(error)}`);
		return null;
	}
}

function extractCandidateIndicesFromParityReport(report: unknown): number[] {
	if (typeof report !== 'object' || report === null) return [];
	const root = report as Record<string, unknown>;
	const mrt = root.mrt;
	if (typeof mrt !== 'object' || mrt === null) return [];
	const detail = (mrt as Record<string, unknown>).detail;
	if (typeof detail !== 'object' || detail === null) return [];
	const worst = (detail as Record<string, unknown>).worstIndices;
	if (!Array.isArray(worst)) return [];
	return worst
		.map((row) => {
			if (typeof row !== 'object' || row === null) return null;
			const index = (row as Record<string, unknown>).index;
			return typeof index === 'number' && Number.isInteger(index) ? index : null;
		})
		.filter((value): value is number => value !== null);
}

function extractCandidateIndicesFromFlipReport(report: unknown): number[] {
	if (typeof report !== 'object' || report === null) return [];
	const root = report as Record<string, unknown>;
	const cells = root.topAffectedCells;
	if (!Array.isArray(cells)) return [];
	return cells
		.map((cell) => {
			if (typeof cell !== 'object' || cell === null) return null;
			const index = (cell as Record<string, unknown>).flatIndex;
			return typeof index === 'number' && Number.isInteger(index) ? index : null;
		})
		.filter((value): value is number => value !== null);
}

function pathKey(pathValue: string): string {
	return resolve(pathValue).replace(/\\/g, '/').toLowerCase();
}

function validateFlipReportCompatibility(report: unknown, expectedBasePath: string, numPositions: number, numHours: number): boolean {
	if (typeof report !== 'object' || report === null) return false;
	const root = report as Record<string, unknown>;
	const reportBasePath = root.basePath;
	if (typeof reportBasePath === 'string' && pathKey(reportBasePath) !== pathKey(expectedBasePath)) {
		return false;
	}
	const counts = root.counts;
	if (typeof counts !== 'object' || counts === null) return false;
	const c = counts as Record<string, unknown>;
	return (
		c.numPoints === numPositions &&
		c.numHours === numHours &&
		c.totalCells === numPositions * numHours
	);
}

function toFixed(value: number | undefined): string {
	if (value === undefined || Number.isNaN(value)) return '';
	return value.toFixed(3);
}

function printTable(rows: Array<Record<string, string>>): void {
	if (rows.length === 0) {
		console.log('No rows to print.');
		return;
	}
	const keys = Object.keys(rows[0]);
	const widths = keys.map((key) =>
		Math.max(
			key.length,
			...rows.map((row) => {
				const value = row[key] ?? '';
				return value.length;
			})
		)
	);
	const render = (values: string[]): string => values.map((v, i) => v.padEnd(widths[i], ' ')).join(' | ');
	console.log(render(keys));
	console.log(widths.map((w) => '-'.repeat(w)).join('-|-'));
	for (const row of rows) {
		console.log(render(keys.map((key) => row[key] ?? '')));
	}
}

async function loadOptionalTerms(basePath: string): Promise<OptionalTermSeries> {
	const terms: OptionalTermSeries = {};
	const webgpu = await loadWebgpuCollectedFromFs(basePath);
	const refMrt = await loadReferenceIntermediatesFromFs(basePath, 'mrt');

	const addMrtTerm = (name: MrtTermName): void => {
		if (name === 'solar' || name === 'sky') return;
		const refSeries = refMrt[name];
		const webgpuSeries = webgpu.mrt?.[name];
		if (refSeries && webgpuSeries) {
			terms[name] = { ref: refSeries, webgpu: webgpuSeries };
		}
	};

	addMrtTerm('short_erf');
	addMrtTerm('long_erf');
	addMrtTerm('short_dmrt');
	addMrtTerm('long_dmrt');

	const refSolar = await loadReferenceIntermediatesFromFs(basePath, 'solar').catch(() => null);
	const webgpuSolar = webgpu.solar;
	if (refSolar?.solarExposure && webgpuSolar?.solarExposure) {
		terms.solar = { ref: refSolar.solarExposure, webgpu: webgpuSolar.solarExposure };
	}

	const refSky = await loadReferenceIntermediatesFromFs(basePath, 'sky').catch(() => null);
	const webgpuSky = webgpu.sky;
	if (refSky?.skyExposure && webgpuSky?.skyExposure) {
		if (refSky.numPositions !== refMrt.numPositions || webgpuSky.numPositions !== refMrt.numPositions) {
			throw new Error(
				`Sky/MRT numPositions mismatch: refSky=${refSky.numPositions}, webgpuSky=${webgpuSky.numPositions}, refMrt=${refMrt.numPositions}`
			);
		}
		const expandedRef = new Array<number>(refMrt.numPositions * refMrt.numHours);
		const expandedWebgpu = new Array<number>(refMrt.numPositions * refMrt.numHours);
		for (let hour = 0; hour < refMrt.numHours; hour++) {
			for (let point = 0; point < refMrt.numPositions; point++) {
				const index = point * refMrt.numHours + hour;
				expandedRef[index] = refSky.skyExposure[point];
				expandedWebgpu[index] = webgpuSky.skyExposure[point];
			}
		}
		terms.sky = { ref: expandedRef, webgpu: expandedWebgpu };
	}

	return terms;
}

async function main(): Promise<void> {
	const { basePath, parityReportPath, flipReportPath, topN } = parseArgs();
	if (!Number.isInteger(topN) || topN <= 0) {
		throw new Error(`--top must be a positive integer, got ${topN}`);
	}

	const refMrt = await loadReferenceIntermediatesFromFs(basePath, 'mrt');
	const webgpu = await loadWebgpuCollectedFromFs(basePath);
	if (!webgpu.mrt) {
		throw new Error(`Missing WebGPU MRT artifact: ${basePath}_webgpu_mrt.json`);
	}
	if (refMrt.numPositions !== webgpu.mrt.numPositions || refMrt.numHours !== webgpu.mrt.numHours) {
		throw new Error(
			`MRT shape mismatch: ref=(${refMrt.numPositions},${refMrt.numHours}) webgpu=(${webgpu.mrt.numPositions},${webgpu.mrt.numHours})`
		);
	}

	const parityPath =
		parityReportPath ??
		(existsSync(resolve(REPO_ROOT, 'viewer/parity-report.json')) ? resolve(REPO_ROOT, 'viewer/parity-report.json') : null);
	const parityReport = parityPath ? safeReadJson(parityPath) : null;
	const defaultFlipPath = `${basePath}_solar_flip_diagnostics.json`;
	const resolvedFlipPath = flipReportPath ?? (existsSync(defaultFlipPath) ? defaultFlipPath : null);
	const flipReport = resolvedFlipPath ? safeReadJson(resolvedFlipPath) : null;
	const flipReportCompatible =
		flipReport != null && validateFlipReportCompatibility(flipReport, basePath, refMrt.numPositions, refMrt.numHours);
	if (flipReport != null && !flipReportCompatible) {
		console.warn(`[diagnose-mrt] ignoring incompatible flip report: ${resolvedFlipPath ?? '(unknown path)'}`);
	}
	const flipIndices = flipReportCompatible ? extractCandidateIndicesFromFlipReport(flipReport) : [];
	const parityIndices = extractCandidateIndicesFromParityReport(parityReport);
	const candidateIndices = flipIndices.length > 0 ? flipIndices : parityIndices;
	const terms = await loadOptionalTerms(basePath);

	const rows = extractTopMrtDeltas({
		refMrt: refMrt.mrt,
		webgpuMrt: webgpu.mrt.mrt,
		numPositions: refMrt.numPositions,
		topN,
		indices: candidateIndices.length > 0 ? candidateIndices : undefined,
		terms
	});

	console.log(`MRT worst-cell diagnostics for ${basePath}`);
	console.log(`numPositions=${refMrt.numPositions}, numHours=${refMrt.numHours}, topN=${topN}`);
	console.log(
		flipIndices.length > 0
			? `candidate source: solar flip report topAffectedCells (${flipIndices.length})`
			: parityIndices.length > 0
				? `candidate source: parity report worstIndices (${parityIndices.length})`
			: 'candidate source: full MRT domain scan'
	);

	if (rows.length === 0) {
		console.log('No MRT rows found. Check artifact consistency.');
		return;
	}

	const tableRows = rows.map((row, rank) => ({
		rank: String(rank + 1),
		hour: String(row.hour),
		point: String(row.pointIndex),
		idx: String(row.index),
		mrtRef: toFixed(row.ref),
		mrtWeb: toFixed(row.webgpu),
		mrtDelta: toFixed(row.diff),
		domTerm: row.dominantTerm ?? '',
		domDelta: toFixed(row.dominantTermDelta),
		termAbsSum: toFixed(row.termAbsSum),
		shortErfD: toFixed(row.termDeltas.short_erf),
		longErfD: toFixed(row.termDeltas.long_erf),
		shortDmrtD: toFixed(row.termDeltas.short_dmrt),
		longDmrtD: toFixed(row.termDeltas.long_dmrt),
		solarD: toFixed(row.termDeltas.solar),
		skyD: toFixed(row.termDeltas.sky)
	}));
	printTable(tableRows);

	const worst = rows[0];
	console.log(
		`Worst MRT cell: hour=${worst.hour}, point=${worst.pointIndex}, idx=${worst.index}, ref=${worst.ref.toFixed(3)}, webgpu=${worst.webgpu.toFixed(3)}, delta=${worst.diff.toFixed(3)}`
	);
}

main().catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
});
