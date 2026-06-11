import { expect, test, type Page } from '@playwright/test';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { compareShadingIndex } from '../parity/compareShadingIndex';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const ANALYSIS_ID = 'Ben-Gurion/20250815_grid_2m_fullday';
const BASE_PATH = resolve(REPO_ROOT, 'data/analyses', ANALYSIS_ID);
const REPORT_PATH = resolve(
	REPO_ROOT,
	'data/batch-parity-results/Ben-Gurion_20250815_grid_2m_fullday_shading-index-parity.json'
);
const TOLERANCE = 1e-6;
const WAIT_MS = Number(process.env.SHADING_INDEX_PARITY_WAIT_MS ?? '180000');

type DebugShadingIndexParitySnapshot = {
	status: 'pending' | 'success' | 'error';
	route: 'debug';
	source: 'debug-shared-host';
	metricType: 'shading_index';
	analysisId: string;
	monthIndex: number;
	startTimeIndex: number;
	timeCount: number;
	numPoints: number;
	outputBytes?: number;
	pythonValues?: number[];
	webgpuValues?: number[];
	positions?: number[];
	error?: string;
};

function loadSunUpCount(): number {
	const metadata = JSON.parse(readFileSync(`${BASE_PATH}.json`, 'utf8')) as {
		sun_positions?: { altitude?: number }[];
	};
	const sunPositions = metadata.sun_positions ?? [];
	return sunPositions.filter((sample) => (sample.altitude ?? -Infinity) > 0).length;
}

function loadKnownSolarBitFlipCounts(numPoints: number): number[] | undefined {
	const diagnosticsPath = `${BASE_PATH}_solar_flip_diagnostics.json`;
	if (!existsSync(diagnosticsPath)) return undefined;
	const diagnostics = JSON.parse(readFileSync(diagnosticsPath, 'utf8')) as {
		counts?: { flipCells?: number; numPoints?: number };
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

async function readSnapshot(page: Page): Promise<DebugShadingIndexParitySnapshot | null> {
	return page.evaluate(() => (window as any).__debugShadingIndexParity__ ?? null);
}

async function waitForSnapshot(page: Page): Promise<DebugShadingIndexParitySnapshot> {
	await page
		.waitForFunction(
			() => {
				const snapshot = (window as any).__debugShadingIndexParity__;
				return snapshot?.status === 'success' || snapshot?.status === 'error';
			},
			undefined,
			{ timeout: WAIT_MS, polling: 1000 }
		)
		.catch(async (error) => {
			const snapshot = await readSnapshot(page).catch((readError) => ({
				readError: readError instanceof Error ? readError.message : String(readError)
			}));
			const diagnostics = await page
				.evaluate(() => (window as any).__onDemandPrototypeDiagnostics__ ?? null)
				.catch(() => null);
			throw new Error(
				[
					'Timed out waiting for debug-route Shading Index parity snapshot.',
					error instanceof Error ? error.message : String(error),
					`snapshot=${JSON.stringify(snapshot)}`,
					`diagnostics=${JSON.stringify(diagnostics)}`
				].join('\n')
			);
		});
	const snapshot = await readSnapshot(page);
	if (!snapshot) throw new Error('Debug route did not expose __debugShadingIndexParity__.');
	if (snapshot.status === 'error') {
		throw new Error(`Debug-route Shading Index parity snapshot failed: ${snapshot.error ?? 'unknown error'}`);
	}
	return snapshot;
}

test.describe('debug route Shading Index .bin vs WebGPU parity', () => {
	test('compares debug-loaded Python .bin shading reference with debug WebGPU shading output', async ({
		page
	}) => {
		test.setTimeout(WAIT_MS + 60_000);
		const requests: string[] = [];
		page.on('request', (request) => requests.push(request.url()));

		await page.goto(
			`/debug?parity=0&analysis=${encodeURIComponent(
				ANALYSIS_ID
			)}&utciRender=auto&utciOnDemand=f32&metric=shading_index&monthIndex=7&shadingIndexParity=1`,
			{ waitUntil: 'domcontentloaded', timeout: 15_000 }
		);
		const hasWebgpu = await page.evaluate(() => Boolean(navigator.gpu));
		test.skip(!hasWebgpu, 'WebGPU unavailable in this browser runtime.');

		const snapshot = await waitForSnapshot(page);
		expect(snapshot.route).toBe('debug');
		expect(snapshot.source).toBe('debug-shared-host');
		expect(snapshot.metricType).toBe('shading_index');
		expect(snapshot.analysisId).toBe(ANALYSIS_ID);
		expect(snapshot.monthIndex).toBe(7);
		expect(snapshot.startTimeIndex).toBe(7 * 24);
		expect(snapshot.timeCount).toBe(24);
		expect(snapshot.outputBytes).toBe(snapshot.numPoints * 4);
		expect(snapshot.pythonValues?.length).toBe(snapshot.numPoints);
		expect(snapshot.webgpuValues?.length).toBe(snapshot.numPoints);
		expect(snapshot.positions?.length).toBe(snapshot.numPoints * 3);
		expect(requests.some((url) => /\/debug(?:[/?#]|$)/.test(new URL(url).pathname))).toBe(true);
		expect(requests.some((url) => /\.bin(?:$|[?#])/.test(url))).toBe(true);

		const sunUpCount = loadSunUpCount();
		const solarBitMismatchCounts = loadKnownSolarBitFlipCounts(snapshot.numPoints);
		const result = compareShadingIndex({
			python: snapshot.pythonValues ?? [],
			webgpu: snapshot.webgpuValues ?? [],
			positions: snapshot.positions,
			tolerance: TOLERANCE,
			sunUpCount,
			solarBitMismatchCounts
		});

		const report = {
			schemaVersion: 1,
			analysisId: ANALYSIS_ID,
			route: 'debug',
			source: 'debug-shared-host',
			reference: 'python-bin',
			metricType: 'shading_index',
			monthIndex: snapshot.monthIndex,
			startTimeIndex: snapshot.startTimeIndex,
			timeCount: snapshot.timeCount,
			sunUpCount,
			tolerance: TOLERANCE,
			numPoints: result.numPoints,
			outputBytes: snapshot.outputBytes,
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
		mkdirSync(resolve(REPO_ROOT, 'data/batch-parity-results'), { recursive: true });
		writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2), 'utf8');

		expect(
			result.pass,
			[
				`Shading Index parity failed; report=${REPORT_PATH}`,
				`maxAbsoluteError=${result.maxAbsoluteError}`,
				`meanAbsoluteError=${result.meanAbsoluteError}`,
				`mismatchCountAboveTolerance=${result.mismatchCountAboveTolerance}`,
				`nonFinitePythonValueCount=${result.nonFinitePythonValueCount}`,
				`nonFiniteWebgpuValueCount=${result.nonFiniteWebgpuValueCount}`,
				`solarBitFlipAttributedMismatchCount=${result.solarBitFlipAttributedMismatchCount}`,
				`worstCells=${JSON.stringify(result.worstCells.slice(0, 3))}`
			].join(' | ')
		).toBe(true);
	});
});
