import { expect, test, type Page } from '@playwright/test';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const RESULTS_DIR = resolve(REPO_ROOT, 'data/performance-results');
const ARTIFACT_FILENAME = 'main-route-cold-start-waterfall.json';
const ARTIFACT_PATH = resolve(RESULTS_DIR, ARTIFACT_FILENAME);
const SOURCE_ROUTE = '/';
const INITIAL_MONTH_INDEX = 7;
const INITIAL_HOUR_INDEX = 0;
const FIRST_SCRUB_HOUR_INDEX = 1;
const FIRST_SCRUB_TIME_INDEX = 169;

type AnalysisCase = {
	projectLabel: string;
	analysisId: string;
	metadataPath: string;
	expectedSelectionKey: string;
	gridResolutionMeters: 2 | 0.5;
};

type AnalysisMetadata = {
	grid_size?: number;
	num_positions?: number;
};

type ProofSnapshot = {
	rendererBackend: string;
	utciRenderResolved: string;
	utciSurfaceSource: string | null;
	baseRenderTransport: string;
	dataTextureBuildCount: number;
	selectedHourRuntimeContract: {
		route: string | null;
		readbackInstrumentation: string | null;
		visibleSelectedHourReadbackCount: number | null;
		strongVisibleGpuPath: boolean | null;
	};
	baseSameDeviceForComputeAndRender: boolean | null;
	selectedMonthIndex: number;
	selectedHourIndex: number;
	selectedTimeIndex: number;
};

type CollectedPhase = {
	firstVisibleMs: number | null;
	timings: Record<string, number | null>;
	renderPublication: Record<string, unknown> | null;
	proof: ProofSnapshot;
};

type CollectedScrubPhase = {
	selectedHourIndex: 1;
	selectedTimeIndex: typeof FIRST_SCRUB_TIME_INDEX;
	visibleMs: number | null;
	surfaceRequestId: number | null;
	timings: Record<string, number | null>;
	renderPublication: Record<string, unknown> | null;
	proof: ProofSnapshot;
};

type CollectedColdCase = {
	projectLabel: string;
	analysisId: string;
	gridResolutionMeters: 2 | 0.5;
	pointCount: number;
	sourceUrl: string;
	initial: CollectedPhase;
	firstPostVisibleScrub: CollectedScrubPhase;
	assertions: {
		pythonBinDebugComparisonFieldsAbsent: true;
		initialForbiddenComparisonFieldsPresent: string[];
		scrubForbiddenComparisonFieldsPresent: string[];
		allForbiddenRequestUrls: string[];
		memoryScope: 'utci-owned-webgpu-buffers';
	};
};

type DiagnosticsSnapshot = Record<string, any>;

const CASES: AnalysisCase[] = [
	{
		projectLabel: 'Ben-Gurion',
		analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
		metadataPath: 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday.json',
		expectedSelectionKey: 'Ben-Gurion/20250815_grid_2m_fullday|7|0',
		gridResolutionMeters: 2
	},
	{
		projectLabel: 'Ness-Tziona',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		metadataPath: 'data/analyses/Ness-Tziona/exploded/nes_tziona_unblock_2.json',
		expectedSelectionKey: 'Ness-Tziona/exploded/nes_tziona_unblock_2|7|0',
		gridResolutionMeters: 2
	},
	{
		projectLabel: 'Ben-Gurion',
		analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
		metadataPath: 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday.json',
		expectedSelectionKey: 'Ben-Gurion/20250815_grid_2m_fullday|7|0',
		gridResolutionMeters: 0.5
	},
	{
		projectLabel: 'Ness-Tziona',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		metadataPath: 'data/analyses/Ness-Tziona/exploded/nes_tziona_unblock_2.json',
		expectedSelectionKey: 'Ness-Tziona/exploded/nes_tziona_unblock_2|7|0',
		gridResolutionMeters: 0.5
	}
];

function formatLocalDate(date: Date) {
	const year = date.getFullYear();
	const month = String(date.getMonth() + 1).padStart(2, '0');
	const day = String(date.getDate()).padStart(2, '0');
	return `${year}-${month}-${day}`;
}

function expectedSelectionKeyForHour(caseConfig: AnalysisCase, hourIndex: number) {
	return caseConfig.expectedSelectionKey.replace('|7|0', `|7|${hourIndex}`);
}

function readMetadata(caseConfig: AnalysisCase): AnalysisMetadata {
	return JSON.parse(readFileSync(resolve(REPO_ROOT, caseConfig.metadataPath), 'utf8')) as AnalysisMetadata;
}

function isForbiddenComparisonRequest(url: string) {
	const parsed = new URL(url);
	const isMainRouteDocument = parsed.pathname === '/' && parsed.searchParams.has('utciRenderDiagnostics');
	if (/\.bin(\?|$)/i.test(url)) return true;
	if (/loadReferenceFromFs/i.test(url)) return true;
	if (/parity/i.test(url) && !isMainRouteDocument) return true;
	return false;
}

async function readUtciRenderDiagnostics(page: Page) {
	return page.evaluate(() => (window as any).__utciRenderDiagnostics__ ?? null);
}

async function waitForSelectedHourPublication(
	page: Page,
	expectedSelectionKey: string,
	options?: { minSurfaceRequestId?: number }
) {
	const diagnostics = await page.waitForFunction(
		({ selectionKey, minSurfaceRequestId }) => {
			const value = (window as any).__utciRenderDiagnostics__;
			if (!value) return null;
			if (
				value.rendererBackend === 'webgpu' &&
				value.baseLiveReady === true &&
				value.utciSurfaceSource === 'compute-buffer-selected-hour' &&
				value.baseRenderTransport === 'compute-buffer-selected-hour' &&
				value.baseSameDeviceForComputeAndRender === true &&
				value.baseSelectionKey === selectionKey &&
				value.baseSceneSelectionKey === selectionKey &&
				typeof value.baseSurfaceRequestId === 'number' &&
				value.baseSurfaceRequestId > (minSurfaceRequestId ?? 0) &&
				value.gpuResidentCopyRequestId === value.baseSurfaceRequestId &&
				value.selectedHourRuntimeContract?.route === 'main' &&
				value.selectedHourRuntimeContract?.readbackInstrumentation === 'instrumented' &&
				value.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount === 0 &&
				value.selectedHourRuntimeContract?.strongVisibleGpuPath === true &&
				typeof value.timings?.renderPublication?.renderPublicationTimeline
					?.controllerVisibleAcknowledgedAtMs === 'number'
			) {
				return value;
			}
			return null;
		},
		{
			selectionKey: expectedSelectionKey,
			minSurfaceRequestId: options?.minSurfaceRequestId ?? 0
		},
		{ timeout: 240_000 }
	).catch(async (error) => {
		const lastDiagnostics = await readUtciRenderDiagnostics(page).catch((readError) => ({
			readError: readError instanceof Error ? readError.message : String(readError)
		}));
		const message = error instanceof Error ? error.message : String(error);
		throw new Error(
			[
				'Timed out waiting for strong main-route selected-hour diagnostics.',
				message,
				'Last window.__utciRenderDiagnostics__:',
				JSON.stringify(lastDiagnostics, null, 2)
			].join('\n')
		);
	});

	return diagnostics.jsonValue() as Promise<DiagnosticsSnapshot>;
}

function collectForbiddenComparisonFields(diagnostics: Record<string, unknown>) {
	return [
		'pythonBinComparisonActive',
		'binComparisonEnabled',
		'binComparisonValid',
		'pythonBinSampleComparison',
		'parityMode',
		'comparisonStats',
		'debugComparison'
	].filter((key) => key in diagnostics);
}

function numberOrNull(value: unknown): number | null {
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function extractTimings(timings: Record<string, unknown> | undefined) {
	return {
		payloadPrepareMs: numberOrNull(timings?.payloadPrepareMs),
		workerBvhMs: numberOrNull(timings?.workerBvhMs),
		pipelineUploadMs: numberOrNull(timings?.pipelineUploadMs),
		exposurePrecomputeMs: numberOrNull(timings?.exposurePrecomputeMs),
		exposureCommandEncodeTotalMs: numberOrNull(timings?.exposureCommandEncodeTotalMs),
		exposureEncodeMs: numberOrNull(timings?.exposureEncodeMs),
		exposureSolarEncodeMs: numberOrNull(timings?.exposureSolarEncodeMs),
		exposureSkyEncodeMs: numberOrNull(timings?.exposureSkyEncodeMs),
		exposureQueueWaitMs: numberOrNull(timings?.exposureQueueWaitMs),
		exposurePointCount: numberOrNull(timings?.exposurePointCount),
		exposureTotalTimeSteps: numberOrNull(timings?.exposureTotalTimeSteps),
		exposureDaylightTimeSteps: numberOrNull(timings?.exposureDaylightTimeSteps),
		exposurePointChunks: numberOrNull(timings?.exposurePointChunks),
		exposureSolarDispatchCount: numberOrNull(timings?.exposureSolarDispatchCount),
		exposureSkyDispatchCount: numberOrNull(timings?.exposureSkyDispatchCount),
		exposureSolarRayBudget: numberOrNull(timings?.exposureSolarRayBudget),
		exposureSkyRayBudget: numberOrNull(timings?.exposureSkyRayBudget),
		oneHourDispatchMs: numberOrNull(timings?.oneHourDispatchMs),
		firstSelectedHourReadyMs: numberOrNull(timings?.firstSelectedHourReadyMs),
		firstSelectedHourVisibleMs: numberOrNull(timings?.firstSelectedHourVisibleMs),
		renderUpdateMs: numberOrNull(timings?.renderUpdateMs),
		renderSceneSyncStartDelayMs: numberOrNull(timings?.renderSceneSyncStartDelayMs),
		renderSceneSyncTotalMs: numberOrNull(timings?.renderSceneSyncTotalMs),
		renderLayoutBuildMs: numberOrNull(timings?.renderLayoutBuildMs),
		renderSurfaceMeshMs: numberOrNull(timings?.renderSurfaceMeshMs),
		renderStorageInitWaitMs: numberOrNull(timings?.renderStorageInitWaitMs),
		renderBufferCopyMs: numberOrNull(timings?.renderBufferCopyMs),
		renderQueueDrainMs: numberOrNull(timings?.renderQueueDrainMs)
	};
}

function extractRenderPublication(
	renderPublication: unknown
): Record<string, unknown> | null {
	if (typeof renderPublication !== 'object' || renderPublication == null) {
		return null;
	}
	return renderPublication as Record<string, unknown>;
}

function getFirstVisibleAcknowledgedAtMs(diagnostics: DiagnosticsSnapshot): number | null {
	return numberOrNull(
		diagnostics.timings?.renderPublication?.renderPublicationTimeline
			?.controllerVisibleAcknowledgedAtMs
	);
}

function buildProof(diagnostics: DiagnosticsSnapshot): ProofSnapshot {
	return {
		rendererBackend: diagnostics.rendererBackend,
		utciRenderResolved: diagnostics.utciRenderResolved,
		utciSurfaceSource: diagnostics.utciSurfaceSource ?? null,
		baseRenderTransport: diagnostics.baseRenderTransport,
		dataTextureBuildCount: diagnostics.dataTextureBuildCount ?? 0,
		selectedHourRuntimeContract: {
			route: diagnostics.selectedHourRuntimeContract?.route ?? null,
			readbackInstrumentation:
				diagnostics.selectedHourRuntimeContract?.readbackInstrumentation ?? null,
			visibleSelectedHourReadbackCount:
				diagnostics.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount ?? null,
			strongVisibleGpuPath:
				diagnostics.selectedHourRuntimeContract?.strongVisibleGpuPath ?? null
		},
		baseSameDeviceForComputeAndRender: diagnostics.baseSameDeviceForComputeAndRender ?? null,
		selectedMonthIndex: diagnostics.baseSelectedMonthIndex,
		selectedHourIndex: diagnostics.baseSelectedHourIndex,
		selectedTimeIndex: diagnostics.baseSelectedTimeIndex
	};
}

function assertColdStartProofBoundary(params: {
	diagnostics: DiagnosticsSnapshot;
	forbiddenComparisonFieldsPresent: string[];
	forbiddenRequestUrls: string[];
	sourceUrl: string;
}) {
	const { diagnostics, forbiddenComparisonFieldsPresent, forbiddenRequestUrls, sourceUrl } = params;
	expect(new URL(sourceUrl, 'http://localhost').pathname).toBe('/');
	expect(diagnostics.rendererBackend).toBe('webgpu');
	expect(diagnostics.utciRenderResolved).toBe('gpuNative');
	expect(diagnostics.utciSurfaceSource).toBe('compute-buffer-selected-hour');
	expect(diagnostics.baseRenderTransport).toBe('compute-buffer-selected-hour');
	expect(diagnostics.dataTextureBuildCount).toBe(0);
	expect(diagnostics.selectedHourRuntimeContract?.route).toBe('main');
	expect(diagnostics.selectedHourRuntimeContract?.readbackInstrumentation).toBe(
		'instrumented'
	);
	expect(diagnostics.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount).toBe(0);
	expect(diagnostics.selectedHourRuntimeContract?.strongVisibleGpuPath).toBe(true);
	expect(forbiddenComparisonFieldsPresent).toEqual([]);
	expect(forbiddenRequestUrls).toEqual([]);
}

function assertLiveDiagnostics(
	diagnostics: DiagnosticsSnapshot,
	caseConfig: AnalysisCase,
	hourIndex: number
) {
	expect(diagnostics.baseMetadataGridSize).toBe(caseConfig.gridResolutionMeters);
	expect(diagnostics.basePointCount).toEqual(expect.any(Number));
	expect(diagnostics.basePointCount).toBeGreaterThan(0);
	expect(diagnostics.baseSelectedMonthIndex).toBe(INITIAL_MONTH_INDEX);
	expect(diagnostics.baseSelectedHourIndex).toBe(hourIndex);
	expect(diagnostics.baseSelectedTimeIndex).toBe(INITIAL_MONTH_INDEX * 24 + hourIndex);
	expect(diagnostics.timings?.renderPublication ?? null).not.toBeNull();
	expect(diagnostics.trackedGpuAllocationBytes).toMatchObject({
		persistentExposureBytes: expect.any(Number),
		allHoursOutputBytes: 0,
		selectedHourOutputBytes: expect.any(Number),
		selectedHourOutputBytesHighWatermark: expect.any(Number),
		renderOwnedSelectedHourBytes: expect.any(Number),
		renderOwnedSelectedHourBytesHighWatermark: expect.any(Number),
		trackingScope: 'utci-owned-webgpu-buffers'
	});
	expect(diagnostics.trackedGpuAllocationBytes.persistentExposureBytes).toBeGreaterThan(0);
	expect(diagnostics.trackedGpuAllocationBytes.renderOwnedSelectedHourBytes).toBeGreaterThan(0);
	expect(diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark).toBeGreaterThanOrEqual(
		diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes
	);
	expect(
		diagnostics.trackedGpuAllocationBytes.renderOwnedSelectedHourBytesHighWatermark
	).toBeGreaterThanOrEqual(diagnostics.trackedGpuAllocationBytes.renderOwnedSelectedHourBytes);
}

async function scrubToFirstPostVisibleHour(
	page: Page,
	caseConfig: AnalysisCase,
	initialDiagnostics: DiagnosticsSnapshot
) {
	const initialRequestId = initialDiagnostics.baseSurfaceRequestId ?? 0;
	const scrubStartedAt = performance.now();
	const hourSlider = page.getByRole('slider', { name: /select analysis hour/i });
	await expect(hourSlider).toBeVisible();
	await hourSlider.focus();
	await hourSlider.press('Home');
	await hourSlider.press('ArrowRight');
	const scrubDiagnostics = await waitForSelectedHourPublication(
		page,
		expectedSelectionKeyForHour(caseConfig, FIRST_SCRUB_HOUR_INDEX),
		{ minSurfaceRequestId: initialRequestId }
	);
	const firstPostVisibleScrubMs = performance.now() - scrubStartedAt;
	return { scrubDiagnostics, firstPostVisibleScrubMs };
}

async function collectCase(
	page: Page,
	caseConfig: AnalysisCase
): Promise<CollectedColdCase> {
	const metadata = readMetadata(caseConfig);
	const requestedUrls: string[] = [];
	page.on('request', (request) => requestedUrls.push(request.url()));

	const gridQuery =
		caseConfig.gridResolutionMeters === 0.5
			? `&gridResolution=${caseConfig.gridResolutionMeters}`
			: '';
	const sourceUrl = `/?analysis=${encodeURIComponent(caseConfig.analysisId)}${gridQuery}&utciRender=auto&utciRenderDiagnostics=1`;
	await page.goto(sourceUrl);
	const initialDiagnostics = await waitForSelectedHourPublication(
		page,
		caseConfig.expectedSelectionKey
	);
	const { scrubDiagnostics, firstPostVisibleScrubMs } = await scrubToFirstPostVisibleHour(
		page,
		caseConfig,
		initialDiagnostics
	);

	const initialForbiddenComparisonFieldsPresent =
		collectForbiddenComparisonFields(initialDiagnostics);
	const scrubForbiddenComparisonFieldsPresent =
		collectForbiddenComparisonFields(scrubDiagnostics);
	const allForbiddenRequestUrls = requestedUrls.filter(isForbiddenComparisonRequest);

	assertColdStartProofBoundary({
		diagnostics: initialDiagnostics,
		forbiddenComparisonFieldsPresent: initialForbiddenComparisonFieldsPresent,
		forbiddenRequestUrls: allForbiddenRequestUrls,
		sourceUrl
	});
	assertColdStartProofBoundary({
		diagnostics: scrubDiagnostics,
		forbiddenComparisonFieldsPresent: scrubForbiddenComparisonFieldsPresent,
		forbiddenRequestUrls: allForbiddenRequestUrls,
		sourceUrl
	});
	assertLiveDiagnostics(initialDiagnostics, caseConfig, INITIAL_HOUR_INDEX);
	assertLiveDiagnostics(scrubDiagnostics, caseConfig, FIRST_SCRUB_HOUR_INDEX);
	expect(scrubDiagnostics.baseSurfaceRequestId).toBeGreaterThan(
		initialDiagnostics.baseSurfaceRequestId ?? 0
	);
	expect(scrubDiagnostics.gpuResidentCopyRequestId).toBe(scrubDiagnostics.baseSurfaceRequestId);

	const firstSelectedHourVisibleMs = getFirstVisibleAcknowledgedAtMs(initialDiagnostics);
	expect(firstSelectedHourVisibleMs).toEqual(expect.any(Number));
	expect(firstPostVisibleScrubMs).toBeGreaterThan(0);

	await page.goto('about:blank');

	return {
		projectLabel: caseConfig.projectLabel,
		analysisId: caseConfig.analysisId,
		gridResolutionMeters: caseConfig.gridResolutionMeters,
		pointCount: initialDiagnostics.basePointCount ?? metadata.num_positions ?? 0,
		sourceUrl,
		initial: {
			firstVisibleMs: firstSelectedHourVisibleMs,
			timings: extractTimings(initialDiagnostics.timings),
			renderPublication: extractRenderPublication(initialDiagnostics.timings?.renderPublication),
			proof: buildProof(initialDiagnostics)
		},
		firstPostVisibleScrub: {
			selectedHourIndex: FIRST_SCRUB_HOUR_INDEX,
			selectedTimeIndex: FIRST_SCRUB_TIME_INDEX,
			visibleMs: firstPostVisibleScrubMs,
			surfaceRequestId: scrubDiagnostics.baseSurfaceRequestId ?? null,
			timings: extractTimings(scrubDiagnostics.timings),
			renderPublication: extractRenderPublication(scrubDiagnostics.timings?.renderPublication),
			proof: buildProof(scrubDiagnostics)
		},
		assertions: {
			pythonBinDebugComparisonFieldsAbsent: true,
			initialForbiddenComparisonFieldsPresent,
			scrubForbiddenComparisonFieldsPresent,
			allForbiddenRequestUrls,
			memoryScope: initialDiagnostics.trackedGpuAllocationBytes.trackingScope
		}
	};
}

test.describe('main route cold-start waterfall collector', () => {
	test.afterEach(async ({ page }) => {
		await page.goto('about:blank').catch(() => undefined);
	});

	test('collects BG and Ness Tziona 2m and 0.5m cold-start waterfall artifacts', async ({
		page
	}, testInfo) => {
		test.setTimeout(600_000);

		const cases: CollectedColdCase[] = [];
		for (const caseConfig of CASES) {
			cases.push(await collectCase(page, caseConfig));
		}

		const artifact = {
			collectedOn: formatLocalDate(new Date()),
			sourceRoute: SOURCE_ROUTE,
			collectionMethod:
				'Main route cold-start waterfall: / with utciRender=auto&utciRenderDiagnostics=1, collecting both initial first visible and first post-visible hour-slider scrub; no debug route and no parity/.bin comparison.',
			cases
		};

		if (!existsSync(RESULTS_DIR)) {
			mkdirSync(RESULTS_DIR, { recursive: true });
		}

		const json = JSON.stringify(artifact, null, 2);
		writeFileSync(ARTIFACT_PATH, json, 'utf8');
		await testInfo.attach(ARTIFACT_FILENAME, {
			body: json,
			contentType: 'application/json'
		});
	});
});
