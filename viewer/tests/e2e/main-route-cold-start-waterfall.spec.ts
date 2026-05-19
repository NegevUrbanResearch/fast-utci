import { expect, test, type Page } from '@playwright/test';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const RESULTS_DIR = resolve(REPO_ROOT, 'data/performance-results');
const ARTIFACT_FILENAME = 'main-route-cold-start-waterfall.json';
const ARTIFACT_PATH = resolve(RESULTS_DIR, ARTIFACT_FILENAME);
const COLLECTED_ON = '2026-05-18';
const SOURCE_ROUTE = '/';

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

type CollectedCase = {
	projectLabel: string;
	analysisId: string;
	gridResolutionMeters: 2 | 0.5;
	colorMode: 'normalized';
	phase: 'cold-initial';
	pointCount: number;
	firstSelectedHourVisibleAtMs: number | null;
	firstSelectedHourVisibleProvenance: string;
	sourceUrl: string;
	timings: Record<string, number | null>;
	coldStart: Record<string, number | null>;
	renderPublication: Record<string, unknown> | null;
	trackedGpuAllocationBytes: {
		persistentExposureBytes: number;
		allHoursOutputBytes: number;
		selectedHourOutputBytes: number;
		selectedHourOutputBytesHighWatermark: number;
		renderOwnedSelectedHourBytes: number;
		renderOwnedSelectedHourBytesHighWatermark: number;
		trackingScope: string;
	};
	ownedGpuMemoryBytes: number;
	proof: {
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
	assertions: {
		pythonBinDebugComparisonFieldsAbsent: true;
		forbiddenComparisonFieldsPresent: string[];
		forbiddenRequestUrls: string[];
		memoryScope: 'utci-owned-webgpu-buffers';
	};
};

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
	expectedSelectionKey: string
) {
	const diagnostics = await page.waitForFunction(
		(selectionKey) => {
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
		expectedSelectionKey,
		{ timeout: 60_000 }
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

	return diagnostics.jsonValue() as Promise<any>;
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

function extractColdStart(coldStart: Record<string, unknown> | undefined) {
	return {
		routeAnalysisLoadStartedAtMs: numberOrNull(coldStart?.routeAnalysisLoadStartedAtMs),
		routeAnalysisLoadCompletedAtMs: numberOrNull(coldStart?.routeAnalysisLoadCompletedAtMs),
		modelLoadStartedAtMs: numberOrNull(coldStart?.modelLoadStartedAtMs),
		modelLoadCompletedAtMs: numberOrNull(coldStart?.modelLoadCompletedAtMs),
		modelProcessingStartedAtMs: numberOrNull(coldStart?.modelProcessingStartedAtMs),
		modelProcessingCompletedAtMs: numberOrNull(coldStart?.modelProcessingCompletedAtMs),
		sessionPrepareStartedAtMs: numberOrNull(coldStart?.sessionPrepareStartedAtMs),
		sessionPrepareCompletedAtMs: numberOrNull(coldStart?.sessionPrepareCompletedAtMs),
		sessionPayloadPrepareStartedAtMs: numberOrNull(coldStart?.sessionPayloadPrepareStartedAtMs),
		sessionPayloadPrepareCompletedAtMs: numberOrNull(coldStart?.sessionPayloadPrepareCompletedAtMs),
		sessionWorkerBvhStartedAtMs: numberOrNull(coldStart?.sessionWorkerBvhStartedAtMs),
		sessionWorkerBvhCompletedAtMs: numberOrNull(coldStart?.sessionWorkerBvhCompletedAtMs),
		sessionPipelineUploadStartedAtMs: numberOrNull(coldStart?.sessionPipelineUploadStartedAtMs),
		sessionPipelineUploadCompletedAtMs: numberOrNull(coldStart?.sessionPipelineUploadCompletedAtMs),
		exposurePrecomputeStartedAtMs: numberOrNull(coldStart?.exposurePrecomputeStartedAtMs),
		exposurePrecomputeCompletedAtMs: numberOrNull(coldStart?.exposurePrecomputeCompletedAtMs),
		firstSelectedHourDispatchStartedAtMs: numberOrNull(
			coldStart?.firstSelectedHourDispatchStartedAtMs
		),
		firstSelectedHourDispatchCompletedAtMs: numberOrNull(
			coldStart?.firstSelectedHourDispatchCompletedAtMs
		),
		firstSelectedHourReadyAtMs: numberOrNull(coldStart?.firstSelectedHourReadyAtMs)
	};
}

function buildColdStartSnapshot(diagnostics: Record<string, any>) {
	const coldStart = extractColdStart(diagnostics.coldStart);
	const visibleAcknowledgedAtMs = numberOrNull(
		diagnostics.timings?.renderPublication?.renderPublicationTimeline
			?.controllerVisibleAcknowledgedAtMs
	);
	const coldStartVisibleAtMs = numberOrNull(diagnostics.coldStart?.firstSelectedHourVisibleAtMs);
	if (coldStartVisibleAtMs != null && visibleAcknowledgedAtMs != null) {
		expect(coldStartVisibleAtMs).toBe(visibleAcknowledgedAtMs);
	}
	return coldStart;
}

function getFirstVisibleAcknowledgedAtMs(diagnostics: Record<string, any>): number | null {
	return numberOrNull(
		diagnostics.timings?.renderPublication?.renderPublicationTimeline
			?.controllerVisibleAcknowledgedAtMs
	);
}

function assertCompletedAfterStarted(
	coldStart: Record<string, unknown>,
	completedField: string,
	startedField: string
) {
	expect(coldStart[completedField], `${completedField} should be numeric`).toEqual(
		expect.any(Number)
	);
	expect(coldStart[startedField], `${startedField} should be numeric`).toEqual(
		expect.any(Number)
	);
	expect(coldStart[completedField] as number).toBeGreaterThanOrEqual(
		coldStart[startedField] as number
	);
}

function assertColdStartProofBoundary(params: {
	diagnostics: Record<string, any>;
	forbiddenComparisonFieldsPresent: string[];
	forbiddenRequestUrls: string[];
	sourceUrl: string;
	gridResolutionMeters: 2 | 0.5;
}) {
	const {
		diagnostics,
		forbiddenComparisonFieldsPresent,
		forbiddenRequestUrls,
		sourceUrl,
		gridResolutionMeters
	} = params;
	expect(new URL(sourceUrl, 'http://localhost').pathname).toBe('/');
	expect(diagnostics.rendererBackend).toBe('webgpu');
	expect(diagnostics.utciRenderResolved).toBe('gpuNative');
	expect(diagnostics.utciSurfaceSource).toBe('compute-buffer-selected-hour');
	expect(diagnostics.baseRenderTransport).toBe('compute-buffer-selected-hour');
	expect(diagnostics.dataTextureBuildCount).toBe(0);
	expect(diagnostics.baseMetadataGridSize).toBe(gridResolutionMeters);
	expect(diagnostics.selectedHourRuntimeContract?.route).toBe('main');
	expect(diagnostics.selectedHourRuntimeContract?.readbackInstrumentation).toBe(
		'instrumented'
	);
	expect(diagnostics.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount).toBe(0);
	expect(diagnostics.selectedHourRuntimeContract?.strongVisibleGpuPath).toBe(true);
	expect(forbiddenComparisonFieldsPresent).toEqual([]);
	expect(forbiddenRequestUrls).toEqual([]);
}

function assertRequiredColdStartFields(coldStart: Record<string, unknown> | undefined) {
	expect(coldStart).toMatchObject({
		routeAnalysisLoadStartedAtMs: expect.any(Number),
		routeAnalysisLoadCompletedAtMs: expect.any(Number),
		modelLoadStartedAtMs: expect.any(Number),
		modelLoadCompletedAtMs: expect.any(Number),
		sessionPrepareStartedAtMs: expect.any(Number),
		sessionPrepareCompletedAtMs: expect.any(Number),
		exposurePrecomputeStartedAtMs: expect.any(Number),
		exposurePrecomputeCompletedAtMs: expect.any(Number),
		firstSelectedHourDispatchStartedAtMs: expect.any(Number),
		firstSelectedHourDispatchCompletedAtMs: expect.any(Number),
		firstSelectedHourReadyAtMs: expect.any(Number)
	});
	if (!coldStart) {
		throw new Error('Missing coldStart diagnostics payload.');
	}
	assertCompletedAfterStarted(
		coldStart,
		'routeAnalysisLoadCompletedAtMs',
		'routeAnalysisLoadStartedAtMs'
	);
	assertCompletedAfterStarted(coldStart, 'modelLoadCompletedAtMs', 'modelLoadStartedAtMs');
	if (
		typeof coldStart.modelProcessingStartedAtMs === 'number' ||
		typeof coldStart.modelProcessingCompletedAtMs === 'number'
	) {
		assertCompletedAfterStarted(
			coldStart,
			'modelProcessingCompletedAtMs',
			'modelProcessingStartedAtMs'
		);
	}
	assertCompletedAfterStarted(
		coldStart,
		'sessionPrepareCompletedAtMs',
		'sessionPrepareStartedAtMs'
	);
	assertCompletedAfterStarted(
		coldStart,
		'sessionPayloadPrepareCompletedAtMs',
		'sessionPayloadPrepareStartedAtMs'
	);
	assertCompletedAfterStarted(
		coldStart,
		'sessionWorkerBvhCompletedAtMs',
		'sessionWorkerBvhStartedAtMs'
	);
	assertCompletedAfterStarted(
		coldStart,
		'sessionPipelineUploadCompletedAtMs',
		'sessionPipelineUploadStartedAtMs'
	);
	assertCompletedAfterStarted(
		coldStart,
		'exposurePrecomputeCompletedAtMs',
		'exposurePrecomputeStartedAtMs'
	);
	assertCompletedAfterStarted(
		coldStart,
		'firstSelectedHourDispatchCompletedAtMs',
		'firstSelectedHourDispatchStartedAtMs'
	);
}

function assertRequiredExposureSplitTimings(timings: Record<string, unknown> | undefined) {
	expect(timings).toMatchObject({
		exposurePrecomputeMs: expect.any(Number),
		exposureCommandEncodeTotalMs: expect.any(Number),
		exposureEncodeMs: expect.any(Number),
		exposureSolarEncodeMs: expect.any(Number),
		exposureSkyEncodeMs: expect.any(Number),
		exposureQueueWaitMs: expect.any(Number),
		exposurePointCount: expect.any(Number),
		exposureTotalTimeSteps: expect.any(Number),
		exposureDaylightTimeSteps: expect.any(Number),
		exposurePointChunks: expect.any(Number),
		exposureSolarDispatchCount: expect.any(Number),
		exposureSkyDispatchCount: expect.any(Number),
		exposureSolarRayBudget: expect.any(Number),
		exposureSkyRayBudget: expect.any(Number)
	});
	if (!timings) throw new Error('Missing diagnostics timings payload.');
	expect(timings.exposureQueueWaitMs as number).toBeGreaterThan(0);
	expect(timings.exposurePointCount).toBeGreaterThan(0);
	expect(timings.exposureTotalTimeSteps).toBeGreaterThan(0);
	expect(timings.exposureDaylightTimeSteps).toBeGreaterThan(0);
	expect(timings.exposureSolarDispatchCount).toBeGreaterThan(0);
	expect(timings.exposureSkyDispatchCount).toBeGreaterThan(0);
	expect(timings.exposureSolarRayBudget).toBe(
		(timings.exposurePointCount as number) * (timings.exposureDaylightTimeSteps as number)
	);
	expect(timings.exposureSkyRayBudget).toBe((timings.exposurePointCount as number) * 145);
}

async function collectCase(
	page: Page,
	caseConfig: AnalysisCase
): Promise<CollectedCase> {
	const metadata = readMetadata(caseConfig);
	const requestedUrls: string[] = [];
	page.on('request', (request) => requestedUrls.push(request.url()));

	const gridQuery =
		caseConfig.gridResolutionMeters === 0.5
			? `&gridResolution=${caseConfig.gridResolutionMeters}`
			: '';
	const sourceUrl = `/?analysis=${encodeURIComponent(caseConfig.analysisId)}${gridQuery}&utciRender=auto&utciRenderDiagnostics=1`;
	await page.goto(sourceUrl);
	const diagnostics = await waitForSelectedHourPublication(page, caseConfig.expectedSelectionKey);
	const forbiddenComparisonFieldsPresent = collectForbiddenComparisonFields(diagnostics);
	const forbiddenRequestUrls = requestedUrls.filter(isForbiddenComparisonRequest);
	const trackedGpuAllocationBytes = diagnostics.trackedGpuAllocationBytes;
	const coldStart = buildColdStartSnapshot(diagnostics);
	const firstSelectedHourVisibleAtMs = getFirstVisibleAcknowledgedAtMs(diagnostics);

	assertColdStartProofBoundary({
		diagnostics,
		forbiddenComparisonFieldsPresent,
		forbiddenRequestUrls,
		sourceUrl,
		gridResolutionMeters: caseConfig.gridResolutionMeters
	});
	assertRequiredColdStartFields(coldStart);
	expect(firstSelectedHourVisibleAtMs).toEqual(expect.any(Number));
	expect(firstSelectedHourVisibleAtMs as number).toBeGreaterThanOrEqual(
		coldStart.firstSelectedHourReadyAtMs as number
	);
	assertRequiredExposureSplitTimings(diagnostics.timings);
	expect(diagnostics.timings?.renderPublication ?? null).not.toBeNull();
	expect(diagnostics.baseSelectedMonthIndex).toBe(7);
	expect(diagnostics.baseSelectedHourIndex).toBe(0);
	expect(diagnostics.baseSelectedTimeIndex).toBe(7 * 24);
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

	await page.goto('about:blank');

	return {
		projectLabel: caseConfig.projectLabel,
		analysisId: caseConfig.analysisId,
		gridResolutionMeters: caseConfig.gridResolutionMeters,
		colorMode: 'normalized',
		phase: 'cold-initial',
		pointCount: diagnostics.basePointCount ?? metadata.num_positions ?? 0,
		firstSelectedHourVisibleAtMs,
		firstSelectedHourVisibleProvenance:
			'renderPublication.renderPublicationTimeline.controllerVisibleAcknowledgedAtMs',
		sourceUrl,
		timings: extractTimings(diagnostics.timings),
		coldStart,
		renderPublication: diagnostics.timings?.renderPublication ?? null,
		trackedGpuAllocationBytes: {
			persistentExposureBytes: trackedGpuAllocationBytes.persistentExposureBytes,
			allHoursOutputBytes: trackedGpuAllocationBytes.allHoursOutputBytes,
			selectedHourOutputBytes: trackedGpuAllocationBytes.selectedHourOutputBytes,
			selectedHourOutputBytesHighWatermark:
				trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark,
			renderOwnedSelectedHourBytes:
				trackedGpuAllocationBytes.renderOwnedSelectedHourBytes,
			renderOwnedSelectedHourBytesHighWatermark:
				trackedGpuAllocationBytes.renderOwnedSelectedHourBytesHighWatermark,
			trackingScope: trackedGpuAllocationBytes.trackingScope
		},
		ownedGpuMemoryBytes:
			trackedGpuAllocationBytes.persistentExposureBytes +
			trackedGpuAllocationBytes.allHoursOutputBytes +
			trackedGpuAllocationBytes.selectedHourOutputBytes +
			trackedGpuAllocationBytes.renderOwnedSelectedHourBytes,
		proof: {
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
			baseSameDeviceForComputeAndRender:
				diagnostics.baseSameDeviceForComputeAndRender ?? null,
			selectedMonthIndex: diagnostics.baseSelectedMonthIndex,
			selectedHourIndex: diagnostics.baseSelectedHourIndex,
			selectedTimeIndex: diagnostics.baseSelectedTimeIndex
		},
		assertions: {
			pythonBinDebugComparisonFieldsAbsent: true,
			forbiddenComparisonFieldsPresent,
			forbiddenRequestUrls,
			memoryScope: trackedGpuAllocationBytes.trackingScope
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
		test.setTimeout(300_000);

		const cases: CollectedCase[] = [];
		for (const caseConfig of CASES) {
			cases.push(await collectCase(page, caseConfig));
		}

		const artifact = {
			collectedOn: COLLECTED_ON,
			sourceRoute: SOURCE_ROUTE,
			collectionMethod:
				'Main route cold-start waterfall: / with utciRender=auto&utciRenderDiagnostics=1, initial selected hour only, no debug route and no parity/.bin comparison.',
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
