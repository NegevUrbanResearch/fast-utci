import { expect, test, type Page } from '@playwright/test';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const RESULTS_DIR = resolve(REPO_ROOT, 'data/performance-results');
const ARTIFACT_PATH = resolve(RESULTS_DIR, 'main-route-selected-hour-0_5m-base.json');
const COLLECTED_ON = '2026-05-15';
const SOURCE_ROUTE = '/';
const TARGET_GRID_RESOLUTION_METERS = 0.5;
const SCRUB_HOUR_INDEX = 1;

type ColorMode = 'normalized' | 'discrete';
type CollectionPhase = 'initial' | 'scrub';

type AnalysisCase = {
	projectLabel: string;
	analysisId: string;
	metadataPath: string;
};

type AnalysisMetadata = {
	grid_size?: number;
	num_positions?: number;
};

type CollectedSample = {
	phase: CollectionPhase;
	colorMode: ColorMode;
	collectionMethod: string;
	selectedMonthIndex: number;
	selectedHourIndex: number;
	selectedTimeIndex: number;
	selectionKey: string | null;
	surfaceRequestId: number | null;
	collectedAt: string;
	sourceRoute: '/';
	sourceUrl: string;
	timings: Record<string, number | null>;
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
	};
};

type CollectedCase = {
	projectLabel: string;
	analysisId: string;
	sourcePointCount: number;
	sourceGridSizeMeters: number;
	pointCount: number;
	gridSizeMeters: number;
	modes: Record<ColorMode, Record<CollectionPhase, CollectedSample>>;
	assertions: {
		pythonBinDebugComparisonFieldsAbsent: true;
		forbiddenComparisonFieldsPresent: string[];
		forbiddenRequestUrls: string[];
		memoryScope: string;
	};
};

const CASES: AnalysisCase[] = [
	{
		projectLabel: 'Ben-Gurion',
		analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
		metadataPath: 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday.json'
	},
	{
		projectLabel: 'Ness-Tziona',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		metadataPath: 'data/analyses/Ness-Tziona/exploded/nes_tziona_unblock_2.json'
	}
];

function expectedSelectionKey(caseConfig: AnalysisCase, hourIndex: number) {
	return `${caseConfig.analysisId}|7|${hourIndex}`;
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
	params: {
		expectedSelectionKey: string;
		colorMode: ColorMode;
		minSurfaceRequestId?: number;
	}
) {
	const diagnostics = await page.waitForFunction(
		({ selectionKey, gridResolution, colorMode, minSurfaceRequestId }) => {
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
				value.baseColorMode === colorMode &&
				(typeof value.baseSurfaceRequestId !== 'number' ||
					value.baseSurfaceRequestId > (minSurfaceRequestId ?? 0)) &&
				value.gpuResidentCopyRequestId === value.baseSurfaceRequestId &&
				value.selectedHourRuntimeContract?.route === 'main' &&
				value.selectedHourRuntimeContract?.readbackInstrumentation === 'instrumented' &&
				value.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount === 0 &&
				value.selectedHourRuntimeContract?.strongVisibleGpuPath === true &&
				value.baseMetadataGridSize === gridResolution &&
				typeof value.basePointCount === 'number' &&
				value.basePointCount > 0
			) {
				return value;
			}
			return null;
		},
		{
			selectionKey: params.expectedSelectionKey,
			gridResolution: TARGET_GRID_RESOLUTION_METERS,
			colorMode: params.colorMode,
			minSurfaceRequestId: params.minSurfaceRequestId ?? 0
		},
		{ timeout: 240_000 }
	).catch(async (error) => {
		const lastDiagnostics = await readUtciRenderDiagnostics(page).catch((readError) => ({
			readError: readError instanceof Error ? readError.message : String(readError)
		}));
		const message = error instanceof Error ? error.message : String(error);
		throw new Error(
			[
				'Timed out waiting for strong main-route 0.5m selected-hour diagnostics.',
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

function expectTimingFields(diagnostics: Record<string, any>) {
	for (const key of [
		'payloadPrepareMs',
		'workerBvhMs',
		'pipelineUploadMs',
		'exposurePrecomputeMs',
		'oneHourDispatchMs',
		'firstSelectedHourReadyMs'
	]) {
		expect(diagnostics.timings?.[key], key).toEqual(expect.any(Number));
		expect(diagnostics.timings[key], key).toBeGreaterThanOrEqual(0);
	}
}

function buildSample(params: {
	diagnostics: Record<string, any>;
	phase: CollectionPhase;
	colorMode: ColorMode;
	collectionMethod: string;
	sourceUrl: string;
}): CollectedSample {
	const { diagnostics, phase, colorMode, collectionMethod, sourceUrl } = params;
	const trackedGpuAllocationBytes = diagnostics.trackedGpuAllocationBytes;
	return {
		phase,
		colorMode,
		collectionMethod,
		selectedMonthIndex: diagnostics.baseSelectedMonthIndex,
		selectedHourIndex: diagnostics.baseSelectedHourIndex,
		selectedTimeIndex: diagnostics.baseSelectedTimeIndex,
		selectionKey: diagnostics.baseSelectionKey ?? null,
		surfaceRequestId: diagnostics.baseSurfaceRequestId ?? null,
		collectedAt: new Date().toISOString(),
		sourceRoute: SOURCE_ROUTE,
		sourceUrl,
		timings: extractTimings(diagnostics.timings),
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
				diagnostics.baseSameDeviceForComputeAndRender ?? null
		}
	};
}

async function collectMode(page: Page, caseConfig: AnalysisCase, colorMode: ColorMode) {
	const sourceUrl = `/?analysis=${encodeURIComponent(caseConfig.analysisId)}&gridResolution=${TARGET_GRID_RESOLUTION_METERS}&utciRender=auto&utciRenderDiagnostics=1`;
	await page.goto(sourceUrl);

	if (colorMode === 'discrete') {
		await page.getByRole('button', { name: 'Per hour' }).click();
	}

	const initialDiagnostics = await waitForSelectedHourPublication(page, {
		expectedSelectionKey: expectedSelectionKey(caseConfig, 0),
		colorMode
	});
	expectTimingFields(initialDiagnostics);
	expect(initialDiagnostics.baseSelectedMonthIndex).toBe(7);
	expect(initialDiagnostics.baseSelectedHourIndex).toBe(0);
	expect(initialDiagnostics.baseSelectedTimeIndex).toBe(7 * 24);

	const initialRequestId = initialDiagnostics.baseSurfaceRequestId ?? 0;
	await page.getByRole('slider', { name: 'Select analysis hour' }).press('ArrowRight');
	const scrubDiagnostics = await waitForSelectedHourPublication(page, {
		expectedSelectionKey: expectedSelectionKey(caseConfig, SCRUB_HOUR_INDEX),
		colorMode,
		minSurfaceRequestId: initialRequestId
	});
	expectTimingFields(scrubDiagnostics);
	expect(scrubDiagnostics.baseSelectedMonthIndex).toBe(7);
	expect(scrubDiagnostics.baseSelectedHourIndex).toBe(SCRUB_HOUR_INDEX);
	expect(scrubDiagnostics.baseSelectedTimeIndex).toBe(7 * 24 + SCRUB_HOUR_INDEX);

	await page.goto('about:blank');

	return {
		initialDiagnostics,
		scrubDiagnostics,
		samples: {
			initial: buildSample({
				diagnostics: initialDiagnostics,
				phase: 'initial',
				colorMode,
				collectionMethod:
					colorMode === 'normalized'
						? 'Initial main-route load at / with default Full day color mode.'
						: 'Initial collected discrete sample after app-visible Color scale mode -> Per hour button interaction; no URL color-mode parameter exists.',
				sourceUrl
			}),
			scrub: buildSample({
				diagnostics: scrubDiagnostics,
				phase: 'scrub',
				colorMode,
				collectionMethod:
					'App-visible keyboard scrub on the main-route Select analysis hour slider from hour 0 to hour 1; no debug route, parity, or .bin path.',
				sourceUrl
			})
		}
	};
}

async function collectCase(
	page: Page,
	caseConfig: AnalysisCase
): Promise<CollectedCase> {
	const metadata = readMetadata(caseConfig);
	const requestedUrls: string[] = [];
	page.on('request', (request) => requestedUrls.push(request.url()));

	const normalized = await collectMode(page, caseConfig, 'normalized');
	const discrete = await collectMode(page, caseConfig, 'discrete');
	const forbiddenComparisonFieldsPresent = [
		...collectForbiddenComparisonFields(normalized.initialDiagnostics),
		...collectForbiddenComparisonFields(normalized.scrubDiagnostics),
		...collectForbiddenComparisonFields(discrete.initialDiagnostics),
		...collectForbiddenComparisonFields(discrete.scrubDiagnostics)
	];
	const forbiddenRequestUrls = requestedUrls.filter(isForbiddenComparisonRequest);
	const representative = discrete.scrubDiagnostics;

	expect(forbiddenComparisonFieldsPresent).toEqual([]);
	expect(forbiddenRequestUrls).toEqual([]);
	expect(representative.baseMetadataGridSize).toBe(TARGET_GRID_RESOLUTION_METERS);
	expect(representative.basePointCount).toBeGreaterThan(metadata.num_positions ?? 0);
	expect(representative.trackedGpuAllocationBytes).toMatchObject({
		persistentExposureBytes: expect.any(Number),
		allHoursOutputBytes: 0,
		selectedHourOutputBytes: expect.any(Number),
		selectedHourOutputBytesHighWatermark: expect.any(Number),
		renderOwnedSelectedHourBytes: expect.any(Number),
		renderOwnedSelectedHourBytesHighWatermark: expect.any(Number),
		trackingScope: 'utci-owned-webgpu-buffers'
	});
	expect(representative.trackedGpuAllocationBytes.persistentExposureBytes).toBeGreaterThan(0);
	expect(representative.trackedGpuAllocationBytes.renderOwnedSelectedHourBytes).toBeGreaterThan(0);

	return {
		projectLabel: caseConfig.projectLabel,
		analysisId: caseConfig.analysisId,
		sourcePointCount: metadata.num_positions ?? 0,
		sourceGridSizeMeters: metadata.grid_size ?? 0,
		pointCount: representative.basePointCount ?? 0,
		gridSizeMeters: representative.baseMetadataGridSize ?? TARGET_GRID_RESOLUTION_METERS,
		modes: {
			normalized: normalized.samples,
			discrete: discrete.samples
		},
		assertions: {
			pythonBinDebugComparisonFieldsAbsent: true,
			forbiddenComparisonFieldsPresent,
			forbiddenRequestUrls,
			memoryScope: representative.trackedGpuAllocationBytes.trackingScope
		}
	};
}

test.describe('main route 0.5m performance collector', () => {
	test.afterEach(async ({ page }) => {
		await page.goto('about:blank').catch(() => undefined);
	});

	test('collects BG and Ness Tziona 0.5m main-route timing, memory, color-mode, and scrub samples', async ({
		page
	}, testInfo) => {
		test.setTimeout(600_000);

		const cases: CollectedCase[] = [];
		for (const caseConfig of CASES) {
			cases.push(await collectCase(page, caseConfig));
		}

		const artifact = {
			collectedOn: COLLECTED_ON,
			sourceRoute: SOURCE_ROUTE,
			targetGridResolutionMeters: TARGET_GRID_RESOLUTION_METERS,
			includedAnalyses: CASES.map((entry) => entry.analysisId),
			excludedBgVariantsExplanation:
				'This 0.5m stress pass intentionally excludes other Ben-Gurion variants and uses only the BG base case plus Ness Tziona base/exploded model.',
			collectionMethod:
				'Main route only: / with gridResolution=0.5&utciRenderDiagnostics=1, app-visible color-mode buttons and hour slider scrub, no debug route and no parity/.bin comparison.',
			cases
		};

		if (!existsSync(RESULTS_DIR)) {
			mkdirSync(RESULTS_DIR, { recursive: true });
		}

		const json = JSON.stringify(artifact, null, 2);
		writeFileSync(ARTIFACT_PATH, json, 'utf8');
		await testInfo.attach('main-route-selected-hour-0_5m-base.json', {
			body: json,
			contentType: 'application/json'
		});
	});
});
