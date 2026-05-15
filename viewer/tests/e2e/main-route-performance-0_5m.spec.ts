import { expect, test, type Page } from '@playwright/test';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const RESULTS_DIR = resolve(REPO_ROOT, 'data/performance-results');
const ARTIFACT_FILENAME = 'main-route-selected-hour-render-diagnostics-next.json';
const ARTIFACT_PATH = resolve(RESULTS_DIR, ARTIFACT_FILENAME);
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

type RenderPublicationTiming = {
	renderPublicationVersion: number | null;
	renderPublicationPath:
		| 'compute-buffer-selected-hour'
		| 'cpu-uploaded-selected-hour'
		| 'none'
		| null;
	renderPublicationPhase: 'initial' | 'scrub' | 'unknown' | null;
	renderPublicationMeshAction: 'created' | 'reused' | 'skipped' | null;
	renderPublicationPointCount: number | null;
	renderPublicationVertexCount: number | null;
	renderPublicationGridWidth: number | null;
	renderPublicationGridHeight: number | null;
	renderPublicationGridSize: number | null;
	renderPublicationSourceByteLength: number | null;
	renderPublicationTargetByteLength: number | null;
	renderPublicationRenderOwnedBytes: number | null;
	renderPublicationTimeline: RenderPublicationTimelineTiming | null;
};

type RenderPublicationTimelineTiming = {
	computeCompletedAtMs: number | null;
	controllerAcceptedAtMs: number | null;
	routePublishedAtMs: number | null;
	routeProjectedAtMs: number | null;
	sceneSurfaceReceivedAtMs: number | null;
	publicationEffectStartedAtMs: number | null;
	renderStorageReadyAtMs: number | null;
	sceneSyncCompletedAtMs: number | null;
};

type ExtractedTimings = {
	payloadPrepareMs: number | null;
	workerBvhMs: number | null;
	pipelineUploadMs: number | null;
	exposurePrecomputeMs: number | null;
	oneHourDispatchMs: number | null;
	firstSelectedHourReadyMs: number | null;
	firstSelectedHourVisibleMs: number | null;
	renderUpdateMs: number | null;
	renderSceneSyncStartDelayMs: number | null;
	renderSceneSyncTotalMs: number | null;
	renderLayoutBuildMs: number | null;
	renderSurfaceMeshMs: number | null;
	renderStorageInitWaitMs: number | null;
	renderBufferCopyMs: number | null;
	renderQueueDrainMs: number | null;
	renderPublication: RenderPublicationTiming | null;
};

type DiagnosticsTimingsInput = {
	payloadPrepareMs?: unknown;
	workerBvhMs?: unknown;
	pipelineUploadMs?: unknown;
	exposurePrecomputeMs?: unknown;
	oneHourDispatchMs?: unknown;
	firstSelectedHourReadyMs?: unknown;
	firstSelectedHourVisibleMs?: unknown;
	renderUpdateMs?: unknown;
	renderSceneSyncStartDelayMs?: unknown;
	renderSceneSyncTotalMs?: unknown;
	renderLayoutBuildMs?: unknown;
	renderSurfaceMeshMs?: unknown;
	renderStorageInitWaitMs?: unknown;
	renderBufferCopyMs?: unknown;
	renderQueueDrainMs?: unknown;
	renderPublication?: unknown;
};

type SelectedHourRuntimeContractDiagnostics = {
	route?: unknown;
	readbackInstrumentation?: unknown;
	visibleSelectedHourReadbackCount?: unknown;
	strongVisibleGpuPath?: unknown;
};

type TrackedGpuAllocationBytesDiagnostics = {
	persistentExposureBytes: number;
	allHoursOutputBytes: number;
	selectedHourOutputBytes: number;
	selectedHourOutputBytesHighWatermark: number;
	renderOwnedSelectedHourBytes: number;
	renderOwnedSelectedHourBytesHighWatermark: number;
	trackingScope: string;
};

type DiagnosticsSnapshot = {
	timings?: DiagnosticsTimingsInput;
	trackedGpuAllocationBytes: TrackedGpuAllocationBytesDiagnostics;
	baseSelectedMonthIndex: number;
	baseSelectedHourIndex: number;
	baseSelectedTimeIndex: number;
	baseSelectionKey?: string | null;
	baseSurfaceRequestId?: number | null;
	rendererBackend: string;
	utciRenderResolved: string;
	utciSurfaceSource?: string | null;
	baseRenderTransport: string;
	dataTextureBuildCount?: number | null;
	selectedHourRuntimeContract?: SelectedHourRuntimeContractDiagnostics;
	baseSameDeviceForComputeAndRender?: boolean | null;
	baseMetadataGridSize?: number | null;
	basePointCount?: number | null;
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
	timings: ExtractedTimings;
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

function stringOrNull(value: unknown): string | null {
	return typeof value === 'string' ? value : null;
}

function booleanOrNull(value: unknown): boolean | null {
	return typeof value === 'boolean' ? value : null;
}

function stringFromSetOrNull<const T extends string>(
	value: unknown,
	allowed: readonly T[]
): T | null {
	return typeof value === 'string' && allowed.includes(value as T) ? (value as T) : null;
}

function extractRenderPublication(
	value: unknown
): RenderPublicationTiming | null {
	if (typeof value !== 'object' || value == null) {
		return null;
	}

	const payload = value as Record<string, unknown>;
	return {
		renderPublicationVersion: numberOrNull(payload.renderPublicationVersion),
		renderPublicationPath: stringFromSetOrNull(payload.renderPublicationPath, [
			'compute-buffer-selected-hour',
			'cpu-uploaded-selected-hour',
			'none'
		]),
		renderPublicationPhase: stringFromSetOrNull(payload.renderPublicationPhase, [
			'initial',
			'scrub',
			'unknown'
		]),
		renderPublicationMeshAction: stringFromSetOrNull(payload.renderPublicationMeshAction, [
			'created',
			'reused',
			'skipped'
		]),
		renderPublicationPointCount: numberOrNull(payload.renderPublicationPointCount),
		renderPublicationVertexCount: numberOrNull(payload.renderPublicationVertexCount),
		renderPublicationGridWidth: numberOrNull(payload.renderPublicationGridWidth),
		renderPublicationGridHeight: numberOrNull(payload.renderPublicationGridHeight),
		renderPublicationGridSize: numberOrNull(payload.renderPublicationGridSize),
		renderPublicationSourceByteLength: numberOrNull(
			payload.renderPublicationSourceByteLength
		),
		renderPublicationTargetByteLength: numberOrNull(
			payload.renderPublicationTargetByteLength
		),
		renderPublicationRenderOwnedBytes: numberOrNull(
			payload.renderPublicationRenderOwnedBytes
		),
		renderPublicationTimeline: extractRenderPublicationTimeline(
			payload.renderPublicationTimeline
		)
	};
}

function extractRenderPublicationTimeline(
	value: unknown
): RenderPublicationTimelineTiming | null {
	if (typeof value !== 'object' || value == null) {
		return null;
	}

	const payload = value as Record<string, unknown>;
	return {
		computeCompletedAtMs: numberOrNull(payload.computeCompletedAtMs),
		controllerAcceptedAtMs: numberOrNull(payload.controllerAcceptedAtMs),
		routePublishedAtMs: numberOrNull(payload.routePublishedAtMs),
		routeProjectedAtMs: numberOrNull(payload.routeProjectedAtMs),
		sceneSurfaceReceivedAtMs: numberOrNull(payload.sceneSurfaceReceivedAtMs),
		publicationEffectStartedAtMs: numberOrNull(payload.publicationEffectStartedAtMs),
		renderStorageReadyAtMs: numberOrNull(payload.renderStorageReadyAtMs),
		sceneSyncCompletedAtMs: numberOrNull(payload.sceneSyncCompletedAtMs)
	};
}

function extractTimings(timings: DiagnosticsTimingsInput | undefined) {
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
		renderQueueDrainMs: numberOrNull(timings?.renderQueueDrainMs),
		renderPublication: extractRenderPublication(timings?.renderPublication)
	};
}

function expectValidRenderPublication(
	sample: CollectedSample,
	label: string
) {
	const renderPublication = sample.timings.renderPublication;
	expect(renderPublication, `${label} renderPublication`).not.toBeNull();
	expect(renderPublication).toMatchObject({
		renderPublicationVersion: 1,
		renderPublicationPath: 'compute-buffer-selected-hour',
		renderPublicationPhase: sample.phase
	});
	expect(
		renderPublication?.renderPublicationMeshAction,
		`${label} renderPublicationMeshAction`
	).toMatch(/^(created|reused|skipped)$/);
	expect(
		renderPublication?.renderPublicationGridWidth,
		`${label} renderPublicationGridWidth`
	).toEqual(expect.any(Number));
	expect(
		renderPublication?.renderPublicationGridHeight,
		`${label} renderPublicationGridHeight`
	).toEqual(expect.any(Number));
	expect(
		renderPublication?.renderPublicationGridSize,
		`${label} renderPublicationGridSize`
	).toBe(TARGET_GRID_RESOLUTION_METERS);
	expect(
		renderPublication?.renderPublicationPointCount,
		`${label} renderPublicationPointCount`
	).toEqual(expect.any(Number));
	expect(
		renderPublication?.renderPublicationVertexCount,
		`${label} renderPublicationVertexCount`
	).toEqual(expect.any(Number));
	expect(
		renderPublication?.renderPublicationSourceByteLength,
		`${label} renderPublicationSourceByteLength`
	).toEqual(expect.any(Number));
	expect(
		renderPublication?.renderPublicationTargetByteLength,
		`${label} renderPublicationTargetByteLength`
	).toEqual(expect.any(Number));
	expect(
		renderPublication?.renderPublicationRenderOwnedBytes,
		`${label} renderPublicationRenderOwnedBytes`
	).toEqual(expect.any(Number));
	expect(
		renderPublication?.renderPublicationGridWidth ?? 0,
		`${label} renderPublicationGridWidth`
	).toBeGreaterThan(0);
	expect(
		renderPublication?.renderPublicationGridHeight ?? 0,
		`${label} renderPublicationGridHeight`
	).toBeGreaterThan(0);
	expect(
		renderPublication?.renderPublicationPointCount ?? 0,
		`${label} renderPublicationPointCount`
	).toBeGreaterThan(0);
	expect(
		renderPublication?.renderPublicationVertexCount ?? 0,
		`${label} renderPublicationVertexCount`
	).toBeGreaterThan(0);
	expect(
		renderPublication?.renderPublicationSourceByteLength ?? 0,
		`${label} renderPublicationSourceByteLength`
	).toBeGreaterThan(0);
	expect(
		renderPublication?.renderPublicationTargetByteLength ?? 0,
		`${label} renderPublicationTargetByteLength`
	).toBeGreaterThan(0);
	expect(
		renderPublication?.renderPublicationRenderOwnedBytes ?? 0,
		`${label} renderPublicationRenderOwnedBytes`
	).toBeGreaterThan(0);
	expectValidRenderPublicationTimeline(renderPublication, label);
}

function expectTimelineOrder(
	timeline: RenderPublicationTimelineTiming,
	keys: readonly (keyof RenderPublicationTimelineTiming)[],
	label: string
) {
	let previousValue: number | undefined;
	for (const key of keys) {
		const value = timeline[key];
		expect(value, `${label} ${key}`).toEqual(expect.any(Number));
		expect(Number.isFinite(value), `${label} ${key} should be finite`).toBe(true);
		if (previousValue !== undefined) {
			expect(value, `${label} ${key} should be ordered`).toBeGreaterThanOrEqual(
				previousValue
			);
		}
		previousValue = value ?? undefined;
	}
}

function expectValidRenderPublicationTimeline(
	renderPublication: RenderPublicationTiming | null,
	label: string
) {
	const timeline = renderPublication?.renderPublicationTimeline;
	expect(timeline, `${label} renderPublicationTimeline`).not.toBeNull();
	if (!timeline) {
		throw new Error(`${label} missing renderPublicationTimeline`);
	}

	expectTimelineOrder(
		timeline,
		[
			'computeCompletedAtMs',
			'controllerAcceptedAtMs',
			'routePublishedAtMs',
			'routeProjectedAtMs'
		],
		label
	);
	expectTimelineOrder(
		timeline,
		[
			'controllerAcceptedAtMs',
			'sceneSurfaceReceivedAtMs',
			'publicationEffectStartedAtMs',
			'renderStorageReadyAtMs',
			'sceneSyncCompletedAtMs'
		],
		label
	);
}

function expectRenderPublicationForAllSamples(collectedCase: CollectedCase) {
	for (const colorMode of ['normalized', 'discrete'] as const) {
		for (const phase of ['initial', 'scrub'] as const) {
			expectValidRenderPublication(
				collectedCase.modes[colorMode][phase],
				`${collectedCase.projectLabel} ${colorMode}.${phase}`
			);
		}
	}
}

function expectTimingFields(diagnostics: DiagnosticsSnapshot) {
	const timings = diagnostics.timings;
	expect(timings, 'timings').toBeDefined();
	if (!timings) {
		throw new Error('Expected diagnostics.timings to be present');
	}

	for (const key of [
		'payloadPrepareMs',
		'workerBvhMs',
		'pipelineUploadMs',
		'exposurePrecomputeMs',
		'oneHourDispatchMs',
		'firstSelectedHourReadyMs'
	] as const satisfies readonly (keyof DiagnosticsTimingsInput)[]) {
		expect(timings[key], key).toEqual(expect.any(Number));
		expect(numberOrNull(timings[key]), key).toBeGreaterThanOrEqual(0);
	}
}

function buildSample(params: {
	diagnostics: DiagnosticsSnapshot;
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
				route: stringOrNull(diagnostics.selectedHourRuntimeContract?.route),
				readbackInstrumentation: stringOrNull(
					diagnostics.selectedHourRuntimeContract?.readbackInstrumentation
				),
				visibleSelectedHourReadbackCount:
					numberOrNull(diagnostics.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount),
				strongVisibleGpuPath: booleanOrNull(
					diagnostics.selectedHourRuntimeContract?.strongVisibleGpuPath
				)
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
	const collectedCase: CollectedCase = {
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

	expectRenderPublicationForAllSamples(collectedCase);

	return collectedCase;
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
		await testInfo.attach(ARTIFACT_FILENAME, {
			body: json,
			contentType: 'application/json'
		});
	});
});
