import { expect, test, type Page } from '@playwright/test';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const RESULTS_DIR = resolve(REPO_ROOT, 'data/performance-results');
const ARTIFACT_FILENAME = 'main-route-selected-hour-render-diagnostics-next.json';
const ARTIFACT_PATH = resolve(RESULTS_DIR, ARTIFACT_FILENAME);
const RESET_PROOF_ARTIFACT_FILENAME =
	'main-route-selected-hour-render-reset-diagnostics-next.json';
const RESET_PROOF_ARTIFACT_PATH = resolve(
	RESULTS_DIR,
	RESET_PROOF_ARTIFACT_FILENAME
);
const COLLECTED_ON = '2026-05-16';
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
	routePendingSurfaceExposedAtMs: number | null;
	routePublishedAtMs: number | null;
	routeProjectedAtMs: number | null;
	scenePendingSurfaceObservedAtMs: number | null;
	sceneSyncAttemptStartedAtMs: number | null;
	sceneSyncAttemptToken: number | null;
	sceneSurfaceReceivedAtMs: number | null;
	publicationEffectStartedAtMs: number | null;
	renderSurfaceMeshTrace: RenderSurfaceMeshTrace | null;
	renderStorageReadyAtMs: number | null;
	renderStorageWaitTrace: RenderStorageWaitTrace | null;
	sceneSyncCompletedAtMs: number | null;
	sceneSyncResetHistory: RenderPublicationSceneSyncResetEvent[];
	sceneSyncActiveWindowResetHistory: RenderPublicationSceneSyncResetEvent[] | null;
};

type RenderSurfaceMeshTrace = {
	action: 'created' | 'updated' | 'update-failed-created' | null;
	totalMs: number | null;
	recreateDecision: RenderSurfaceMeshRecreateDecision | null;
	disposeResetMeshRemovalMs: number | null;
	createComputeBufferSurfaceMeshMs: number | null;
	updateComputeBufferSurfaceMeshMs: number | null;
	fallbackDecisionMs: number | null;
	applySurfaceMeshStateMs: number | null;
	setCreatedSurfacePendingStorageInitMs: number | null;
	setPostSurfacePendingStorageInitMs: number | null;
	sceneAddMs: number | null;
	publishUtciSurfaceDiagnosticsMs: number | null;
};

type RenderSurfaceMeshRecreateDecision = {
	missingSurface: boolean | null;
	notComputeBufferSurface: boolean | null;
	analysisIdentityChanged: boolean | null;
	layoutCompatible: boolean | null;
};

type RenderStorageWaitReadState = {
	deviceAvailable: boolean | null;
	backendEntryAvailable: boolean | null;
	bufferAvailable: boolean | null;
};

type RenderStorageWaitSample = RenderStorageWaitReadState & {
	atMs: number | null;
};

type RenderStorageWaitTrace = {
	waitStartedAtMs: number | null;
	waitFinishedAtMs: number | null;
	waitMs: number | null;
	readAttemptCount: number | null;
	frameWaitCount: number | null;
	deviceAvailableCount: number | null;
	backendEntryAvailableCount: number | null;
	bufferAvailableCount: number | null;
	firstDeviceAtMs: number | null;
	firstBackendEntryAtMs: number | null;
	firstBufferAtMs: number | null;
	lastReadState: RenderStorageWaitReadState | null;
	samples: RenderStorageWaitSample[];
};

type RenderPublicationSceneSyncResetEvent = {
	resetAtMs: number | null;
	resetReason: string | null;
	invalidateActiveRun: boolean | null;
	previousCopyRunToken: number | null;
	nextCopyRunToken: number | null;
	previousSyncRunKey: string | null;
};

type RenderPublicationTimelineNumberKey = Exclude<
	keyof RenderPublicationTimelineTiming,
	| 'renderStorageWaitTrace'
	| 'renderSurfaceMeshTrace'
	| 'sceneSyncResetHistory'
	| 'sceneSyncActiveWindowResetHistory'
>;

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
		routePendingSurfaceExposedAtMs: numberOrNull(
			payload.routePendingSurfaceExposedAtMs
		),
		routePublishedAtMs: numberOrNull(payload.routePublishedAtMs),
		routeProjectedAtMs: numberOrNull(payload.routeProjectedAtMs),
		scenePendingSurfaceObservedAtMs: numberOrNull(
			payload.scenePendingSurfaceObservedAtMs
		),
		sceneSyncAttemptStartedAtMs: numberOrNull(payload.sceneSyncAttemptStartedAtMs),
		sceneSyncAttemptToken: numberOrNull(payload.sceneSyncAttemptToken),
		sceneSurfaceReceivedAtMs: numberOrNull(payload.sceneSurfaceReceivedAtMs),
		publicationEffectStartedAtMs: numberOrNull(payload.publicationEffectStartedAtMs),
		renderSurfaceMeshTrace: extractRenderSurfaceMeshTrace(
			payload.renderSurfaceMeshTrace
		),
		renderStorageReadyAtMs: numberOrNull(payload.renderStorageReadyAtMs),
		renderStorageWaitTrace: extractRenderStorageWaitTrace(
			payload.renderStorageWaitTrace
		),
		sceneSyncCompletedAtMs: numberOrNull(payload.sceneSyncCompletedAtMs),
		sceneSyncResetHistory: extractSceneSyncResetHistory(
			payload.sceneSyncResetHistory
		),
		sceneSyncActiveWindowResetHistory: extractPresentSceneSyncResetHistory(
			payload.sceneSyncActiveWindowResetHistory
		)
	};
}

function extractRenderSurfaceMeshTrace(value: unknown): RenderSurfaceMeshTrace | null {
	if (typeof value !== 'object' || value == null) {
		return null;
	}

	const payload = value as Record<string, unknown>;
	return {
		action: stringFromSetOrNull(payload.action, [
			'created',
			'updated',
			'update-failed-created'
		]),
		totalMs: numberOrNull(payload.totalMs),
		recreateDecision: extractRenderSurfaceMeshRecreateDecision(
			payload.recreateDecision
		),
		disposeResetMeshRemovalMs: numberOrNull(payload.disposeResetMeshRemovalMs),
		createComputeBufferSurfaceMeshMs: numberOrNull(
			payload.createComputeBufferSurfaceMeshMs
		),
		updateComputeBufferSurfaceMeshMs: numberOrNull(
			payload.updateComputeBufferSurfaceMeshMs
		),
		fallbackDecisionMs: numberOrNull(payload.fallbackDecisionMs),
		applySurfaceMeshStateMs: numberOrNull(payload.applySurfaceMeshStateMs),
		setCreatedSurfacePendingStorageInitMs: numberOrNull(
			payload.setCreatedSurfacePendingStorageInitMs
		),
		setPostSurfacePendingStorageInitMs: numberOrNull(
			payload.setPostSurfacePendingStorageInitMs
		),
		sceneAddMs: numberOrNull(payload.sceneAddMs),
		publishUtciSurfaceDiagnosticsMs: numberOrNull(
			payload.publishUtciSurfaceDiagnosticsMs
		)
	};
}

function extractRenderSurfaceMeshRecreateDecision(
	value: unknown
): RenderSurfaceMeshRecreateDecision | null {
	if (typeof value !== 'object' || value == null) {
		return null;
	}

	const payload = value as Record<string, unknown>;
	return {
		missingSurface: booleanOrNull(payload.missingSurface),
		notComputeBufferSurface: booleanOrNull(payload.notComputeBufferSurface),
		analysisIdentityChanged: booleanOrNull(payload.analysisIdentityChanged),
		layoutCompatible: booleanOrNull(payload.layoutCompatible)
	};
}

function extractRenderStorageWaitTrace(
	value: unknown
): RenderStorageWaitTrace | null {
	if (typeof value !== 'object' || value == null) {
		return null;
	}

	const payload = value as Record<string, unknown>;
	return {
		waitStartedAtMs: numberOrNull(payload.waitStartedAtMs),
		waitFinishedAtMs: numberOrNull(payload.waitFinishedAtMs),
		waitMs: numberOrNull(payload.waitMs),
		readAttemptCount: numberOrNull(payload.readAttemptCount),
		frameWaitCount: numberOrNull(payload.frameWaitCount),
		deviceAvailableCount: numberOrNull(payload.deviceAvailableCount),
		backendEntryAvailableCount: numberOrNull(payload.backendEntryAvailableCount),
		bufferAvailableCount: numberOrNull(payload.bufferAvailableCount),
		firstDeviceAtMs: numberOrNull(payload.firstDeviceAtMs),
		firstBackendEntryAtMs: numberOrNull(payload.firstBackendEntryAtMs),
		firstBufferAtMs: numberOrNull(payload.firstBufferAtMs),
		lastReadState: extractRenderStorageWaitReadState(payload.lastReadState),
		samples: extractRenderStorageWaitSamples(payload.samples)
	};
}

function extractRenderStorageWaitReadState(
	value: unknown
): RenderStorageWaitReadState | null {
	if (typeof value !== 'object' || value == null) {
		return null;
	}
	const payload = value as Record<string, unknown>;
	return {
		deviceAvailable: booleanOrNull(payload.deviceAvailable),
		backendEntryAvailable: booleanOrNull(payload.backendEntryAvailable),
		bufferAvailable: booleanOrNull(payload.bufferAvailable)
	};
}

function extractRenderStorageWaitSamples(value: unknown): RenderStorageWaitSample[] {
	if (!Array.isArray(value)) {
		return [];
	}
	return value
		.filter((entry): entry is Record<string, unknown> => {
			return typeof entry === 'object' && entry != null;
		})
		.map((entry) => ({
			atMs: numberOrNull(entry.atMs),
			deviceAvailable: booleanOrNull(entry.deviceAvailable),
			backendEntryAvailable: booleanOrNull(entry.backendEntryAvailable),
			bufferAvailable: booleanOrNull(entry.bufferAvailable)
		}));
}

function extractPresentSceneSyncResetHistory(
	value: unknown
): RenderPublicationSceneSyncResetEvent[] | null {
	if (!Array.isArray(value)) {
		return null;
	}
	return extractSceneSyncResetHistory(value);
}

function extractSceneSyncResetHistory(
	value: unknown
): RenderPublicationSceneSyncResetEvent[] {
	if (!Array.isArray(value)) {
		return [];
	}
	return value
		.filter((entry): entry is Record<string, unknown> => {
			return typeof entry === 'object' && entry != null;
		})
		.map((entry) => ({
			resetAtMs: numberOrNull(entry.resetAtMs),
			resetReason: stringOrNull(entry.resetReason),
			invalidateActiveRun: booleanOrNull(entry.invalidateActiveRun),
			previousCopyRunToken: numberOrNull(entry.previousCopyRunToken),
			nextCopyRunToken: numberOrNull(entry.nextCopyRunToken),
			previousSyncRunKey: stringOrNull(entry.previousSyncRunKey)
		}));
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
	keys: readonly RenderPublicationTimelineNumberKey[],
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

function requireTimelineNumber(
	timeline: RenderPublicationTimelineTiming,
	key: RenderPublicationTimelineNumberKey,
	label: string
): number {
	const value = timeline[key];
	expect(value, `${label} ${key}`).toEqual(expect.any(Number));
	expect(Number.isFinite(value), `${label} ${key} should be finite`).toBe(true);
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		throw new Error(`${label} ${key} should be a finite number`);
	}
	return value;
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
		['computeCompletedAtMs', 'controllerAcceptedAtMs'],
		label
	);
	const controllerAcceptedAtMs = requireTimelineNumber(
		timeline,
		'controllerAcceptedAtMs',
		label
	);
	const routePendingSurfaceExposedAtMs = requireTimelineNumber(
		timeline,
		'routePendingSurfaceExposedAtMs',
		label
	);
	const routePublishedAtMs = requireTimelineNumber(
		timeline,
		'routePublishedAtMs',
		label
	);
	const routeProjectedAtMs = requireTimelineNumber(
		timeline,
		'routeProjectedAtMs',
		label
	);
	const sceneSurfaceReceivedAtMs = requireTimelineNumber(
		timeline,
		'sceneSurfaceReceivedAtMs',
		label
	);
	const scenePendingSurfaceObservedAtMs = timeline.scenePendingSurfaceObservedAtMs;
	const sceneSyncAttemptStartedAtMs = timeline.sceneSyncAttemptStartedAtMs;
	const sceneSyncAttemptToken = requireTimelineNumber(
		timeline,
		'sceneSyncAttemptToken',
		label
	);
	expect(
		routePendingSurfaceExposedAtMs,
		`${label} routePendingSurfaceExposedAtMs should not precede controllerAcceptedAtMs`
	).toBeGreaterThanOrEqual(controllerAcceptedAtMs);
	expect(
		routePublishedAtMs,
		`${label} routePublishedAtMs should not precede controllerAcceptedAtMs`
	).toBeGreaterThanOrEqual(controllerAcceptedAtMs);
	expect(
		routeProjectedAtMs,
		`${label} routeProjectedAtMs should not precede controllerAcceptedAtMs`
	).toBeGreaterThanOrEqual(controllerAcceptedAtMs);
	expect(
		sceneSurfaceReceivedAtMs,
		`${label} sceneSurfaceReceivedAtMs should not precede controllerAcceptedAtMs`
	).toBeGreaterThanOrEqual(controllerAcceptedAtMs);
	if (scenePendingSurfaceObservedAtMs != null) {
		expect(
			scenePendingSurfaceObservedAtMs,
			`${label} scenePendingSurfaceObservedAtMs should be finite when present`
		).toEqual(expect.any(Number));
		expect(
			scenePendingSurfaceObservedAtMs,
			`${label} scenePendingSurfaceObservedAtMs should not precede controllerAcceptedAtMs`
		).toBeGreaterThanOrEqual(controllerAcceptedAtMs);
		expect(
			sceneSurfaceReceivedAtMs,
			`${label} sceneSurfaceReceivedAtMs should not precede scenePendingSurfaceObservedAtMs`
		).toBeGreaterThanOrEqual(scenePendingSurfaceObservedAtMs);
	}
	expect(sceneSyncAttemptToken, `${label} sceneSyncAttemptToken`).toBeGreaterThan(0);
	if (sceneSyncAttemptStartedAtMs != null) {
		expect(
			sceneSyncAttemptStartedAtMs,
			`${label} sceneSyncAttemptStartedAtMs should be finite when present`
		).toEqual(expect.any(Number));
		if (scenePendingSurfaceObservedAtMs != null) {
			expect(
				sceneSyncAttemptStartedAtMs,
				`${label} sceneSyncAttemptStartedAtMs should not precede scenePendingSurfaceObservedAtMs`
			).toBeGreaterThanOrEqual(scenePendingSurfaceObservedAtMs);
		}
		expect(
			sceneSyncAttemptStartedAtMs,
			`${label} sceneSyncAttemptStartedAtMs should not precede sceneSurfaceReceivedAtMs`
		).toBeGreaterThanOrEqual(sceneSurfaceReceivedAtMs);
	}
	expectTimelineOrder(
		timeline,
		[
			'sceneSurfaceReceivedAtMs',
			'publicationEffectStartedAtMs',
			'renderStorageReadyAtMs',
			'sceneSyncCompletedAtMs'
		],
		label
	);
}

function expectResetProofRenderStorageWaitTrace(
	timeline: RenderPublicationTimelineTiming,
	label: string
) {
	const trace = timeline.renderStorageWaitTrace;
	expect(trace, `${label} renderStorageWaitTrace`).not.toBeNull();
	if (!trace) {
		throw new Error(`${label} missing renderStorageWaitTrace`);
	}
	for (const key of [
		'waitStartedAtMs',
		'waitFinishedAtMs',
		'waitMs',
		'readAttemptCount',
		'frameWaitCount',
		'deviceAvailableCount',
		'backendEntryAvailableCount',
		'bufferAvailableCount'
	] as const) {
		expect(trace[key], `${label} renderStorageWaitTrace.${key}`).toEqual(
			expect.any(Number)
		);
		expect(
			Number.isFinite(trace[key]),
			`${label} renderStorageWaitTrace.${key} should be finite`
		).toBe(true);
	}
	const waitStartedAtMs = trace.waitStartedAtMs;
	const waitFinishedAtMs = trace.waitFinishedAtMs;
	const waitMs = trace.waitMs;
	if (
		typeof waitStartedAtMs !== 'number' ||
		typeof waitFinishedAtMs !== 'number' ||
		typeof waitMs !== 'number'
	) {
		throw new Error(`${label} renderStorageWaitTrace wait fields must be numeric`);
	}
	expect(waitFinishedAtMs).toBeGreaterThanOrEqual(waitStartedAtMs);
	expect(waitMs).toBeGreaterThanOrEqual(0);
	expect(
		Math.abs(waitMs - (waitFinishedAtMs - waitStartedAtMs)),
		`${label} renderStorageWaitTrace.waitMs should match finish-start`
	).toBeLessThanOrEqual(1);
	expect(trace.readAttemptCount).toBeGreaterThan(0);
	expect(trace.frameWaitCount).toBeGreaterThanOrEqual(0);
	expect(trace.deviceAvailableCount).toBeGreaterThan(0);
	expect(trace.backendEntryAvailableCount).toBeGreaterThan(0);
	expect(trace.bufferAvailableCount).toBeGreaterThan(0);
	expect(
		trace.lastReadState,
		`${label} renderStorageWaitTrace.lastReadState`
	).toEqual({
		deviceAvailable: true,
		backendEntryAvailable: true,
		bufferAvailable: true
	});
	const firstReadyAtMs = [
		['firstDeviceAtMs', trace.firstDeviceAtMs],
		['firstBackendEntryAtMs', trace.firstBackendEntryAtMs],
		['firstBufferAtMs', trace.firstBufferAtMs]
	] as const;
	for (const [key, value] of firstReadyAtMs) {
		expect(value, `${label} renderStorageWaitTrace.${key}`).toEqual(
			expect.any(Number)
		);
		expect(
			Number.isFinite(value),
			`${label} renderStorageWaitTrace.${key} should be finite`
		).toBe(true);
		if (typeof value !== 'number' || !Number.isFinite(value)) {
			throw new Error(`${label} renderStorageWaitTrace.${key} should be finite`);
		}
		expect(
			value,
			`${label} renderStorageWaitTrace.${key} should not precede wait start`
		).toBeGreaterThanOrEqual(waitStartedAtMs);
		expect(
			value,
			`${label} renderStorageWaitTrace.${key} should not exceed wait finish`
		).toBeLessThanOrEqual(waitFinishedAtMs);
	}
	expect(trace.samples, `${label} renderStorageWaitTrace.samples`).toEqual(
		expect.any(Array)
	);
	expect(
		trace.samples.length,
		`${label} bounded renderStorageWaitTrace.samples`
	).toBeLessThanOrEqual(8);
	for (const [index, sample] of trace.samples.entries()) {
		expect(sample.atMs, `${label} renderStorageWaitTrace sample ${index} atMs`).toEqual(
			expect.any(Number)
		);
		expect(
			Number.isFinite(sample.atMs),
			`${label} renderStorageWaitTrace sample ${index} atMs should be finite`
		).toBe(true);
		expect(sample.deviceAvailable).toEqual(expect.any(Boolean));
		expect(sample.backendEntryAvailable).toEqual(expect.any(Boolean));
		expect(sample.bufferAvailable).toEqual(expect.any(Boolean));
	}
}

function expectResetProofRenderSurfaceMeshTrace(
	timeline: RenderPublicationTimelineTiming,
	label: string
) {
	const trace = timeline.renderSurfaceMeshTrace;
	expect(trace, `${label} renderSurfaceMeshTrace`).not.toBeNull();
	if (!trace) {
		throw new Error(`${label} missing renderSurfaceMeshTrace`);
	}
	expect(trace.action, `${label} renderSurfaceMeshTrace.action`).toMatch(
		/^(created|updated|update-failed-created)$/
	);
	expect(
		trace.recreateDecision,
		`${label} renderSurfaceMeshTrace.recreateDecision`
	).not.toBeNull();
	if (!trace.recreateDecision) {
		throw new Error(`${label} missing renderSurfaceMeshTrace.recreateDecision`);
	}
	for (const key of [
		'missingSurface',
		'notComputeBufferSurface',
		'analysisIdentityChanged',
		'layoutCompatible'
	] as const) {
		expect(
			trace.recreateDecision[key],
			`${label} renderSurfaceMeshTrace.recreateDecision.${key}`
		).toEqual(expect.any(Boolean));
	}
	for (const key of [
		'totalMs',
		'disposeResetMeshRemovalMs',
		'createComputeBufferSurfaceMeshMs',
		'updateComputeBufferSurfaceMeshMs',
		'fallbackDecisionMs',
		'applySurfaceMeshStateMs',
		'setCreatedSurfacePendingStorageInitMs',
		'setPostSurfacePendingStorageInitMs',
		'sceneAddMs',
		'publishUtciSurfaceDiagnosticsMs'
	] as const) {
		const value = trace[key];
		if (value == null) {
			continue;
		}
		expect(value, `${label} renderSurfaceMeshTrace.${key}`).toEqual(
			expect.any(Number)
		);
		expect(
			Number.isFinite(value),
			`${label} renderSurfaceMeshTrace.${key} should be finite`
		).toBe(true);
		expect(
			value,
			`${label} renderSurfaceMeshTrace.${key} should be nonnegative`
		).toBeGreaterThanOrEqual(0);
	}
	expect(trace.totalMs, `${label} renderSurfaceMeshTrace.totalMs`).toEqual(
		expect.any(Number)
	);
	const knownSubstepMs = [
		trace.disposeResetMeshRemovalMs,
		trace.createComputeBufferSurfaceMeshMs,
		trace.updateComputeBufferSurfaceMeshMs,
		trace.fallbackDecisionMs,
		trace.applySurfaceMeshStateMs,
		trace.setCreatedSurfacePendingStorageInitMs,
		trace.setPostSurfacePendingStorageInitMs,
		trace.sceneAddMs,
		trace.publishUtciSurfaceDiagnosticsMs
	].reduce<number>((sum, value) => sum + (value ?? 0), 0);
	expect(
		knownSubstepMs,
		`${label} renderSurfaceMeshTrace known substeps should fit within totalMs`
	).toBeLessThanOrEqual((trace.totalMs ?? 0) + 2);
	if (trace.action === 'created') {
		expect(
			trace.recreateDecision.missingSurface ||
				trace.recreateDecision.notComputeBufferSurface ||
				trace.recreateDecision.layoutCompatible === false,
			`${label} created trace should record a recreate guard`
		).toBe(true);
		expect(trace.createComputeBufferSurfaceMeshMs).toEqual(expect.any(Number));
		expect(trace.setCreatedSurfacePendingStorageInitMs).toEqual(
			expect.any(Number)
		);
		expect(trace.setPostSurfacePendingStorageInitMs).toEqual(expect.any(Number));
		expect(trace.sceneAddMs).toEqual(expect.any(Number));
		expect(trace.publishUtciSurfaceDiagnosticsMs).toEqual(expect.any(Number));
	}
	if (trace.action === 'updated') {
		expect(trace.recreateDecision.missingSurface).toBe(false);
		expect(trace.recreateDecision.notComputeBufferSurface).toBe(false);
		expect(trace.recreateDecision.layoutCompatible).toBe(true);
		expect(trace.updateComputeBufferSurfaceMeshMs).toEqual(expect.any(Number));
		expect(trace.setCreatedSurfacePendingStorageInitMs).toBeNull();
		expect(trace.setPostSurfacePendingStorageInitMs).toEqual(expect.any(Number));
	}
	if (trace.action === 'update-failed-created') {
		expect(trace.recreateDecision.missingSurface).toBe(false);
		expect(trace.recreateDecision.notComputeBufferSurface).toBe(false);
		expect(trace.recreateDecision.analysisIdentityChanged).toBe(false);
		expect(trace.recreateDecision.layoutCompatible).toBe(false);
		expect(trace.updateComputeBufferSurfaceMeshMs).toEqual(expect.any(Number));
		expect(trace.createComputeBufferSurfaceMeshMs).toEqual(expect.any(Number));
		expect(trace.setCreatedSurfacePendingStorageInitMs).toEqual(
			expect.any(Number)
		);
		expect(trace.setPostSurfacePendingStorageInitMs).toEqual(expect.any(Number));
	}
}

function expectResetProofTimeline(sample: CollectedSample, label: string) {
	const timeline = sample.timings.renderPublication?.renderPublicationTimeline;
	expect(timeline, `${label} renderPublicationTimeline`).not.toBeNull();
	if (!timeline) {
		throw new Error(`${label} missing renderPublicationTimeline`);
	}
	expect(timeline.sceneSyncAttemptToken, `${label} sceneSyncAttemptToken`).toEqual(
		expect.any(Number)
	);
	expectResetProofRenderSurfaceMeshTrace(timeline, label);
	expectResetProofRenderStorageWaitTrace(timeline, label);
	expect(timeline.sceneSyncResetHistory, `${label} sceneSyncResetHistory`).toEqual(
		expect.any(Array)
	);
	for (const [index, event] of timeline.sceneSyncResetHistory.entries()) {
		expect(event.resetAtMs, `${label} reset ${index} resetAtMs`).toEqual(
			expect.any(Number)
		);
		expect(event.resetReason, `${label} reset ${index} resetReason`).toEqual(
			expect.any(String)
		);
		expect(
			event.invalidateActiveRun,
			`${label} reset ${index} invalidateActiveRun`
		).toEqual(expect.any(Boolean));
		expect(
			event.previousCopyRunToken,
			`${label} reset ${index} previousCopyRunToken`
		).toEqual(expect.any(Number));
		expect(
			event.nextCopyRunToken,
			`${label} reset ${index} nextCopyRunToken`
		).toEqual(expect.any(Number));
	}

	expect(
		timeline.sceneSyncActiveWindowResetHistory,
		`${label} sceneSyncActiveWindowResetHistory`
	).toEqual(expect.any(Array));
	if (!Array.isArray(timeline.sceneSyncActiveWindowResetHistory)) {
		throw new Error(`${label} missing sceneSyncActiveWindowResetHistory`);
	}
	const scenePendingSurfaceObservedAtMs = timeline.scenePendingSurfaceObservedAtMs;
	const sceneSyncAttemptStartedAtMs = timeline.sceneSyncAttemptStartedAtMs;
	expect(scenePendingSurfaceObservedAtMs).toEqual(expect.any(Number));
	expect(sceneSyncAttemptStartedAtMs).toEqual(expect.any(Number));
	if (
		scenePendingSurfaceObservedAtMs == null ||
		sceneSyncAttemptStartedAtMs == null
	) {
		throw new Error(`${label} missing active reset-proof window timestamps`);
	}
	for (const [index, event] of timeline.sceneSyncActiveWindowResetHistory.entries()) {
		expect(
			event.resetAtMs,
			`${label} active-window reset ${index} resetAtMs`
		).toEqual(expect.any(Number));
		if (event.resetAtMs == null) {
			throw new Error(`${label} active-window reset ${index} missing resetAtMs`);
		}
		expect(
			event.resetAtMs,
			`${label} active-window reset ${index} should not precede scenePendingSurfaceObservedAtMs`
		).toBeGreaterThanOrEqual(scenePendingSurfaceObservedAtMs);
		expect(
			event.resetAtMs,
			`${label} active-window reset ${index} should not follow sceneSyncAttemptStartedAtMs`
		).toBeLessThanOrEqual(sceneSyncAttemptStartedAtMs);
	}
	expect(
		timeline.sceneSyncActiveWindowResetHistory.filter(
			(event) => event.resetReason === 'compute-surface-recreation'
		),
		`${label} should not reset compute surface after pending surface observation before sync attempt`
	).toEqual([]);
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

	test('reset proof: collects Ness Tziona normalized scrub render reset diagnostics', async ({
		page
	}, testInfo) => {
		test.setTimeout(300_000);

		const caseConfig = CASES.find(
			(entry) => entry.projectLabel === 'Ness-Tziona'
		);
		if (!caseConfig) {
			throw new Error('Ness Tziona case config is missing');
		}

		const requestedUrls: string[] = [];
		page.on('request', (request) => requestedUrls.push(request.url()));
		const normalized = await collectMode(page, caseConfig, 'normalized');
		expectValidRenderPublication(
			normalized.samples.initial,
			'Ness-Tziona normalized.initial reset proof'
		);
		expectValidRenderPublication(
			normalized.samples.scrub,
			'Ness-Tziona normalized.scrub reset proof'
		);
		expectResetProofTimeline(
			normalized.samples.scrub,
			'Ness-Tziona normalized.scrub reset proof'
		);

		const forbiddenRequestUrls = requestedUrls.filter(isForbiddenComparisonRequest);
		expect(forbiddenRequestUrls).toEqual([]);

		const artifact = {
			collectedOn: COLLECTED_ON,
			sourceRoute: SOURCE_ROUTE,
			targetGridResolutionMeters: TARGET_GRID_RESOLUTION_METERS,
			includedAnalyses: [caseConfig.analysisId],
			collectionMethod:
				'Narrow reset proof: main route only, Ness Tziona 0.5m, normalized initial load then app-visible hour scrub from 0 to 1.',
			forbiddenRequestUrls,
			samples: normalized.samples
		};

		if (!existsSync(RESULTS_DIR)) {
			mkdirSync(RESULTS_DIR, { recursive: true });
		}

		const json = JSON.stringify(artifact, null, 2);
		writeFileSync(RESET_PROOF_ARTIFACT_PATH, json, 'utf8');
		await testInfo.attach(RESET_PROOF_ARTIFACT_FILENAME, {
			body: json,
			contentType: 'application/json'
		});
	});
});
