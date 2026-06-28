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
const COLLECTED_ON = '2026-05-17';
const SOURCE_ROUTE = '/';
const TARGET_GRID_RESOLUTION_METERS = 0.5;
const WARMUP_SCRUB_HOUR_INDEX = 1;
const SCRUB_HOUR_INDEX = 2;
const REPEATED_SCRUB_HOUR_INDICES = [SCRUB_HOUR_INDEX, 3, 4] as const;

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
	controllerSessionRunStartedAtMs: number | null;
	controllerSessionRunCompletedAtMs: number | null;
	controllerAcceptStartedAtMs: number | null;
	controllerDiagnosticsMergedAtMs: number | null;
	controllerStatePublishedAtMs: number | null;
	sessionComputeOutputReturnedAtMs: number | null;
	sessionDiagnosticsAppliedAtMs: number | null;
	sessionGpuOutputHandleReadyAtMs: number | null;
	sessionPreferGpuResidentResolvedAtMs: number | null;
	sessionDebugReadbackStartedAtMs: number | null;
	sessionDebugReadbackCompletedAtMs: number | null;
	sessionSelectedHourRangeScanStartedAtMs: number | null;
	sessionSelectedHourRangeScanCompletedAtMs: number | null;
	sessionSelectedHourAnalysisBuildStartedAtMs: number | null;
	sessionSelectedHourAnalysisBuildCompletedAtMs: number | null;
	sessionRangeResolveStartedAtMs: number | null;
	sessionRangeResolveCompletedAtMs: number | null;
	sessionCpuFallbackSetupStartedAtMs: number | null;
	sessionCpuFallbackSetupCompletedAtMs: number | null;
	sessionGpuResidentRangeResolveStartedAtMs: number | null;
	sessionGpuResidentRangeResolveCompletedAtMs: number | null;
	sessionTooltipValuesHandoffStartedAtMs: number | null;
	sessionTooltipValuesHandoffCompletedAtMs: number | null;
	sessionGpuResidentResultAssemblyStartedAtMs: number | null;
	sessionGpuResidentResultAssemblyCompletedAtMs: number | null;
	sessionResultReadyAtMs: number | null;
	sessionResultReturnedAtMs: number | null;
	computeCompletedAtMs: number | null;
	selectedHourValuePublicationStartedAtMs: number | null;
	controllerAcceptedAtMs: number | null;
	routePendingSurfaceExposedAtMs: number | null;
	routePublishedAtMs: number | null;
	routeProjectedAtMs: number | null;
	routeProjectionEvaluationStartedAtMs: number | null;
	routeProjectionEvaluationCompletedAtMs: number | null;
	scenePendingSurfaceObservedAtMs: number | null;
	sceneReactiveBlockEnteredAtMs: number | null;
	sceneRenderStateResolvedAtMs: number | null;
	sceneAcceptedKeyResolvedAtMs: number | null;
	sceneSyncInvocationQueuedAtMs: number | null;
	sceneStartSyncEnteredAtMs: number | null;
	sceneStartSyncReturnedAtMs: number | null;
	sceneSyncAttemptStartedAtMs: number | null;
	sceneSyncAttemptToken: number | null;
	sceneSurfaceReceivedAtMs: number | null;
	controllerVisibleAcknowledgedAtMs: number | null;
	publicationEffectStartedAtMs: number | null;
	sceneLayoutKeyStartedAtMs: number | null;
	sceneLayoutKeyCompletedAtMs: number | null;
	scenePublicationPlanReadyAtMs: number | null;
	renderLayoutBuildTrace: RenderLayoutBuildTrace | null;
	renderLayoutReuseProofTrace: RenderLayoutReuseProofTrace | null;
	renderLayoutReuseAction: 'reuse-candidate' | 'build-required' | 'reused' | null;
	renderLayoutReuseReason: string | null;
	renderLayoutReuseDecisionMs: number | null;
	renderLayoutReuseKeyMs: number | null;
	renderLayoutReuseSourceSignatureMs: number | null;
	renderLayoutReusePositionsSignatureMs: number | null;
	renderLayoutReusePositionsSignatureCacheHit: boolean | null;
	renderLayoutReuseFrameCacheLookupMs: number | null;
	renderLayoutReuseFrameDerivationMs: number | null;
	renderLayoutReuseFrameCacheHit: boolean | null;
	renderLayoutReuseFrameCacheKind:
		| 'analysis-object'
		| 'structural'
		| 'miss'
		| null;
	renderLayoutPublicationPlanMs: number | null;
	renderLayoutCompatibilityMs: number | null;
	renderLayoutCompatibilityRequiredExpensiveMappingComparison: boolean | null;
	renderLayoutCompatibilityPerformedExpensiveMappingComparison: boolean | null;
	renderLayoutReuseProofMs: number | null;
	renderLayoutReuseKeyMatch: boolean | null;
	renderLayoutReuseProofSource:
		| 'fresh-build-proof'
		| 'previous-publication-proof'
		| 'refreshed-runtime-proof'
		| null;
	renderLayoutReusePreviousKey: string | null;
	renderLayoutReusePreviousRequestId: number | null;
	renderLayoutReusePreviousSelectionKey: string | null;
	activeLayoutCandidateCount: number | null;
	renderSurfaceMeshTrace: RenderSurfaceMeshTrace | null;
	sceneSurfacePendingStorageInitAtMs: number | null;
	renderStorageWaitStartedAtMs: number | null;
	renderStoragePreWaitMs: number | null;
	renderStorageReadyAtMs: number | null;
	renderStorageWaitTrace: RenderStorageWaitTrace | null;
	renderBufferCopyEncoderCreateMs: number | null;
	renderBufferCopyCommandRecordMs: number | null;
	renderBufferCopySubmitMs: number | null;
	sceneSyncCompletedAtMs: number | null;
	sceneSyncResetHistory: RenderPublicationSceneSyncResetEvent[];
	sceneSyncActiveWindowResetHistory: RenderPublicationSceneSyncResetEvent[] | null;
};

const REUSED_LAYOUT_PROOF_SOURCES = [
	'previous-publication-proof',
	'refreshed-runtime-proof'
] as const;

type RenderLayoutBuildTrace = {
	totalMs: number | null;
	arrayAllocationMs: number | null;
	transformBoundsPassMs: number | null;
	coordinateAssignmentMs: number | null;
	indexToTexelFillMs: number | null;
	cellToPointIndexBuildMs: number | null;
	colorBufferAllocationMs: number | null;
};

type RenderLayoutNormalizationSignature = {
	enabled: boolean | null;
	offset: {
		x: number | null;
		y: number | null;
		z: number | null;
	} | null;
	provenance: string | null;
};

type RenderLayoutReuseProofTrace = {
	decision: 'reuse-safe' | 'rebuild-required' | 'proof-inconclusive' | null;
	hoverCellLookupProofStatus:
		| 'same-point-confirmed'
		| 'not-compatible'
		| 'proof-inconclusive'
		| null;
	previousLayoutPresent: boolean | null;
	canonicalRuntimeCompatibilityWouldReuse: boolean | null;
	proofMatchesCanonicalRuntimeCompatibility: boolean | null;
	positionsReferenceMatch: boolean | null;
	pointCountMatch: boolean | null;
	gridSizeMatch: boolean | null;
	coordinateSystemMatch: boolean | null;
	normalizationSignature: RenderLayoutNormalizationSignature | null;
	previousNormalizationSignature: RenderLayoutNormalizationSignature | null;
	normalizationSignatureMatch: boolean | null;
	constructionMode: string | null;
	previousConstructionMode: string | null;
	constructionModeMatch: boolean | null;
	dimensionsMatch: boolean | null;
	placementMatch: boolean | null;
	cellToPointMappingMatch: boolean | null;
	proofCostMs: number | null;
	estimatedRetainedCpuLayoutBytes: number | null;
};

type RenderSurfaceMeshTrace = {
	action: 'created' | 'updated' | 'update-failed-created' | null;
	totalMs: number | null;
	recreateDecision: RenderSurfaceMeshRecreateDecision | null;
	disposeResetMeshRemovalMs: number | null;
	createComputeBufferSurfaceMeshMs: number | null;
	updateComputeBufferSurfaceMeshMs: number | null;
	updateComputeBufferSurfaceRangeUniformMs: number | null;
	updateComputeBufferSurfacePendingSourceMs: number | null;
	updateComputeBufferSurfaceLayoutUserDataMs: number | null;
	updateComputeBufferSurfaceByteAccountingMs: number | null;
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

type RenderPublicationTimelineNumberKey = {
	[K in keyof RenderPublicationTimelineTiming]: RenderPublicationTimelineTiming[K] extends
		| number
		| null
		? K
		: never;
}[keyof RenderPublicationTimelineTiming];

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

type RepeatedScrubSoakSample = {
	hourIndex: number;
	selectionKey: string | null;
	surfaceRequestId: number | null;
	renderLayoutReuseAction: RenderPublicationTimelineTiming['renderLayoutReuseAction'];
	renderLayoutReuseReason: string | null;
	activeLayoutCandidateCount: number | null;
	reusedLayoutIdentity: string | null;
	retainedCpuLayoutBytes: number | null;
	ownedGpuMemoryBytes: number;
	renderOwnedSelectedHourBytes: number;
	renderOwnedSelectedHourBytesHighWatermark: number;
	hoverCellLookupProofStatus: RenderLayoutReuseProofTrace['hoverCellLookupProofStatus'];
	hoverProbe: {
		positionIndex: number;
		value: number;
		position: { x: number; y: number; z: number };
	};
};

type RepeatedScrubSoakResult = {
	projectLabel: string;
	analysisId: string;
	colorMode: ColorMode;
	query: string;
	warmupHourIndex: number;
	reusedSamples: RepeatedScrubSoakSample[];
	plateauRetainedCpuLayoutBytes: number | null;
	plateauOwnedGpuMemoryBytes: number;
	plateauRenderOwnedSelectedHourBytes: number;
	plateauRenderOwnedSelectedHourBytesHighWatermark: number;
	stableReusedLayoutIdentity: string | null;
	rebuildReplacement: {
		hourIndex: number;
		selectionKey: string | null;
		surfaceRequestId: number | null;
		renderLayoutReuseAction: RenderPublicationTimelineTiming['renderLayoutReuseAction'];
		renderLayoutReuseReason: string | null;
		activeLayoutCandidateCount: number | null;
		releasedPreviousLayout: string | null;
		ownedGpuMemoryBytes: number;
		renderOwnedSelectedHourBytes: number;
		renderOwnedSelectedHourBytesHighWatermark: number;
	};
};

type MainRouteTooltipProbe = {
	clientX: number;
	clientY: number;
	positionIndex: number;
	value: number;
	position: { x: number; y: number; z: number };
	tooltipHourIndex: number;
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

async function readMainRouteTooltipProbe(page: Page): Promise<MainRouteTooltipProbe | null> {
	return page.evaluate(() => {
		return (window as Window & {
			__mainRouteTooltipProbe__?: (() => MainRouteTooltipProbe | null) | undefined;
		}).__mainRouteTooltipProbe__?.() ?? null;
	});
}

async function readMainRouteTooltipAt(
	page: Page,
	clientX: number,
	clientY: number
): Promise<MainRouteTooltipProbe | null> {
	return page.evaluate(
		([x, y]) => {
			return (window as Window & {
				__mainRouteTooltipAt__?:
					| ((clientX: number, clientY: number) => MainRouteTooltipProbe | null)
					| undefined;
			}).__mainRouteTooltipAt__?.(x, y) ?? null;
		},
		[clientX, clientY] as const
	);
}

async function readMainRouteLastTooltip(
	page: Page
): Promise<MainRouteTooltipProbe | null> {
	return page.evaluate(() => {
		return (window as Window & {
			__mainRouteLastTooltip__?: MainRouteTooltipProbe | null | undefined;
		}).__mainRouteLastTooltip__ ?? null;
	});
}

async function clearMainRouteLastTooltip(page: Page) {
	await page.evaluate(() => {
		(window as Window & {
			__mainRouteLastTooltip__?: MainRouteTooltipProbe | null | undefined;
		}).__mainRouteLastTooltip__ = null;
	});
}

async function requestDiagnosticsGridResolutionChange(
	page: Page,
	resolutionMeters: number
) {
	const armed = await page.evaluate((resolution) => {
		return (
			(window as Window & {
				__mainRouteDiagnosticsSetGridResolution?: (
					resolutionMeters: number
				) => boolean;
			}).__mainRouteDiagnosticsSetGridResolution?.(resolution) ?? false
		);
	}, resolutionMeters);
	expect(
		armed,
		'diagnostics-only grid change hook should preserve the active base surface candidate'
	).toBe(true);
}

function formatTooltipPosition(position: { x: number; y: number; z: number }): string {
	return `X ${position.x.toFixed(3)} / Y ${position.y.toFixed(3)} / Z ${position.z.toFixed(3)}`;
}

async function hoverMainRouteProbe(page: Page): Promise<{
	probe: MainRouteTooltipProbe;
	oracle: MainRouteTooltipProbe;
	liveTooltip: MainRouteTooltipProbe;
}> {
	for (const [dx, dy] of [
		[0, 0],
		[2, 0],
		[0, 2],
		[-2, 0],
		[0, -2]
	] as const) {
		const probe = await readMainRouteTooltipProbe(page);
		if (!probe) {
			throw new Error('Main route never exposed a concrete tooltip probe point.');
		}

		await clearMainRouteLastTooltip(page);
		await page.waitForTimeout(20);
		await page.mouse.move(probe.clientX + dx, probe.clientY + dy);
		await expect(page.getByRole('tooltip')).toBeVisible({ timeout: 5_000 });
		const liveTooltipHandle = await page.waitForFunction(
			() => {
				const tooltip = (window as Window & {
					__mainRouteLastTooltip__?: MainRouteTooltipProbe | null | undefined;
				}).__mainRouteLastTooltip__;
				return tooltip ?? null;
			},
			undefined,
			{ timeout: 5_000 }
		).catch(() => null);
		const liveTooltip =
			(await liveTooltipHandle?.jsonValue().catch(() => null)) ??
			(await readMainRouteLastTooltip(page));
		if (liveTooltip) {
			const oracle = await readMainRouteTooltipAt(
				page,
				liveTooltip.clientX,
				liveTooltip.clientY
			);
			if (!oracle) {
				continue;
			}
			return { probe, oracle, liveTooltip };
		}
	}

	throw new Error('Expected a visible main-route tooltip hit, but no live tooltip payload was recorded.');
}

async function expectTooltipProbeMatch(
	page: Page,
	label: string
): Promise<{
	probe: MainRouteTooltipProbe;
	oracle: MainRouteTooltipProbe;
	liveTooltip: MainRouteTooltipProbe;
}> {
	const hovered = await hoverMainRouteProbe(page);
	expect(hovered.liveTooltip.positionIndex, `${label} hovered point index`).toBe(
		hovered.oracle.positionIndex
	);
	expect(hovered.liveTooltip.value, `${label} hovered UTCI value`).toBeCloseTo(
		hovered.oracle.value,
		6
	);
	expect(hovered.liveTooltip.position, `${label} hovered position`).toEqual(
		hovered.oracle.position
	);
	await expect(
		page.locator('[role="tooltip"] .tooltip-value'),
		`${label} tooltip value`
	).toHaveText(hovered.oracle.value.toFixed(1));
	await expect(
		page.locator('[role="tooltip"] .tooltip-position'),
		`${label} tooltip position`
	).toHaveText(formatTooltipPosition(hovered.oracle.position));
	return hovered;
}

async function waitForSelectedHourPublication(
	page: Page,
	params: {
		expectedSelectionKey: string;
		colorMode: ColorMode;
		minSurfaceRequestId?: number;
		expectedGridResolution?: number;
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
			gridResolution:
				params.expectedGridResolution ?? TARGET_GRID_RESOLUTION_METERS,
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
		controllerSessionRunStartedAtMs: numberOrNull(
			payload.controllerSessionRunStartedAtMs
		),
		controllerSessionRunCompletedAtMs: numberOrNull(
			payload.controllerSessionRunCompletedAtMs
		),
		controllerAcceptStartedAtMs: numberOrNull(payload.controllerAcceptStartedAtMs),
		controllerDiagnosticsMergedAtMs: numberOrNull(
			payload.controllerDiagnosticsMergedAtMs
		),
		controllerStatePublishedAtMs: numberOrNull(payload.controllerStatePublishedAtMs),
		sessionComputeOutputReturnedAtMs: numberOrNull(
			payload.sessionComputeOutputReturnedAtMs
		),
		sessionDiagnosticsAppliedAtMs: numberOrNull(payload.sessionDiagnosticsAppliedAtMs),
		sessionGpuOutputHandleReadyAtMs: numberOrNull(
			payload.sessionGpuOutputHandleReadyAtMs
		),
		sessionPreferGpuResidentResolvedAtMs: numberOrNull(
			payload.sessionPreferGpuResidentResolvedAtMs
		),
		sessionDebugReadbackStartedAtMs: numberOrNull(
			payload.sessionDebugReadbackStartedAtMs
		),
		sessionDebugReadbackCompletedAtMs: numberOrNull(
			payload.sessionDebugReadbackCompletedAtMs
		),
		sessionSelectedHourRangeScanStartedAtMs: numberOrNull(
			payload.sessionSelectedHourRangeScanStartedAtMs
		),
		sessionSelectedHourRangeScanCompletedAtMs: numberOrNull(
			payload.sessionSelectedHourRangeScanCompletedAtMs
		),
		sessionSelectedHourAnalysisBuildStartedAtMs: numberOrNull(
			payload.sessionSelectedHourAnalysisBuildStartedAtMs
		),
		sessionSelectedHourAnalysisBuildCompletedAtMs: numberOrNull(
			payload.sessionSelectedHourAnalysisBuildCompletedAtMs
		),
		sessionRangeResolveStartedAtMs: numberOrNull(
			payload.sessionRangeResolveStartedAtMs
		),
		sessionRangeResolveCompletedAtMs: numberOrNull(
			payload.sessionRangeResolveCompletedAtMs
		),
		sessionCpuFallbackSetupStartedAtMs: numberOrNull(
			payload.sessionCpuFallbackSetupStartedAtMs
		),
		sessionCpuFallbackSetupCompletedAtMs: numberOrNull(
			payload.sessionCpuFallbackSetupCompletedAtMs
		),
		sessionGpuResidentRangeResolveStartedAtMs: numberOrNull(
			payload.sessionGpuResidentRangeResolveStartedAtMs
		),
		sessionGpuResidentRangeResolveCompletedAtMs: numberOrNull(
			payload.sessionGpuResidentRangeResolveCompletedAtMs
		),
		sessionTooltipValuesHandoffStartedAtMs: numberOrNull(
			payload.sessionTooltipValuesHandoffStartedAtMs
		),
		sessionTooltipValuesHandoffCompletedAtMs: numberOrNull(
			payload.sessionTooltipValuesHandoffCompletedAtMs
		),
		sessionGpuResidentResultAssemblyStartedAtMs: numberOrNull(
			payload.sessionGpuResidentResultAssemblyStartedAtMs
		),
		sessionGpuResidentResultAssemblyCompletedAtMs: numberOrNull(
			payload.sessionGpuResidentResultAssemblyCompletedAtMs
		),
		sessionResultReadyAtMs: numberOrNull(payload.sessionResultReadyAtMs),
		sessionResultReturnedAtMs: numberOrNull(payload.sessionResultReturnedAtMs),
		computeCompletedAtMs: numberOrNull(payload.computeCompletedAtMs),
		selectedHourValuePublicationStartedAtMs: numberOrNull(
			payload.selectedHourValuePublicationStartedAtMs
		),
		controllerAcceptedAtMs: numberOrNull(payload.controllerAcceptedAtMs),
		routePendingSurfaceExposedAtMs: numberOrNull(
			payload.routePendingSurfaceExposedAtMs
		),
		routePublishedAtMs: numberOrNull(payload.routePublishedAtMs),
		routeProjectedAtMs: numberOrNull(payload.routeProjectedAtMs),
		routeProjectionEvaluationStartedAtMs: numberOrNull(
			payload.routeProjectionEvaluationStartedAtMs
		),
		routeProjectionEvaluationCompletedAtMs: numberOrNull(
			payload.routeProjectionEvaluationCompletedAtMs
		),
		scenePendingSurfaceObservedAtMs: numberOrNull(
			payload.scenePendingSurfaceObservedAtMs
		),
		sceneReactiveBlockEnteredAtMs: numberOrNull(payload.sceneReactiveBlockEnteredAtMs),
		sceneRenderStateResolvedAtMs: numberOrNull(payload.sceneRenderStateResolvedAtMs),
		sceneAcceptedKeyResolvedAtMs: numberOrNull(payload.sceneAcceptedKeyResolvedAtMs),
		sceneSyncInvocationQueuedAtMs: numberOrNull(
			payload.sceneSyncInvocationQueuedAtMs
		),
		sceneStartSyncEnteredAtMs: numberOrNull(payload.sceneStartSyncEnteredAtMs),
		sceneStartSyncReturnedAtMs: numberOrNull(payload.sceneStartSyncReturnedAtMs),
		sceneSyncAttemptStartedAtMs: numberOrNull(payload.sceneSyncAttemptStartedAtMs),
		sceneSyncAttemptToken: numberOrNull(payload.sceneSyncAttemptToken),
		sceneSurfaceReceivedAtMs: numberOrNull(payload.sceneSurfaceReceivedAtMs),
		controllerVisibleAcknowledgedAtMs: numberOrNull(
			payload.controllerVisibleAcknowledgedAtMs
		),
		publicationEffectStartedAtMs: numberOrNull(payload.publicationEffectStartedAtMs),
		sceneLayoutKeyStartedAtMs: numberOrNull(payload.sceneLayoutKeyStartedAtMs),
		sceneLayoutKeyCompletedAtMs: numberOrNull(payload.sceneLayoutKeyCompletedAtMs),
		scenePublicationPlanReadyAtMs: numberOrNull(payload.scenePublicationPlanReadyAtMs),
		renderLayoutBuildTrace: extractRenderLayoutBuildTrace(
			payload.renderLayoutBuildTrace
		),
		renderLayoutReuseProofTrace: extractRenderLayoutReuseProofTrace(
			payload.renderLayoutReuseProofTrace
		),
		renderLayoutReuseAction: stringFromSetOrNull(payload.renderLayoutReuseAction, [
			'reuse-candidate',
			'build-required',
			'reused'
		]),
		renderLayoutReuseReason: stringOrNull(payload.renderLayoutReuseReason),
		renderLayoutReuseDecisionMs: numberOrNull(payload.renderLayoutReuseDecisionMs),
		renderLayoutReuseKeyMs: numberOrNull(payload.renderLayoutReuseKeyMs),
		renderLayoutReuseSourceSignatureMs: numberOrNull(
			payload.renderLayoutReuseSourceSignatureMs
		),
		renderLayoutReusePositionsSignatureMs: numberOrNull(
			payload.renderLayoutReusePositionsSignatureMs
		),
		renderLayoutReusePositionsSignatureCacheHit: booleanOrNull(
			payload.renderLayoutReusePositionsSignatureCacheHit
		),
		renderLayoutReuseFrameCacheLookupMs: numberOrNull(
			payload.renderLayoutReuseFrameCacheLookupMs
		),
		renderLayoutReuseFrameDerivationMs: numberOrNull(
			payload.renderLayoutReuseFrameDerivationMs
		),
		renderLayoutReuseFrameCacheHit: booleanOrNull(
			payload.renderLayoutReuseFrameCacheHit
		),
		renderLayoutReuseFrameCacheKind: stringFromSetOrNull(
			payload.renderLayoutReuseFrameCacheKind,
			['analysis-object', 'structural', 'miss']
		),
		renderLayoutPublicationPlanMs: numberOrNull(
			payload.renderLayoutPublicationPlanMs
		),
		renderLayoutCompatibilityMs: numberOrNull(payload.renderLayoutCompatibilityMs),
		renderLayoutCompatibilityRequiredExpensiveMappingComparison: booleanOrNull(
			payload.renderLayoutCompatibilityRequiredExpensiveMappingComparison
		),
		renderLayoutCompatibilityPerformedExpensiveMappingComparison: booleanOrNull(
			payload.renderLayoutCompatibilityPerformedExpensiveMappingComparison
		),
		renderLayoutReuseProofMs: numberOrNull(payload.renderLayoutReuseProofMs),
		renderLayoutReuseKeyMatch: booleanOrNull(payload.renderLayoutReuseKeyMatch),
		renderLayoutReuseProofSource: stringFromSetOrNull(
			payload.renderLayoutReuseProofSource,
			['fresh-build-proof', 'previous-publication-proof', 'refreshed-runtime-proof']
		),
		renderLayoutReusePreviousKey: stringOrNull(payload.renderLayoutReusePreviousKey),
		renderLayoutReusePreviousRequestId: numberOrNull(
			payload.renderLayoutReusePreviousRequestId
		),
		renderLayoutReusePreviousSelectionKey: stringOrNull(
			payload.renderLayoutReusePreviousSelectionKey
		),
		activeLayoutCandidateCount: numberOrNull(payload.activeLayoutCandidateCount),
		renderSurfaceMeshTrace: extractRenderSurfaceMeshTrace(
			payload.renderSurfaceMeshTrace
		),
		sceneSurfacePendingStorageInitAtMs: numberOrNull(
			payload.sceneSurfacePendingStorageInitAtMs
		),
		renderStorageWaitStartedAtMs: numberOrNull(payload.renderStorageWaitStartedAtMs),
		renderStoragePreWaitMs: numberOrNull(payload.renderStoragePreWaitMs),
		renderStorageReadyAtMs: numberOrNull(payload.renderStorageReadyAtMs),
		renderStorageWaitTrace: extractRenderStorageWaitTrace(
			payload.renderStorageWaitTrace
		),
		renderBufferCopyEncoderCreateMs: numberOrNull(
			payload.renderBufferCopyEncoderCreateMs
		),
		renderBufferCopyCommandRecordMs: numberOrNull(
			payload.renderBufferCopyCommandRecordMs
		),
		renderBufferCopySubmitMs: numberOrNull(payload.renderBufferCopySubmitMs),
		sceneSyncCompletedAtMs: numberOrNull(payload.sceneSyncCompletedAtMs),
		sceneSyncResetHistory: extractSceneSyncResetHistory(
			payload.sceneSyncResetHistory
		),
		sceneSyncActiveWindowResetHistory: extractPresentSceneSyncResetHistory(
			payload.sceneSyncActiveWindowResetHistory
		)
	};
}

function extractRenderLayoutBuildTrace(value: unknown): RenderLayoutBuildTrace | null {
	if (typeof value !== 'object' || value == null) {
		return null;
	}

	const payload = value as Record<string, unknown>;
	return {
		totalMs: numberOrNull(payload.totalMs),
		arrayAllocationMs: numberOrNull(payload.arrayAllocationMs),
		transformBoundsPassMs: numberOrNull(payload.transformBoundsPassMs),
		coordinateAssignmentMs: numberOrNull(payload.coordinateAssignmentMs),
		indexToTexelFillMs: numberOrNull(payload.indexToTexelFillMs),
		cellToPointIndexBuildMs: numberOrNull(payload.cellToPointIndexBuildMs),
		colorBufferAllocationMs: numberOrNull(payload.colorBufferAllocationMs)
	};
}

function extractRenderLayoutNormalizationSignature(
	value: unknown
): RenderLayoutNormalizationSignature | null {
	if (typeof value !== 'object' || value == null) {
		return null;
	}

	const payload = value as Record<string, unknown>;
	return {
		enabled: booleanOrNull(payload.enabled),
		offset:
			typeof payload.offset === 'object' && payload.offset != null
				? {
						x: numberOrNull((payload.offset as Record<string, unknown>).x),
						y: numberOrNull((payload.offset as Record<string, unknown>).y),
						z: numberOrNull((payload.offset as Record<string, unknown>).z)
					}
				: null,
		provenance: stringOrNull(payload.provenance)
	};
}

function extractRenderLayoutReuseProofTrace(
	value: unknown
): RenderLayoutReuseProofTrace | null {
	if (typeof value !== 'object' || value == null) {
		return null;
	}

	const payload = value as Record<string, unknown>;
	return {
		decision: stringFromSetOrNull(payload.decision, [
			'reuse-safe',
			'rebuild-required',
			'proof-inconclusive'
		]),
		hoverCellLookupProofStatus: stringFromSetOrNull(payload.hoverCellLookupProofStatus, [
			'same-point-confirmed',
			'not-compatible',
			'proof-inconclusive'
		]),
		previousLayoutPresent: booleanOrNull(payload.previousLayoutPresent),
		canonicalRuntimeCompatibilityWouldReuse: booleanOrNull(
			payload.canonicalRuntimeCompatibilityWouldReuse
		),
		proofMatchesCanonicalRuntimeCompatibility: booleanOrNull(
			payload.proofMatchesCanonicalRuntimeCompatibility
		),
		positionsReferenceMatch: booleanOrNull(payload.positionsReferenceMatch),
		pointCountMatch: booleanOrNull(payload.pointCountMatch),
		gridSizeMatch: booleanOrNull(payload.gridSizeMatch),
		coordinateSystemMatch: booleanOrNull(payload.coordinateSystemMatch),
		normalizationSignature: extractRenderLayoutNormalizationSignature(
			payload.normalizationSignature
		),
		previousNormalizationSignature: extractRenderLayoutNormalizationSignature(
			payload.previousNormalizationSignature
		),
		normalizationSignatureMatch: booleanOrNull(payload.normalizationSignatureMatch),
		constructionMode: stringOrNull(payload.constructionMode),
		previousConstructionMode: stringOrNull(payload.previousConstructionMode),
		constructionModeMatch: booleanOrNull(payload.constructionModeMatch),
		dimensionsMatch: booleanOrNull(payload.dimensionsMatch),
		placementMatch: booleanOrNull(payload.placementMatch),
		cellToPointMappingMatch: booleanOrNull(payload.cellToPointMappingMatch),
		proofCostMs: numberOrNull(payload.proofCostMs),
		estimatedRetainedCpuLayoutBytes: numberOrNull(
			payload.estimatedRetainedCpuLayoutBytes
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
		updateComputeBufferSurfaceRangeUniformMs: numberOrNull(
			payload.updateComputeBufferSurfaceRangeUniformMs
		),
		updateComputeBufferSurfacePendingSourceMs: numberOrNull(
			payload.updateComputeBufferSurfacePendingSourceMs
		),
		updateComputeBufferSurfaceLayoutUserDataMs: numberOrNull(
			payload.updateComputeBufferSurfaceLayoutUserDataMs
		),
		updateComputeBufferSurfaceByteAccountingMs: numberOrNull(
			payload.updateComputeBufferSurfaceByteAccountingMs
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

function deriveReusedLayoutIdentity(
	timeline: RenderPublicationTimelineTiming | null | undefined
): string | null {
	return timeline?.renderLayoutReuseAction === 'reused'
		? timeline.renderLayoutReusePreviousKey
		: null;
}

function expectValidRenderPublication(
	sample: CollectedSample,
	label: string
) {
	expect(
		sample.proof.rendererBackend,
		`${label} renderer backend should stay on WebGPU for compute-buffer render diagnostics`
	).toBe('webgpu');
	expect(
		sample.proof.utciRenderResolved,
		`${label} utciRenderResolved should stay gpuNative for compute-buffer render diagnostics`
	).toBe('gpuNative');
	expect(
		sample.proof.utciSurfaceSource,
		`${label} utciSurfaceSource regressed before compute-buffer render-publication validation`
	).toBe('compute-buffer-selected-hour');
	expect(
		sample.proof.baseRenderTransport,
		`${label} baseRenderTransport regressed before compute-buffer render-publication validation`
	).toBe('compute-buffer-selected-hour');
	expect(
		sample.proof.dataTextureBuildCount,
		`${label} dataTextureBuildCount regressed before compute-buffer render-publication validation`
	).toBe(0);
	expect(
		sample.proof.baseSameDeviceForComputeAndRender,
		`${label} same-device proof regressed before compute-buffer render-publication validation`
	).toBe(true);
	expect(
		sample.proof.selectedHourRuntimeContract.route,
		`${label} selectedHourRuntimeContract.route regressed before compute-buffer render-publication validation`
	).toBe('main');
	expect(
		sample.proof.selectedHourRuntimeContract.readbackInstrumentation,
		`${label} selectedHourRuntimeContract.readbackInstrumentation regressed before compute-buffer render-publication validation`
	).toBe('instrumented');
	expect(
		sample.proof.selectedHourRuntimeContract.visibleSelectedHourReadbackCount,
		`${label} visibleSelectedHourReadbackCount regressed before compute-buffer render-publication validation`
	).toBe(0);
	expect(
		sample.proof.selectedHourRuntimeContract.strongVisibleGpuPath,
		`${label} strongVisibleGpuPath regressed before compute-buffer render-publication validation`
	).toBe(true);
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
		previousValue = typeof value === 'number' ? value : undefined;
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

	requireTimelineNumber(timeline, 'computeCompletedAtMs', label);
	const controllerAcceptedAtMs = requireTimelineNumber(
		timeline,
		'controllerAcceptedAtMs',
		label
	);
	const requireSessionTimeline =
		renderPublication?.renderPublicationPhase === 'initial' ||
		renderPublication?.renderPublicationPhase === 'scrub' ||
		timeline.sessionSelectedHourRangeScanStartedAtMs != null ||
		timeline.sessionSelectedHourAnalysisBuildStartedAtMs != null ||
		timeline.sessionGpuResidentRangeResolveStartedAtMs != null;
	const selectedHourValuePublicationStartedAtMs = requireTimelineNumber(
		timeline,
		'selectedHourValuePublicationStartedAtMs',
		label
	);
	let controllerDiagnosticsMergedAtMs: number | null = null;
	if (requireSessionTimeline) {
		const sessionComputeOutputReturnedAtMs = requireTimelineNumber(
			timeline,
			'sessionComputeOutputReturnedAtMs',
			label
		);
		const sessionDiagnosticsAppliedAtMs = requireTimelineNumber(
			timeline,
			'sessionDiagnosticsAppliedAtMs',
			label
		);
		const sessionGpuOutputHandleReadyAtMs = requireTimelineNumber(
			timeline,
			'sessionGpuOutputHandleReadyAtMs',
			label
		);
		const sessionPreferGpuResidentResolvedAtMs = requireTimelineNumber(
			timeline,
			'sessionPreferGpuResidentResolvedAtMs',
			label
		);
		const sessionDebugReadbackStartedAtMs = requireTimelineNumber(
			timeline,
			'sessionDebugReadbackStartedAtMs',
			label
		);
		const sessionDebugReadbackCompletedAtMs = requireTimelineNumber(
			timeline,
			'sessionDebugReadbackCompletedAtMs',
			label
		);
		const sessionSelectedHourRangeScanStartedAtMs = requireTimelineNumber(
			timeline,
			'sessionSelectedHourRangeScanStartedAtMs',
			label
		);
		const sessionSelectedHourRangeScanCompletedAtMs = requireTimelineNumber(
			timeline,
			'sessionSelectedHourRangeScanCompletedAtMs',
			label
		);
		const sessionRangeResolveStartedAtMs = requireTimelineNumber(
			timeline,
			'sessionRangeResolveStartedAtMs',
			label
		);
		const sessionRangeResolveCompletedAtMs = requireTimelineNumber(
			timeline,
			'sessionRangeResolveCompletedAtMs',
			label
		);
		const sessionSelectedHourAnalysisBuildStartedAtMs = requireTimelineNumber(
			timeline,
			'sessionSelectedHourAnalysisBuildStartedAtMs',
			label
		);
		const sessionSelectedHourAnalysisBuildCompletedAtMs = requireTimelineNumber(
			timeline,
			'sessionSelectedHourAnalysisBuildCompletedAtMs',
			label
		);
		const sessionCpuFallbackSetupStartedAtMs = requireTimelineNumber(
			timeline,
			'sessionCpuFallbackSetupStartedAtMs',
			label
		);
		const sessionCpuFallbackSetupCompletedAtMs = requireTimelineNumber(
			timeline,
			'sessionCpuFallbackSetupCompletedAtMs',
			label
		);
		const sessionGpuResidentRangeResolveStartedAtMs = requireTimelineNumber(
			timeline,
			'sessionGpuResidentRangeResolveStartedAtMs',
			label
		);
		const sessionGpuResidentRangeResolveCompletedAtMs = requireTimelineNumber(
			timeline,
			'sessionGpuResidentRangeResolveCompletedAtMs',
			label
		);
		const sessionTooltipValuesHandoffStartedAtMs = requireTimelineNumber(
			timeline,
			'sessionTooltipValuesHandoffStartedAtMs',
			label
		);
		const sessionTooltipValuesHandoffCompletedAtMs = requireTimelineNumber(
			timeline,
			'sessionTooltipValuesHandoffCompletedAtMs',
			label
		);
		const sessionGpuResidentResultAssemblyStartedAtMs = requireTimelineNumber(
			timeline,
			'sessionGpuResidentResultAssemblyStartedAtMs',
			label
		);
		const sessionGpuResidentResultAssemblyCompletedAtMs = requireTimelineNumber(
			timeline,
			'sessionGpuResidentResultAssemblyCompletedAtMs',
			label
		);
		const sessionResultReadyAtMs = requireTimelineNumber(
			timeline,
			'sessionResultReadyAtMs',
			label
		);
		const sessionResultReturnedAtMs = requireTimelineNumber(
			timeline,
			'sessionResultReturnedAtMs',
			label
		);
		expect(
			sessionDiagnosticsAppliedAtMs,
			`${label} sessionDiagnosticsAppliedAtMs should not precede sessionComputeOutputReturnedAtMs`
		).toBeGreaterThanOrEqual(sessionComputeOutputReturnedAtMs);
		expect(
			sessionGpuOutputHandleReadyAtMs,
			`${label} sessionGpuOutputHandleReadyAtMs should not precede sessionDiagnosticsAppliedAtMs`
		).toBeGreaterThanOrEqual(sessionDiagnosticsAppliedAtMs);
		expect(
			sessionPreferGpuResidentResolvedAtMs,
			`${label} sessionPreferGpuResidentResolvedAtMs should not precede sessionGpuOutputHandleReadyAtMs`
		).toBeGreaterThanOrEqual(sessionGpuOutputHandleReadyAtMs);
		expect(
			sessionDebugReadbackStartedAtMs,
			`${label} sessionDebugReadbackStartedAtMs should not precede sessionPreferGpuResidentResolvedAtMs`
		).toBeGreaterThanOrEqual(sessionPreferGpuResidentResolvedAtMs);
		expect(
			sessionDebugReadbackCompletedAtMs,
			`${label} sessionDebugReadbackCompletedAtMs should not precede sessionDebugReadbackStartedAtMs`
		).toBeGreaterThanOrEqual(sessionDebugReadbackStartedAtMs);
		expect(
			sessionSelectedHourRangeScanStartedAtMs,
			`${label} sessionSelectedHourRangeScanStartedAtMs should not precede sessionDebugReadbackCompletedAtMs`
		).toBeGreaterThanOrEqual(sessionDebugReadbackCompletedAtMs);
		expect(
			sessionSelectedHourRangeScanCompletedAtMs,
			`${label} sessionSelectedHourRangeScanCompletedAtMs should not precede sessionSelectedHourRangeScanStartedAtMs`
		).toBeGreaterThanOrEqual(sessionSelectedHourRangeScanStartedAtMs);
		expect(
			sessionRangeResolveStartedAtMs,
			`${label} sessionRangeResolveStartedAtMs should not precede sessionSelectedHourRangeScanCompletedAtMs`
		).toBeGreaterThanOrEqual(sessionSelectedHourRangeScanCompletedAtMs);
		expect(
			sessionRangeResolveCompletedAtMs,
			`${label} sessionRangeResolveCompletedAtMs should not precede sessionRangeResolveStartedAtMs`
		).toBeGreaterThanOrEqual(sessionRangeResolveStartedAtMs);
		expect(
			sessionSelectedHourAnalysisBuildStartedAtMs,
			`${label} sessionSelectedHourAnalysisBuildStartedAtMs should not precede sessionRangeResolveCompletedAtMs`
		).toBeGreaterThanOrEqual(sessionRangeResolveCompletedAtMs);
		expect(
			sessionSelectedHourAnalysisBuildCompletedAtMs,
			`${label} sessionSelectedHourAnalysisBuildCompletedAtMs should not precede sessionSelectedHourAnalysisBuildStartedAtMs`
		).toBeGreaterThanOrEqual(sessionSelectedHourAnalysisBuildStartedAtMs);
		expect(
			sessionCpuFallbackSetupStartedAtMs,
			`${label} sessionCpuFallbackSetupStartedAtMs should not precede sessionSelectedHourAnalysisBuildCompletedAtMs`
		).toBeGreaterThanOrEqual(sessionSelectedHourAnalysisBuildCompletedAtMs);
		expect(
			sessionCpuFallbackSetupCompletedAtMs,
			`${label} sessionCpuFallbackSetupCompletedAtMs should not precede sessionCpuFallbackSetupStartedAtMs`
		).toBeGreaterThanOrEqual(sessionCpuFallbackSetupStartedAtMs);
		expect(
			sessionGpuResidentResultAssemblyStartedAtMs,
			`${label} sessionGpuResidentResultAssemblyStartedAtMs should not precede sessionCpuFallbackSetupCompletedAtMs`
		).toBeGreaterThanOrEqual(sessionCpuFallbackSetupCompletedAtMs);
		expect(
			sessionGpuResidentRangeResolveStartedAtMs,
			`${label} sessionGpuResidentRangeResolveStartedAtMs should not precede sessionGpuResidentResultAssemblyStartedAtMs`
		).toBeGreaterThanOrEqual(sessionGpuResidentResultAssemblyStartedAtMs);
		expect(
			sessionGpuResidentRangeResolveCompletedAtMs,
			`${label} sessionGpuResidentRangeResolveCompletedAtMs should not precede sessionGpuResidentRangeResolveStartedAtMs`
		).toBeGreaterThanOrEqual(sessionGpuResidentRangeResolveStartedAtMs);
		expect(
			sessionTooltipValuesHandoffStartedAtMs,
			`${label} sessionTooltipValuesHandoffStartedAtMs should not precede sessionGpuResidentRangeResolveCompletedAtMs`
		).toBeGreaterThanOrEqual(sessionGpuResidentRangeResolveCompletedAtMs);
		expect(
			sessionTooltipValuesHandoffCompletedAtMs,
			`${label} sessionTooltipValuesHandoffCompletedAtMs should not precede sessionTooltipValuesHandoffStartedAtMs`
		).toBeGreaterThanOrEqual(sessionTooltipValuesHandoffStartedAtMs);
		expect(
			sessionGpuResidentResultAssemblyCompletedAtMs,
			`${label} sessionGpuResidentResultAssemblyCompletedAtMs should not precede sessionTooltipValuesHandoffCompletedAtMs`
		).toBeGreaterThanOrEqual(sessionTooltipValuesHandoffCompletedAtMs);
		expect(
			sessionResultReturnedAtMs,
			`${label} sessionResultReturnedAtMs should not precede sessionResultReadyAtMs`
		).toBeGreaterThanOrEqual(sessionResultReadyAtMs);
		expect(
			sessionResultReadyAtMs,
			`${label} sessionResultReadyAtMs should not precede sessionGpuResidentResultAssemblyCompletedAtMs`
		).toBeGreaterThanOrEqual(sessionGpuResidentResultAssemblyCompletedAtMs);
		const controllerSessionRunStartedAtMs = requireTimelineNumber(
			timeline,
			'controllerSessionRunStartedAtMs',
			label
		);
		const controllerSessionRunCompletedAtMs = requireTimelineNumber(
			timeline,
			'controllerSessionRunCompletedAtMs',
			label
		);
		expect(
			controllerSessionRunCompletedAtMs,
			`${label} controllerSessionRunCompletedAtMs should not precede controllerSessionRunStartedAtMs`
		).toBeGreaterThanOrEqual(controllerSessionRunStartedAtMs);
		expect(
			sessionResultReturnedAtMs,
			`${label} sessionResultReturnedAtMs should not precede selectedHourValuePublicationStartedAtMs`
		).toBeGreaterThanOrEqual(selectedHourValuePublicationStartedAtMs);
		const controllerAcceptStartedAtMs = requireTimelineNumber(
			timeline,
			'controllerAcceptStartedAtMs',
			label
		);
		controllerDiagnosticsMergedAtMs = requireTimelineNumber(
			timeline,
			'controllerDiagnosticsMergedAtMs',
			label
		);
		expect(
			controllerAcceptStartedAtMs,
			`${label} controllerAcceptStartedAtMs should not precede sessionResultReturnedAtMs`
		).toBeGreaterThanOrEqual(sessionResultReturnedAtMs);
		expect(
			controllerDiagnosticsMergedAtMs,
			`${label} controllerDiagnosticsMergedAtMs should not precede controllerAcceptStartedAtMs`
		).toBeGreaterThanOrEqual(controllerAcceptStartedAtMs);
		const eagerAnalysisBuildMs =
			sessionSelectedHourAnalysisBuildCompletedAtMs -
			sessionSelectedHourAnalysisBuildStartedAtMs;
		expect(
			eagerAnalysisBuildMs,
			`${label} eager selected-hour analysis build should be below 5ms after supplied range optimization`
		).toBeLessThan(5);

		const gpuResidentRangeResolveMs =
			sessionGpuResidentRangeResolveCompletedAtMs -
			sessionGpuResidentRangeResolveStartedAtMs;
		expect(
			gpuResidentRangeResolveMs,
			`${label} GPU-resident range resolve should be below 5ms after range reuse`
		).toBeLessThan(5);
	}
	expect(
		selectedHourValuePublicationStartedAtMs,
		`${label} selectedHourValuePublicationStartedAtMs should not follow controllerAcceptedAtMs`
	).toBeLessThanOrEqual(controllerAcceptedAtMs);
	if (controllerDiagnosticsMergedAtMs != null) {
		expect(
			controllerDiagnosticsMergedAtMs,
			`${label} controllerDiagnosticsMergedAtMs should not precede controllerAcceptedAtMs`
		).toBeGreaterThanOrEqual(controllerAcceptedAtMs);
	}
	const controllerStatePublishedAtMs = requireTimelineNumber(
		timeline,
		'controllerStatePublishedAtMs',
		label
	);
	expect(
		controllerStatePublishedAtMs,
		`${label} controllerStatePublishedAtMs should not precede controllerAcceptedAtMs`
	).toBeGreaterThanOrEqual(controllerAcceptedAtMs);
	if (controllerDiagnosticsMergedAtMs != null) {
		expect(
			controllerStatePublishedAtMs,
			`${label} controllerStatePublishedAtMs should not precede controllerDiagnosticsMergedAtMs`
		).toBeGreaterThanOrEqual(controllerDiagnosticsMergedAtMs);
	}
	const routePendingSurfaceExposedAtMs = requireTimelineNumber(
		timeline,
		'routePendingSurfaceExposedAtMs',
		label
	);
	const routeProjectionEvaluationStartedAtMs = requireTimelineNumber(
		timeline,
		'routeProjectionEvaluationStartedAtMs',
		label
	);
	const routeProjectionEvaluationCompletedAtMs = requireTimelineNumber(
		timeline,
		'routeProjectionEvaluationCompletedAtMs',
		label
	);
	expect(
		routeProjectionEvaluationCompletedAtMs,
		`${label} routeProjectionEvaluationCompletedAtMs should not precede routeProjectionEvaluationStartedAtMs`
	).toBeGreaterThanOrEqual(routeProjectionEvaluationStartedAtMs);
	const routePublishedAtMs = requireTimelineNumber(
		timeline,
		'routePublishedAtMs',
		label
	);
	expect(
		timeline.renderLayoutReuseProofTrace,
		`${label} renderLayoutReuseProofTrace`
	).not.toBeNull();
	expect(
		timeline.renderLayoutReuseProofTrace?.decision,
		`${label} renderLayoutReuseProofTrace.decision`
	).toMatch(/^(reuse-safe|rebuild-required|proof-inconclusive)$/);
	expect(
		timeline.renderLayoutReuseProofTrace?.hoverCellLookupProofStatus,
		`${label} renderLayoutReuseProofTrace.hoverCellLookupProofStatus`
	).toMatch(/^(same-point-confirmed|not-compatible|proof-inconclusive)$/);
	expect(
		timeline.renderLayoutReuseProofTrace?.constructionMode,
		`${label} renderLayoutReuseProofTrace.constructionMode`
	).toMatch(/^(world-positions|metadata-bounds-fallback)$/);
	expect(
		timeline.renderLayoutReuseProofTrace?.normalizationSignature?.provenance,
		`${label} renderLayoutReuseProofTrace.normalizationSignature.provenance`
	).toMatch(/^(normalization-disabled|anchor-offset-minus-origin)$/);
	expect(
		timeline.renderLayoutReuseProofTrace?.normalizationSignature?.enabled,
		`${label} renderLayoutReuseProofTrace.normalizationSignature.enabled`
	).toEqual(expect.any(Boolean));
	expect(
		timeline.renderLayoutReuseProofTrace?.normalizationSignature?.offset,
		`${label} renderLayoutReuseProofTrace.normalizationSignature.offset`
	).toEqual({
		x: expect.any(Number),
		y: expect.any(Number),
		z: expect.any(Number)
	});
	for (const key of [
		'canonicalRuntimeCompatibilityWouldReuse',
		'hoverCellLookupProofStatus',
		'proofMatchesCanonicalRuntimeCompatibility',
		'normalizationSignatureMatch',
		'placementMatch',
		'cellToPointMappingMatch',
		'previousLayoutPresent'
	] as const) {
		expect(
			timeline.renderLayoutReuseProofTrace,
			`${label} renderLayoutReuseProofTrace.${key} should be present`
		).toHaveProperty(key);
	}
	expect(
		timeline.renderLayoutReuseProofTrace?.proofCostMs,
		`${label} renderLayoutReuseProofTrace.proofCostMs`
	).toEqual(expect.any(Number));
	expect(
		timeline.renderLayoutReuseProofTrace?.estimatedRetainedCpuLayoutBytes,
		`${label} renderLayoutReuseProofTrace.estimatedRetainedCpuLayoutBytes`
	).toEqual(expect.any(Number));
	expect(
		timeline.renderLayoutReuseAction,
		`${label} renderLayoutReuseAction`
	).toMatch(/^(reuse-candidate|build-required|reused)$/);
	expect(
		timeline.renderLayoutReuseReason,
		`${label} renderLayoutReuseReason`
	).toEqual(expect.any(String));
	expect(
		timeline.renderLayoutReuseDecisionMs,
		`${label} renderLayoutReuseDecisionMs`
	).toEqual(expect.any(Number));
	expect(
		timeline.renderLayoutReuseKeyMs,
		`${label} renderLayoutReuseKeyMs`
	).toEqual(expect.any(Number));
	expect(
		timeline.renderLayoutReuseSourceSignatureMs,
		`${label} renderLayoutReuseSourceSignatureMs`
	).toEqual(expect.any(Number));
	expect(
		timeline.renderLayoutReusePositionsSignatureMs,
		`${label} renderLayoutReusePositionsSignatureMs`
	).toEqual(expect.any(Number));
	expect(
		timeline.renderLayoutReusePositionsSignatureCacheHit,
		`${label} renderLayoutReusePositionsSignatureCacheHit`
	).toEqual(expect.any(Boolean));
	expect(
		timeline.renderLayoutReuseFrameCacheLookupMs,
		`${label} renderLayoutReuseFrameCacheLookupMs`
	).toEqual(expect.any(Number));
	expect(
		timeline.renderLayoutReuseFrameCacheHit,
		`${label} renderLayoutReuseFrameCacheHit`
	).toEqual(expect.any(Boolean));
	expect(
		timeline.renderLayoutReuseKeyMatch,
		`${label} renderLayoutReuseKeyMatch`
	).toEqual(expect.any(Boolean));
	expect(
		timeline.activeLayoutCandidateCount,
		`${label} activeLayoutCandidateCount`
	).toEqual(expect.any(Number));
	expect(
		timeline.activeLayoutCandidateCount ?? -1,
		`${label} activeLayoutCandidateCount should be nonnegative`
	).toBeGreaterThanOrEqual(0);
	if (timeline.renderLayoutReuseAction === 'reused') {
		expect(timeline.renderLayoutBuildTrace, `${label} renderLayoutBuildTrace`).toBeNull();
		expect(
			REUSED_LAYOUT_PROOF_SOURCES,
			`${label} renderLayoutReuseProofSource`
		).toContain(timeline.renderLayoutReuseProofSource);
		expect(
			timeline.renderLayoutReusePreviousKey,
			`${label} renderLayoutReusePreviousKey`
		).toEqual(expect.any(String));
		expect(
			timeline.renderLayoutReusePreviousRequestId,
			`${label} renderLayoutReusePreviousRequestId`
		).toEqual(expect.any(Number));
		expect(
			timeline.renderLayoutReusePreviousSelectionKey,
			`${label} renderLayoutReusePreviousSelectionKey`
		).toEqual(expect.any(String));
		expect(
			timeline.renderLayoutReuseFrameDerivationMs,
			`${label} renderLayoutReuseFrameDerivationMs`
		).toEqual(expect.any(Number));
	} else {
		expect(timeline.renderLayoutBuildTrace, `${label} renderLayoutBuildTrace`).not.toBeNull();
		expect(timeline.renderLayoutBuildTrace).toEqual({
			totalMs: expect.any(Number),
			arrayAllocationMs: expect.any(Number),
			transformBoundsPassMs: expect.any(Number),
			coordinateAssignmentMs: expect.any(Number),
			indexToTexelFillMs: expect.any(Number),
			cellToPointIndexBuildMs: expect.any(Number),
			colorBufferAllocationMs: expect.any(Number)
		});
		expect(
			timeline.renderLayoutReuseProofSource,
			`${label} renderLayoutReuseProofSource`
		).toBe('fresh-build-proof');
		expect(
			timeline.renderLayoutReusePreviousKey,
			`${label} renderLayoutReusePreviousKey`
		).toBeNull();
		expect(
			timeline.renderLayoutReusePreviousRequestId,
			`${label} renderLayoutReusePreviousRequestId`
		).toBeNull();
		expect(
			timeline.renderLayoutReusePreviousSelectionKey,
			`${label} renderLayoutReusePreviousSelectionKey`
		).toBeNull();
	}
	const routeProjectedAtMs = requireTimelineNumber(
		timeline,
		'routeProjectedAtMs',
		label
	);
	const sceneReactiveBlockEnteredAtMs = requireTimelineNumber(
		timeline,
		'sceneReactiveBlockEnteredAtMs',
		label
	);
	const sceneRenderStateResolvedAtMs = requireTimelineNumber(
		timeline,
		'sceneRenderStateResolvedAtMs',
		label
	);
	const sceneAcceptedKeyResolvedAtMs = requireTimelineNumber(
		timeline,
		'sceneAcceptedKeyResolvedAtMs',
		label
	);
	const sceneSyncInvocationQueuedAtMs = requireTimelineNumber(
		timeline,
		'sceneSyncInvocationQueuedAtMs',
		label
	);
	const sceneStartSyncEnteredAtMs = requireTimelineNumber(
		timeline,
		'sceneStartSyncEnteredAtMs',
		label
	);
	const sceneLayoutKeyStartedAtMs = requireTimelineNumber(
		timeline,
		'sceneLayoutKeyStartedAtMs',
		label
	);
	const sceneLayoutKeyCompletedAtMs = requireTimelineNumber(
		timeline,
		'sceneLayoutKeyCompletedAtMs',
		label
	);
	const scenePublicationPlanReadyAtMs = requireTimelineNumber(
		timeline,
		'scenePublicationPlanReadyAtMs',
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
	const controllerVisibleAcknowledgedAtMs = requireTimelineNumber(
		timeline,
		'controllerVisibleAcknowledgedAtMs',
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
	expect(
		sceneReactiveBlockEnteredAtMs,
		`${label} sceneReactiveBlockEnteredAtMs should not precede controllerAcceptedAtMs`
	).toBeGreaterThanOrEqual(controllerAcceptedAtMs);
	expect(
		sceneRenderStateResolvedAtMs,
		`${label} sceneRenderStateResolvedAtMs should not precede sceneReactiveBlockEnteredAtMs`
	).toBeGreaterThanOrEqual(sceneReactiveBlockEnteredAtMs);
	expect(
		sceneAcceptedKeyResolvedAtMs,
		`${label} sceneAcceptedKeyResolvedAtMs should not precede sceneRenderStateResolvedAtMs`
	).toBeGreaterThanOrEqual(sceneRenderStateResolvedAtMs);
	expect(
		sceneSyncInvocationQueuedAtMs,
		`${label} sceneSyncInvocationQueuedAtMs should not precede sceneAcceptedKeyResolvedAtMs`
	).toBeGreaterThanOrEqual(sceneAcceptedKeyResolvedAtMs);
	expect(
		sceneStartSyncEnteredAtMs,
		`${label} sceneStartSyncEnteredAtMs should not precede sceneSyncInvocationQueuedAtMs`
	).toBeGreaterThanOrEqual(sceneSyncInvocationQueuedAtMs);
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
		expect(
			sceneLayoutKeyStartedAtMs,
			`${label} sceneLayoutKeyStartedAtMs should not precede sceneSyncAttemptStartedAtMs`
		).toBeGreaterThanOrEqual(sceneSyncAttemptStartedAtMs);
	}
	expect(
		sceneLayoutKeyCompletedAtMs,
		`${label} sceneLayoutKeyCompletedAtMs should not precede sceneLayoutKeyStartedAtMs`
	).toBeGreaterThanOrEqual(sceneLayoutKeyStartedAtMs);
	expect(
		scenePublicationPlanReadyAtMs,
		`${label} scenePublicationPlanReadyAtMs should not precede sceneLayoutKeyCompletedAtMs`
	).toBeGreaterThanOrEqual(sceneLayoutKeyCompletedAtMs);
	if (timeline.renderLayoutPublicationPlanMs != null) {
		expect(
			timeline.renderLayoutPublicationPlanMs,
			`${label} renderLayoutPublicationPlanMs should be nonnegative`
		).toBeGreaterThanOrEqual(0);
		expect(
			timeline.renderLayoutCompatibilityMs,
			`${label} renderLayoutCompatibilityMs`
		).toEqual(expect.any(Number));
		expect(
			timeline.renderLayoutCompatibilityMs ?? -1,
			`${label} renderLayoutCompatibilityMs should be nonnegative`
		).toBeGreaterThanOrEqual(0);
		expect(
			timeline.renderLayoutReuseProofMs,
			`${label} renderLayoutReuseProofMs`
		).toEqual(expect.any(Number));
		expect(
			timeline.renderLayoutReuseProofMs ?? -1,
			`${label} renderLayoutReuseProofMs should be nonnegative`
		).toBeGreaterThanOrEqual(0);
		if (timeline.renderLayoutCompatibilityRequiredExpensiveMappingComparison != null) {
			expect(
				timeline.renderLayoutCompatibilityRequiredExpensiveMappingComparison,
				`${label} renderLayoutCompatibilityRequiredExpensiveMappingComparison`
			).toEqual(expect.any(Boolean));
		}
		if (timeline.renderLayoutCompatibilityPerformedExpensiveMappingComparison != null) {
			expect(
				timeline.renderLayoutCompatibilityPerformedExpensiveMappingComparison,
				`${label} renderLayoutCompatibilityPerformedExpensiveMappingComparison`
			).toEqual(expect.any(Boolean));
		}
		expect(
			(timeline.renderLayoutPublicationPlanMs ?? 0) +
				(timeline.renderLayoutCompatibilityMs ?? 0) +
				(timeline.renderLayoutReuseProofMs ?? 0),
			`${label} post-key planning split should fit within key-complete to plan-ready`
		).toBeLessThanOrEqual(scenePublicationPlanReadyAtMs - sceneLayoutKeyCompletedAtMs + 1);
	}
	expect(
		controllerVisibleAcknowledgedAtMs,
		`${label} controllerVisibleAcknowledgedAtMs should not precede controllerAcceptedAtMs`
	).toBeGreaterThanOrEqual(controllerAcceptedAtMs);
	expect(
		controllerVisibleAcknowledgedAtMs,
		`${label} controllerVisibleAcknowledgedAtMs should not precede selectedHourValuePublicationStartedAtMs`
	).toBeGreaterThanOrEqual(selectedHourValuePublicationStartedAtMs);
	const publicationEffectStartedAtMs = requireTimelineNumber(
		timeline,
		'publicationEffectStartedAtMs',
		label
	);
	if (sceneSyncAttemptStartedAtMs != null) {
		expect(
			publicationEffectStartedAtMs,
			`${label} publicationEffectStartedAtMs should not precede sceneSyncAttemptStartedAtMs`
		).toBeGreaterThanOrEqual(sceneSyncAttemptStartedAtMs);
		expect(
			sceneLayoutKeyStartedAtMs,
			`${label} sceneLayoutKeyStartedAtMs should not precede publicationEffectStartedAtMs`
		).toBeGreaterThanOrEqual(publicationEffectStartedAtMs);
	}
	const sceneSurfacePendingStorageInitAtMs = requireTimelineNumber(
		timeline,
		'sceneSurfacePendingStorageInitAtMs',
		label
	);
	expect(
		sceneSurfacePendingStorageInitAtMs,
		`${label} sceneSurfacePendingStorageInitAtMs should not precede scenePublicationPlanReadyAtMs`
	).toBeGreaterThanOrEqual(scenePublicationPlanReadyAtMs);
	const renderStorageWaitStartedAtMs = requireTimelineNumber(
		timeline,
		'renderStorageWaitStartedAtMs',
		label
	);
	expect(
		renderStorageWaitStartedAtMs,
		`${label} renderStorageWaitStartedAtMs should not precede sceneSurfacePendingStorageInitAtMs`
	).toBeGreaterThanOrEqual(sceneSurfacePendingStorageInitAtMs);
	expect(
		timeline.renderStoragePreWaitMs,
		`${label} renderStoragePreWaitMs`
	).toEqual(expect.any(Number));
	expect(
		timeline.renderStoragePreWaitMs ?? -1,
		`${label} renderStoragePreWaitMs should be nonnegative`
	).toBeGreaterThanOrEqual(0);
	if (sceneSyncAttemptStartedAtMs != null) {
		expect(
			Math.abs(
				(timeline.renderStoragePreWaitMs ?? 0) -
					(renderStorageWaitStartedAtMs - sceneSyncAttemptStartedAtMs)
			),
			`${label} renderStoragePreWaitMs should match waitStarted - sceneSyncAttemptStarted`
		).toBeLessThanOrEqual(1);
	}
	expectTimelineOrder(
		timeline,
		[
			'sceneSurfaceReceivedAtMs',
			'sceneLayoutKeyStartedAtMs',
			'sceneLayoutKeyCompletedAtMs',
			'scenePublicationPlanReadyAtMs',
			'sceneSurfacePendingStorageInitAtMs',
			'renderStorageWaitStartedAtMs',
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
	expect(
		timeline.renderStorageWaitStartedAtMs,
		`${label} renderStorageWaitStartedAtMs`
	).toBe(waitStartedAtMs);
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
		'updateComputeBufferSurfaceRangeUniformMs',
		'updateComputeBufferSurfacePendingSourceMs',
		'updateComputeBufferSurfaceLayoutUserDataMs',
		'updateComputeBufferSurfaceByteAccountingMs',
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
		expect(trace.updateComputeBufferSurfaceRangeUniformMs).toEqual(
			expect.any(Number)
		);
		expect(trace.updateComputeBufferSurfacePendingSourceMs).toEqual(
			expect.any(Number)
		);
		expect(trace.updateComputeBufferSurfaceLayoutUserDataMs).toEqual(
			expect.any(Number)
		);
		expect(trace.updateComputeBufferSurfaceByteAccountingMs).toEqual(
			expect.any(Number)
		);
		expect(trace.setCreatedSurfacePendingStorageInitMs).toBeNull();
		expect(trace.setPostSurfacePendingStorageInitMs).toEqual(expect.any(Number));
	}
	if (trace.action === 'update-failed-created') {
		expect(trace.recreateDecision.missingSurface).toBe(false);
		expect(trace.recreateDecision.notComputeBufferSurface).toBe(false);
		expect(trace.recreateDecision.analysisIdentityChanged).toBe(false);
		expect(trace.recreateDecision.layoutCompatible).toBe(false);
		expect(trace.updateComputeBufferSurfaceMeshMs).toEqual(expect.any(Number));
		expect(trace.updateComputeBufferSurfaceRangeUniformMs).toEqual(
			expect.any(Number)
		);
		expect(trace.updateComputeBufferSurfacePendingSourceMs).toEqual(
			expect.any(Number)
		);
		expect(trace.updateComputeBufferSurfaceLayoutUserDataMs).toEqual(
			expect.any(Number)
		);
		expect(trace.updateComputeBufferSurfaceByteAccountingMs).toEqual(
			expect.any(Number)
		);
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
			const sample = collectedCase.modes[colorMode][phase];
			const label = `${collectedCase.projectLabel} ${colorMode}.${phase}`;
			expectValidRenderPublication(sample, label);
			if (phase === 'scrub') {
				expect(
					sample.timings.renderPublication?.renderPublicationTimeline
						?.renderLayoutReuseAction,
					`${label} scrub action should report actual layout reuse`
				).toBe('reused');
				expect(
					sample.timings.renderPublication?.renderPublicationTimeline
						?.renderLayoutReuseReason,
					`${label} scrub reuse reason`
				).toBe('reuse-safe');
				expect(
					sample.timings.renderPublication?.renderPublicationTimeline
						?.renderLayoutBuildTrace,
					`${label} scrub should skip layout rebuild`
				).toBeNull();
				expect(
					REUSED_LAYOUT_PROOF_SOURCES,
					`${label} scrub should use a safe reused-layout proof`
				).toContain(
					sample.timings.renderPublication?.renderPublicationTimeline
						?.renderLayoutReuseProofSource
				);
				expect(
					sample.timings.renderPublication?.renderPublicationTimeline
						?.renderLayoutReusePreviousKey,
					`${label} scrub previous key linkage`
				).toEqual(expect.any(String));
				expect(
					sample.timings.renderPublication?.renderPublicationTimeline
						?.renderLayoutReusePreviousRequestId,
					`${label} scrub previous request linkage`
				).toEqual(expect.any(Number));
				expect(
					sample.timings.renderPublication?.renderPublicationTimeline
						?.renderLayoutReusePreviousSelectionKey,
					`${label} scrub previous selection linkage`
				).toEqual(expect.any(String));
				expect(
					sample.timings.renderPublication?.renderPublicationTimeline
						?.activeLayoutCandidateCount,
					`${label} scrub active layout candidate count`
				).toBe(1);
				expect(
					sample.timings.renderPublication?.renderPublicationTimeline
						?.renderLayoutReuseFrameCacheHit,
					`${label} renderLayoutReuseFrameCacheHit`
				).toBe(true);
				expect(
					sample.timings.renderPublication?.renderPublicationTimeline
						?.renderLayoutReuseFrameCacheKind,
					`${label} renderLayoutReuseFrameCacheKind`
				).toBe('structural');
				expect(
					sample.timings.renderPublication?.renderPublicationTimeline
						?.renderLayoutReuseFrameDerivationMs ?? 0,
					`${label} renderLayoutReuseFrameDerivationMs should be zero on cache hit`
				).toBe(0);
			}
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
	const dataTextureBuildCount = numberOrNull(diagnostics.dataTextureBuildCount);
	expect(
		dataTextureBuildCount,
		`${phase} ${colorMode} dataTextureBuildCount diagnostic`
	).toEqual(expect.any(Number));
	expect(
		dataTextureBuildCount,
		`${phase} ${colorMode} dataTextureBuildCount regressed before sample build`
	).toBe(0);
	if (typeof dataTextureBuildCount !== 'number') {
		throw new Error(`${phase} ${colorMode} missing dataTextureBuildCount diagnostic`);
	}
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
			dataTextureBuildCount,
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
	const warmupScrubDiagnostics = await waitForSelectedHourPublication(page, {
		expectedSelectionKey: expectedSelectionKey(caseConfig, WARMUP_SCRUB_HOUR_INDEX),
		colorMode,
		minSurfaceRequestId: initialRequestId
	});
	const warmupRequestId = warmupScrubDiagnostics.baseSurfaceRequestId ?? initialRequestId;
	await page.getByRole('slider', { name: 'Select analysis hour' }).press('ArrowRight');
	const scrubDiagnostics = await waitForSelectedHourPublication(page, {
		expectedSelectionKey: expectedSelectionKey(caseConfig, SCRUB_HOUR_INDEX),
		colorMode,
		minSurfaceRequestId: warmupRequestId
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
					'App-visible keyboard scrub on the main-route Select analysis hour slider from hour 0 to hour 2, with hour 1 as the in-session warmup scrub that establishes previous-publication proof; no debug route, parity, or .bin path.',
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

async function verifyRouteReuseAndRebuildHoverTruth(page: Page) {
	const bgCase = CASES.find((entry) => entry.projectLabel === 'Ben-Gurion');
	if (!bgCase) {
		throw new Error('Expected the Ben-Gurion collector case.');
	}

	const bgUrl = `/?analysis=${encodeURIComponent(bgCase.analysisId)}&gridResolution=${TARGET_GRID_RESOLUTION_METERS}&utciRender=auto&utciRenderDiagnostics=1`;
	await page.goto(bgUrl);
	const bgInitialDiagnostics = await waitForSelectedHourPublication(page, {
		expectedSelectionKey: expectedSelectionKey(bgCase, 0),
		colorMode: 'normalized'
	});
	const bgInitialRenderPublication = extractTimings(bgInitialDiagnostics.timings).renderPublication;
	expect(
		bgInitialRenderPublication?.renderPublicationTimeline?.renderLayoutReuseAction,
		'BG initial should build a fresh layout'
	).toBe('build-required');
	const bgInitialRequestId = bgInitialDiagnostics.baseSurfaceRequestId ?? 0;
	await page.getByRole('slider', { name: 'Select analysis hour' }).press('ArrowRight');
	const bgWarmupDiagnostics = await waitForSelectedHourPublication(page, {
		expectedSelectionKey: expectedSelectionKey(bgCase, WARMUP_SCRUB_HOUR_INDEX),
		colorMode: 'normalized',
		minSurfaceRequestId: bgInitialRequestId
	});
	const bgWarmupTimeline =
		extractTimings(bgWarmupDiagnostics.timings).renderPublication?.renderPublicationTimeline;
	expect(
		bgWarmupTimeline?.renderLayoutReuseAction,
		'BG warmup scrub should build while establishing prior proof'
	).toBe('build-required');
	const bgWarmupRequestId = bgWarmupDiagnostics.baseSurfaceRequestId ?? bgInitialRequestId;
	await page.getByRole('slider', { name: 'Select analysis hour' }).press('ArrowRight');
	const bgScrubDiagnostics = await waitForSelectedHourPublication(page, {
		expectedSelectionKey: expectedSelectionKey(bgCase, SCRUB_HOUR_INDEX),
		colorMode: 'normalized',
		minSurfaceRequestId: bgWarmupRequestId
	});
	const bgScrubTimeline = extractTimings(bgScrubDiagnostics.timings).renderPublication
		?.renderPublicationTimeline;
	expect(bgScrubTimeline?.renderLayoutReuseAction, 'BG scrub should reuse layout').toBe(
		'reused'
	);
	expect(bgScrubTimeline?.renderLayoutReuseReason, 'BG scrub reuse reason').toBe(
		'reuse-safe'
	);
	expect(bgScrubTimeline?.renderLayoutBuildTrace, 'BG scrub build trace').toBeNull();
	expect(
		REUSED_LAYOUT_PROOF_SOURCES,
		'BG scrub proof source'
	).toContain(bgScrubTimeline?.renderLayoutReuseProofSource);
	expect(bgScrubTimeline?.renderLayoutReusePreviousKey).toEqual(expect.any(String));
	expect(bgScrubTimeline?.renderLayoutReusePreviousRequestId).toEqual(expect.any(Number));
	expect(bgScrubTimeline?.renderLayoutReusePreviousSelectionKey).toEqual(
		expect.any(String)
	);
	expect(bgScrubTimeline?.activeLayoutCandidateCount).toBe(1);
	const bgHover = await expectTooltipProbeMatch(page, 'BG scrub reused publication');

	await requestDiagnosticsGridResolutionChange(page, 2);
	const rebuiltDiagnostics = await waitForSelectedHourPublication(page, {
		expectedSelectionKey: expectedSelectionKey(bgCase, SCRUB_HOUR_INDEX),
		colorMode: 'normalized',
		expectedGridResolution: 2
	});
	const rebuiltTimeline = extractTimings(rebuiltDiagnostics.timings).renderPublication
		?.renderPublicationTimeline;
	expect(
		rebuiltTimeline?.renderLayoutReuseAction,
		'BG in-session grid change should rebuild with prior active layout state present'
	).toBe('build-required');
	expect(
		rebuiltTimeline?.renderLayoutBuildTrace,
		'BG in-session grid change should include a fresh build trace'
	).not.toBeNull();
	expect(
		rebuiltTimeline?.renderLayoutReuseReason,
		'BG in-session grid change should fail reuse on key mismatch'
	).toBe('layout-key-mismatch');
	expect(
		rebuiltTimeline?.renderLayoutReuseKeyMatch,
		'BG in-session grid change should report the layout key mismatch explicitly'
	).toBe(false);
	expect(
		rebuiltTimeline?.renderLayoutReuseProofTrace?.previousLayoutPresent,
		'BG in-session grid change should compare against the prior active layout'
	).toBe(true);
	expect(
		rebuiltTimeline?.renderLayoutReuseProofTrace?.gridSizeMatch,
		'BG in-session grid change should prove the grid size changed'
	).toBe(false);
	expect(
		rebuiltTimeline?.renderLayoutReusePreviousKey,
		'BG in-session grid change should retain previous layout candidate linkage'
	).toEqual(expect.any(String));
	expect(
		rebuiltTimeline?.activeLayoutCandidateCount,
		'BG in-session grid change active layout candidate count'
	).toBe(1);
	expect(
		rebuiltTimeline?.renderLayoutReuseProofSource,
		'BG in-session grid change proof source'
	).toBe('fresh-build-proof');
	const rebuiltHover = await expectTooltipProbeMatch(
		page,
		'BG rebuilt publication after in-session grid change'
	);

	expect(
		rebuiltDiagnostics.baseSelectionKey,
		'grid change should keep the selected hour while rebuilding the surface'
	).toBe(bgScrubDiagnostics.baseSelectionKey);
	expect(
		rebuiltDiagnostics.baseMetadataGridSize,
		'grid change should publish the requested grid resolution'
	).toBe(2);
	expect(
		`${rebuiltHover.liveTooltip.positionIndex}|${rebuiltHover.liveTooltip.value.toFixed(3)}|${formatTooltipPosition(rebuiltHover.liveTooltip.position)}`,
		'in-session grid change should alter hover truth'
	).not.toBe(
		`${bgHover.liveTooltip.positionIndex}|${bgHover.liveTooltip.value.toFixed(3)}|${formatTooltipPosition(bgHover.liveTooltip.position)}`
	);
}

function buildRepeatedScrubSoakSample(
	diagnostics: DiagnosticsSnapshot,
	hoverProbe: MainRouteTooltipProbe
): RepeatedScrubSoakSample {
	const timeline = extractTimings(diagnostics.timings).renderPublication?.renderPublicationTimeline;
	return {
		hourIndex: diagnostics.baseSelectedHourIndex,
		selectionKey: diagnostics.baseSelectionKey ?? null,
		surfaceRequestId: diagnostics.baseSurfaceRequestId ?? null,
		renderLayoutReuseAction: timeline?.renderLayoutReuseAction ?? null,
		renderLayoutReuseReason: timeline?.renderLayoutReuseReason ?? null,
		activeLayoutCandidateCount: timeline?.activeLayoutCandidateCount ?? null,
		reusedLayoutIdentity: deriveReusedLayoutIdentity(timeline),
		retainedCpuLayoutBytes:
			timeline?.renderLayoutReuseProofTrace?.estimatedRetainedCpuLayoutBytes ?? null,
		ownedGpuMemoryBytes:
			diagnostics.trackedGpuAllocationBytes.persistentExposureBytes +
			diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes +
			diagnostics.trackedGpuAllocationBytes.renderOwnedSelectedHourBytes,
		renderOwnedSelectedHourBytes:
			diagnostics.trackedGpuAllocationBytes.renderOwnedSelectedHourBytes,
		renderOwnedSelectedHourBytesHighWatermark:
			diagnostics.trackedGpuAllocationBytes.renderOwnedSelectedHourBytesHighWatermark,
		hoverCellLookupProofStatus:
			timeline?.renderLayoutReuseProofTrace?.hoverCellLookupProofStatus ?? null,
		hoverProbe: {
			positionIndex: hoverProbe.positionIndex,
			value: hoverProbe.value,
			position: hoverProbe.position
		}
	};
}

async function collectRepeatedScrubSoak(
	page: Page,
	caseConfig: AnalysisCase
): Promise<RepeatedScrubSoakResult> {
	const sourceUrl = `/?analysis=${encodeURIComponent(caseConfig.analysisId)}&gridResolution=${TARGET_GRID_RESOLUTION_METERS}&utciRender=auto&utciRenderDiagnostics=1`;
	await page.goto(sourceUrl);

	const initialDiagnostics = await waitForSelectedHourPublication(page, {
		expectedSelectionKey: expectedSelectionKey(caseConfig, 0),
		colorMode: 'normalized'
	});
	const initialRequestId = initialDiagnostics.baseSurfaceRequestId ?? 0;
	await page.getByRole('slider', { name: 'Select analysis hour' }).press('ArrowRight');
	const warmupDiagnostics = await waitForSelectedHourPublication(page, {
		expectedSelectionKey: expectedSelectionKey(caseConfig, WARMUP_SCRUB_HOUR_INDEX),
		colorMode: 'normalized',
		minSurfaceRequestId: initialRequestId
	});
	const warmupTimeline =
		extractTimings(warmupDiagnostics.timings).renderPublication?.renderPublicationTimeline;
	expect(warmupTimeline?.renderLayoutReuseAction, 'Repeated scrub warmup should build').toBe(
		'build-required'
	);
	let previousRequestId = warmupDiagnostics.baseSurfaceRequestId ?? initialRequestId;

	const reusedSamples: RepeatedScrubSoakSample[] = [];
	for (const hourIndex of REPEATED_SCRUB_HOUR_INDICES) {
		await page.getByRole('slider', { name: 'Select analysis hour' }).press('ArrowRight');
		const diagnostics = await waitForSelectedHourPublication(page, {
			expectedSelectionKey: expectedSelectionKey(caseConfig, hourIndex),
			colorMode: 'normalized',
			minSurfaceRequestId: previousRequestId
		});
		const timeline =
			extractTimings(diagnostics.timings).renderPublication?.renderPublicationTimeline;
		expect(
			timeline?.renderLayoutReuseAction,
			`Repeated scrub hour ${hourIndex} should reuse layout`
		).toBe('reused');
		expect(
			timeline?.renderLayoutReuseReason,
			`Repeated scrub hour ${hourIndex} reuse reason`
		).toBe('reuse-safe');
		expect(
			timeline?.activeLayoutCandidateCount,
			`Repeated scrub hour ${hourIndex} active layout count`
		).toBe(1);
		expect(
			timeline?.renderLayoutReuseProofTrace?.hoverCellLookupProofStatus,
			`Repeated scrub hour ${hourIndex} hover/cell proof`
		).toBe('same-point-confirmed');
		expect(
			deriveReusedLayoutIdentity(timeline),
			`Repeated scrub hour ${hourIndex} reused layout identity`
		).toEqual(expect.any(String));
		const hovered = await expectTooltipProbeMatch(
			page,
			`Repeated scrub hour ${hourIndex} reused publication`
		);
		reusedSamples.push(buildRepeatedScrubSoakSample(diagnostics, hovered.liveTooltip));
		previousRequestId = diagnostics.baseSurfaceRequestId ?? previousRequestId;
	}

	const stableReusedLayoutIdentity = reusedSamples[0]?.reusedLayoutIdentity ?? null;
	for (const sample of reusedSamples) {
		expect(sample.reusedLayoutIdentity, 'Repeated scrub reused layout identity should plateau').toBe(
			stableReusedLayoutIdentity
		);
		expect(sample.retainedCpuLayoutBytes, 'Repeated scrub retained bytes should plateau').toBe(
			reusedSamples[0]?.retainedCpuLayoutBytes ?? null
		);
		expect(sample.ownedGpuMemoryBytes, 'Repeated scrub app-owned bytes should plateau').toBe(
			reusedSamples[0]?.ownedGpuMemoryBytes ?? 0
		);
		expect(
			sample.renderOwnedSelectedHourBytes,
			'Repeated scrub render-owned selected-hour bytes should plateau'
		).toBe(reusedSamples[0]?.renderOwnedSelectedHourBytes ?? 0);
		expect(
			sample.renderOwnedSelectedHourBytesHighWatermark,
			'Repeated scrub render-owned selected-hour high-watermark should plateau'
		).toBe(reusedSamples[0]?.renderOwnedSelectedHourBytesHighWatermark ?? 0);
	}

	await requestDiagnosticsGridResolutionChange(page, 2);
	const rebuildDiagnostics = await waitForSelectedHourPublication(page, {
		expectedSelectionKey: expectedSelectionKey(
			caseConfig,
			REPEATED_SCRUB_HOUR_INDICES[REPEATED_SCRUB_HOUR_INDICES.length - 1]
		),
		colorMode: 'normalized',
		expectedGridResolution: 2
	});
	const rebuildTimeline =
		extractTimings(rebuildDiagnostics.timings).renderPublication?.renderPublicationTimeline;
	expect(
		rebuildTimeline?.renderLayoutReuseAction,
		'Repeated scrub rebuild replacement should rebuild layout'
	).toBe('build-required');
	expect(
		rebuildTimeline?.renderLayoutReuseReason,
		'Repeated scrub rebuild replacement should report key mismatch'
	).toBe('layout-key-mismatch');
	expect(
		rebuildTimeline?.activeLayoutCandidateCount,
		'Repeated scrub rebuild replacement active layout count'
	).toBe(1);
	expect(
		rebuildTimeline?.renderLayoutReusePreviousKey,
		'Repeated scrub rebuild replacement should stamp the released previous layout'
	).toBe(stableReusedLayoutIdentity);

	await page.goto('about:blank');

	return {
		projectLabel: caseConfig.projectLabel,
		analysisId: caseConfig.analysisId,
		colorMode: 'normalized',
		query: 'gridResolution=0.5&utciRender=auto&utciRenderDiagnostics=1',
		warmupHourIndex: WARMUP_SCRUB_HOUR_INDEX,
		reusedSamples,
		plateauRetainedCpuLayoutBytes: reusedSamples[0]?.retainedCpuLayoutBytes ?? null,
		plateauOwnedGpuMemoryBytes: reusedSamples[0]?.ownedGpuMemoryBytes ?? 0,
		plateauRenderOwnedSelectedHourBytes:
			reusedSamples[0]?.renderOwnedSelectedHourBytes ?? 0,
		plateauRenderOwnedSelectedHourBytesHighWatermark:
			reusedSamples[0]?.renderOwnedSelectedHourBytesHighWatermark ?? 0,
		stableReusedLayoutIdentity,
		rebuildReplacement: {
			hourIndex: rebuildDiagnostics.baseSelectedHourIndex,
			selectionKey: rebuildDiagnostics.baseSelectionKey ?? null,
			surfaceRequestId: rebuildDiagnostics.baseSurfaceRequestId ?? null,
			renderLayoutReuseAction: rebuildTimeline?.renderLayoutReuseAction ?? null,
			renderLayoutReuseReason: rebuildTimeline?.renderLayoutReuseReason ?? null,
			activeLayoutCandidateCount: rebuildTimeline?.activeLayoutCandidateCount ?? null,
			releasedPreviousLayout: rebuildTimeline?.renderLayoutReusePreviousKey ?? null,
			ownedGpuMemoryBytes:
				rebuildDiagnostics.trackedGpuAllocationBytes.persistentExposureBytes +
				rebuildDiagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes +
				rebuildDiagnostics.trackedGpuAllocationBytes.renderOwnedSelectedHourBytes,
			renderOwnedSelectedHourBytes:
				rebuildDiagnostics.trackedGpuAllocationBytes.renderOwnedSelectedHourBytes,
			renderOwnedSelectedHourBytesHighWatermark:
				rebuildDiagnostics.trackedGpuAllocationBytes
					.renderOwnedSelectedHourBytesHighWatermark
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
		await verifyRouteReuseAndRebuildHoverTruth(page);
		const nessTzionaCase = CASES.find((entry) => entry.projectLabel === 'Ness-Tziona');
		if (!nessTzionaCase) {
			throw new Error('Expected the Ness Tziona collector case.');
		}
		const repeatedScrubSoak = await collectRepeatedScrubSoak(page, nessTzionaCase);

		const artifact = {
			collectedOn: COLLECTED_ON,
			sourceRoute: SOURCE_ROUTE,
			targetGridResolutionMeters: TARGET_GRID_RESOLUTION_METERS,
			includedAnalyses: CASES.map((entry) => entry.analysisId),
			excludedBgVariantsExplanation:
				'This 0.5m stress pass intentionally excludes other Ben-Gurion variants and uses only the BG base case plus Ness Tziona base/exploded model.',
			collectionMethod:
				'Main route only: / with gridResolution=0.5&utciRenderDiagnostics=1, app-visible color-mode buttons and hour slider scrub, no debug route and no parity/.bin comparison.',
			cases,
			repeatedScrubSoak
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
