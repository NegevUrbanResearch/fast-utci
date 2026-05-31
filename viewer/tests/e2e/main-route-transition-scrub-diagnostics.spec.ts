import { expect, test, type Page } from '@playwright/test';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const RESULTS_DIR = resolve(REPO_ROOT, 'data/performance-results');
const ARTIFACT_FILENAME = 'main-route-transition-scrub-diagnostics.json';
const ARTIFACT_PATH = resolve(RESULTS_DIR, ARTIFACT_FILENAME);
const PROGRESS_ARTIFACT_FILENAME = 'main-route-transition-scrub-diagnostics-progress.json';
const PROGRESS_ARTIFACT_PATH = resolve(RESULTS_DIR, PROGRESS_ARTIFACT_FILENAME);
const SOURCE_ROUTE = '/';
const NZ_ANALYSIS_ID = 'Ness-Tziona/exploded/nes_tziona_unblock_2';
const BG_ANALYSIS_ID = 'Ben-Gurion/20250815_grid_2m_fullday';
const TARGET_GRID_RESOLUTION_METERS = 0.5;
const BG_ENTRY_GRID_RESOLUTION_METERS = 2;
const INITIAL_MONTH_INDEX = 7;
const HOUR_SEQUENCE = [0, 1, 2, 3] as const;
const MONTH_SEQUENCE = [8, 0, 7] as const;

type CollectorCase = {
	caseId: string;
	entry: 'direct-nz' | 'bg-then-nz';
	analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2';
	gridResolutionMeters: 0.5;
	queryParams?: Record<string, string>;
	expectedInitialSelectionKey: string;
};

type ActionKind = 'initial-visible' | 'hour-scrub' | 'month-change';
type DiagnosticsSnapshot = Record<string, any>;
type RenderPublicationSnapshot = Record<string, unknown> & {
	renderPublicationTimeline?: Record<string, unknown> | null;
};

const CASES: CollectorCase[] = [
	{
		caseId: 'direct-nz-0_5m-single-submit',
		entry: 'direct-nz',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		gridResolutionMeters: 0.5,
		expectedInitialSelectionKey: `${NZ_ANALYSIS_ID}|7|0`
	},
	{
		caseId: 'bg-then-nz-0_5m-single-submit',
		entry: 'bg-then-nz',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		gridResolutionMeters: 0.5,
		expectedInitialSelectionKey: `${NZ_ANALYSIS_ID}|7|0`
	},
	{
		caseId: 'direct-nz-0_5m-chunked-2048',
		entry: 'direct-nz',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		gridResolutionMeters: 0.5,
		queryParams: {
			utciExposureSchedule: 'chunked',
			utciExposureMaxWorkgroupsPerSlice: '2048'
		},
		expectedInitialSelectionKey: `${NZ_ANALYSIS_ID}|7|0`
	},
	{
		caseId: 'bg-then-nz-0_5m-chunked-2048',
		entry: 'bg-then-nz',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		gridResolutionMeters: 0.5,
		queryParams: {
			utciExposureSchedule: 'chunked',
			utciExposureMaxWorkgroupsPerSlice: '2048'
		},
		expectedInitialSelectionKey: `${NZ_ANALYSIS_ID}|7|0`
	}
];

function formatLocalDate(date: Date) {
	const year = date.getFullYear();
	const month = String(date.getMonth() + 1).padStart(2, '0');
	const day = String(date.getDate()).padStart(2, '0');
	return `${year}-${month}-${day}`;
}

function selectionKey(analysisId: string, monthIndex: number, hourIndex: number) {
	return `${analysisId}|${monthIndex}|${hourIndex}`;
}

function buildMainRouteUrl(
	analysisId: string,
	caseConfig: CollectorCase,
	gridResolutionMeters: number = caseConfig.gridResolutionMeters
) {
	const params = new URLSearchParams({
		analysis: analysisId,
		gridResolution: String(gridResolutionMeters),
		utciRender: 'auto',
		utciRenderDiagnostics: '1'
	});
	for (const [key, value] of Object.entries(caseConfig.queryParams ?? {})) {
		params.set(key, value);
	}
	return `${SOURCE_ROUTE}?${params.toString()}`;
}

function writeJsonArtifact(path: string, value: unknown) {
	if (!existsSync(RESULTS_DIR)) {
		mkdirSync(RESULTS_DIR, { recursive: true });
	}
	writeFileSync(path, JSON.stringify(value, null, 2), 'utf8');
}

function logCollectorProgress(message: string) {
	console.log(`[transition-scrub-diagnostics] ${message}`);
}

function isForbiddenComparisonRequest(url: string) {
	const parsed = new URL(url);
	const isMainRouteDocument =
		parsed.pathname === SOURCE_ROUTE && parsed.searchParams.has('utciRenderDiagnostics');
	if (/\.bin(\?|$)/i.test(url)) return true;
	if (/loadReferenceFromFs/i.test(url)) return true;
	if (/parity/i.test(url) && !isMainRouteDocument) return true;
	return false;
}

function collectForbiddenComparisonFields(diagnostics: DiagnosticsSnapshot) {
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

function exposureSchedulerModeOrNull(value: unknown): 'single-submit' | 'chunked' | null {
	return value === 'single-submit' || value === 'chunked' ? value : null;
}

async function readUtciRenderDiagnostics(page: Page) {
	return page.evaluate(() => (window as any).__utciRenderDiagnostics__ ?? null);
}

async function waitForSelectedHourPublication(
	page: Page,
	expectedSelectionKey: string,
	options?: { minSurfaceRequestId?: number; expectedGridResolutionMeters?: number }
) {
	logCollectorProgress(
		`waiting for ${expectedSelectionKey} request>${options?.minSurfaceRequestId ?? 0} grid=${
			options?.expectedGridResolutionMeters ?? 'any'
		}`
	);
	const diagnostics = await page.waitForFunction(
		({ selectionKey: expectedSelectionKey, minSurfaceRequestId, expectedGridResolutionMeters }) => {
			const value = (window as any).__utciRenderDiagnostics__;
			if (!value) return null;
			if (
				value.rendererBackend === 'webgpu' &&
				value.baseLiveReady === true &&
				value.utciSurfaceSource === 'compute-buffer-selected-hour' &&
				value.baseRenderTransport === 'compute-buffer-selected-hour' &&
				value.baseSameDeviceForComputeAndRender === true &&
				value.baseSelectionKey === expectedSelectionKey &&
				value.baseSceneSelectionKey === expectedSelectionKey &&
				(expectedGridResolutionMeters == null ||
					value.baseMetadataGridSize === expectedGridResolutionMeters) &&
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
			minSurfaceRequestId: options?.minSurfaceRequestId ?? 0,
			expectedGridResolutionMeters: options?.expectedGridResolutionMeters ?? null
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

async function setHourSelection(page: Page, hourIndex: number) {
	const dayButton = page.getByRole('button', { name: /^day$/i });
	await expect(dayButton).toBeVisible();
	await dayButton.click();

	const slider = page.getByRole('slider', { name: /select analysis hour/i });
	await expect(slider).toBeVisible();
	await slider.focus();
	await slider.press('Home');
	for (let step = 0; step < hourIndex; step += 1) {
		await slider.press('ArrowRight');
	}
	await expect(slider).toHaveAttribute('aria-valuenow', String(hourIndex));
}

async function setMonthSelection(page: Page, monthIndex: number) {
	const monthButton = page.getByRole('button', { name: /^month$/i });
	await expect(monthButton).toBeVisible();
	await monthButton.click();

	const slider = page.getByRole('slider', { name: /select month/i });
	await expect(slider).toBeVisible();
	await slider.focus();
	await slider.press('Home');
	for (let step = 0; step < monthIndex; step += 1) {
		await slider.press('ArrowRight');
	}
	await expect(slider).toHaveAttribute('aria-valuenow', String(monthIndex));
}

async function selectNessTzionaProject(page: Page) {
	const selector = page.getByTestId('project-select');
	await expect(selector).toBeVisible();
	await selector.selectOption('Ness-Tziona');
}

async function requestDiagnosticsGridResolutionChange(page: Page, resolutionMeters: number) {
	const armed = await page.evaluate((resolution) => {
		return (
			(window as Window & {
				__mainRouteDiagnosticsSetGridResolution?: (resolutionMeters: number) => boolean;
			}).__mainRouteDiagnosticsSetGridResolution?.(resolution) ?? false
		);
	}, resolutionMeters);
	expect(
		armed,
		'diagnostics-only grid change hook should preserve the active base surface candidate'
	).toBe(true);
}

async function enterNzCase(page: Page, caseConfig: CollectorCase) {
	if (caseConfig.entry === 'direct-nz') {
		const entryUrl = buildMainRouteUrl(caseConfig.analysisId, caseConfig);
		logCollectorProgress(`${caseConfig.caseId}: goto direct NZ 0.5m`);
		const startedAt = performance.now();
		await page.goto(entryUrl);
		return { entryUrl, targetUrl: entryUrl, startedAt };
	}

	const entryUrl = buildMainRouteUrl(
		BG_ANALYSIS_ID,
		caseConfig,
		BG_ENTRY_GRID_RESOLUTION_METERS
	);
	const targetUrl = buildMainRouteUrl(caseConfig.analysisId, caseConfig);
	logCollectorProgress(`${caseConfig.caseId}: goto BG ${BG_ENTRY_GRID_RESOLUTION_METERS}m`);
	await page.goto(entryUrl);
	await waitForSelectedHourPublication(
		page,
		selectionKey(BG_ANALYSIS_ID, 7, 0),
		{ expectedGridResolutionMeters: BG_ENTRY_GRID_RESOLUTION_METERS }
	);
	logCollectorProgress(`${caseConfig.caseId}: select NZ project at BG grid`);
	const startedAt = performance.now();
	await selectNessTzionaProject(page);
	await waitForSelectedHourPublication(
		page,
		caseConfig.expectedInitialSelectionKey,
		{ expectedGridResolutionMeters: BG_ENTRY_GRID_RESOLUTION_METERS }
	);
	logCollectorProgress(`${caseConfig.caseId}: request diagnostics grid change to NZ 0.5m`);
	await requestDiagnosticsGridResolutionChange(page, caseConfig.gridResolutionMeters);
	return {
		entryUrl,
		targetUrl,
		startedAt
	};
}

function extractTimings(timings: Record<string, unknown> | undefined) {
	return {
		firstSelectedHourVisibleMs: numberOrNull(timings?.firstSelectedHourVisibleMs),
		oneHourDispatchMs: numberOrNull(timings?.oneHourDispatchMs),
		renderUpdateMs: numberOrNull(timings?.renderUpdateMs),
		renderSceneSyncStartDelayMs: numberOrNull(timings?.renderSceneSyncStartDelayMs),
		renderSceneSyncTotalMs: numberOrNull(timings?.renderSceneSyncTotalMs),
		exposurePrecomputeMs: numberOrNull(timings?.exposurePrecomputeMs),
		exposureQueueWaitMs: numberOrNull(timings?.exposureQueueWaitMs),
		exposurePointChunks: numberOrNull(timings?.exposurePointChunks),
		exposureSchedulerMode: exposureSchedulerModeOrNull(timings?.exposureSchedulerMode),
		exposureSchedulerSliceCount: numberOrNull(timings?.exposureSchedulerSliceCount),
		exposureSchedulerMaxWorkgroupsPerSlice: numberOrNull(
			timings?.exposureSchedulerMaxWorkgroupsPerSlice
		),
		exposureSchedulerQueueWaitTotalMs: numberOrNull(
			timings?.exposureSchedulerQueueWaitTotalMs
		),
		exposureSchedulerQueueWaitMaxMs: numberOrNull(timings?.exposureSchedulerQueueWaitMaxMs),
		exposureSchedulerQueueWaitMinMs: numberOrNull(timings?.exposureSchedulerQueueWaitMinMs),
		exposureSchedulerYieldCount: numberOrNull(timings?.exposureSchedulerYieldCount),
		exposureSchedulerSubmitCount: numberOrNull(timings?.exposureSchedulerSubmitCount)
	};
}

function pickTimelineFields(renderPublication: RenderPublicationSnapshot | null) {
	const timeline = renderPublication?.renderPublicationTimeline;
	if (!timeline) return null;
	return {
		controllerSessionRunStartedAtMs: numberOrNull(timeline.controllerSessionRunStartedAtMs),
		controllerSessionRunCompletedAtMs: numberOrNull(timeline.controllerSessionRunCompletedAtMs),
		controllerAcceptStartedAtMs: numberOrNull(timeline.controllerAcceptStartedAtMs),
		controllerAcceptedAtMs: numberOrNull(timeline.controllerAcceptedAtMs),
		controllerStatePublishedAtMs: numberOrNull(timeline.controllerStatePublishedAtMs),
		sessionComputeOutputReturnedAtMs: numberOrNull(timeline.sessionComputeOutputReturnedAtMs),
		sessionGpuOutputHandleReadyAtMs: numberOrNull(timeline.sessionGpuOutputHandleReadyAtMs),
		sessionResultReadyAtMs: numberOrNull(timeline.sessionResultReadyAtMs),
		sessionResultReturnedAtMs: numberOrNull(timeline.sessionResultReturnedAtMs),
		routePendingSurfaceExposedAtMs: numberOrNull(timeline.routePendingSurfaceExposedAtMs),
		routePublishedAtMs: numberOrNull(timeline.routePublishedAtMs),
		routeProjectedAtMs: numberOrNull(timeline.routeProjectedAtMs),
		scenePendingSurfaceObservedAtMs: numberOrNull(timeline.scenePendingSurfaceObservedAtMs),
		sceneReactiveBlockEnteredAtMs: numberOrNull(timeline.sceneReactiveBlockEnteredAtMs),
		sceneSyncInvocationQueuedAtMs: numberOrNull(timeline.sceneSyncInvocationQueuedAtMs),
		sceneReactiveToSyncQueuedMs: numberOrNull(timeline.sceneReactiveToSyncQueuedMs),
		sceneSyncQueuedToStartMs: numberOrNull(timeline.sceneSyncQueuedToStartMs),
		sceneSyncAttemptStartedAtMs: numberOrNull(timeline.sceneSyncAttemptStartedAtMs),
		sceneLayoutKeyStartedAtMs: numberOrNull(timeline.sceneLayoutKeyStartedAtMs),
		sceneLayoutKeyCompletedAtMs: numberOrNull(timeline.sceneLayoutKeyCompletedAtMs),
		scenePublicationPlanReadyAtMs: numberOrNull(timeline.scenePublicationPlanReadyAtMs),
		sceneSurfacePendingStorageInitAtMs: numberOrNull(timeline.sceneSurfacePendingStorageInitAtMs),
		renderStorageReadyAtMs: numberOrNull(timeline.renderStorageReadyAtMs),
		sceneSyncCompletedAtMs: numberOrNull(timeline.sceneSyncCompletedAtMs),
		controllerVisibleAcknowledgedAtMs: numberOrNull(
			timeline.controllerVisibleAcknowledgedAtMs
		),
		renderLayoutReuseAction: stringOrNull(timeline.renderLayoutReuseAction),
		renderLayoutReuseReason: stringOrNull(timeline.renderLayoutReuseReason),
		renderLayoutReuseProofSource: stringOrNull(timeline.renderLayoutReuseProofSource),
		renderLayoutBuildTrace: timeline.renderLayoutBuildTrace ?? null,
		renderLayoutReuseFrameCacheKind: stringOrNull(timeline.renderLayoutReuseFrameCacheKind),
		renderLayoutReuseFrameCacheHit:
			typeof timeline.renderLayoutReuseFrameCacheHit === 'boolean'
				? timeline.renderLayoutReuseFrameCacheHit
				: null,
		activeLayoutCandidateCount: numberOrNull(timeline.activeLayoutCandidateCount),
		renderSurfaceMeshTrace: timeline.renderSurfaceMeshTrace ?? null
	};
}

function pickLayoutFields(renderPublication: RenderPublicationSnapshot | null) {
	const timeline = renderPublication?.renderPublicationTimeline;
	if (!timeline) return null;
	return {
		renderLayoutReuseAction: stringOrNull(timeline.renderLayoutReuseAction),
		renderLayoutReuseReason: stringOrNull(timeline.renderLayoutReuseReason),
		renderLayoutReuseProofSource: stringOrNull(timeline.renderLayoutReuseProofSource),
		renderLayoutBuildTrace: timeline.renderLayoutBuildTrace ?? null,
		renderLayoutReuseFrameCacheKind: stringOrNull(timeline.renderLayoutReuseFrameCacheKind),
		renderLayoutReuseFrameCacheHit:
			typeof timeline.renderLayoutReuseFrameCacheHit === 'boolean'
				? timeline.renderLayoutReuseFrameCacheHit
				: null,
		activeLayoutCandidateCount: numberOrNull(timeline.activeLayoutCandidateCount),
		renderSurfaceMeshTrace: timeline.renderSurfaceMeshTrace ?? null
	};
}


function buildProof(diagnostics: DiagnosticsSnapshot) {
	return {
		rendererBackend: diagnostics.rendererBackend,
		utciRenderResolved: diagnostics.utciRenderResolved,
		utciSurfaceSource: diagnostics.utciSurfaceSource ?? null,
		baseRenderTransport: diagnostics.baseRenderTransport,
		baseSameDeviceForComputeAndRender: diagnostics.baseSameDeviceForComputeAndRender ?? null,
		dataTextureBuildCount: diagnostics.dataTextureBuildCount ?? null,
		selectedHourRuntimeContract: {
			route: diagnostics.selectedHourRuntimeContract?.route ?? null,
			readbackInstrumentation:
				diagnostics.selectedHourRuntimeContract?.readbackInstrumentation ?? null,
			visibleSelectedHourReadbackCount:
				diagnostics.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount ?? null,
			strongVisibleGpuPath:
				diagnostics.selectedHourRuntimeContract?.strongVisibleGpuPath ?? null
		}
	};
}

function assertStrongGpuProof(diagnostics: DiagnosticsSnapshot, sourceUrl: string) {
	expect(new URL(sourceUrl, 'http://localhost').pathname).toBe(SOURCE_ROUTE);
	expect(diagnostics.rendererBackend).toBe('webgpu');
	expect(diagnostics.utciSurfaceSource).toBe('compute-buffer-selected-hour');
	expect(diagnostics.baseRenderTransport).toBe('compute-buffer-selected-hour');
	expect(diagnostics.baseSameDeviceForComputeAndRender).toBe(true);
	expect(diagnostics.selectedHourRuntimeContract?.route).toBe('main');
	expect(diagnostics.selectedHourRuntimeContract?.readbackInstrumentation).toBe(
		'instrumented'
	);
	expect(diagnostics.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount).toBe(0);
	expect(diagnostics.selectedHourRuntimeContract?.strongVisibleGpuPath).toBe(true);
}

function buildSample(params: {
	caseId: string;
	actionKind: ActionKind;
	actionLabel: string;
	wallVisibleMs: number | null;
	diagnostics: DiagnosticsSnapshot;
	entryUrl: string;
	targetUrl: string;
}) {
	const renderPublication = params.diagnostics.timings?.renderPublication ?? null;
	return {
		caseId: params.caseId,
		actionKind: params.actionKind,
		actionLabel: params.actionLabel,
		selectedMonthIndex: params.diagnostics.baseSelectedMonthIndex ?? null,
		selectedHourIndex: params.diagnostics.baseSelectedHourIndex ?? null,
		selectedTimeIndex: params.diagnostics.baseSelectedTimeIndex ?? null,
		selectionKey: params.diagnostics.baseSelectionKey ?? null,
		surfaceRequestId: params.diagnostics.baseSurfaceRequestId ?? null,
		entryUrl: params.entryUrl,
		targetUrl: params.targetUrl,
		wallVisibleMs: params.wallVisibleMs,
		timings: extractTimings(params.diagnostics.timings),
		renderPublication: {
			renderPublicationVersion: renderPublication?.renderPublicationVersion ?? null,
			renderPublicationPath: renderPublication?.renderPublicationPath ?? null,
			renderPublicationPhase: renderPublication?.renderPublicationPhase ?? null,
			renderPublicationMeshAction: renderPublication?.renderPublicationMeshAction ?? null,
			renderPublicationPointCount: renderPublication?.renderPublicationPointCount ?? null,
			renderPublicationVertexCount: renderPublication?.renderPublicationVertexCount ?? null,
			timeline: pickTimelineFields(renderPublication)
		},
		layout: pickLayoutFields(renderPublication),
		proof: buildProof(params.diagnostics),
		forbiddenComparisonFieldsPresent: collectForbiddenComparisonFields(params.diagnostics)
	};
}

async function collectInteractionSample(params: {
	page: Page;
	caseConfig: CollectorCase;
	actionKind: Exclude<ActionKind, 'initial-visible'>;
	actionLabel: string;
	targetMonthIndex: number;
	targetHourIndex: number;
	previousRequestId: number;
	interact: () => Promise<void>;
	entryUrl: string;
	targetUrl: string;
}) {
	const startedAt = performance.now();
	await params.interact();
	const diagnostics = await waitForSelectedHourPublication(
		params.page,
		selectionKey(params.caseConfig.analysisId, params.targetMonthIndex, params.targetHourIndex),
		{ minSurfaceRequestId: params.previousRequestId }
	);
	const wallVisibleMs = performance.now() - startedAt;
	return {
		diagnostics,
		sample: buildSample({
			caseId: params.caseConfig.caseId,
			actionKind: params.actionKind,
			actionLabel: params.actionLabel,
			wallVisibleMs,
			diagnostics,
			entryUrl: params.entryUrl,
			targetUrl: params.targetUrl
		})
	};
}

async function collectCase(page: Page, caseConfig: CollectorCase) {
	const requestedUrls: string[] = [];
	const onRequest = (request: { url: () => string }) => requestedUrls.push(request.url());
	page.on('request', onRequest);

	try {
		logCollectorProgress(`${caseConfig.caseId}: start`);
		const sourceUrl = buildMainRouteUrl(caseConfig.analysisId, caseConfig);
		const entry = await enterNzCase(page, caseConfig);
		const initialDiagnostics = await waitForSelectedHourPublication(
			page,
			caseConfig.expectedInitialSelectionKey,
			{ expectedGridResolutionMeters: caseConfig.gridResolutionMeters }
		);
		const initialWallVisibleMs = performance.now() - entry.startedAt;
		assertStrongGpuProof(initialDiagnostics, sourceUrl);
		logCollectorProgress(
			`${caseConfig.caseId}: initial NZ 0.5m visible in ${initialWallVisibleMs.toFixed(1)}ms`
		);

		const samples = [
			buildSample({
				caseId: caseConfig.caseId,
				actionKind: 'initial-visible',
				actionLabel: caseConfig.entry,
				wallVisibleMs: initialWallVisibleMs,
				diagnostics: initialDiagnostics,
				entryUrl: entry.entryUrl,
				targetUrl: entry.targetUrl
			})
		];

		let previousRequestId = initialDiagnostics.baseSurfaceRequestId ?? 0;
		for (const hourIndex of HOUR_SEQUENCE.slice(1)) {
			logCollectorProgress(`${caseConfig.caseId}: scrub hour ${hourIndex}`);
			const result = await collectInteractionSample({
				page,
				caseConfig,
				actionKind: 'hour-scrub',
				actionLabel: `hour-${hourIndex}`,
				targetMonthIndex: INITIAL_MONTH_INDEX,
				targetHourIndex: hourIndex,
				previousRequestId,
				interact: () => setHourSelection(page, hourIndex),
				entryUrl: entry.entryUrl,
				targetUrl: entry.targetUrl
			});
			assertStrongGpuProof(result.diagnostics, sourceUrl);
			samples.push(result.sample);
			previousRequestId = result.diagnostics.baseSurfaceRequestId ?? previousRequestId;
		}

		const stableHourIndex = HOUR_SEQUENCE[HOUR_SEQUENCE.length - 1];
		for (const monthIndex of MONTH_SEQUENCE) {
			logCollectorProgress(`${caseConfig.caseId}: change month ${monthIndex}`);
			const result = await collectInteractionSample({
				page,
				caseConfig,
				actionKind: 'month-change',
				actionLabel: `month-${monthIndex}`,
				targetMonthIndex: monthIndex,
				targetHourIndex: stableHourIndex,
				previousRequestId,
				interact: () => setMonthSelection(page, monthIndex),
				entryUrl: entry.entryUrl,
				targetUrl: entry.targetUrl
			});
			assertStrongGpuProof(result.diagnostics, sourceUrl);
			samples.push(result.sample);
			previousRequestId = result.diagnostics.baseSurfaceRequestId ?? previousRequestId;
		}

		const forbiddenRequestUrls = requestedUrls.filter(isForbiddenComparisonRequest);
		const forbiddenComparisonFieldsPresent = samples.flatMap(
			(sample) => sample.forbiddenComparisonFieldsPresent
		);
		expect(forbiddenRequestUrls).toEqual([]);
		expect(forbiddenComparisonFieldsPresent).toEqual([]);

		await page.goto('about:blank');
		logCollectorProgress(`${caseConfig.caseId}: complete`);

		return {
			caseId: caseConfig.caseId,
			entry: caseConfig.entry,
			analysisId: caseConfig.analysisId,
			gridResolutionMeters: caseConfig.gridResolutionMeters,
			queryParams: caseConfig.queryParams ?? null,
			entryUrl: entry.entryUrl,
			targetUrl: entry.targetUrl,
			proofRouteUrl: sourceUrl,
			samples,
			assertions: {
				strongGpuPath: true,
				forbiddenRequestUrls,
				forbiddenComparisonFieldsPresent
			}
		};
	} finally {
		page.off('request', onRequest);
	}
}

test.describe('main route transition scrub diagnostics collector', () => {
	test.afterEach(async ({ page }) => {
		await page.goto('about:blank').catch(() => undefined);
	});

	test('collects direct and BG-to-NZ 0.5m scrub diagnostics in normal and chunked-2048 modes', async ({
		page
	}, testInfo) => {
		test.setTimeout(900_000);

		const cases = [];
		for (const caseConfig of CASES) {
			const collectedCase = await collectCase(page, caseConfig);
			cases.push(collectedCase);
			writeJsonArtifact(PROGRESS_ARTIFACT_PATH, {
				collectedOn: formatLocalDate(new Date()),
				sourceRoute: SOURCE_ROUTE,
				artifact: PROGRESS_ARTIFACT_FILENAME,
				completedCaseIds: cases.map((item) => item.caseId),
				cases
			});
		}

		const artifact = {
			collectedOn: formatLocalDate(new Date()),
			sourceRoute: SOURCE_ROUTE,
			artifact: ARTIFACT_FILENAME,
			collectionMethod:
				'Main route only: compares direct Ness Tziona 0.5m loads with Ben-Gurion-to-Ness-Tziona project-selector transitions, then records repeated app-visible hour and month publications. Chunked mode is supplied only through utciExposureSchedule=chunked&utciExposureMaxWorkgroupsPerSlice=2048 query params.',
			cases
		};

		const json = JSON.stringify(artifact, null, 2);
		writeJsonArtifact(ARTIFACT_PATH, artifact);
		expect(existsSync(ARTIFACT_PATH)).toBe(true);
		await testInfo.attach(ARTIFACT_FILENAME, {
			body: json,
			contentType: 'application/json'
		});
	});
});
