import { test, type Page } from '@playwright/test';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const RESULTS_DIR = resolve(REPO_ROOT, 'data/performance-results');
const ARTIFACT_FILENAME = 'main-route-visual-freeze-map.json';
const ARTIFACT_PATH = resolve(RESULTS_DIR, ARTIFACT_FILENAME);
const DIAGNOSTIC_ARTIFACT_FILENAME = 'main-route-exposure-and-raf-diagnostics.json';
const DIAGNOSTIC_ARTIFACT_PATH = resolve(RESULTS_DIR, DIAGNOSTIC_ARTIFACT_FILENAME);
const SOURCE_ROUTE = '/';
const DIAGNOSTICS_POLL_INTERVAL_MS = 500;
const DIAGNOSTIC_FOCUS =
	'exposure breathing during cold load plus render-publication rAF correlation on main route /';

type AnalysisCase = {
	caseId: string;
	projectLabel: string;
	analysisId: string;
	expectedSelectionKey: string;
	gridResolutionMeters: 2 | 0.5;
	queryParams?: Record<string, string>;
};

type DiagnosticsSnapshot = Record<string, any>;

type BrowserProbeSnapshot = {
	createdAtMs: number;
	rafGaps: Array<Record<string, unknown>>;
	intervalGaps: Array<Record<string, unknown>>;
	longTasks: Array<Record<string, unknown>>;
	gpuEvents: Array<Record<string, unknown>>;
	marks: Array<Record<string, unknown>>;
	counters: Record<string, unknown>;
};

type PollSample = {
	pollIndex: number;
	nodePollStartedAtMs: number;
	nodePollEndedAtMs: number;
	nodePollDurationMs: number;
	pagePerformanceNowMs: number | null;
	hasDiagnostics: boolean;
	readError?: string;
	diagnostics: Record<string, unknown> | null;
};

type TimingWindow = {
	label: string;
	startMs: number;
	endMs: number;
	durationMs: number;
};

type OverlapSummaryEntry = {
	startMs: number | null;
	endMs: number | null;
	durationMs: number | null;
	overlapExposureSliceCount: number;
	overlapExposureSliceIndexes: number[];
	overlapRenderPublicationWindowLabels: string[];
};

type PageEvents = {
	console: Array<Record<string, unknown>>;
	pageErrors: string[];
	requestFailures: Array<Record<string, unknown>>;
	crashes: Array<Record<string, unknown>>;
};

type PageEventCollector = {
	events: PageEvents;
	dispose: () => void;
};

type CollectedCase = {
	caseId: string;
	projectLabel: string;
	analysisId: string;
	gridResolutionMeters: 2 | 0.5;
	sourceUrl: string;
	startedAt: string;
	finishedAt: string;
	durationMs: number;
	publicationReached: boolean;
	failure?: {
		message: string;
		stack?: string;
	};
	summary: Record<string, unknown>;
	raw: {
		topRafGaps: BrowserProbeSnapshot['rafGaps'];
		topIntervalGaps: BrowserProbeSnapshot['intervalGaps'];
		longTasks: BrowserProbeSnapshot['longTasks'];
		gpuEvents: BrowserProbeSnapshot['gpuEvents'];
		phaseMarks: BrowserProbeSnapshot['marks'];
		probeCounters: BrowserProbeSnapshot['counters'];
		diagnosticsSamples: PollSample[];
		finalDiagnostics: Record<string, unknown>;
		pageEvents: PageEvents;
	};
};

const CASES: AnalysisCase[] = [
	{
		caseId: 'ness-tziona-0_5m',
		projectLabel: 'Ness-Tziona',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		expectedSelectionKey: 'Ness-Tziona/exploded/nes_tziona_unblock_2|7|0',
		gridResolutionMeters: 0.5
	},
	{
		caseId: 'ben-gurion-0_5m',
		projectLabel: 'Ben-Gurion',
		analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
		expectedSelectionKey: 'Ben-Gurion/20250815_grid_2m_fullday|7|0',
		gridResolutionMeters: 0.5
	},
	{
		caseId: 'ness-tziona-2m',
		projectLabel: 'Ness-Tziona',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		expectedSelectionKey: 'Ness-Tziona/exploded/nes_tziona_unblock_2|7|0',
		gridResolutionMeters: 2
	}
];

function numberOrNull(value: unknown): number | null {
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function exposureSchedulerModeOrNull(value: unknown): 'single-submit' | 'chunked' | null {
	return value === 'single-submit' || value === 'chunked' ? value : null;
}

function topByDuration<T extends Record<string, unknown>>(items: T[], limit: number): T[] {
	return [...items]
		.sort((left, right) => (numberOrNull(right.durationMs) ?? 0) - (numberOrNull(left.durationMs) ?? 0))
		.slice(0, limit);
}

function recordOrNull(value: unknown): Record<string, unknown> | null {
	return typeof value === 'object' && value != null ? (value as Record<string, unknown>) : null;
}

function asRecordArray(value: unknown): Record<string, unknown>[] {
	return Array.isArray(value)
		? value.filter(
				(item): item is Record<string, unknown> =>
					typeof item === 'object' && item != null
			)
		: [];
}

function createWindow(
	label: string,
	startMs: unknown,
	endMs: unknown
): TimingWindow | null {
	const normalizedStartMs = numberOrNull(startMs);
	const normalizedEndMs = numberOrNull(endMs);
	if (normalizedStartMs == null || normalizedEndMs == null || normalizedEndMs < normalizedStartMs) {
		return null;
	}
	return {
		label,
		startMs: normalizedStartMs,
		endMs: normalizedEndMs,
		durationMs: normalizedEndMs - normalizedStartMs
	};
}

function windowsOverlap(left: TimingWindow, right: TimingWindow): boolean {
	return left.startMs < right.endMs && left.endMs > right.startMs;
}

function buildGapWindow(
	entry: Record<string, unknown>,
	kind: 'raf' | 'interval' | 'longTask',
	index: number
): TimingWindow | null {
	if (kind === 'longTask') {
		const startMs = numberOrNull(entry.startTimeMs);
		const durationMs = numberOrNull(entry.durationMs);
		if (startMs == null || durationMs == null) return null;
		return createWindow(`longTask:${index}`, startMs, startMs + durationMs);
	}
	return createWindow(`${kind}:${index}`, entry.startMs, entry.endMs);
}

function collectExposureSliceWindows(diagnostics: Record<string, unknown> | null): TimingWindow[] {
	const timings = recordOrNull(diagnostics?.timings);
	const breathingTrace = recordOrNull(timings?.exposureSchedulerBreathingTrace);
	return asRecordArray(breathingTrace?.allSliceWindows)
		.map((window, index) =>
			createWindow(`slice:${numberOrNull(window.sliceIndex) ?? index}`, window.startMs, window.endMs)
		)
		.filter((window): window is TimingWindow => window != null);
}

function collectRenderPublicationWindows(
	diagnostics: Record<string, unknown> | null
): TimingWindow[] {
	const timings = recordOrNull(diagnostics?.timings);
	const renderPublication = recordOrNull(timings?.renderPublication);
	const timeline = recordOrNull(renderPublication?.renderPublicationTimeline);
	return [
		createWindow(
			'renderPublicationTotal',
			timeline?.sceneSurfaceReceivedAtMs,
			timeline?.controllerVisibleAcknowledgedAtMs
		),
		createWindow(
			'renderPublicationPreStorage',
			timeline?.renderPublicationPreStorageStartedAtMs,
			timeline?.renderPublicationPreStorageCompletedAtMs
		),
		createWindow(
			'renderStorageFirstWaitFrame',
			timeline?.renderStorageFirstWaitFrameRequestedAtMs,
			timeline?.renderStorageFirstWaitFrameCompletedAtMs
		),
		createWindow(
			'renderStorageWait',
			timeline?.renderStorageWaitStartedAtMs,
			timeline?.renderStorageReadyAtMs
		),
		createWindow(
			'renderCopyQueueDrain',
			timeline?.renderCopyQueueDrainStartedAtMs,
			timeline?.renderCopyQueueDrainCompletedAtMs
		)
	].filter((window): window is TimingWindow => window != null);
}

function summarizeEventOverlaps(
	items: Array<Record<string, unknown>>,
	kind: 'raf' | 'interval' | 'longTask',
	exposureSliceWindows: TimingWindow[],
	renderPublicationWindows: TimingWindow[]
): OverlapSummaryEntry[] {
	return items.map((item, index) => {
		const gapWindow = buildGapWindow(item, kind, index);
		const overlappingExposureWindows =
			gapWindow == null
				? []
				: exposureSliceWindows.filter((window) => windowsOverlap(gapWindow, window));
		const overlappingRenderPublicationWindows =
			gapWindow == null
				? []
				: renderPublicationWindows.filter((window) => windowsOverlap(gapWindow, window));
		return {
			startMs: gapWindow?.startMs ?? null,
			endMs: gapWindow?.endMs ?? null,
			durationMs: numberOrNull(item.durationMs),
			overlapExposureSliceCount: overlappingExposureWindows.length,
			overlapExposureSliceIndexes: overlappingExposureWindows
				.map((window) => Number(window.label.replace('slice:', '')))
				.filter(Number.isFinite)
				.slice(0, 12),
			overlapRenderPublicationWindowLabels: overlappingRenderPublicationWindows
				.map((window) => window.label)
				.slice(0, 12)
		};
	});
}

function selectOverlapDiagnostics(params: {
	finalDiagnostics: DiagnosticsSnapshot | null;
	pollSamples: PollSample[];
}): Record<string, unknown> | null {
	const summarizedFinalDiagnostics = summarizeDiagnostics(params.finalDiagnostics);
	if (summarizedFinalDiagnostics) return summarizedFinalDiagnostics;
	for (let index = params.pollSamples.length - 1; index >= 0; index -= 1) {
		const diagnostics = recordOrNull(params.pollSamples[index]?.diagnostics);
		if (diagnostics) return diagnostics;
	}
	return null;
}

function summarizeDiagnostics(value: DiagnosticsSnapshot | null): Record<string, unknown> | null {
	if (!value) return null;
	return {
		rendererBackend: value.rendererBackend ?? null,
		utciRenderResolved: value.utciRenderResolved ?? null,
		baseLiveReady: value.baseLiveReady ?? null,
		utciSurfaceSource: value.utciSurfaceSource ?? null,
		baseRenderTransport: value.baseRenderTransport ?? null,
		baseSameDeviceForComputeAndRender: value.baseSameDeviceForComputeAndRender ?? null,
		baseSelectionKey: value.baseSelectionKey ?? null,
		baseSceneSelectionKey: value.baseSceneSelectionKey ?? null,
		baseSurfaceRequestId: value.baseSurfaceRequestId ?? null,
		gpuResidentCopyRequestId: value.gpuResidentCopyRequestId ?? null,
		baseSelectedMonthIndex: value.baseSelectedMonthIndex ?? null,
		baseSelectedHourIndex: value.baseSelectedHourIndex ?? null,
		baseSelectedTimeIndex: value.baseSelectedTimeIndex ?? null,
		basePointCount: value.basePointCount ?? null,
		baseMetadataGridSize: value.baseMetadataGridSize ?? null,
		dataTextureBuildCount: value.dataTextureBuildCount ?? null,
		selectedHourRuntimeContract: value.selectedHourRuntimeContract ?? null,
		timings: {
			payloadPrepareMs: numberOrNull(value.timings?.payloadPrepareMs),
			workerBvhMs: numberOrNull(value.timings?.workerBvhMs),
			pipelineUploadMs: numberOrNull(value.timings?.pipelineUploadMs),
			staticUploadTrace: value.timings?.staticUploadTrace ?? null,
			exposurePrecomputeMs: numberOrNull(value.timings?.exposurePrecomputeMs),
			exposureWeatherBufferEnsureMs: numberOrNull(
				value.timings?.exposureWeatherBufferEnsureMs
			),
			exposureCommandEncodeTotalMs: numberOrNull(
				value.timings?.exposureCommandEncodeTotalMs
			),
			exposureSolarEncodeMs: numberOrNull(value.timings?.exposureSolarEncodeMs),
			exposureSkyEncodeMs: numberOrNull(value.timings?.exposureSkyEncodeMs),
			exposureQueueWaitMs: numberOrNull(value.timings?.exposureQueueWaitMs),
			exposurePointCount: numberOrNull(value.timings?.exposurePointCount),
			exposureTotalTimeSteps: numberOrNull(value.timings?.exposureTotalTimeSteps),
			exposureDaylightTimeSteps: numberOrNull(value.timings?.exposureDaylightTimeSteps),
			exposurePointChunks: numberOrNull(value.timings?.exposurePointChunks),
			exposureSchedulerMode: exposureSchedulerModeOrNull(value.timings?.exposureSchedulerMode),
			exposureSchedulerSliceCount: numberOrNull(value.timings?.exposureSchedulerSliceCount),
			exposurePointDispatchChunkCount: numberOrNull(
				value.timings?.exposurePointDispatchChunkCount
			),
			exposureSchedulerMaxWorkgroupsPerSlice: numberOrNull(
				value.timings?.exposureSchedulerMaxWorkgroupsPerSlice
			),
			exposureSchedulerQueueWaitTotalMs: numberOrNull(
				value.timings?.exposureSchedulerQueueWaitTotalMs
			),
			exposureSchedulerQueueWaitMaxMs: numberOrNull(
				value.timings?.exposureSchedulerQueueWaitMaxMs
			),
			exposureSchedulerBreathingTrace:
				value.timings?.exposureSchedulerBreathingTrace ?? null,
			exposureSchedulerYieldCount: numberOrNull(
				value.timings?.exposureSchedulerYieldCount
			),
			exposureSchedulerSubmitCount: numberOrNull(
				value.timings?.exposureSchedulerSubmitCount
			),
			exposureSolarDispatchCount: numberOrNull(value.timings?.exposureSolarDispatchCount),
			exposureSkyDispatchCount: numberOrNull(value.timings?.exposureSkyDispatchCount),
			exposureSolarRayBudget: numberOrNull(value.timings?.exposureSolarRayBudget),
			exposureSkyRayBudget: numberOrNull(value.timings?.exposureSkyRayBudget),
			oneHourDispatchMs: numberOrNull(value.timings?.oneHourDispatchMs),
			firstSelectedHourReadyMs: numberOrNull(value.timings?.firstSelectedHourReadyMs),
			firstSelectedHourVisibleMs: numberOrNull(value.timings?.firstSelectedHourVisibleMs),
			renderUpdateMs: numberOrNull(value.timings?.renderUpdateMs),
			renderSceneSyncStartDelayMs: numberOrNull(value.timings?.renderSceneSyncStartDelayMs),
			renderSceneSyncTotalMs: numberOrNull(value.timings?.renderSceneSyncTotalMs),
			renderLayoutBuildMs: numberOrNull(value.timings?.renderLayoutBuildMs),
			renderSurfaceMeshMs: numberOrNull(value.timings?.renderSurfaceMeshMs),
			renderStorageInitWaitMs: numberOrNull(value.timings?.renderStorageInitWaitMs),
			renderBufferCopyMs: numberOrNull(value.timings?.renderBufferCopyMs),
			renderQueueDrainMs: numberOrNull(value.timings?.renderQueueDrainMs),
			renderPublicationPreStorageMs: numberOrNull(
				value.timings?.renderPublication?.renderPublicationTimeline
					?.renderPublicationPreStorageMs
			),
			renderCopyQueueDrainMs: numberOrNull(
				value.timings?.renderPublication?.renderPublicationTimeline
					?.renderCopyQueueDrainMs
			),
			renderPublication: value.timings?.renderPublication ?? null
		},
		trackedGpuAllocationBytes: value.trackedGpuAllocationBytes ?? null
	};
}

function summarizeTimingBuckets(value: DiagnosticsSnapshot | null): Record<string, unknown> | null {
	const diagnostics = summarizeDiagnostics(value);
	const timings = recordOrNull(diagnostics?.timings);
	if (!timings) return null;
	const {
		exposureSchedulerBreathingTrace: _exposureSchedulerBreathingTrace,
		renderPublication: _renderPublication,
		firstSelectedHourVisibleMs,
		...compactTimings
	} = timings;
	return {
		...compactTimings,
		pipelineFirstSelectedHourVisibleMs: numberOrNull(firstSelectedHourVisibleMs)
	};
}

function ownedGpuMemoryBytes(diagnostics: DiagnosticsSnapshot | null): number | null {
	const tracked = diagnostics?.trackedGpuAllocationBytes;
	if (!tracked) return null;
	const fields = [
		tracked.persistentExposureBytes,
		tracked.allHoursOutputBytes,
		tracked.selectedHourOutputBytes,
		tracked.renderOwnedSelectedHourBytes
	];
	if (!fields.every((value) => typeof value === 'number' && Number.isFinite(value))) {
		return null;
	}
	return fields.reduce((sum, value) => sum + value, 0);
}

function installPageEventCollectors(page: Page): PageEventCollector {
	const events: PageEvents = {
		console: [],
		pageErrors: [],
		requestFailures: [],
		crashes: []
	};

	const onConsole = (message: any) => {
		if (events.console.length >= 200) return;
		events.console.push({
			type: message.type(),
			text: message.text(),
			location: message.location()
		});
	};
	const onPageError = (error: Error) => {
		if (events.pageErrors.length >= 50) return;
		events.pageErrors.push(error.stack ?? error.message);
	};
	const onRequestFailed = (request: any) => {
		if (events.requestFailures.length >= 100) return;
		events.requestFailures.push({
			url: request.url(),
			method: request.method(),
			failure: request.failure()?.errorText ?? null
		});
	};
	const onCrash = () => {
		events.crashes.push({
			nodeTimeMs: performance.now()
		});
	};

	page.on('console', onConsole);
	page.on('pageerror', onPageError);
	page.on('requestfailed', onRequestFailed);
	page.on('crash', onCrash);

	return {
		events,
		dispose: () => {
			page.off('console', onConsole);
			page.off('pageerror', onPageError);
			page.off('requestfailed', onRequestFailed);
			page.off('crash', onCrash);
		}
	};
}

function snapshotPageEvents(events: PageEvents): PageEvents {
	return {
		console: [...events.console],
		pageErrors: [...events.pageErrors],
		requestFailures: [...events.requestFailures],
		crashes: [...events.crashes]
	};
}

async function installBrowserProbe(page: Page) {
	await page.addInitScript(() => {
		const win = window as any;
		const now = () => performance.now();
		const limitPush = (items: unknown[], item: unknown, limit = 500) => {
			items.push(item);
			if (items.length > limit) items.shift();
		};
		const probe = {
			createdAtMs: now(),
			rafGaps: [] as Array<Record<string, unknown>>,
			intervalGaps: [] as Array<Record<string, unknown>>,
			longTasks: [] as Array<Record<string, unknown>>,
			gpuEvents: [] as Array<Record<string, unknown>>,
			marks: [] as Array<Record<string, unknown>>,
			counters: {
				rafFrames: 0,
				intervalTicks: 0,
				longTaskCount: 0,
				requestDeviceCalls: 0,
				uncapturedErrors: 0,
				deviceLostEvents: 0
			}
		};
		const mark = (name: string, extra: Record<string, unknown> = {}) => {
			limitPush(probe.marks, { name, atMs: now(), ...extra });
		};

		win.__visualFreezeProbe__ = probe;
		win.__visualFreezeMark__ = mark;
		mark('init-script-installed', {
			readyState: document.readyState,
			visibilityState: document.visibilityState
		});

		let lastRaf: number | null = null;
		const rafLoop = (timestamp: number) => {
			probe.counters.rafFrames = Number(probe.counters.rafFrames) + 1;
			if (lastRaf != null) {
				const durationMs = timestamp - lastRaf;
				if (durationMs >= 50) {
					limitPush(probe.rafGaps, {
						startMs: lastRaf,
						endMs: timestamp,
						durationMs,
						visibilityState: document.visibilityState
					});
				}
			}
			lastRaf = timestamp;
			requestAnimationFrame(rafLoop);
		};
		requestAnimationFrame(rafLoop);

		let lastInterval = now();
		setInterval(() => {
			const current = now();
			probe.counters.intervalTicks = Number(probe.counters.intervalTicks) + 1;
			const durationMs = current - lastInterval;
			if (durationMs >= 100) {
				limitPush(probe.intervalGaps, {
					startMs: lastInterval,
					endMs: current,
					durationMs,
					visibilityState: document.visibilityState
				});
			}
			lastInterval = current;
		}, 50);

		try {
			const Observer = (window as any).PerformanceObserver;
			if (Observer?.supportedEntryTypes?.includes('longtask')) {
				const observer = new Observer((list: PerformanceObserverEntryList) => {
					for (const entry of list.getEntries()) {
						probe.counters.longTaskCount = Number(probe.counters.longTaskCount) + 1;
						limitPush(probe.longTasks, {
							name: entry.name,
							startTimeMs: entry.startTime,
							durationMs: entry.duration,
							entryType: entry.entryType
						});
					}
				});
				observer.observe({ entryTypes: ['longtask'] });
				mark('longtask-observer-installed');
			} else {
				mark('longtask-observer-unavailable');
			}
		} catch (error) {
			mark('longtask-observer-error', {
				error: error instanceof Error ? error.message : String(error)
			});
		}

		const recordGpuEvent = (type: string, extra: Record<string, unknown> = {}) => {
			limitPush(probe.gpuEvents, { type, atMs: now(), ...extra });
		};
		const patchDevice = (device: any) => {
			if (!device || device.__visualFreezeProbePatched) return device;
			Object.defineProperty(device, '__visualFreezeProbePatched', { value: true });
			device.addEventListener?.('uncapturederror', (event: any) => {
				probe.counters.uncapturedErrors = Number(probe.counters.uncapturedErrors) + 1;
				recordGpuEvent('uncapturederror', {
					message: event?.error?.message ?? String(event?.error ?? event)
				});
			});
			device.lost?.then?.((info: any) => {
				probe.counters.deviceLostEvents = Number(probe.counters.deviceLostEvents) + 1;
				recordGpuEvent('device-lost', {
					reason: info?.reason ?? null,
					message: info?.message ?? null
				});
			});
			return device;
		};
		const patchAdapter = (adapter: any) => {
			if (!adapter || adapter.__visualFreezeProbePatched) return adapter;
			const originalRequestDevice = adapter.requestDevice?.bind(adapter);
			if (!originalRequestDevice) return adapter;
			Object.defineProperty(adapter, '__visualFreezeProbePatched', { value: true });
			adapter.requestDevice = async (...args: unknown[]) => {
				const callId = Number(probe.counters.requestDeviceCalls) + 1;
				probe.counters.requestDeviceCalls = callId;
				recordGpuEvent('requestDevice-start', { callId });
				try {
					const device = await originalRequestDevice(...args);
					recordGpuEvent('requestDevice-resolved', { callId });
					return patchDevice(device);
				} catch (error) {
					recordGpuEvent('requestDevice-rejected', {
						callId,
						message: error instanceof Error ? error.message : String(error)
					});
					throw error;
				}
			};
			return adapter;
		};

		try {
			const gpu = (navigator as any).gpu;
			const originalRequestAdapter = gpu?.requestAdapter?.bind(gpu);
			if (originalRequestAdapter) {
				gpu.requestAdapter = async (...args: unknown[]) => {
					recordGpuEvent('requestAdapter-start');
					const adapter = await originalRequestAdapter(...args);
					recordGpuEvent('requestAdapter-resolved', { hasAdapter: Boolean(adapter) });
					return patchAdapter(adapter);
				};
				mark('webgpu-request-adapter-wrapped');
			} else {
				mark('webgpu-request-adapter-unavailable');
			}
		} catch (error) {
			mark('webgpu-wrapper-error', {
				error: error instanceof Error ? error.message : String(error)
			});
		}

		document.addEventListener('DOMContentLoaded', () => mark('dom-content-loaded'));
		window.addEventListener('load', () => mark('window-load'));
		window.addEventListener('pageshow', () => mark('page-show'));
		window.addEventListener('pagehide', () => mark('page-hide'));
		document.addEventListener('visibilitychange', () =>
			mark('visibility-change', { visibilityState: document.visibilityState })
		);
	});
}

async function readUtciRenderDiagnostics(page: Page) {
	return page.evaluate(() => (window as any).__utciRenderDiagnostics__ ?? null);
}

async function waitForSelectedHourPublication(page: Page, expectedSelectionKey: string) {
	const diagnostics = await page
		.waitForFunction(
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
					typeof value.baseSurfaceRequestId === 'number' &&
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
			{ timeout: 240_000 }
		)
		.catch(async (error) => {
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

async function pollDiagnosticsUntilStopped(
	page: Page,
	caseStartedAtMs: number,
	stopSignal: { stopped: boolean }
): Promise<PollSample[]> {
	const samples: PollSample[] = [];
	let pollIndex = 0;
	while (!stopSignal.stopped) {
		const nodePollStartedAtMs = performance.now() - caseStartedAtMs;
		const sample: PollSample = {
			pollIndex,
			nodePollStartedAtMs,
			nodePollEndedAtMs: nodePollStartedAtMs,
			nodePollDurationMs: 0,
			pagePerformanceNowMs: null,
			hasDiagnostics: false,
			diagnostics: null
		};

		try {
			const payload = await page.evaluate(() => ({
				pagePerformanceNowMs: performance.now(),
				diagnostics: (window as any).__utciRenderDiagnostics__ ?? null
			}));
			sample.pagePerformanceNowMs = numberOrNull(payload.pagePerformanceNowMs);
			sample.hasDiagnostics = Boolean(payload.diagnostics);
			sample.diagnostics = summarizeDiagnostics(payload.diagnostics as DiagnosticsSnapshot | null);
		} catch (error) {
			sample.readError = error instanceof Error ? error.message : String(error);
		}

		const nodePollEndedAtMs = performance.now() - caseStartedAtMs;
		sample.nodePollEndedAtMs = nodePollEndedAtMs;
		sample.nodePollDurationMs = nodePollEndedAtMs - nodePollStartedAtMs;
		samples.push(sample);
		pollIndex += 1;

		if (samples.length >= 600 || stopSignal.stopped) break;
		await page.waitForTimeout(DIAGNOSTICS_POLL_INTERVAL_MS).catch(() => undefined);
	}
	return samples;
}

async function readBrowserProbe(page: Page): Promise<BrowserProbeSnapshot> {
	const fallback = {
		createdAtMs: 0,
		rafGaps: [],
		intervalGaps: [],
		longTasks: [],
		gpuEvents: [],
		marks: [],
		counters: {}
	};
	return page
		.evaluate(() => (window as any).__visualFreezeProbe__ ?? null)
		.then((value) => value ?? fallback)
		.catch(() => fallback);
}

function buildSourceUrl(caseConfig: AnalysisCase) {
	const params = new URLSearchParams({
		analysis: caseConfig.analysisId,
		gridResolution: String(caseConfig.gridResolutionMeters),
		utciRender: 'auto',
		utciRenderDiagnostics: '1'
	});
	for (const [key, value] of Object.entries(caseConfig.queryParams ?? {})) {
		params.set(key, value);
	}
	return `${SOURCE_ROUTE}?${params.toString()}`;
}

function buildSummary(params: {
	finalDiagnostics: DiagnosticsSnapshot | null;
	browserProbe: BrowserProbeSnapshot;
	pollSamples: PollSample[];
	pageEvents: PageEvents;
	durationMs: number;
	publicationReached: boolean;
}) {
	const {
		finalDiagnostics,
		browserProbe,
		pollSamples,
		pageEvents,
		durationMs,
		publicationReached
	} = params;
	const topRafGaps = topByDuration(browserProbe.rafGaps, 10);
	const topIntervalGaps = topByDuration(browserProbe.intervalGaps, 10);
	const longTasks = topByDuration(browserProbe.longTasks, 25);
	const pollDurations = pollSamples.map((sample) => sample.nodePollDurationMs);
	const overlapDiagnostics = selectOverlapDiagnostics({ finalDiagnostics, pollSamples });
	const exposureSliceWindows = collectExposureSliceWindows(overlapDiagnostics);
	const renderPublicationWindows = collectRenderPublicationWindows(overlapDiagnostics);
	return {
		publicationReached,
		durationMs,
		firstSelectedHourVisibleMs: numberOrNull(
			finalDiagnostics?.timings?.renderPublication?.renderPublicationTimeline
				?.controllerVisibleAcknowledgedAtMs
		),
		finalTimingBuckets: summarizeTimingBuckets(finalDiagnostics),
		trackedGpuAllocationBytes: finalDiagnostics?.trackedGpuAllocationBytes ?? null,
		ownedGpuMemoryBytes: ownedGpuMemoryBytes(finalDiagnostics),
		drawIndices:
			finalDiagnostics?.timings?.renderPublication?.renderPublicationDrawIndexCount ??
			finalDiagnostics?.drawIndices ??
			finalDiagnostics?.baseDrawIndexCount ??
			null,
		topRafGapMs: numberOrNull(topRafGaps[0]?.durationMs),
		topIntervalGapMs: numberOrNull(topIntervalGaps[0]?.durationMs),
		topLongTaskMs: numberOrNull(longTasks[0]?.durationMs),
		rafGapCount: browserProbe.rafGaps.length,
		intervalGapCount: browserProbe.intervalGaps.length,
		longTaskCount: browserProbe.longTasks.length,
		gapOverlapSummary: {
			exposureSliceWindowCount: exposureSliceWindows.length,
			renderPublicationWindowCount: renderPublicationWindows.length,
			topRafGaps: summarizeEventOverlaps(
				topRafGaps,
				'raf',
				exposureSliceWindows,
				renderPublicationWindows
			),
			topIntervalGaps: summarizeEventOverlaps(
				topIntervalGaps,
				'interval',
				exposureSliceWindows,
				renderPublicationWindows
			),
			longTasks: summarizeEventOverlaps(
				longTasks,
				'longTask',
				exposureSliceWindows,
				renderPublicationWindows
			)
		},
		gpuEventCount: browserProbe.gpuEvents.length,
		pageErrorCount: pageEvents.pageErrors.length,
		requestFailureCount: pageEvents.requestFailures.length,
		crashCount: pageEvents.crashes.length,
		pollCount: pollSamples.length,
		maxPollDurationMs: pollDurations.length > 0 ? Math.max(...pollDurations) : null
	};
}

async function collectCase(
	page: Page,
	caseConfig: AnalysisCase,
	pageEvents: PageEvents
): Promise<CollectedCase> {
	const sourceUrl = buildSourceUrl(caseConfig);
	const startedAt = new Date();
	const caseStartedAtMs = performance.now();
	const stopSignal = { stopped: false };
	const pollPromise = pollDiagnosticsUntilStopped(page, caseStartedAtMs, stopSignal);
	let finalDiagnostics: DiagnosticsSnapshot | null = null;
	let publicationReached = false;
	let failure: CollectedCase['failure'] | undefined;

	try {
		await page.goto(sourceUrl);
		finalDiagnostics = await waitForSelectedHourPublication(
			page,
			caseConfig.expectedSelectionKey
		);
		publicationReached = true;
	} catch (error) {
		failure = {
			message: error instanceof Error ? error.message : String(error),
			stack: error instanceof Error ? error.stack : undefined
		};
		finalDiagnostics = await readUtciRenderDiagnostics(page).catch(() => null);
	}
	stopSignal.stopped = true;
	const pollSamples = await pollPromise;
	const browserProbe = await readBrowserProbe(page);
	const durationMs = performance.now() - caseStartedAtMs;
	const finishedAt = new Date();
	const pageEventsSnapshot = snapshotPageEvents(pageEvents);

	return {
		caseId: caseConfig.caseId,
		projectLabel: caseConfig.projectLabel,
		analysisId: caseConfig.analysisId,
		gridResolutionMeters: caseConfig.gridResolutionMeters,
		sourceUrl,
		startedAt: startedAt.toISOString(),
		finishedAt: finishedAt.toISOString(),
		durationMs,
		publicationReached,
		failure,
		summary: buildSummary({
			finalDiagnostics,
			browserProbe,
			pollSamples,
			pageEvents: pageEventsSnapshot,
			durationMs,
			publicationReached
		}),
		raw: {
			topRafGaps: topByDuration(browserProbe.rafGaps, 50),
			topIntervalGaps: topByDuration(browserProbe.intervalGaps, 50),
			longTasks: topByDuration(browserProbe.longTasks, 100),
			gpuEvents: browserProbe.gpuEvents,
			phaseMarks: browserProbe.marks,
			probeCounters: browserProbe.counters,
			diagnosticsSamples: pollSamples,
			finalDiagnostics: summarizeDiagnostics(finalDiagnostics) ?? {},
			pageEvents: pageEventsSnapshot
		}
	};
}

function assertCollectorConfig(testInfo: {
	config: { configFile?: string };
	project: { name: string; use: Record<string, unknown> };
}) {
	const configFile = testInfo.config.configFile ?? '';
	const projectUse = testInfo.project.use as {
		headless?: boolean;
		launchOptions?: { args?: string[] };
	};
	if (!configFile.endsWith('playwright.collect.config.ts')) {
		throw new Error(
			`Visual-freeze collector must run with viewer/playwright.collect.config.ts; got ${configFile || 'unknown config'}.`
		);
	}
	if (testInfo.project.name !== 'chromium') {
		throw new Error(
			`Visual-freeze collector must run the chromium project; got ${testInfo.project.name}.`
		);
	}
	if (projectUse.headless !== false) {
		throw new Error('Visual-freeze collector must run headed Chromium.');
	}
	if (!projectUse.launchOptions?.args?.includes('--enable-unsafe-webgpu')) {
		throw new Error('Visual-freeze collector must enable --enable-unsafe-webgpu.');
	}
}

function buildFocusedDiagnosticArtifact(artifact: {
	collectedAt: string;
	sourceRoute: string;
	headedCollectorConfig: string;
	collectionMethod: string;
	pollIntervalMs: number;
	cases: CollectedCase[];
}) {
	return {
		...artifact,
		diagnosticFocus: DIAGNOSTIC_FOCUS,
		cases: artifact.cases.map((entry) => ({
			...entry,
			raw: {
				topRafGaps: entry.raw.topRafGaps,
				topIntervalGaps: entry.raw.topIntervalGaps,
				longTasks: entry.raw.longTasks,
				gpuEvents: entry.raw.gpuEvents,
				probeCounters: entry.raw.probeCounters,
				finalDiagnostics: entry.raw.finalDiagnostics,
				pageEvents: entry.raw.pageEvents
			}
		}))
	};
}

test.describe('main route desktop visual freeze map collector', () => {
	test.afterEach(async ({ page }) => {
		await page.goto('about:blank').catch(() => undefined);
	});

	test('maps RAF, event-loop, long-task, WebGPU, and diagnostics timing during cold load', async ({
		page
	}, testInfo) => {
		test.setTimeout(600_000);
		assertCollectorConfig(testInfo);

		await installBrowserProbe(page);
		const cases: CollectedCase[] = [];
		for (const caseConfig of CASES) {
			const pageEvents = installPageEventCollectors(page);
			try {
				const collected = await collectCase(page, caseConfig, pageEvents.events);
				cases.push(collected);
			} finally {
				pageEvents.dispose();
				await page.goto('about:blank');
			}
		}

		const artifact = {
			collectedAt: new Date().toISOString(),
			sourceRoute: SOURCE_ROUTE,
			headedCollectorConfig: 'viewer/playwright.collect.config.ts',
			collectionMethod:
				'Diagnostics-only desktop visual freeze map: navigate to / with analysis, gridResolution, utciRender=auto, and utciRenderDiagnostics=1; browser init script records RAF gaps, event-loop interval gaps, long tasks, WebGPU requestDevice/uncapturederror/device-lost events, and lifecycle marks; Playwright polls __utciRenderDiagnostics__ roughly every 500ms while waiting only for the strong main-route selected-hour publication contract.',
			pollIntervalMs: DIAGNOSTICS_POLL_INTERVAL_MS,
			cases
		};
		const diagnosticArtifact = buildFocusedDiagnosticArtifact(artifact);

		if (!existsSync(RESULTS_DIR)) {
			mkdirSync(RESULTS_DIR, { recursive: true });
		}

		const json = JSON.stringify(artifact, null, 2);
		const diagnosticJson = JSON.stringify(diagnosticArtifact, null, 2);
		writeFileSync(ARTIFACT_PATH, json, 'utf8');
		writeFileSync(DIAGNOSTIC_ARTIFACT_PATH, diagnosticJson, 'utf8');
		await testInfo.attach(ARTIFACT_FILENAME, {
			body: json,
			contentType: 'application/json'
		});
		await testInfo.attach(DIAGNOSTIC_ARTIFACT_FILENAME, {
			body: diagnosticJson,
			contentType: 'application/json'
		});

		const failedCases = cases.filter((entry) => !entry.publicationReached);
		if (failedCases.length > 0) {
			throw new Error(
				`Strong selected-hour publication was not reached for: ${failedCases
					.map((entry) => entry.caseId)
					.join(', ')}. Freeze-map artifact was written before this failure.`
			);
		}
	});
});
