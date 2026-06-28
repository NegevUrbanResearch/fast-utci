import { expect, test, type Page } from '@playwright/test';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const RESULTS_DIR = resolve(REPO_ROOT, 'data/performance-results');
const ARTIFACT_PATH = resolve(RESULTS_DIR, 'main-route-selected-hour-current-head.json');
const COLLECTED_ON = '2026-05-15';
const SOURCE_ROUTE = '/';

type AnalysisCase = {
	projectLabel: string;
	analysisId: string;
	metadataPath: string;
	expectedSelectionKey: string;
};

type AnalysisMetadata = {
	grid_size?: number;
	num_positions?: number;
};

type CollectedCase = {
	projectLabel: string;
	analysisId: string;
	pointCount: number;
	gridSizeMeters: number;
	selectedMonthIndex: number;
	selectedHourIndex: number;
	selectedTimeIndex: number;
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
		metadataPath: 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday.json',
		expectedSelectionKey: 'Ben-Gurion/20250815_grid_2m_fullday|7|0'
	},
	{
		projectLabel: 'Ness-Tziona',
		analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		metadataPath: 'data/analyses/Ness-Tziona/exploded/nes_tziona_unblock_2.json',
		expectedSelectionKey: 'Ness-Tziona/exploded/nes_tziona_unblock_2|7|0'
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
				value.selectedHourRuntimeContract?.strongVisibleGpuPath === true
			) {
				return value;
			}
			return null;
		},
		expectedSelectionKey,
		{ timeout: 15_000 }
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

async function collectCase(
	page: Page,
	caseConfig: AnalysisCase
): Promise<CollectedCase> {
	const metadata = readMetadata(caseConfig);
	const requestedUrls: string[] = [];
	page.on('request', (request) => requestedUrls.push(request.url()));

	const sourceUrl = `/?analysis=${encodeURIComponent(caseConfig.analysisId)}&utciRender=auto&utciRenderDiagnostics=1`;
	await page.goto(sourceUrl);
	const diagnostics = await waitForSelectedHourPublication(page, caseConfig.expectedSelectionKey);
	const forbiddenComparisonFieldsPresent = collectForbiddenComparisonFields(diagnostics);
	const forbiddenRequestUrls = requestedUrls.filter(isForbiddenComparisonRequest);
	const trackedGpuAllocationBytes = diagnostics.trackedGpuAllocationBytes;

	expect(forbiddenComparisonFieldsPresent).toEqual([]);
	expect(forbiddenRequestUrls).toEqual([]);
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
		pointCount: metadata.num_positions ?? 0,
		gridSizeMeters: metadata.grid_size ?? 0,
		selectedMonthIndex: diagnostics.baseSelectedMonthIndex,
		selectedHourIndex: diagnostics.baseSelectedHourIndex,
		selectedTimeIndex: diagnostics.baseSelectedTimeIndex,
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
		},
		assertions: {
			pythonBinDebugComparisonFieldsAbsent: true,
			forbiddenComparisonFieldsPresent,
			forbiddenRequestUrls,
			memoryScope: trackedGpuAllocationBytes.trackingScope
		}
	};
}

test.describe('main route performance baseline collector', () => {
	test.afterEach(async ({ page }) => {
		await page.goto('about:blank').catch(() => undefined);
	});

	test('collects the fresh BG base and Ness Tziona main-route timing baseline', async ({
		page
	}, testInfo) => {
		test.setTimeout(45_000);

		const cases: CollectedCase[] = [];
		for (const caseConfig of CASES) {
			cases.push(await collectCase(page, caseConfig));
		}

		const artifact = {
			collectedOn: COLLECTED_ON,
			sourceRoute: SOURCE_ROUTE,
			includedAnalyses: CASES.map((entry) => entry.analysisId),
			excludedBgVariantsExplanation:
				'This baseline intentionally excludes other Ben-Gurion variants so the current-head timing pass stays limited to the BG 2m base case and Ness Tziona 2m.',
			cases
		};

		if (!existsSync(RESULTS_DIR)) {
			mkdirSync(RESULTS_DIR, { recursive: true });
		}

		const json = JSON.stringify(artifact, null, 2);
		writeFileSync(ARTIFACT_PATH, json, 'utf8');
		await testInfo.attach('main-route-selected-hour-current-head.json', {
			body: json,
			contentType: 'application/json'
		});
	});
});
