import { expect, test, type Page } from '@playwright/test';

const BEN_GURION_BASE_ANALYSIS_ID = 'Ben-Gurion/20250815_grid_2m_fullday';
const UTCI_SELECTION_PREFIX = `${BEN_GURION_BASE_ANALYSIS_ID}|utci`;
const SHADING_INDEX_SELECTION_PREFIX = `${BEN_GURION_BASE_ANALYSIS_ID}|shading_index`;

type MainRouteDiagnostics = {
	rendererBackend?: string;
	utciRenderRequested?: string;
	utciRenderResolved?: string;
	utciSurfaceSource?: string;
	dataTextureBuildCount?: number;
	baseRenderTransport?: string;
	baseSameDeviceForComputeAndRender?: boolean | null;
	baseLiveReady?: boolean;
	baseSurfaceRequestId?: number;
	gpuResidentCopyRequestId?: number;
	baseSelectionKey?: string;
	baseSceneSelectionKey?: string;
	baseSelectedMonthIndex?: number;
	baseSelectedHourIndex?: number;
	baseSelectedTimeIndex?: number;
	baseColorMode?: string;
	tooltipInteraction?: {
		hoverSampleCount?: number;
		sampleCount?: number;
		hitCount?: number;
		metricPointReadbackCount?: number;
		metricPointReadbackBytes?: number;
		metricPointReadbackLastBytes?: number | null;
		metricPointReadbackCacheEntries?: number;
		metricPointReadbackCacheHitCount?: number;
		metricPointReadbackCacheMissCount?: number;
		metricPointReadbackLastLatencyMs?: number | null;
		metricPointReadbackMaxLatencyMs?: number;
	};
	timings?: {
		shadingIndexDispatchMs?: number;
		shadingIndexQueueWaitMs?: number;
		shadingIndexOutputBytes?: number;
		shadingIndexSnapshotBytes?: number;
		renderPublication?: {
			renderPublicationTimeline?: {
				sessionMetricType?: string;
				sessionMetricPeriodKind?: string;
				sessionMetricPeriodIndex?: number;
				sessionMetricPeriodStartTimeIndex?: number;
				sessionMetricPeriodTimeCount?: number;
				sessionOutputBytes?: number;
				sessionCompactSummaryBytes?: number;
				sessionShadingIndexDispatchMs?: number;
				sessionShadingIndexQueueWaitMs?: number;
				sessionShadingIndexOutputBytes?: number;
				sessionShadingIndexSnapshotBytes?: number;
				sessionShadingIndexSource?: string;
				sessionShadingIndexMonthCacheHit?: boolean;
				sessionFullSolarReadbackCount?: number;
				sessionTooltipPointReadbackCount?: number;
				sessionTooltipPointReadbackBytes?: number;
				sessionTooltipPointReadbackCacheHitCount?: number;
				sessionTooltipPointReadbackCacheMissCount?: number;
				sessionTooltipPointReadbackLastLatencyMs?: number | null;
				sessionTooltipPointReadbackMaxLatencyMs?: number;
				sessionSelectedHourRangeResolutionPath?: string;
				sessionSelectedHourRangeSummaryReadbackBytes?: number;
				sessionSelectedHourRangeSummaryReadbackCount?: number;
				sessionSelectedHourRangeFullReadbackAvoidedCount?: number;
			};
		};
	};
	selectedHourRuntimeContract?: {
		route?: string;
		selectedHourEngine?: string;
		renderTransport?: string;
		utciSurfaceSource?: string;
		strongVisibleGpuPath?: boolean;
		visibleSelectedHourReadbackCount?: number;
		readbackInstrumentation?: string;
		dataTextureBuildCount?: number;
	};
};

type TooltipProbe = {
	clientX: number;
	clientY: number;
	positionIndex: number;
};

async function readDiagnostics(page: Page): Promise<MainRouteDiagnostics | null> {
	return page.evaluate(() => (window as any).__utciRenderDiagnostics__ ?? null);
}

async function skipIfWebgpuUnavailable(page: Page): Promise<void> {
	const navigatorGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
	const requireWebgpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

	test.skip(
		!navigatorGpuAvailable && !requireWebgpu,
		'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
	);
}

async function waitForDiagnostics(
	page: Page,
	predicateSource: string,
	errorLabel: string,
	args: Record<string, unknown> = {},
	timeoutMs = 90_000
): Promise<MainRouteDiagnostics> {
	const handle = await page
		.waitForFunction(
			({ predicateSource: source, args: predicateArgs }) => {
				const value = (window as any).__utciRenderDiagnostics__;
				if (!value) return null;
				const predicate = new Function(
					'value',
					'args',
					`return (${source})(value, args);`
				) as (value: unknown, args: unknown) => boolean;
				return predicate(value, predicateArgs) ? value : null;
			},
			{ predicateSource, args },
			{ timeout: timeoutMs }
		)
		.catch(async (error) => {
			const lastDiagnostics = await readDiagnostics(page).catch((readError) => ({
				readError: readError instanceof Error ? readError.message : String(readError)
			}));
			const message = error instanceof Error ? error.message : String(error);
			throw new Error(
				[
					errorLabel,
					message,
					'Last window.__utciRenderDiagnostics__:',
					JSON.stringify(lastDiagnostics, null, 2)
				].join('\n')
			);
		});
	return handle.jsonValue() as Promise<MainRouteDiagnostics>;
}

async function waitForUtcSelectedHourPublication(
	page: Page,
	options: {
		monthIndex?: number;
		hourIndex?: number;
		previousRequestId?: number;
		colorMode?: 'normalized' | 'discrete';
	} = {}
): Promise<MainRouteDiagnostics> {
	const monthIndex = options.monthIndex ?? 7;
	const hourIndex = options.hourIndex ?? 0;
	return waitForDiagnostics(
		page,
		`(value, args) => {
			const timeline = value.timings?.renderPublication?.renderPublicationTimeline;
			const minSurfaceRequestId = args.previousRequestId ?? 0;
			return value.rendererBackend === 'webgpu' &&
				value.utciRenderRequested === 'auto' &&
				value.utciRenderResolved === 'gpuNative' &&
				value.baseLiveReady === true &&
				value.baseRenderTransport === 'compute-buffer-selected-hour' &&
				value.utciSurfaceSource === 'compute-buffer-selected-hour' &&
				value.baseSameDeviceForComputeAndRender === true &&
				value.dataTextureBuildCount === 0 &&
				value.baseSelectionKey === args.expectedSelectionKey &&
				value.baseSceneSelectionKey === args.expectedSelectionKey &&
				value.baseSelectedMonthIndex === args.monthIndex &&
				value.baseSelectedHourIndex === args.hourIndex &&
				value.baseSelectedTimeIndex === args.monthIndex * 24 + args.hourIndex &&
				(args.colorMode == null || value.baseColorMode === args.colorMode) &&
				typeof value.baseSurfaceRequestId === 'number' &&
				value.baseSurfaceRequestId > minSurfaceRequestId &&
				value.gpuResidentCopyRequestId === value.baseSurfaceRequestId &&
				value.selectedHourRuntimeContract?.route === 'main' &&
				value.selectedHourRuntimeContract?.selectedHourEngine === 'shared-host' &&
				value.selectedHourRuntimeContract?.renderTransport === 'compute-buffer-selected-hour' &&
				value.selectedHourRuntimeContract?.utciSurfaceSource === 'compute-buffer-selected-hour' &&
				value.selectedHourRuntimeContract?.strongVisibleGpuPath === true &&
				value.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount === 0 &&
				value.selectedHourRuntimeContract?.dataTextureBuildCount === 0 &&
				timeline?.sessionMetricType === 'utci';
		}`,
		'Timed out waiting for live WebGPU UTCI selected-hour publication.',
		{
			monthIndex,
			hourIndex,
			previousRequestId: options.previousRequestId,
			colorMode: options.colorMode,
			expectedSelectionKey: `${UTCI_SELECTION_PREFIX}|${monthIndex}|${hourIndex}`
		}
	);
}

async function waitForUtcCompactRangeProof(
	page: Page,
	options: {
		monthIndex?: number;
		hourIndex?: number;
		previousRequestId?: number;
		colorMode?: 'normalized' | 'discrete';
	} = {}
): Promise<MainRouteDiagnostics> {
	const monthIndex = options.monthIndex ?? 7;
	const hourIndex = options.hourIndex ?? 0;
	return waitForDiagnostics(
		page,
		`(value, args) => {
			const timeline = value.timings?.renderPublication?.renderPublicationTimeline;
			const minSurfaceRequestId = args.previousRequestId ?? 0;
			return value.rendererBackend === 'webgpu' &&
				value.utciRenderRequested === 'auto' &&
				value.utciRenderResolved === 'gpuNative' &&
				value.baseLiveReady === true &&
				value.baseRenderTransport === 'compute-buffer-selected-hour' &&
				value.utciSurfaceSource === 'compute-buffer-selected-hour' &&
				value.baseSameDeviceForComputeAndRender === true &&
				value.dataTextureBuildCount === 0 &&
				value.baseSelectionKey === args.expectedSelectionKey &&
				value.baseSceneSelectionKey === args.expectedSelectionKey &&
				value.baseSelectedMonthIndex === args.monthIndex &&
				value.baseSelectedHourIndex === args.hourIndex &&
				value.baseSelectedTimeIndex === args.monthIndex * 24 + args.hourIndex &&
				(args.colorMode == null || value.baseColorMode === args.colorMode) &&
				typeof value.baseSurfaceRequestId === 'number' &&
				value.baseSurfaceRequestId > minSurfaceRequestId &&
				value.gpuResidentCopyRequestId === value.baseSurfaceRequestId &&
				value.selectedHourRuntimeContract?.route === 'main' &&
				value.selectedHourRuntimeContract?.selectedHourEngine === 'shared-host' &&
				value.selectedHourRuntimeContract?.renderTransport === 'compute-buffer-selected-hour' &&
				value.selectedHourRuntimeContract?.utciSurfaceSource === 'compute-buffer-selected-hour' &&
				value.selectedHourRuntimeContract?.strongVisibleGpuPath === true &&
				value.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount === 0 &&
				value.selectedHourRuntimeContract?.dataTextureBuildCount === 0 &&
				timeline?.sessionMetricType === 'utci' &&
				timeline?.sessionSelectedHourRangeResolutionPath === 'compact-gpu-summary' &&
				timeline?.sessionSelectedHourRangeSummaryReadbackBytes === 16 &&
				timeline?.sessionSelectedHourRangeSummaryReadbackCount >= 1 &&
				timeline?.sessionSelectedHourRangeFullReadbackAvoidedCount >= 1;
		}`,
		'Timed out waiting for UTCI compact range-summary proof.',
		{
			monthIndex,
			hourIndex,
			previousRequestId: options.previousRequestId,
			colorMode: options.colorMode,
			expectedSelectionKey: `${UTCI_SELECTION_PREFIX}|${monthIndex}|${hourIndex}`
		}
	);
}

async function waitForShadingIndexPublication(
	page: Page,
	options: { monthIndex: number; previousRequestId?: number } = { monthIndex: 7 }
): Promise<MainRouteDiagnostics> {
	return waitForDiagnostics(
		page,
		`(value, args) => {
			const timeline = value.timings?.renderPublication?.renderPublicationTimeline;
			const expectedSelectionKey = args.expectedSelectionKey;
			return value.rendererBackend === 'webgpu' &&
				value.utciRenderRequested === 'auto' &&
				value.utciRenderResolved === 'gpuNative' &&
				value.baseLiveReady === true &&
				value.baseRenderTransport === 'compute-buffer-selected-hour' &&
				value.utciSurfaceSource === 'compute-buffer-selected-hour' &&
				value.baseSameDeviceForComputeAndRender === true &&
				value.dataTextureBuildCount === 0 &&
				value.baseSelectionKey === expectedSelectionKey &&
				value.baseSceneSelectionKey === expectedSelectionKey &&
				value.baseSelectedMonthIndex === args.monthIndex &&
				value.baseSelectedHourIndex === 0 &&
				value.baseSelectedTimeIndex === args.monthIndex * 24 &&
				typeof value.baseSurfaceRequestId === 'number' &&
				(typeof args.previousRequestId !== 'number' ||
					value.baseSurfaceRequestId > args.previousRequestId) &&
				value.selectedHourRuntimeContract?.route === 'main' &&
				value.selectedHourRuntimeContract?.selectedHourEngine === 'shared-host' &&
				value.selectedHourRuntimeContract?.renderTransport === 'compute-buffer-selected-hour' &&
				value.selectedHourRuntimeContract?.utciSurfaceSource === 'compute-buffer-selected-hour' &&
				value.selectedHourRuntimeContract?.strongVisibleGpuPath === true &&
				value.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount === 0 &&
				value.selectedHourRuntimeContract?.dataTextureBuildCount === 0 &&
				timeline?.sessionMetricType === 'shading_index' &&
				timeline?.sessionMetricPeriodKind === 'month-index' &&
				timeline?.sessionMetricPeriodIndex === args.monthIndex &&
				timeline?.sessionMetricPeriodStartTimeIndex === args.monthIndex * 24 &&
				timeline?.sessionMetricPeriodTimeCount === 24 &&
				timeline?.sessionOutputBytes > 4 &&
				timeline?.sessionCompactSummaryBytes === 0 &&
				timeline?.sessionShadingIndexDispatchMs >= 0 &&
				timeline?.sessionShadingIndexQueueWaitMs >= 0 &&
				timeline?.sessionShadingIndexOutputBytes === timeline?.sessionOutputBytes &&
				timeline?.sessionShadingIndexSnapshotBytes === timeline?.sessionOutputBytes &&
				timeline?.sessionShadingIndexSource === 'fresh-dispatch' &&
				timeline?.sessionShadingIndexMonthCacheHit === false &&
				value.timings?.shadingIndexDispatchMs >= 0 &&
				value.timings?.shadingIndexQueueWaitMs >= 0 &&
				value.timings?.shadingIndexOutputBytes === timeline?.sessionOutputBytes &&
				value.timings?.shadingIndexSnapshotBytes === timeline?.sessionOutputBytes &&
				timeline?.sessionFullSolarReadbackCount === 0 &&
				typeof timeline?.sessionTooltipPointReadbackCount === 'number' &&
				typeof timeline?.sessionTooltipPointReadbackBytes === 'number';
		}`,
		'Timed out waiting for live WebGPU Shading Index publication.',
		{
			monthIndex: options.monthIndex,
			previousRequestId: options.previousRequestId,
			expectedSelectionKey: `${SHADING_INDEX_SELECTION_PREFIX}|${options.monthIndex}`
		}
	);
}

async function setStaleSelectedHourState(page: Page, hourIndex: number): Promise<number> {
	return page.evaluate(async (nextHourIndex) => {
		const { setCurrentHour, viewerStore } = (await new Function(
			'return import("/src/lib/stores/viewerStore.ts")'
		)()) as typeof import('$lib/stores/viewerStore');
		setCurrentHour(nextHourIndex);
		await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
		let currentHour = -1;
		const unsubscribe = viewerStore.subscribe((state) => {
			currentHour = state.currentHour;
		});
		unsubscribe();
		return currentHour;
	}, hourIndex);
}

async function setColorScaleMode(page: Page, mode: 'full day' | 'per hour'): Promise<void> {
	const button = page.getByRole('button', { name: new RegExp(`^${mode}$`, 'i') });
	await expect(button).toBeVisible();
	await button.click();
	await expect(button).toHaveAttribute('aria-pressed', 'true');
}

async function setMonthSelection(page: Page, monthIndex: number, monthLabel: string): Promise<void> {
	const slider = page.getByRole('slider', { name: /select month/i });
	await expect(slider).toBeVisible();
	await slider.click();
	await slider.focus();
	await expect(slider).toBeFocused();

	await slider.press('Home');
	for (let step = 0; step < monthIndex; step += 1) {
		await slider.press('ArrowRight');
	}

	await expect(slider).toHaveAttribute('aria-valuenow', String(monthIndex));
	await expect(slider).toHaveAttribute('aria-valuetext', new RegExp(`Month\\s+${monthLabel}`, 'i'));
}

async function readMainRouteTooltipProbe(page: Page): Promise<TooltipProbe> {
	const probe = await page
		.waitForFunction(() => {
			const resolver = (window as any).__mainRouteTooltipProbePosition__;
			return typeof resolver === 'function' ? resolver() : null;
		}, undefined, { timeout: 15_000 })
		.catch(async (error) => {
			const diagnostics = await readDiagnostics(page);
			throw new Error(
				[
					'Timed out waiting for a current main-route tooltip probe point.',
					error instanceof Error ? error.message : String(error),
					JSON.stringify(diagnostics, null, 2)
				].join('\n')
			);
		});
	return probe.jsonValue() as Promise<TooltipProbe>;
}

async function hoverUntilShadingTooltip(page: Page, probe: TooltipProbe): Promise<void> {
	for (const [dx, dy] of [
		[0, 0],
		[3, 0],
		[0, 3],
		[-3, 0],
		[0, -3],
		[6, 0],
		[0, 6]
	] as const) {
		await page.mouse.move(probe.clientX + dx, probe.clientY + dy);
		await page.waitForTimeout(48);
		if (await page.getByRole('tooltip').isVisible().catch(() => false)) {
			await expect(page.getByRole('tooltip')).toContainText(/Shading Index/i);
			await expect(page.getByRole('tooltip')).toContainText(
				/Poor Shading|Acceptable Shading|Good Shading|Excellent Shading/i
			);
			return;
		}
	}

	throw new Error('Expected a visible Shading Index tooltip over the live shaded surface.');
}

async function waitForPointSizedTooltipReadback(
	page: Page,
	before: MainRouteDiagnostics
): Promise<MainRouteDiagnostics> {
	return waitForDiagnostics(
		page,
		`(value, args) => {
			const tooltip = value.tooltipInteraction;
			return tooltip?.hitCount > (args.hitCount ?? 0) &&
				tooltip?.metricPointReadbackCount > (args.readbackCount ?? 0) &&
				tooltip?.metricPointReadbackBytes > (args.readbackBytes ?? 0) &&
				tooltip?.metricPointReadbackLastBytes === 4 &&
				tooltip?.metricPointReadbackCacheEntries >= 1 &&
				tooltip?.metricPointReadbackCacheMissCount > (args.cacheMissCount ?? 0) &&
				tooltip?.metricPointReadbackLastLatencyMs >= 0 &&
				tooltip?.metricPointReadbackMaxLatencyMs >= tooltip?.metricPointReadbackLastLatencyMs;
		}`,
		'Timed out waiting for point-sized Shading Index tooltip readback diagnostics.',
		{
			hitCount: before.tooltipInteraction?.hitCount ?? 0,
			readbackCount: before.tooltipInteraction?.metricPointReadbackCount ?? 0,
			readbackBytes: before.tooltipInteraction?.metricPointReadbackBytes ?? 0,
			cacheMissCount: before.tooltipInteraction?.metricPointReadbackCacheMissCount ?? 0
		},
		15_000
	);
}

function expectGpuOnlyMainRouteShadingSource(
	diagnostics: MainRouteDiagnostics,
	newBinRequests: string[]
): void {
	expect(newBinRequests).toEqual([]);
	expect(diagnostics.utciSurfaceSource).toBe('compute-buffer-selected-hour');
	expect(diagnostics.baseRenderTransport).toBe('compute-buffer-selected-hour');
	expect(diagnostics.selectedHourRuntimeContract).toMatchObject({
		route: 'main',
		selectedHourEngine: 'shared-host',
		renderTransport: 'compute-buffer-selected-hour',
		utciSurfaceSource: 'compute-buffer-selected-hour',
		strongVisibleGpuPath: true,
		visibleSelectedHourReadbackCount: 0,
		dataTextureBuildCount: 0
	});
}

test.describe('live WebGPU Shading Index proof', () => {
	test('main route renders Shading Index from GPU-resident WebGPU output without full readback', async ({
		page
	}) => {
		test.setTimeout(120_000);
		const binRequests: string[] = [];
		page.on('request', (request) => {
			if (/\.bin(?:$|[?#])/.test(request.url())) {
				binRequests.push(request.url());
			}
		});

		await page.goto(
			`/?analysis=${encodeURIComponent(BEN_GURION_BASE_ANALYSIS_ID)}&utciRender=auto&utciRenderDiagnostics=1`
		);
		await skipIfWebgpuUnavailable(page);

		const initialUtciDiagnostics = await waitForUtcSelectedHourPublication(page, {
			colorMode: 'normalized'
		});
		await setColorScaleMode(page, 'per hour');
		const rangeProof = await waitForUtcCompactRangeProof(page, {
			previousRequestId: initialUtciDiagnostics.baseSurfaceRequestId,
			colorMode: 'discrete'
		});
		const dataTextureBuildCountBefore = rangeProof.dataTextureBuildCount ?? 0;
		const binRequestCountBeforeShading = binRequests.length;

		await page.getByRole('button', { name: /^Shading$/i }).click();
		await expect(page.getByRole('button', { name: /^Shading$/i })).toHaveAttribute(
			'aria-pressed',
			'true'
		);
		await expect(page.getByRole('button', { name: /^Day$/i })).toHaveCount(0);
		await expect(page.getByRole('slider', { name: /select analysis hour/i })).toHaveCount(0);
		await expect(page.getByRole('slider', { name: /select month/i })).toBeVisible();
		await expect(page.getByText('Shading Index').first()).toBeVisible();
		await expect(page.getByText('Poor').first()).toBeVisible();
		await expect(page.getByText('Acceptable').first()).toBeVisible();
		await expect(page.getByText('Good').first()).toBeVisible();
		await expect(page.getByText('Excellent').first()).toBeVisible();

		const shadingDiagnostics = await waitForShadingIndexPublication(page, {
			monthIndex: 7,
			previousRequestId: rangeProof.baseSurfaceRequestId
		});
		expect(shadingDiagnostics.dataTextureBuildCount ?? 0).toBe(dataTextureBuildCountBefore);
		expectGpuOnlyMainRouteShadingSource(
			shadingDiagnostics,
			binRequests.slice(binRequestCountBeforeShading)
		);
		expect(
			shadingDiagnostics.timings?.renderPublication?.renderPublicationTimeline
				?.sessionTooltipPointReadbackCount
		).toBe(0);
		expect(
			shadingDiagnostics.timings?.renderPublication?.renderPublicationTimeline
				?.sessionTooltipPointReadbackBytes
		).toBe(0);
		const staleHourIndex = await setStaleSelectedHourState(page, 17);
		expect(staleHourIndex).toBe(17);
		const staleHourDiagnostics = await readDiagnostics(page);
		expect(staleHourDiagnostics).toMatchObject({
			baseSelectionKey: `${SHADING_INDEX_SELECTION_PREFIX}|7`,
			baseSceneSelectionKey: `${SHADING_INDEX_SELECTION_PREFIX}|7`,
			baseSelectedMonthIndex: 7,
			baseSelectedHourIndex: 0,
			baseSelectedTimeIndex: 7 * 24
		});
		expect(staleHourDiagnostics?.baseSurfaceRequestId).toBe(
			shadingDiagnostics.baseSurfaceRequestId
		);
		expect(staleHourDiagnostics?.gpuResidentCopyRequestId).toBe(
			shadingDiagnostics.gpuResidentCopyRequestId
		);
		expect(
			staleHourDiagnostics?.timings?.renderPublication?.renderPublicationTimeline
		).toMatchObject({
			sessionMetricType: 'shading_index',
			sessionMetricPeriodKind: 'month-index',
			sessionMetricPeriodIndex: 7,
			sessionMetricPeriodStartTimeIndex: 7 * 24,
			sessionMetricPeriodTimeCount: 24
		});

		const tooltipProbe = await readMainRouteTooltipProbe(page);
		await hoverUntilShadingTooltip(page, tooltipProbe);
		const tooltipDiagnostics = await waitForPointSizedTooltipReadback(
			page,
			shadingDiagnostics
		);
		const timeline =
			tooltipDiagnostics.timings?.renderPublication?.renderPublicationTimeline;
		expect(tooltipDiagnostics.tooltipInteraction?.metricPointReadbackLastBytes).toBe(4);
		expect(tooltipDiagnostics.tooltipInteraction?.metricPointReadbackBytes ?? 0).toBeLessThan(
			timeline?.sessionOutputBytes ?? 0
		);
		expect(timeline?.sessionFullSolarReadbackCount).toBe(0);
		expect(timeline?.sessionTooltipPointReadbackCount).toBe(
			tooltipDiagnostics.tooltipInteraction?.metricPointReadbackCount
		);
		expect(timeline?.sessionTooltipPointReadbackBytes).toBe(
			tooltipDiagnostics.tooltipInteraction?.metricPointReadbackBytes
		);
		expect(timeline?.sessionTooltipPointReadbackCacheMissCount).toBeGreaterThan(0);
		expect(timeline?.sessionTooltipPointReadbackLastLatencyMs).toEqual(expect.any(Number));

		await setMonthSelection(page, 8, 'Sep');
		const septemberDiagnostics = await waitForShadingIndexPublication(page, {
			monthIndex: 8,
			previousRequestId: shadingDiagnostics.baseSurfaceRequestId
		});
		expect(septemberDiagnostics.dataTextureBuildCount ?? 0).toBe(dataTextureBuildCountBefore);
		expectGpuOnlyMainRouteShadingSource(
			septemberDiagnostics,
			binRequests.slice(binRequestCountBeforeShading)
		);
	});
});
