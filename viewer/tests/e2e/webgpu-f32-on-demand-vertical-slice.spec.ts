import { expect, test, type Page } from '@playwright/test';

type StrictExposureOnlyDiagnostics = {
	navigatorGpu?: boolean;
	rendererBackend?: string;
	utciRenderRequested?: string;
	utciRenderResolved?: string;
	utciOnDemand?: string;
	utciSurfaceSource?: string;
	selectedHourTransferCount?: number;
	path?: string;
	usedExposureOnlyPrecompute?: boolean;
	usedRunAllForSelectedHour?: boolean;
	liveAnalysisConstructedForSelectedHour?: boolean;
	allHoursUtciBytesAllocated?: number;
	allHoursMrtBytesAllocated?: number;
	oneHourOutputBytes?: number;
	debugReadbackCount?: number;
	dataTextureBuildCount?: number;
	bridgeAttached?: boolean;
	visibleColorVariance?: number;
	debugComparisonReference?: string;
	pythonBinComparisonActive?: boolean;
	debugComparisonMonthIndex?: number;
	pythonComparisonHourIndex?: number;
	webgpuComparisonHourIndex?: number;
	pythonBinSampleComparison?: {
		numCompared?: number;
		maxAbsDiff?: number;
		samples?: Array<{
			pointIndex: number;
			debugValue: number;
			referenceValue: number;
			absDiff: number;
		}>;
	};
	timings?: {
		exposurePrecomputeMs?: number;
		oneHourDispatchMs?: number;
		debugReadbackMs?: number;
	};
};

type MultiHourComparisonHourResult = {
	hour: number;
	numCompared: number;
	maxAbsDiff: number;
	rmse: number;
	onDemandAt31079?: number;
	baselineAt31079?: number;
	diffAt31079?: number;
};

type MultiHourComparisonKnownPoint = {
	pointIndex: number;
	hours: Array<{
		hour: number;
		onDemand: number;
		baseline: number;
		diff: number;
	}>;
};

type MultiHourComparisonResult = {
	baselineSource: string;
	baselineMonthContext?: {
		monthIndex: number;
		sliceKind: string;
		note: string;
	};
	strictPath: StrictExposureOnlyDiagnostics;
	hours: number[];
	hourResults: MultiHourComparisonHourResult[];
	knownPoint31079?: MultiHourComparisonKnownPoint;
};

const KNOWN_LOCALIZED_HOTSPOT_POINT_INDEX = 31079;
const NON_HOTSPOT_SAMPLE_TOLERANCE = 0.05;
const KNOWN_HOTSPOT_MIN_ABS_DIFF = 1.7;
const KNOWN_HOTSPOT_MAX_ABS_DIFF = 1.95;

async function readDiagnostics(page: Page) {
	return page.evaluate(() => {
		return (window as Window & {
			__onDemandPrototypeDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__onDemandPrototypeDiagnostics__;
	});
}

async function readMultiHourComparison(page: Page) {
	return page.evaluate(() => {
		return (window as Window & {
			__onDemandMultiHourComparison__?: MultiHourComparisonResult;
		}).__onDemandMultiHourComparison__;
	});
}

async function switchQueryMode(
	page: Page,
	params: Record<string, string | null>
) {
	await page.evaluate((nextParams) => {
		const nextUrl = new URL(window.location.href);
		for (const [key, value] of Object.entries(nextParams)) {
			if (value === null) {
				nextUrl.searchParams.delete(key);
			} else {
				nextUrl.searchParams.set(key, value);
			}
		}
		window.history.pushState({}, '', nextUrl);
		window.dispatchEvent(new PopStateEvent('popstate'));
	}, params);
}

test('strict static-upload path computes one selected hour without constructing live analysis', async ({
	page
}) => {
	await page.goto(
		'/debug-webgpu-utci?onDemandPrototype=1&strictExposureOnly=1&timeIndex=12'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

	test.skip(
		!hasWebGpu && !requireWebGpu,
		'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
	);

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 120_000
	});

	const diagnostics = await page.evaluate(() => {
		return (window as Window & {
			__onDemandPrototypeDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__onDemandPrototypeDiagnostics__;
	});

	expect(diagnostics).toBeTruthy();
	expect(diagnostics?.path).toBe('exposure-only-f32');
	expect(diagnostics?.usedExposureOnlyPrecompute).toBe(true);
	expect(diagnostics?.usedRunAllForSelectedHour).toBe(false);
	expect(diagnostics?.liveAnalysisConstructedForSelectedHour).toBe(false);
	expect(diagnostics?.allHoursUtciBytesAllocated).toBe(0);
	expect(diagnostics?.allHoursMrtBytesAllocated).toBe(0);
	expect(diagnostics?.oneHourOutputBytes ?? 0).toBeGreaterThan(0);
	expect(diagnostics?.debugReadbackCount).toBe(0);
	expect(diagnostics?.dataTextureBuildCount).toBe(0);
});

test('same-route synthetic bridge to strict mode clears stale bridge diagnostics', async ({
	page
}) => {
	await page.goto('/debug-webgpu-utci?onDemandPrototype=1&syntheticBridge=1');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

	test.skip(
		!hasWebGpu && !requireWebGpu,
		'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
	);

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & {
			__onDemandPrototypeDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__onDemandPrototypeDiagnostics__;
		return diagnostics?.bridgeAttached === true && (diagnostics.visibleColorVariance ?? 0) > 0;
	});

	await switchQueryMode(page, {
		syntheticBridge: null,
		strictExposureOnly: '1',
		timeIndex: '12'
	});

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 120_000
	});

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & {
			__onDemandPrototypeDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__onDemandPrototypeDiagnostics__;
		return (
			diagnostics?.path === 'exposure-only-f32' &&
			diagnostics?.usedExposureOnlyPrecompute === true &&
			diagnostics?.usedRunAllForSelectedHour === false &&
			diagnostics?.liveAnalysisConstructedForSelectedHour === false &&
			(diagnostics.oneHourOutputBytes ?? 0) > 0
		);
	});

	const diagnostics = await readDiagnostics(page);
	expect(diagnostics?.bridgeAttached).toBe(false);
	expect(diagnostics?.visibleColorVariance).toBe(0);
	expect(diagnostics?.debugReadbackCount).toBe(0);
	expect(diagnostics?.dataTextureBuildCount).toBe(0);
	expect(diagnostics?.path).toBe('exposure-only-f32');
	expect(diagnostics?.usedExposureOnlyPrecompute).toBe(true);
});

test('same-route compareOneHour to strict mode clears stale comparison counters', async ({
	page
}) => {
	test.setTimeout(120_000);
	await page.goto('/debug-webgpu-utci?parity=1&onDemandPrototype=1&compareOneHour=1');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

	test.skip(
		!hasWebGpu && !requireWebGpu,
		'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
	);

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & {
			__onDemandPrototypeDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__onDemandPrototypeDiagnostics__;
		return (diagnostics?.debugReadbackCount ?? 0) > 0;
	}, undefined, { timeout: 120_000 });

	await switchQueryMode(page, {
		compareOneHour: null,
		strictExposureOnly: '1',
		timeIndex: '12'
	});

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 120_000
	});

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & {
			__onDemandPrototypeDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__onDemandPrototypeDiagnostics__;
		return (
			diagnostics?.path === 'exposure-only-f32' &&
			diagnostics?.usedExposureOnlyPrecompute === true &&
			diagnostics?.usedRunAllForSelectedHour === false &&
			diagnostics?.liveAnalysisConstructedForSelectedHour === false &&
			(diagnostics.oneHourOutputBytes ?? 0) > 0 &&
			(diagnostics.timings?.exposurePrecomputeMs ?? 0) > 0 &&
			(diagnostics.timings?.oneHourDispatchMs ?? -1) >= 0 &&
			diagnostics.timings?.debugReadbackMs === undefined
		);
	});

	const diagnostics = await readDiagnostics(page);
	expect(diagnostics?.debugReadbackCount).toBe(0);
	expect(diagnostics?.dataTextureBuildCount).toBe(0);
	expect(diagnostics?.path).toBe('exposure-only-f32');
	expect(diagnostics?.usedExposureOnlyPrecompute).toBe(true);
	expect(diagnostics?.timings?.exposurePrecomputeMs ?? 0).toBeGreaterThan(0);
	expect(diagnostics?.timings?.oneHourDispatchMs ?? -1).toBeGreaterThanOrEqual(0);
	expect(diagnostics?.timings?.debugReadbackMs).toBeUndefined();
});

test('strict exposure-only compareHours matches a separate runAll baseline across multiple hours', async ({
	page
}) => {
	test.setTimeout(120_000);
	await page.goto(
		'/debug-webgpu-utci?onDemandPrototype=1&strictExposureOnly=1&compareHours=12,23,16,17&baseline=separateRunAll'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

	test.skip(
		!hasWebGpu && !requireWebGpu,
		'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
	);

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 120_000
	});

	const result = await readMultiHourComparison(page);
	const expectedHours = [12, 23, 16, 17];

	expect(result).toBeTruthy();
	expect(result?.strictPath.usedRunAllForSelectedHour).toBe(false);
	expect(result?.strictPath.debugReadbackCount).toBe(0);
	expect(result?.baselineSource).toBe('separateRunAll');
	expect(result?.baselineMonthContext).toEqual({
		monthIndex: 0,
		sliceKind: 'representative-day-full-year',
		note: 'compareHours uses the separate runAll baseline monthIndex 0 representative-day slice.'
	});
	expect(result?.hours).toEqual(expectedHours);
	expect(result?.hourResults).toHaveLength(expectedHours.length);
	expect(result?.hourResults.map((hourResult) => hourResult.hour)).toEqual(expectedHours);

	for (const hourResult of result?.hourResults ?? []) {
		expect(hourResult.numCompared).toBeGreaterThan(0);
		expect(hourResult.maxAbsDiff).toBeLessThanOrEqual(1e-5);
	}

	if (result?.knownPoint31079) {
		expect(result.knownPoint31079.pointIndex).toBe(31079);
		expect(result.knownPoint31079.hours).toEqual([
			{
				hour: 16,
				onDemand: expect.any(Number),
				baseline: expect.any(Number),
				diff: expect.any(Number)
			},
			{
				hour: 17,
				onDemand: expect.any(Number),
				baseline: expect.any(Number),
				diff: expect.any(Number)
			}
		]);
	}
});

test('strict exposure-only diagnostics publish timing fields on the window object', async ({
	page
}) => {
	test.setTimeout(120_000);
	await page.goto('/debug-webgpu-utci?onDemandPrototype=1&strictExposureOnly=1&timeIndex=12');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

	test.skip(
		!hasWebGpu && !requireWebGpu,
		'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
	);

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 120_000
	});

	const diagnostics = await readDiagnostics(page);
	expect(diagnostics).toBeTruthy();
	expect(diagnostics?.timings?.exposurePrecomputeMs ?? 0).toBeGreaterThan(0);
	expect(diagnostics?.timings?.oneHourDispatchMs ?? -1).toBeGreaterThanOrEqual(0);
});

test('main route publishes f32 on-demand diagnostics and preserves dataTexture fallback', async ({
	page
}) => {
	test.setTimeout(120_000);
	await page.goto('/?utciRenderDiagnostics=1&utciOnDemand=f32&utciRender=gpu');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

	test.skip(
		!hasWebGpu && !requireWebGpu,
		'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
	);

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & {
			__utciRenderDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__utciRenderDiagnostics__;

		return (
			diagnostics?.utciOnDemand === 'f32' &&
			diagnostics?.utciRenderResolved === 'gpuNative' &&
			typeof diagnostics?.utciSurfaceSource === 'string' &&
			typeof diagnostics?.selectedHourTransferCount === 'number' &&
			diagnostics?.dataTextureBuildCount === 0
		);
	});

	const gpuDiagnostics = await page.evaluate(() => {
		return (window as Window & {
			__utciRenderDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__utciRenderDiagnostics__;
	});

	expect(gpuDiagnostics).toBeTruthy();
	expect(gpuDiagnostics?.utciSurfaceSource).toMatch(/selected-hour/);
	expect(gpuDiagnostics?.dataTextureBuildCount).toBe(0);
	expect(gpuDiagnostics?.selectedHourTransferCount ?? -1).toBeGreaterThanOrEqual(0);

	await page.goto('/?utciRenderDiagnostics=1&utciOnDemand=f32&utciRender=data');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & {
			__utciRenderDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__utciRenderDiagnostics__;

		return (
			diagnostics?.utciRenderResolved === 'dataTexture' &&
			typeof diagnostics?.dataTextureBuildCount === 'number'
		);
	});

	const fallbackDiagnostics = await page.evaluate(() => {
		return (window as Window & {
			__utciRenderDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__utciRenderDiagnostics__;
	});

	expect(fallbackDiagnostics).toBeTruthy();
	expect(fallbackDiagnostics?.utciRenderResolved).toBe('dataTexture');
});

test('debug route can use f32 on-demand as the visible WebGPU side while keeping comparison active', async ({
	page
}) => {
	test.setTimeout(180_000);
	await page.goto(
		'/debug-webgpu-utci?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&timeIndex=12'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return (
			diagnostics?.path === 'exposure-only-f32' &&
			diagnostics?.usedExposureOnlyPrecompute === true &&
			diagnostics?.usedRunAllForSelectedHour === false &&
			diagnostics?.completedTimeIndex === 12 &&
			diagnostics?.pythonComparisonHourIndex === 12 &&
			diagnostics?.webgpuComparisonHourIndex === 12 &&
			diagnostics?.appVisibleSelectedHour === true
		);
	}, undefined, { timeout: 180_000 });

	await expect(page.getByRole('slider', { name: /select analysis hour/i })).toHaveAttribute(
		'aria-valuenow',
		'12'
	);
	await expect(page.getByRole('button', { name: /exit comparison mode/i })).toBeVisible();
	await expect(page.getByRole('slider', { name: /comparison curtain position/i })).toBeVisible();

	const diagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

	expect(diagnostics.renderTransport).toMatch(/selected-hour|none/);
	expect(diagnostics.debugComparisonReference).toBe('python-bin');
	expect(diagnostics.pythonBinComparisonActive).toBe(true);
	expect(diagnostics.pythonComparisonHourIndex).toBe(12);
	expect(diagnostics.webgpuComparisonHourIndex).toBe(12);
	expect(diagnostics.selectedHourReadbackCount).toBeLessThanOrEqual(1);
	expect(typeof diagnostics.dataTextureBuildCount === 'number' || diagnostics.dataTextureBuildCount === undefined).toBe(true);

	await switchQueryMode(page, {
		timeIndex: '13'
	});

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return (
			diagnostics?.path === 'exposure-only-f32' &&
			diagnostics?.usedExposureOnlyPrecompute === true &&
			diagnostics?.usedRunAllForSelectedHour === false &&
			diagnostics?.completedTimeIndex === 13 &&
			diagnostics?.pythonComparisonHourIndex === 13 &&
			diagnostics?.webgpuComparisonHourIndex === 13 &&
			diagnostics?.appVisibleSelectedHour === true
		);
	}, undefined, { timeout: 180_000 });

	await expect(page.getByRole('slider', { name: /select analysis hour/i })).toHaveAttribute(
		'aria-valuenow',
		'13'
	);
	await expect(page.getByRole('button', { name: /exit comparison mode/i })).toBeVisible();

	const switchedDiagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

	expect(switchedDiagnostics.pythonComparisonHourIndex).toBe(13);
	expect(switchedDiagnostics.webgpuComparisonHourIndex).toBe(13);
});

test('debug route preserves explicit dataTexture fallback when f32 on-demand is selected', async ({
	page
}) => {
	test.setTimeout(120_000);
	await page.goto(
		'/debug-webgpu-utci?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=data&timeIndex=12'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & {
			__onDemandPrototypeDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__onDemandPrototypeDiagnostics__;

		return (
			diagnostics?.path === 'exposure-only-f32' &&
			diagnostics?.usedExposureOnlyPrecompute === true &&
			diagnostics?.usedRunAllForSelectedHour === false &&
			diagnostics?.completedTimeIndex === 12 &&
			diagnostics?.pythonBinComparisonActive === true &&
			(diagnostics?.oneHourOutputBytes ?? 0) > 0 &&
			diagnostics?.allHoursUtciBytesAllocated === 0 &&
			diagnostics?.allHoursMrtBytesAllocated === 0 &&
			diagnostics?.utciRenderResolved === 'dataTexture' &&
			typeof diagnostics?.dataTextureBuildCount === 'number' &&
			(diagnostics.dataTextureBuildCount ?? 0) >= 1
		);
	}, undefined, { timeout: 120_000 });

	const diagnostics = await readDiagnostics(page);

	expect(diagnostics?.path).toBe('exposure-only-f32');
	expect(diagnostics?.usedExposureOnlyPrecompute).toBe(true);
	expect(diagnostics?.usedRunAllForSelectedHour).toBe(false);
	expect(diagnostics?.completedTimeIndex).toBe(12);
	expect(diagnostics?.pythonBinComparisonActive).toBe(true);
	expect(diagnostics?.oneHourOutputBytes ?? 0).toBeGreaterThan(0);
	expect(diagnostics?.allHoursUtciBytesAllocated).toBe(0);
	expect(diagnostics?.allHoursMrtBytesAllocated).toBe(0);
	expect(diagnostics?.utciRenderResolved).toBe('dataTexture');
	expect(diagnostics?.dataTextureBuildCount ?? 0).toBeGreaterThanOrEqual(1);
});

test('debug on-demand discards stale scrub results and ends on the final selected hour', async ({
	page
}) => {
	test.setTimeout(180_000);
	await page.goto(
		'/debug-webgpu-utci?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&timeIndex=12'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return diagnostics?.completedTimeIndex === 12;
	}, undefined, { timeout: 180_000 });

	await page.evaluate((nextTimeIndex) => {
		const url = new URL(window.location.href);
		url.searchParams.set('timeIndex', String(nextTimeIndex));
		url.searchParams.set('forceOnDemandPostAcceptDelayMs', '75');
		window.history.pushState({}, '', url);
		window.dispatchEvent(new PopStateEvent('popstate'));
	}, 13);

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return (
			diagnostics?.completedTimeIndex === 13 &&
			diagnostics?.pendingReadbackRequestId != null &&
			diagnostics?.pendingReadbackTimeIndex === 13
		);
	}, undefined, { timeout: 180_000 });

	await page.evaluate(async () => {
		const pushScrubSelection = async (nextTimeIndex: number) => {
			await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
			const url = new URL(window.location.href);
			url.searchParams.set('timeIndex', String(nextTimeIndex));
			url.searchParams.delete('forceOnDemandPostAcceptDelayMs');
			window.history.pushState({}, '', url);
			window.dispatchEvent(new PopStateEvent('popstate'));
		};

		for (const timeIndex of [16, 17, 23]) {
			await pushScrubSelection(timeIndex);
		}
	});

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		return (
			diagnostics?.selectedTimeIndex === 23 &&
			diagnostics?.completedTimeIndex === 23 &&
			diagnostics?.pythonComparisonHourIndex === 23 &&
			diagnostics?.webgpuComparisonHourIndex === 23 &&
			diagnostics?.inFlightCount === 0 &&
			diagnostics?.pendingReadbackRequestId == null &&
			(diagnostics?.selectedHourTransferCount ?? 0) > 0 &&
			(diagnostics?.timings?.renderUpdateMs ?? 0) > 0
		);
	}, undefined, { timeout: 180_000 });

	const diagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

	expect(diagnostics.usedRunAllForSelectedHour).toBe(false);
	expect(diagnostics.usedExposureOnlyPrecompute).toBe(true);
	expect(diagnostics.allHoursUtciBytesAllocated).toBe(0);
	expect(diagnostics.allHoursMrtBytesAllocated).toBe(0);
	expect(diagnostics.trackedGpuAllocationBytes.allHoursOutputBytes).toBe(0);
	expect(diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes).toBeGreaterThan(0);
	expect(diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark).toBe(
		diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes
	);
	expect(diagnostics.dataTextureBuildCount).toBe(0);
	expect(diagnostics.selectedTimeIndex).toBe(23);
	expect(diagnostics.completedTimeIndex).toBe(23);
	expect(diagnostics.pythonComparisonHourIndex).toBe(23);
	expect(diagnostics.webgpuComparisonHourIndex).toBe(23);
	expect(diagnostics.selectedHourTransferCount).toBeGreaterThan(0);
	expect(diagnostics.timings.renderUpdateMs).toBeGreaterThan(0);
	expect(diagnostics.scrubSampleCount).toBeGreaterThanOrEqual(4);
	expect(diagnostics.pendingReadbackRequestId).toBeUndefined();
	expect(diagnostics.staleResultDiscardCount).toBeGreaterThan(0);
});

test('debug on-demand publishes sampled python-bin comparison metadata for the selected hour', async ({
	page
}) => {
	test.setTimeout(180_000);
	await page.goto(
		'/debug-webgpu-utci?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&timeIndex=17'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & {
			__onDemandPrototypeDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__onDemandPrototypeDiagnostics__;
		return (
			diagnostics?.completedTimeIndex === 17 &&
			diagnostics?.pythonBinComparisonActive === true
		);
	}, undefined, { timeout: 180_000 });

	const diagnostics = await readDiagnostics(page);

	expect(diagnostics?.debugComparisonReference).toBe('python-bin');
	expect(diagnostics?.pythonBinComparisonActive).toBe(true);
	expect(diagnostics?.usedRunAllForSelectedHour).toBe(false);
	const sampleComparison = diagnostics?.pythonBinSampleComparison;
	expect(sampleComparison?.numCompared ?? 0).toBeGreaterThan(0);

	const samples = sampleComparison?.samples ?? [];
	expect(samples).toHaveLength(sampleComparison?.numCompared ?? 0);

	const samplesOverTightTolerance = samples.filter(
		(sample) => sample.absDiff > NON_HOTSPOT_SAMPLE_TOLERANCE,
	);
	const nonHotspotSamples = samples.filter(
		(sample) => sample.pointIndex !== KNOWN_LOCALIZED_HOTSPOT_POINT_INDEX,
	);

	for (const sample of nonHotspotSamples) {
		expect(sample.absDiff).toBeLessThanOrEqual(NON_HOTSPOT_SAMPLE_TOLERANCE);
	}

	if (samplesOverTightTolerance.length === 0) {
		expect(sampleComparison?.maxAbsDiff ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(
			NON_HOTSPOT_SAMPLE_TOLERANCE,
		);
		return;
	}

	expect(samplesOverTightTolerance).toHaveLength(1);
	const [knownHotspotSample] = samplesOverTightTolerance;
	expect(knownHotspotSample?.pointIndex).toBe(KNOWN_LOCALIZED_HOTSPOT_POINT_INDEX);
	expect(knownHotspotSample?.absDiff ?? 0).toBeGreaterThanOrEqual(KNOWN_HOTSPOT_MIN_ABS_DIFF);
	expect(knownHotspotSample?.absDiff ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(
		KNOWN_HOTSPOT_MAX_ABS_DIFF,
	);
	expect(sampleComparison?.maxAbsDiff).toBe(knownHotspotSample?.absDiff);
});
