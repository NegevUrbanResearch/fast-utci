import { expect, test, type Page } from '@playwright/test';

type StrictExposureOnlyDiagnostics = {
	navigatorGpu?: boolean;
	error?: string;
	rendererBackend?: string;
	utciRenderRequested?: string;
	utciRenderResolved?: string;
	utciOnDemand?: string;
	rendererRequestedMaxStorageBufferBindingSize?: number;
	rendererRequestedMaxBufferSize?: number;
	rendererDeviceMaxStorageBufferBindingSize?: number;
	rendererDeviceMaxBufferSize?: number;
	utciSurfaceSource?: string;
	selectedHourTransferCount?: number;
	selectedHourReadbackCount?: number;
	gpuResidentRenderAvailable?: boolean;
	sameDeviceForComputeAndRender?: boolean | null;
	gpuResidentCopyStatus?: 'idle' | 'pending' | 'complete' | 'failed';
	gpuResidentCopyError?: string;
	acceptedGpuResidentUtciRange?: { min: number; max: number };
	renderTransport?: 'none' | 'compute-buffer-selected-hour' | 'cpu-uploaded-selected-hour';
	path?: string;
	usedExposureOnlyPrecompute?: boolean;
	usedRunAllForSelectedHour?: boolean;
	liveAnalysisConstructedForSelectedHour?: boolean;
	allHoursUtciBytesAllocated?: number;
	allHoursMrtBytesAllocated?: number;
	oneHourOutputBytes?: number;
	selectedMonthIndex?: number;
	selectedTimeIndex?: number;
	completedMonthIndex?: number;
	completedTimeIndex?: number;
	inFlightCount?: number;
	scrubSampleCount?: number;
	staleResultDiscardCount?: number;
	pendingReadbackRequestId?: number;
	pendingReadbackTimeIndex?: number;
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
	trackedGpuAllocationBytes?: {
		persistentExposureBytes?: number;
		allHoursOutputBytes?: number;
		selectedHourOutputBytes?: number;
		selectedHourOutputBytesHighWatermark?: number;
		trackingScope?: string;
	};
	timings?: {
		exposurePrecomputeMs?: number;
		oneHourDispatchMs?: number;
		debugReadbackMs?: number;
		selectedHourReadbackMs?: number;
		selectedHourAnalysisBuildMs?: number;
		cpuColorBuildMs?: number;
		gpuSurfaceUpdateMs?: number;
		renderUpdateMs?: number;
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

type MonthHourComparisonPairResult = {
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	numCompared: number;
	maxAbsDiff: number;
	rmse: number;
	onDemandAt31079?: number;
	baselineAt31079?: number;
	diffAt31079?: number;
};

type MonthHourComparisonResult = {
	status: 'idle' | 'running' | 'complete' | 'error';
	baselineSource: string;
	pairs: MonthHourComparisonPairResult[];
	error?: string;
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

async function readMonthHourComparison(page: Page) {
	return page.evaluate(() => {
		return (window as Window & {
			__onDemandMonthHourComparison__?: MonthHourComparisonResult;
		}).__onDemandMonthHourComparison__;
	});
}

function expectSelectedHourTransportMatchesFeasibility(
	diagnostics: StrictExposureOnlyDiagnostics | undefined
) {
	expect(diagnostics?.error).toBeUndefined();
	if (diagnostics?.sameDeviceForComputeAndRender === true) {
		expect(diagnostics.gpuResidentRenderAvailable).toBe(true);
		expect(diagnostics.renderTransport).toBe('compute-buffer-selected-hour');
		expect(diagnostics.selectedHourReadbackCount).toBe(0);
		expect(diagnostics.selectedHourTransferCount).toBe(0);
		expect(diagnostics.dataTextureBuildCount).toBe(0);
		expect(diagnostics.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(diagnostics.gpuResidentCopyStatus).toBe('complete');
		expect(diagnostics.gpuResidentCopyError).toBeUndefined();
		expect(diagnostics.liveAnalysisConstructedForSelectedHour).toBe(false);
	} else {
		expect(diagnostics?.gpuResidentRenderAvailable).toBe(false);
		expect(diagnostics?.renderTransport).toBe('cpu-uploaded-selected-hour');
		expect(diagnostics?.selectedHourReadbackCount).toBe(1);
		expect(diagnostics?.utciSurfaceSource).toBe('cpu-uploaded-selected-hour');
		expect(diagnostics?.liveAnalysisConstructedForSelectedHour).toBe(true);
		expect(diagnostics?.gpuResidentCopyStatus).not.toBe('complete');
		if (diagnostics?.gpuResidentCopyError !== undefined) {
			expect(diagnostics.gpuResidentCopyError).toMatch(/GPU-resident render feasibility gate failed:/);
		}
	}
}

function expectGpuResidentSelectedHourTransport(diagnostics: StrictExposureOnlyDiagnostics | undefined) {
	expect(diagnostics?.error).toBeUndefined();
	expect(diagnostics?.gpuResidentRenderAvailable).toBe(true);
	expect(diagnostics?.sameDeviceForComputeAndRender).toBe(true);
	expect(diagnostics?.renderTransport).toBe('compute-buffer-selected-hour');
	expect(diagnostics?.gpuResidentCopyStatus).toBe('complete');
	expect(diagnostics?.gpuResidentCopyError).toBeUndefined();
	expect(diagnostics?.selectedHourReadbackCount).toBe(0);
	expect(diagnostics?.selectedHourTransferCount).toBe(0);
	expect(diagnostics?.dataTextureBuildCount).toBe(0);
	expect(diagnostics?.utciSurfaceSource).toBe('compute-buffer-selected-hour');
	expect(diagnostics?.liveAnalysisConstructedForSelectedHour).toBe(false);
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

test.afterEach(async ({ page }) => {
	await page.goto('about:blank', { timeout: 5_000 }).catch(() => undefined);
});

test('strict static-upload path computes one selected hour without constructing live analysis', async ({
	page
}) => {
	await page.goto(
		'/debug?onDemandPrototype=1&strictExposureOnly=1&timeIndex=12'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

	test.skip(
		!hasWebGpu && !requireWebGpu,
		'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
	);

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 60_000
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
	await page.goto('/debug?onDemandPrototype=1&syntheticBridge=1');

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
		timeout: 60_000
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
	test.setTimeout(60_000);
	await page.goto('/debug?parity=1&onDemandPrototype=1&compareOneHour=1');

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
	}, undefined, { timeout: 60_000 });

	const compareOneHourDiagnostics = await readDiagnostics(page);
	expect(compareOneHourDiagnostics?.debugComparisonReference).toBeUndefined();
	expect(compareOneHourDiagnostics?.pythonBinComparisonActive).not.toBe(true);

	await switchQueryMode(page, {
		compareOneHour: null,
		strictExposureOnly: '1',
		timeIndex: '12'
	});

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 60_000
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
			diagnostics.timings?.debugReadbackMs === undefined &&
			diagnostics.timings?.selectedHourReadbackMs === undefined &&
			diagnostics.timings?.selectedHourAnalysisBuildMs === undefined
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
	expect(diagnostics?.timings?.selectedHourReadbackMs).toBeUndefined();
	expect(diagnostics?.timings?.selectedHourAnalysisBuildMs).toBeUndefined();
	expect(diagnostics?.debugComparisonReference).toBeUndefined();
	expect(diagnostics?.pythonBinComparisonActive).toBe(false);
	expect(diagnostics?.debugComparisonMonthIndex).toBeUndefined();
	expect(diagnostics?.pythonComparisonHourIndex).toBeUndefined();
	expect(diagnostics?.webgpuComparisonHourIndex).toBeUndefined();
	expect(diagnostics?.pythonBinSampleComparison).toBeUndefined();
	expect(diagnostics?.selectedHourReadbackCount).toBe(0);
});

test('strict exposure-only compareHours matches a separate runAll baseline across multiple hours', async ({
	page
}) => {
	test.setTimeout(60_000);
	await page.goto(
		'/debug?onDemandPrototype=1&strictExposureOnly=1&compareHours=12,23,16,17&baseline=separateRunAll'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

	test.skip(
		!hasWebGpu && !requireWebGpu,
		'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
	);

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 60_000
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

test('strict exposure-only month/hour outputs compareMonthHours against a separate runAll baseline', async ({
	page
}) => {
	test.setTimeout(60_000);
	await page.goto(
		'/debug?onDemandPrototype=1&strictExposureOnly=1&compareMonthHours=0:12,3:15,7:23,10:18&baseline=separateRunAll'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

	test.skip(
		!hasWebGpu && !requireWebGpu,
		'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
	);

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 60_000
	});

	await page.waitForFunction(() => {
		const result = (window as Window & {
			__onDemandMonthHourComparison__?: MonthHourComparisonResult;
		}).__onDemandMonthHourComparison__;
		return result?.status === 'complete' || result?.status === 'error';
	}, undefined, { timeout: 60_000 });

	const result = await readMonthHourComparison(page);
	const diagnostics = await readDiagnostics(page);
	const expectedPairs = [
		{ monthIndex: 0, hourIndex: 12, timeIndex: 12 },
		{ monthIndex: 3, hourIndex: 15, timeIndex: 87 },
		{ monthIndex: 7, hourIndex: 23, timeIndex: 191 },
		{ monthIndex: 10, hourIndex: 18, timeIndex: 258 }
	];

	expect(result).toBeTruthy();
	expect(result?.status).toBe('complete');
	expect(result?.error).toBeUndefined();
	expect(result?.baselineSource).toBe('separateRunAll');
	expect(diagnostics?.usedRunAllForSelectedHour).toBe(false);
	expect(diagnostics?.debugReadbackCount).toBe(0);
	expect(diagnostics?.path).toBe('exposure-only-f32');
	expect(diagnostics?.usedExposureOnlyPrecompute).toBe(true);
	expect(diagnostics?.liveAnalysisConstructedForSelectedHour).toBe(false);
	expect(diagnostics?.dataTextureBuildCount).toBe(0);
	expect(diagnostics?.selectedHourReadbackCount).toBe(0);
	expect(result?.pairs).toHaveLength(expectedPairs.length);
	expect(
		result?.pairs.map((pair) => ({
			monthIndex: pair.monthIndex,
			hourIndex: pair.hourIndex,
			timeIndex: pair.timeIndex
		}))
	).toEqual(expectedPairs);

	for (const pair of result?.pairs ?? []) {
		expect(pair.numCompared).toBeGreaterThan(1000);
		expect(pair.maxAbsDiff).toBeLessThanOrEqual(1e-5);
		expect(pair.rmse).toBeLessThanOrEqual(1e-6);
	}
});

test('strict exposure-only diagnostics publish timing fields on the window object', async ({
	page
}) => {
	test.setTimeout(60_000);
	await page.goto('/debug?onDemandPrototype=1&strictExposureOnly=1&timeIndex=12');

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

	test.skip(
		!hasWebGpu && !requireWebGpu,
		'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
	);

	await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
		timeout: 60_000
	});

	const diagnostics = await readDiagnostics(page);
	expect(diagnostics).toBeTruthy();
	expect(diagnostics?.timings?.exposurePrecomputeMs ?? 0).toBeGreaterThan(0);
	expect(diagnostics?.timings?.oneHourDispatchMs ?? -1).toBeGreaterThanOrEqual(0);
});

test('main route publishes f32 on-demand diagnostics and preserves dataTexture fallback', async ({
	page
}) => {
	test.setTimeout(60_000);
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
	test.setTimeout(60_000);
	await page.goto(
		'/debug?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&timeIndex=12'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		if (diagnostics?.error) return true;
		return (
			diagnostics?.path === 'exposure-only-f32' &&
			diagnostics?.usedExposureOnlyPrecompute === true &&
			diagnostics?.usedRunAllForSelectedHour === false &&
			diagnostics?.completedTimeIndex === 7 * 24 + 12 &&
			diagnostics?.pythonComparisonHourIndex === 12 &&
			diagnostics?.webgpuComparisonHourIndex === 12 &&
			diagnostics?.appVisibleSelectedHour === true
		);
	}, undefined, { timeout: 60_000 });

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

	expectGpuResidentSelectedHourTransport(diagnostics);
	expect(diagnostics.debugComparisonReference).toBe('python-bin');
	expect(diagnostics.pythonBinComparisonActive).toBe(true);
	expect(diagnostics.pythonComparisonHourIndex).toBe(12);
	expect(diagnostics.webgpuComparisonHourIndex).toBe(12);
	expect(diagnostics.acceptedGpuResidentUtciRange).toBeTruthy();
	expect(diagnostics.rendererRequestedMaxStorageBufferBindingSize).toBeGreaterThanOrEqual(
		512 * 1024 * 1024
	);
	expect(diagnostics.rendererRequestedMaxBufferSize).toBeGreaterThanOrEqual(1024 * 1024 * 1024);
	expect(diagnostics.rendererDeviceMaxStorageBufferBindingSize).toBeGreaterThanOrEqual(
		512 * 1024 * 1024
	);
	expect(diagnostics.rendererDeviceMaxBufferSize).toBeGreaterThanOrEqual(1024 * 1024 * 1024);
	expect(diagnostics.timings.oneHourDispatchMs).toBeGreaterThanOrEqual(0);

	const fullDayRange = diagnostics.acceptedGpuResidentUtciRange;
	await page.getByRole('button', { name: /per hour/i }).click();
	await page.waitForFunction((previousRange) => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		const range = diagnostics?.acceptedGpuResidentUtciRange;
		return (
			range &&
			previousRange &&
			(range.min !== previousRange.min || range.max !== previousRange.max) &&
			diagnostics?.gpuResidentCopyStatus === 'complete'
		);
	}, fullDayRange, { timeout: 60_000 });

	const perHourDiagnostics = await readDiagnostics(page);
	expect(perHourDiagnostics?.acceptedGpuResidentUtciRange).toBeTruthy();
	expect(perHourDiagnostics?.acceptedGpuResidentUtciRange).not.toEqual(fullDayRange);

	await switchQueryMode(page, {
		timeIndex: '13'
	});

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		if (diagnostics?.error) return true;
		return (
			diagnostics?.path === 'exposure-only-f32' &&
			diagnostics?.usedExposureOnlyPrecompute === true &&
			diagnostics?.usedRunAllForSelectedHour === false &&
			diagnostics?.completedTimeIndex === 7 * 24 + 13 &&
			diagnostics?.pythonComparisonHourIndex === 13 &&
			diagnostics?.webgpuComparisonHourIndex === 13 &&
			diagnostics?.appVisibleSelectedHour === true
		);
	}, undefined, { timeout: 60_000 });

	await expect(page.getByRole('slider', { name: /select analysis hour/i })).toHaveAttribute(
		'aria-valuenow',
		'13'
	);
	await expect(page.getByRole('button', { name: /exit comparison mode/i })).toBeVisible();

	const switchedDiagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

	expectGpuResidentSelectedHourTransport(switchedDiagnostics);
	expect(switchedDiagnostics.pythonComparisonHourIndex).toBe(13);
	expect(switchedDiagnostics.webgpuComparisonHourIndex).toBe(13);
});

test('debug route only reports GPU-resident feasibility when compute and render share the same GPUDevice', async ({
	page
}) => {
	test.setTimeout(60_000);
	await page.goto(
		'/debug?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&timeIndex=12'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & {
			__onDemandPrototypeDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__onDemandPrototypeDiagnostics__;
		if (diagnostics?.error) return true;

		return (
			diagnostics?.path === 'exposure-only-f32' &&
			diagnostics?.usedExposureOnlyPrecompute === true &&
			diagnostics?.usedRunAllForSelectedHour === false &&
			diagnostics?.completedTimeIndex === 7 * 24 + 12 &&
			diagnostics?.pythonBinComparisonActive === true &&
			diagnostics?.appVisibleSelectedHour === true
		);
	}, undefined, { timeout: 60_000 });

	const diagnostics = await readDiagnostics(page);

	expect(diagnostics?.error).toBeUndefined();
	expect(typeof diagnostics?.gpuResidentRenderAvailable).toBe('boolean');
	expect(['idle', 'pending', 'complete', 'failed']).toContain(diagnostics?.gpuResidentCopyStatus);

	expectSelectedHourTransportMatchesFeasibility(diagnostics);
});

test('debug route preserves explicit dataTexture fallback when f32 on-demand is selected', async ({
	page
}) => {
	test.setTimeout(60_000);
	await page.goto(
		'/debug?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=data&timeIndex=12'
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
			diagnostics?.completedTimeIndex === 7 * 24 + 12 &&
			diagnostics?.pythonBinComparisonActive === true &&
			(diagnostics?.oneHourOutputBytes ?? 0) > 0 &&
			diagnostics?.allHoursUtciBytesAllocated === 0 &&
			diagnostics?.allHoursMrtBytesAllocated === 0 &&
			diagnostics?.utciRenderResolved === 'dataTexture' &&
			typeof diagnostics?.dataTextureBuildCount === 'number' &&
			(diagnostics.dataTextureBuildCount ?? 0) >= 1
		);
	}, undefined, { timeout: 60_000 });

	const diagnostics = await readDiagnostics(page);

	expect(diagnostics?.path).toBe('exposure-only-f32');
	expect(diagnostics?.usedExposureOnlyPrecompute).toBe(true);
	expect(diagnostics?.usedRunAllForSelectedHour).toBe(false);
	expect(diagnostics?.completedTimeIndex).toBe(7 * 24 + 12);
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
	test.setTimeout(60_000);
	await page.goto(
		'/debug?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&timeIndex=12'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		if (diagnostics?.error) return true;
		return diagnostics?.completedTimeIndex === 7 * 24 + 12;
	}, undefined, { timeout: 60_000 });

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
		if (diagnostics?.error) return true;
		return (
			diagnostics?.completedTimeIndex === 7 * 24 + 13 &&
			diagnostics?.pendingReadbackRequestId != null &&
			diagnostics?.pendingReadbackTimeIndex === 7 * 24 + 13
		);
	}, undefined, { timeout: 60_000 });

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
		if (diagnostics?.error) return true;
		return (
			diagnostics?.selectedTimeIndex === 7 * 24 + 23 &&
			diagnostics?.completedTimeIndex === 7 * 24 + 23 &&
			diagnostics?.pythonComparisonHourIndex === 23 &&
			diagnostics?.webgpuComparisonHourIndex === 23 &&
			diagnostics?.inFlightCount === 0 &&
			diagnostics?.pendingReadbackRequestId == null &&
			diagnostics?.appVisibleSelectedHour === true &&
			(diagnostics?.timings?.renderUpdateMs ?? 0) > 0
		);
	}, undefined, { timeout: 60_000 });

	const diagnostics = await page.evaluate(() => {
		return (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
	});

	expectSelectedHourTransportMatchesFeasibility(diagnostics);
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
	expect(diagnostics.selectedTimeIndex).toBe(7 * 24 + 23);
	expect(diagnostics.completedTimeIndex).toBe(7 * 24 + 23);
	expect(diagnostics.pythonComparisonHourIndex).toBe(23);
	expect(diagnostics.webgpuComparisonHourIndex).toBe(23);
	expect(diagnostics.timings.renderUpdateMs).toBeGreaterThan(0);
	expect(diagnostics.timings.gpuSurfaceUpdateMs).toBeGreaterThan(0);
	expect(diagnostics.scrubSampleCount).toBeGreaterThanOrEqual(4);
	expect(diagnostics.pendingReadbackRequestId).toBeUndefined();
	expect(diagnostics.staleResultDiscardCount).toBeGreaterThan(0);
});

test('debug on-demand honors the selected month when computing a selected hour', async ({
	page
}) => {
	test.setTimeout(60_000);
	await page.goto(
		'/debug?onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&monthIndex=7&timeIndex=12'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & { __onDemandPrototypeDiagnostics__?: any })
			.__onDemandPrototypeDiagnostics__;
		if (diagnostics?.error) return true;
		return (
			diagnostics?.selectedMonthIndex === 7 &&
			diagnostics?.completedMonthIndex === 7 &&
			diagnostics?.selectedTimeIndex === 7 * 24 + 12 &&
			diagnostics?.completedTimeIndex === 7 * 24 + 12 &&
			diagnostics?.appVisibleSelectedHour === true
		);
	}, undefined, { timeout: 60_000 });

	const diagnostics = await readDiagnostics(page);

	expect(diagnostics?.error).toBeUndefined();
	expect(diagnostics?.usedRunAllForSelectedHour).toBe(false);
	expect(diagnostics?.usedExposureOnlyPrecompute).toBe(true);
	expect(diagnostics?.allHoursUtciBytesAllocated).toBe(0);
	expect(diagnostics?.allHoursMrtBytesAllocated).toBe(0);
	expect(diagnostics?.dataTextureBuildCount).toBe(0);
	if (diagnostics?.sameDeviceForComputeAndRender === true) {
		expect(diagnostics?.selectedHourReadbackCount).toBe(0);
	} else {
		expect(diagnostics?.selectedHourReadbackCount).toBe(1);
	}
	expect(diagnostics?.pythonBinComparisonActive).toBe(false);
	expect(diagnostics?.debugComparisonReference).toBeUndefined();
	expect(diagnostics?.pythonComparisonHourIndex).toBeUndefined();
	expect(diagnostics?.webgpuComparisonHourIndex).toBeUndefined();
});

test('debug on-demand publishes sampled python-bin comparison metadata for the selected hour', async ({
	page
}) => {
	test.setTimeout(60_000);
	await page.goto(
		'/debug?parity=1&onDemandPrototype=1&utciOnDemand=f32&utciRender=gpu&timeIndex=17'
	);

	const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
	test.skip(!hasWebGpu && process.env.REQUIRE_WEBGPU_ON_DEMAND !== '1', 'WebGPU unavailable.');

	await page.waitForFunction(() => {
		const diagnostics = (window as Window & {
			__onDemandPrototypeDiagnostics__?: StrictExposureOnlyDiagnostics;
		}).__onDemandPrototypeDiagnostics__;
		if (diagnostics?.error) return true;
		return (
			diagnostics?.completedTimeIndex === 7 * 24 + 17 &&
			diagnostics?.pythonBinComparisonActive === true
		);
	}, undefined, { timeout: 60_000 });

	const diagnostics = await readDiagnostics(page);

	expect(diagnostics?.error).toBeUndefined();
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
