import { expect, test, type Page } from '@playwright/test';

async function readUtciRenderDiagnostics(page: Page) {
	return page.evaluate(() => {
		return (window as any).__utciRenderDiagnostics__ ?? null;
	});
}

async function waitForSelectedHourPublication(page: Page, options?: {
	previousRequestId?: number;
	expectedSelectionKey?: string;
}) {
	const diagnostics = await page.waitForFunction(
		(args) => {
			const value = (window as any).__utciRenderDiagnostics__;
			if (!value) return null;
			if (
				value.baseLiveReady === true &&
				value.utciSurfaceSource === 'compute-buffer-selected-hour' &&
				value.baseRenderTransport === 'compute-buffer-selected-hour' &&
				value.baseSameDeviceForComputeAndRender === true &&
				value.selectedHourRuntimeContract?.route === 'main' &&
				value.selectedHourRuntimeContract?.readbackInstrumentation === 'instrumented' &&
				value.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount === 0 &&
				value.selectedHourRuntimeContract?.strongVisibleGpuPath === true &&
				(typeof args.previousRequestId !== 'number' ||
					value.baseSurfaceRequestId !== args.previousRequestId) &&
				(!args.expectedSelectionKey ||
					value.baseSelectionKey === args.expectedSelectionKey) &&
				(!args.expectedSelectionKey ||
					value.baseSceneSelectionKey === args.expectedSelectionKey) &&
				value.gpuResidentCopyRequestId === value.baseSurfaceRequestId
			) {
				return value;
			}
			return null;
		},
		{
			previousRequestId: options?.previousRequestId,
			expectedSelectionKey: options?.expectedSelectionKey
		},
		{ timeout: 15_000 }
	).catch(async (error) => {
		const lastDiagnostics = await readUtciRenderDiagnostics(page).catch((readError) => ({
			readError: readError instanceof Error ? readError.message : String(readError)
		}));
		const message = error instanceof Error ? error.message : String(error);
		throw new Error(
			[
				'Timed out waiting for selected-hour publication diagnostics.',
				message,
				'Last window.__utciRenderDiagnostics__:',
				JSON.stringify(lastDiagnostics, null, 2)
			].join('\n')
		);
	});
	return diagnostics.jsonValue() as Promise<any>;
}

async function waitForComparisonReadbackAccounting(page: Page) {
	const diagnostics = await page.waitForFunction(
		() => {
			const value = (window as any).__utciRenderDiagnostics__;
			const comparisonCount =
				value?.selectedHourRuntimeContract?.readbackReasonCounts?.comparison ?? 0;
			if (
				value?.selectedHourRuntimeContract?.strongVisibleGpuPath === true &&
				comparisonCount >= 1
			) {
				return value;
			}
			return null;
		},
		undefined,
		{ timeout: 15_000 }
	).catch(async (error) => {
		const lastDiagnostics = await readUtciRenderDiagnostics(page).catch((readError) => ({
			readError: readError instanceof Error ? readError.message : String(readError)
		}));
		const message = error instanceof Error ? error.message : String(error);
		throw new Error(
			[
				'Timed out waiting for comparison readback accounting diagnostics.',
				message,
				'Last window.__utciRenderDiagnostics__:',
				JSON.stringify(lastDiagnostics, null, 2)
			].join('\n')
		);
	});
	return diagnostics.jsonValue() as Promise<any>;
}

async function setHourSelection(page: Page, hourIndex: number) {
	const modeButton = page.getByRole('button', { name: /^day$/i });
	await expect(modeButton).toBeVisible();
	await page.bringToFront();
	await page.waitForFunction(() => document.hasFocus());
	await modeButton.click();

	const slider = page.getByRole('slider', { name: /select analysis hour/i });
	await expect(slider).toBeVisible();
	await slider.click();
	await slider.focus();
	await expect(slider).toBeFocused();

	await slider.press('Home');
	for (let step = 0; step < hourIndex; step += 1) {
		await slider.press('ArrowRight');
	}
	await expect(slider).toHaveAttribute('aria-valuenow', String(hourIndex));
	await expect(slider).toHaveAttribute(
		'aria-valuetext',
		new RegExp(`Time\\s+${hourIndex.toString().padStart(2, '0')}:00`, 'i')
	);
}

async function setColorScaleMode(page: Page, mode: 'full day' | 'per hour') {
	const button = page.getByRole('button', { name: new RegExp(`^${mode}$`, 'i') });
	await expect(button).toBeVisible();
	await button.click();
	await expect(button).toHaveAttribute('aria-pressed', 'true');
}

async function setMonthSelection(page: Page, monthIndex: number) {
	const modeButton = page.getByRole('button', { name: /^month$/i });
	await expect(modeButton).toBeVisible();
	await page.bringToFront();
	await page.waitForFunction(() => document.hasFocus());
	await modeButton.click();

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
}

function getTooltipHoverSampleCount(diagnostics: any): number {
	return diagnostics?.tooltipInteraction?.hoverSampleCount ?? 0;
}

function getCameraWheelEventCount(diagnostics: any): number {
	return diagnostics?.cameraInteraction?.wheelEventCount ?? 0;
}

async function exerciseMainRouteCanvasInteractions(page: Page) {
	const canvas = page.locator('canvas').first();
	await expect(canvas).toBeVisible();
	const canvasBox = await canvas.boundingBox();
	expect(canvasBox).toBeTruthy();
	if (!canvasBox) {
		throw new Error('Expected the main route canvas to expose a bounding box.');
	}

	const clientX = canvasBox.x + canvasBox.width * 0.5;
	const clientY = canvasBox.y + canvasBox.height * 0.5;
	await page.mouse.move(clientX, clientY);
	await canvas.dispatchEvent('wheel', {
		deltaX: 0,
		deltaY: 500,
		deltaMode: 0,
		clientX,
		clientY
	});
}

async function waitForMainRouteInteractionDiagnostics(
	page: Page,
	before: { hoverSampleCount: number; wheelEventCount: number }
) {
	const handle = await page.waitForFunction(
		(args) => {
			const value = (window as any).__utciRenderDiagnostics__;
			const hoverSampleCount = value?.tooltipInteraction?.hoverSampleCount ?? 0;
			const wheelEventCount = value?.cameraInteraction?.wheelEventCount ?? 0;
			if (
				hoverSampleCount > args.hoverSampleCount &&
				wheelEventCount > args.wheelEventCount
			) {
				return value;
			}
			return null;
		},
		before,
		{ timeout: 10_000 }
	).catch(async (error) => {
		const lastDiagnostics = await readUtciRenderDiagnostics(page).catch((readError) => ({
			readError: readError instanceof Error ? readError.message : String(readError)
		}));
		const message = error instanceof Error ? error.message : String(error);
		throw new Error(
			[
				'Timed out waiting for main route canvas interaction diagnostics.',
				message,
				'Last window.__utciRenderDiagnostics__:',
				JSON.stringify(lastDiagnostics, null, 2)
			].join('\n')
		);
	});
	return handle.jsonValue() as Promise<any>;
}

function expectFiniteUtciRange(range: unknown) {
	expect(range).toEqual({
		min: expect.any(Number),
		max: expect.any(Number)
	});
	const typedRange = range as { min: number; max: number };
	expect(Number.isFinite(typedRange.min)).toBe(true);
	expect(Number.isFinite(typedRange.max)).toBe(true);
	expect(typedRange.max).toBeGreaterThan(typedRange.min);
}

function expectFiniteRenderPublicationTimeline(
	renderPublication: any
): Record<string, number> {
	expect(renderPublication).toBeDefined();
	const timeline = renderPublication?.renderPublicationTimeline;
	expect(timeline).toBeDefined();

	const requiredKeys = [
		'computeCompletedAtMs',
		'controllerAcceptedAtMs',
		'routePublishedAtMs',
		'routeProjectedAtMs',
		'sceneSurfaceReceivedAtMs',
		'publicationEffectStartedAtMs',
		'renderStorageReadyAtMs',
		'sceneSyncCompletedAtMs'
	] as const;

	for (const key of requiredKeys) {
		const value = timeline?.[key];
		expect(typeof value, `${key} should be a finite number`).toBe('number');
		expect(Number.isFinite(value), `${key} should be finite`).toBe(true);
	}

	return timeline as Record<string, number>;
}

function expectTruthfulRenderPublicationTimeline(renderPublication: any) {
	const timeline = expectFiniteRenderPublicationTimeline(renderPublication);

	expect(
		timeline.controllerAcceptedAtMs,
		'controllerAcceptedAtMs should not precede computeCompletedAtMs'
	).toBeGreaterThanOrEqual(timeline.computeCompletedAtMs);
	expect(
		timeline.routePublishedAtMs,
		'routePublishedAtMs should not precede controllerAcceptedAtMs'
	).toBeGreaterThanOrEqual(timeline.controllerAcceptedAtMs);
	expect(
		timeline.routeProjectedAtMs,
		'routeProjectedAtMs should not precede routePublishedAtMs'
	).toBeGreaterThanOrEqual(timeline.routePublishedAtMs);
	expect(
		timeline.sceneSurfaceReceivedAtMs,
		'sceneSurfaceReceivedAtMs should not precede controllerAcceptedAtMs'
	).toBeGreaterThanOrEqual(timeline.controllerAcceptedAtMs);
	expect(
		timeline.publicationEffectStartedAtMs,
		'publicationEffectStartedAtMs should not precede sceneSurfaceReceivedAtMs'
	).toBeGreaterThanOrEqual(timeline.sceneSurfaceReceivedAtMs);
	expect(
		timeline.renderStorageReadyAtMs,
		'renderStorageReadyAtMs should not precede publicationEffectStartedAtMs'
	).toBeGreaterThanOrEqual(timeline.publicationEffectStartedAtMs);
	expect(
		timeline.sceneSyncCompletedAtMs,
		'sceneSyncCompletedAtMs should not precede renderStorageReadyAtMs'
	).toBeGreaterThanOrEqual(timeline.renderStorageReadyAtMs);
}

async function readUtciLegendValues(page: Page): Promise<number[]> {
	return page
		.locator('.color-legend .label')
		.evaluateAll((nodes) =>
			nodes
				.map((node) => node.textContent ?? '')
				.map((text) => Number(text.replace(/[^0-9.+-]/g, '')))
				.filter((value) => Number.isFinite(value))
		);
}

function isForbiddenDebugBoundaryRequest(url: string) {
	const parsed = new URL(url);
	const isMainRouteDocument = parsed.pathname === '/' && parsed.searchParams.has('utciRenderDiagnostics');
	if (/\.bin(\?|$)|loadReferenceFromFs/i.test(url)) return true;
	if (/parity/i.test(url) && !isMainRouteDocument) return true;
	return false;
}

test.describe('main route manual diagnostics probe', () => {
	let requestedUrls: string[] = [];
	let allowedForbiddenRequestPatterns: RegExp[] = [];

	test.beforeEach(({ page }) => {
		requestedUrls = [];
		allowedForbiddenRequestPatterns = [];
		page.on('request', (request) => requestedUrls.push(request.url()));
	});

	test.afterEach(async ({ page }) => {
		expect(
			requestedUrls.filter(
				(url) =>
					isForbiddenDebugBoundaryRequest(url) &&
					!allowedForbiddenRequestPatterns.some((pattern) => pattern.test(url))
			)
		).toEqual([]);
		await page.goto('about:blank').catch(() => undefined);
	});

	test('publishes selected-hour diagnostics without waiting for the full e2e suite', async ({
		page
	}) => {
		test.setTimeout(30_000);
		await page.goto('/?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=auto&utciRenderDiagnostics=1');

		await page.waitForFunction(() => {
			const diagnostics = (window as any).__utciRenderDiagnostics__;
			return diagnostics?.rendererBackend === 'webgpu';
		}, undefined, { timeout: 10_000 }).catch(async (error) => {
			const lastDiagnostics = await readUtciRenderDiagnostics(page).catch((readError) => ({
				readError: readError instanceof Error ? readError.message : String(readError)
			}));
			const message = error instanceof Error ? error.message : String(error);
			throw new Error(
				[
					'Timed out waiting for WebGPU renderer backend diagnostics.',
					message,
					'Last window.__utciRenderDiagnostics__:',
					JSON.stringify(lastDiagnostics, null, 2)
				].join('\n')
			);
		});

		const diagnostics = await page.waitForFunction(
			() => {
				const value = (window as any).__utciRenderDiagnostics__;
				if (!value) return null;
				if (
					value.baseLiveReady === true &&
					value.utciSurfaceSource === 'compute-buffer-selected-hour' &&
					value.baseRenderTransport === 'compute-buffer-selected-hour' &&
					value.baseSameDeviceForComputeAndRender === true &&
					value.selectedHourRuntimeContract?.route === 'main' &&
					value.selectedHourRuntimeContract?.readbackInstrumentation === 'instrumented' &&
					value.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount === 0 &&
					value.selectedHourRuntimeContract?.strongVisibleGpuPath === true
				) {
					return value;
				}
				return null;
			},
			undefined,
			{ timeout: 15_000 }
		).catch(async (error) => {
			const lastDiagnostics = await readUtciRenderDiagnostics(page).catch((readError) => ({
				readError: readError instanceof Error ? readError.message : String(readError)
			}));
			const message = error instanceof Error ? error.message : String(error);
			throw new Error(
				[
					'Timed out waiting for selected-hour publication diagnostics.',
					message,
					'Last window.__utciRenderDiagnostics__:',
					JSON.stringify(lastDiagnostics, null, 2)
				].join('\n')
			);
		});

		const value = await diagnostics.jsonValue() as any;
		await page.getByRole('button', { name: /performance/i }).click();
		await expect(page.getByTestId('performance-panel')).toBeVisible();
		await expect(page.getByText(/Total calculation time/i)).toBeVisible();
		await expect(page.getByText(/UTCI calculation/i)).toBeVisible();
		await expect(page.getByText(/Render prep/i)).toHaveCount(0);
		await expect(page.getByText(/GPU VRAM/i)).toBeVisible();
		await expect(page.getByText(/Grid size/i)).toBeVisible();
		await expect(page.getByText(/Validation vs Grasshopper/i)).toHaveCount(0);
		expect(value.utciRenderResolved).toBe('gpuNative');
		expect(value.baseRenderTransport).toBe('compute-buffer-selected-hour');
		expect(value.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(value.dataTextureBuildCount).toBe(0);
		expect(value.baseLiveReady).toBe(true);
		expect(value.baseSameDeviceForComputeAndRender).toBe(true);
		expect(value.baseSelectionKey).toBe(value.baseSceneSelectionKey);
		expect(value.selectedHourRuntimeContract).toMatchObject({
			route: 'main',
			selectedHourEngine: 'shared-host',
			readbackInstrumentation: 'instrumented',
			visibleSelectedHourReadbackCount: 0,
			visibleSelectedHourReadbackCountInstrumented: true,
			acceptedRequestId: value.baseSurfaceRequestId,
			sceneRequestId: value.baseSceneSurfaceRequestId,
			strongVisibleGpuPath: true
		});
		expect(value.baseSelectedTimeIndex).toBe(value.baseRenderContextTimeIndex);
		expect(value.baseAcceptedUtciRange).toBeDefined();
		expect(value.timings?.oneHourDispatchMs ?? -1).toBeGreaterThanOrEqual(0);
		expect(value.timings?.firstSelectedHourVisibleMs ?? -1).toBeGreaterThanOrEqual(0);
		expect(value.timings?.renderPublication).toMatchObject({
			renderPublicationVersion: 1,
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationMeshAction: expect.stringMatching(/^(created|reused)$/)
		});
		expect(value.timings?.renderPublication?.renderPublicationPointCount ?? 0).toBeGreaterThan(0);
		expect(
			value.timings?.renderPublication?.renderPublicationRenderOwnedBytes ?? 0
		).toBeGreaterThan(0);
		expect(
			value.timings?.renderPublication?.renderPublicationSourceByteLength ?? 0
		).toBeGreaterThan(0);
		expect(
			value.timings?.renderPublication?.renderPublicationTargetByteLength ?? 0
		).toBeGreaterThan(0);
		if (
			typeof value.timings?.renderPublication?.renderPublicationSourceByteLength === 'number' &&
			typeof value.timings?.renderPublication?.renderPublicationTargetByteLength === 'number'
		) {
			expect(
				value.timings.renderPublication.renderPublicationTargetByteLength
			).toBeGreaterThanOrEqual(
				value.timings.renderPublication.renderPublicationSourceByteLength
			);
		}
		expectTruthfulRenderPublicationTimeline(value.timings?.renderPublication);
		expect(value.trackedGpuAllocationBytes).toMatchObject({
			persistentExposureBytes: expect.any(Number),
			allHoursOutputBytes: 0,
			selectedHourOutputBytes: expect.any(Number),
			selectedHourOutputBytesHighWatermark: expect.any(Number),
			renderOwnedSelectedHourBytes: expect.any(Number),
			renderOwnedSelectedHourBytesHighWatermark: expect.any(Number),
			trackingScope: 'utci-owned-webgpu-buffers'
		});
		expect(value.trackedGpuAllocationBytes.persistentExposureBytes).toBeGreaterThan(0);
		expect(value.trackedGpuAllocationBytes.selectedHourOutputBytes).toBeGreaterThan(0);
		expect(value.trackedGpuAllocationBytes.renderOwnedSelectedHourBytes).toBeGreaterThan(0);
		expect(value.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark).toBeGreaterThanOrEqual(
			value.trackedGpuAllocationBytes.selectedHourOutputBytes
		);
		expect(
			value.trackedGpuAllocationBytes.renderOwnedSelectedHourBytesHighWatermark
		).toBeGreaterThanOrEqual(value.trackedGpuAllocationBytes.renderOwnedSelectedHourBytes);
		expect(JSON.stringify(value)).not.toMatch(
			/pythonBin|binComparison|__onDemandPrototypeDiagnostics__|performance\.memory/i
		);

		const beforeInteraction = {
			hoverSampleCount: getTooltipHoverSampleCount(value),
			wheelEventCount: getCameraWheelEventCount(value)
		};
		await exerciseMainRouteCanvasInteractions(page);
		const afterInteraction = await waitForMainRouteInteractionDiagnostics(
			page,
			beforeInteraction
		);
		expect(getTooltipHoverSampleCount(afterInteraction)).toBeGreaterThan(
			beforeInteraction.hoverSampleCount
		);
		expect(getCameraWheelEventCount(afterInteraction)).toBeGreaterThan(
			beforeInteraction.wheelEventCount
		);
		expect(afterInteraction?.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(afterInteraction?.baseRenderTransport).toBe('compute-buffer-selected-hour');
		expect(afterInteraction?.baseSameDeviceForComputeAndRender).toBe(true);
	});

	test('ignores debug parity query params on the main route without bin requests', async ({
		page
	}) => {
		test.setTimeout(30_000);
		await page.goto(
			'/?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=auto&utciRenderDiagnostics=1&parity=1&utciOnDemand=f32&monthIndex=7'
		);

		const value = await waitForSelectedHourPublication(page, {
			expectedSelectionKey: 'Ben-Gurion/20250815_grid_2m_fullday|7|0'
		});

		expect(value.utciRenderResolved).toBe('gpuNative');
		expect(JSON.stringify(value)).not.toMatch(
			/pythonBin|binComparison|__onDemandPrototypeDiagnostics__|parityMode/i
		);
	});

	test('publishes a new compute-buffer surface after changing the selected hour', async ({
		page
	}) => {
		test.setTimeout(45_000);
		const analysisId = 'Ben-Gurion/20250815_grid_2m_fullday';
		await page.goto('/?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=auto&utciRenderDiagnostics=1');

		const initial = await waitForSelectedHourPublication(page, {
			expectedSelectionKey: `${analysisId}|7|0`
		});
		expect(initial.baseSurfaceRequestId).toBeGreaterThan(0);
		expect(initial.baseSceneSurfaceRequestId).toBe(initial.baseSurfaceRequestId);
		expect(initial.baseSelectedTimeIndex).toBe(7 * 24);
		expect(initial.baseRenderContextTimeIndex).toBe(7 * 24);

		await setHourSelection(page, 1);

		const updated = await waitForSelectedHourPublication(page, {
			previousRequestId: initial.baseSurfaceRequestId,
			expectedSelectionKey: `${analysisId}|7|1`
		});
		expect(updated.baseSurfaceRequestId).toBeGreaterThan(initial.baseSurfaceRequestId);
		expect(updated.baseSceneSurfaceRequestId).toBe(updated.baseSurfaceRequestId);
		expect(updated.gpuResidentCopyRequestId).toBe(updated.baseSurfaceRequestId);
		expect(updated.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(updated.baseRenderTransport).toBe('compute-buffer-selected-hour');
		expect(updated.baseLiveReady).toBe(true);
		expect(updated.baseSelectionKey).toBe(updated.baseSceneSelectionKey);
		expect(updated.baseSelectedTimeIndex).toBe(7 * 24 + 1);
		expect(updated.baseRenderContextTimeIndex).toBe(7 * 24 + 1);
		expect(updated.baseAcceptedUtciRange).toEqual(initial.baseAcceptedUtciRange);
		expectTruthfulRenderPublicationTimeline(updated.timings?.renderPublication);

		await setColorScaleMode(page, 'per hour');

		const perHour = await waitForSelectedHourPublication(page, {
			previousRequestId: updated.baseSurfaceRequestId,
			expectedSelectionKey: `${analysisId}|7|1`
		});
		expectFiniteUtciRange(perHour.baseAcceptedUtciRange);
		expect(perHour.baseAcceptedUtciRange.min).toBeGreaterThanOrEqual(
			initial.baseAcceptedUtciRange.min
		);
		expect(perHour.baseAcceptedUtciRange.max).toBeLessThanOrEqual(
			initial.baseAcceptedUtciRange.max
		);
		expectTruthfulRenderPublicationTimeline(perHour.timings?.renderPublication);
	});

	test('publishes a new compute-buffer surface after changing the selected month', async ({
		page
	}) => {
		test.setTimeout(45_000);
		const analysisId = 'Ben-Gurion/20250815_grid_2m_fullday';
		await page.goto('/?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=auto&utciRenderDiagnostics=1');

		const initial = await waitForSelectedHourPublication(page, {
			expectedSelectionKey: `${analysisId}|7|0`
		});
		expect(initial.baseSelectedTimeIndex).toBe(7 * 24);

		await setMonthSelection(page, 8);

		const updated = await waitForSelectedHourPublication(page, {
			previousRequestId: initial.baseSurfaceRequestId,
			expectedSelectionKey: `${analysisId}|8|0`
		});
		expect(updated.baseSurfaceRequestId).toBeGreaterThan(initial.baseSurfaceRequestId);
		expect(updated.gpuResidentCopyRequestId).toBe(updated.baseSurfaceRequestId);
		expect(updated.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(updated.baseSelectionKey).toBe(updated.baseSceneSelectionKey);
		expect(updated.baseSelectedTimeIndex).toBe(8 * 24);
		expect(updated.baseRenderContextTimeIndex).toBe(8 * 24);
		expectTruthfulRenderPublicationTimeline(updated.timings?.renderPublication);
	});

	test('keeps visible GPU path strong while comparison readbacks are accounted separately', async ({
		page
	}) => {
		await page.goto(
			'/?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=auto&utciRenderDiagnostics=1'
		);

		await waitForSelectedHourPublication(page, {
			expectedSelectionKey: 'Ben-Gurion/20250815_grid_2m_fullday|7|0'
		});

		allowedForbiddenRequestPatterns = [/\/Ben-Gurion\/existing_trees\/existing_trees_01\.bin(?:\?|$)/i];
		await page.getByRole('button', { name: /browse variants/i }).click();
		await page.getByRole('button', { name: /Existing Tree Cover/i }).click();

		const value = await waitForComparisonReadbackAccounting(page);
		expect(value.selectedHourRuntimeContract).toMatchObject({
			route: 'main',
			selectedHourEngine: 'shared-host',
			readbackInstrumentation: 'instrumented',
			visibleSelectedHourReadbackCount: 0,
			strongVisibleGpuPath: true,
			visibleRenderPathAvoidsCpuReadback: true
		});
		expect(value.selectedHourRuntimeContract.readbackReasons).toContain('comparison');
		expect(value.selectedHourRuntimeContract.readbackReasonCounts?.comparison ?? 0).toBeGreaterThan(0);
		expect(value.selectedHourReadbackReasons).toContain('comparison');
		expect(value.selectedHourReadbackReasonCounts?.comparison ?? 0).toBeGreaterThan(0);
	});

	test('uses live WebGPU UTCI range independent of .bin metadata for Ness Tziona', async ({
		page
	}) => {
		test.setTimeout(75_000);
		const analysisId = 'Ness-Tziona/exploded/nes_tziona_unblock_2';
		await page.goto('/?analysis=Ness-Tziona%2Fexploded%2Fnes_tziona_unblock_2&utciRender=auto&utciRenderDiagnostics=1');

		const initial = await waitForSelectedHourPublication(page, {
			expectedSelectionKey: `${analysisId}|7|0`
		});
		expectFiniteUtciRange(initial.baseAcceptedUtciRange);
		expect(initial.baseAcceptedUtciRange).not.toEqual({
			min: 23.165335456154907,
			max: 37.661211206353144
		});
		expect(initial.baseAcceptedUtciRange).not.toEqual({ min: -20, max: 60 });
		const initialLegend = await readUtciLegendValues(page);
		expect(initialLegend[0]).toBeCloseTo(initial.baseAcceptedUtciRange.max, 1);
		expect(initialLegend[initialLegend.length - 1]).toBeCloseTo(
			initial.baseAcceptedUtciRange.min,
			1
		);

		await setMonthSelection(page, 0);

		const updated = await waitForSelectedHourPublication(page, {
			previousRequestId: initial.baseSurfaceRequestId,
			expectedSelectionKey: `${analysisId}|0|0`
		});
		expect(updated.baseSelectedTimeIndex).toBe(0);
		expect(updated.baseRenderContextTimeIndex).toBe(0);
		expect(updated.baseSelectionKey).toBe(updated.baseSceneSelectionKey);
		expectFiniteUtciRange(updated.baseAcceptedUtciRange);
		expect(updated.baseAcceptedUtciRange).not.toEqual(initial.baseAcceptedUtciRange);
		expectTruthfulRenderPublicationTimeline(updated.timings?.renderPublication);
	});
});
