import { expect, test, type Page } from '@playwright/test';

async function readDebugDiagnostics(page: Page) {
	return page.evaluate(() => {
		return (window as any).__onDemandPrototypeDiagnostics__ ?? null;
	});
}

function getTooltipHoverSampleCount(diagnostics: any): number {
	return (
		diagnostics?.tooltipInteraction?.hoverSampleCount ??
		diagnostics?.tooltipInteraction?.sampleCount ??
		0
	);
}

function getCameraWheelEventCount(diagnostics: any): number {
	return diagnostics?.cameraInteraction?.wheelEventCount ?? 0;
}

async function exerciseDebugCanvasInteractions(page: Page) {
	const canvas = page.locator('canvas').first();
	await expect(canvas).toBeVisible();
	const canvasBox = await canvas.boundingBox();
	expect(canvasBox).toBeTruthy();
	if (!canvasBox) {
		throw new Error('Expected the debug route canvas to expose a bounding box.');
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

async function waitForDebugInteractionDiagnostics(
	page: Page,
	before: { hoverSampleCount: number; wheelEventCount: number }
) {
	const handle = await page
		.waitForFunction(
			(args) => {
				const value = (window as any).__onDemandPrototypeDiagnostics__;
				const hoverSampleCount =
					value?.tooltipInteraction?.hoverSampleCount ??
					value?.tooltipInteraction?.sampleCount ??
					0;
				const wheelEventCount =
					value?.cameraInteraction?.wheelEventCount ?? 0;
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
		)
		.catch(async (error) => {
			const lastDiagnostics = await readDebugDiagnostics(page);
			throw new Error(
				[
					'Timed out waiting for debug canvas interaction diagnostics.',
					error instanceof Error ? error.message : String(error),
					'Last window.__onDemandPrototypeDiagnostics__:',
					JSON.stringify(lastDiagnostics, null, 2)
				].join('\n')
			);
		});
	return handle.jsonValue() as Promise<any>;
}

async function waitForSharedHostPublication(
	page: Page,
	options?: { previousRequestId?: number; expectedSelectionKey?: string }
) {
	const handle = await page
		.waitForFunction(
			(args) => {
				const value = (window as any).__onDemandPrototypeDiagnostics__;
				if (!value) return null;
				if (
					value.selectedHourEngine === 'shared-host' &&
					value.utciSurfaceSource === 'compute-buffer-selected-hour' &&
					value.renderTransport === 'compute-buffer-selected-hour' &&
					value.sameDeviceForComputeAndRender === true &&
					value.dataTextureBuildCount === 0 &&
					value.selectedHourRuntimeContract?.route === 'debug' &&
					value.selectedHourRuntimeContract?.selectedHourEngine === 'shared-host' &&
					value.selectedHourRuntimeContract?.readbackInstrumentation === 'instrumented' &&
					value.selectedHourRuntimeContract?.visibleSelectedHourReadbackCount === 0 &&
					value.selectedHourRuntimeContract?.strongVisibleGpuPath === true &&
					value.legacySelectedHourDispatchCount === 0 &&
					value.legacyScrubScheduleCount === 0 &&
					(typeof args.previousRequestId !== 'number' ||
						value.surfaceRequestId !== args.previousRequestId) &&
					(!args.expectedSelectionKey || value.selectionKey === args.expectedSelectionKey) &&
					(!args.expectedSelectionKey || value.sceneSelectionKey === args.expectedSelectionKey)
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
		)
		.catch(async (error) => {
			const lastDiagnostics = await readDebugDiagnostics(page);
			throw new Error(
				[
					'Timed out waiting for debug shared-host diagnostics.',
					error instanceof Error ? error.message : String(error),
					'Last window.__onDemandPrototypeDiagnostics__:',
					JSON.stringify(lastDiagnostics, null, 2)
				].join('\n')
			);
		});
	return handle.jsonValue() as Promise<any>;
}

async function setDebugHourSelection(page: Page, hourIndex: number) {
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
}

async function setDebugMonthSelection(page: Page, monthIndex: number) {
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

test.describe('debug route shared-host selected-hour diagnostics', () => {
	test.afterEach(async ({ page }) => {
		await page.goto('about:blank').catch(() => undefined);
	});

	test('uses shared host for normal f32 selected-hour rendering without legacy dispatch', async ({
		page
	}) => {
		test.setTimeout(30_000);
		const analysisId = 'Ben-Gurion/20250815_grid_2m_fullday';
		await page.goto(
			'/debug?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu&parity=0'
		);

		const initial = await waitForSharedHostPublication(page, {
			expectedSelectionKey: `${analysisId}|7|0`
		});

		expect(initial.surfaceRequestId).toBeGreaterThan(0);
		expect(initial.sceneSurfaceRequestId).toBe(initial.surfaceRequestId);
		expect(initial.selectedHourRuntimeContract.acceptedRequestId).toBe(initial.surfaceRequestId);
		expect(initial.selectedHourRuntimeContract.sceneRequestId).toBe(initial.sceneSurfaceRequestId);
		expect(initial.selectedHourRuntimeContract).toMatchObject({
			route: 'debug',
			selectedHourEngine: 'shared-host',
			readbackInstrumentation: 'instrumented',
			visibleSelectedHourReadbackCount: 0,
			visibleSelectedHourReadbackCountInstrumented: true,
			strongVisibleGpuPath: true,
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0
		});
		expect(initial.selectedHourRuntimeContract.readbackReasons).toEqual(
			expect.any(Array)
		);
		expect(initial.selectedTimeIndex).toBe(initial.renderContextTimeIndex);
		expect(initial.acceptedUtciRange).toEqual({
			min: expect.any(Number),
			max: expect.any(Number)
		});
		await expect(page.locator('.model-loading-overlay')).toBeHidden();

		const beforeInteraction = {
			hoverSampleCount: getTooltipHoverSampleCount(initial),
			wheelEventCount: getCameraWheelEventCount(initial)
		};
		await exerciseDebugCanvasInteractions(page);
		const afterInteraction = await waitForDebugInteractionDiagnostics(page, beforeInteraction);
		expect(getTooltipHoverSampleCount(afterInteraction)).toBeGreaterThan(
			beforeInteraction.hoverSampleCount
		);
		expect(getCameraWheelEventCount(afterInteraction)).toBeGreaterThan(
			beforeInteraction.wheelEventCount
		);
	});

	test('converges shared-host diagnostics after rapid month and hour changes', async ({ page }) => {
		test.setTimeout(30_000);
		const analysisId = 'Ben-Gurion/20250815_grid_2m_fullday';
		await page.goto(
			'/debug?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu&parity=0'
		);

		const initial = await waitForSharedHostPublication(page, {
			expectedSelectionKey: `${analysisId}|7|0`
		});

		await setDebugHourSelection(page, 2);
		await setDebugMonthSelection(page, 8);
		await setDebugHourSelection(page, 3);

		const finalValue = await waitForSharedHostPublication(page, {
			previousRequestId: initial.surfaceRequestId,
			expectedSelectionKey: `${analysisId}|8|3`
		});

		expect(finalValue.selectedTimeIndex).toBe(8 * 24 + 3);
		expect(finalValue.renderContextTimeIndex).toBe(8 * 24 + 3);
		expect(finalValue.selectionKey).toBe(finalValue.sceneSelectionKey);
		expect(finalValue.surfaceRequestId).toBe(finalValue.sceneSurfaceRequestId);
		expect(finalValue.selectedHourRuntimeContract).toMatchObject({
			route: 'debug',
			selectedHourEngine: 'shared-host',
			readbackInstrumentation: 'instrumented',
			visibleSelectedHourReadbackCount: 0,
			visibleSelectedHourReadbackCountInstrumented: true,
			acceptedRequestId: finalValue.surfaceRequestId,
			sceneRequestId: finalValue.sceneSurfaceRequestId,
			strongVisibleGpuPath: true
		});
		expect(finalValue.acceptedUtciRange).toEqual({
			min: expect.any(Number),
			max: expect.any(Number)
		});
		expect(finalValue.legacySelectedHourDispatchCount).toBe(0);
		expect(finalValue.legacyScrubScheduleCount).toBe(0);
		await expect(page.locator('.model-loading-overlay')).toBeHidden();
	});
});
