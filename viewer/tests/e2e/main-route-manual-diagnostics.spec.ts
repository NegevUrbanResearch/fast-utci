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

test.describe('main route manual diagnostics probe', () => {
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
					value.baseSameDeviceForComputeAndRender === true
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
		expect(value.utciRenderResolved).toBe('gpuNative');
		expect(value.baseRenderTransport).toBe('compute-buffer-selected-hour');
		expect(value.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(value.baseLiveReady).toBe(true);
		expect(value.baseSameDeviceForComputeAndRender).toBe(true);
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
		expect(updated.baseSelectedTimeIndex).toBe(7 * 24 + 1);
		expect(updated.baseRenderContextTimeIndex).toBe(7 * 24 + 1);
		expect(updated.baseAcceptedUtciRange).toEqual(initial.baseAcceptedUtciRange);

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
		expect(updated.baseSelectedTimeIndex).toBe(8 * 24);
		expect(updated.baseRenderContextTimeIndex).toBe(8 * 24);
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
		expectFiniteUtciRange(updated.baseAcceptedUtciRange);
		expect(updated.baseAcceptedUtciRange).not.toEqual(initial.baseAcceptedUtciRange);
	});
});
