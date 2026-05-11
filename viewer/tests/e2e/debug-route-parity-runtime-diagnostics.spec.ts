import { expect, test, type Page } from '@playwright/test';

async function readDebugDiagnostics(page: Page) {
	return page.evaluate(() => {
		return (window as any).__onDemandPrototypeDiagnostics__ ?? null;
	});
}

async function waitForDebugDiagnostics(
	page: Page,
	predicate: (value: any) => boolean,
	timeoutMessage: string
) {
	const handle = await page
		.waitForFunction(
			(predicateText) => {
				const value = (window as any).__onDemandPrototypeDiagnostics__;
				if (!value) return null;
				const predicate = new Function('value', `return (${predicateText})(value);`);
				return predicate(value) ? value : null;
			},
			predicate.toString(),
			{ timeout: 12_000 }
		)
		.catch(async (error) => {
			const lastDiagnostics = await readDebugDiagnostics(page);
			throw new Error(
				[
					timeoutMessage,
					error instanceof Error ? error.message : String(error),
					'Last window.__onDemandPrototypeDiagnostics__:',
					JSON.stringify(lastDiagnostics, null, 2)
				].join('\n')
			);
		});
	return handle.jsonValue() as Promise<any>;
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

test.describe('debug route parity runtime diagnostics', () => {
	test.afterEach(async ({ page }) => {
		await page.goto('about:blank').catch(() => undefined);
	});

	test('defaults to August Python bin comparison on debug route', async ({ page }) => {
		test.setTimeout(30_000);
		await page.goto(
			'/debug?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu',
			{ waitUntil: 'domcontentloaded', timeout: 10_000 }
		);

		const diagnostics = await waitForDebugDiagnostics(
			page,
			(value) =>
				value?.binComparisonEnabled === true &&
				value?.binComparisonValid === true &&
				value?.pythonBinComparisonActive === true &&
				value?.debugComparisonReference === 'python-bin' &&
				value?.pythonBinSampleComparison?.numCompared > 0,
			'Timed out waiting for default Python bin comparison diagnostics.'
		);

		expect(diagnostics.selectedHourEngine).toBe('legacy-debug');
		expect(diagnostics.selectedHourRuntimeContract).toMatchObject({
			route: 'debug',
			selectedHourEngine: 'legacy-debug',
			readbackInstrumentation: 'not-instrumented',
			strongVisibleGpuPath: false
		});
		expect(diagnostics.pythonBaselineStatus).toBe('available-august');
		expect(diagnostics.selectedMonthIndex).toBe(7);
		expect(diagnostics.selectedTimeIndex).toBe(168);
		await expect(page.getByText('Live WebGPU UTCI')).toBeVisible();
		await expect(page.locator('[data-testid="on-demand-prototype-status"]')).toHaveCount(0);
	});

	test('drops Python bin validity when the default comparison route changes away from August', async ({
		page
	}) => {
		test.setTimeout(30_000);
		await page.goto(
			'/debug?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu',
			{ waitUntil: 'domcontentloaded', timeout: 10_000 }
		);

		await waitForDebugDiagnostics(
			page,
			(value) =>
				value?.binComparisonEnabled === true &&
				value?.binComparisonValid === true &&
				value?.pythonBinComparisonActive === true &&
				value?.pythonBinSampleComparison?.numCompared > 0,
			'Timed out waiting for default August Python bin comparison before month change.'
		);

		await setDebugMonthSelection(page, 3);

		const diagnostics = await waitForDebugDiagnostics(
			page,
			(value) =>
				value?.binComparisonEnabled === true &&
				value?.binComparisonValid === false &&
				value?.pythonBaselineStatus === 'unavailable-non-august' &&
				value?.selectedMonthIndex === 3 &&
				value?.pythonBinComparisonActive !== true &&
				value?.pythonBinSampleComparison === undefined,
			'Timed out waiting for Python bin comparison to disable after non-August month selection.'
		);

		expect(diagnostics.debugComparisonReference).not.toBe('python-bin');
		expect(diagnostics.selectedTimeIndex).toBe(72);
	});

	test('allows August Python bin comparison only on debug route', async ({ page }) => {
		test.setTimeout(30_000);
		await page.goto(
			'/debug?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu&parity=1&utciOnDemand=f32&monthIndex=7&timeIndex=0',
			{ waitUntil: 'domcontentloaded', timeout: 10_000 }
		);

		const diagnostics = await waitForDebugDiagnostics(
			page,
			(value) =>
				value?.binComparisonEnabled === true &&
				value?.binComparisonValid === true &&
				value?.pythonBinComparisonActive === true &&
				value?.debugComparisonReference === 'python-bin' &&
				value?.pythonBinSampleComparison?.numCompared > 0,
			'Timed out waiting for August Python bin comparison diagnostics.'
		);

		expect(diagnostics.selectedHourEngine).toBe('legacy-debug');
		expect(diagnostics.selectedHourRuntimeContract).toMatchObject({
			route: 'debug',
			selectedHourEngine: 'legacy-debug',
			readbackInstrumentation: 'not-instrumented',
			strongVisibleGpuPath: false
		});
		expect(diagnostics.pythonBaselineStatus).toBe('available-august');
		expect(diagnostics.selectedMonthIndex).toBe(7);
		expect(diagnostics.selectedTimeIndex).toBe(168);
		await expect(page.getByText('Live WebGPU UTCI')).toBeVisible();
	});

	test('does not claim non-August Python bin validity', async ({ page }) => {
		test.setTimeout(30_000);
		await page.goto(
			'/debug?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu&parity=1&utciOnDemand=f32&monthIndex=3&timeIndex=9',
			{ waitUntil: 'domcontentloaded', timeout: 10_000 }
		);

		const diagnostics = await waitForDebugDiagnostics(
			page,
			(value) =>
				value?.binComparisonEnabled === true &&
				value?.binComparisonValid === false &&
				value?.pythonBaselineStatus === 'unavailable-non-august' &&
				value?.selectedMonthIndex === 3 &&
				value?.selectedTimeIndex === 81,
			'Timed out waiting for non-August Python bin invalid diagnostics.'
		);

		expect(diagnostics.selectedHourEngine).toBe('legacy-debug');
		expect(diagnostics.selectedHourRuntimeContract).toMatchObject({
			route: 'debug',
			selectedHourEngine: 'legacy-debug',
			readbackInstrumentation: 'not-instrumented',
			strongVisibleGpuPath: false
		});
		expect(diagnostics.selectedMonthIndex).toBe(3);
		expect(diagnostics.selectedTimeIndex).toBe(81);
		expect(diagnostics.pythonBinComparisonActive).not.toBe(true);
		expect(diagnostics.debugComparisonReference).not.toBe('python-bin');
		expect(diagnostics.pythonBinSampleComparison).toBeUndefined();
		expect(diagnostics.pythonBaselineStatus).toBe('unavailable-non-august');
	});
});
