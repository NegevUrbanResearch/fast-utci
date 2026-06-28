import { expect, test, type Page } from '@playwright/test';

async function readDebugDiagnostics(page: Page) {
	return page.evaluate(() => {
		return (window as any).__onDemandPrototypeDiagnostics__ ?? null;
	});
}

test.describe('debug route baseline diagnostics probe', () => {
	test.afterEach(async ({ page }) => {
		await page.goto('about:blank').catch(() => undefined);
	});

	test('publishes selected-hour GPU diagnostics on the legacy parity debug baseline', async ({ page }) => {
		test.setTimeout(30_000);
		await page.goto(
			'/debug?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu&parity=1&utciOnDemand=f32&monthIndex=7&timeIndex=0'
		);

		const diagnosticsHandle = await page
			.waitForFunction(
				() => {
					const value = (window as any).__onDemandPrototypeDiagnostics__;
					if (
						value?.utciSurfaceSource === 'compute-buffer-selected-hour' &&
						value?.renderTransport === 'compute-buffer-selected-hour' &&
						value?.sameDeviceForComputeAndRender === true
					) {
						return value;
					}
					return null;
				},
				undefined,
				{ timeout: 20_000 }
			)
			.catch(async (error) => {
				const lastDiagnostics = await readDebugDiagnostics(page);
				throw new Error(
					[
						'Timed out waiting for legacy debug diagnostics.',
						error instanceof Error ? error.message : String(error),
						'Last window.__onDemandPrototypeDiagnostics__:',
						JSON.stringify(lastDiagnostics, null, 2)
					].join('\n')
				);
			});

		const diagnostics = (await diagnosticsHandle.jsonValue()) as any;
		expect(diagnostics.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(diagnostics.renderTransport).toBe('compute-buffer-selected-hour');
		expect(diagnostics.sameDeviceForComputeAndRender).toBe(true);
		expect(diagnostics.selectedHourEngine).toBe('legacy-debug');
		expect(diagnostics.legacySelectedHourDispatchCount).toBeGreaterThanOrEqual(1);
		expect(diagnostics.legacyScrubScheduleCount).toBeGreaterThanOrEqual(0);
		expect(diagnostics.pythonBinComparisonActive).toBe(true);
		expect(diagnostics.debugComparisonReference).toBe('python-bin');
	});
});
