import { expect, test } from '@playwright/test';

test.describe('debug route baseline diagnostics probe', () => {
	test('publishes selected-hour GPU diagnostics on the frozen debug baseline', async ({ page }) => {
		test.setTimeout(30_000);
		await page.goto(
			'/debug-webgpu-utci?analysis=Ben-Gurion%2F20250815_grid_2m_fullday&utciRender=gpu'
		);

		const diagnosticsHandle = await page.waitForFunction(
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
		);

		const diagnostics = (await diagnosticsHandle.jsonValue()) as any;
		expect(diagnostics.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(diagnostics.renderTransport).toBe('compute-buffer-selected-hour');
		expect(diagnostics.sameDeviceForComputeAndRender).toBe(true);
	});
});
