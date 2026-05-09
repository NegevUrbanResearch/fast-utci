import { expect, test, type Page } from '@playwright/test';

type PrototypeDiagnostics = {
	navigatorGpu?: boolean;
	rendererBackend?: string;
	utciRenderRequested?: string;
	utciRenderResolved?: string;
	bridgeAttached?: boolean;
	debugReadbackCount?: number;
	dataTextureBuildCount?: number;
	visibleColorVariance?: number;
};

type MainRouteUtciRenderDiagnostics = {
	rendererBackend?: string;
	utciRenderRequested?: string;
	utciRenderResolved?: string;
};

type PrototypeComparison = {
	timeIndex: number;
	numCompared: number;
	maxAbsDiff: number;
	rmse: number;
	debugReadbackCount: number;
};

async function readPrototypeDiagnostics(page: Page) {
	return page.evaluate(() => {
		return (window as Window & {
			__onDemandPrototypeDiagnostics__?: PrototypeDiagnostics;
		}).__onDemandPrototypeDiagnostics__;
	});
}

async function readMainRouteUtciRenderDiagnostics(page: Page) {
	for (let attempt = 0; attempt < 3; attempt += 1) {
		try {
			return await page.evaluate(() => {
				return (window as Window & {
					__utciRenderDiagnostics__?: MainRouteUtciRenderDiagnostics;
				}).__utciRenderDiagnostics__;
			});
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			if (!message.includes('Execution context was destroyed') || attempt === 2) {
				throw error;
			}
			await page.waitForLoadState('domcontentloaded').catch(() => undefined);
		}
	}
}

test.describe('WebGPU on-demand prototype diagnostics', () => {
	test('main route default resolves to gpuNative when WebGPU is available', async ({ page }) => {
		await page.goto('/?utciRenderDiagnostics=1');

		const navigatorGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
		test.skip(!navigatorGpuAvailable, 'WebGPU unavailable in this runtime.');

		await page.waitForFunction(() => {
			const diagnostics = (
				window as Window & {
					__utciRenderDiagnostics__?: MainRouteUtciRenderDiagnostics;
				}
			).__utciRenderDiagnostics__;

			return diagnostics?.rendererBackend === 'webgpu' && diagnostics?.utciRenderResolved === 'gpuNative';
		});

		const diagnostics = await readMainRouteUtciRenderDiagnostics(page);
		expect(diagnostics).toBeTruthy();
		expect(diagnostics?.utciRenderRequested).toBe('auto');
		expect(diagnostics?.rendererBackend).toBe('webgpu');
		expect(diagnostics?.utciRenderResolved).toBe('gpuNative');
	});

	test('main route honors utciRender=data override with dataTexture resolution', async ({ page }) => {
		await page.goto('/?utciRenderDiagnostics=1&utciRender=data');

		await page.waitForFunction(() => {
			const diagnostics = (
				window as Window & {
					__utciRenderDiagnostics__?: MainRouteUtciRenderDiagnostics;
				}
			).__utciRenderDiagnostics__;

			return diagnostics?.utciRenderRequested === 'data' && diagnostics?.utciRenderResolved === 'dataTexture';
		});

		const diagnostics = await readMainRouteUtciRenderDiagnostics(page);
		expect(diagnostics).toBeTruthy();
		expect(diagnostics?.utciRenderRequested).toBe('data');
		expect(diagnostics?.utciRenderResolved).toBe('dataTexture');
	});

	test('main route diagnostics update and clear on same-route query changes', async ({ page }) => {
		await page.goto('/?utciRenderDiagnostics=1');

		await page.waitForFunction(() => {
			const diagnostics = (
				window as Window & {
					__utciRenderDiagnostics__?: MainRouteUtciRenderDiagnostics;
				}
			).__utciRenderDiagnostics__;

			return diagnostics?.utciRenderRequested === 'auto';
		});

		await page.evaluate(() => {
			const nextUrl = new URL(window.location.href);
			nextUrl.searchParams.set('utciRender', 'data');
			window.history.pushState({}, '', nextUrl);
			window.dispatchEvent(new PopStateEvent('popstate'));
		});

		await page.waitForFunction(() => {
			const diagnostics = (
				window as Window & {
					__utciRenderDiagnostics__?: MainRouteUtciRenderDiagnostics;
				}
			).__utciRenderDiagnostics__;

			return diagnostics?.utciRenderRequested === 'data' && diagnostics?.utciRenderResolved === 'dataTexture';
		});

		await page.evaluate(() => {
			const nextUrl = new URL(window.location.href);
			nextUrl.searchParams.delete('utciRenderDiagnostics');
			window.history.pushState({}, '', nextUrl);
			window.dispatchEvent(new PopStateEvent('popstate'));
		});

		await page.waitForFunction(() => {
			return !(window as Window & {
				__utciRenderDiagnostics__?: MainRouteUtciRenderDiagnostics;
			}).__utciRenderDiagnostics__;
		});
	});

	test('reports prototype diagnostics on the debug route', async ({ page }) => {
		await page.goto('/debug-webgpu-utci?onDemandPrototype=1');

		const navigatorGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
		const requireWebgpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

		test.skip(
			!navigatorGpuAvailable && !requireWebgpu,
			'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
		);

		const prototypeStatus = page.getByTestId('on-demand-prototype-status');
		await expect(prototypeStatus).toContainText(/diagnostics|ready|error/i);
		if (navigatorGpuAvailable) {
			await expect(prototypeStatus).toContainText(/ready/i);
		}

		const diagnostics = await page.evaluate(() => {
			return (window as Window & {
				__onDemandPrototypeDiagnostics__?: {
					navigatorGpu: boolean;
					rendererBackend: string;
					utciRenderRequested?: string;
					utciRenderResolved?: string;
				};
			}).__onDemandPrototypeDiagnostics__;
		});

		expect(diagnostics).toBeTruthy();
		expect(diagnostics?.rendererBackend).toMatch(/webgpu|unknown/i);
		expect(typeof diagnostics?.navigatorGpu).toBe('boolean');
		expect(diagnostics?.utciRenderRequested).toBe('auto');
		expect(diagnostics?.utciRenderResolved).toBe(
			diagnostics?.rendererBackend === 'webgpu' ? 'gpuNative' : 'dataTexture'
		);
		if (navigatorGpuAvailable) {
			expect(diagnostics?.rendererBackend).toMatch(/webgpu/i);
		}
	});

	test('honors explicit UTCI render override diagnostics on the debug route', async ({ page }) => {
		await page.goto('/debug-webgpu-utci?onDemandPrototype=1&utciRender=data');

		const navigatorGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
		const requireWebgpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

		test.skip(
			!navigatorGpuAvailable && !requireWebgpu,
			'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
		);

		await expect(page.getByTestId('on-demand-render-selection')).toContainText(
			/utciRender data -> dataTexture/
		);

		const diagnostics = await readPrototypeDiagnostics(page);
		expect(diagnostics?.utciRenderRequested).toBe('data');
		expect(diagnostics?.utciRenderResolved).toBe('dataTexture');
	});

	test('reports synthetic bridge diagnostics on the debug route', async ({ page }) => {
		await page.goto('/debug-webgpu-utci?onDemandPrototype=1&syntheticBridge=1');

		const navigatorGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
		const requireWebgpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

		test.skip(
			!navigatorGpuAvailable && !requireWebgpu,
			'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
		);

		const prototypeStatus = page.getByTestId('on-demand-prototype-status');
		await expect(prototypeStatus).toContainText(/ready|error/i);

		await page.waitForFunction(() => {
			const diagnostics = (
				window as Window & {
					__onDemandPrototypeDiagnostics__?: PrototypeDiagnostics;
				}
			).__onDemandPrototypeDiagnostics__;

			return diagnostics?.bridgeAttached === true && (diagnostics.visibleColorVariance ?? 0) > 0;
		});

		const diagnostics = await readPrototypeDiagnostics(page);

		expect(diagnostics?.bridgeAttached).toBe(true);
		expect(diagnostics?.debugReadbackCount).toBe(0);
		expect(diagnostics?.dataTextureBuildCount).toBe(0);
		expect(diagnostics?.visibleColorVariance ?? 0).toBeGreaterThan(0);

		await page.getByTestId('project-select').selectOption('Ness-Tziona');

		await page.waitForFunction(() => {
			const diagnostics = (
				window as Window & {
					__onDemandPrototypeDiagnostics__?: PrototypeDiagnostics;
				}
			).__onDemandPrototypeDiagnostics__;

			return diagnostics?.bridgeAttached === false && diagnostics?.visibleColorVariance === 0;
		});

		const resetDiagnostics = await readPrototypeDiagnostics(page);
		expect(resetDiagnostics?.bridgeAttached).toBe(false);
		expect(resetDiagnostics?.visibleColorVariance).toBe(0);
		await expect(prototypeStatus).not.toContainText(/ready/i);
	});

	test('one-hour f32 on-demand output matches all-hours UTCI slice', async ({ page }) => {
		await page.goto('/debug-webgpu-utci?parity=1&onDemandPrototype=1&compareOneHour=1');

		const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
		const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

		test.skip(
			!hasWebGpu && !requireWebGpu,
			'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
		);

		await expect(page.getByTestId('on-demand-prototype-status')).toContainText(/ready|error/i, {
			timeout: 120_000
		});

		const result = await page.evaluate(() => {
			return (window as Window & {
				__onDemandPrototypeComparison__?: PrototypeComparison;
			}).__onDemandPrototypeComparison__;
		});

		expect(result).toBeTruthy();
		expect(result?.timeIndex).toBe(12);
		expect(result?.numCompared).toBeGreaterThan(0);
		expect(result?.maxAbsDiff).toBeLessThanOrEqual(1e-5);
		expect(result?.debugReadbackCount).toBeGreaterThan(0);

		const diagnostics = await readPrototypeDiagnostics(page);
		expect(diagnostics?.dataTextureBuildCount).toBe(0);
	});
});
