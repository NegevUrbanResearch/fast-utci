import { expect, test, type Page } from '@playwright/test';

type PrototypeDiagnostics = {
	navigatorGpu?: boolean;
	rendererBackend?: string;
	utciRenderRequested?: string;
	utciRenderResolved?: string;
	renderTransport?: string;
	utciSurfaceSource?: string;
	bridgeAttached?: boolean;
	debugReadbackCount?: number;
	dataTextureBuildCount?: number;
	visibleColorVariance?: number;
	tooltipInteraction?: {
		enabled?: boolean;
		disabledByQuery?: boolean;
		sampleCount?: number;
		hitCount?: number;
		missCount?: number;
	};
	cameraInteraction?: {
		slowThresholdMs?: number;
		sampleCount?: number;
		overBudgetCount?: number;
		lastFrameMs?: number | null;
		maxFrameMs?: number;
		p95FrameMs?: number | null;
	};
	tooltipProbeClientPoint?: {
		clientX?: number;
		clientY?: number;
	} | null;
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

async function readTooltipProbePoint(page: Page) {
	return page.evaluate(() => {
		const resolver = (window as Window & {
			__debugTooltipProbe__?: (() => { clientX: number; clientY: number } | null) | undefined;
		}).__debugTooltipProbe__;
		return resolver?.() ?? null;
	});
}

async function moveMouseOverRouteCanvas(page: Page) {
	const probe = await readTooltipProbePoint(page);
	if (probe?.clientX == null || probe?.clientY == null) {
		throw new Error("Debug route never exposed a concrete tooltip probe point.");
	}
	await page.mouse.move(probe.clientX, probe.clientY);
	await page.waitForTimeout(24);
}

async function hoverUntilTooltipHit(page: Page) {
	for (const [dx, dy] of [
		[0, 0],
		[2, 0],
		[0, 2],
		[-2, 0],
		[0, -2]
	] as const) {
		const probe = await readTooltipProbePoint(page);
		if (probe?.clientX == null || probe?.clientY == null) {
			throw new Error("Debug route never exposed a concrete tooltip probe point.");
		}
		await page.mouse.move(probe.clientX + dx, probe.clientY + dy);
		await page.waitForTimeout(24);
		const diagnostics = await readPrototypeDiagnostics(page);
		const hitCount = diagnostics?.tooltipInteraction?.hitCount ?? 0;
		const tooltipVisible = await page.getByRole('tooltip').isVisible().catch(() => false);
		if (hitCount > 0 && tooltipVisible) {
			return diagnostics;
		}
	}

	throw new Error('Expected a real tooltip hit on the debug route, but no visible tooltip/hit count was recorded.');
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
		if (
			diagnostics?.rendererBackend === 'webgpu' ||
			diagnostics?.renderTransport === 'compute-buffer-selected-hour' ||
			diagnostics?.utciSurfaceSource === 'compute-buffer-selected-hour'
		) {
			await expect(prototypeStatus).not.toContainText(/unsupported/i);
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

	test('debug route hover populates tooltip diagnostics by default', async ({ page }) => {
		await page.goto('/debug-webgpu-utci?onDemandPrototype=1');
		await expect(page.getByTestId('on-demand-prototype-status')).toContainText(
			/diagnostics|ready|error/i
		);

		await page.waitForFunction(() => {
			const diagnostics = (window as Window & {
				__onDemandPrototypeDiagnostics__?: PrototypeDiagnostics;
			}).__onDemandPrototypeDiagnostics__;
			return (
				diagnostics &&
				diagnostics.tooltipInteraction &&
				diagnostics.tooltipInteraction.enabled === true
			);
		});
		await expect
			.poll(async () => {
				const probe = await readTooltipProbePoint(page);
				return probe?.clientX != null && probe?.clientY != null;
			}, { timeout: 20_000 })
			.toBe(true);

		const before = await readPrototypeDiagnostics(page);
		expect(before?.tooltipInteraction?.disabledByQuery).toBe(false);
		expect(before?.tooltipInteraction?.enabled).toBe(true);
		expect(before?.tooltipInteraction?.sampleCount ?? 0).toBe(0);

		const after = await hoverUntilTooltipHit(page);
		expect(after?.tooltipInteraction?.disabledByQuery).toBe(false);
		expect(after?.tooltipInteraction?.enabled).toBe(true);
		expect(after?.tooltipInteraction?.sampleCount ?? 0).toBeGreaterThan(0);
		expect(after?.tooltipInteraction?.hitCount ?? 0).toBeGreaterThan(0);
		expect(
			(after?.tooltipInteraction?.hitCount ?? 0) + (after?.tooltipInteraction?.missCount ?? 0)
		).toBe(after?.tooltipInteraction?.sampleCount);
		await expect(page.getByRole('tooltip')).toBeVisible();
	});

	test('debug route wheel interaction temporarily suppresses tooltip hover sampling until motion settles', async ({
		page
	}) => {
		await page.goto('/debug-webgpu-utci?onDemandPrototype=1');
		await expect(page.getByTestId('on-demand-prototype-status')).toContainText(
			/diagnostics|ready|error/i
		);
		await expect(page.locator('canvas')).toBeVisible();
		await page.waitForFunction(() => !document.querySelector('.model-loading-overlay'));
		await page.waitForFunction(() => {
			const diagnostics = (window as Window & {
				__onDemandPrototypeDiagnostics__?: PrototypeDiagnostics;
			}).__onDemandPrototypeDiagnostics__;
			return diagnostics?.tooltipInteraction?.enabled === true;
		});
		await expect
			.poll(async () => {
				const probe = await readTooltipProbePoint(page);
				return probe?.clientX != null && probe?.clientY != null;
			}, { timeout: 20_000 })
			.toBe(true);

		const beforeHover = await hoverUntilTooltipHit(page);
		const baselineSampleCount = beforeHover?.tooltipInteraction?.sampleCount ?? 0;
		expect(baselineSampleCount).toBeGreaterThan(0);

		const canvasBox = await page.locator('canvas').boundingBox();
		expect(canvasBox).toBeTruthy();
		if (!canvasBox) {
			throw new Error('Expected the debug route canvas to expose a bounding box.');
		}

		const startX = canvasBox.x + canvasBox.width * 0.5;
		const startY = canvasBox.y + canvasBox.height * 0.5;
		await page.locator('canvas').dispatchEvent('wheel', {
			deltaX: 0,
			deltaY: 500,
			deltaMode: 0,
			clientX: startX,
			clientY: startY
		});

		await moveMouseOverRouteCanvas(page);
		const suppressedDiagnostics = await readPrototypeDiagnostics(page);
		expect(suppressedDiagnostics?.tooltipInteraction?.sampleCount ?? 0).toBe(
			baselineSampleCount
		);
		expect(await page.getByRole('tooltip').isVisible().catch(() => false)).toBe(false);

		await page.waitForTimeout(475);

		const resumedDiagnostics = await hoverUntilTooltipHit(page);
		expect(resumedDiagnostics?.tooltipInteraction?.sampleCount ?? 0).toBeGreaterThan(
			baselineSampleCount
		);
		await expect(page.getByRole('tooltip')).toBeVisible();
	});

	test('disableTooltip=1 keeps tooltip hover diagnostics at zero after mousemove attempts', async ({
		page
	}) => {
		await page.goto('/debug-webgpu-utci?onDemandPrototype=1&disableTooltip=1');
		await expect(page.getByTestId('on-demand-prototype-status')).toContainText(
			/diagnostics|ready|error/i
		);

		await page.waitForFunction(() => {
			const diagnostics = (window as Window & {
				__onDemandPrototypeDiagnostics__?: PrototypeDiagnostics;
			}).__onDemandPrototypeDiagnostics__;
			return (
				diagnostics &&
				diagnostics.tooltipInteraction &&
				diagnostics.tooltipInteraction.disabledByQuery === true
			);
		});
		await expect
			.poll(async () => {
				const probe = await readTooltipProbePoint(page);
				return probe?.clientX != null && probe?.clientY != null;
			}, { timeout: 20_000 })
			.toBe(true);

		const before = await readPrototypeDiagnostics(page);
		expect(before?.tooltipInteraction?.disabledByQuery).toBe(true);
		expect(before?.tooltipInteraction?.enabled).toBe(false);
		expect(before?.tooltipInteraction?.sampleCount ?? 0).toBe(0);

		await moveMouseOverRouteCanvas(page);

		const after = await readPrototypeDiagnostics(page);
		expect(after?.tooltipInteraction?.disabledByQuery).toBe(true);
		expect(after?.tooltipInteraction?.enabled).toBe(false);
		expect(after?.tooltipInteraction?.sampleCount ?? 0).toBe(0);
	});

	test('debug route wheel zoom populates camera interaction diagnostics only after movement', async ({
		page
	}) => {
		await page.goto('/debug-webgpu-utci?onDemandPrototype=1');
		await expect(page.getByTestId('on-demand-prototype-status')).toContainText(
			/diagnostics|ready|error/i
		);

		await expect(page.locator('canvas')).toBeVisible();
		await page.waitForFunction(() => !document.querySelector('.model-loading-overlay'));

		const before = await readPrototypeDiagnostics(page);
		expect(before?.cameraInteraction?.sampleCount ?? 0).toBe(0);
		expect(before?.cameraInteraction?.lastFrameMs ?? null).toBeNull();

		const canvasBox = await page.locator('canvas').boundingBox();
		expect(canvasBox).toBeTruthy();
		if (!canvasBox) {
			throw new Error('Expected the debug route canvas to expose a bounding box.');
		}

		const startX = canvasBox.x + canvasBox.width * 0.5;
		const startY = canvasBox.y + canvasBox.height * 0.5;
		await page.locator('canvas').dispatchEvent('wheel', {
			deltaX: 0,
			deltaY: 500,
			deltaMode: 0,
			clientX: startX,
			clientY: startY
		});

		await expect
			.poll(async () => {
				const diagnostics = await readPrototypeDiagnostics(page);
				return diagnostics?.cameraInteraction?.sampleCount ?? 0;
			}, { timeout: 20_000 })
			.toBeGreaterThan(0);

		const after = await readPrototypeDiagnostics(page);
		expect(after?.cameraInteraction?.slowThresholdMs).toBeGreaterThan(0);
		expect(after?.cameraInteraction?.sampleCount ?? 0).toBeGreaterThan(0);
		expect(after?.cameraInteraction?.lastFrameMs ?? 0).toBeGreaterThan(0);
		expect(after?.cameraInteraction?.maxFrameMs ?? 0).toBeGreaterThan(0);
		expect(after?.cameraInteraction?.p95FrameMs ?? 0).toBeGreaterThan(0);
		expect(after?.cameraInteraction?.overBudgetCount ?? 0).toBeLessThanOrEqual(
			after?.cameraInteraction?.sampleCount ?? 0
		);
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
