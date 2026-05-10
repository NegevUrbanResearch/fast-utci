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
	utciOnDemand?: string;
	rendererBackend?: string;
	utciRenderRequested?: string;
	utciRenderResolved?: string;
	utciSurfaceSource?: string;
	selectedHourTransferCount?: number;
	dataTextureBuildCount?: number;
	baseRenderTransport?: string;
	comparisonRenderTransport?: string;
	baseLiveReady?: boolean;
	comparisonLiveReady?: boolean;
	baseSurfaceRequestId?: number;
	baseSelectionKey?: string;
	baseSceneSurfaceRequestId?: number;
	baseSceneSelectionKey?: string;
	comparisonSurfaceRequestId?: number;
	comparisonSelectionKey?: string;
	comparisonUtciSurfaceSource?: string;
	comparisonSelectedHourTransferCount?: number;
	comparisonDataTextureBuildCount?: number;
};

type PrototypeComparison = {
	timeIndex: number;
	numCompared: number;
	maxAbsDiff: number;
	rmse: number;
	debugReadbackCount: number;
};

const BEN_GURION_BASE_ANALYSIS_ID = 'Ben-Gurion/20250815_grid_2m_fullday';

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

async function skipIfMainRouteLiveComputeUnavailable(page: Page) {
	const navigatorGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
	const requireWebgpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

	test.skip(
		!navigatorGpuAvailable && !requireWebgpu,
		'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
	);
}

function expectPublishedBaseSurface(diagnostics: MainRouteUtciRenderDiagnostics | undefined) {
	expect(diagnostics?.baseLiveReady).toBe(true);
	expect(['cpu-uploaded-selected-hour', 'compute-buffer-selected-hour']).toContain(
		diagnostics?.baseRenderTransport
	);

	if (diagnostics?.baseRenderTransport === 'compute-buffer-selected-hour') {
		expect(diagnostics?.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(diagnostics?.dataTextureBuildCount ?? 0).toBe(0);
		return;
	}

	expect(diagnostics?.utciSurfaceSource).toBe('cpu-uploaded-selected-hour');
	if (diagnostics?.utciRenderResolved === 'dataTexture') {
		expect(diagnostics?.dataTextureBuildCount ?? 0).toBeGreaterThan(0);
		return;
	}

	expect(diagnostics?.selectedHourTransferCount ?? 0).toBeGreaterThan(0);
	expect(diagnostics?.dataTextureBuildCount ?? 0).toBe(0);
}

function expectPublishedComparisonSurface(
	diagnostics: MainRouteUtciRenderDiagnostics | undefined
) {
	expect(diagnostics?.comparisonLiveReady).toBe(true);
	expect(['cpu-uploaded-selected-hour', 'compute-buffer-selected-hour']).toContain(
		diagnostics?.comparisonRenderTransport
	);

	if (diagnostics?.comparisonRenderTransport === 'compute-buffer-selected-hour') {
		expect(diagnostics?.comparisonUtciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(diagnostics?.comparisonDataTextureBuildCount ?? 0).toBe(0);
		return;
	}

	expect(diagnostics?.comparisonUtciSurfaceSource).toBe('cpu-uploaded-selected-hour');
	if (diagnostics?.utciRenderResolved === 'dataTexture') {
		expect(diagnostics?.comparisonDataTextureBuildCount ?? 0).toBeGreaterThan(0);
		return;
	}

	expect(diagnostics?.comparisonSelectedHourTransferCount ?? 0).toBeGreaterThan(0);
	expect(diagnostics?.comparisonDataTextureBuildCount ?? 0).toBe(0);
}

async function setRadialPickerSelection(
	page: Page,
	params: {
		mode: 'day' | 'month';
		index: number;
		expectedValueText: RegExp;
	}
) {
	const modeButton = page.getByRole('button', { name: new RegExp(`^${params.mode}$`, 'i') });
	await expect(modeButton).toBeVisible();
	await page.bringToFront();
	await page.waitForFunction(() => document.hasFocus());
	await modeButton.scrollIntoViewIfNeeded();
	await modeButton.click();
	const slider = page.getByRole('slider', {
		name: params.mode === 'month' ? /select month/i : /select analysis hour/i
	});
	await expect(slider).toBeVisible();
	await slider.click();
	await slider.focus();
	await expect(slider).toBeFocused();
	const dispatchSliderKey = async (key: 'Home' | 'ArrowRight') => {
		await slider.evaluate((node, keyValue: 'Home' | 'ArrowRight') => {
			node.dispatchEvent(
				new KeyboardEvent('keydown', {
					key: keyValue,
					bubbles: true,
					cancelable: true
				})
			);
		}, key);
	};
	await dispatchSliderKey('Home');
	for (let step = 0; step < params.index; step += 1) {
		await dispatchSliderKey('ArrowRight');
	}
	const sliderValueAfterKeyboard = await slider.getAttribute('aria-valuenow');
	if (sliderValueAfterKeyboard !== String(params.index)) {
		const box = await slider.boundingBox();
		if (!box) {
			throw new Error('Expected radial picker slider to expose a bounding box.');
		}
		const angleDeg =
			(params.index /
				(params.mode === 'month' ? 12 : 24)) *
				360 -
			90;
		const angleRad = (angleDeg * Math.PI) / 180;
		const radius = Math.min(box.width, box.height) / 2 - 6;
		await page.mouse.click(
			box.x + box.width / 2 + Math.cos(angleRad) * radius,
			box.y + box.height / 2 + Math.sin(angleRad) * radius
		);
	}
	await expect(slider).toHaveAttribute('aria-valuenow', String(params.index));
	await expect(slider).toHaveAttribute('aria-valuetext', params.expectedValueText);
}

async function setMonthSelection(page: Page, monthIndex: number, monthLabel: string) {
	await setRadialPickerSelection(page, {
		mode: 'month',
		index: monthIndex,
		expectedValueText: new RegExp(`Month\\s+${monthLabel}`, 'i')
	});
}

async function setHourSelection(page: Page, hourIndex: number) {
	await setRadialPickerSelection(page, {
		mode: 'day',
		index: hourIndex,
		expectedValueText: new RegExp(`Time\\s+${hourIndex.toString().padStart(2, '0')}:00`, 'i')
	});
}

function expectMainRouteBaseSelectedHourContract(
	diagnostics: MainRouteUtciRenderDiagnostics | undefined
) {
	expect(diagnostics?.baseLiveReady).toBe(true);
	expect(diagnostics?.utciRenderResolved).toBe('gpuNative');
	expect(['cpu-uploaded-selected-hour', 'compute-buffer-selected-hour']).toContain(
		diagnostics?.baseRenderTransport
	);

	if (diagnostics?.baseRenderTransport === 'compute-buffer-selected-hour') {
		expect(diagnostics?.utciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(diagnostics?.dataTextureBuildCount ?? 0).toBe(0);
		return;
	}

	expect(diagnostics?.utciSurfaceSource).toBe('cpu-uploaded-selected-hour');
	expect(diagnostics?.selectedHourTransferCount ?? 0).toBeGreaterThan(0);
	expect(diagnostics?.dataTextureBuildCount ?? 0).toBe(0);
}

function expectMainRouteComparisonSelectedHourContract(
	diagnostics: MainRouteUtciRenderDiagnostics | undefined
) {
	expect(diagnostics?.comparisonLiveReady).toBe(true);
	expect(['cpu-uploaded-selected-hour', 'compute-buffer-selected-hour']).toContain(
		diagnostics?.comparisonRenderTransport
	);
	if (diagnostics?.comparisonRenderTransport === 'compute-buffer-selected-hour') {
		expect(diagnostics?.comparisonUtciSurfaceSource).toBe('compute-buffer-selected-hour');
		expect(diagnostics?.comparisonDataTextureBuildCount ?? 0).toBe(0);
		return;
	}
	expect(diagnostics?.comparisonUtciSurfaceSource).toBe('cpu-uploaded-selected-hour');
	expect(diagnostics?.comparisonSelectedHourTransferCount ?? 0).toBeGreaterThan(0);
	expect(diagnostics?.comparisonDataTextureBuildCount ?? 0).toBe(0);
}

function hasMainRouteBaseSelectedHourContract(
	diagnostics: MainRouteUtciRenderDiagnostics | undefined
): boolean {
	if (
		diagnostics?.baseLiveReady !== true ||
		diagnostics?.utciRenderResolved !== 'gpuNative' ||
		diagnostics?.baseSelectionKey == null ||
		diagnostics?.baseSurfaceRequestId == null ||
		(diagnostics?.baseRenderTransport !== 'cpu-uploaded-selected-hour' &&
			diagnostics?.baseRenderTransport !== 'compute-buffer-selected-hour')
	) {
		return false;
	}

	if (diagnostics.baseRenderTransport === 'compute-buffer-selected-hour') {
		return (
			diagnostics.utciSurfaceSource === 'compute-buffer-selected-hour' &&
			(diagnostics.dataTextureBuildCount ?? 0) === 0
		);
	}

	return (
		diagnostics.utciSurfaceSource === 'cpu-uploaded-selected-hour' &&
		(diagnostics.selectedHourTransferCount ?? 0) > 0 &&
		(diagnostics.dataTextureBuildCount ?? 0) === 0
	);
}

function hasMainRouteComparisonSelectedHourContract(
	diagnostics: MainRouteUtciRenderDiagnostics | undefined
): boolean {
	const comparisonTransport = diagnostics?.comparisonRenderTransport;
	const comparisonContractSatisfied =
		comparisonTransport === 'compute-buffer-selected-hour'
			? diagnostics?.comparisonUtciSurfaceSource === 'compute-buffer-selected-hour' &&
				(diagnostics?.comparisonDataTextureBuildCount ?? 0) === 0
			: comparisonTransport === 'cpu-uploaded-selected-hour'
				? diagnostics?.comparisonUtciSurfaceSource === 'cpu-uploaded-selected-hour' &&
					(diagnostics?.comparisonSelectedHourTransferCount ?? 0) > 0 &&
					(diagnostics?.comparisonDataTextureBuildCount ?? 0) === 0
				: false;
	return (
		diagnostics?.baseLiveReady === true &&
		diagnostics?.comparisonLiveReady === true &&
		diagnostics?.comparisonSelectionKey != null &&
		diagnostics?.comparisonSurfaceRequestId != null &&
		hasMainRouteBaseSelectedHourContract(diagnostics) &&
		comparisonContractSatisfied
	);
}

function getMainRouteBaseSelectedHourSignature(
	diagnostics: MainRouteUtciRenderDiagnostics | undefined
): string {
	return JSON.stringify({
		utciRenderResolved: diagnostics?.utciRenderResolved ?? null,
		baseRenderTransport: diagnostics?.baseRenderTransport ?? null,
		baseLiveReady: diagnostics?.baseLiveReady ?? null,
		baseSurfaceRequestId: diagnostics?.baseSurfaceRequestId ?? null,
		baseSelectionKey: diagnostics?.baseSelectionKey ?? null,
		utciSurfaceSource: diagnostics?.utciSurfaceSource ?? null,
		selectedHourTransferCount: diagnostics?.selectedHourTransferCount ?? null,
		dataTextureBuildCount: diagnostics?.dataTextureBuildCount ?? null
	});
}

function getMainRouteComparisonSelectedHourSignature(
	diagnostics: MainRouteUtciRenderDiagnostics | undefined
): string {
	return JSON.stringify({
		base: getMainRouteBaseSelectedHourSignature(diagnostics),
		comparisonRenderTransport: diagnostics?.comparisonRenderTransport ?? null,
		comparisonLiveReady: diagnostics?.comparisonLiveReady ?? null,
		comparisonSurfaceRequestId: diagnostics?.comparisonSurfaceRequestId ?? null,
		comparisonSelectionKey: diagnostics?.comparisonSelectionKey ?? null,
		comparisonUtciSurfaceSource: diagnostics?.comparisonUtciSurfaceSource ?? null,
		comparisonSelectedHourTransferCount:
			diagnostics?.comparisonSelectedHourTransferCount ?? null,
		comparisonDataTextureBuildCount: diagnostics?.comparisonDataTextureBuildCount ?? null
	});
}

async function waitForMainRouteInteractiveLiveReady(page: Page) {
	await page.waitForFunction(() => {
		const diagnostics = (window as Window & {
			__utciRenderDiagnostics__?: MainRouteUtciRenderDiagnostics;
		}).__utciRenderDiagnostics__;
		if (!diagnostics) {
			return false;
		}
		return (
			diagnostics.rendererBackend === 'webgpu' &&
			diagnostics.utciRenderResolved === 'gpuNative'
		);
	});
	await expect(page.getByRole('button', { name: /^day$/i })).toBeVisible();
	await expect(page.getByRole('button', { name: /^month$/i })).toBeVisible();
	await expect(page.getByRole('slider', { name: /select analysis hour/i })).toBeVisible();
}

async function waitForMainRouteBaseSelectedHourContract(
	page: Page,
	options: { previous?: MainRouteUtciRenderDiagnostics | undefined } = {}
) {
	await waitForMainRouteInteractiveLiveReady(page);
	const previousSignature =
		options.previous == null ? null : getMainRouteBaseSelectedHourSignature(options.previous);
	let diagnostics: MainRouteUtciRenderDiagnostics | undefined;
	await expect
		.poll(async () => {
			diagnostics = await readMainRouteUtciRenderDiagnostics(page);
			if (!hasMainRouteBaseSelectedHourContract(diagnostics)) {
				return false;
			}
			if (
				previousSignature != null &&
				getMainRouteBaseSelectedHourSignature(diagnostics) === previousSignature
			) {
				return false;
			}
			return true;
		}, { timeout: 90_000 })
		.toBe(true);

	expectMainRouteBaseSelectedHourContract(diagnostics);
	return diagnostics;
}

async function waitForMainRouteComparisonSelectedHourContract(
	page: Page,
	options: { previous?: MainRouteUtciRenderDiagnostics | undefined } = {}
) {
	const previousSignature =
		options.previous == null
			? null
			: getMainRouteComparisonSelectedHourSignature(options.previous);
	let diagnostics: MainRouteUtciRenderDiagnostics | undefined;
	await expect
		.poll(async () => {
			diagnostics = await readMainRouteUtciRenderDiagnostics(page);
			if (!hasMainRouteComparisonSelectedHourContract(diagnostics)) {
				return false;
			}
			if (
				previousSignature != null &&
				getMainRouteComparisonSelectedHourSignature(diagnostics) === previousSignature
			) {
				return false;
			}
			return true;
		}, { timeout: 90_000 })
		.toBe(true);

	expectMainRouteBaseSelectedHourContract(diagnostics);
	expectMainRouteComparisonSelectedHourContract(diagnostics);
	return diagnostics;
}

async function waitForDebugRouteSelectedHourSurface(page: Page) {
	return waitForDebugRouteSelectedHourSurfaceAfterSelection(page);
}

async function waitForDebugRouteInteractiveReady(page: Page) {
	await page.waitForFunction(() => {
		const diagnostics = (window as Window & {
			__onDemandPrototypeDiagnostics__?: PrototypeDiagnostics;
		}).__onDemandPrototypeDiagnostics__;
		return diagnostics?.rendererBackend === 'webgpu' && diagnostics?.utciRenderResolved === 'gpuNative';
	});
	await expect(page.getByRole('button', { name: /^day$/i })).toBeVisible();
	await expect(page.getByRole('button', { name: /^month$/i })).toBeVisible();
	await expect(page.getByRole('slider', { name: /select analysis hour/i })).toBeVisible();
}

function getDebugRouteSelectedHourSignature(diagnostics: PrototypeDiagnostics | undefined): string {
	return JSON.stringify({
		rendererBackend: diagnostics?.rendererBackend ?? null,
		utciRenderResolved: diagnostics?.utciRenderResolved ?? null,
		renderTransport: diagnostics?.renderTransport ?? null,
		utciSurfaceSource: diagnostics?.utciSurfaceSource ?? null,
		debugReadbackCount: diagnostics?.debugReadbackCount ?? null,
		dataTextureBuildCount: diagnostics?.dataTextureBuildCount ?? null
	});
}

function hasDebugRouteSelectedHourContract(diagnostics: PrototypeDiagnostics | undefined): boolean {
	return (
		diagnostics?.rendererBackend === 'webgpu' &&
		diagnostics?.utciRenderResolved === 'gpuNative' &&
		diagnostics?.renderTransport === 'compute-buffer-selected-hour' &&
		diagnostics?.utciSurfaceSource === 'compute-buffer-selected-hour'
	);
}

async function waitForDebugRouteSelectedHourSurfaceAfterSelection(
	page: Page,
	options: { previous?: PrototypeDiagnostics | undefined } = {}
) {
	const previousSignature =
		options.previous == null ? null : getDebugRouteSelectedHourSignature(options.previous);
	let diagnostics: PrototypeDiagnostics | undefined;
	await expect
		.poll(async () => {
			diagnostics = await readPrototypeDiagnostics(page);
			if (!hasDebugRouteSelectedHourContract(diagnostics)) {
				return false;
			}
			if (
				previousSignature != null &&
				getDebugRouteSelectedHourSignature(diagnostics) === previousSignature
			) {
				return false;
			}
			return true;
		}, { timeout: 90_000 })
		.toBe(true);

	expect(diagnostics?.renderTransport).toBe('compute-buffer-selected-hour');
	expect(diagnostics?.utciSurfaceSource).toBe('compute-buffer-selected-hour');
	return diagnostics;
}

test.describe('WebGPU on-demand prototype diagnostics', () => {
	test('main route default resolves to gpuNative when WebGPU is available', async ({ page }) => {
		await page.goto('/?utciRenderDiagnostics=1');

		await skipIfMainRouteLiveComputeUnavailable(page);

		await page.waitForFunction(() => {
			const diagnostics = (
				window as Window & {
					__utciRenderDiagnostics__?: MainRouteUtciRenderDiagnostics;
				}
			).__utciRenderDiagnostics__;

			return (
				diagnostics?.rendererBackend === 'webgpu' &&
				diagnostics?.utciRenderResolved === 'gpuNative' &&
				diagnostics?.baseLiveReady === true &&
				(diagnostics?.baseRenderTransport === 'compute-buffer-selected-hour'
					? diagnostics?.utciSurfaceSource === 'compute-buffer-selected-hour'
					: (diagnostics?.selectedHourTransferCount ?? 0) > 0)
			);
		});

		const diagnostics = await readMainRouteUtciRenderDiagnostics(page);
		expect(diagnostics).toBeTruthy();
		expect(diagnostics?.utciOnDemand).toBe('f32');
		expect(diagnostics?.utciRenderRequested).toBe('auto');
		expect(diagnostics?.rendererBackend).toBe('webgpu');
		expect(diagnostics?.utciRenderResolved).toBe('gpuNative');
		expectPublishedBaseSurface(diagnostics);
	});

	test('main route honors utciRender=data override with dataTexture resolution', async ({ page }) => {
		await page.goto('/?utciRenderDiagnostics=1&utciRender=data');
		await skipIfMainRouteLiveComputeUnavailable(page);

		await page.waitForFunction(() => {
			const diagnostics = (
				window as Window & {
					__utciRenderDiagnostics__?: MainRouteUtciRenderDiagnostics;
				}
			).__utciRenderDiagnostics__;

			return (
				diagnostics?.utciRenderRequested === 'data' &&
				diagnostics?.utciRenderResolved === 'dataTexture' &&
				diagnostics?.baseLiveReady === true &&
				diagnostics?.baseRenderTransport === 'cpu-uploaded-selected-hour' &&
				(diagnostics?.dataTextureBuildCount ?? 0) > 0
			);
		});

		const diagnostics = await readMainRouteUtciRenderDiagnostics(page);
		expect(diagnostics).toBeTruthy();
		expect(diagnostics?.utciRenderRequested).toBe('data');
		expect(diagnostics?.utciRenderResolved).toBe('dataTexture');
		expectPublishedBaseSurface(diagnostics);
	});

	test('main route diagnostics update and clear on same-route query changes', async ({ page }) => {
		await page.goto('/?utciRenderDiagnostics=1');
		await skipIfMainRouteLiveComputeUnavailable(page);

		await page.waitForFunction(() => {
			const diagnostics = (
				window as Window & {
					__utciRenderDiagnostics__?: MainRouteUtciRenderDiagnostics;
				}
			).__utciRenderDiagnostics__;

			return (
				diagnostics?.utciRenderRequested === 'auto' &&
				diagnostics?.baseLiveReady === true &&
				(diagnostics?.baseRenderTransport === 'compute-buffer-selected-hour'
					? diagnostics?.utciSurfaceSource === 'compute-buffer-selected-hour'
					: (diagnostics?.selectedHourTransferCount ?? 0) > 0)
			);
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

			return (
				diagnostics?.utciRenderRequested === 'data' &&
				diagnostics?.utciRenderResolved === 'dataTexture' &&
				diagnostics?.baseLiveReady === true &&
				diagnostics?.baseRenderTransport === 'cpu-uploaded-selected-hour' &&
				(diagnostics?.dataTextureBuildCount ?? 0) > 0
			);
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

	test('main route scenario comparison keeps live UTCI active on both sides', async ({ page }) => {
		test.setTimeout(60_000);
		await page.goto('/?utciRenderDiagnostics=1');
		await skipIfMainRouteLiveComputeUnavailable(page);

		await page.waitForFunction(() => {
			const diagnostics = (window as Window & {
				__utciRenderDiagnostics__?: MainRouteUtciRenderDiagnostics;
			}).__utciRenderDiagnostics__;
			return (
				diagnostics?.baseLiveReady === true &&
				(diagnostics?.baseRenderTransport === 'compute-buffer-selected-hour'
					? diagnostics?.utciSurfaceSource === 'compute-buffer-selected-hour'
					: (diagnostics?.selectedHourTransferCount ?? 0) > 0)
			);
		});

		await page.getByRole('button', { name: /browse variants/i }).click();
		await page.getByRole('button', { name: /existing tree cover/i }).click();

		await page.waitForFunction(() => {
			const diagnostics = (window as Window & {
				__utciRenderDiagnostics__?: MainRouteUtciRenderDiagnostics;
			}).__utciRenderDiagnostics__;
			return (
				diagnostics?.baseLiveReady === true &&
				diagnostics?.comparisonLiveReady === true &&
				(diagnostics?.baseRenderTransport === 'compute-buffer-selected-hour'
					? diagnostics?.utciSurfaceSource === 'compute-buffer-selected-hour'
					: diagnostics?.utciRenderResolved === 'dataTexture'
						? (diagnostics?.dataTextureBuildCount ?? 0) > 0
						: (diagnostics?.selectedHourTransferCount ?? 0) > 0) &&
				(diagnostics?.comparisonRenderTransport === 'compute-buffer-selected-hour'
					? diagnostics?.comparisonUtciSurfaceSource ===
						'compute-buffer-selected-hour'
					: diagnostics?.comparisonRenderTransport ===
						  'cpu-uploaded-selected-hour' &&
						(diagnostics?.utciRenderResolved === 'dataTexture'
							? (diagnostics?.comparisonDataTextureBuildCount ?? 0) > 0
							: (diagnostics?.comparisonSelectedHourTransferCount ?? 0) > 0))
			);
		});

		await expect(page.getByRole('button', { name: /exit comparison mode/i })).toBeVisible();
		await expect(page.getByRole('slider', { name: /comparison curtain position/i })).toBeVisible();

		const diagnostics = await readMainRouteUtciRenderDiagnostics(page);
		expectPublishedBaseSurface(diagnostics);
		expectPublishedComparisonSurface(diagnostics);
	});

	test('main route matches debug selected-hour behavior', async ({ page }) => {
		test.setTimeout(90_000);
		await page.goto(
			`/?analysis=${encodeURIComponent(BEN_GURION_BASE_ANALYSIS_ID)}&utciRender=auto&utciRenderDiagnostics=1`
		);
		await skipIfMainRouteLiveComputeUnavailable(page);

		const initialDiagnostics = await waitForMainRouteBaseSelectedHourContract(page);
		expect(initialDiagnostics?.baseLiveReady).toBe(true);
		expect(initialDiagnostics?.utciRenderRequested).toBe('auto');
		expect(initialDiagnostics?.utciRenderResolved).toBe('gpuNative');

		await setMonthSelection(page, 7, 'Aug');
		const augustDiagnostics = await waitForMainRouteBaseSelectedHourContract(page, {
			previous: initialDiagnostics
		});
		expect(augustDiagnostics?.baseLiveReady).toBe(true);

		await page.getByRole('button', { name: /browse variants/i }).click();
		await page.getByRole('button', { name: /existing tree cover/i }).click();
		const comparisonDiagnostics = await waitForMainRouteComparisonSelectedHourContract(page, {
			previous: augustDiagnostics
		});
		expect(comparisonDiagnostics?.comparisonLiveReady).toBe(true);

		await setMonthSelection(page, 0, 'Jan');
		const januaryDiagnostics = await waitForMainRouteComparisonSelectedHourContract(page, {
			previous: comparisonDiagnostics
		});
		expect(januaryDiagnostics?.baseLiveReady).toBe(true);
		expect(januaryDiagnostics?.comparisonLiveReady).toBe(true);
	});

	test('main route matches debug selected-hour baseline for the same selection', async ({
		page,
		context
	}) => {
		test.setTimeout(90_000);
		const debugPage = await context.newPage();

		try {
			await page.goto(
				`/?analysis=${encodeURIComponent(BEN_GURION_BASE_ANALYSIS_ID)}&utciRender=auto&utciRenderDiagnostics=1`
			);
			await skipIfMainRouteLiveComputeUnavailable(page);
			await waitForMainRouteInteractiveLiveReady(page);
			const initialMainDiagnostics = await readMainRouteUtciRenderDiagnostics(page);

			await debugPage.goto(
				`/debug?analysis=${encodeURIComponent(BEN_GURION_BASE_ANALYSIS_ID)}&utciRender=auto&monthIndex=0&timeIndex=0`
			);
			const debugNavigatorGpuAvailable = await debugPage.evaluate(() => Boolean(navigator.gpu));
			const requireWebgpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';
			test.skip(
				!debugNavigatorGpuAvailable && !requireWebgpu,
				'WebGPU unavailable on the debug route and REQUIRE_WEBGPU_ON_DEMAND is not set.'
			);

			await waitForDebugRouteInteractiveReady(debugPage);
			const initialDebugDiagnostics = await readPrototypeDiagnostics(debugPage);
			await setMonthSelection(page, 7, 'Aug');
			await setHourSelection(page, 12);
			await setMonthSelection(debugPage, 7, 'Aug');
			await setHourSelection(debugPage, 12);

			const [mainDiagnostics, debugDiagnostics] = await Promise.all([
				waitForMainRouteBaseSelectedHourContract(page, {
					previous: initialMainDiagnostics
				}),
				waitForDebugRouteSelectedHourSurfaceAfterSelection(debugPage, {
					previous: initialDebugDiagnostics
				})
			]);

			expect(debugDiagnostics?.utciSurfaceSource).toBe('compute-buffer-selected-hour');
			expect(debugDiagnostics?.renderTransport).toBe('compute-buffer-selected-hour');
			expect(debugDiagnostics?.utciRenderResolved).toBe('gpuNative');
			expectMainRouteBaseSelectedHourContract(mainDiagnostics);
		} finally {
			await debugPage.close();
		}
	});

test('main route diagnostics export base scene identity during gpu-native bootstrap', async ({
	page
}) => {
	test.setTimeout(45_000);
	await page.goto(
		`/?analysis=${encodeURIComponent(BEN_GURION_BASE_ANALYSIS_ID)}&utciRender=auto&utciRenderDiagnostics=1`
	);
	await skipIfMainRouteLiveComputeUnavailable(page);
	await waitForMainRouteInteractiveLiveReady(page);

	let diagnostics: MainRouteUtciRenderDiagnostics | undefined;
	await expect
		.poll(async () => {
			diagnostics = await readMainRouteUtciRenderDiagnostics(page);
			return (
				diagnostics?.baseRenderTransport === 'compute-buffer-selected-hour' &&
				diagnostics?.utciSurfaceSource === 'compute-buffer-selected-hour' &&
				diagnostics?.baseSameDeviceForComputeAndRender === true &&
				diagnostics?.baseLiveReady === false &&
				diagnostics?.baseSceneSurfaceRequestId != null &&
				diagnostics?.baseSceneSelectionKey != null &&
				(diagnostics?.baseSurfaceRequestId == null ||
					diagnostics.baseSurfaceRequestId !==
						diagnostics.baseSceneSurfaceRequestId) &&
				(diagnostics?.baseSelectionKey == null ||
					diagnostics.baseSelectionKey !== diagnostics.baseSceneSelectionKey)
			);
		}, { timeout: 30_000 })
		.toBe(true);

	expect(diagnostics?.baseLiveReady).toBe(false);
	expect(diagnostics?.baseSceneSurfaceRequestId).not.toBeNull();
	expect(diagnostics?.baseSceneSelectionKey).toBeTruthy();
	expect(diagnostics?.baseSurfaceRequestId ?? null).not.toBe(
		diagnostics?.baseSceneSurfaceRequestId ?? null
	);
	expect(diagnostics?.baseSelectionKey ?? null).not.toBe(
		diagnostics?.baseSceneSelectionKey ?? null
	);
});

	test('plain debug route defaults to on-demand diagnostics', async ({ page }) => {
		await page.goto('/debug');

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

	test('debug route honors utciOnDemand=off explicit opt-out', async ({ page }) => {
		await page.goto('/debug?utciOnDemand=off');
		await expect(page.locator('[data-testid="on-demand-prototype-status"]')).toHaveCount(0);

		await page.waitForLoadState('domcontentloaded');
		await expect(page.getByTestId('project-select')).toBeVisible();

		await expect
			.poll(async () => {
				return page.evaluate(() => {
					return (window as Window & {
						__onDemandPrototypeDiagnostics__?: PrototypeDiagnostics;
					}).__onDemandPrototypeDiagnostics__;
				});
			})
			.toBeUndefined();
	});

	test('parity-only debug route does not silently enable on-demand diagnostics', async ({ page }) => {
		await page.goto('/debug?parity=1');
		await expect(page.locator('[data-testid="on-demand-prototype-status"]')).toHaveCount(0);

		await page.waitForLoadState('domcontentloaded');
		await expect(page.getByTestId('project-select')).toBeVisible();

		await expect
			.poll(async () => {
				return page.evaluate(() => {
					return (window as Window & {
						__onDemandPrototypeDiagnostics__?: PrototypeDiagnostics;
					}).__onDemandPrototypeDiagnostics__;
				});
			})
			.toBeUndefined();
	});

	test('normal collect route preserves full-day collection harness by default', async ({ page }) => {
		test.setTimeout(120_000);
		await page.goto('/debug?collect=normal');
		await expect(page.locator('[data-testid="on-demand-prototype-status"]')).toHaveCount(0);

		await expect
			.poll(async () => {
				return page.evaluate(() => {
					const win = window as Window & {
						__normalUtciResults__?: {
							numPoints: number;
							numHours: number;
							monthIndex: number;
							utciByHour: number[][];
						};
						__onDemandPrototypeDiagnostics__?: PrototypeDiagnostics;
					};
					return {
						hasNormalResults: win.__normalUtciResults__ != null,
						numHours: win.__normalUtciResults__?.numHours ?? null,
						monthIndex: win.__normalUtciResults__?.monthIndex ?? null,
						hourCount: win.__normalUtciResults__?.utciByHour?.length ?? null,
						hasOnDemandDiagnostics: win.__onDemandPrototypeDiagnostics__ != null
					};
				});
			}, { timeout: 120_000 })
			.toMatchObject({
				hasNormalResults: true,
				numHours: 24,
				monthIndex: 7,
				hourCount: 24,
				hasOnDemandDiagnostics: false
			});
	});

	test('honors explicit UTCI render override diagnostics on the debug route', async ({ page }) => {
		await page.goto('/debug?onDemandPrototype=1&utciRender=data');

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
		await page.goto('/debug?onDemandPrototype=1');
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
		await page.goto('/debug?onDemandPrototype=1');
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
		await page.goto('/debug?onDemandPrototype=1&disableTooltip=1');
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
		await page.goto('/debug?onDemandPrototype=1');
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
		await page.goto('/debug?onDemandPrototype=1&syntheticBridge=1');

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
		test.setTimeout(120_000);
		await page.goto('/debug?parity=1&onDemandPrototype=1&compareOneHour=1');

		const hasWebGpu = await page.evaluate(() => Boolean(navigator.gpu));
		const requireWebGpu = process.env.REQUIRE_WEBGPU_ON_DEMAND === '1';

		test.skip(
			!hasWebGpu && !requireWebGpu,
			'WebGPU unavailable in this runtime and REQUIRE_WEBGPU_ON_DEMAND is not set.'
		);

		await expect
			.poll(async () => {
				return page.evaluate(() => {
					return (window as Window & {
						__onDemandPrototypeComparison__?: PrototypeComparison;
					}).__onDemandPrototypeComparison__;
				});
			}, { timeout: 120_000 })
			.toBeTruthy();

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
