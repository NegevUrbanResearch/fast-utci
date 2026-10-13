import { expect, test, type Page } from '@playwright/test';
import { writeFile } from 'node:fs/promises';

const BG = 'Ben-Gurion/20250815_grid_2m_fullday';
const NZ = 'Ness-Tziona/exploded/nes_tziona_unblock_2';
const url = (id = BG) => `/?analysis=${encodeURIComponent(id)}&utciRender=auto&utciRenderDiagnostics=1`;

async function diagnostics(page: Page) {
	return page.evaluate(() => (window as any).__utciRenderDiagnostics__);
}

async function publication(
	page: Page,
	id: string,
	month: number,
	hour: number,
	source = 'compute-buffer-selected-hour'
) {
	const handle = await page.waitForFunction(({ key, month, hour, source }) => {
		const d = (window as any).__utciRenderDiagnostics__;
		return d?.baseLiveReady === true && d.baseSelectionKey === key &&
			d.baseSceneSelectionKey === key && d.baseSelectedMonthIndex === month &&
			d.baseSelectedHourIndex === hour && d.baseRenderContextTimeIndex === month * 24 + hour &&
			d.utciSurfaceSource === source && d.baseRenderTransport === source ? d : null;
	}, { key: `${id}|utci|${month}|${hour}`, month, hour, source }, {
		timeout: source === 'compute-buffer-selected-hour' ? 90_000 : 30_000
	});
	const d = await handle.jsonValue();
	if (source === 'compute-buffer-selected-hour') {
		expect(d.baseSameDeviceForComputeAndRender).toBe(true);
		expect(d.gpuResidentCopyRequestId).toBe(d.baseSurfaceRequestId);
		expect(d.selectedHourRuntimeContract.strongVisibleGpuPath).toBe(true);
		expect(d.selectedHourRuntimeContract.visibleSelectedHourReadbackCount).toBe(0);
	}
	return d;
}

async function selectTime(page: Page, kind: 'hour' | 'month', value: number) {
	await page.getByRole('button', { name: kind === 'hour' ? /^day$/i : /^month$/i }).click();
	const slider = page.getByRole('slider', {
		name: kind === 'hour' ? /select analysis hour/i : /select month/i
	});
	await slider.focus();
	await slider.press('Home');
	for (let i = 0; i < value; i++) {
		await slider.press('ArrowRight');
	}
	await expect(slider).toHaveAttribute('aria-valuenow', String(value));
	await page.mouse.move(0, 0);
}

async function hover(page: Page) {
	const handle = await page.waitForFunction(
		() => (window as any).__mainRouteTooltipProbePosition__?.(),
		undefined,
		{ timeout: 15_000 }
	);
	const point = await handle.jsonValue();
	await page.mouse.move(point.clientX, point.clientY);
	await expect(page.getByRole('tooltip')).toBeVisible();
	await expect(page.getByRole('tooltip')).toContainText(/UTCI/i);
	const result = await page.evaluate(() => (window as any).__mainRouteLastTooltip__);
	expect(result.positionIndex).toBe(point.positionIndex);
	expect(Number.isFinite(result.value)).toBe(true);
	return { point, result };
}

test.describe('main route selection ownership and localized UTCI', () => {
	test.setTimeout(180000);

	test.afterEach(async ({ page }, info) => {
		const evidence = await page.evaluate(() => ({
			diagnostics: (window as any).__utciRenderDiagnostics__,
			copyFailure: (window as any).__copyFailureEvidence,
			tooltip: (window as any).__mainRouteLastTooltip__
		})).catch(error => ({ error: String(error) }));
		const path = info.outputPath('actual-main-route-diagnostics.json');
		await writeFile(path, JSON.stringify(evidence, null, 2));
		await info.attach('actual-main-route-diagnostics', { path, contentType: 'application/json' });
	});

	test('a delayed old GLB cannot publish while the next project metadata is pending', async ({ page }) => {
		let releaseOld!: () => void;
		let releaseNext!: () => void;
		let oldStarted!: () => void;
		let nextStarted!: () => void;
		let oldDelivered!: () => void;
		const oldGate = new Promise<void>(resolve => { releaseOld = resolve; });
		const nextGate = new Promise<void>(resolve => { releaseNext = resolve; });
		const oldRequest = new Promise<void>(resolve => { oldStarted = resolve; });
		const nextRequest = new Promise<void>(resolve => { nextStarted = resolve; });
		const delivered = new Promise<void>(resolve => { oldDelivered = resolve; });
		await page.route('**/original_with_layers.glb*', async (route) => {
			const response = await route.fetch();
			oldStarted();
			await oldGate;
			await route.fulfill({ response });
			oldDelivered();
		});
		await page.route('**/Ness-Tziona/exploded/nes_tziona_unblock_2.json', async (route) => {
			const response = await route.fetch();
			nextStarted();
			await nextGate;
			await route.fulfill({ response });
		});
		try {
			await page.goto(url());
			await oldRequest;
			await page.getByTestId('project-select').selectOption('Ness-Tziona');
			await nextRequest;
			releaseOld();
			await delivered;
			// Give the real GLTF parser and modelLoaded event an opportunity to complete.
			await page.waitForTimeout(750);
			expect((await diagnostics(page))?.baseLiveReady).not.toBe(true);
			releaseNext();
			const latest = await publication(page, NZ, 7, 0);
			expect(latest.basePointCount).toBeGreaterThan(0);
			await expect(page.getByTestId('project-select')).toHaveValue('Ness-Tziona');
			const metadata = await page.evaluate(async () => {
				const { analysisStore } = await new Function('return import("/src/lib/stores/analysisStore.ts")')();
				let value: any;
				const stop = analysisStore.subscribe((a: any) => { value = a?.metadata; });
				stop();
				return value;
			});
			expect(metadata.source_analysis_id).toBe(NZ);
			expect(metadata.model_file).toContain('Ness-Tziona/nes_tziona_unblock_2.glb');
		}
		finally {
			releaseOld();
			releaseNext();
		}
	});

	test('UTCI hover reads four bytes, reuses the point, and invalidates for hour and month', async ({ page }) => {
		await page.goto(url());
		await publication(page, BG, 7, 0);
		const initial = await diagnostics(page);
		const first = await hover(page);
		await expect.poll(async () => (await diagnostics(page)).tooltipInteraction.metricPointReadbackCount).toBe((initial.tooltipInteraction.metricPointReadbackCount ?? 0) + 1);
		const read = (await diagnostics(page)).tooltipInteraction;
		expect(read.metricPointReadbackBytes - initial.tooltipInteraction.metricPointReadbackBytes).toBe(4);
		expect(read.metricPointReadbackLastBytes).toBe(4);
		await page.mouse.move(0, 0);
		await expect(page.getByRole('tooltip')).toBeHidden();
		await page.mouse.move(first.point.clientX, first.point.clientY);
		await expect(page.getByRole('tooltip')).toBeVisible();
		await expect.poll(async () => (await diagnostics(page)).tooltipInteraction.metricPointReadbackCacheHitCount).toBeGreaterThan(read.metricPointReadbackCacheHitCount);
		expect((await diagnostics(page)).tooltipInteraction.metricPointReadbackCount).toBe(read.metricPointReadbackCount);
		for (const selection of [{ kind: 'hour' as const, value: 1, month: 7, hour: 1 }, { kind: 'month' as const, value: 8, month: 8, hour: 1 }]) {
			await selectTime(page, selection.kind, selection.value);
			await expect(page.getByRole('tooltip')).toBeHidden();
			await publication(page, BG, selection.month, selection.hour);
			const before = (await diagnostics(page)).tooltipInteraction;
			const current = await hover(page);
			expect(current.result.tooltipHourIndex).toBe(selection.month * 24 + selection.hour);
			await expect.poll(async () => (await diagnostics(page)).tooltipInteraction.metricPointReadbackCount).toBe(before.metricPointReadbackCount + 1);
			expect((await diagnostics(page)).tooltipInteraction.metricPointReadbackBytes - before.metricPointReadbackBytes).toBe(4);
		}
	});

});
