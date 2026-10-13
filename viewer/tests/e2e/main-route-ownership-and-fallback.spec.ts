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

});
