import { test, expect } from '@playwright/test';
import { resolve } from 'node:path';
import { loadReferenceFromFs } from '../../src/lib/parity/loadReferenceFromFs';
import { compareParityFullDay } from '../../src/lib/parity/compareParity';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
/** Wait up to 60s for compute (typically ~10s in browser). Poll every 1s to avoid hammering the page. */
const PARITY_WAIT_MS = 60_000;
const POLL_INTERVAL_MS = 1000;

/**
 * Smoke test: WebGPU live UTCI completes and exposes results.
 * Point-to-point parity vs .bin is deferred; focus is on intermediate stages (solar, sky, MRT) next.
 */
test.describe('WebGPU live UTCI (Ben-Gurion base)', () => {
	test('debug viewer completes compute and sets __parityResults__', async ({ page }) => {
		test.setTimeout(90_000);
		const basePath = resolve(REPO_ROOT, 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday');
		const ref = await loadReferenceFromFs(basePath);
		expect(ref.data.utciByHour.length).toBe(24);
		expect(ref.data.numPositions).toBeGreaterThan(0);

		await page.goto(
			`/debug-webgpu-utci?parity=1&analysis=${encodeURIComponent('Ben-Gurion/20250815_grid_2m_fullday')}`
		);

		await page.waitForFunction(
			() => (window as unknown as { __parityResults__?: unknown }).__parityResults__ != null,
			{ timeout: PARITY_WAIT_MS, polling: POLL_INTERVAL_MS }
		);

		const webgpu = (await page.evaluate(() => (window as unknown as { __parityResults__?: unknown }).__parityResults__)) as {
			utciByHour: number[][];
			positions: number[];
			numPoints: number;
			numHours: number;
		};
		const debug = (await page.evaluate(() => (window as unknown as { __parityDebug__?: unknown }).__parityDebug__)) as
			| { parityMode: boolean; useParityGrid: boolean; search: string; baseNumPositions?: number }
			| undefined;
		if (debug) console.log('[parity debug]', debug);

		expect(webgpu).toBeDefined();
		expect(webgpu.numHours).toBe(24);
		expect(webgpu.numPoints).toBeGreaterThan(0);

		// Optional: if grid sizes match, log UTCI comparison metrics (no assertion).
		if (webgpu.numPoints === ref.data.numPositions) {
			const utciWebgpuByHour = webgpu.utciByHour.map((arr) => new Float32Array(arr));
			const { byHour, overallPass, worstHour } = compareParityFullDay({
				utciRefByHour: ref.data.utciByHour,
				utciWebgpuByHour,
				toleranceC: 2,
			});
			const worst = byHour[worstHour];
			console.log(
				`[parity] Same grid: overallPass=${overallPass}, worstHour=${worstHour}, worstMaxError=${worst?.maxError?.toFixed(3)}°C, rmse=${worst?.rmse?.toFixed(3)}`
			);
		} else {
			console.log(
				`[parity] Grid mismatch (webgpu=${webgpu.numPoints}, ref=${ref.data.numPositions}); point-to-point comparison skipped. Focus on intermediate stages (solar, sky, MRT).`
			);
		}
	});
});
