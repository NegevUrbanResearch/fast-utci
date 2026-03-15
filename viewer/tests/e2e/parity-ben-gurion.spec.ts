import { test, expect } from '@playwright/test';
import { resolve } from 'node:path';
import { loadReferenceFromFs } from '../../src/lib/parity/loadReferenceFromFs';
import { compareUtciPointwise } from '../../src/lib/parity/compareUtciPointwise';

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
			() => {
				const w = window as unknown as {
					__parityResults__?: unknown;
					__parityCollectionError__?: string;
					__parityIntermediatesError__?: string;
					__parityCollectionStatus__?: { state: 'running' | 'success' | 'error' | 'timeout' };
				};
				if (typeof w.__parityCollectionError__ === 'string') return true;
				if (typeof w.__parityIntermediatesError__ === 'string') return true;
				if (w.__parityCollectionStatus__?.state === 'error') return true;
				if (w.__parityCollectionStatus__?.state === 'timeout') return true;
				return w.__parityResults__ != null;
			},
			{ timeout: PARITY_WAIT_MS, polling: POLL_INTERVAL_MS }
		);
		const collectError = await page.evaluate(() => {
			const w = window as unknown as {
				__parityCollectionError__?: string;
				__parityIntermediatesError__?: string;
				__parityCollectionStatus__?: unknown;
				__parityCollectionLog__?: unknown;
			};
			return {
				error: w.__parityCollectionError__ ?? w.__parityIntermediatesError__ ?? null,
				status: w.__parityCollectionStatus__ ?? null,
				log: w.__parityCollectionLog__ ?? null
			};
		});
		if (collectError.error) {
			throw new Error(
				`Parity compute failed before results: ${collectError.error}\nstatus=${JSON.stringify(collectError.status)}\nlog=${JSON.stringify(collectError.log)}`
			);
		}

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

		// If grid sizes match, assert pointwise UTCI parity with actionable diagnostics.
		if (webgpu.numPoints === ref.data.numPositions) {
			const pointwise = compareUtciPointwise({
				ref: ref.data.utciByHour.map((arr) => Array.from(arr)),
				webgpu: webgpu.utciByHour,
				tolerance: 2
			});
			expect(
				pointwise.pass,
				[
					`Pointwise UTCI parity failed`,
					`phase=${(collectError.status as { phase?: string } | null)?.phase ?? 'unknown'}`,
					`status=${JSON.stringify(collectError.status)}`,
					`worstHour=${pointwise.worst?.hour ?? 'n/a'}`,
					`worstPointIndex=${pointwise.worst?.pointIndex ?? 'n/a'}`,
					`ref=${pointwise.worst?.ref?.toFixed(6) ?? 'n/a'}`,
					`webgpu=${pointwise.worst?.webgpu?.toFixed(6) ?? 'n/a'}`,
					`diff=${pointwise.worst?.diff?.toFixed(6) ?? 'n/a'}`,
					`rmse=${pointwise.rmse.toFixed(6)}`,
					`maxError=${pointwise.maxError.toFixed(6)}`
				].join(' | ')
			).toBe(true);
		} else {
			console.log(
				`[parity] Grid mismatch (webgpu=${webgpu.numPoints}, ref=${ref.data.numPositions}); point-to-point comparison skipped. Focus on intermediate stages (solar, sky, MRT).`
			);
		}
	});
});
