import { afterEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { loadWebgpuCollectedFromFs } from '$lib/parity/loadWebgpuCollectedFromFs';

const tempDirs: string[] = [];

afterEach(() => {
	for (const dir of tempDirs.splice(0)) {
		rmSync(dir, { recursive: true, force: true });
	}
});

describe('WebGPU parity artifact contract', () => {
	it('rejects UTCI artifacts when per-hour point counts do not match numPoints', async () => {
		const dir = mkdtempSync(join(tmpdir(), 'utci-contract-'));
		tempDirs.push(dir);
		const basePath = join(dir, 'analysis');

		writeFileSync(
			`${basePath}_webgpu_utci.json`,
			JSON.stringify({
				numPoints: 2,
				numHours: 2,
				utciByHour: [
					[25, 26],
					[27]
				],
				utci_range: { min: 25, max: 27, mean: 26 }
			})
		);

		await expect(loadWebgpuCollectedFromFs(basePath)).rejects.toThrow(/utciByHour/i);
	});

	it('rejects MRT artifacts when optional ERF arrays are present with invalid lengths', async () => {
		const dir = mkdtempSync(join(tmpdir(), 'mrt-contract-'));
		tempDirs.push(dir);
		const basePath = join(dir, 'analysis');

		writeFileSync(
			`${basePath}_webgpu_mrt.json`,
			JSON.stringify({
				numPositions: 3,
				numHours: 2,
				mrt: [10, 11, 12, 13, 14, 15],
				short_erf: [0.1, 0.2]
			})
		);

		await expect(loadWebgpuCollectedFromFs(basePath)).rejects.toThrow(/short_erf/i);
	});
});
