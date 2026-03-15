import { afterEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { buildParityReport } from '$lib/parity/buildParityReport';

const tempDirs: string[] = [];

afterEach(() => {
	for (const dir of tempDirs.splice(0)) {
		rmSync(dir, { recursive: true, force: true });
	}
});

function writeReferenceBin(path: string, numPositions: number, numHours: number, positions: number[], utci: number[]): void {
	const bytes = 8 + numPositions * 3 * 4 + numPositions * numHours * 4;
	const buffer = new ArrayBuffer(bytes);
	const view = new DataView(buffer);
	let offset = 0;
	view.setUint32(offset, numPositions, true);
	offset += 4;
	view.setUint32(offset, numHours, true);
	offset += 4;
	for (const value of positions) {
		view.setFloat32(offset, value, true);
		offset += 4;
	}
	for (const value of utci) {
		view.setFloat32(offset, value, true);
		offset += 4;
	}
	writeFileSync(path, Buffer.from(buffer));
}

describe('buildParityReport', () => {
	it('keeps pointwise UTCI as diagnostics without failing stats summary', async () => {
		const dir = mkdtempSync(join(tmpdir(), 'parity-report-'));
		tempDirs.push(dir);
		const basePath = join(dir, 'sample');

		writeFileSync(
			`${basePath}.json`,
			JSON.stringify({
				num_positions: 2,
				hours: [0],
				analysis_type: 'utci',
				utci_range: { min: 20, max: 25, mean: 22.5 }
			})
		);
		writeReferenceBin(`${basePath}.bin`, 2, 1, [0, 0, 0, 1, 0, 0], [20, 20]);
		writeFileSync(
			`${basePath}_webgpu_utci.json`,
			JSON.stringify({
				numPoints: 2,
				numHours: 1,
				positions: [0, 0, 0, 1, 0, 0],
				utciByHour: [[20, 25]],
				utci_range: { min: 20, max: 25, mean: 22.5 }
			})
		);

		const report = await buildParityReport(basePath);
		expect(report.utci?.pass).toBe(true);
		expect(report.utciPointwise?.pass).toBe(false);
		expect(report.summary.pass).toBe(true);
		expect(report.summary.failCount).toBe(0);
	});
});
