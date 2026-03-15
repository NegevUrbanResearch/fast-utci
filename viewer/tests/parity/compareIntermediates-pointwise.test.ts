import { describe, it, expect } from 'vitest';
import { compareIntermediates } from '$lib/parity/compareIntermediates';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';

describe('strict pointwise parity', () => {
	it('fails immediately on length mismatch', () => {
		expect(() =>
			compareIntermediates({
				ref: new Float32Array([1, 2, 3]),
				webgpu: new Float32Array([1, 2]),
				tolerance: 1e-5
			})
		).toThrow(/Length mismatch/);
	});

	it('strict mode fails when only one side has short_erf', () => {
		const tempDir = mkdtempSync(join(tmpdir(), 'parity-erf-policy-'));
		const basePath = join(tempDir, 'fixture');
		try {
			writeFileSync(
				`${basePath}_mrt.json`,
				JSON.stringify({
					numPositions: 2,
					numHours: 1,
					mrt: [20, 21],
					short_erf: [5, 5]
				})
			);
			writeFileSync(
				`${basePath}_webgpu_mrt.json`,
				JSON.stringify({
					numPositions: 2,
					numHours: 1,
					mrt: [20, 21]
				})
			);

			const proc = spawnSync(
				'npx',
				['tsx', 'scripts/compare-parity.ts', '--base-path', basePath, '--mode', 'strict'],
				{
					cwd: join(process.cwd()),
					encoding: 'utf8',
					shell: process.platform === 'win32'
				}
			);

			expect(proc.status).toBe(1);
			expect(`${proc.stdout}\n${proc.stderr}`).toMatch(/short_erf: FAIL .*present in ref only/i);
		} finally {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});
});
