import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('webgpuUtciPipeline implementation locks', () => {
	const source = readFileSync(
		resolve(process.cwd(), 'src/lib/compute/webgpuUtciPipeline.ts'),
		'utf8'
	);

	it('uses async compute pipeline creation', () => {
		expect(source.includes('createComputePipelineAsync')).toBe(true);
		expect(source.includes('createComputePipeline(')).toBe(false);
	});

	it('handles GPU device loss by clearing cached device promise', () => {
		expect(source.includes('device.lost.then')).toBe(true);
		expect(source.includes('cachedDevicePromise = null')).toBe(true);
	});

	it('reads UTCI by gathered hour slice instead of full-buffer cache field', () => {
		expect(source.includes('GATHER_SLICE_SHADER')).toBe(true);
		expect(source.includes('cachedUtciData')).toBe(false);
	});
});
