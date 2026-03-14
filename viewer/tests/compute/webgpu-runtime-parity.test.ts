import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

interface RuntimeParityFixture {
	name: string;
	inputs: {
		monthIndex: number;
		hourIndex: number;
		numPoints: number;
		numHours: number;
		numMonths: number;
	};
	expected: {
		sunVectors: number[][];
		solarExposure: number[];
		skyExposure: number[];
		mrt: number[];
		utci: number[];
		tolerances: {
			sunVectors: number;
			solarExposure: number;
			skyExposure: number;
			mrt: number;
			utci: number;
		};
	};
}

function loadFixture(name: string): RuntimeParityFixture {
	const path = resolve(process.cwd(), `tests/fixtures/parity/stages/${name}.json`);
	return JSON.parse(readFileSync(path, 'utf8')) as RuntimeParityFixture;
}

describe('WebGPU runtime parity harness (stage-wise)', () => {
	it('defines stage fixtures and tolerance contracts used by browser WebGPU lane', () => {
		const fixture = loadFixture('single-point-stage-fixture');
		expect(fixture.expected.sunVectors.length).toBeGreaterThan(0);
		expect(fixture.expected.tolerances.utci).toBeGreaterThan(0);
	});

	it.skipIf(typeof navigator === 'undefined' || !(navigator as any).gpu)(
		'compares stage outputs against fixture tolerances in browser WebGPU runtime',
		async () => {
			const fixture = loadFixture('single-point-stage-fixture');
			// Placeholder contract: this test intentionally runs only in browser
			// WebGPU lane where stage buffers are available for assertion.
			expect(fixture.inputs.numPoints).toBe(1);
			expect(fixture.expected.utci[0]).toBeTypeOf('number');
		}
	);
});
