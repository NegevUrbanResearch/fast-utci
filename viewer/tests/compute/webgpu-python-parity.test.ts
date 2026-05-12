import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { calculateUTCI } from '$lib/compute/core/utci';

interface ParityFixture {
	name: string;
	description: string;
	inputs: {
		airTempC: number;
		mrtC: number;
		windSpeedMs: number;
		relativeHumidityPct: number;
	};
	expected: {
		pythonUtciC: number;
		webgpuUtciC: number;
		toleranceC: number;
	};
}

function loadFixture(name: string): ParityFixture {
	const path = resolve(process.cwd(), `tests/fixtures/parity/${name}.json`);
	return JSON.parse(readFileSync(path, 'utf8')) as ParityFixture;
}

describe('WebGPU vs Python parity fixture', () => {
	it('matches python fixture within tolerance for one-point one-hour case', () => {
		const fixture = loadFixture('single-point-single-hour');
		const calcUtci = calculateUTCI(
			fixture.inputs.airTempC,
			fixture.inputs.mrtC,
			fixture.inputs.windSpeedMs,
			fixture.inputs.relativeHumidityPct
		);

		// Fixture check: CPU UTCI stays close to stored Python benchmark.
		expect(Math.abs(calcUtci - fixture.expected.pythonUtciC)).toBeLessThanOrEqual(
			fixture.expected.toleranceC
		);

		// Fixture check: stored WebGPU benchmark remains close to Python benchmark.
		expect(
			Math.abs(fixture.expected.webgpuUtciC - fixture.expected.pythonUtciC)
		).toBeLessThanOrEqual(fixture.expected.toleranceC);
	});
});
