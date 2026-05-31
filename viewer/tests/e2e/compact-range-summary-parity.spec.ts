import { expect, test, type Page } from '@playwright/test';
import type * as WebgpuUtciPipelineModule from '$lib/compute/gpu/webgpuUtciPipeline';

const F32_MAX_VALUE = 3.4028234663852886e38;

type RangeSummary = {
	timeIndex: number;
	range: { min: number; max: number } | null;
	validCount: number;
	readbackBytes: number;
	reductionPassCount: number;
	debugLabel: 'webgpu-on-demand-f32-utci-range-summary';
};

function cpuRange(values: number[]): {
	range: { min: number; max: number } | null;
	validCount: number;
} {
	let min = F32_MAX_VALUE;
	let max = -F32_MAX_VALUE;
	let validCount = 0;
	for (const value of values) {
		if (Number.isFinite(value) && Math.abs(value) <= F32_MAX_VALUE) {
			min = Math.min(min, value);
			max = Math.max(max, value);
			validCount += 1;
		}
	}
	return {
		range: validCount > 0 ? { min, max } : null,
		validCount
	};
}

async function runGpuCompactSummary(page: Page, values: number[]): Promise<RangeSummary> {
	await page.goto('/');
	return page.evaluate(async (rawValues) => {
		if (!navigator.gpu) {
			throw new Error('WebGPU is not available in this browser context');
		}
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) {
			throw new Error('WebGPU adapter is not available in this browser context');
		}
		const device = await adapter.requestDevice();
		const modulePath = '/src/lib/compute/gpu/webgpuUtciPipeline.ts';
		const { __TEST_ONLY_WebgpuUtciComputePipeline } = (await import(
			/* @vite-ignore */ modulePath
		)) as typeof WebgpuUtciPipelineModule;
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device, false);
		try {
			const summary = await pipeline.__TEST_ONLY_reduceRangeValuesForDebug(
				new Float32Array(rawValues)
			);
			return summary;
		} finally {
			pipeline.dispose();
			device.destroy();
		}
	}, values);
}

async function expectGpuMatchesCpu(page: Page, values: number[]) {
	const gpuSummary = await runGpuCompactSummary(page, values);
	const expected = cpuRange(Array.from(new Float32Array(values)));

	expect(gpuSummary.readbackBytes).toBe(16);
	expect(gpuSummary.validCount).toBe(expected.validCount);
	expect(gpuSummary.debugLabel).toBe('webgpu-on-demand-f32-utci-range-summary');
	if (expected.range === null) {
		expect(gpuSummary.range).toBeNull();
		return;
	}

	expect(gpuSummary.range).not.toBeNull();
	expect(gpuSummary.range?.min).toBeCloseTo(expected.range.min, 5);
	expect(gpuSummary.range?.max).toBeCloseTo(expected.range.max, 5);
}

test.describe('compact WebGPU range summaries', () => {
	test('match CPU range for mixed values spanning more than one workgroup', async ({ page }) => {
		const values = Array.from({ length: 600 }, (_, index) => {
			const wave = Math.sin(index * 0.173) * 18;
			const ramp = (index % 47) - 23;
			return wave + ramp * 0.25;
		});
		values[17] = -42.5;
		values[511] = 58.75;

		await expectGpuMatchesCpu(page, values);
	});

	test('match CPU range for equal values', async ({ page }) => {
		await expectGpuMatchesCpu(page, Array(300).fill(12.25));
	});

	test('ignore invalid values while matching CPU range', async ({ page }) => {
		const values = Array.from({ length: 300 }, (_, index) => (index % 31) - 15.5);
		values[0] = Number.NaN;
		values[1] = Number.POSITIVE_INFINITY;
		values[2] = Number.NEGATIVE_INFINITY;
		values[299] = -22.75;

		await expectGpuMatchesCpu(page, values);
	});

	test('return null range for all-invalid values', async ({ page }) => {
		const values = Array.from({ length: 300 }, (_, index) => {
			if (index % 3 === 0) return Number.NaN;
			if (index % 3 === 1) return Number.POSITIVE_INFINITY;
			return Number.NEGATIVE_INFINITY;
		});

		await expectGpuMatchesCpu(page, values);
	});
});
