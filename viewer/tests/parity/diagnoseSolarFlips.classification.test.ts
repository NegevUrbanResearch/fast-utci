import { describe, expect, it } from 'vitest';
import { classifyMismatch } from '../../scripts/diagnose-solar-flips';

describe('diagnose-solar-flips classification', () => {
	it('derives flip direction, margin delta, and heuristic class', () => {
		expect(
			classifyMismatch({
				refBinary: 0,
				webgpuBinary: 1,
				refMarginToThreshold: -0.2,
				webgpuMarginToThreshold: 0.15,
				shortWaveCompositeDelta: 1.5,
				longWaveCompositeDelta: -0.5
			})
		).toEqual({
			binaryFlipDirection: '0->1',
			marginDelta: 0.35,
			mismatchClass: 'promotion-shortwave-dominant'
		});
	});

	it('labels demotions symmetrically', () => {
		expect(
			classifyMismatch({
				refBinary: 1,
				webgpuBinary: 0,
				refMarginToThreshold: 0.25,
				webgpuMarginToThreshold: -0.4,
				shortWaveCompositeDelta: -2,
				longWaveCompositeDelta: -1
			})
		).toMatchObject({
			binaryFlipDirection: '1->0',
			marginDelta: -0.65,
				mismatchClass: 'demotion-shortwave-dominant'
			});
	});

	it('returns unknown when either composite delta is missing', () => {
		expect(
			classifyMismatch({
				refBinary: 0,
				webgpuBinary: 1,
				refMarginToThreshold: -0.1,
				webgpuMarginToThreshold: 0.2,
				shortWaveCompositeDelta: null,
				longWaveCompositeDelta: 0.4
			})
		).toMatchObject({
			binaryFlipDirection: '0->1',
			marginDelta: 0.30000000000000004,
			mismatchClass: 'unknown'
		});
	});
});
