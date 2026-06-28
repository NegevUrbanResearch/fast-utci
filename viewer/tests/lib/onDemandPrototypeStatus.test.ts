import { describe, expect, it } from 'vitest';
import {
	deriveOnDemandPrototypeStatus,
	type OnDemandPrototypeStatusInputs
} from '$lib/compute/on-demand/onDemandPrototypeStatus';

function createInputs(
	overrides: Partial<OnDemandPrototypeStatusInputs> = {}
): OnDemandPrototypeStatusInputs {
	return {
		diagnostics: {
			navigatorGpu: true,
			rendererBackend: 'unknown',
			path: 'idle',
			usedExposureOnlyPrecompute: false,
			usedRunAllForSelectedHour: false,
			oneHourOutputBytes: 0,
			renderTransport: 'none'
		},
		syntheticBridgeEnabled: false,
		strictExposureOnlyEnabled: false,
		compareOneHourEnabled: false,
		hasOnDemandPrototypeComparison: false,
		compareHoursEnabled: false,
		hasOnDemandMultiHourComparison: false,
		compareMonthHoursEnabled: false,
		hasCompletedOnDemandMonthHourComparison: false,
		...overrides
	};
}

describe('deriveOnDemandPrototypeStatus', () => {
	it('returns unsupported when navigator.gpu is absent and no runtime proof exists', () => {
		const status = deriveOnDemandPrototypeStatus(
			createInputs({
				diagnostics: {
					navigatorGpu: false,
					rendererBackend: 'unknown',
					path: 'idle',
					usedExposureOnlyPrecompute: false,
					usedRunAllForSelectedHour: false,
					oneHourOutputBytes: 0,
					renderTransport: 'none'
				}
			})
		);

		expect(status).toBe('unsupported');
	});

	it('keeps gpuNative resolution alone from suppressing unsupported', () => {
		const status = deriveOnDemandPrototypeStatus(
			createInputs({
				diagnostics: {
					navigatorGpu: false,
					rendererBackend: 'unknown',
					utciRenderResolved: 'gpuNative',
					path: 'idle',
					usedExposureOnlyPrecompute: false,
					usedRunAllForSelectedHour: false,
					oneHourOutputBytes: 0,
					renderTransport: 'none'
				}
			})
		);

		expect(status).toBe('unsupported');
	});

	it('prefers runtime WebGPU proof over navigator.gpu=false on the standard route path', () => {
		const status = deriveOnDemandPrototypeStatus(
			createInputs({
				diagnostics: {
					navigatorGpu: false,
					rendererBackend: 'webgpu',
					renderTransport: 'compute-buffer-selected-hour',
					utciSurfaceSource: 'compute-buffer-selected-hour',
					path: 'idle',
					usedExposureOnlyPrecompute: false,
					usedRunAllForSelectedHour: false,
					oneHourOutputBytes: 0
				}
			})
		);

		expect(status).toBe('ready');
	});

	it('keeps the synthetic bridge flow in diagnostics until its route-specific proof appears', () => {
		const status = deriveOnDemandPrototypeStatus(
			createInputs({
				syntheticBridgeEnabled: true,
				diagnostics: {
					navigatorGpu: false,
					rendererBackend: 'webgpu',
					path: 'idle',
					bridgeAttached: false,
					visibleColorVariance: 0,
					usedExposureOnlyPrecompute: false,
					usedRunAllForSelectedHour: false,
					oneHourOutputBytes: 0,
					renderTransport: 'none'
				}
			})
		);

		expect(status).toBe('diagnostics');
	});

	it('keeps the strict exposure-only flow honest after bypassing the weak capability short-circuit', () => {
		const status = deriveOnDemandPrototypeStatus(
			createInputs({
				strictExposureOnlyEnabled: true,
				diagnostics: {
					navigatorGpu: false,
					rendererBackend: 'webgpu',
					path: 'exposure-only-f32',
					usedExposureOnlyPrecompute: true,
					usedRunAllForSelectedHour: false,
					liveAnalysisConstructedForSelectedHour: false,
					oneHourOutputBytes: 4096,
					renderTransport: 'compute-buffer-selected-hour'
				}
			})
		);

		expect(status).toBe('ready');
	});

	it('preserves explicit error status above capability and proof checks', () => {
		const status = deriveOnDemandPrototypeStatus(
			createInputs({
				diagnostics: {
					navigatorGpu: false,
					rendererBackend: 'webgpu',
					path: 'idle',
					error: 'Synthetic bridge validation failed.',
					usedExposureOnlyPrecompute: false,
					usedRunAllForSelectedHour: false,
					oneHourOutputBytes: 0,
					renderTransport: 'compute-buffer-selected-hour'
				}
			})
		);

		expect(status).toBe('error');
	});
});
