import { describe, expect, it, vi } from 'vitest';
import {
	createSelectedHourOutputHandle,
	disposeSelectedHourOutputHandle
} from '$lib/compute/selected-hour/selectedHourOutputHandle';

describe('selectedHourOutputHandle', () => {
	it('disposes the owned GPU buffer once', () => {
		const destroy = vi.fn();
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy } as unknown as GPUBuffer,
			byteLength: 16,
			requestId: 5,
			timeIndex: 12,
			source: 'webgpu-on-demand-snapshot'
		});

		disposeSelectedHourOutputHandle(handle);
		disposeSelectedHourOutputHandle(handle);

		expect(destroy).toHaveBeenCalledTimes(1);
		expect(handle.disposed).toBe(true);
	});

	it('keeps request identity with the buffer handle', () => {
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy: vi.fn() } as unknown as GPUBuffer,
			byteLength: 32,
			requestId: 17,
			timeIndex: 90,
			source: 'webgpu-on-demand-snapshot'
		});

		expect(handle.requestId).toBe(17);
		expect(handle.timeIndex).toBe(90);
		expect(handle.byteLength).toBe(32);
		expect(handle.source).toBe('webgpu-on-demand-snapshot');
	});
});
