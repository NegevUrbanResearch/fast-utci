import { vi } from 'vitest';

export const setSizeSpy = vi.fn();
export const initSpy = vi.fn(async () => undefined);

const rendererInstances: unknown[] = [];
let lastRendererOptions: unknown = null;

export function resetMockWebgpuRenderer() {
	setSizeSpy.mockReset();
	initSpy.mockClear();
	rendererInstances.length = 0;
	lastRendererOptions = null;
}

export function getLastRendererOptions() {
	return lastRendererOptions;
}

export class WebGPURenderer {
	backend = {
		isWebGPUBackend: true,
		device: {
			limits: {
				maxStorageBufferBindingSize: 512 * 1024 * 1024,
				maxBufferSize: 1024 * 1024 * 1024
			}
		}
	};
	info = { dispose: () => {} };
	setSize = (width: number, height: number) => setSizeSpy(width, height);
	init = initSpy;

	constructor(options: unknown) {
		lastRendererOptions = options;
		rendererInstances.push(this);
	}
}
