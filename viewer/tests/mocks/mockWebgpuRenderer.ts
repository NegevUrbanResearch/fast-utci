import { vi } from 'vitest';

export const setSizeSpy = vi.fn();
export const initSpy = vi.fn(async () => undefined);

const rendererInstances: unknown[] = [];

export function resetMockWebgpuRenderer() {
	setSizeSpy.mockReset();
	initSpy.mockClear();
	rendererInstances.length = 0;
}

export class WebGPURenderer {
	backend = { isWebGPUBackend: true };
	info = { dispose: () => {} };
	setSize = (width: number, height: number) => setSizeSpy(width, height);
	init = initSpy;

	constructor(_options: unknown) {
		rendererInstances.push(this);
	}
}
