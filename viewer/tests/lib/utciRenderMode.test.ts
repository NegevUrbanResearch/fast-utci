import { describe, expect, it } from 'vitest';
import {
	DEFAULT_MAIN_UTCI_RENDER_MODE,
	resolveMainRouteUtciSurfaceBackend,
	resolveUtciSurfaceBackend,
} from '$lib/utciRenderMode';

describe('utciRenderMode', () => {
	it('defaults the main route to auto render mode', () => {
		expect(DEFAULT_MAIN_UTCI_RENDER_MODE).toBe('auto');
	});

	it('preserves explicit gpu resolution outside compare mode', () => {
		expect(resolveUtciSurfaceBackend('gpu', 'webgpu')).toBe('gpuNative');
		expect(
			resolveMainRouteUtciSurfaceBackend({
				mode: 'gpu',
				rendererBackend: 'webgpu',
				isComparing: false,
			}),
		).toBe('gpuNative');
	});

	it('uses gpuNative in compare mode when webgpu can drive auto or gpu', () => {
		expect(
			resolveMainRouteUtciSurfaceBackend({
				mode: 'gpu',
				rendererBackend: 'webgpu',
				isComparing: true,
			}),
		).toBe('gpuNative');
		expect(
			resolveMainRouteUtciSurfaceBackend({
				mode: 'auto',
				rendererBackend: 'webgpu',
				isComparing: true,
			}),
		).toBe('gpuNative');
	});

	it('keeps the explicit data fallback in compare mode', () => {
		expect(
			resolveMainRouteUtciSurfaceBackend({
				mode: 'data',
				rendererBackend: 'webgpu',
				isComparing: true,
			}),
		).toBe('dataTexture');
		expect(
			resolveMainRouteUtciSurfaceBackend({
				mode: 'auto',
				rendererBackend: 'unknown',
				isComparing: true,
			}),
		).toBe('dataTexture');
	});
});
