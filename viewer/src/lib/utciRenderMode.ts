import type { UtciSurfaceBackendType } from '$lib/services/pointCloudService';

export type UtciRenderMode = 'auto' | 'gpu' | 'data';
export type UtciRendererBackend = 'webgpu' | 'unknown';

export const DEFAULT_DEBUG_UTCI_RENDER_MODE: UtciRenderMode = 'auto';
export const DEFAULT_MAIN_UTCI_RENDER_MODE: UtciRenderMode = 'auto';

export function parseUtciRenderMode(
	searchParams: URLSearchParams,
	fallback: UtciRenderMode
): UtciRenderMode {
	const requested = searchParams.get('utciRender');
	return requested === 'auto' || requested === 'gpu' || requested === 'data'
		? requested
		: fallback;
}

export function resolveUtciSurfaceBackend(
	mode: UtciRenderMode,
	rendererBackend: UtciRendererBackend
): UtciSurfaceBackendType {
	switch (mode) {
		case 'gpu':
			return 'gpuNative';
		case 'data':
			return 'dataTexture';
		case 'auto':
		default:
			return rendererBackend === 'webgpu' ? 'gpuNative' : 'dataTexture';
	}
}

export function resolveMainRouteUtciSurfaceBackend(params: {
	mode: UtciRenderMode;
	rendererBackend: UtciRendererBackend;
	isComparing: boolean;
}): UtciSurfaceBackendType {
	return resolveUtciSurfaceBackend(params.mode, params.rendererBackend);
}
