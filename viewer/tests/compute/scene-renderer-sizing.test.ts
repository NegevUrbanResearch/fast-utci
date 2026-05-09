import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from '@testing-library/svelte/svelte5';
import { getLastRenderer, resetLastRenderer } from '../mocks/threlteCanvasHarness';
import {
	getLastRendererOptions,
	resetMockWebgpuRenderer,
	setSizeSpy
} from '../mocks/mockWebgpuRenderer';
import { LARGE_BUFFER_REQUIRED_LIMITS } from '$lib/compute/webgpuDeviceLimits';

vi.mock('@threlte/core', async () => {
	const Canvas = (await import('../mocks/MockCanvas.svelte')).default;
	return {
		Canvas,
		useScene: () => ({ scene: null }),
		useThrelte: () => ({ invalidate: vi.fn() })
	};
});

vi.mock('$lib/components/scene/SceneBackground.svelte', async () => ({
	default: (await import('../mocks/Empty.svelte')).default
}));

vi.mock('$lib/components/scene/SceneInvalidateSetup.svelte', async () => ({
	default: (await import('../mocks/Empty.svelte')).default
}));

vi.mock('three/webgpu', async () => await import('../mocks/mockWebgpuRenderer'));

import Scene from '$lib/components/scene/Scene.svelte';

describe('Scene WebGPU renderer sizing guard', () => {
	beforeEach(() => {
		resetLastRenderer();
		resetMockWebgpuRenderer();
	});

	it('ignores transient zero width or height sizes and forwards valid positive sizes', () => {
		render(Scene, {
			backgroundColor: 0x000000,
			enableShadows: false
		});

		const renderer = getLastRenderer() as { setSize: (width: number, height: number) => void } | null;

		expect(renderer).not.toBeNull();

		renderer!.setSize(0, 240);
		renderer!.setSize(320, 0);
		renderer!.setSize(320, 240);

		expect(setSizeSpy).toHaveBeenCalledTimes(1);
		expect(setSizeSpy).toHaveBeenCalledWith(320, 240);
	});

	it('requests large WebGPU buffer limits when enabled for GPU-resident compute rendering', () => {
		const onRendererDiagnostics = vi.fn();

		render(Scene, {
			backgroundColor: 0x000000,
			enableShadows: false,
			requestLargeWebgpuLimits: true,
			onRendererDiagnostics
		});

		expect(getLastRendererOptions()).toMatchObject({
			requiredLimits: LARGE_BUFFER_REQUIRED_LIMITS
		});
		expect(onRendererDiagnostics).toHaveBeenCalledWith(
			expect.objectContaining({
				rendererRequiredLimits: LARGE_BUFFER_REQUIRED_LIMITS
			})
		);
	});

	it('keeps default renderer construction unchanged when large limits are not requested', () => {
		render(Scene, {
			backgroundColor: 0x000000,
			enableShadows: false
		});

		expect(getLastRendererOptions()).not.toMatchObject({
			requiredLimits: expect.anything()
		});
	});
});
