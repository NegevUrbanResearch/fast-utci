import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from '@testing-library/svelte/svelte5';
import { getLastRenderer, resetLastRenderer } from '../mocks/threlteCanvasHarness';
import { resetMockWebgpuRenderer, setSizeSpy } from '../mocks/mockWebgpuRenderer';

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
});
