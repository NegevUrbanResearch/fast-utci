<script lang="ts">
	import { Canvas } from '@threlte/core';
	import * as THREE from 'three';
	import { WebGPURenderer } from 'three/webgpu';
	import {
		createLargeBufferRequiredLimits,
		readLargeBufferDeviceLimits,
		type WebgpuLargeBufferDeviceLimits,
		type WebgpuLargeBufferRequiredLimits
	} from '$lib/compute/webgpuDeviceLimits';
	import SceneBackground from './SceneBackground.svelte';
	import SceneInvalidateSetup from './SceneInvalidateSetup.svelte';

	type RendererDiagnostics = {
		rendererBackend: 'webgpu' | 'unknown';
		rendererDevice?: GPUDevice;
		rendererRequiredLimits?: WebgpuLargeBufferRequiredLimits;
		rendererDeviceLimits?: WebgpuLargeBufferDeviceLimits;
		error?: string;
	};

	// Default background; can be overridden by parent for theme-aware colors
	export let backgroundColor: number = 0x4b5563;
	export let enableShadows: boolean = true;
	export let onRendererDiagnostics:
		| ((diagnostics: RendererDiagnostics) => void)
		| undefined = undefined;
	export let requestLargeWebgpuLimits = false;

	let canvasElement: HTMLCanvasElement | null = null;

	function createRenderer(canvas: HTMLCanvasElement) {
		canvasElement = canvas;
		const rendererRequiredLimits = requestLargeWebgpuLimits
			? createLargeBufferRequiredLimits()
			: undefined;
		const renderer = new WebGPURenderer({
			canvas,
			antialias: true,
			alpha: false,
			...(rendererRequiredLimits ? { requiredLimits: rendererRequiredLimits } : {})
		});
		const originalSetSize = renderer.setSize.bind(renderer);
		// Layout/resize observers can briefly emit 0x0 during transitions; sending
		// that through to Dawn/WebGPU produces invalid swapchain/texture warnings,
		// and the renderer still recovers once a later positive size arrives.
		renderer.setSize = ((width: number, height: number, updateStyle?: boolean) => {
			if (width <= 0 || height <= 0) return;
			originalSetSize(width, height, updateStyle);
		}) as typeof renderer.setSize;
		// Guard so dispose() does not throw when this.info is undefined (e.g. if dispose runs before async init() completes).
		const r = renderer as unknown as { info?: { dispose: () => void } };
		if (!r.info) r.info = { dispose: () => {} };

		const backend = (
			renderer as unknown as { backend?: { isWebGPUBackend?: boolean; device?: GPUDevice } }
		).backend;
		onRendererDiagnostics?.({
			rendererBackend: backend?.isWebGPUBackend ? 'webgpu' : 'unknown',
			rendererRequiredLimits
		});

		// WebGPURenderer requires async initialization before first use. We fire
		// and forget here; the renderer will still be usable for subsequent
		// frames once init resolves.
		if (typeof (renderer as any).init === 'function') {
			// eslint-disable-next-line @typescript-eslint/no-floating-promises
			(renderer as any)
				.init()
				.then(() => {
					const initializedBackend = (
						renderer as unknown as { backend?: { isWebGPUBackend?: boolean; device?: GPUDevice } }
					).backend;
					onRendererDiagnostics?.({
						rendererBackend: initializedBackend?.isWebGPUBackend ? 'webgpu' : 'unknown',
						rendererDevice: initializedBackend?.device,
						rendererRequiredLimits,
						rendererDeviceLimits: readLargeBufferDeviceLimits(initializedBackend?.device)
					});
				})
				.catch((error: unknown) => {
					onRendererDiagnostics?.({
						rendererBackend: 'unknown',
						rendererRequiredLimits,
						error: error instanceof Error ? error.message : String(error)
					});
				});
		}
		// Let <Canvas> drive tone mapping via its props.
		return renderer;
	}

	// Expose canvas element
	export { canvasElement };
</script>

<div class="scene-wrapper">
	<Canvas
		{createRenderer}
		toneMapping={THREE.NoToneMapping}
		shadows={enableShadows ? THREE.BasicShadowMap : false}
	>
		<SceneBackground {backgroundColor} />
		<SceneInvalidateSetup />
		<slot />
	</Canvas>
</div>

<style>
	.scene-wrapper {
		width: 100%;
		height: 100%;
		position: relative;
	}
</style>

