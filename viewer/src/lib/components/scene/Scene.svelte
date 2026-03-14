<script lang="ts">
import { Canvas } from '@threlte/core';
	import * as THREE from 'three';
	import { WebGPURenderer } from 'three/webgpu';
	import SceneBackground from './SceneBackground.svelte';
	import SceneInvalidateSetup from './SceneInvalidateSetup.svelte';
	import { onMount } from 'svelte';

	// Default background; can be overridden by parent for theme-aware colors
	export let backgroundColor: number = 0x4b5563;
	export let enableShadows: boolean = true;

	let canvasElement: HTMLCanvasElement | null = null;

function createRenderer(canvas: HTMLCanvasElement) {
		canvasElement = canvas;
		const renderer = new WebGPURenderer({
			canvas,
			antialias: true,
			alpha: false
		});
		// Guard so dispose() does not throw when this.info is undefined (e.g. if dispose runs before async init() completes).
		const r = renderer as unknown as { info?: { dispose: () => void } };
		if (!r.info) r.info = { dispose: () => {} };
		// WebGPURenderer requires async initialization before first use. We fire
		// and forget here; the renderer will still be usable for subsequent
		// frames once init resolves.
		if (typeof (renderer as any).init === 'function') {
			// eslint-disable-next-line @typescript-eslint/no-floating-promises
			(renderer as any).init();
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

