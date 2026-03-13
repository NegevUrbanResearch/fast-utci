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

	async function createRenderer(canvas: HTMLCanvasElement) {
		canvasElement = canvas;
		const renderer = new WebGPURenderer({
			canvas,
			antialias: true
		});
		// WebGPURenderer requires async initialization before first use
		if (typeof renderer.init === 'function') {
			// Threlte supports async createRenderer returning a Promise
			// eslint-disable-next-line @typescript-eslint/await-thenable
			await renderer.init();
		}
		renderer.toneMapping = THREE.NoToneMapping;
		renderer.toneMappingExposure = 1.0;
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

