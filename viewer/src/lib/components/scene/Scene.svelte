<script lang="ts">
	import { Canvas } from '@threlte/core';
	import * as THREE from 'three';
	import SceneBackground from './SceneBackground.svelte';
	import SceneInvalidateSetup from './SceneInvalidateSetup.svelte';
	import type { WebGLRenderer } from 'three';
	import { onMount } from 'svelte';

	// Default background; can be overridden by parent for theme-aware colors
	export let backgroundColor: number = 0x4b5563;
	export let enableShadows: boolean = true;

	let canvasElement: HTMLCanvasElement | null = null;

	function createRenderer(canvas: HTMLCanvasElement): WebGLRenderer {
		canvasElement = canvas;
		const renderer = new THREE.WebGLRenderer({
			canvas,
			antialias: true,
			logarithmicDepthBuffer: true
		});
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

