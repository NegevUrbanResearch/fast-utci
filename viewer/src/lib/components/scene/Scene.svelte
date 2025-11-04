<script lang="ts">
	import { Canvas } from '@threlte/core';
	import * as THREE from 'three';
	import SceneBackground from './SceneBackground.svelte';
	import SceneInvalidateSetup from './SceneInvalidateSetup.svelte';
	import type { WebGLRenderer } from 'three';

	export let backgroundColor: number = 0xd3d3d3;
	export let enableShadows: boolean = true;

	function createRenderer(canvas: HTMLCanvasElement): WebGLRenderer {
		const renderer = new THREE.WebGLRenderer({
			canvas,
			antialias: true,
			logarithmicDepthBuffer: true
		});
		renderer.toneMapping = THREE.NoToneMapping;
		renderer.toneMappingExposure = 1.0;
		return renderer;
	}
</script>

<Canvas
	{createRenderer}
	toneMapping={THREE.NoToneMapping}
	shadows={enableShadows ? THREE.BasicShadowMap : false}
>
	<SceneBackground {backgroundColor} />
	<SceneInvalidateSetup />
	<slot />
</Canvas>

