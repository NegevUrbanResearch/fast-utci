<script lang="ts">
	import { T } from '@threlte/core';
	import * as THREE from 'three';

	export let ambientIntensity: number = 1.2;
	export let directionalIntensity: number = 0.6;
	export let directionalPosition: [number, number, number] = [100, 200, 100];

	let dirLightRef: THREE.DirectionalLight | undefined;

	// Set shadow camera properties
	$: if (dirLightRef) {
		dirLightRef.shadow.camera.left = -500;
		dirLightRef.shadow.camera.right = 500;
		dirLightRef.shadow.camera.top = 500;
		dirLightRef.shadow.camera.bottom = -500;
		dirLightRef.shadow.mapSize.width = 512;
		dirLightRef.shadow.mapSize.height = 512;
	}
</script>

<T.AmbientLight color={0xffffff} intensity={ambientIntensity} />
<T.DirectionalLight
	bind:ref={dirLightRef}
	color={0xffffff}
	intensity={directionalIntensity}
	position={directionalPosition}
	castShadow
/>

