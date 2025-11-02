<script lang="ts">
	import { T } from '@threlte/core';
	import { OrbitControls } from '@threlte/extras';
	import { cameraStore } from '$lib/stores/cameraStore';
	import * as THREE from 'three';

	export let fov: number = 60;
	export let near: number = 0.1;
	export let far: number = 5000;

	let cameraRef: THREE.PerspectiveCamera | undefined;

	// Sync camera with store
	$: if (cameraRef && $cameraStore) {
		cameraRef.position.copy($cameraStore.position);
		cameraRef.lookAt($cameraStore.target);
	}
</script>

<T.PerspectiveCamera
	bind:ref={cameraRef}
	{fov}
	{near}
	{far}
	position={[$cameraStore.position.x, $cameraStore.position.y, $cameraStore.position.z]}
	makeDefault
>
	<OrbitControls
		target={[$cameraStore.target.x, $cameraStore.target.y, $cameraStore.target.z]}
		enableDamping={true}
		dampingFactor={0.05}
		minDistance={$cameraStore.minDistance}
		maxDistance={$cameraStore.maxDistance}
		zoomSpeed={$cameraStore.zoomSpeed}
		screenSpacePanning={false}
		maxPolarAngle={Math.PI / 2}
	/>
</T.PerspectiveCamera>

