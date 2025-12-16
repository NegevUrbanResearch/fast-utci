<script lang="ts">
	import { T } from '@threlte/core';
	import { OrbitControls } from '@threlte/extras';
	import { cameraStore, setCameraPosition, setCameraTarget } from '$lib/stores/cameraStore';
	import * as THREE from 'three';
	import { onMount, onDestroy } from 'svelte';

	export let fov: number = 60;
	export let near: number = 0.1;
	export let far: number = 5000;

	// Expose camera ref for tooltip raycasting
	export let cameraRef: THREE.PerspectiveCamera | undefined;
	let controlsRef: any = undefined; // OrbitControls from Threlte
	let lastStoreUpdate = 0;
	const UPDATE_THROTTLE_MS = 100; // Throttle store updates to avoid excessive writes
	let animationFrameId: number | null = null;

	// Sync camera with store (one-way: store -> camera)
	// Only sync if store was updated externally (not from our own updates)
	$: if (cameraRef && $cameraStore) {
		const now = Date.now();
		// Only apply store updates if they're recent (within throttle window)
		// This prevents applying our own updates back to the camera
		if (now - lastStoreUpdate > UPDATE_THROTTLE_MS) {
			cameraRef.position.copy($cameraStore.position);
			cameraRef.lookAt($cameraStore.target);
		}
	}

	// Sync OrbitControls target with store
	$: if (controlsRef && $cameraStore) {
		const now = Date.now();
		if (now - lastStoreUpdate > UPDATE_THROTTLE_MS) {
			controlsRef.target.copy($cameraStore.target);
		}
	}

	// Sync OrbitControls changes back to store
	// Use throttled polling to avoid performance issues
	let lastSyncTime = 0;
	const SYNC_INTERVAL_MS = 100; // Sync every 100ms instead of every frame

	function syncControlsToStore() {
		if (!cameraRef || !controlsRef) return;

		const now = Date.now();
		if (now - lastSyncTime < SYNC_INTERVAL_MS) {
			return; // Throttle updates
		}
		lastSyncTime = now;

		const currentPos = cameraRef.position.clone();
		const currentTarget = controlsRef.target.clone();
		const storePos = $cameraStore.position;
		const storeTarget = $cameraStore.target;
		
		// Only update if there's a significant difference (avoid unnecessary updates)
		const posDiff = currentPos.distanceTo(storePos);
		const targetDiff = currentTarget.distanceTo(storeTarget);
		
		if (posDiff > 0.01 || targetDiff > 0.01) {
			lastStoreUpdate = Date.now();
			setCameraPosition(currentPos);
			setCameraTarget(currentTarget);
		}
	}

	function animationLoop() {
		syncControlsToStore();
		animationFrameId = requestAnimationFrame(animationLoop);
	}

	onMount(() => {
		animationFrameId = requestAnimationFrame(animationLoop);
	});

	onDestroy(() => {
		if (animationFrameId !== null) {
			cancelAnimationFrame(animationFrameId);
		}
	});
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
		bind:ref={controlsRef}
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
