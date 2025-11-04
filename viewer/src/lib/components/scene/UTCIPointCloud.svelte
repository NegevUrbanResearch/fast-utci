<script lang="ts">
	import { T } from '@threlte/core';
	import * as THREE from 'three';
	import type { Analysis } from '$lib/types/analysis';
	import { createPointCloudGeometry, updatePointCloudColors } from '$lib/services/pointCloudService';
	import { applyCoordinateTransform } from '$lib/utils/coordinates';
	import { viewerStore } from '$lib/stores/viewerStore';
	import type { Group, Points as ThreePoints } from 'three';

	export let analysis: Analysis | null = null;
	// Model is used for coordinate system matching in reactive statement below
	export let model: Group | null = null;

	let geometry: THREE.BufferGeometry | null = null;
	let material: THREE.PointsMaterial | null = null;
	let pointsRef: ThreePoints | undefined = undefined;

	// Create geometry and material when analysis loads
	$: if (analysis) {
		const currentHour = $viewerStore?.currentHour ?? 0;
		const colorMode = $viewerStore?.colorMode ?? 'normalized';
		const result = createPointCloudGeometry(analysis, currentHour, colorMode);
		geometry = result.geometry;
		material = result.material;
	}

	// Update colors when hour or color mode changes
	$: if (pointsRef && analysis && $viewerStore) {
		const currentHour = $viewerStore.currentHour;
		const colorMode = $viewerStore.colorMode;
		updatePointCloudColors(pointsRef, analysis, currentHour, colorMode);
	}

	// Apply coordinate transformation and tiny up-offset when model or analysis changes
	$: if (pointsRef && analysis && model) {
		const coordinateSystem = analysis.metadata.coordinate_system || 'xy_ground';
		applyCoordinateTransform(pointsRef, coordinateSystem);
		// Apply tiny world-up offset post-transform to avoid z-fighting with ground
		pointsRef.position.y = 0.05;
	}
</script>

{#if analysis && geometry && material && $viewerStore.utciVisible}
	<T.Points
		bind:ref={pointsRef}
		{geometry}
		{material}
		renderOrder={2}
	/>
{/if}

