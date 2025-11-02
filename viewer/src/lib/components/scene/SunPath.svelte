<script lang="ts">
	import { T } from '@threlte/core';
	import * as THREE from 'three';
	import { onDestroy } from 'svelte';
	import type { AnalysisMetadata } from '$lib/types/analysis';
	import { calculateModelCenter, calculateModelSize } from '$lib/utils/bounds';
	import { viewerStore } from '$lib/stores/viewerStore';
	import type { Group as ThreeGroup } from 'three';

	export let metadata: AnalysisMetadata | undefined = undefined;
	export let model: ThreeGroup | null = null;

	let sunPathGroup: ThreeGroup | null = null;
	let arcGeometry: THREE.TubeGeometry | null = null;
	let arcMaterial: THREE.MeshBasicMaterial | null = null;
	let markerPositions: THREE.Vector3[] = [];
	let markerData: Array<{ hour: number; isUp: boolean; color: number; opacity: number }> = [];
	let currentHourHighlight: number = -1;

	// Create/recreate sun path when model or metadata changes
	// Also create when visibility is toggled on
	$: if (model && metadata?.sun_positions && $viewerStore.sunPathVisible) {
		createSunPath();
	}

	// Update current sun indicator when hour changes
	$: if (sunPathGroup && metadata?.sun_positions && $viewerStore.sunPathVisible && currentHourHighlight !== $viewerStore.currentHour) {
		currentHourHighlight = $viewerStore.currentHour;
		updateCurrentSunIndicator($viewerStore.currentHour);
	}

	// Clean up when sun path is hidden or model/metadata becomes unavailable
	$: if ((!$viewerStore.sunPathVisible || !model || !metadata?.sun_positions) && sunPathGroup) {
		disposeSunPath();
	}

	function createSunPath() {
		if (!model || !metadata?.sun_positions) return;

		// Dispose old sun path if it exists
		disposeSunPath();

		const center = calculateModelCenter(model);
		const size = calculateModelSize(model);
		const scale = size.length() * 0.6; // Sun path 0.6x model size

		// Create points for sun path
		const points = createSunPathPoints(metadata, center, scale);
		if (points.length === 0) return;

		// Create curve
		const curve = new THREE.CatmullRomCurve3(points, true);

		// Create group
		sunPathGroup = new THREE.Group();
		sunPathGroup.name = 'SunPath';
		sunPathGroup.visible = $viewerStore.sunPathVisible;

		// Create arc geometry and material
		arcGeometry = new THREE.TubeGeometry(curve, 64, 2, 8, false);
		arcMaterial = new THREE.MeshBasicMaterial({
			color: 0xffaa00,
			transparent: true,
			opacity: 0.6
		});

		// Create marker data arrays
		markerPositions = [];
		markerData = [];

		metadata.sun_positions.forEach((sun, hour) => {
			markerPositions.push(points[hour]);
			markerData.push({
				hour,
				isUp: sun.is_up,
				color: sun.is_up ? 0xffdd00 : 0x666666,
				opacity: sun.is_up ? 0.9 : 0.4
			});
		});

		currentHourHighlight = $viewerStore.currentHour;
		updateCurrentSunIndicator($viewerStore.currentHour);

		console.log(`[SUNPATH] Created sun path with ${markerPositions.length} hour markers`);
	}

	function createSunPathPoints(
		metadata: AnalysisMetadata,
		modelCenter: THREE.Vector3,
		scale: number
	): THREE.Vector3[] {
		if (!metadata.sun_positions || metadata.sun_positions.length === 0) {
			return [];
		}

		const sunPositions = metadata.sun_positions;
		const result: THREE.Vector3[] = [];

		// Convert sun positions to 3D coordinates
		sunPositions.forEach((sun) => {
			if (!sun.is_up) {
				// Below horizon - calculate position from altitude/azimuth
				const alt = Math.max(sun.altitude, -90) * Math.PI / 180;
				const azi = sun.azimuth * Math.PI / 180;

				const x = Math.sin(azi) * Math.cos(alt) * scale;
				const y = Math.sin(alt) * scale;
				const z = Math.cos(azi) * Math.cos(alt) * scale;

				result.push(new THREE.Vector3(x, y, z).add(modelCenter));
			} else {
				// Use pre-calculated vector from Python, scaled and transformed
				// Python: X=East, Y=North, Z=Up
				// Three.js: X=East, Y=Up, Z=South
				result.push(new THREE.Vector3(
					sun.vector[0] * scale,
					sun.vector[2] * scale,
					-sun.vector[1] * scale
				).add(modelCenter));
			}
		});

		return result;
	}

	function updateCurrentSunIndicator(currentHour: number) {
		if (!metadata?.sun_positions || markerData.length === 0) return;

		// Update marker data to highlight current hour
		metadata.sun_positions.forEach((sun, hour) => {
			if (hour < markerData.length) {
				if (hour === currentHour) {
					// Highlight current hour
					markerData[hour] = {
						hour,
						isUp: sun.is_up,
						color: 0xff3300, // Bright red-orange
						opacity: 1.0 // Full opacity
					};
				} else {
					// Reset to default
					markerData[hour] = {
						hour,
						isUp: sun.is_up,
						color: sun.is_up ? 0xffdd00 : 0x666666,
						opacity: sun.is_up ? 0.9 : 0.4
					};
				}
			}
		});

		// Force reactivity by reassigning
		markerData = [...markerData];
	}

	function disposeSunPath() {
		// Dispose geometries and materials
		if (arcGeometry) {
			arcGeometry.dispose();
			arcGeometry = null;
		}
		if (arcMaterial) {
			arcMaterial.dispose();
			arcMaterial = null;
		}
		
		markerPositions = [];
		markerData = [];
		
		sunPathGroup = null;
		currentHourHighlight = -1;
	}

	onDestroy(() => {
		disposeSunPath();
	});
</script>

{#if sunPathGroup && $viewerStore.sunPathVisible && arcGeometry && arcMaterial && markerPositions.length > 0}
	<T.Group bind:ref={sunPathGroup} visible={$viewerStore.sunPathVisible}>
		<!-- Sun path arc -->
		<T.Mesh geometry={arcGeometry} material={arcMaterial} />

		<!-- Hour markers -->
		{#each markerData as markerDataItem, hour}
			<T.Mesh
				position={markerPositions[hour]}
				name="sun_marker_{hour}"
				scale={hour === $viewerStore.currentHour ? [4, 4, 4] : [1, 1, 1]}
			>
				<T.SphereGeometry args={[4, 16, 16]} />
				<T.MeshBasicMaterial
					color={markerDataItem.color}
					transparent={true}
					opacity={markerDataItem.opacity}
				/>
			</T.Mesh>
		{/each}
	</T.Group>
{/if}
