<script lang="ts">
	import { Grid } from '@threlte/extras';
	import * as THREE from 'three';
	import { calculateModelBounds } from '$lib/utils/bounds';
	import type { Group } from 'three';

	export let model: Group | null = null;
	export let visible: boolean = false;

	let gridSize: number = 100;
	let divisions: number = 20;
	let position: [number, number, number] = [0, 0, 0];

	$: if (model) {
		const bounds = calculateModelBounds(model);
		const groundLevel = bounds.min.y;
		const size = bounds.getSize(new THREE.Vector3());
		const maxDim = Math.max(size.x, size.z);
		gridSize = Math.ceil(maxDim * 1.2 / 100) * 100;
		divisions = Math.min(50, Math.max(20, Math.floor(gridSize / 50)));
		
		const center = bounds.getCenter(new THREE.Vector3());
		position = [center.x, groundLevel, center.z];
	}
</script>

{#if visible}
	<Grid
		cellSize={gridSize}
		cellThickness={1}
		cellColor={0x888888}
		sectionSize={gridSize / divisions}
		sectionThickness={2}
		sectionColor={0x444444}
		fadeDistance={0}
		fadeStrength={0}
		position={position}
	/>
{/if}

