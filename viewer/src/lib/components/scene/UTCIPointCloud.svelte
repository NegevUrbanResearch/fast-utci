<script lang="ts">
	import { useThrelte } from '@threlte/core';
	import { onDestroy } from 'svelte';
	import type { Analysis } from '$lib/types/analysis';
	import {
		createUtciSurfaceMesh,
		updateUtciSurfaceTexture
	} from '$lib/services/pointCloudService';
	import { viewerStore } from '$lib/stores/viewerStore';
	import type { Group, Mesh, MeshBasicMaterial } from 'three';

	export let analysis: Analysis | null = null;
	export let model: Group | null = null;

	let utciSurface: Mesh | null = null;
	let lastAnalysis: Analysis | null = null;
	const { scene, invalidate } = useThrelte();

	function disposeUtciSurface() {
		if (!utciSurface) return;

		scene.remove(utciSurface);

		const materials = Array.isArray(utciSurface.material)
			? utciSurface.material
			: [utciSurface.material];

		materials.forEach((mat) => {
			const material = mat as MeshBasicMaterial;
			material.map?.dispose();
			material.dispose();
		});

		utciSurface.geometry.dispose();
		utciSurface = null;
		lastAnalysis = null;
	}

	$: viewerState = $viewerStore;

	$: {
		if (!analysis) {
			disposeUtciSurface();
		} else if (analysis !== lastAnalysis) {
			disposeUtciSurface();
			utciSurface = createUtciSurfaceMesh(
				analysis,
				viewerState?.currentHour ?? 0,
				viewerState?.colorMode ?? 'normalized'
			);
			scene.add(utciSurface);
			lastAnalysis = analysis;
			invalidate();
		} else if (utciSurface && viewerState) {
			updateUtciSurfaceTexture(
				utciSurface,
				analysis,
				viewerState.currentHour,
				viewerState.colorMode
			);
			invalidate();
		}
	}

	$: {
		if (utciSurface) {
			utciSurface.visible = Boolean(analysis && viewerState?.utciVisible);
			if (utciSurface.visible) {
				invalidate();
			}
		}
	}

	onDestroy(() => {
		disposeUtciSurface();
	});
</script>

