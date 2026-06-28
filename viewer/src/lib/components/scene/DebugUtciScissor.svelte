<script lang="ts">
	/**
	 * DebugUtciScissor Component
	 *
	 * ABOUTME: Minimal scissor-based renderer that compares two UTCI meshes
	 * (left: .bin-backed, right: live-computed) within a single Three.js scene.
	 * It reuses the existing curtainPosition store and follows the same
	 * WebGPU-safe scissor strategy as ComparisonRenderer.
	 */
	import { onMount, onDestroy } from 'svelte';
	import { useThrelte, useTask } from '@threlte/core';
	import * as THREE from 'three';
	import type { Mesh, PerspectiveCamera } from 'three';
	import { curtainPosition, comparisonStore } from '$lib/stores/comparisonStore';
	import { viewerStore } from '$lib/stores/viewerStore';
	import { get } from 'svelte/store';

	// Props
	export let baseCamera: PerspectiveCamera | undefined = undefined;
	export let binUtciMesh: Mesh | null = null;
	export let liveUtciMesh: Mesh | null = null;

	const { renderer, scene, autoRender, renderStage } = useThrelte();

	/**
	 * Render function that performs two passes with scissor rectangles:
	 * - Left pass: .bin UTCI mesh visible, live mesh hidden
	 * - Right pass: live UTCI mesh visible, .bin mesh hidden
	 */
	function renderDebugComparison() {
		if (!renderer || !scene) return;
		const camera = baseCamera;
		if (!camera) return;
		if (!binUtciMesh || !liveUtciMesh) return;

		// Respect global UTCI visibility flag – if disabled, skip rendering overlays.
		const viewerState = get(viewerStore);
		const utciVisible = viewerState.utciVisible ?? true;
		if (!utciVisible) {
			return;
		}

		// Use renderer size rather than DOM element size for WebGPU safety.
		const size = new THREE.Vector2();
		renderer.getSize(size);
		const width = size.x;
		const height = size.y;
		if (width === 0 || height === 0) return;

		const curtain = get(curtainPosition);
		const curtainX = Math.floor(width * curtain);

		// WebGPURenderer does not expose getScissorTest / setScissorTest; guard calls
		// so this remains compatible with both WebGLRenderer and WebGPURenderer.
		const canToggleScissorTest =
			typeof (renderer as any).getScissorTest === 'function' &&
			typeof (renderer as any).setScissorTest === 'function';
		const originalScissorTest = canToggleScissorTest
			? (renderer as any).getScissorTest()
			: undefined;

		if (canToggleScissorTest) {
			(renderer as any).setScissorTest(true);
		}

		// Clear entire canvas first.
		renderer.setViewport(0, 0, width, height);
		renderer.setScissor(0, 0, width, height);
		renderer.clear();

		// Render LEFT side: .bin UTCI visible, live hidden.
		if (curtainX > 0) {
			binUtciMesh.visible = true;
			liveUtciMesh.visible = false;

			renderer.setViewport(0, 0, width, height);
			renderer.setScissor(0, 0, curtainX, height);
			renderer.render(scene, camera);
		}

		// Render RIGHT side: live UTCI visible, .bin hidden.
		if (curtainX < width) {
			binUtciMesh.visible = false;
			liveUtciMesh.visible = true;

			renderer.setViewport(0, 0, width, height);
			renderer.setScissor(curtainX, 0, width - curtainX, height);
			renderer.render(scene, camera);
		}

		// Restore scissor test state if supported.
		if (canToggleScissorTest && originalScissorTest !== undefined) {
			(renderer as any).setScissorTest(originalScissorTest);
		}
	}

	// Dedicated render task that replaces Threlte's autoRender while mounted.
	// autoInvalidate=true ensures this runs every frame while active so that
	// curtain movement and other interactions are reflected immediately.
	const { start: startDebugRender, stop: stopDebugRender } = useTask(renderDebugComparison, {
		autoStart: false,
		autoInvalidate: true,
		stage: renderStage
	});

	let renderActive = false;
	let previousAutoRender: boolean | null = null;

	onMount(() => {
		previousAutoRender = autoRender.current;
		autoRender.set(false);
		startDebugRender();
		renderActive = true;

		// Mark comparison mode as active for UI elements like the curtain.
		comparisonStore.update((state) => ({
			...state,
			isComparing: true
		}));
	});

	onDestroy(() => {
		if (renderActive) {
			stopDebugRender();
			renderActive = false;
		}
		if (previousAutoRender !== null) {
			autoRender.set(previousAutoRender);
			previousAutoRender = null;
		}

		// Do not forcibly reset meshes; parent component owns their lifecycle.
	});
</script>

<!--
	This component does not render DOM content itself; it hooks into Threlte's
	render loop to implement scissor-based dual rendering for two UTCI meshes.
-->

