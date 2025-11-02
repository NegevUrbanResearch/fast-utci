<script lang="ts">
	/**
	 * SceneInvalidateSetup Component
	 * 
	 * Sets up the invalidate callback for layer manager service.
	 * This allows services to trigger Threlte re-renders when needed.
	 * Must be inside a Threlte <Canvas> component.
	 */
	import { onDestroy } from 'svelte';
	import { useThrelte } from '@threlte/core';
	import { setInvalidateCallback } from '$lib/services/layerManagerService';

	// Get Threlte's invalidate function
	const { invalidate } = useThrelte();

	// Pass it to the layer manager service
	setInvalidateCallback(invalidate);

	// Clean up on destroy
	onDestroy(() => {
		setInvalidateCallback(null);
	});
</script>

<!-- This component has no visible output, it just sets up the invalidate callback -->

