<script lang="ts">
	import { useScene } from '@threlte/core';
	import * as THREE from 'three';
	import { onMount } from 'svelte';

	export let backgroundColor: number = 0xacd8eb;

	onMount(() => {
		const { scene } = useScene();
		if (scene) {
			scene.background = new THREE.Color(backgroundColor);
		}
	});

	$: {
		const { scene } = useScene();
		if (scene) {
			scene.background = new THREE.Color(backgroundColor);
		}
	}
</script>


