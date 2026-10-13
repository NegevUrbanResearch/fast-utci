<script lang="ts">
 import { onMount } from 'svelte';
 import type { ThrelteGltf } from '@threlte/extras';
 import { modelLoadHandlers } from './modelLoadHarness';
 export let url: string;
 export let onload: (gltf: ThrelteGltf) => void;
 onMount(() => { modelLoadHandlers.push({ url, onload }); });
</script>
