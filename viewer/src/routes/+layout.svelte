<script lang="ts">
	import { onMount } from 'svelte';
	import favicon from '$lib/assets/nur_negev_urban_research_logo.webp';
	import { viewerStore, setTheme } from '$lib/stores/viewerStore';

	let { children } = $props();

	onMount(() => {
		if (typeof window === 'undefined') return;

		// Always set dark mode as default
		const next: 'dark' | 'light' = 'dark';

		setTheme(next);
		document.documentElement.dataset.theme = next;

		const unsubscribe = viewerStore.subscribe((state) => {
			// Always enforce dark theme
			document.documentElement.dataset.theme = 'dark';
		});

		return () => {
			unsubscribe();
		};
	});
</script>

<svelte:head>
	<title>NUR UTCI</title>
	<link rel="icon" href={favicon} />
</svelte:head>

{@render children()}
