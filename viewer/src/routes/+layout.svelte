<script lang="ts">
	import { onMount } from 'svelte';
	import favicon from '$lib/assets/nur_negev_urban_research_logo.webp';
	import { viewerStore, setTheme } from '$lib/stores/viewerStore';

	let { children } = $props();

	onMount(() => {
		if (typeof window === 'undefined') return;

		const stored = window.localStorage.getItem('fast_utci_theme');
		let next: 'dark' | 'light' = 'dark';

		if (stored === 'dark' || stored === 'light') {
			next = stored;
		} else if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
			next = 'dark';
		}

		setTheme(next);
		document.documentElement.dataset.theme = next;

		const unsubscribe = viewerStore.subscribe((state) => {
			const theme = state.theme ?? 'dark';
			document.documentElement.dataset.theme = theme;
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
