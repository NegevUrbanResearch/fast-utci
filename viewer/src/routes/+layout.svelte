<script lang="ts">
	import favicon from '$lib/assets/nur_negev_urban_research_logo.webp';

	let { children } = $props();
</script>

<svelte:head>
	<title>NUR UTCI</title>
	<link rel="icon" href={favicon} />
</svelte:head>

{@render children()}
