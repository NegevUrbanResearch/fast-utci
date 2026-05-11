<script lang="ts">
	import ComparisonCurtain from "$lib/components/ui/ComparisonCurtain.svelte";

	export let loading: boolean;
	export let error: string | null;
	export let baseLiveError: string | null | undefined;
	export let comparisonLiveError: string | null | undefined;
	export let showMainRouteOverlay: boolean;
	export let showMainRouteComparisonOverlay: boolean;
	export let curtainPosition: number;
	export let modelLoading: boolean;
	export let comparisonModelLoading: boolean;
	export let useLiveUtciOnMainRoute: boolean;
	export let isComparing: boolean;
	export let mainViewportElement: HTMLElement | null;
	export let comparisonScenarioName: string;
</script>

{#if loading}
	<div class="overlay-message">Loading analysis data...</div>
{/if}

{#if error}
	<div class="overlay-message error">Error: {error}</div>
{/if}

{#if baseLiveError}
	<div class="overlay-message error">Live UTCI error: {baseLiveError}</div>
{/if}

{#if comparisonLiveError}
	<div class="overlay-message error comparison-note">
		Scenario live UTCI error: {comparisonLiveError}
	</div>
{/if}

{#if showMainRouteOverlay}
	<div
		class="model-loading-backdrop"
		class:comparison-mode={showMainRouteComparisonOverlay}
		style={showMainRouteComparisonOverlay
			? `--curtain-position: ${curtainPosition}`
			: ""}
		aria-hidden="true"
	></div>
	<div
		class="model-loading-overlay"
		class:comparison-mode={showMainRouteComparisonOverlay}
		style={showMainRouteComparisonOverlay
			? `--curtain-position: ${curtainPosition}`
			: ""}
		aria-live="polite"
	>
		<div class="spinner"></div>
		<div class="loading-text">
			{#if modelLoading || comparisonModelLoading}
				Preparing model...
			{:else if useLiveUtciOnMainRoute}
				Computing live UTCI...
			{:else}
				Loading analysis...
			{/if}
		</div>
	</div>
{/if}

{#if isComparing}
	<ComparisonCurtain
		containerElement={mainViewportElement}
		{comparisonScenarioName}
	/>
{/if}

<style>
	.model-loading-backdrop.comparison-mode {
		left: calc(var(--curtain-position) * 100%);
		right: 0;
	}

	.model-loading-overlay.comparison-mode {
		left: calc(50% + var(--curtain-position) * 50%);
		transform: translate(-50%, -50%);
	}
</style>
