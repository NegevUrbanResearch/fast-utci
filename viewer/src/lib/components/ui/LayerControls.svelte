<script lang="ts">
	import {
		layerStore,
		toggleLayer,
		discoveredLayersStore,
	} from "$lib/stores/layerStore";
	import { STANDARD_LAYER_TYPES } from "$lib/types/layerMaterials";
	import { LAYER_MATERIALS } from "$lib/types/layerMaterials";
	import { viewerStore, setUtciVisible } from "$lib/stores/viewerStore";

	export let placement: "header" | "sidebar" = "header";

	// Layers to hide from UI (we don't care about them currently)
	const HIDDEN_LAYERS = ["base", "sidewalk", "unknown", "ignored"];

	$: onlyBaseLayer =
		$discoveredLayersStore.length === 1 &&
		$discoveredLayersStore[0] === "base";

	function shouldHideLayer(layerId: string) {
		if (layerId === "base" && onlyBaseLayer) return false;
		return HIDDEN_LAYERS.includes(layerId);
	}

	function handleToggle(layerId: string) {
		toggleLayer(layerId);
	}

	function handleUtciToggle() {
		setUtciVisible(!$viewerStore.utciVisible);
	}
</script>

<div class="layer-controls" data-placement={placement}>
	{#each STANDARD_LAYER_TYPES as layer}
		{#if $discoveredLayersStore.includes(layer.id) && !shouldHideLayer(layer.id)}
			<button
				type="button"
				class="layer-button"
				class:active={$layerStore[layer.id] ?? layer.defaultVisible}
				on:click={() => handleToggle(layer.id)}
				aria-label="Toggle {layer.displayName} layer visibility"
				aria-pressed={$layerStore[layer.id] ?? layer.defaultVisible}
			>
				<div
					class="layer-color"
					style="background-color: {LAYER_MATERIALS[layer.id]
						?.color || '#95a5a6'};"
				></div>
				<span class="layer-label">{layer.displayName}</span>
			</button>
		{/if}
	{/each}

	<!-- Show any discovered layers that aren't in the standard list and aren't hidden -->
	{#each $discoveredLayersStore as layerId}
		{#if !STANDARD_LAYER_TYPES.some((l) => l.id === layerId) && !shouldHideLayer(layerId)}
			<button
				type="button"
				class="layer-button"
				class:active={$layerStore[layerId] ?? true}
				on:click={() => handleToggle(layerId)}
				aria-label="Toggle {layerId} layer visibility"
				aria-pressed={$layerStore[layerId] ?? true}
			>
				<div
					class="layer-color"
					style="background-color: {LAYER_MATERIALS[layerId]?.color ||
						'#95a5a6'};"
				></div>
				<span class="layer-label"
					>{layerId.charAt(0).toUpperCase() + layerId.slice(1)}</span
				>
			</button>
		{/if}
	{/each}

	<!-- UTCI/Shading Index Data Layer -->
	<button
		type="button"
		class="layer-button"
		class:active={$viewerStore.utciVisible}
		on:click={handleUtciToggle}
		aria-label="Toggle {$viewerStore.metricType === 'shading_index'
			? 'Shading Index'
			: 'UTCI'} data layer visibility"
		aria-pressed={$viewerStore.utciVisible}
	>
		<div class="layer-color utci-gradient"></div>
		<span class="layer-label">
			{$viewerStore.metricType === "shading_index"
				? "Shading Index Data"
				: "UTCI Data"}
		</span>
	</button>
</div>

<style>
	.layer-controls {
		display: flex;
		flex-direction: row;
		align-items: center;
		gap: 6px;
		font-family: var(--font-family);
		min-width: 0;
	}

	.layer-button {
		display: flex;
		align-items: center;
		justify-content: flex-start;
		gap: 6px;
		height: 26px;
		min-width: 100px;
		padding: 4px 10px;
		cursor: pointer;
		border-radius: var(--radius-control);
		border: 1px solid var(--color-border-subtle);
		background: transparent;
		transition:
			background-color 0.15s ease,
			border-color 0.15s ease,
			opacity 0.15s ease,
			transform 0.08s ease;
		opacity: 0.7;
		font-size: var(--font-xxs);
		font-weight: 500;
		color: var(--color-text-secondary);
		user-select: none;
		-webkit-user-select: none;
		-moz-user-select: none;
		-ms-user-select: none;
		-ms-user-select: none;
		white-space: nowrap;
		font-family: var(--font-family);
	}

	.layer-button.active {
		opacity: 1;
		background-color: var(--color-accent-soft);
		border-color: rgba(56, 189, 248, 0.8);
		border-width: 1.5px;
		color: var(--color-text-primary);
	}

	.layer-button:hover {
		background-color: rgba(148, 163, 184, 0.12);
		border-color: var(--color-border-strong);
		transform: scale(1.02);
	}

	.layer-button.active:hover {
		background-color: var(--color-accent-soft);
		border-color: var(--color-accent);
	}

	.layer-button:active {
		transform: scale(0.98);
	}

	.layer-button:focus {
		outline: 2px solid var(--color-accent);
		outline-offset: 2px;
	}

	.layer-button:focus:not(:focus-visible) {
		outline: none;
	}

	.layer-color {
		width: 10px;
		height: 10px;
		border-radius: 999px;
		flex-shrink: 0;
		border: 1px solid rgba(15, 23, 42, 0.3);
		transition: opacity 0.15s ease;
	}

	.layer-button.active .layer-color {
		opacity: 1;
	}

	.layer-button:not(.active) .layer-color {
		opacity: 0.6;
	}

	.layer-label {
		line-height: 1.2;
		flex-shrink: 0;
	}

	.layer-controls[data-placement="sidebar"] {
		flex-direction: column;
		align-items: stretch;
		gap: 8px;
	}

	.layer-controls[data-placement="sidebar"] .layer-button {
		width: 100%;
		min-width: 0;
		height: 32px;
		padding: 8px 12px;
		font-size: var(--font-xs);
	}

	.layer-controls[data-placement="sidebar"] .layer-button:hover {
		transform: none;
	}

	.layer-controls[data-placement="sidebar"] .layer-button:active {
		transform: translateY(1px);
	}

	.layer-controls[data-placement="sidebar"] .layer-label {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.utci-gradient {
		background: linear-gradient(
			to bottom,
			#d73027 0%,
			#fc8d59 20%,
			#fee08b 40%,
			#d9ef8b 60%,
			#91cf60 80%,
			#1a9850 100%
		);
	}
</style>
