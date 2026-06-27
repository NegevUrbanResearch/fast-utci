<script lang="ts">
	import type { IconifyIcon } from "@iconify/types";
	import building2Icon from "@iconify-icons/lucide/building-2";
	import circleIcon from "@iconify-icons/lucide/circle";
	import mapIcon from "@iconify-icons/lucide/map";
	import trainTrackIcon from "@iconify-icons/lucide/train-track";
	import treesIcon from "@iconify-icons/lucide/trees";
	import wavesIcon from "@iconify-icons/lucide/waves";
	import {
		layerStore,
		toggleLayer,
		discoveredLayersStore,
	} from "$lib/stores/layerStore";
	import { STANDARD_LAYER_TYPES } from "$lib/types/layerMaterials";
	import { LAYER_MATERIALS } from "$lib/types/layerMaterials";

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

	function getLayerColor(layerId: string): string {
		return LAYER_MATERIALS[layerId]?.color || "#95a5a6";
	}

	function getLayerIconColor(layerId: string): string {
		if (layerId === "road") {
			return LAYER_MATERIALS.road.outlineColor || getLayerColor(layerId);
		}
		if (layerId === "train_track") {
			return "#cbd5e1";
		}
		return getLayerColor(layerId);
	}

	function getLayerIcon(layerId: string): IconifyIcon {
		switch (layerId) {
			case "building":
			case "new_building":
				return building2Icon;
			case "vegetation":
			case "new_vegetation":
				return treesIcon;
			case "road":
				return mapIcon;
			case "train_track":
				return trainTrackIcon;
			case "water":
				return wavesIcon;
			default:
				return circleIcon;
		}
	}
</script>

<div class="layer-controls" data-placement={placement}>
	{#each STANDARD_LAYER_TYPES as layer}
		{#if $discoveredLayersStore.includes(layer.id) && !shouldHideLayer(layer.id)}
			{@const layerColor = getLayerIconColor(layer.id)}
			{@const layerIcon = getLayerIcon(layer.id)}
			<button
				type="button"
				class="layer-button"
				class:active={$layerStore[layer.id] ?? layer.defaultVisible}
				on:click={() => handleToggle(layer.id)}
				aria-label="Toggle {layer.displayName} layer visibility"
				aria-pressed={$layerStore[layer.id] ?? layer.defaultVisible}
			>
				<span class="layer-icon" style="color: {layerColor};">
					<svg
						viewBox="0 0 {layerIcon.width ?? 24} {layerIcon.height ?? 24}"
						width="15"
						height="15"
						aria-hidden="true"
					>
						{@html layerIcon.body}
					</svg>
				</span>
				<span class="layer-label">{layer.displayName}</span>
			</button>
		{/if}
	{/each}

	<!-- Show any discovered layers that aren't in the standard list and aren't hidden -->
	{#each $discoveredLayersStore as layerId}
		{#if !STANDARD_LAYER_TYPES.some((l) => l.id === layerId) && !shouldHideLayer(layerId)}
			{@const layerColor = getLayerIconColor(layerId)}
			{@const layerIcon = getLayerIcon(layerId)}
			<button
				type="button"
				class="layer-button"
				class:active={$layerStore[layerId] ?? true}
				on:click={() => handleToggle(layerId)}
				aria-label="Toggle {layerId} layer visibility"
				aria-pressed={$layerStore[layerId] ?? true}
			>
				<span class="layer-icon" style="color: {layerColor};">
					<svg
						viewBox="0 0 {layerIcon.width ?? 24} {layerIcon.height ?? 24}"
						width="15"
						height="15"
						aria-hidden="true"
					>
						{@html layerIcon.body}
					</svg>
				</span>
				<span class="layer-label"
					>{layerId.charAt(0).toUpperCase() + layerId.slice(1)}</span
				>
			</button>
		{/if}
	{/each}
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

	.layer-icon {
		width: 20px;
		height: 20px;
		border-radius: 5px;
		flex-shrink: 0;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		background: color-mix(in srgb, currentColor 24%, rgba(15, 23, 42, 0.52));
		border: 1px solid color-mix(in srgb, currentColor 72%, transparent);
		box-shadow:
			inset 0 0 0 1px rgba(255, 255, 255, 0.1),
			0 0 0 1px rgba(2, 6, 23, 0.18);
		transition: opacity 0.15s ease;
	}

	.layer-icon svg {
		filter: drop-shadow(0 1px 1px rgba(2, 6, 23, 0.45));
	}

	.layer-button.active .layer-icon {
		opacity: 1;
	}

	.layer-button:not(.active) .layer-icon {
		opacity: 0.55;
	}

	.layer-label {
		line-height: 1.2;
		flex-shrink: 0;
	}

	.layer-controls[data-placement="sidebar"] {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		align-items: stretch;
		gap: 8px;
	}

	.layer-controls[data-placement="sidebar"] .layer-button {
		width: 100%;
		min-width: 0;
		min-height: 34px;
		height: auto;
		padding: 8px 12px;
		font-size: var(--font-xs);
		white-space: normal;
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
		overflow-wrap: anywhere;
	}
</style>
