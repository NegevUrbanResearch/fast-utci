<script lang="ts">
	import { layerStore, toggleLayer, discoveredLayersStore } from '$lib/stores/layerStore';
	import { STANDARD_LAYER_TYPES } from '$lib/types/layerMaterials';
	import { LAYER_MATERIALS } from '$lib/types/layerMaterials';
	import { viewerStore, setUtciVisible } from '$lib/stores/viewerStore';

	// Layers to hide from UI (we don't care about them currently)
	const HIDDEN_LAYERS = ['base', 'road', 'sidewalk', 'unknown'];

	function handleToggle(layerId: string) {
		toggleLayer(layerId);
	}

	function handleUtciToggle() {
		setUtciVisible(!$viewerStore.utciVisible);
	}
</script>

<div class="layer-controls">
	<div class="layer-header">Model Layers</div>
	{#each STANDARD_LAYER_TYPES as layer}
		{#if $discoveredLayersStore.includes(layer.id) && !HIDDEN_LAYERS.includes(layer.id)}
			<div
				class="layer-item"
				class:active={$layerStore[layer.id] ?? layer.defaultVisible}
				role="button"
				tabindex="0"
				on:click={() => handleToggle(layer.id)}
				on:keydown={(e) => {
					if (e.key === 'Enter' || e.key === ' ') {
						e.preventDefault();
						handleToggle(layer.id);
					}
				}}
			>
				<div class="layer-color" style="background-color: {LAYER_MATERIALS[layer.id]?.color || '#95a5a6'};"></div>
				<div class="layer-label">{layer.displayName}</div>
			</div>
		{/if}
	{/each}
	
	<!-- Show any discovered layers that aren't in the standard list and aren't hidden -->
	{#each $discoveredLayersStore as layerId}
		{#if !STANDARD_LAYER_TYPES.some(l => l.id === layerId) && !HIDDEN_LAYERS.includes(layerId)}
			<div
				class="layer-item"
				class:active={$layerStore[layerId] ?? true}
				role="button"
				tabindex="0"
				on:click={() => handleToggle(layerId)}
				on:keydown={(e) => {
					if (e.key === 'Enter' || e.key === ' ') {
						e.preventDefault();
						handleToggle(layerId);
					}
				}}
			>
				<div class="layer-color" style="background-color: {LAYER_MATERIALS[layerId]?.color || '#95a5a6'};"></div>
				<div class="layer-label">{layerId.charAt(0).toUpperCase() + layerId.slice(1)}</div>
			</div>
		{/if}
	{/each}
	
	<!-- UTCI Data Layer -->
	<div
		class="layer-item"
		class:active={$viewerStore.utciVisible}
		role="button"
		tabindex="0"
		on:click={handleUtciToggle}
		on:keydown={(e) => {
			if (e.key === 'Enter' || e.key === ' ') {
				e.preventDefault();
				handleUtciToggle();
			}
		}}
	>
		<div class="layer-color utci-gradient"></div>
		<div class="layer-label">UTCI Data</div>
	</div>
</div>

<style>
	.layer-controls {
		position: absolute;
		top: 120px;
		left: 20px;
		background: rgba(255, 255, 255, 0.95);
		padding: 10px;
		border-radius: 4px;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
		z-index: 1000;
		min-width: 180px;
		font-family: Arial, sans-serif;
		font-size: 13px;
	}

	.layer-header {
		font-weight: bold;
		margin-bottom: 10px;
		font-size: 13px;
		color: #333;
		padding-bottom: 5px;
		border-bottom: 1px solid #ddd;
	}

	.layer-item {
		display: flex;
		align-items: center;
		padding: 6px 8px;
		cursor: pointer;
		border-radius: 3px;
		margin-bottom: 2px;
		transition: background-color 0.2s;
		opacity: 0.5;
		text-decoration: line-through;
	}

	.layer-item.active {
		opacity: 1;
		text-decoration: none;
	}

	.layer-item:hover {
		background-color: rgba(0, 0, 0, 0.05);
	}

	.layer-item:focus {
		outline: 2px solid #3498db;
		outline-offset: 2px;
	}

	.layer-color {
		width: 16px;
		height: 16px;
		border-radius: 2px;
		margin-right: 8px;
		border: 1px solid rgba(0, 0, 0, 0.2);
		flex-shrink: 0;
	}

	.layer-label {
		flex: 1;
		color: #333;
		user-select: none;
	}

	.utci-gradient {
		background: linear-gradient(to bottom, 
			#d73027 0%, 
			#fc8d59 20%, 
			#fee08b 40%, 
			#d9ef8b 60%, 
			#91cf60 80%, 
			#1a9850 100%
		);
	}
</style>


