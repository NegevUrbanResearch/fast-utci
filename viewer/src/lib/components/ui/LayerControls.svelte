<script lang="ts">
	import { layerStore, toggleLayer, discoveredLayersStore } from '$lib/stores/layerStore';
	import { STANDARD_LAYER_TYPES } from '$lib/types/layerMaterials';
	import { LAYER_MATERIALS } from '$lib/types/layerMaterials';
	import { viewerStore, setUtciVisible } from '$lib/stores/viewerStore';
	import type { MetricType } from '$lib/types/viewer';

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
	<div class="layers-caption">Toggle visibility in the 3D model</div>

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
	
	<!-- UTCI/Shading Index Data Layer -->
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
		<div class="layer-label">
			{$viewerStore.metricType === 'shading_index' ? 'Shading Index Data' : 'UTCI Data'}
		</div>
	</div>
</div>

<style>
	.layer-controls {
		font-family: var(--font-family);
		font-size: 13px;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.layer-item {
		display: flex;
		align-items: center;
		padding: 6px 9px;
		cursor: pointer;
		border-radius: var(--radius-control);
		border: 1px solid transparent;
		position: relative;
		padding-right: 40px;
		transition:
			background-color 0.15s ease,
			border-color 0.15s ease,
			color 0.15s ease,
			opacity 0.15s ease,
			transform 0.08s ease;
		opacity: 0.65;
	}

	.layer-item.active {
		opacity: 1;
		background-color: var(--color-accent-soft);
		border-color: var(--color-border-subtle);
	}

	.layer-item:hover {
		background-color: rgba(148, 163, 184, 0.16);
	}

	.layer-item:active {
		transform: translateY(1px);
	}

	.layer-item:focus {
		outline: 2px solid var(--color-accent);
		outline-offset: 2px;
	}

	.layer-item::after {
		content: '';
		position: absolute;
		right: 8px;
		top: 50%;
		transform: translateY(-50%);
		width: 26px;
		height: 14px;
		border-radius: 999px;
		background: rgba(148, 163, 184, 0.45);
		transition: background-color 0.15s ease;
	}

	.layer-item::before {
		content: '';
		position: absolute;
		right: 18px;
		top: 50%;
		transform: translate(0, -50%);
		width: 12px;
		height: 12px;
		border-radius: 999px;
		background: #f9fafb;
		box-shadow: 0 1px 2px rgba(15, 23, 42, 0.5);
		border: 1px solid rgba(15, 23, 42, 0.55);
		transition:
			transform 0.15s ease,
			background-color 0.15s ease;
	}

	.layer-item.active::after {
		background: rgba(56, 189, 248, 0.55);
	}

	.layer-item.active::before {
		transform: translate(10px, -50%);
	}

	.layer-color {
		width: 16px;
		height: 16px;
		border-radius: 999px;
		margin-right: 9px;
		border: 1px solid var(--color-border-subtle);
		flex-shrink: 0;
		box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.6);
	}

	.layer-label {
		flex: 1;
		color: var(--color-text-primary);
		user-select: none;
	}

	.layers-caption {
		font-size: 12px;
		color: var(--color-text-secondary);
		margin-bottom: 2px;
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


