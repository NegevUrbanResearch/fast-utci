<script lang="ts">
	import { loadAnalysisData } from '$lib/stores/analysisStore';
	import { cameraStore, focusCameraOnModel } from '$lib/stores/cameraStore';
	import { get } from 'svelte/store';
	import type { Writable } from 'svelte/store';

	let isExpanded = false;
	let selectedCategory = '';
	let selectedVariant = 1;

	const categories = [
		{ value: 'existing_buildings', label: 'Existing Buildings' },
		{ value: 'existing_trees', label: 'Existing Trees' },
		{ value: 'new_high_buildings', label: 'New High Buildings' },
		{ value: 'new_low_buildings', label: 'New Low Buildings' },
		{ value: 'new_trees', label: 'New Trees' }
	];

	function togglePanel() {
		isExpanded = !isExpanded;
	}

	async function handleCategoryChange(event: Event) {
		const target = event.target as HTMLSelectElement;
		selectedCategory = target.value;

		if (selectedCategory) {
			// Auto-load variant 1 when category is selected
			selectedVariant = 1;
			await loadScenario(selectedCategory, 1);
		}
	}

	async function handleVariantChange(event: Event) {
		const target = event.target as HTMLInputElement;
		selectedVariant = parseInt(target.value, 10);
		await loadScenario(selectedCategory, selectedVariant);
	}

	async function loadScenario(category: string, variant: number) {
		try {
			// Preserve current camera state
			const currentCameraState = get(cameraStore);

			// Construct analysis ID: category/category_variant (e.g., "existing_buildings/existing_buildings_01")
			const variantStr = variant.toString().padStart(2, '0');
			const analysisId = `${category}/${category}_${variantStr}`;

			console.log(`[SCENARIO] Loading: ${analysisId}`);

			// Load analysis data (this will update the analysisStore)
			// dataDir defaults to base path + "/data/analyses" for GitHub Pages compatibility
			await loadAnalysisData(analysisId);

			// Preserve camera position after model loads (model loaded event will handle camera focus)
			// Note: Camera will be repositioned when model loads, but state is preserved in analysisStore
		} catch (error) {
			console.error('[SCENARIO] Failed to load scenario:', error);
		}
	}
</script>

<div class="scenario-panel">
	<button class="scenario-toggle" on:click={togglePanel}>
		Select Scenario {isExpanded ? '▲' : '▼'}
	</button>
	{#if isExpanded}
		<div class="scenario-content">
			<select
				class="scenario-dropdown"
				value={selectedCategory}
				on:change={handleCategoryChange}
			>
				<option value="">Choose Category...</option>
				{#each categories as category}
					<option value={category.value}>{category.label}</option>
				{/each}
			</select>

			{#if selectedCategory}
				<div class="scenario-variant">
					<label for="scenario-slider">Variant:</label>
					<input
						type="range"
						id="scenario-slider"
						min="1"
						max="10"
						value={selectedVariant}
						step="1"
						on:input={handleVariantChange}
					/>
					<span class="scenario-number">{selectedVariant}</span>
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.scenario-panel {
	width: 100%;
	}

	.scenario-toggle {
		width: 100%;
		padding: 12px 20px;
		background: var(--color-accent);
		color: white;
		border: none;
		border-radius: 5px;
		cursor: pointer;
		font-size: 14px;
		font-weight: bold;
		transition: background 0.2s;
	font-family: var(--font-family);
	}

	.scenario-toggle:hover {
		background: color-mix(in srgb, var(--color-accent) 80%, #000000 20%);
	}

	.scenario-content {
		margin-top: 8px;
		background: var(--color-bg-panel-soft);
		padding: 15px;
		border-radius: 8px;
		box-shadow: var(--shadow-panel);
		max-height: 500px;
		overflow-y: auto;
		transition: max-height 0.3s ease, opacity 0.3s ease;
	}

	.scenario-dropdown {
		width: 100%;
		padding: 10px;
		font-size: 14px;
		border: 1px solid var(--color-border-subtle);
		border-radius: 4px;
		margin-bottom: 10px;
	font-family: var(--font-family);
		background: var(--color-bg-panel);
	}

	.scenario-variant {
		display: flex;
		align-items: center;
		gap: 10px;
		padding-top: 10px;
		border-top: 1px solid var(--color-border-subtle);
	}

	.scenario-variant label {
		font-weight: bold;
		font-size: 13px;
		color: var(--color-text-primary);
	}

	.scenario-variant input[type="range"] {
		flex: 1;
	}

	.scenario-number {
		min-width: 30px;
		font-weight: bold;
		color: var(--color-accent);
		text-align: center;
	}
</style>

