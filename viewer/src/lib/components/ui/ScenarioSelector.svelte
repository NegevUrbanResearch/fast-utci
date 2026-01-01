<script lang="ts">
	import { loadAnalysisData } from '$lib/stores/analysisStore';
	import { cameraStore, focusCameraOnModel } from '$lib/stores/cameraStore';
	import { get } from 'svelte/store';
	import type { Writable } from 'svelte/store';

	let isExpanded = false;
	let selectedCategory = '';
	let selectedVariant = 1;

	const categories = [
		{
			value: 'existing_buildings',
			label: 'Existing buildings with added mass',
			description: 'Current buildings made higher'
		},
		{
			value: 'existing_trees',
			label: 'Existing Tree Cover',
			description: 'From no trees up to current canopy'
		},
		{
			value: 'new_high_buildings',
			label: 'New Highrise Buildings',
			description: 'Adds more tall buildings to the site'
		},
		{
			value: 'new_low_buildings',
			label: 'New Lowrise Buildings',
			description: 'Adds more low and mid-rise buildings'
		},
		{
			value: 'new_trees',
			label: 'New Tree Cover',
			description: 'Adds more tree cover'
		}
	];

	function togglePanel() {
		isExpanded = !isExpanded;
	}

	async function applyCategory(category: string) {
		selectedCategory = category;

		if (selectedCategory) {
			// Auto-load variant 1 when category is selected
			selectedVariant = 1;
			await loadScenario(selectedCategory, 1);
		}
	}

	async function handleCategoryChange(event: Event) {
		const target = event.target as HTMLSelectElement;
		await applyCategory(target.value);
	}

	async function handleVariantClick(variant: number) {
		selectedVariant = variant;
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
	<div class="scenario-summary">
		<div class="summary-label">Active scenario</div>
		<div class="summary-value">
			{#if selectedCategory}
				{categories.find((c) => c.value === selectedCategory)?.label ?? 'Custom'} · Variant&nbsp;{selectedVariant}
			{:else}
				No scenario selected
			{/if}
		</div>
	</div>

	<button class="scenario-toggle" type="button" on:click={togglePanel}>
		<span class="toggle-title">Browse variants</span>
		<span class="toggle-meta">{isExpanded ? 'Hide options' : 'Compare design scenarios'}</span>
		<span class="chevron" aria-hidden="true">{isExpanded ? '▴' : '▾'}</span>
	</button>

	{#if isExpanded}
		<div class="scenario-content">
			<div class="category-list" role="list">
				{#each categories as category}
					<button
						type="button"
						class="category-item"
						class:category-item-active={selectedCategory === category.value}
						on:click={() => applyCategory(category.value)}
					>
						<div class="category-content">
							<div class="category-header">
								<div class="category-title">{category.label}</div>
								{#if selectedCategory === category.value}
									<div class="category-indicator" aria-hidden="true">✓</div>
								{/if}
							</div>
							<div class="category-description">{category.description}</div>
						</div>
					</button>
				{/each}
			</div>

			{#if selectedCategory}
				<div class="variant-buttons">
					<div class="variant-header">
						<div class="variant-title">Variant</div>
						<span class="variant-badge">#{selectedVariant}</span>
					</div>
					<div class="variant-grid" role="group" aria-label="Select variant">
						{#each Array(10) as _, idx}
							<button
								type="button"
								class:selected={selectedVariant === idx + 1}
								on:click={() => handleVariantClick(idx + 1)}
								aria-pressed={selectedVariant === idx + 1}
							>
								{idx + 1}
							</button>
						{/each}
					</div>
					<div class="variant-hints">
						<span>1 · Less</span>
						<span>More · 10</span>
					</div>
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.scenario-panel {
		width: 100%;
		min-width: 0;
		max-width: 100%;
		box-sizing: border-box;
		contain: layout;
	}

	.scenario-toggle {
		width: 100%;
		padding: 10px 12px;
		background: var(--color-bg-panel-soft);
		color: var(--color-text-primary);
		border: none;
		border-radius: var(--radius-control);
		cursor: pointer;
		font-size: 13px;
		font-weight: 500;
		transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease;
		font-family: var(--font-family);
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.scenario-toggle:hover {
		background: var(--color-accent-soft);
		box-shadow: 0 0 0 1px var(--color-border-subtle);
	}

	.scenario-toggle:active {
		transform: translateY(1px);
	}

	.toggle-title {
		text-transform: uppercase;
		letter-spacing: 0.08em;
		font-size: 11px;
		color: var(--color-text-secondary);
	}

	.toggle-meta {
		font-size: 11px;
		color: var(--color-text-muted);
		margin-left: 8px;
		flex: 1;
		text-align: left;
	}

	.chevron {
		margin-left: 8px;
	}

	.scenario-summary {
		margin-bottom: 8px;
		min-width: 0;
		max-width: 100%;
		box-sizing: border-box;
	}

	.summary-label {
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--color-text-secondary);
		margin-bottom: 2px;
	}

	.summary-value {
		font-size: 13px;
		color: var(--color-text-primary);
		min-height: 2.6em;
		height: auto;
		word-wrap: break-word;
		overflow-wrap: break-word;
		width: 100%;
		max-width: 100%;
		min-width: 0;
		box-sizing: border-box;
		overflow: hidden;
		display: block;
		white-space: normal;
		line-height: 1.3;
	}

	.scenario-content {
		margin-top: 8px;
		background: var(--color-bg-panel-soft);
		padding: 12px;
		border-radius: var(--radius-panel);
		box-shadow: var(--shadow-panel);
		width: 100%;
		min-width: 0;
		max-width: 100%;
		box-sizing: border-box;
	}

	.category-list {
		display: flex;
		flex-direction: column;
		gap: 6px;
		margin-bottom: 10px;
		width: 100%;
		min-width: 0;
		max-width: 100%;
		box-sizing: border-box;
	}

	.category-item {
		width: 100%;
		min-width: 0;
		max-width: 100%;
		padding: 0;
		border-radius: var(--radius-control);
		border: 1px solid var(--color-border-subtle);
		background: var(--color-bg-panel);
		cursor: pointer;
		text-align: left;
		transition: all 0.2s ease;
		font-family: var(--font-family);
		box-sizing: border-box;
		overflow: hidden;
		position: relative;
	}

	.category-item::before {
		content: '';
		position: absolute;
		left: 0;
		top: 0;
		bottom: 0;
		width: 3px;
		background: var(--color-accent);
		opacity: 0;
		transition: opacity 0.2s ease;
	}

	.category-item:hover {
		background: var(--color-accent-soft);
		border-color: var(--color-border-strong);
		transform: translateX(2px);
	}

	.category-item:hover::before {
		opacity: 0.6;
	}

	.category-item:active {
		transform: translateX(1px);
	}

	.category-item-active {
		border-color: var(--color-accent);
		background: var(--color-accent-soft);
		box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.2);
	}

	.category-item-active::before {
		opacity: 1;
	}

	.category-content {
		padding: 10px 12px;
		display: flex;
		flex-direction: column;
		gap: 4px;
		width: 100%;
		box-sizing: border-box;
	}

	.category-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		width: 100%;
	}

	.category-title {
		font-size: 13px;
		font-weight: 600;
		letter-spacing: 0.02em;
		color: var(--color-text-primary);
		overflow-wrap: break-word;
		word-wrap: break-word;
		min-width: 0;
		flex: 1;
		line-height: 1.4;
	}

	.category-indicator {
		flex-shrink: 0;
		width: 18px;
		height: 18px;
		border-radius: 50%;
		background: var(--color-accent);
		color: var(--color-bg-page);
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 11px;
		font-weight: 700;
		line-height: 1;
	}

	.category-description {
		font-size: 11px;
		color: var(--color-text-secondary);
		line-height: 1.4;
		margin-top: 2px;
		overflow-wrap: break-word;
		word-wrap: break-word;
	}
	.variant-buttons {
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding-top: 12px;
		border-top: 1px solid var(--color-border-subtle);
		width: 100%;
		min-width: 0;
		max-width: 100%;
		box-sizing: border-box;
	}

	.variant-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		width: 100%;
	}

	.variant-title {
		font-weight: 600;
		font-size: 12px;
		color: var(--color-text-primary);
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}

	.variant-badge {
		font-size: 11px;
		padding: 2px 8px;
		border-radius: 999px;
		background: var(--color-accent-soft);
		color: var(--color-text-primary);
	}

	.variant-grid {
		display: grid;
		grid-template-columns: repeat(5, minmax(0, 1fr));
		gap: 6px;
		width: 100%;
		min-width: 0;
		max-width: 100%;
		box-sizing: border-box;
	}

	.variant-grid button {
		width: 100%;
		padding: 8px 0;
		border-radius: var(--radius-control);
		border: 1px solid var(--color-border-subtle);
		background: var(--color-bg-panel);
		color: var(--color-text-primary);
		font-weight: 600;
		font-size: 12px;
		cursor: pointer;
		transition: background 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
	}

	.variant-grid button:hover {
		background: var(--color-accent-soft);
		border-color: var(--color-border-strong);
	}

	.variant-grid button:active {
		transform: translateY(1px);
	}

	.variant-grid button.selected {
		border-color: var(--color-accent);
		box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.4);
		background: var(--color-accent-soft);
	}

	.variant-hints {
		display: flex;
		justify-content: space-between;
		width: 100%;
		font-size: 10px;
		color: var(--color-text-muted);
	}
</style>
