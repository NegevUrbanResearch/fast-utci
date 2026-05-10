<script lang="ts">
	/**
	 * ScenarioSelector Component
	 *
	 * ABOUTME: UI component for selecting and comparing scenario variants.
	 * When a scenario is selected, it triggers comparison mode instead of replacing
	 * the base analysis, allowing side-by-side comparison with a curtain slider.
	 */
	import { comparisonStore, startComparison } from "$lib/stores/comparisonStore";

	export let projectId: string = "Ben-Gurion";
	export let mode: "compare" | "replace-analysis" = "compare";
	export let activeAnalysisId: string | null = null;
	export let onSelectScenarioAnalysisId:
		| ((analysisId: string) => Promise<void> | void)
		| undefined = undefined;

	let isExpanded = false;
	let selectedCategory = "";
	let selectedVariant = 1;

	const categories = [
		{
			value: "existing_buildings",
			label: "Existing buildings with added mass",
			description: "Current buildings made higher",
		},
		{
			value: "existing_trees",
			label: "Existing Tree Cover",
			description: "From no trees up to current canopy",
		},
		{
			value: "new_high_buildings",
			label: "New Highrise Buildings",
			description: "Adds more tall buildings to the site",
		},
		{
			value: "new_low_buildings",
			label: "New Lowrise Buildings",
			description: "Adds more low and mid-rise buildings",
		},
		{
			value: "new_trees",
			label: "New Tree Cover",
			description: "Adds more tree cover",
		},
	];

	// Export selected category label for use by parent components
	export function getSelectedCategoryLabel(): string {
		if (!selectedCategory) return "";
		return (
			categories.find((c) => c.value === selectedCategory)?.label ??
			"Custom"
		);
	}

	// Export selected variant for use by parent components
	export function getSelectedVariant(): number {
		return selectedVariant;
	}

	// Get scenario name for comparison curtain label
	export function getScenarioName(): string {
		if (!selectedCategory) return "Comparison";
		const label =
			categories.find((c) => c.value === selectedCategory)?.label ??
			"Comparison";
		return `${label} #${selectedVariant}`;
	}

	function togglePanel() {
		isExpanded = !isExpanded;
	}

	function resetSelection() {
		selectedCategory = "";
		selectedVariant = 1;
	}

	function syncSelectionFromAnalysisId(analysisId: string | null) {
		if (mode !== "replace-analysis" || projectId !== "Ben-Gurion") return;
		if (!analysisId) {
			resetSelection();
			return;
		}

		const match = analysisId.match(
			/^Ben-Gurion\/([^/]+)\/\1_(\d{2})$/,
		);
		if (!match) {
			resetSelection();
			return;
		}

		const [, category, variantToken] = match;
		const categoryExists = categories.some(({ value }) => value === category);
		const variant = Number.parseInt(variantToken, 10);
		if (!categoryExists || !Number.isInteger(variant) || variant < 1 || variant > 10) {
			resetSelection();
			return;
		}

		selectedCategory = category;
		selectedVariant = variant;
	}

	async function applyCategory(category: string) {
		if (projectId !== "Ben-Gurion") return;
		if (!category) {
			resetSelection();
			return;
		}

		if (mode === "replace-analysis") {
			const previousCategory = selectedCategory;
			const previousVariant = selectedVariant;
			const success = await loadScenario(category, 1);
			if (success) {
				selectedCategory = category;
				selectedVariant = 1;
			} else {
				selectedCategory = previousCategory;
				selectedVariant = previousVariant;
			}
			return;
		}

		selectedCategory = category;
		selectedVariant = 1;
		await loadScenario(selectedCategory, 1);
	}

	async function handleCategoryChange(event: Event) {
		if (projectId !== "Ben-Gurion") return;
		const target = event.target as HTMLSelectElement;
		await applyCategory(target.value);
	}

	async function handleVariantClick(variant: number) {
		if (projectId !== "Ben-Gurion") return;
		if (mode === "replace-analysis") {
			const previousVariant = selectedVariant;
			const success = await loadScenario(selectedCategory, variant);
			selectedVariant = success ? variant : previousVariant;
			return;
		}

		selectedVariant = variant;
		await loadScenario(selectedCategory, selectedVariant);
	}

	async function loadScenario(category: string, variant: number): Promise<boolean> {
		try {
			// Construct analysis ID: category/category_variant (e.g., "existing_buildings/existing_buildings_01")
			const variantStr = variant.toString().padStart(2, "0");
			const analysisId = `Ben-Gurion/${category}/${category}_${variantStr}`;

			if (mode === "replace-analysis") {
				if (!onSelectScenarioAnalysisId) return false;
				console.log(`[SCENARIO] Replacing active analysis: ${analysisId}`);
				await onSelectScenarioAnalysisId(analysisId);
				return true;
			}

			console.log(`[SCENARIO] Starting comparison: ${analysisId}`);

			// Start comparison mode instead of replacing base analysis
			// This loads the comparison analysis and enables the curtain slider
			await startComparison(analysisId);
			return true;
		} catch (error) {
			console.error("[SCENARIO] Failed to load scenario:", error);
			return false;
		}
	}

	// Check if currently comparing
	$: isComparing = $comparisonStore.isComparing;
	$: isLoading = $comparisonStore.isLoading;
	$: isProjectSupported = projectId === "Ben-Gurion";

	// Reset selection when comparison is exited (via curtain or any other means)
	$: if (mode === "compare" && !isComparing && selectedCategory) {
		resetSelection();
	}

	$: if (!isProjectSupported && selectedCategory) {
		resetSelection();
	}

	$: if (mode === "replace-analysis") {
		syncSelectionFromAnalysisId(activeAnalysisId);
	}
</script>

<div class="scenario-panel">
	<div class="scenario-summary">
		<div class="summary-label">
			{#if mode === "compare" && isComparing}
				Comparing with
			{:else if mode === "replace-analysis"}
				Selected scenario
			{:else}
				Active scenario
			{/if}
		</div>
		<div class="summary-value">
			{#if mode === "compare" && isLoading}
				Loading scenario...
			{:else if selectedCategory}
				{categories.find((c) => c.value === selectedCategory)?.label ??
					"Custom"} · Variant&nbsp;{selectedVariant}
			{:else if mode === "replace-analysis"}
				Base analysis
			{:else}
				No scenario selected
			{/if}
		</div>
	</div>

	<button class="scenario-toggle" type="button" on:click={togglePanel}>
		<span class="toggle-title">Browse variants</span>
		<span class="toggle-meta">
			{#if isExpanded}
				Hide options
			{:else if mode === "replace-analysis"}
				Load scenario analysis
			{:else}
				Compare design scenarios
			{/if}
		</span>
		<span class="chevron" aria-hidden="true">{isExpanded ? "▴" : "▾"}</span>
	</button>

	{#if isExpanded}
		<div class="scenario-content">
			{#if isProjectSupported}
				<div class="category-list" role="list">
					{#each categories as category}
						<button
							type="button"
							class="category-item"
							class:category-item-active={selectedCategory ===
								category.value}
							on:click={() => applyCategory(category.value)}
						>
							<div class="category-content">
								<div class="category-header">
									<div class="category-title">
										{category.label}
									</div>
								</div>
								<div class="category-description">
									{category.description}
								</div>
							</div>
						</button>
					{/each}
				</div>
			{:else}
				<div class="scenario-empty">
					Scenarios are only available for Ben-Gurion right now.
				</div>
			{/if}

			{#if selectedCategory && isProjectSupported}
				<div class="variant-buttons">
					<div class="variant-header">
						<div class="variant-title">Variant</div>
						<span class="variant-badge">#{selectedVariant}</span>
					</div>
					<div
						class="variant-grid"
						role="group"
						aria-label="Select variant"
					>
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
	:global(button) {
		border: none;
		outline: none;
	}

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
		font-size: var(--font-sm);
		font-weight: 500;
		transition:
			background 0.15s ease,
			box-shadow 0.15s ease,
			transform 0.1s ease;
		font-family: var(--font-family);
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.scenario-toggle:hover {
		background: var(--color-accent-soft);
	}

	.scenario-toggle:active {
		transform: translateY(1px);
	}

	.toggle-title {
		text-transform: uppercase;
		letter-spacing: 0.08em;
		font-size: var(--font-xxs);
		color: var(--color-text-secondary);
	}

	.toggle-meta {
		font-size: var(--font-xxs);
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
		font-size: var(--font-xxs);
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--color-text-secondary);
		margin-bottom: 2px;
	}

	.summary-value {
		font-size: var(--font-sm);
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

	.scenario-empty {
		padding: 10px 12px;
		font-size: var(--font-xs);
		color: var(--color-text-muted);
		background: var(--color-bg-panel);
		border-radius: var(--radius-control);
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
		border: none;
		border-radius: var(--radius-control);
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
		content: "";
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
		transform: translateX(2px);
	}

	.category-item:hover::before {
		opacity: 0.6;
	}

	.category-item:active {
		transform: translateX(1px);
	}

	.category-item-active {
		background: var(--color-accent-soft);
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
		font-size: var(--font-sm);
		font-weight: 600;
		letter-spacing: 0.02em;
		color: var(--color-text-primary);
		overflow-wrap: break-word;
		word-wrap: break-word;
		min-width: 0;
		flex: 1;
		line-height: 1.4;
	}

	.category-description {
		font-size: var(--font-xxs);
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
		padding: 12px;
		margin-top: 12px;
		background: rgba(15, 23, 42, 0.4);
		border-radius: var(--radius-control);
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
		font-size: var(--font-xs);
		color: var(--color-text-primary);
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}

	.variant-badge {
		font-size: var(--font-xxs);
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
		border: none;
		border-radius: var(--radius-control);
		background: var(--color-bg-panel);
		color: var(--color-text-primary);
		font-weight: 600;
		font-size: var(--font-xs);
		cursor: pointer;
		transition:
			background 0.15s ease,
			transform 0.1s ease;
	}

	.variant-grid button:hover {
		background: var(--color-accent-soft);
	}

	.variant-grid button:active {
		transform: translateY(1px);
	}

	.variant-grid button.selected {
		background: var(--color-accent-soft);
	}

	.variant-hints {
		display: flex;
		justify-content: space-between;
		width: 100%;
		font-size: var(--font-xxs); /* Really small hints */
		color: var(--color-text-muted);
	}
</style>
