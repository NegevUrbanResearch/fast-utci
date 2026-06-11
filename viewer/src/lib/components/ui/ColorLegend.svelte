<script lang="ts">
	import { analysisStore } from "$lib/stores/analysisStore";
	import { comparisonStore, unifiedUtciRange } from "$lib/stores/comparisonStore";
	import { viewerStore, setMetricType } from "$lib/stores/viewerStore";
	import {
		LADYBUG_NUANCED_COLORS,
		SHADING_INDEX_COLORS,
		createShadingIndexLegendData,
	} from "$lib/services/colorScale";
	import { getUtciRangeForDisplay } from "$lib/utils/effectiveHourIndex";
	import type { Analysis } from "$lib/types/analysis";
	import type { MetricType } from "$lib/types/viewer";

	/** When set (e.g. debug page showing only live layer), use this for legend range instead of analysisStore. */
	export let displayAnalysis: Analysis | null = null;
	export let utciRangeOverride: { min: number; max: number } | null | undefined = undefined;

	let utciMin = 0;
	let utciMax = 100;
	let shadingIndexMin = 0;
	let shadingIndexMax = 1;

	$: effectiveAnalysis = displayAnalysis ?? $analysisStore;

	$: if (effectiveAnalysis) {
		if ($viewerStore.metricType === "utci") {
			const inComparison = $comparisonStore.isComparing && $unifiedUtciRange;
			if (inComparison) {
				utciMin = $unifiedUtciRange.utciMin;
				utciMax = $unifiedUtciRange.utciMax;
			} else if (
				utciRangeOverride &&
				Number.isFinite(utciRangeOverride.min) &&
				Number.isFinite(utciRangeOverride.max) &&
				utciRangeOverride.max > utciRangeOverride.min
			) {
				utciMin = utciRangeOverride.min;
				utciMax = utciRangeOverride.max;
			} else {
				const range = getUtciRangeForDisplay(
					effectiveAnalysis.metadata,
					$viewerStore.colorMode,
					$viewerStore.currentHour,
					$viewerStore.currentMonth ?? 7
				);
				utciMin = range.utciMin;
				utciMax = range.utciMax;
			}
		} else {
			// Shading Index
			if (effectiveAnalysis.metadata.shading_index_range) {
				shadingIndexMin =
					effectiveAnalysis.metadata.shading_index_range.min;
				shadingIndexMax =
					effectiveAnalysis.metadata.shading_index_range.max;
			}
		}
	}

	// Make these reactive to ensure updates
	$: isUTCI = $viewerStore.metricType === "utci";
	$: isShadingIndex = $viewerStore.metricType === "shading_index";
	$: hasShadingIndex = effectiveAnalysis?.metadata.has_shading_index ?? false;

	function selectUTCI() {
		if (!isUTCI) {
			setMetricType("utci");
		}
	}

	function selectShadingIndex() {
		if (!isShadingIndex && hasShadingIndex) {
			setMetricType("shading_index");
		}
	}

	// Create UTCI gradient (reactive to metricType to force update)
	$: utciGradient =
		$viewerStore.metricType === "utci"
			? [...LADYBUG_NUANCED_COLORS]
					.reverse()
					.map((color, i) => {
						const stepSize = 100 / LADYBUG_NUANCED_COLORS.length;
						const start = (i * stepSize).toFixed(2);
						const end = ((i + 1) * stepSize).toFixed(2);
						return `${color} ${start}%, ${color} ${end}%`;
					})
					.join(", ")
			: "";

	// Create Shading Index gradient (reactive to metricType to force update)
	// Reverse the gradient so high values (excellent, green) are at top, low values (poor, red) at bottom
	$: shadingIndexGradient =
		$viewerStore.metricType === "shading_index"
			? [...createShadingIndexLegendData()]
					.reverse()
					.map((item, i) => {
						const stepSize = 100 / 4; // 4 categories
						const start = (i * stepSize).toFixed(2);
						const end = ((i + 1) * stepSize).toFixed(2);
						return `${item.color} ${start}%, ${item.color} ${end}%`;
					})
					.join(", ")
			: "";

	$: shadingIndexLabels =
		$viewerStore.metricType === "shading_index"
			? createShadingIndexLegendData()
			: [];
	$: shadingIndexLabelsReversed =
		shadingIndexLabels.length > 0 ? [...shadingIndexLabels].reverse() : [];
</script>

{#if effectiveAnalysis}
	<div class="color-legend">
		<div class="legend-header">
			<div class="title">
				{#if isUTCI}
					UTCI
				{:else}
					Shading Index
				{/if}
			</div>
		</div>

		<div class="gradient-row" class:shading-row={isShadingIndex}>
			{#if isUTCI}
				<div class="gradient-container">
					<div
						class="gradient"
						style="background: linear-gradient(to bottom, {utciGradient})"
					></div>
					<div class="labels">
						{#each Array(6) as _, i}
							{@const temp =
								utciMax - (i * (utciMax - utciMin)) / 5}
							{@const position = (i / 5) * 100}
							<div class="label" style="top: {position}%">
								{temp.toFixed(1)}°C
							</div>
						{/each}
					</div>
				</div>
			{:else}
				<div class="gradient-container shading-gradient-container">
					<div
						class="gradient"
						style="background: linear-gradient(to bottom, {shadingIndexGradient})"
						aria-hidden="true"
					></div>
					<div class="labels shading-labels" aria-label="Shading Index legend">
						<div class="label shading-limit-label" style="top: 0%">
							{shadingIndexMax.toFixed(1)}
						</div>
						{#each shadingIndexLabelsReversed as item, i}
							<div class="shading-category-label" style="top: {(i + 0.5) * 25}%">
								<div class="shading-category-name">{item.abbrev}</div>
								<div class="shading-category-range">
									{item.range[0].toFixed(1)}-{item.range[1].toFixed(1)}
								</div>
							</div>
						{/each}
						<div class="label shading-limit-label" style="top: 100%">
							{shadingIndexMin.toFixed(1)}
						</div>
					</div>
				</div>
			{/if}

			{#if hasShadingIndex}
				<div class="metric-column">
					<div class="mode-caption">
						<span>Metric Type</span>
					</div>
					<div
						class="mode-toggle-vertical"
						aria-label="Metric type"
						role="toolbar"
					>
						<button
							type="button"
							class="mode-pill-vertical"
							class:mode-pill-vertical-active={isUTCI}
							on:click={selectUTCI}
							aria-pressed={isUTCI}
						>
							<span class="mode-pill-label">UTCI</span>
						</button>
						<button
							type="button"
							class="mode-pill-vertical"
							class:mode-pill-vertical-active={isShadingIndex}
							on:click={selectShadingIndex}
							aria-pressed={isShadingIndex}
							disabled={!hasShadingIndex}
						>
							<span class="mode-pill-label">Shading</span>
						</button>
					</div>
					<div class="mode-help"></div>
				</div>
			{/if}
		</div>
	</div>
{/if}

<style>
	.color-legend {
		background: var(--color-bg-panel-soft);
		padding: var(--spacing-lg);
		border-radius: var(--radius-panel);
		box-shadow: var(--shadow-panel);
		z-index: var(--z-panel);
		min-width: 200px;
		max-width: 280px;
		box-sizing: border-box;
		overflow: hidden;
	}

	.title {
		font-weight: 600;
		font-size: var(--font-md);
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--color-text-primary);
	}

	.legend-header {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		margin-bottom: var(--spacing-sm);
	}

	.gradient-row {
		display: flex;
		align-items: flex-start;
		gap: 20px;
	}

	.shading-row {
		gap: 18px;
	}

	.gradient-container {
		position: relative;
		display: flex;
		align-items: stretch;
		gap: var(--spacing-md);
		margin-bottom: var(--spacing-md);
		width: 103px;
		flex-shrink: 0;
	}

	.gradient {
		width: 35px;
		height: 250px;
		border-radius: 5px;
		box-shadow:
			inset 0 0 5px rgba(0, 0, 0, 0.1),
			0 0 0 1px rgba(148, 163, 184, 0.2);
		flex-shrink: 0;
	}

	.labels {
		position: relative;
		height: 250px;
		width: 60px;
		flex-shrink: 0;
	}

	.label {
		position: absolute;
		right: 0px;
		transform: translateY(-50%);
		font-size: var(--font-xs);
		font-weight: 500;
		white-space: nowrap;
		color: var(--color-text-primary);
		text-align: left;
		min-width: 50px;
	}

	.shading-gradient-container {
		width: 110px;
	}

	.shading-labels {
		width: 66px;
	}

	.shading-limit-label {
		left: 0;
		right: auto;
		min-width: 0;
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}

	.shading-category-label {
		position: absolute;
		left: 0;
		transform: translateY(-50%);
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 2px;
		font-size: var(--font-xs);
		line-height: 1.05;
		color: var(--color-text-primary);
	}

	.shading-category-name {
		font-weight: 600;
		white-space: nowrap;
	}

	.shading-category-range {
		font-size: 10px;
		color: var(--color-text-secondary);
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}

	.metric-column {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 8px;
		min-width: 0;
		max-width: 130px;
	}

	.mode-caption {
		display: flex;
		flex-direction: column;
		align-items: center;
		font-size: var(--font-xs);
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--color-text-secondary);
		line-height: 1.1;
		text-align: center;
	}

	.mode-toggle-vertical {
		display: flex;
		flex-direction: column;
		align-items: stretch;
		background: var(--color-bg-panel);
		border-radius: 18px;
		padding: 4px;
		gap: 2px;
		border: 1px solid rgba(148, 163, 184, 0.45);
		box-shadow: 0 10px 24px rgba(15, 23, 42, 0.35);
		overflow: hidden;
	}

	.mode-pill-vertical {
		border: none;
		background: transparent;
		color: var(--color-text-secondary);
		font-size: var(--font-xs);
		padding: 6px 12px;
		border-radius: 999px;
		cursor: pointer;
		text-align: center;
		transition:
			background 0.16s ease,
			color 0.16s ease;
		font-family: var(--font-family);
	}

	.mode-pill-vertical:hover {
		background: rgba(148, 163, 184, 0.08);
	}

	.mode-pill-vertical-active {
		background: linear-gradient(
			to bottom,
			rgba(56, 189, 248, 0.24),
			rgba(56, 189, 248, 0.55)
		);
		color: var(--color-bg-elevated);
	}

	.mode-pill-vertical:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.mode-pill-label {
		display: inline-block;
		font-weight: 500;
		letter-spacing: 0.04em;
		text-transform: uppercase;
	}

	.mode-help {
		font-size: var(--font-xs);
		color: var(--color-text-muted);
		line-height: 1.4;
		min-height: 2.8em;
		text-align: center;
		display: flex;
		align-items: center;
		justify-content: center;
	}
</style>
