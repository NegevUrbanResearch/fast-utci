<script lang="ts">
	import { analysisStore } from '$lib/stores/analysisStore';
	import { viewerStore, setMetricType } from '$lib/stores/viewerStore';
	import { 
		LADYBUG_NUANCED_COLORS, 
		SHADING_INDEX_COLORS,
		createShadingIndexLegendData 
	} from '$lib/services/colorScale';
	import type { MetricType } from '$lib/types/viewer';

	let utciMin = 0;
	let utciMax = 100;
	let shadingIndexMin = 0;
	let shadingIndexMax = 1;

	$: if ($analysisStore) {
		if ($viewerStore.metricType === 'utci') {
			if ($viewerStore.colorMode === 'normalized') {
				utciMin = $analysisStore.metadata.utci_range.min;
				utciMax = $analysisStore.metadata.utci_range.max;
			} else {
				const hourStat = $analysisStore.metadata.hour_statistics?.[$viewerStore.currentHour];
				if (hourStat) {
					utciMin = hourStat.min;
					utciMax = hourStat.max;
				} else {
					utciMin = $analysisStore.metadata.utci_range.min;
					utciMax = $analysisStore.metadata.utci_range.max;
				}
			}
		} else {
			// Shading Index
			if ($analysisStore.metadata.shading_index_range) {
				shadingIndexMin = $analysisStore.metadata.shading_index_range.min;
				shadingIndexMax = $analysisStore.metadata.shading_index_range.max;
			}
		}
	}

	// Make these reactive to ensure updates
	$: isUTCI = $viewerStore.metricType === 'utci';
	$: isShadingIndex = $viewerStore.metricType === 'shading_index';
	$: hasShadingIndex = $analysisStore?.metadata.has_shading_index ?? false;

	function selectUTCI() {
		if (!isUTCI) {
			setMetricType('utci');
		}
	}

	function selectShadingIndex() {
		if (!isShadingIndex && hasShadingIndex) {
			setMetricType('shading_index');
		}
	}

	// Create UTCI gradient (reactive to metricType to force update)
	$: utciGradient = $viewerStore.metricType === 'utci' ? [...LADYBUG_NUANCED_COLORS].reverse().map((color, i) => {
		const stepSize = 100 / LADYBUG_NUANCED_COLORS.length;
		const start = (i * stepSize).toFixed(2);
		const end = ((i + 1) * stepSize).toFixed(2);
		return `${color} ${start}%, ${color} ${end}%`;
	}).join(', ') : '';

	// Create Shading Index gradient (reactive to metricType to force update)
	// Reverse the gradient so high values (excellent, green) are at top, low values (poor, red) at bottom
	$: shadingIndexGradient = $viewerStore.metricType === 'shading_index' ? [...createShadingIndexLegendData()].reverse().map((item, i) => {
		const stepSize = 100 / 4; // 4 categories
		const start = (i * stepSize).toFixed(2);
		const end = ((i + 1) * stepSize).toFixed(2);
		return `${item.color} ${start}%, ${item.color} ${end}%`;
	}).join(', ') : '';

	$: shadingIndexLabels = $viewerStore.metricType === 'shading_index' ? createShadingIndexLegendData() : [];
	$: shadingIndexLabelsReversed = shadingIndexLabels.length > 0 ? [...shadingIndexLabels].reverse() : [];
</script>

{#if $analysisStore}
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

		<div class="gradient-row">
			<div class="gradient-container">
				{#if isUTCI}
					<div
						class="gradient"
						style="background: linear-gradient(to bottom, {utciGradient})"
					></div>
					<div class="labels">
						{#each Array(6) as _, i}
							{@const temp = utciMax - (i * (utciMax - utciMin) / 5)}
							{@const position = (i / 5) * 100}
							<div class="label" style="top: {position}%">
								{temp.toFixed(1)}°C
							</div>
						{/each}
					</div>
				{:else}
					<div
						class="gradient"
						style="background: linear-gradient(to bottom, {shadingIndexGradient})"
					></div>
					<div class="labels">
						{#if shadingIndexLabels.length > 0}
							<!-- Show all category boundaries: 1.0, 0.9, 0.7, 0.5, 0.0 -->
							<!-- Top: 1.0 (excellent max) -->
							<div class="label" style="top: 0%">
								1.0
							</div>
							<!-- 0.9 (boundary between excellent and good) -->
							<div class="label" style="top: 25%">
								0.9
							</div>
							<!-- 0.7 (boundary between good and acceptable) -->
							<div class="label" style="top: 50%">
								0.7
							</div>
							<!-- 0.5 (boundary between acceptable and poor) -->
							<div class="label" style="top: 75%">
								0.5
							</div>
							<!-- Bottom: 0.0 (poor min) -->
							<div class="label" style="top: 100%">
								0.0
							</div>
						{/if}
					</div>
				{/if}
			</div>

			{#if hasShadingIndex}
				<div class="metric-column">
					<div class="mode-caption">
						<span>Metric Type</span>
					</div>
					<div class="mode-toggle-vertical" aria-label="Metric type" role="toolbar">
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
					<div class="mode-help">
					</div>
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
		font-size: 14px;
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

	.range {
		font-size: 12px;
	}

	.range-values {
		color: var(--color-text-primary);
	}

	.hour-label {
		font-size: 11px;
		color: var(--color-text-secondary);
		margin-left: 4px;
	}

	.gradient-row {
		display: flex;
		align-items: flex-start;
		gap: 20px;
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
		font-size: 12px;
		font-weight: 500;
		white-space: nowrap;
		color: var(--color-text-primary);
		text-align: left;
		min-width: 50px;
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
		font-size: 12px;
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
		font-size: 12px;
		padding: 6px 12px;
		border-radius: 999px;
		cursor: pointer;
		text-align: center;
		transition:
			background 0.16s ease,
			color 0.16s ease;
	}

	.mode-pill-vertical:hover {
		background: rgba(148, 163, 184, 0.08);
	}

	.mode-pill-vertical-active {
		background: linear-gradient(to bottom, rgba(56, 189, 248, 0.24), rgba(56, 189, 248, 0.55));
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
		font-size: 12px;
		color: var(--color-text-muted);
		line-height: 1.4;
		min-height: 2.8em;
		text-align: center;
		display: flex;
		align-items: center;
		justify-content: center;
	}
</style>


