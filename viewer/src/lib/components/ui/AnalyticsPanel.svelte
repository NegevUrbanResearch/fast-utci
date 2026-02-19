<script lang="ts">
	/**
	 * AnalyticsPanel Component
	 *
	 * ABOUTME: Displays analysis metrics and statistics. When in comparison mode,
	 * shows side-by-side metrics for base vs comparison scenarios with delta values.
	 */
	import { onMount } from "svelte";
	import { analysisStore } from "$lib/stores/analysisStore";
	import { viewerStore } from "$lib/stores/viewerStore";
	import {
		comparisonStore,
		comparisonAnalysis,
	} from "$lib/stores/comparisonStore";
	import {
		loadValidationData,
		compareWithValidation,
		calculateAvgMeanDiffAllHours,
	} from "$lib/services/validationService";
	import { getShadingIndex } from "$lib/services/dataLoader";
	import type { ComparisonStats } from "$lib/services/validationService";
	import type { Analysis } from "$lib/types/analysis";

	let validationStats: ComparisonStats | null = null;
	let avgMeanDiffAllHours: number | null = null;
	let validationLoaded = false;
	let showValidation = false;

	$: metadata = $analysisStore?.metadata;
	$: currentHour = $viewerStore.currentHour;
	$: metricType = $viewerStore.metricType;
	$: isComparing = $comparisonStore.isComparing;

	// Calculate UTCI statistics for current hour/selection (base analysis)
	$: baseUtciStats = calculateCurrentUtciStats(
		$analysisStore,
		$viewerStore.currentHour,
	);

	// Calculate UTCI statistics for comparison analysis
	$: comparisonUtciStats = isComparing
		? calculateCurrentUtciStats(
				$comparisonAnalysis,
				$viewerStore.currentHour,
			)
		: null;

	// Calculate Shading Index statistics (base)
	$: baseShadingStats = calculateShadingIndexStats($analysisStore);

	// Calculate Shading Index statistics (comparison)
	$: comparisonShadingStats = isComparing
		? calculateShadingIndexStats($comparisonAnalysis)
		: null;

	// Calculate delta values for comparison
	$: utciDelta =
		baseUtciStats && comparisonUtciStats
			? {
					mean: comparisonUtciStats.mean - baseUtciStats.mean,
					min: comparisonUtciStats.min - baseUtciStats.min,
					max: comparisonUtciStats.max - baseUtciStats.max,
				}
			: null;

	$: shadingDelta =
		baseShadingStats && comparisonShadingStats
			? {
					mean: comparisonShadingStats.mean - baseShadingStats.mean,
					min: comparisonShadingStats.min - baseShadingStats.min,
					max: comparisonShadingStats.max - baseShadingStats.max,
				}
			: null;

	let lastAnalysisId: string | null = null;

	// Load validation data when analysis changes (use date or model_file as unique identifier)
	$: if (metadata && (metadata.date || metadata.model_file)) {
		const currentId = metadata.date || metadata.model_file || "";
		if (currentId !== lastAnalysisId) {
			lastAnalysisId = currentId;
			validationLoaded = false;
			validationStats = null;
			avgMeanDiffAllHours = null;
			loadValidation();
		}
	}

	// Update validation stats when hour changes (if validation is loaded)
	$: if (validationStats && $analysisStore && validationLoaded) {
		updateComparison($viewerStore.currentHour);
	}

	let validationCache: any = null;

	async function loadValidation() {
		try {
			if (!validationCache) {
				validationCache = await loadValidationData();
			}
			const validation = validationCache;

			if ($analysisStore && validation) {
				validationStats = compareWithValidation(
					$analysisStore,
					validation,
					$viewerStore.currentHour,
				);
				if ($analysisStore.metadata.analysis_type === "full_day") {
					avgMeanDiffAllHours = calculateAvgMeanDiffAllHours(
						$analysisStore,
						validation,
					);
				}
				validationLoaded = true;
			}
		} catch (error) {
			console.warn("[WARN] Could not load validation data:", error);
			validationLoaded = true; // Mark as loaded to prevent retry loops
		}
	}

	async function updateComparison(hourIndex: number) {
		if (!$analysisStore || !validationLoaded || !validationCache) return;

		try {
			validationStats = compareWithValidation(
				$analysisStore,
				validationCache,
				hourIndex,
			);
		} catch (error) {
			console.warn(
				"[WARN] Could not update validation comparison:",
				error,
			);
		}
	}

	/**
	 * Calculate UTCI statistics for the current hour/selection
	 * For single_hour: uses the single utciValues array
	 * For full_day: uses the utciByHour array for the selected hour
	 */
	function calculateCurrentUtciStats(
		analysis: Analysis | null,
		hourIndex: number,
	): { min: number; max: number; mean: number } | null {
		if (!analysis) return null;

		let utciValues: Float32Array;

		if (analysis.metadata.analysis_type === "single_hour") {
			// Single hour analysis - use utciValues directly
			utciValues = (analysis.data as any).utciValues;
		} else {
			// Full day analysis - use utciByHour for selected hour
			const utciByHour = (analysis.data as any).utciByHour;
			if (
				!utciByHour ||
				hourIndex < 0 ||
				hourIndex >= utciByHour.length
			) {
				return null;
			}
			utciValues = utciByHour[hourIndex];
		}

		if (!utciValues || utciValues.length === 0) return null;

		// Calculate min/max/mean from the array
		let min = Infinity;
		let max = -Infinity;
		let sum = 0;

		for (let i = 0; i < utciValues.length; i++) {
			const val = utciValues[i];
			if (val < min) min = val;
			if (val > max) max = val;
			sum += val;
		}

		const mean = sum / utciValues.length;

		return { min, max, mean };
	}

	/**
	 * Calculate Shading Index statistics
	 * Shading Index is a full-day metric (not hour-specific)
	 */
	function calculateShadingIndexStats(
		analysis: Analysis | null,
	): { min: number; max: number; mean: number } | null {
		if (!analysis) return null;

		const shadingIndexValues = getShadingIndex(analysis.data);
		if (!shadingIndexValues || shadingIndexValues.length === 0) return null;

		// Calculate min/max/mean from the array
		let min = Infinity;
		let max = -Infinity;
		let sum = 0;
		let count = 0;

		for (let i = 0; i < shadingIndexValues.length; i++) {
			const val = shadingIndexValues[i];
			if (!isNaN(val) && isFinite(val)) {
				if (val < min) min = val;
				if (val > max) max = val;
				sum += val;
				count++;
			}
		}

		if (count === 0) return null;

		const mean = sum / count;

		return { min, max, mean };
	}

	/**
	 * Format delta value with sign
	 */
	function formatDelta(value: number, decimals: number = 1): string {
		const sign = value >= 0 ? "+" : "";
		return `${sign}${value.toFixed(decimals)}`;
	}
</script>

{#if metadata}
	<div class="analytics-panel">
		<div class="panel-header">Analysis Info</div>

		<div class="panel-section">
			{#if metadata.date}
				<strong>Date:</strong> {metadata.date}<br />
			{/if}
			<strong>Grid Size:</strong>
			{metadata.grid_size}m<br />
			<strong>Positions:</strong>
			{metadata.num_positions.toLocaleString()}<br />
			{#if "runtime_seconds" in metadata && typeof metadata.runtime_seconds === "number"}
				<strong>Runtime:</strong> {metadata.runtime_seconds.toFixed(1)}s
			{/if}
		</div>

		<!-- Scenario Comparison Section (when comparing) -->
		{#if isComparing}
			<div class="comparison-section">
				<div class="comparison-header">Scenario Comparison</div>

				{#if metricType === "shading_index"}
					<!-- Shading Index Comparison Table -->
					<table class="comparison-table">
						<thead>
							<tr>
								<th>Metric</th>
								<th>Base</th>
								<th>Scenario</th>
								<th>Delta</th>
							</tr>
						</thead>
						<tbody>
							<tr>
								<td>Mean</td>
								<td
									>{baseShadingStats?.mean.toFixed(2) ??
										"-"}</td
								>
								<td
									>{comparisonShadingStats?.mean.toFixed(2) ??
										"-"}</td
								>
								<td
									class:positive={shadingDelta &&
										shadingDelta.mean > 0}
									class:negative={shadingDelta &&
										shadingDelta.mean < 0}
								>
									{shadingDelta
										? formatDelta(shadingDelta.mean, 2)
										: "-"}
								</td>
							</tr>
							<tr>
								<td>Min</td>
								<td
									>{baseShadingStats?.min.toFixed(2) ??
										"-"}</td
								>
								<td
									>{comparisonShadingStats?.min.toFixed(2) ??
										"-"}</td
								>
								<td
									class:positive={shadingDelta &&
										shadingDelta.min > 0}
									class:negative={shadingDelta &&
										shadingDelta.min < 0}
								>
									{shadingDelta
										? formatDelta(shadingDelta.min, 2)
										: "-"}
								</td>
							</tr>
							<tr>
								<td>Max</td>
								<td
									>{baseShadingStats?.max.toFixed(2) ??
										"-"}</td
								>
								<td
									>{comparisonShadingStats?.max.toFixed(2) ??
										"-"}</td
								>
								<td
									class:positive={shadingDelta &&
										shadingDelta.max > 0}
									class:negative={shadingDelta &&
										shadingDelta.max < 0}
								>
									{shadingDelta
										? formatDelta(shadingDelta.max, 2)
										: "-"}
								</td>
							</tr>
						</tbody>
					</table>
					<div class="comparison-note">
						Shading Index (higher = more shade)
					</div>
				{:else}
					<!-- UTCI Comparison Table -->
					<table class="comparison-table">
						<thead>
							<tr>
								<th>Metric</th>
								<th>Base</th>
								<th>Scenario</th>
								<th>Delta</th>
							</tr>
						</thead>
						<tbody>
							<tr>
								<td>Mean</td>
								<td>{baseUtciStats?.mean.toFixed(1) ?? "-"}C</td
								>
								<td
									>{comparisonUtciStats?.mean.toFixed(1) ??
										"-"}C</td
								>
								<td
									class:positive={utciDelta &&
										utciDelta.mean < 0}
									class:negative={utciDelta &&
										utciDelta.mean > 0}
								>
									{utciDelta
										? formatDelta(utciDelta.mean, 1) + "C"
										: "-"}
								</td>
							</tr>
							<tr>
								<td>Min</td>
								<td>{baseUtciStats?.min.toFixed(1) ?? "-"}C</td>
								<td
									>{comparisonUtciStats?.min.toFixed(1) ??
										"-"}C</td
								>
								<td
									class:positive={utciDelta &&
										utciDelta.min < 0}
									class:negative={utciDelta &&
										utciDelta.min > 0}
								>
									{utciDelta
										? formatDelta(utciDelta.min, 1) + "C"
										: "-"}
								</td>
							</tr>
							<tr>
								<td>Max</td>
								<td>{baseUtciStats?.max.toFixed(1) ?? "-"}C</td>
								<td
									>{comparisonUtciStats?.max.toFixed(1) ??
										"-"}C</td
								>
								<td
									class:positive={utciDelta &&
										utciDelta.max < 0}
									class:negative={utciDelta &&
										utciDelta.max > 0}
								>
									{utciDelta
										? formatDelta(utciDelta.max, 1) + "C"
										: "-"}
								</td>
							</tr>
						</tbody>
					</table>
					<div class="comparison-note">
						UTCI Hour {currentHour} (lower = cooler)
					</div>
				{/if}
			</div>
		{:else}
			<!-- Standard metrics display (when not comparing) -->
			<div class="panel-section">
				{#if metricType === "shading_index"}
					<strong>Shading Index (Full Day):</strong><br />
					{#if baseShadingStats}
						Min: {baseShadingStats.min.toFixed(2)}<br />
						Max: {baseShadingStats.max.toFixed(2)}<br />
						Mean: {baseShadingStats.mean.toFixed(2)}
					{:else}
						<span class="loading-text"
							>No Shading Index data available</span
						>
					{/if}
				{:else}
					<strong>Data (Hour {currentHour}):</strong><br />
					{#if baseUtciStats}
						Min: {baseUtciStats.min.toFixed(1)}C<br />
						Max: {baseUtciStats.max.toFixed(1)}C<br />
						Mean: {baseUtciStats.mean.toFixed(1)}C
					{:else}
						<span class="loading-text">Loading...</span>
					{/if}
				{/if}
			</div>
		{/if}

		{#if validationStats && !isComparing}
			<button
				type="button"
				class="validation-toggle"
				on:click={() => (showValidation = !showValidation)}
			>
				<span class="validation-title">Validation vs Grasshopper</span>
				<span class:open={showValidation} class="chevron">v</span>
			</button>

			{#if showValidation}
				<div class="panel-header panel-header-secondary">
					Grasshopper Comparison
				</div>
				<div class="panel-section">
					<strong>Validation Data:</strong><br />
					Min: {validationStats.validation.min.toFixed(1)}C<br />
					Max: {validationStats.validation.max.toFixed(1)}C<br />
					Mean: {validationStats.validation.mean.toFixed(1)}C
				</div>
				<div class="panel-section">
					<strong>Comparison Metrics:</strong><br />
					Min Diff: {validationStats.comparison.minDiff >= 0
						? "+"
						: ""}{validationStats.comparison.minDiff.toFixed(2)}C<br
					/>
					Max Diff: {validationStats.comparison.maxDiff >= 0
						? "+"
						: ""}{validationStats.comparison.maxDiff.toFixed(2)}C<br
					/>
					Mean Diff: {validationStats.comparison.meanDiff >= 0
						? "+"
						: ""}{validationStats.comparison.meanDiff.toFixed(2)}C
					{#if avgMeanDiffAllHours !== null}
						<br />24-Hour Avg: {avgMeanDiffAllHours >= 0
							? "+"
							: ""}{avgMeanDiffAllHours.toFixed(2)}C
					{/if}
				</div>
			{/if}
		{/if}
	</div>
{/if}

<style>
	.analytics-panel {
		font-family: var(--font-family);
		font-size: var(--font-xs);
		color: var(--color-text-primary);
	}

	button {
		font-family: var(--font-family);
	}

	.panel-header {
		font-weight: 600;
		font-size: var(--font-sm);
		margin-bottom: 8px;
		padding-bottom: 4px;
		color: var(--color-text-primary);
	}

	.panel-header-secondary {
		margin-top: 6px;
	}

	.panel-section {
		margin-bottom: 8px;
		color: var(--color-text-secondary);
		line-height: 1.5;
	}

	.validation-toggle {
		margin-top: 6px;
		margin-bottom: 2px;
		width: 100%;
		display: flex;
		align-items: center;
		justify-content: space-between;
		border: none;
		background: transparent;
		color: var(--color-text-muted);
		font-size: var(--font-xxs);
		cursor: pointer;
		padding: 2px 0;
	}

	.validation-title {
		text-align: left;
	}

	.chevron {
		transition: transform 0.15s ease;
	}

	.chevron.open {
		transform: rotate(180deg);
	}

	.loading-text {
		color: var(--color-text-muted);
	}

	/* Comparison section styles */
	.comparison-section {
		margin-top: 8px;
		padding: 8px;
		background: var(--color-bg-panel-soft);
		border-radius: var(--radius-control);
	}

	.comparison-header {
		font-weight: 600;
		font-size: var(--font-xs);
		margin-bottom: 8px;
		color: var(--color-accent);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.comparison-table {
		width: 100%;
		border-collapse: collapse;
		font-size: var(--font-xxs);
	}

	.comparison-table th,
	.comparison-table td {
		padding: 4px 6px;
		text-align: right;
	}

	.comparison-table th:first-child,
	.comparison-table td:first-child {
		text-align: left;
	}

	.comparison-table th {
		font-weight: 600;
		color: var(--color-text-secondary);
		border-bottom: 1px solid var(--color-border-subtle);
		font-size: var(--font-xxs); /* Keep grid headers small */
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.comparison-table td {
		color: var(--color-text-primary);
		border-bottom: 1px solid rgba(148, 163, 184, 0.15);
	}

	.comparison-table tr:last-child td {
		border-bottom: none;
	}

	.comparison-table td.positive {
		color: #34d399; /* Green - improvement */
	}

	.comparison-table td.negative {
		color: #fb7185; /* Red - worse */
	}

	.comparison-note {
		margin-top: 6px;
		font-size: var(--font-xxs);
		color: var(--color-text-muted);
		font-style: italic;
	}
</style>
