<script lang="ts">
	import { onMount } from 'svelte';
	import { analysisStore } from '$lib/stores/analysisStore';
	import { viewerStore } from '$lib/stores/viewerStore';
	import { loadValidationData, compareWithValidation, calculateAvgMeanDiffAllHours } from '$lib/services/validationService';
	import { getShadingIndex } from '$lib/services/dataLoader';
	import type { ComparisonStats } from '$lib/services/validationService';

	let comparisonStats: ComparisonStats | null = null;
	let avgMeanDiffAllHours: number | null = null;
	let validationLoaded = false;
	let showValidation = false;

	$: metadata = $analysisStore?.metadata;
	$: currentHour = $viewerStore.currentHour;
	$: metricType = $viewerStore.metricType;

	// Calculate UTCI statistics for current hour/selection
	$: currentUtciStats = calculateCurrentUtciStats($analysisStore, $viewerStore.currentHour);
	
	// Calculate Shading Index statistics
	$: shadingIndexStats = calculateShadingIndexStats($analysisStore);

	let lastAnalysisId: string | null = null;

	// Load validation data when analysis changes (use date or model_file as unique identifier)
	$: if (metadata && (metadata.date || metadata.model_file)) {
		const currentId = metadata.date || metadata.model_file || '';
		if (currentId !== lastAnalysisId) {
			lastAnalysisId = currentId;
			validationLoaded = false;
			comparisonStats = null;
			avgMeanDiffAllHours = null;
			loadValidation();
		}
	}

	// Update comparison stats when hour changes (if validation is loaded)
	$: if (comparisonStats && $analysisStore && validationLoaded) {
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
				comparisonStats = compareWithValidation($analysisStore, validation, $viewerStore.currentHour);
				if ($analysisStore.metadata.analysis_type === 'full_day') {
					avgMeanDiffAllHours = calculateAvgMeanDiffAllHours($analysisStore, validation);
				}
				validationLoaded = true;
			}
		} catch (error) {
			console.warn('[WARN] Could not load validation data:', error);
			validationLoaded = true; // Mark as loaded to prevent retry loops
		}
	}

	async function updateComparison(hourIndex: number) {
		if (!$analysisStore || !validationLoaded || !validationCache) return;
		
		try {
			comparisonStats = compareWithValidation($analysisStore, validationCache, hourIndex);
		} catch (error) {
			console.warn('[WARN] Could not update validation comparison:', error);
		}
	}

	/**
	 * Calculate UTCI statistics for the current hour/selection
	 * For single_hour: uses the single utciValues array
	 * For full_day: uses the utciByHour array for the selected hour
	 */
	function calculateCurrentUtciStats(analysis: typeof $analysisStore, hourIndex: number): { min: number; max: number; mean: number } | null {
		if (!analysis) return null;

		let utciValues: Float32Array;

		if (analysis.metadata.analysis_type === 'single_hour') {
			// Single hour analysis - use utciValues directly
			utciValues = (analysis.data as any).utciValues;
		} else {
			// Full day analysis - use utciByHour for selected hour
			const utciByHour = (analysis.data as any).utciByHour;
			if (!utciByHour || hourIndex < 0 || hourIndex >= utciByHour.length) {
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
	function calculateShadingIndexStats(analysis: typeof $analysisStore): { min: number; max: number; mean: number } | null {
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
</script>

{#if metadata}
	<div class="analytics-panel">
		<div class="panel-header">Analysis Info</div>
		
		<div class="panel-section">
			{#if metadata.date}
				<strong>Date:</strong> {metadata.date}<br />
			{/if}
			<strong>Grid Size:</strong> {metadata.grid_size}m<br />
			<strong>Positions:</strong> {metadata.num_positions.toLocaleString()}<br />
			{#if 'runtime_seconds' in metadata && typeof metadata.runtime_seconds === 'number'}
				<strong>Runtime:</strong> {metadata.runtime_seconds.toFixed(1)}s
			{/if}
		</div>
		
		<div class="panel-section">
			{#if metricType === 'shading_index'}
				<strong>Shading Index (Full Day):</strong><br />
				{#if shadingIndexStats}
					Min: {shadingIndexStats.min.toFixed(2)}<br />
					Max: {shadingIndexStats.max.toFixed(2)}<br />
					Mean: {shadingIndexStats.mean.toFixed(2)}
				{:else}
					<span class="loading-text">No Shading Index data available</span>
				{/if}
			{:else}
				<strong>Data (Hour {currentHour}):</strong><br />
				{#if currentUtciStats}
					Min: {currentUtciStats.min.toFixed(1)}°C<br />
					Max: {currentUtciStats.max.toFixed(1)}°C<br />
					Mean: {currentUtciStats.mean.toFixed(1)}°C
				{:else}
					<span class="loading-text">Loading...</span>
				{/if}
			{/if}
		</div>

		{#if comparisonStats}
			<button type="button" class="validation-toggle" on:click={() => (showValidation = !showValidation)}>
				<span class="validation-title">Validation vs Grasshopper</span>
				<span class:open={showValidation} class="chevron">▾</span>
			</button>

			{#if showValidation}
				<div class="panel-header panel-header-secondary">Grasshopper Comparison</div>
				<div class="panel-section">
					<strong>Validation Data:</strong><br />
					Min: {comparisonStats.validation.min.toFixed(1)}°C<br />
					Max: {comparisonStats.validation.max.toFixed(1)}°C<br />
					Mean: {comparisonStats.validation.mean.toFixed(1)}°C
				</div>
				<div class="panel-section">
					<strong>Comparison Metrics:</strong><br />
					Min Diff: {comparisonStats.comparison.minDiff >= 0 ? '+' : ''}{comparisonStats.comparison.minDiff.toFixed(2)}°C<br />
					Max Diff: {comparisonStats.comparison.maxDiff >= 0 ? '+' : ''}{comparisonStats.comparison.maxDiff.toFixed(2)}°C<br />
					Mean Diff: {comparisonStats.comparison.meanDiff >= 0 ? '+' : ''}{comparisonStats.comparison.meanDiff.toFixed(2)}°C
					{#if avgMeanDiffAllHours !== null}
						<br />24-Hour Avg: {avgMeanDiffAllHours >= 0 ? '+' : ''}{avgMeanDiffAllHours.toFixed(2)}°C
					{/if}
				</div>
			{/if}
		{/if}
	</div>
{/if}

<style>
	.analytics-panel {
		font-family: var(--font-family);
		font-size: 12px;
		color: var(--color-text-primary);
	}

	.panel-header {
		font-weight: 600;
		font-size: 13px;
		margin-bottom: 8px;
		border-bottom: 1px solid var(--color-border-subtle);
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
		font-size: 11px;
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
</style>

