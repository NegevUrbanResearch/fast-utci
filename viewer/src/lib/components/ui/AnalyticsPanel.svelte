<script lang="ts">
	import { onMount } from 'svelte';
	import { analysisStore } from '$lib/stores/analysisStore';
	import { viewerStore } from '$lib/stores/viewerStore';
	import { loadValidationData, compareWithValidation, calculateAvgMeanDiffAllHours } from '$lib/services/validationService';
	import type { ComparisonStats } from '$lib/services/validationService';

	let comparisonStats: ComparisonStats | null = null;
	let avgMeanDiffAllHours: number | null = null;
	let validationLoaded = false;

	$: metadata = $analysisStore?.metadata;
	$: currentHour = $viewerStore.currentHour;

	// Calculate UTCI statistics for current hour/selection
	$: currentUtciStats = calculateCurrentUtciStats($analysisStore, $viewerStore.currentHour);

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
			<strong>Data (Hour {currentHour}):</strong><br />
			{#if currentUtciStats}
				Min: {currentUtciStats.min.toFixed(1)}°C<br />
				Max: {currentUtciStats.max.toFixed(1)}°C<br />
				Mean: {currentUtciStats.mean.toFixed(1)}°C
			{:else}
				<span style="color: #999;">Loading...</span>
			{/if}
		</div>

		{#if comparisonStats}
			<div class="panel-header">Grasshopper Comparison</div>
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
	</div>
{/if}

<style>
	.analytics-panel {
		position: absolute;
		top: 20px;
		right: 20px;
		background: rgba(255, 255, 255, 0.95);
		padding: 15px;
		border-radius: 8px;
		box-shadow: 0 2px 15px rgba(0, 0, 0, 0.3);
		font-family: Arial, sans-serif;
		font-size: 12px;
		max-width: 300px;
		z-index: 100;
	}

	.panel-header {
		font-weight: bold;
		font-size: 14px;
		margin-bottom: 10px;
		border-bottom: 2px solid #333;
		padding-bottom: 5px;
	}

	.panel-section {
		margin-bottom: 8px;
		color: #333;
		line-height: 1.5;
	}
</style>

