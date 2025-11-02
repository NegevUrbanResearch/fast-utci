<script lang="ts">
	import { analysisStore } from '$lib/stores/analysisStore';
	import { viewerStore, setColorMode } from '$lib/stores/viewerStore';
	import { LADYBUG_NUANCED_COLORS } from '$lib/services/colorScale';

	let utciMin = 0;
	let utciMax = 100;

	$: if ($analysisStore) {
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
	}

	function handleColorModeToggle(event: Event) {
		const target = event.target as HTMLInputElement;
		setColorMode(target.checked ? 'discrete' : 'normalized');
	}

	// Create stepped gradient
	$: {
		const colors = [...LADYBUG_NUANCED_COLORS].reverse();
		const numColors = colors.length;
		const stepSize = 100 / numColors;
		const gradientStops: string[] = [];
		
		for (let i = 0; i < numColors; i++) {
			const startPercent = (i * stepSize).toFixed(2);
			const endPercent = ((i + 1) * stepSize).toFixed(2);
			gradientStops.push(`${colors[i]} ${startPercent}%`);
			gradientStops.push(`${colors[i]} ${endPercent}%`);
		}
		
		gradientStops.join(', ');
	}
</script>

{#if $analysisStore}
	<div class="color-legend">
		<div class="legend-header">
			<div class="title">UTCI</div>
			<div class="range">
				Range: {utciMin.toFixed(1)} - {utciMax.toFixed(1)}°C
				{#if $viewerStore.colorMode === 'discrete' && $analysisStore.metadata.analysis_type === 'full_day'}
					<span class="hour-label">(Hour {$analysisStore.metadata.hours[$viewerStore.currentHour]})</span>
				{/if}
			</div>
		</div>

		<div class="gradient-container">
			<div
				class="gradient"
				style="background: linear-gradient(to bottom, {[...LADYBUG_NUANCED_COLORS].reverse().map((color, i) => {
					const stepSize = 100 / LADYBUG_NUANCED_COLORS.length;
					const start = (i * stepSize).toFixed(2);
					const end = ((i + 1) * stepSize).toFixed(2);
					return `${color} ${start}%, ${color} ${end}%`;
				}).join(', ')})"
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
		</div>

		{#if $analysisStore.metadata.analysis_type === 'full_day'}
			<div class="color-mode-toggle">
				<span class="toggle-label">Color Scale</span>
				<div class="switch-container">
					<span class="switch-label">Full Day</span>
					<label class="switch">
						<input
							type="checkbox"
							checked={$viewerStore.colorMode === 'discrete'}
							on:change={handleColorModeToggle}
						/>
						<span class="slider"></span>
					</label>
					<span class="switch-label">Per Hour</span>
				</div>
			</div>
		{/if}
	</div>
{/if}

<style>
	.color-legend {
		position: absolute;
		bottom: 20px;
		right: 20px;
		background: var(--color-bg-panel);
		padding: var(--spacing-lg);
		border-radius: var(--radius-panel);
		box-shadow: var(--shadow-panel);
		z-index: var(--z-panel);
		min-width: 200px;
	}

	.legend-header {
		margin-bottom: var(--spacing-md);
	}

	.title {
		font-weight: bold;
		font-size: 15px;
		margin-bottom: var(--spacing-sm);
		color: var(--color-text-primary);
	}

	.range {
		font-size: 12px;
		color: var(--color-text-secondary);
	}

	.hour-label {
		font-size: 11px;
		color: var(--color-text-secondary);
	}

	.gradient-container {
		position: relative;
		display: inline-flex;
		align-items: stretch;
		gap: var(--spacing-md);
		margin-bottom: var(--spacing-md);
	}

	.gradient {
		width: 35px;
		height: 250px;
		border: 2px solid #666;
		border-radius: 5px;
		box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.1);
	}

	.labels {
		position: relative;
		height: 250px;
		width: 70px;
	}

	.label {
		position: absolute;
		left: 8px;
		transform: translateY(-50%);
		font-size: 12px;
		font-weight: 500;
		white-space: nowrap;
		color: var(--color-text-primary);
	}

	.color-mode-toggle {
		margin-top: var(--spacing-md);
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--spacing-md);
	}

	.toggle-label {
		font-size: 12px;
		color: var(--color-text-primary);
	}

	.switch-container {
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
	}

	.switch-label {
		font-size: 11px;
		color: var(--color-text-secondary);
	}

	.switch {
		position: relative;
		display: inline-block;
		width: 42px;
		height: 22px;
	}

	.switch input {
		opacity: 0;
		width: 0;
		height: 0;
	}

	.switch .slider {
		position: absolute;
		cursor: pointer;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		background-color: #ccc;
		transition: 0.4s;
		border-radius: 22px;
	}

	.switch .slider:before {
		position: absolute;
		content: '';
		height: 18px;
		width: 18px;
		left: 2px;
		bottom: 2px;
		background-color: white;
		transition: 0.4s;
		border-radius: 50%;
	}

	.switch input:checked + .slider {
		background-color: #3498db;
	}

	.switch input:checked + .slider:before {
		transform: translateX(20px);
	}
</style>


