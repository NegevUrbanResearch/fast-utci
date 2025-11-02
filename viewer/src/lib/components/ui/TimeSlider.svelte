<script lang="ts">
	import { viewerStore, setCurrentHour } from '$lib/stores/viewerStore';
	import { analysisStore } from '$lib/stores/analysisStore';

	let sliderValue = $viewerStore.currentHour;

	$: if ($analysisStore && $analysisStore.metadata.analysis_type === 'full_day') {
		const hours = $analysisStore.metadata.hours;
		sliderValue = $viewerStore.currentHour;
	}

	function handleSliderChange(event: Event) {
		const target = event.target as HTMLInputElement;
		const hourIndex = parseInt(target.value, 10);
		setCurrentHour(hourIndex);
		sliderValue = hourIndex;
	}

	$: if ($analysisStore) {
		const hours = $analysisStore.metadata.hours;
		const currentHourLabel = hours[$viewerStore.currentHour] || `${$viewerStore.currentHour}:00`;
		const displayText = currentHourLabel;
	}
</script>

{#if $analysisStore && $analysisStore.metadata.analysis_type === 'full_day'}
	<div class="time-controls">
		<div class="time-display">
			<label for="hour-slider">Time: {$analysisStore.metadata.hours[$viewerStore.currentHour] || `${$viewerStore.currentHour}:00`}</label>
		</div>
		<input
			id="hour-slider"
			type="range"
			min="0"
			max={$analysisStore.metadata.hours.length - 1}
			value={$viewerStore.currentHour}
			on:input={handleSliderChange}
			class="slider"
		/>
	</div>
{/if}

<style>
	.time-controls {
		position: absolute;
		bottom: 20px;
		left: 50%;
		transform: translateX(-50%);
		background: var(--color-bg-panel);
		padding: var(--spacing-lg);
		border-radius: var(--radius-panel);
		box-shadow: var(--shadow-panel);
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		min-width: 300px;
		z-index: var(--z-panel);
	}

	.time-display {
		font-size: 14px;
		font-weight: 500;
		color: var(--color-text-primary);
		min-width: 80px;
	}

	.slider {
		flex: 1;
		height: 6px;
		border-radius: 3px;
		background: #ddd;
		outline: none;
		cursor: pointer;
	}

	.slider::-webkit-slider-thumb {
		appearance: none;
		width: 18px;
		height: 18px;
		border-radius: 50%;
		background: #3498db;
		cursor: pointer;
	}

	.slider::-moz-range-thumb {
		width: 18px;
		height: 18px;
		border-radius: 50%;
		background: #3498db;
		cursor: pointer;
		border: none;
	}
</style>


