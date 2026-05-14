<script lang="ts">
	import { performanceStore } from '$lib/stores/performanceStore';
	import {
		formatDuration,
		formatMemory
	} from '$lib/performance/mainRoutePerformanceTelemetry';

	const GRID_RESOLUTION_STEPS = [10, 8, 6, 4, 2, 1, 0.5] as const;

	type Props = {
		selectedGridResolutionMeters?: number;
		onGridResolutionChange?: (value: number) => void;
	};

	let {
		selectedGridResolutionMeters = 2,
		onGridResolutionChange = () => undefined
	}: Props = $props();

	let snapshot = $derived($performanceStore);
	let selectedResolutionIndex = $derived(Math.max(
		0,
		GRID_RESOLUTION_STEPS.findIndex((value) => value === selectedGridResolutionMeters)
	));
	let selectedGridLabel = $derived(`${selectedGridResolutionMeters} m`);
	let pointCountLabel = $derived(
		snapshot.pointCount === null ? 'Loading' : `${snapshot.pointCount.toLocaleString()} pts`
	);

	function handleResolutionInput(event: Event): void {
		const slider = event.currentTarget as HTMLInputElement;
		const index = Number(slider.value);
		const resolution = GRID_RESOLUTION_STEPS[index];
		if (resolution !== undefined && resolution !== selectedGridResolutionMeters) {
			onGridResolutionChange(resolution);
		}
	}
</script>

<div class="performance-panel" data-testid="performance-panel">
	<div class="resolution-control">
		<div class="resolution-control-header">
			<label for="performance-grid-resolution">Resolution</label>
			<output for="performance-grid-resolution">{selectedGridLabel}</output>
		</div>
		<input
			id="performance-grid-resolution"
			data-testid="performance-grid-resolution-slider"
			type="range"
			min="0"
			max={GRID_RESOLUTION_STEPS.length - 1}
			step="1"
			value={selectedResolutionIndex}
			aria-label="Live UTCI grid resolution"
			oninput={handleResolutionInput}
		/>
	</div>
	<div class="metric-row metric-row-primary">
		<span>Total calculation time</span>
		<strong>{formatDuration(snapshot.totalToVisibleMs)}</strong>
	</div>
	<div class="metric-row">
		<span>UTCI calculation</span>
		<strong>{formatDuration(snapshot.utciComputeMs)}</strong>
	</div>
	<div class="metric-row">
		<span>GPU VRAM</span>
		<strong>{formatMemory(snapshot.ownedGpuMemoryBytes)}</strong>
	</div>
	<div class="metric-row">
		<span>Grid points</span>
		<strong>{pointCountLabel}</strong>
	</div>
</div>

<style>
	.performance-panel {
		display: grid;
		gap: 6px;
		font-family: var(--font-family);
		font-size: var(--font-xs);
		color: var(--color-text-primary);
	}

	.resolution-control {
		display: grid;
		gap: 4px;
		padding-bottom: 4px;
		border-bottom: 1px solid var(--color-border-subtle);
	}

	.resolution-control-header {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 8px;
	}

	.resolution-control label {
		color: var(--color-text-secondary);
	}

	.resolution-control output {
		font-size: var(--font-xs);
		font-weight: 600;
		color: var(--color-text-primary);
		text-align: right;
	}

	.resolution-control input {
		width: 100%;
		margin: 0;
		accent-color: var(--color-accent);
	}

	.metric-row {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 12px;
		padding: 4px 0;
		border-bottom: 1px solid var(--color-border-subtle);
	}

	.metric-row span {
		color: var(--color-text-secondary);
	}

	.metric-row strong {
		font-size: var(--font-xs);
		font-weight: 600;
		color: var(--color-text-primary);
		text-align: right;
	}

	.metric-row-primary strong {
		color: var(--color-accent);
	}
</style>
