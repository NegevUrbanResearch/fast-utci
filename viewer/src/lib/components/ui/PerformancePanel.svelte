<script lang="ts">
	import { performanceStore } from '$lib/stores/performanceStore';
	import {
		formatDuration,
		formatMemory
	} from '$lib/performance/mainRoutePerformanceTelemetry';

	$: snapshot = $performanceStore;
	$: statusLabel =
		snapshot.status === 'ready'
			? 'Ready'
			: snapshot.status === 'loading'
				? 'Preparing'
				: snapshot.status === 'fallback'
					? 'Preparing live result'
					: snapshot.status === 'error'
						? 'Needs attention'
						: 'Waiting';
</script>

<div class="performance-panel" data-testid="performance-panel">
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
		<span>Grid size</span>
		<strong>
			{snapshot.gridSizeMeters === null
				? 'Loading'
				: `${snapshot.gridSizeMeters} m${
						snapshot.pointCount === null ? '' : ` (${snapshot.pointCount.toLocaleString()} pts)`
					}`}
		</strong>
	</div>
	<div class="performance-status">{statusLabel}</div>
</div>

<style>
	.performance-panel {
		display: grid;
		gap: 6px;
		font-family: var(--font-family);
		font-size: var(--font-xs);
		color: var(--color-text-primary);
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

	.performance-status {
		padding-top: 2px;
		font-size: var(--font-xxs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--color-text-muted);
	}
</style>
