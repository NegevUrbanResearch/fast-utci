<script lang="ts">
	import type { MetricType } from '$lib/types/viewer';
	import { getUTCILabel } from '$lib/services/colorScale';
	import { SHADING_INDEX_CATEGORIES } from '$lib/services/colorScale';

	export let visible: boolean = false;
	export let x: number = 0;
	export let y: number = 0;
	export let value: number | null = null;
	export let position: { x: number; y: number; z: number } | null = null;
	export let metricType: MetricType = 'utci';

	// Format value based on metric type
	$: formattedValue = value !== null ? formatValue(value, metricType) : null;
	$: formattedPosition = position ? formatPosition(position) : null;
	$: metricLabel = metricType === 'utci' ? 'UTCI' : 'Shading Index';
	$: description = value !== null ? getDescription(value, metricType) : null;

	function formatValue(val: number, type: MetricType): string {
		if (type === 'utci') {
			return val.toFixed(1);
		} else {
			// Shading Index: show 2 decimal places
			return val.toFixed(2);
		}
	}

	function getDescription(val: number, type: MetricType): string {
		if (type === 'utci') {
			return getUTCILabel(val, false);
		} else {
			// Shading Index: find matching category
			for (const category of SHADING_INDEX_CATEGORIES) {
				const [min, max] = category.range;
				// Handle edge case for last category (0.9-1.0) - include 1.0
				if (val >= min && (val < max || (val === 1.0 && max === 1.0))) {
					return category.label;
				}
			}
			return '';
		}
	}

	function formatPosition(pos: { x: number; y: number; z: number }): string {
		return `X ${pos.x.toFixed(3)} / Y ${pos.y.toFixed(3)} / Z ${pos.z.toFixed(3)}`;
	}
</script>

{#if visible && formattedValue !== null}
	<div
		class="tooltip"
		style="left: {x}px; top: {y}px;"
		role="tooltip"
		aria-live="polite"
	>
		<div class="tooltip-header">
			<span class="metric-label">{metricLabel}</span>
			{#if metricType === 'utci'}
				<span class="metric-unit">°C</span>
			{/if}
		</div>
		<div class="tooltip-value">{formattedValue}</div>
		{#if description}
			<div class="tooltip-description">{description}</div>
		{/if}
		{#if formattedPosition}
			<div class="tooltip-position">{formattedPosition}</div>
		{/if}
	</div>
{/if}

<style>
	.tooltip {
		position: fixed;
		pointer-events: none;
		z-index: var(--z-tooltip);
		background: var(--color-bg-panel);
		border-radius: var(--radius-panel);
		padding: var(--spacing-sm) var(--spacing-md);
		box-shadow: var(--shadow-tooltip);
		backdrop-filter: blur(12px);
		font-family: var(--font-family);
		font-size: 13px;
		line-height: 1.4;
		min-width: 120px;
		transform: translate(-50%, calc(-100% - 12px));
		opacity: 0;
		animation: tooltipFadeIn 0.15s ease-out forwards;
	}

	@keyframes tooltipFadeIn {
		from {
			opacity: 0;
			transform: translate(-50%, calc(-100% - 8px));
		}
		to {
			opacity: 1;
			transform: translate(-50%, calc(-100% - 12px));
		}
	}

	.tooltip-header {
		display: flex;
		align-items: baseline;
		gap: 4px;
		margin-bottom: 2px;
	}

	.metric-label {
		color: var(--color-text-secondary);
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		font-weight: 600;
	}

	.metric-unit {
		color: var(--color-text-muted);
		font-size: 10px;
	}

	.tooltip-value {
		color: var(--color-text-primary);
		font-size: 18px;
		font-weight: 600;
		line-height: 1.2;
		margin: 2px 0;
	}

	.tooltip-description {
		margin-top: 6px;
		padding-top: 6px;
		color: var(--color-text-secondary);
		font-size: 12px;
		font-weight: 500;
		line-height: 1.3;
	}

	.tooltip-position {
		margin-top: 6px;
		padding-top: 6px;
		border-top: 1px solid var(--color-border-subtle);
		color: var(--color-text-muted);
		font-size: 11px;
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}
</style>

