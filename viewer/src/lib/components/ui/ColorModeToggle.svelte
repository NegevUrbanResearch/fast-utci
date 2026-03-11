<script lang="ts">
	import { analysisStore } from "$lib/stores/analysisStore";
	import { viewerStore, setColorMode } from "$lib/stores/viewerStore";

	$: show =
		$analysisStore?.metadata.analysis_type === "full_day" &&
		$viewerStore.metricType === "utci";

	$: isFullDayMode = $viewerStore.colorMode === "normalized";
	$: isPerHourMode = $viewerStore.colorMode === "discrete";

	function selectFullDayMode() {
		if (!isFullDayMode) setColorMode("normalized");
	}

	function selectPerHourMode() {
		if (!isPerHourMode) setColorMode("discrete");
	}
</script>

{#if show}
	<div class="color-mode-section">
		<div class="mode-caption">
			<span>Color scale Mode</span>
		</div>
		<div
			class="mode-toggle-vertical"
			aria-label="Color scale mode"
			role="toolbar"
		>
			<button
				type="button"
				class="mode-pill-vertical"
				class:mode-pill-vertical-active={isFullDayMode}
				on:click={selectFullDayMode}
				aria-pressed={isFullDayMode}
			>
				<span class="mode-pill-label">Full day</span>
			</button>
			<button
				type="button"
				class="mode-pill-vertical"
				class:mode-pill-vertical-active={isPerHourMode}
				on:click={selectPerHourMode}
				aria-pressed={isPerHourMode}
			>
				<span class="mode-pill-label">Per hour</span>
			</button>
		</div>
		<div class="mode-help">
			Switch between the full‑day range and the range for the selected hour.
		</div>
	</div>
{/if}

<style>
	.color-mode-section {
		display: flex;
		flex-direction: column;
		align-items: stretch;
		gap: 8px;
		margin-top: var(--spacing-lg);
		padding-top: var(--spacing-lg);
		border-top: 1px solid rgba(148, 163, 184, 0.2);
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
	}
</style>
