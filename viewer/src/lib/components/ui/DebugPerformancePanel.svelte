<script lang="ts">
	import {
		buildDebugPerformanceComparisonRows,
		type DebugPerformanceComparisonDiagnosticsLike
	} from '$lib/performance/debugPerformanceComparison';

	export let diagnostics: DebugPerformanceComparisonDiagnosticsLike | null = null;

	$: rows = buildDebugPerformanceComparisonRows(diagnostics);
</script>

<div class="debug-performance-panel" data-testid="debug-performance-panel">
	<table>
		<thead>
			<tr>
				<th scope="col">Metric</th>
				<th scope="col">Python</th>
				<th scope="col">WebGPU</th>
				<th scope="col">Diff</th>
			</tr>
		</thead>
		<tbody>
			{#each rows as row (row.metric)}
				<tr>
					<th scope="row">{row.metric}</th>
					<td>{row.python}</td>
					<td>{row.webgpu}</td>
					<td>{row.diff}</td>
				</tr>
			{/each}
		</tbody>
	</table>
	<div class="debug-note">Debug comparison only</div>
</div>

<style>
	.debug-performance-panel {
		display: grid;
		gap: 6px;
		font-family: var(--font-family);
		font-size: var(--font-xxs);
		color: var(--color-text-primary);
	}

	table {
		width: 100%;
		border-collapse: collapse;
		table-layout: fixed;
	}

	thead {
		color: var(--color-text-muted);
	}

	th,
	td {
		padding: 4px 0;
		text-align: left;
		vertical-align: top;
		border-bottom: 1px solid var(--color-border-subtle);
	}

	thead th {
		font-size: var(--font-xxs);
		font-weight: 600;
	}

	tbody th {
		font-weight: 500;
		color: var(--color-text-secondary);
	}

	td {
		color: var(--color-text-primary);
		overflow-wrap: anywhere;
	}

	.debug-note {
		font-size: var(--font-xxs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--color-text-muted);
	}
</style>
