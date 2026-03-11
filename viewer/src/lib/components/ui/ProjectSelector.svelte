<script lang="ts">
	import { projects } from "$lib/config/projects";
	import { resolveProjectId } from "$lib/utils/analysisPaths";

	export let analysisId: string;
	export let onSelect: (analysisId: string) => void;

	let selectedProjectId = projects[0]?.id ?? "";
	let selectedModelId = projects[0]?.models[0]?.id ?? "";

	function syncFromAnalysisId() {
		const projectId = resolveProjectId(analysisId) ?? projects[0]?.id;
		const project = projects.find((p) => p.id === projectId) ?? projects[0];
		selectedProjectId = project?.id ?? "";
		const model = project?.models.find((m) => m.analysisId === analysisId) ?? project?.models[0];
		selectedModelId = model?.id ?? "";
	}

	$: syncFromAnalysisId();

	function handleProjectChange(event: Event) {
		const target = event.target as HTMLSelectElement;
		selectedProjectId = target.value;
		const project = projects.find((p) => p.id === selectedProjectId);
		const model = project?.models[0];
		selectedModelId = model?.id ?? "";
		if (model && onSelect) {
			onSelect(model.analysisId);
		}
	}

</script>

<div class="project-selector">
	<label class="selector-label" for="project-select">Project</label>
	<select
		id="project-select"
		data-testid="project-select"
		class="selector-select"
		bind:value={selectedProjectId}
		on:change={handleProjectChange}
	>
		{#each projects as project}
			<option value={project.id}>{project.label}</option>
		{/each}
	</select>
</div>

<style>
	.project-selector {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: var(--font-xs);
		color: var(--color-text-primary);
	}

	.selector-label {
		font-size: var(--font-xxs);
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--color-text-secondary);
	}

	.selector-select {
		background: var(--color-bg-panel);
		border: 1px solid rgba(148, 163, 184, 0.3);
		border-radius: 8px;
		padding: 6px 10px;
		color: var(--color-text-primary);
		font-size: var(--font-xs);
		font-family: var(--font-family);
	}
</style>
