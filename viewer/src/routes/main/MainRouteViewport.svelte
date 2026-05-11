<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { Group, Mesh, PerspectiveCamera } from 'three';

	import type { Analysis } from '$lib/types/analysis';
	import type { LiveSelectedHourControllerSurfaceDiagnostics } from '$lib/compute/liveSelectedHourController';
	import type { LiveSelectedHourPublishedRenderContext } from '$lib/compute/liveSelectedHourRenderContext';
	import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/liveSelectedHourSurfaceIdentity';
	import type { SelectedHourGpuResidentOutput } from '$lib/compute/liveUtciSelectedHourSession';
	import type {
		WebgpuLargeBufferDeviceLimits,
		WebgpuLargeBufferRequiredLimits
	} from '$lib/compute/webgpuDeviceLimits';
	import type { UtciSurfaceBackendType } from '$lib/services/pointCloudService';
	import type { UtciRendererBackend } from '$lib/utciRenderMode';
	import { resolveAnalysisModelPath } from '$lib/utils/analysisPaths';
	import Camera from '$lib/components/scene/Camera.svelte';
	import ComparisonRenderer from '$lib/components/scene/ComparisonRenderer.svelte';
	import GridHelper from '$lib/components/scene/GridHelper.svelte';
	import Lights from '$lib/components/scene/Lights.svelte';
	import Model from '$lib/components/scene/Model.svelte';
	import Scene from '$lib/components/scene/Scene.svelte';
	import UTCIPointCloud from '$lib/components/scene/UTCIPointCloud.svelte';

	import type { MainRouteAcceptedGpuResidentOutputReleaseParams } from './liveSelectedHour';

	type RendererDiagnostics = {
		rendererBackend: UtciRendererBackend;
		rendererDevice?: GPUDevice;
		rendererRequiredLimits?: WebgpuLargeBufferRequiredLimits;
		rendererDeviceLimits?: WebgpuLargeBufferDeviceLimits;
		error?: string;
	};

	const dispatch = createEventDispatcher<{
		modelLoaded: Group;
		layersDiscovered: string[];
	}>();

	export let analysis: Analysis | null;
	export let analysisId: string;
	export let dataBasePath: string;
	export let theme: 'light' | 'dark';
	export let requestLargeWebgpuLimits: boolean;
	export let cameraNear: number;
	export let cameraFar: number;
	export let gridVisible: boolean;
	export let model: Group | null;
	export let isComparing: boolean;
	export let baseSceneAnalysis: Analysis | null;
	export let comparisonSceneAnalysis: Analysis | null | undefined;
	export let basePendingGpuResidentOutput: SelectedHourGpuResidentOutput | null;
	export let comparisonPendingGpuResidentOutput: SelectedHourGpuResidentOutput | null;
	export let baseSceneRenderContext: LiveSelectedHourPublishedRenderContext | null;
	export let comparisonSceneRenderContext:
		| LiveSelectedHourPublishedRenderContext
		| null
		| undefined;
	export let baseSceneSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	export let comparisonSceneSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null | undefined;
	export let basePendingRenderUpdateStartedAt: number | undefined;
	export let comparisonPendingRenderUpdateStartedAt: number | undefined;
	export let resolvedUtciSurfaceBackend: UtciSurfaceBackendType;
	export let onRendererDiagnostics: (diagnostics: RendererDiagnostics) => void;
	export let onBaseUtciSurfaceDiagnostics: (
		diagnostics: LiveSelectedHourControllerSurfaceDiagnostics
	) => void;
	export let onComparisonUtciSurfaceDiagnostics: (
		diagnostics: LiveSelectedHourControllerSurfaceDiagnostics
	) => void;
	export let onBaseAcceptedGpuResidentOutputRelease: (
		params: MainRouteAcceptedGpuResidentOutputReleaseParams
	) => void;
	export let onComparisonAcceptedGpuResidentOutputRelease: (
		params: MainRouteAcceptedGpuResidentOutputReleaseParams
	) => void;

	export let canvasElement: HTMLCanvasElement | null = null;
	export let cameraRef: PerspectiveCamera | undefined = undefined;
	export let utciMesh: Mesh | null = null;
	export let comparisonRenderer: ComparisonRenderer;

	$: modelPath =
		analysis == null
			? null
			: resolveAnalysisModelPath(analysis.metadata, analysisId).replace(
					'data/',
					`${dataBasePath}/data/`
				);
</script>

{#key requestLargeWebgpuLimits}
	<Scene
		backgroundColor={theme === 'light' ? 0x4b5563 : 0x111827}
		bind:canvasElement
		{requestLargeWebgpuLimits}
		{onRendererDiagnostics}
	>
		<Camera bind:cameraRef near={cameraNear} far={cameraFar} />
		<Lights />

		{#if analysis && modelPath}
			{#key analysis.metadata.model_file}
				<Model
					{modelPath}
					coordinateSystem={analysis.metadata.coordinate_system || 'xy_ground'}
					metadata={analysis.metadata}
					on:modelLoaded={(event) => dispatch('modelLoaded', event.detail)}
					on:layersDiscovered={(event) => dispatch('layersDiscovered', event.detail)}
				/>
			{/key}

			{#if model}
				<GridHelper {model} visible={gridVisible} />
				<UTCIPointCloud
					analysis={baseSceneAnalysis}
					{model}
					bind:utciSurface={utciMesh}
					acceptedGpuResidentOutput={basePendingGpuResidentOutput}
					selectedHourRenderContext={baseSceneRenderContext}
					liveSelectedHourSurfaceIdentity={baseSceneSurfaceIdentity}
					onUtciSurfaceDiagnostics={onBaseUtciSurfaceDiagnostics}
					onAcceptedGpuResidentOutputRelease={onBaseAcceptedGpuResidentOutputRelease}
					pendingRenderUpdateStartedAt={basePendingRenderUpdateStartedAt}
					utciSurfaceBackend={resolvedUtciSurfaceBackend}
				/>
			{/if}

			{#if isComparing}
				<ComparisonRenderer
					bind:this={comparisonRenderer}
					acceptedGpuResidentOutput={comparisonPendingGpuResidentOutput}
					baseCamera={cameraRef}
					displayAnalysis={comparisonSceneAnalysis}
					selectedHourRenderContext={comparisonSceneRenderContext}
					liveSelectedHourSurfaceIdentity={comparisonSceneSurfaceIdentity}
					onUtciSurfaceDiagnostics={onComparisonUtciSurfaceDiagnostics}
					onAcceptedGpuResidentOutputRelease={onComparisonAcceptedGpuResidentOutputRelease}
					pendingRenderUpdateStartedAt={comparisonPendingRenderUpdateStartedAt}
					utciSurfaceBackend={resolvedUtciSurfaceBackend}
				/>
			{/if}
		{/if}
	</Scene>
{/key}
