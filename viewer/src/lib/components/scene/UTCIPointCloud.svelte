<script lang="ts">
	import { useThrelte } from '@threlte/core';
	import { onDestroy } from 'svelte';
	import type { Analysis } from '$lib/types/analysis';
	import type { SelectedHourGpuResidentOutput } from '$lib/compute/selected-hour/liveUtciSelectedHourSession';
	import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/selected-hour/liveSelectedHourSurfaceIdentity';
	import {
		resolveLiveSelectedHourSurfaceRenderState,
		type LiveSelectedHourPublishedRenderContext
	} from '$lib/compute/selected-hour/liveSelectedHourRenderContext';
	import {
		applySurfaceMeshState,
		buildUtciGridLayoutReuseProofDiagnostics,
		buildUtciGridLayout,
		createUtciLayoutReusePublicationState,
		createUtciLayoutReuseKeyForAnalysis,
		createUtciSurfaceMesh,
		disposeUtciSurfaceMesh,
		getUtciLayoutIdentity,
		planUtciLayoutPublication,
		resolveUtciLayoutReusePublicationStateAfterSync,
		type UtciGridLayout,
		type UtciGridLayoutReuseProofDiagnostics,
		type UtciLayoutReuseKey,
		type UtciLayoutReuseKeyDiagnostics,
		type UtciLayoutReusePublicationState,
		type UtciSurfaceBackendType,
		updateUtciSurfaceMesh
	} from '$lib/services/pointCloudService';
	import {
		createComputeBufferUtciSurfaceMesh,
		evaluateComputeBufferUtciSurfaceLayoutCompatibility,
		getComputeBufferUtciSurfaceLayoutCompatibilityState,
		getComputeBufferUtciStorageAttribute,
		updateComputeBufferUtciSurfaceMesh
	} from '$lib/services/gpuUtciRenderBridge';
	import { viewerStore } from '$lib/stores/viewerStore';
	import { unifiedUtciRange } from '$lib/stores/comparisonStore';
	import type { UnifiedUtciRange } from '$lib/stores/comparisonStore';
	import {
		invokeDiagnosticsCallbackSafely,
		type SelectedHourRenderTimingSubsteps
	} from '$lib/compute/on-demand/onDemandDiagnostics';
	import {
		createRenderPublicationDiagnostics,
		type SelectedHourRenderLayoutBuildTrace,
		type SelectedHourRenderLayoutReuseProofTrace,
		type SelectedHourRenderSurfaceMeshTrace
	} from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';
	import {
		buildCpuPublicationDiagnostics,
		buildUtciSurfaceDiagnostics,
		getAcceptedGpuResidentKey,
		isComputeBufferUtciSurface,
		shouldRecreateComputeBufferUtciSurface,
		type GpuResidentCopyStatus,
		type UtciSurfaceDiagnostics
	} from '$lib/components/scene/utciSurfaceSync';
	import {
		type AcceptedGpuResidentOutputReleaseCallback
	} from '$lib/components/scene/acceptedGpuResidentOutputRelease';
	import {
		createAcceptedGpuResidentSurfaceSync,
		type AcceptedGpuResidentSurfaceSyncRun,
		type AcceptedGpuResidentSurfaceSyncTerminalResult
	} from '$lib/components/scene/acceptedGpuResidentSurfaceSync';
	import {
		copyComputeBufferToRenderStorage,
		waitForRenderStorageBuffer
	} from './utciComputeBufferRenderBridge';
	import type { Group, Mesh } from 'three';

	export let analysis: Analysis | null = null;
	export let model: Group | null = null;
	export let utciSurfaceBackend: UtciSurfaceBackendType = 'dataTexture';
	export let acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null = null;
	export let liveSelectedHourSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null = null;
	export let selectedHourRenderContext:
		| LiveSelectedHourPublishedRenderContext
		| null
		| undefined = undefined;
	export let rangeOverride: UnifiedUtciRange | null | undefined = undefined;
	export let pendingRenderUpdateStartedAt: number | undefined = undefined;
	export let onUtciSurfaceDiagnostics:
		| ((diagnostics: UtciSurfaceDiagnostics) => void | Promise<void>)
		| undefined = undefined;
	export let onAcceptedGpuResidentOutputRelease:
		| AcceptedGpuResidentOutputReleaseCallback
		| undefined = undefined;
	export let utciSurface: Mesh | null = null;
	let lastAnalysis: Analysis | null = null;
	let lastBackend: UtciSurfaceBackendType | null = null;
	let gpuResidentCopyStatus: GpuResidentCopyStatus = 'idle';
	let gpuResidentCopyError: string | undefined = undefined;
	let gpuResidentCopyRequestId: number | undefined = undefined;
	let gpuResidentRenderTimings: SelectedHourRenderTimingSubsteps | undefined = undefined;
	let activeUtciLayoutReuseState: UtciLayoutReusePublicationState | null = null;
	// Observation is per sync key; attempt timing/token below are per startSync retry.
	let lastObservedPendingSurface: {
		syncRunKey: string;
		observedAtMs: number;
	} | null = null;
	const { renderer, scene, invalidate } = useThrelte();
	$: void model;

	function setComputeBufferSurfacePendingStorageInit(mesh: Mesh): void {
		mesh.visible = true;
		mesh.scale.setScalar(0);
	}

	function setComputeBufferSurfacePublicationVisibility(
		mesh: Mesh,
		shouldBeVisible: boolean
	): void {
		mesh.scale.setScalar(1);
		mesh.visible = shouldBeVisible;
	}

	function disposeUtciSurface(
		options: {
			invalidateGpuResidentCopies?: boolean;
			preserveActiveSyncRunKey?: string | null;
		} = {}
	) {
		if (options.invalidateGpuResidentCopies === false) {
			acceptedGpuResidentSurfaceSync.resetUnlessActiveSyncRunKeyMatches({
				expectedActiveSyncRunKey: options.preserveActiveSyncRunKey ?? null,
				invalidateActiveRun: false,
				reason: 'compute-surface-recreation'
			});
		} else {
			acceptedGpuResidentSurfaceSync.reset({
				invalidateActiveRun: true,
				reason: 'dispose-utci-surface'
			});
		}
		if (utciSurface) {
			utciSurface.parent?.remove(utciSurface);
			disposeUtciSurfaceMesh(utciSurface);
			utciSurface = null;
		}
		activeUtciLayoutReuseState = null;
		lastAnalysis = null;
		lastBackend = null;
		invokeUtciSurfaceDiagnostics({ renderOwnedSelectedHourBytes: 0 });
	}

	function invokeUtciSurfaceDiagnostics(diagnostics: UtciSurfaceDiagnostics): void {
		invokeDiagnosticsCallbackSafely(
			onUtciSurfaceDiagnostics,
			diagnostics,
			'UTCIPointCloud onUtciSurfaceDiagnostics'
		);
	}

	function publishUtciSurfaceDiagnostics(): void {
		const cpuPublicationDiagnostics = buildCpuPublicationDiagnostics({
			mesh: utciSurface,
			liveSelectedHourSurfaceIdentity
		});
		invokeUtciSurfaceDiagnostics(buildUtciSurfaceDiagnostics({
			mesh: utciSurface,
			cpuPublicationDiagnostics,
			gpuResidentCopyStatus,
			gpuResidentCopyError,
			gpuResidentCopyRequestId,
			gpuResidentRenderTimings
		}));
	}

	function getSceneSyncActiveWindowResetHistory(params: {
		scenePendingSurfaceObservedAtMs: number | undefined;
		sceneSyncAttemptStartedAtMs: number;
	}) {
		const { scenePendingSurfaceObservedAtMs, sceneSyncAttemptStartedAtMs } = params;
		if (scenePendingSurfaceObservedAtMs === undefined) {
			return [];
		}
		return acceptedGpuResidentSurfaceSync.getResetHistory().filter((event) => {
			return (
				event.resetAtMs >= scenePendingSurfaceObservedAtMs &&
				event.resetAtMs <= sceneSyncAttemptStartedAtMs
			);
		});
	}

	function setGpuResidentCopyDiagnostics(
		status: GpuResidentCopyStatus,
		options?: {
			error?: string;
			requestId?: number;
			renderTimings?: SelectedHourRenderTimingSubsteps;
		}
	): void {
		gpuResidentCopyStatus = status;
		gpuResidentCopyError = options?.error;
		gpuResidentCopyRequestId = options?.requestId;
		gpuResidentRenderTimings = status === 'complete' ? options?.renderTimings : undefined;
		publishUtciSurfaceDiagnostics();
	}

	const acceptedGpuResidentSurfaceSync =
		createAcceptedGpuResidentSurfaceSync({
			componentName: 'UTCIPointCloud',
			getOnAcceptedGpuResidentOutputRelease: () =>
				onAcceptedGpuResidentOutputRelease,
			setCopyDiagnostics: setGpuResidentCopyDiagnostics
		});

	function getCurrentGpuResidentSyncLiveState() {
		return {
			acceptedGpuResidentOutput,
			liveSelectedHourSurfaceIdentity
		};
	}

	function waitForNextFrame(): Promise<void> {
		return new Promise((resolve) => requestAnimationFrame(() => resolve()));
	}

	function buildSurfaceOptions(renderState: NonNullable<ReturnType<typeof resolveLiveSelectedHourSurfaceRenderState>>) {
		return {
			analysis: renderState.analysis,
			hourIndex: renderState.hourIndex,
			colorMode: renderState.colorMode,
			metricType: renderState.metricType,
			rangeOverride: renderState.rangeOverride,
			monthIndex: renderState.monthIndex,
			backend: utciSurfaceBackend
		} as const;
	}

	function recreateUtciSurface(
		renderState: NonNullable<ReturnType<typeof resolveLiveSelectedHourSurfaceRenderState>>
	): void {
		disposeUtciSurface();
		utciSurface = createUtciSurfaceMesh(buildSurfaceOptions(renderState));
		scene.add(utciSurface);
		lastAnalysis = renderState.analysis;
		lastBackend = utciSurfaceBackend;
		publishUtciSurfaceDiagnostics();
	}

	function extractUtciLayout(
		activeAnalysis: Analysis,
		layoutBuildTrace?: SelectedHourRenderLayoutBuildTrace
	) {
		const layout = buildUtciGridLayout(activeAnalysis, {
			diagnostics: layoutBuildTrace
		});
		if (!layout) {
			throw new Error('UTCI surface layout was unavailable for compute-buffer rendering.');
		}
		return layout;
	}

	function addSurfaceTraceTiming(
		trace: SelectedHourRenderSurfaceMeshTrace,
		key: Exclude<
			keyof SelectedHourRenderSurfaceMeshTrace,
			'action' | 'totalMs' | 'recreateDecision'
		>,
		durationMs: number
	): void {
		trace[key] = (trace[key] ?? 0) + durationMs;
	}

	function recreateComputeBufferSurface(
		activeAnalysis: Analysis,
		acceptedOutput: SelectedHourGpuResidentOutput,
		layout: ReturnType<typeof extractUtciLayout>,
		trace?: SelectedHourRenderSurfaceMeshTrace
	): void {
		const sourceBuffer = acceptedOutput.output.gpuBuffer as GPUBuffer | undefined;
		if (!sourceBuffer) {
			throw new Error('Accepted GPU-resident UTCI output is missing its GPUBuffer handle.');
		}

		const disposeStartedAt = performance.now();
		disposeUtciSurface({
			invalidateGpuResidentCopies: false,
			preserveActiveSyncRunKey: acceptedGpuResidentSurfaceSync.getSyncRunKey({
				acceptedGpuResidentOutput: acceptedOutput,
				liveSelectedHourSurfaceIdentity
			})
		});
		if (trace) {
			addSurfaceTraceTiming(
				trace,
				'disposeResetMeshRemovalMs',
				performance.now() - disposeStartedAt
			);
		}

		const createStartedAt = performance.now();
		utciSurface = createComputeBufferUtciSurfaceMesh({
			layout,
			utciBuffer: sourceBuffer,
			utciRange: acceptedOutput.utciRange,
			metricType: acceptedOutput.metricType ?? 'utci',
			trace,
			now: performance.now.bind(performance)
		});
		if (trace) {
			addSurfaceTraceTiming(
				trace,
				'createComputeBufferSurfaceMeshMs',
				performance.now() - createStartedAt
			);
		}

		const applyStartedAt = performance.now();
		applySurfaceMeshState(utciSurface, layout, 'gpuNative');
		if (trace) {
			addSurfaceTraceTiming(
				trace,
				'applySurfaceMeshStateMs',
				performance.now() - applyStartedAt
			);
		}

		const pendingStartedAt = performance.now();
		setComputeBufferSurfacePendingStorageInit(utciSurface);
		if (trace) {
			addSurfaceTraceTiming(
				trace,
				'setCreatedSurfacePendingStorageInitMs',
				performance.now() - pendingStartedAt
			);
		}

		const sceneAddStartedAt = performance.now();
		scene.add(utciSurface);
		if (trace) {
			addSurfaceTraceTiming(trace, 'sceneAddMs', performance.now() - sceneAddStartedAt);
		}
		lastAnalysis = activeAnalysis;
		lastBackend = utciSurfaceBackend;
		gpuResidentCopyError = undefined;
		const diagnosticsStartedAt = performance.now();
		publishUtciSurfaceDiagnostics();
		if (trace) {
			addSurfaceTraceTiming(
				trace,
				'publishUtciSurfaceDiagnosticsMs',
				performance.now() - diagnosticsStartedAt
			);
		}
	}

	async function copyComputeBufferIntoRenderOwnedStorage(params: {
		mesh: Mesh;
		acceptedOutput: SelectedHourGpuResidentOutput;
		activeSyncRun: AcceptedGpuResidentSurfaceSyncRun;
		syncStartedAt: number;
		scenePendingSurfaceObservedAtMs: number | undefined;
		sceneSyncAttemptStartedAtMs: number;
		sceneSyncAttemptToken: number;
		sceneSurfaceReceivedAtMs: number;
		publicationEffectStartedAtMs: number;
		reactiveTiming?: {
			sceneReactiveBlockEnteredAtMs?: number;
			sceneRenderStateResolvedAtMs?: number;
			sceneAcceptedKeyResolvedAtMs?: number;
			sceneSyncInvocationQueuedAtMs?: number;
			sceneStartSyncEnteredAtMs?: number;
			sceneStartSyncReturnedAtMs?: number;
		};
		sceneSurfacePendingStorageInitAtMs: number | undefined;
		renderTimings: SelectedHourRenderTimingSubsteps;
		meshAction: 'created' | 'reused';
		layout: ReturnType<typeof extractUtciLayout>;
		lastRenderTargetByteLength: { value: number | undefined };
	}): Promise<AcceptedGpuResidentSurfaceSyncTerminalResult> {
		const {
			mesh,
			acceptedOutput,
			activeSyncRun,
			syncStartedAt,
			scenePendingSurfaceObservedAtMs,
			sceneSyncAttemptStartedAtMs,
			sceneSyncAttemptToken,
			sceneSurfaceReceivedAtMs,
			publicationEffectStartedAtMs,
			reactiveTiming,
			sceneSurfacePendingStorageInitAtMs,
			renderTimings,
			meshAction,
			layout,
			lastRenderTargetByteLength
		} = params;
		const sourceBuffer = acceptedOutput.output.gpuBuffer as GPUBuffer | undefined;
		if (!sourceBuffer) {
			throw new Error('Accepted GPU-resident UTCI output is missing its GPUBuffer handle.');
		}

		const storageAttribute = getComputeBufferUtciStorageAttribute(mesh);
		if (!storageAttribute) {
			throw new Error('Compute-buffer UTCI storage attribute was not available.');
		}
		let lastDevice: GPUDevice | undefined;
		let renderStorageInvalidateRequestedAtMs: number | undefined;
		let renderStorageFirstWaitFrameRequestedAtMs: number | undefined;
		let renderStorageFirstWaitFrameCompletedAtMs: number | undefined;
		const isSuperseded = () =>
			acceptedGpuResidentSurfaceSync.isSuperseded(
				activeSyncRun,
				getCurrentGpuResidentSyncLiveState()
			);
		const { device, targetBuffer, waitMs, waitTrace } = await waitForRenderStorageBuffer({
			deadlineMs: 1000,
			now: performance.now.bind(performance),
			waitForNextFrame: async () => {
				renderStorageInvalidateRequestedAtMs ??= performance.now();
				invalidate();
				renderStorageFirstWaitFrameRequestedAtMs ??= performance.now();
				await waitForNextFrame();
				renderStorageFirstWaitFrameCompletedAtMs ??= performance.now();
			},
			isSuperseded,
			collectDiagnostics: true,
			readStorageState: () => {
				const rendererBackend = (
					renderer as unknown as {
						backend?: {
							device?: GPUDevice;
							get?: (resource: unknown) => { buffer?: GPUBuffer } | undefined;
						};
					}
				).backend;
				const backendEntry = rendererBackend?.get?.(storageAttribute);
				const device = rendererBackend?.device;
				const targetBuffer = backendEntry?.buffer;
				lastDevice = device;
				return {
					device,
					backendEntryAvailable: Boolean(backendEntry),
					targetBuffer
				};
			},
			readStorageBuffer: () => {
				const rendererBackend = (
					renderer as unknown as {
						backend?: {
							device?: GPUDevice;
							get?: (resource: unknown) => { buffer?: GPUBuffer } | undefined;
						};
					}
				).backend;
				const device = rendererBackend?.device;
				const targetBuffer = rendererBackend?.get?.(storageAttribute)?.buffer;
				lastDevice = device;
				return device && targetBuffer ? { device, targetBuffer } : null;
			},
			getTimeoutErrorMessage: () =>
				lastDevice
					? 'Three storage buffer was not initialized within the GPU-resident render timeout.'
					: 'Renderer WebGPU device was not available within the GPU-resident render timeout.'
		});
		lastRenderTargetByteLength.value = targetBuffer.size;
		const renderStorageReadyAtMs = performance.now();
		const renderStorageWaitStartedAtMs =
			waitTrace?.waitStartedAtMs ?? renderStorageReadyAtMs;
		const renderStoragePreWaitMs =
			typeof renderStorageWaitStartedAtMs === 'number'
				? Math.max(0, renderStorageWaitStartedAtMs - sceneSyncAttemptStartedAtMs)
				: undefined;
		renderTimings.renderStorageInitWaitMs = waitMs;
		if (isSuperseded()) {
			return acceptedGpuResidentSurfaceSync.supersedeSync(activeSyncRun);
		}

		const copyTimings = await copyComputeBufferToRenderStorage({
			device,
			queue: device.queue,
			sourceBuffer,
			targetBuffer,
			byteLength: sourceBuffer.size,
			now: performance.now.bind(performance),
			isSuperseded
		});
		renderTimings.renderBufferCopyMs = copyTimings.bufferCopyMs;
		renderTimings.renderQueueDrainMs = copyTimings.queueDrainMs;
		if (isSuperseded()) {
			return acceptedGpuResidentSurfaceSync.supersedeSync(activeSyncRun);
		}

		mesh.userData.utciSurfaceSource = 'compute-buffer-selected-hour';
		mesh.userData.selectedHourTransferCount = 0;
		mesh.userData.dataTextureBuildCount = 0;
		setComputeBufferSurfacePublicationVisibility(
			mesh,
			Boolean(analysis && $viewerStore?.utciVisible)
		);
		renderTimings.renderSceneSyncTotalMs = performance.now() - syncStartedAt;
		const sceneSyncCompletedAtMs = performance.now();
		const positionVertexCount = mesh.geometry.getAttribute('position')?.count ?? null;
		const indexCount = mesh.geometry.index?.count ?? null;
		const currentPublicationPhase =
			selectedHourRenderContext?.publicationPhase ??
			(activeUtciLayoutReuseState != null ? 'scrub' : 'initial');
		renderTimings.renderPublication = createRenderPublicationDiagnostics({
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: currentPublicationPhase,
			renderPublicationMeshAction: meshAction,
			renderPublicationPointCount: layout.numPositions,
			renderPublicationVertexCount: positionVertexCount ?? undefined,
			renderPublicationIndexCount: indexCount ?? undefined,
			renderPublicationDrawIndexCount: indexCount ?? positionVertexCount ?? undefined,
			renderPublicationGridWidth: layout.width,
			renderPublicationGridHeight: layout.height,
			renderPublicationGridSize: layout.gridSize,
			renderPublicationSourceByteLength: sourceBuffer.size,
			renderPublicationTargetByteLength: lastRenderTargetByteLength.value,
			renderPublicationRenderOwnedBytes:
				mesh.userData.renderOwnedSelectedHourBytes as number | undefined,
			renderPublicationTimeline: {
					scenePendingSurfaceObservedAtMs,
					sceneReactiveBlockEnteredAtMs:
						reactiveTiming?.sceneReactiveBlockEnteredAtMs,
					sceneRenderStateResolvedAtMs:
						reactiveTiming?.sceneRenderStateResolvedAtMs,
					sceneAcceptedKeyResolvedAtMs:
						reactiveTiming?.sceneAcceptedKeyResolvedAtMs,
					sceneSyncInvocationQueuedAtMs:
						reactiveTiming?.sceneSyncInvocationQueuedAtMs,
					sceneReactiveToSyncQueuedMs:
						reactiveTiming?.sceneSyncInvocationQueuedAtMs != null &&
						reactiveTiming.sceneReactiveBlockEnteredAtMs != null
							? Math.max(
									0,
									reactiveTiming.sceneSyncInvocationQueuedAtMs -
										reactiveTiming.sceneReactiveBlockEnteredAtMs
								)
							: undefined,
					sceneSyncQueuedToStartMs:
						reactiveTiming?.sceneSyncInvocationQueuedAtMs != null
							? Math.max(
									0,
									sceneSyncAttemptStartedAtMs -
										reactiveTiming.sceneSyncInvocationQueuedAtMs
								)
							: undefined,
					sceneStartSyncEnteredAtMs:
						reactiveTiming?.sceneStartSyncEnteredAtMs,
					sceneStartSyncReturnedAtMs:
						reactiveTiming?.sceneStartSyncReturnedAtMs,
					sceneSyncAttemptStartedAtMs,
					sceneSyncAttemptToken,
					sceneSurfaceReceivedAtMs,
					publicationEffectStartedAtMs,
					sceneLayoutKeyStartedAtMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.sceneLayoutKeyStartedAtMs,
					sceneLayoutKeyCompletedAtMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.sceneLayoutKeyCompletedAtMs,
					scenePublicationPlanReadyAtMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.scenePublicationPlanReadyAtMs,
					renderLayoutBuildTrace:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutBuildTrace,
					renderLayoutReuseProofTrace:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseProofTrace,
					renderLayoutReuseAction:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseAction,
					renderLayoutReuseReason:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseReason,
					renderLayoutReuseDecisionMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseDecisionMs,
					renderLayoutReuseKeyMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseKeyMs,
					renderLayoutReuseSourceSignatureMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseSourceSignatureMs,
					renderLayoutReusePositionsSignatureMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReusePositionsSignatureMs,
					renderLayoutReusePositionsSignatureCacheHit:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReusePositionsSignatureCacheHit,
					renderLayoutReuseFrameCacheLookupMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseFrameCacheLookupMs,
					renderLayoutReuseFrameDerivationMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseFrameDerivationMs,
					renderLayoutReuseFrameCacheHit:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseFrameCacheHit,
					renderLayoutReuseFrameCacheKind:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseFrameCacheKind,
					renderLayoutPublicationPlanMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutPublicationPlanMs,
					renderLayoutCompatibilityMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutCompatibilityMs,
					renderLayoutCompatibilityRequiredExpensiveMappingComparison:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutCompatibilityRequiredExpensiveMappingComparison,
					renderLayoutCompatibilityPerformedExpensiveMappingComparison:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutCompatibilityPerformedExpensiveMappingComparison,
					renderLayoutReuseProofMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseProofMs,
					renderLayoutReuseKeyMatch:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseKeyMatch,
					renderLayoutReuseProofSource:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReuseProofSource,
					renderLayoutReusePreviousKey:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReusePreviousKey,
					renderLayoutReusePreviousRequestId:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReusePreviousRequestId,
					renderLayoutReusePreviousSelectionKey:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderLayoutReusePreviousSelectionKey,
					activeLayoutCandidateCount:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.activeLayoutCandidateCount,
					renderSurfaceMeshTrace:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderSurfaceMeshTrace,
					sceneSurfacePendingStorageInitAtMs,
					renderPublicationPreStorageStartedAtMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderPublicationPreStorageStartedAtMs,
					renderPublicationPreStorageCompletedAtMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderPublicationPreStorageCompletedAtMs,
					renderPublicationPreStorageMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderPublicationPreStorageMs,
					renderStoragePendingFlagStartedAtMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderStoragePendingFlagStartedAtMs,
					renderStoragePendingFlagCompletedAtMs:
						renderTimings.renderPublication?.renderPublicationTimeline
							?.renderStoragePendingFlagCompletedAtMs,
					renderStorageInvalidateRequestedAtMs,
					renderStorageFirstWaitFrameRequestedAtMs,
					renderStorageFirstWaitFrameCompletedAtMs,
					renderStorageWaitStartedAtMs,
					renderStoragePreWaitMs,
					renderStorageReadyAtMs,
					renderStorageWaitTrace: waitTrace,
					renderBufferCopyEncoderCreateMs: copyTimings.copyEncoderCreateMs,
					renderBufferCopyCommandRecordMs: copyTimings.copyCommandRecordMs,
					renderBufferCopySubmitMs: copyTimings.copySubmitMs,
					renderCopyQueueDrainStartedAtMs: copyTimings.queueDrainStartedAtMs,
					renderCopyQueueDrainCompletedAtMs:
						copyTimings.queueDrainCompletedAtMs,
					renderCopyQueueDrainMs: copyTimings.queueDrainMs,
					sceneSyncCompletedAtMs,
				sceneSyncResetHistory:
					acceptedGpuResidentSurfaceSync.getResetHistory(),
				sceneSyncActiveWindowResetHistory:
					getSceneSyncActiveWindowResetHistory({
						scenePendingSurfaceObservedAtMs,
						sceneSyncAttemptStartedAtMs
					})
			}
		});
		const syncResult = acceptedGpuResidentSurfaceSync.completeSync(activeSyncRun, {
			...getCurrentGpuResidentSyncLiveState(),
			renderTimings
		});
		if (syncResult !== 'complete') {
			return syncResult;
		}
		invalidate();
		return syncResult;
	}

	async function syncAcceptedGpuResidentSurface(
		activeAnalysis: Analysis,
		acceptedOutput: SelectedHourGpuResidentOutput,
		sceneSurfaceReceivedAtMs: number,
		scenePendingSurfaceObservedAtMs?: number,
		reactiveTiming?: {
			sceneReactiveBlockEnteredAtMs?: number;
			sceneRenderStateResolvedAtMs?: number;
			sceneAcceptedKeyResolvedAtMs?: number;
			sceneSyncInvocationQueuedAtMs?: number;
		}
	): Promise<void> {
		const sceneStartSyncEnteredAtMs = performance.now();
		const activeSyncRun =
			acceptedGpuResidentSurfaceSync.startSync({
				acceptedOutput,
				liveSelectedHourSurfaceIdentity
			});
		const sceneStartSyncReturnedAtMs = performance.now();
		if (!activeSyncRun) {
			return;
		}

		try {
			const sceneSyncAttemptStartedAtMs = performance.now();
			const sceneSyncAttemptToken = activeSyncRun.copyRunToken;
			const publicationEffectStartedAtMs = performance.now();
			const syncStartedAt = sceneSyncAttemptStartedAtMs;
			const renderTimings: SelectedHourRenderTimingSubsteps = {};
			if (pendingRenderUpdateStartedAt !== undefined) {
				renderTimings.renderSceneSyncStartDelayMs = Math.max(
					0,
					syncStartedAt - pendingRenderUpdateStartedAt
				);
			}
			const previousLayout = (utciSurface?.userData.utciLayout as UtciGridLayout | undefined) ?? null;
			const previousReuseState = activeUtciLayoutReuseState;
			const previousProof = activeUtciLayoutReuseState?.proof ?? null;
			const previousKey = activeUtciLayoutReuseState?.key ?? null;
			const layoutReuseKeyDiagnostics: UtciLayoutReuseKeyDiagnostics = {};
			const renderPublicationPreStorageStartedAtMs = performance.now();
			const sceneLayoutKeyStartedAtMs = performance.now();
			const layoutReuseDecisionStartedAt = performance.now();
			const currentKey = createUtciLayoutReuseKeyForAnalysis({
				analysis: activeAnalysis,
				utciSurfaceSource: 'compute-buffer-selected-hour',
				rendererBackend: 'webgpu',
				diagnostics: layoutReuseKeyDiagnostics
			});
			const sceneLayoutKeyCompletedAtMs = performance.now();
			const publicationPhase =
				selectedHourRenderContext?.publicationPhase ??
				(previousReuseState != null ? 'scrub' : 'initial');
			const layoutPublicationPlanStartedAt = performance.now();
			let layoutPublicationPlan = planUtciLayoutPublication({
				previousLayout,
				previousProof,
				previousKey,
				currentKey,
				currentSurfaceSource:
					(utciSurface?.userData.utciSurfaceSource as string | undefined) ?? null,
				currentRendererBackend: 'webgpu',
				publicationPhase
			});
			let runtimeLayoutCompatibility:
				ReturnType<typeof evaluateComputeBufferUtciSurfaceLayoutCompatibility> | null = null;
			let acceptedRefreshedLayoutReuseProof: SelectedHourRenderLayoutReuseProofTrace | null = null;
			let refreshedLayoutCompatibilityMs = 0;
			let refreshedLayoutProofMs = 0;
			if (
				layoutPublicationPlan.action === 'build-new' &&
				layoutPublicationPlan.reason === 'canonical-mismatch' &&
				previousLayout
			) {
				const refreshCompatibilityStartedAt = performance.now();
				const refreshLayoutCompatibility = evaluateComputeBufferUtciSurfaceLayoutCompatibility({
					state: getComputeBufferUtciSurfaceLayoutCompatibilityState(utciSurface),
					previousLayout,
					nextLayout: previousLayout,
					metricType: acceptedOutput.metricType ?? 'utci',
					allowExpensiveMappingComparison: true
				});
				refreshedLayoutCompatibilityMs =
					performance.now() - refreshCompatibilityStartedAt;
				const refreshProofStartedAt = performance.now();
				const refreshedLayoutReuseProof = buildUtciGridLayoutReuseProofDiagnostics({
					previousLayout,
					nextLayout: previousLayout,
					canonicalRuntimeCompatibilityWouldReuse:
						refreshLayoutCompatibility.compatible ?? false,
					canonicalPointCompatibility:
						refreshLayoutCompatibility.pointCompatibility
				});
				refreshedLayoutProofMs = performance.now() - refreshProofStartedAt;
				layoutPublicationPlan = planUtciLayoutPublication({
					previousLayout,
					previousProof,
					previousKey,
					currentKey,
					currentSurfaceSource:
						(utciSurface?.userData.utciSurfaceSource as string | undefined) ?? null,
					currentRendererBackend: 'webgpu',
					publicationPhase,
					refreshedProof: refreshedLayoutReuseProof
				});
				if (layoutPublicationPlan.action === 'reuse-existing') {
					runtimeLayoutCompatibility = refreshLayoutCompatibility;
					acceptedRefreshedLayoutReuseProof = refreshedLayoutReuseProof;
				}
			}
			const layoutStartedAt = performance.now();
			let layoutBuildTrace: SelectedHourRenderLayoutBuildTrace | null =
				layoutPublicationPlan.action === 'reuse-existing' ? null : { totalMs: 0 };
			const layout =
				layoutPublicationPlan.action === 'reuse-existing'
					? layoutPublicationPlan.layout
					: extractUtciLayout(
							activeAnalysis,
							layoutBuildTrace ?? undefined
						);
			const layoutCompatibilityStartedAt = performance.now();
			runtimeLayoutCompatibility ??= evaluateComputeBufferUtciSurfaceLayoutCompatibility({
					state: getComputeBufferUtciSurfaceLayoutCompatibilityState(utciSurface),
					previousLayout,
					nextLayout: layout,
					metricType: acceptedOutput.metricType ?? 'utci',
					allowExpensiveMappingComparison: true
				});
			const renderLayoutCompatibilityMs =
				performance.now() - layoutCompatibilityStartedAt + refreshedLayoutCompatibilityMs;
			const layoutReuseProofStartedAt = performance.now();
			const layoutReuseProofTrace: SelectedHourRenderLayoutReuseProofTrace =
				acceptedRefreshedLayoutReuseProof ??
				(layoutPublicationPlan.action === 'reuse-existing'
					? layoutPublicationPlan.layout === previousLayout && previousProof
						? previousProof
						: buildUtciGridLayoutReuseProofDiagnostics({
								previousLayout,
								nextLayout: layout,
								canonicalRuntimeCompatibilityWouldReuse:
									runtimeLayoutCompatibility.compatible ?? false,
								canonicalPointCompatibility:
									runtimeLayoutCompatibility.pointCompatibility
							})
					: buildUtciGridLayoutReuseProofDiagnostics({
							previousLayout,
							nextLayout: layout,
							canonicalRuntimeCompatibilityWouldReuse:
								previousLayout != null
									? (runtimeLayoutCompatibility.compatible ?? null)
									: null,
							canonicalPointCompatibility:
								previousLayout != null
									? runtimeLayoutCompatibility.pointCompatibility
								: null
						}));
			const renderLayoutReuseProofMs =
				performance.now() - layoutReuseProofStartedAt + refreshedLayoutProofMs;
			const renderLayoutPublicationPlanMs =
				performance.now() - layoutPublicationPlanStartedAt;
			const renderLayoutReuseDecisionMs =
				performance.now() - layoutReuseDecisionStartedAt;
			const scenePublicationPlanReadyAtMs = performance.now();
			let meshAction: 'created' | 'reused' = 'reused';
			let renderSurfaceMeshTrace: SelectedHourRenderSurfaceMeshTrace | undefined;
			const lastRenderTargetByteLength: { value: number | undefined } = {
				value: undefined
			};
			renderTimings.renderLayoutBuildMs = performance.now() - layoutStartedAt;
			const surfaceMeshStartedAt = performance.now();
			const missingSurface = !utciSurface;
			const notComputeBufferSurface = !missingSurface && !isComputeBufferUtciSurface(utciSurface);
			const analysisIdentityChanged = activeAnalysis !== lastAnalysis;
			const layoutCompatible = runtimeLayoutCompatibility.compatible ?? false;
			const recreateDecision = {
				missingSurface,
				notComputeBufferSurface,
				analysisIdentityChanged,
				layoutCompatible
			};
			const { shouldRecreate } =
				shouldRecreateComputeBufferUtciSurface(recreateDecision);

			if (shouldRecreate) {
				meshAction = 'created';
				renderSurfaceMeshTrace = {
					action: 'created',
					totalMs: 0,
					recreateDecision
				};
				recreateComputeBufferSurface(
					activeAnalysis,
					acceptedOutput,
					layout,
					renderSurfaceMeshTrace
				);
			} else {
				const sourceBuffer = acceptedOutput.output.gpuBuffer as GPUBuffer | undefined;
				if (!sourceBuffer) {
					throw new Error('Accepted GPU-resident UTCI output is missing its GPUBuffer handle.');
				}
				const existingSurface = utciSurface;
				if (!existingSurface) {
					throw new Error('Compute-buffer UTCI surface was unavailable for update.');
				}
				renderSurfaceMeshTrace = {
					action: 'updated',
					totalMs: 0,
					recreateDecision
				};
				const updateStartedAt = performance.now();
				const updated = updateComputeBufferUtciSurfaceMesh(existingSurface, {
					layout,
					utciBuffer: sourceBuffer,
					utciRange: acceptedOutput.utciRange,
					metricType: acceptedOutput.metricType ?? 'utci',
					compatibilityEvaluation: runtimeLayoutCompatibility,
					trace: renderSurfaceMeshTrace
				});
				addSurfaceTraceTiming(
					renderSurfaceMeshTrace,
					'updateComputeBufferSurfaceMeshMs',
					performance.now() - updateStartedAt
				);
				const fallbackDecisionStartedAt = performance.now();
				if (!updated) {
					meshAction = 'created';
					renderSurfaceMeshTrace.action = 'update-failed-created';
					addSurfaceTraceTiming(
						renderSurfaceMeshTrace,
						'fallbackDecisionMs',
						performance.now() - fallbackDecisionStartedAt
					);
					recreateComputeBufferSurface(
						activeAnalysis,
						acceptedOutput,
						layout,
						renderSurfaceMeshTrace
					);
				} else {
					addSurfaceTraceTiming(
						renderSurfaceMeshTrace,
						'fallbackDecisionMs',
						performance.now() - fallbackDecisionStartedAt
					);
					const applyStartedAt = performance.now();
					applySurfaceMeshState(existingSurface, layout, 'gpuNative');
					lastAnalysis = activeAnalysis;
					lastBackend = utciSurfaceBackend;
					addSurfaceTraceTiming(
						renderSurfaceMeshTrace,
						'applySurfaceMeshStateMs',
						performance.now() - applyStartedAt
					);
				}
			}

			if (!utciSurface) {
				throw new Error('Compute-buffer UTCI surface was not created.');
			}
			const pendingReuseState = createUtciLayoutReusePublicationState({
				proof: layoutReuseProofTrace,
				key: currentKey,
				requestId: acceptedOutput.requestId,
				selectionKey: liveSelectedHourSurfaceIdentity?.selectionKey ?? null
			});

			const renderStoragePendingFlagStartedAtMs = performance.now();
			setComputeBufferSurfacePendingStorageInit(utciSurface);
			const renderStoragePendingFlagCompletedAtMs = performance.now();
			const renderPublicationPreStorageCompletedAtMs =
				renderStoragePendingFlagCompletedAtMs;
			const renderPublicationPreStorageMs =
				renderPublicationPreStorageCompletedAtMs != null &&
				renderPublicationPreStorageStartedAtMs != null
					? renderPublicationPreStorageCompletedAtMs -
						renderPublicationPreStorageStartedAtMs
					: undefined;
			const sceneSurfacePendingStorageInitAtMs =
				renderPublicationPreStorageCompletedAtMs ?? performance.now();
			if (renderSurfaceMeshTrace) {
				addSurfaceTraceTiming(
					renderSurfaceMeshTrace,
					'setPostSurfacePendingStorageInitMs',
					renderStoragePendingFlagStartedAtMs != null
						? sceneSurfacePendingStorageInitAtMs -
							renderStoragePendingFlagStartedAtMs
						: 0
				);
				renderTimings.renderSurfaceMeshMs = performance.now() - surfaceMeshStartedAt;
				renderSurfaceMeshTrace.totalMs = renderTimings.renderSurfaceMeshMs;
				renderTimings.renderPublication = createRenderPublicationDiagnostics({
					renderPublicationPath: 'compute-buffer-selected-hour',
					renderPublicationPhase: publicationPhase,
					renderPublicationMeshAction: meshAction,
					renderPublicationTimeline: {
						sceneLayoutKeyStartedAtMs,
						sceneLayoutKeyCompletedAtMs,
						scenePublicationPlanReadyAtMs,
						renderLayoutBuildTrace: layoutBuildTrace,
						renderLayoutReuseProofTrace: layoutReuseProofTrace,
						renderLayoutReuseAction:
							layoutPublicationPlan.action === 'reuse-existing'
								? 'reused'
								: 'build-required',
						renderLayoutReuseReason: layoutPublicationPlan.reason,
						renderLayoutReuseDecisionMs,
						renderLayoutReuseKeyMs:
							layoutReuseKeyDiagnostics.keyBuildMs,
						renderLayoutReuseSourceSignatureMs:
							layoutReuseKeyDiagnostics.layoutSourceSignatureMs,
						renderLayoutReusePositionsSignatureMs:
							layoutReuseKeyDiagnostics.positionsSourceSignatureMs,
						renderLayoutReusePositionsSignatureCacheHit:
							layoutReuseKeyDiagnostics.positionsSourceSignatureCacheHit,
						renderLayoutReuseFrameCacheLookupMs:
							layoutReuseKeyDiagnostics.frameCacheLookupMs,
						renderLayoutReuseFrameDerivationMs:
							layoutReuseKeyDiagnostics.frameDerivationMs,
						renderLayoutReuseFrameCacheHit:
							layoutReuseKeyDiagnostics.frameCacheHit,
						renderLayoutReuseFrameCacheKind:
							layoutReuseKeyDiagnostics.frameCacheKind,
						renderLayoutPublicationPlanMs,
						renderLayoutCompatibilityMs,
						renderLayoutCompatibilityRequiredExpensiveMappingComparison:
							runtimeLayoutCompatibility.pointCompatibility
								?.requiredExpensiveMappingComparison,
						renderLayoutCompatibilityPerformedExpensiveMappingComparison:
							runtimeLayoutCompatibility.pointCompatibility
								?.performedExpensiveMappingComparison,
						renderLayoutReuseProofMs,
						renderLayoutReuseKeyMatch: layoutPublicationPlan.keyMatch,
						renderLayoutReuseProofSource:
							acceptedRefreshedLayoutReuseProof
								? 'refreshed-runtime-proof'
								: layoutPublicationPlan.action === 'reuse-existing'
								? 'previous-publication-proof'
								: 'fresh-build-proof',
						renderLayoutReusePreviousKey:
							previousReuseState?.layoutIdentity ??
							(previousKey ? getUtciLayoutIdentity(previousKey) : null),
						renderLayoutReusePreviousRequestId:
							layoutPublicationPlan.action === 'reuse-existing'
								? previousReuseState?.requestId ?? null
								: null,
						renderLayoutReusePreviousSelectionKey:
							layoutPublicationPlan.action === 'reuse-existing'
								? previousReuseState?.selectionKey ?? null
								: null,
						activeLayoutCandidateCount: pendingReuseState ? 1 : 0,
						renderSurfaceMeshTrace,
						sceneSurfacePendingStorageInitAtMs,
						renderPublicationPreStorageStartedAtMs,
						renderPublicationPreStorageCompletedAtMs,
						renderPublicationPreStorageMs,
						renderStoragePendingFlagStartedAtMs,
						renderStoragePendingFlagCompletedAtMs
					}
				});
			}
			const syncResult = await copyComputeBufferIntoRenderOwnedStorage({
				mesh: utciSurface,
				acceptedOutput,
				activeSyncRun,
				syncStartedAt,
				scenePendingSurfaceObservedAtMs,
				sceneSyncAttemptStartedAtMs,
				sceneSyncAttemptToken,
				sceneSurfaceReceivedAtMs,
				publicationEffectStartedAtMs,
				reactiveTiming: {
					...reactiveTiming,
					sceneStartSyncEnteredAtMs,
					sceneStartSyncReturnedAtMs
				},
				sceneSurfacePendingStorageInitAtMs,
				renderTimings,
				meshAction,
				layout,
				lastRenderTargetByteLength
			});
			activeUtciLayoutReuseState = resolveUtciLayoutReusePublicationStateAfterSync({
				currentState: previousReuseState,
				pendingState: pendingReuseState,
				syncResult
			});
		} catch (error) {
			const errorMessage = error instanceof Error ? error.message : String(error);
			if (
				acceptedGpuResidentSurfaceSync.failSync(activeSyncRun, {
					...getCurrentGpuResidentSyncLiveState(),
					errorMessage
				}) === 'superseded'
			) {
				return;
			}

			if (utciSurface) {
				setComputeBufferSurfacePublicationVisibility(utciSurface, false);
			}
		}
	}

	// Track last update state to avoid redundant texture updates
	let lastUpdateState: {
		hour: number;
		month: number;
		colorMode: string;
		metricType: string;
		unifiedRangeMin: number | null;
		unifiedRangeMax: number | null;
	} | null = null;

	/**
	 * Check if the update state has changed and we need to refresh the texture.
	 * This consolidates all texture update triggers into a single reactive block.
	 */
	function hasStateChanged(
		renderState: NonNullable<ReturnType<typeof resolveLiveSelectedHourSurfaceRenderState>>
	): boolean {
		const currentState = {
			hour: renderState.hourIndex,
			month: renderState.monthIndex,
			colorMode: renderState.colorMode,
			metricType: renderState.metricType,
			unifiedRangeMin: renderState.rangeOverride?.utciMin ?? null,
			unifiedRangeMax: renderState.rangeOverride?.utciMax ?? null
		};

		if (!lastUpdateState) {
			lastUpdateState = currentState;
			return true;
		}

		const changed =
			lastUpdateState.hour !== currentState.hour ||
			lastUpdateState.month !== currentState.month ||
			lastUpdateState.colorMode !== currentState.colorMode ||
			lastUpdateState.metricType !== currentState.metricType ||
			lastUpdateState.unifiedRangeMin !== currentState.unifiedRangeMin ||
			lastUpdateState.unifiedRangeMax !== currentState.unifiedRangeMax;

		if (changed) {
			lastUpdateState = currentState;
		}

		return changed;
	}

	$: {
		const sceneReactiveBlockEnteredAtMs = performance.now();
		const viewerState = $viewerStore;
		const currentUnifiedRange =
			rangeOverride !== undefined ? rangeOverride : $unifiedUtciRange;
		const renderState = resolveLiveSelectedHourSurfaceRenderState({
			analysis,
			viewerState,
			publishedRenderContext: selectedHourRenderContext,
			rangeOverride: currentUnifiedRange
		});
		const sceneRenderStateResolvedAtMs = performance.now();
		const acceptedKey =
			renderState?.analysis && utciSurfaceBackend === 'gpuNative'
				? getAcceptedGpuResidentKey(acceptedGpuResidentOutput)
				: null;
		const acceptedSyncRunKey =
			renderState?.analysis && utciSurfaceBackend === 'gpuNative'
				? acceptedGpuResidentSurfaceSync.getSyncRunKey({
						acceptedGpuResidentOutput,
						liveSelectedHourSurfaceIdentity
					})
				: null;
		const sceneAcceptedKeyResolvedAtMs = performance.now();
		const useGpuResidentComputeSurface =
			Boolean(renderState?.analysis) &&
			utciSurfaceBackend === 'gpuNative' &&
			acceptedGpuResidentOutput != null;

		if (!renderState) {
			disposeUtciSurface();
			lastUpdateState = null;
			lastObservedPendingSurface = null;
		} else if (useGpuResidentComputeSurface && acceptedGpuResidentOutput && acceptedKey) {
			lastUpdateState = null;
			if (!acceptedSyncRunKey) {
				acceptedGpuResidentSurfaceSync.reset({
					invalidateActiveRun: true,
					reason: 'missing-accepted-sync-run-key'
				});
				lastObservedPendingSurface = null;
			} else if (
				acceptedGpuResidentSurfaceSync.getActiveSyncRunKey() !== acceptedSyncRunKey ||
				renderState.analysis !== lastAnalysis ||
				!isComputeBufferUtciSurface(utciSurface)
			) {
				const sceneSyncInvocationQueuedAtMs = performance.now();
				if (lastObservedPendingSurface?.syncRunKey !== acceptedSyncRunKey) {
					lastObservedPendingSurface = {
						syncRunKey: acceptedSyncRunKey,
						observedAtMs: sceneSyncInvocationQueuedAtMs
					};
				}
				void syncAcceptedGpuResidentSurface(
					renderState.analysis,
					acceptedGpuResidentOutput,
					performance.now(),
					lastObservedPendingSurface?.observedAtMs,
					{
						sceneReactiveBlockEnteredAtMs,
						sceneRenderStateResolvedAtMs,
						sceneAcceptedKeyResolvedAtMs,
						sceneSyncInvocationQueuedAtMs
					}
				);
			}
		} else {
			acceptedGpuResidentSurfaceSync.reset({
				reason: 'fallback-cpu-surface'
			});
			lastObservedPendingSurface = null;

			if (utciSurface && isComputeBufferUtciSurface(utciSurface)) {
				disposeUtciSurface();
			}

			if (
				!utciSurface ||
				renderState.analysis !== lastAnalysis ||
				utciSurfaceBackend !== lastBackend
			) {
				recreateUtciSurface(renderState);
				lastUpdateState = {
					hour: renderState.hourIndex,
					month: renderState.monthIndex,
					colorMode: renderState.colorMode,
					metricType: renderState.metricType,
					unifiedRangeMin: renderState.rangeOverride?.utciMin ?? null,
					unifiedRangeMax: renderState.rangeOverride?.utciMax ?? null
				};
				invalidate();
			} else if (utciSurface && hasStateChanged(renderState)) {
				const updated = updateUtciSurfaceMesh(utciSurface, buildSurfaceOptions(renderState));
				if (!updated) {
					recreateUtciSurface(renderState);
				} else {
					publishUtciSurfaceDiagnostics();
				}
				invalidate();
			}
		}
	}

	$: {
		if (utciSurface) {
			const isComputeSurface = isComputeBufferUtciSurface(utciSurface);
			const shouldRenderForStorageInit =
				isComputeSurface && gpuResidentCopyStatus === 'pending';
			const shouldBeVisible =
				Boolean(analysis && $viewerStore?.utciVisible) &&
				(!isComputeSurface || gpuResidentCopyStatus === 'complete');
			if (shouldRenderForStorageInit) {
				setComputeBufferSurfacePendingStorageInit(utciSurface);
			} else {
				setComputeBufferSurfacePublicationVisibility(utciSurface, shouldBeVisible);
			}
			if (utciSurface.visible) {
				invalidate();
			}
		}
	}

	$: if (
		utciSurface &&
		!isComputeBufferUtciSurface(utciSurface) &&
		liveSelectedHourSurfaceIdentity
	) {
		publishUtciSurfaceDiagnostics();
	}

	onDestroy(() => {
		disposeUtciSurface();
	});
</script>

