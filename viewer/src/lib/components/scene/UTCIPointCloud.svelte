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
		buildUtciGridLayout,
		createUtciSurfaceMesh,
		disposeUtciSurfaceMesh,
		type UtciSurfaceBackendType,
		updateUtciSurfaceMesh
	} from '$lib/services/pointCloudService';
	import {
		createComputeBufferUtciSurfaceMesh,
		getComputeBufferUtciStorageAttribute,
		isComputeBufferUtciSurfaceLayoutCompatible,
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
		type AcceptedGpuResidentSurfaceSyncRun
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

	function extractUtciLayout(activeAnalysis: Analysis) {
		const layout = buildUtciGridLayout(activeAnalysis);
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
			utciRange: acceptedOutput.utciRange
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
		renderTimings: SelectedHourRenderTimingSubsteps;
		meshAction: 'created' | 'reused';
		layout: ReturnType<typeof extractUtciLayout>;
		lastRenderTargetByteLength: { value: number | undefined };
	}): Promise<void> {
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
		const isSuperseded = () =>
			acceptedGpuResidentSurfaceSync.isSuperseded(
				activeSyncRun,
				getCurrentGpuResidentSyncLiveState()
			);
		const { device, targetBuffer, waitMs, waitTrace } = await waitForRenderStorageBuffer({
			deadlineMs: 1000,
			now: performance.now.bind(performance),
			waitForNextFrame: async () => {
				invalidate();
				await waitForNextFrame();
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
		renderTimings.renderStorageInitWaitMs = waitMs;
		if (isSuperseded()) {
			acceptedGpuResidentSurfaceSync.supersedeSync(activeSyncRun);
			return;
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
			acceptedGpuResidentSurfaceSync.supersedeSync(activeSyncRun);
			return;
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
		renderTimings.renderPublication = createRenderPublicationDiagnostics({
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: acceptedOutput.requestId <= 1 ? 'initial' : 'scrub',
			renderPublicationMeshAction: meshAction,
			renderPublicationPointCount: layout.numPositions,
			renderPublicationVertexCount: layout.width * layout.height * 6,
			renderPublicationGridWidth: layout.width,
			renderPublicationGridHeight: layout.height,
			renderPublicationGridSize: layout.gridSize,
			renderPublicationSourceByteLength: sourceBuffer.size,
			renderPublicationTargetByteLength: lastRenderTargetByteLength.value,
			renderPublicationRenderOwnedBytes:
				mesh.userData.renderOwnedSelectedHourBytes as number | undefined,
			renderPublicationTimeline: {
				scenePendingSurfaceObservedAtMs,
				sceneSyncAttemptStartedAtMs,
				sceneSyncAttemptToken,
				sceneSurfaceReceivedAtMs,
				publicationEffectStartedAtMs,
				renderSurfaceMeshTrace:
					renderTimings.renderPublication?.renderPublicationTimeline
						?.renderSurfaceMeshTrace,
				renderStorageReadyAtMs,
				renderStorageWaitTrace: waitTrace,
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
		if (
			acceptedGpuResidentSurfaceSync.completeSync(activeSyncRun, {
				...getCurrentGpuResidentSyncLiveState(),
				renderTimings
			}) !== 'complete'
		) {
			return;
		}
		invalidate();
	}

	async function syncAcceptedGpuResidentSurface(
		activeAnalysis: Analysis,
		acceptedOutput: SelectedHourGpuResidentOutput,
		sceneSurfaceReceivedAtMs: number,
		scenePendingSurfaceObservedAtMs?: number
	): Promise<void> {
		const activeSyncRun =
			acceptedGpuResidentSurfaceSync.startSync({
				acceptedOutput,
				liveSelectedHourSurfaceIdentity
			});
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
			const layoutStartedAt = performance.now();
			const layout = extractUtciLayout(activeAnalysis);
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
			const layoutCompatible = isComputeBufferUtciSurfaceLayoutCompatible(
				utciSurface,
				layout
			);
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
					utciRange: acceptedOutput.utciRange
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

			const pendingStorageStartedAt = performance.now();
			setComputeBufferSurfacePendingStorageInit(utciSurface);
			if (renderSurfaceMeshTrace) {
				addSurfaceTraceTiming(
					renderSurfaceMeshTrace,
					'setPostSurfacePendingStorageInitMs',
					performance.now() - pendingStorageStartedAt
				);
				renderTimings.renderSurfaceMeshMs = performance.now() - surfaceMeshStartedAt;
				renderSurfaceMeshTrace.totalMs = renderTimings.renderSurfaceMeshMs;
				renderTimings.renderPublication = createRenderPublicationDiagnostics({
					renderPublicationPath: 'compute-buffer-selected-hour',
					renderPublicationPhase: acceptedOutput.requestId <= 1 ? 'initial' : 'scrub',
					renderPublicationMeshAction: meshAction,
					renderPublicationTimeline: {
						renderSurfaceMeshTrace
					}
				});
			}
			await copyComputeBufferIntoRenderOwnedStorage({
				mesh: utciSurface,
				acceptedOutput,
				activeSyncRun,
				syncStartedAt,
				scenePendingSurfaceObservedAtMs,
				sceneSyncAttemptStartedAtMs,
				sceneSyncAttemptToken,
				sceneSurfaceReceivedAtMs,
				publicationEffectStartedAtMs,
				renderTimings,
				meshAction,
				layout,
				lastRenderTargetByteLength
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
		const viewerState = $viewerStore;
		const currentUnifiedRange =
			rangeOverride !== undefined ? rangeOverride : $unifiedUtciRange;
		const renderState = resolveLiveSelectedHourSurfaceRenderState({
			analysis,
			viewerState,
			publishedRenderContext: selectedHourRenderContext,
			rangeOverride: currentUnifiedRange
		});
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
				if (lastObservedPendingSurface?.syncRunKey !== acceptedSyncRunKey) {
					lastObservedPendingSurface = {
						syncRunKey: acceptedSyncRunKey,
						observedAtMs: performance.now()
					};
				}
				void syncAcceptedGpuResidentSurface(
					renderState.analysis,
					acceptedGpuResidentOutput,
					performance.now(),
					lastObservedPendingSurface?.observedAtMs
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

