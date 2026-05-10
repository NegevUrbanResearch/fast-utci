<script lang="ts">
	import { useThrelte } from '@threlte/core';
	import { onDestroy } from 'svelte';
	import type { Analysis } from '$lib/types/analysis';
	import type { SelectedHourGpuResidentOutput } from '$lib/compute/liveUtciSelectedHourSession';
	import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/liveSelectedHourSurfaceIdentity';
	import {
		resolveLiveSelectedHourSurfaceRenderState,
		type LiveSelectedHourPublishedRenderContext
	} from '$lib/compute/liveSelectedHourRenderContext';
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
		updateComputeBufferUtciSurfaceMesh
	} from '$lib/services/gpuUtciRenderBridge';
	import { viewerStore } from '$lib/stores/viewerStore';
	import { unifiedUtciRange } from '$lib/stores/comparisonStore';
	import type { UnifiedUtciRange } from '$lib/stores/comparisonStore';
	import {
		invokeDiagnosticsCallbackSafely,
		type SelectedHourRenderTimingSubsteps
	} from '$lib/compute/onDemandDiagnostics';
	import {
		buildCpuPublicationDiagnostics,
		buildUtciSurfaceDiagnostics,
		getAcceptedGpuResidentKey,
		isComputeBufferUtciSurface,
		type GpuResidentCopyStatus,
		type UtciSurfaceDiagnostics
	} from '$lib/components/scene/utciSurfaceSync';
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

	export let utciSurface: Mesh | null = null;
	let lastAnalysis: Analysis | null = null;
	let lastBackend: UtciSurfaceBackendType | null = null;
	let activeGpuResidentSyncKey: string | null = null;
	let gpuResidentCopyStatus: GpuResidentCopyStatus = 'idle';
	let gpuResidentCopyError: string | undefined = undefined;
	let gpuResidentCopyRequestId: number | undefined = undefined;
	let gpuResidentRenderTimings: SelectedHourRenderTimingSubsteps | undefined = undefined;
	let gpuResidentCopyRunToken = 0;
	const { renderer, scene, invalidate } = useThrelte();

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

	function disposeUtciSurface(options: { invalidateGpuResidentCopies?: boolean } = {}) {
		if (options.invalidateGpuResidentCopies ?? true) {
			gpuResidentCopyRunToken += 1;
		}
		if (utciSurface) {
			disposeUtciSurfaceMesh(utciSurface);
			utciSurface = null;
		}
		lastAnalysis = null;
		lastBackend = null;
		activeGpuResidentSyncKey = null;
		gpuResidentCopyStatus = 'idle';
		gpuResidentCopyError = undefined;
		gpuResidentCopyRequestId = undefined;
		gpuResidentRenderTimings = undefined;
		invokeUtciSurfaceDiagnostics({});
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

	async function waitForRenderStorageBuffer(params: {
		storageAttribute: unknown;
		copyRunToken: number;
		syncKey: string;
		requestId: number;
	}): Promise<{ device: GPUDevice; targetBuffer: GPUBuffer }> {
		const { storageAttribute, copyRunToken, syncKey, requestId } = params;
		const deadline = performance.now() + 1000;
		let lastDevice: GPUDevice | undefined;

		while (performance.now() < deadline) {
			invalidate();
			await waitForNextFrame();
			if (
				copyRunToken !== gpuResidentCopyRunToken ||
				activeGpuResidentSyncKey !== syncKey ||
				acceptedGpuResidentOutput?.requestId !== requestId
			) {
				throw new Error('GPU-resident render copy was superseded before storage initialization.');
			}

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
			if (device && targetBuffer) {
				return { device, targetBuffer };
			}
		}

		throw new Error(
			lastDevice
				? 'Three storage buffer was not initialized within the GPU-resident render timeout.'
				: 'Renderer WebGPU device was not available within the GPU-resident render timeout.'
		);
	}

	function recreateComputeBufferSurface(
		activeAnalysis: Analysis,
		acceptedOutput: SelectedHourGpuResidentOutput,
		layout: ReturnType<typeof extractUtciLayout>
	): void {
		const sourceBuffer = acceptedOutput.output.gpuBuffer as GPUBuffer | undefined;
		if (!sourceBuffer) {
			throw new Error('Accepted GPU-resident UTCI output is missing its GPUBuffer handle.');
		}

		disposeUtciSurface({ invalidateGpuResidentCopies: false });
		utciSurface = createComputeBufferUtciSurfaceMesh({
			layout,
			utciBuffer: sourceBuffer,
			utciRange: acceptedOutput.utciRange
		});
		applySurfaceMeshState(utciSurface, layout, 'gpuNative');
		setComputeBufferSurfacePendingStorageInit(utciSurface);
		scene.add(utciSurface);
		lastAnalysis = activeAnalysis;
		lastBackend = utciSurfaceBackend;
		gpuResidentCopyStatus = 'idle';
		gpuResidentCopyError = undefined;
		gpuResidentCopyRequestId = undefined;
		publishUtciSurfaceDiagnostics();
	}

	async function copyComputeBufferIntoRenderOwnedStorage(params: {
		mesh: Mesh;
		acceptedOutput: SelectedHourGpuResidentOutput;
		copyRunToken: number;
		syncKey: string;
		syncStartedAt: number;
		renderTimings: SelectedHourRenderTimingSubsteps;
	}): Promise<void> {
		const { mesh, acceptedOutput, copyRunToken, syncKey, syncStartedAt, renderTimings } = params;
		const sourceBuffer = acceptedOutput.output.gpuBuffer as GPUBuffer | undefined;
		if (!sourceBuffer) {
			throw new Error('Accepted GPU-resident UTCI output is missing its GPUBuffer handle.');
		}

		const storageAttribute = getComputeBufferUtciStorageAttribute(mesh);
		if (!storageAttribute) {
			throw new Error('Compute-buffer UTCI storage attribute was not available.');
		}
		const storageInitWaitStartedAt = performance.now();
		const { device, targetBuffer } = await waitForRenderStorageBuffer({
			storageAttribute,
			copyRunToken,
			syncKey,
			requestId: acceptedOutput.requestId
		});
		renderTimings.renderStorageInitWaitMs = performance.now() - storageInitWaitStartedAt;
		if (
			copyRunToken !== gpuResidentCopyRunToken ||
			activeGpuResidentSyncKey !== syncKey ||
			acceptedGpuResidentOutput?.requestId !== acceptedOutput.requestId
		) {
			return;
		}
		if (targetBuffer.size < sourceBuffer.size) {
			throw new Error('Three storage buffer is smaller than the accepted compute output buffer.');
		}

		const bufferCopyStartedAt = performance.now();
		const encoder = device.createCommandEncoder();
		encoder.copyBufferToBuffer(sourceBuffer, 0, targetBuffer, 0, sourceBuffer.size);
		device.queue.submit([encoder.finish()]);
		renderTimings.renderBufferCopyMs = performance.now() - bufferCopyStartedAt;
		const queueDrainStartedAt = performance.now();
		await device.queue.onSubmittedWorkDone();
		renderTimings.renderQueueDrainMs = performance.now() - queueDrainStartedAt;
		if (
			copyRunToken !== gpuResidentCopyRunToken ||
			activeGpuResidentSyncKey !== syncKey ||
			acceptedGpuResidentOutput?.requestId !== acceptedOutput.requestId
		) {
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
		setGpuResidentCopyDiagnostics('complete', {
			requestId: acceptedOutput.requestId,
			renderTimings
		});
		invalidate();
	}

	async function syncAcceptedGpuResidentSurface(
		activeAnalysis: Analysis,
		acceptedOutput: SelectedHourGpuResidentOutput
	): Promise<void> {
		const syncKey = getAcceptedGpuResidentKey(acceptedOutput);
		if (!syncKey) return;

		const copyRunToken = ++gpuResidentCopyRunToken;
		activeGpuResidentSyncKey = syncKey;
		setGpuResidentCopyDiagnostics('pending', {
			requestId: acceptedOutput.requestId
		});

		try {
			const syncStartedAt = performance.now();
			const renderTimings: SelectedHourRenderTimingSubsteps = {};
			if (pendingRenderUpdateStartedAt !== undefined) {
				renderTimings.renderSceneSyncStartDelayMs = Math.max(
					0,
					syncStartedAt - pendingRenderUpdateStartedAt
				);
			}
			const layoutStartedAt = performance.now();
			const layout = extractUtciLayout(activeAnalysis);
			renderTimings.renderLayoutBuildMs = performance.now() - layoutStartedAt;

			if (!utciSurface || !isComputeBufferUtciSurface(utciSurface) || activeAnalysis !== lastAnalysis) {
				const surfaceMeshStartedAt = performance.now();
				recreateComputeBufferSurface(activeAnalysis, acceptedOutput, layout);
				renderTimings.renderSurfaceMeshMs = performance.now() - surfaceMeshStartedAt;
			} else {
				const sourceBuffer = acceptedOutput.output.gpuBuffer as GPUBuffer | undefined;
				if (!sourceBuffer) {
					throw new Error('Accepted GPU-resident UTCI output is missing its GPUBuffer handle.');
				}
				const surfaceMeshStartedAt = performance.now();
				const updated = updateComputeBufferUtciSurfaceMesh(utciSurface, {
					layout,
					utciBuffer: sourceBuffer,
					utciRange: acceptedOutput.utciRange
				});
				if (!updated) {
					recreateComputeBufferSurface(activeAnalysis, acceptedOutput, layout);
				} else {
					applySurfaceMeshState(utciSurface, layout, 'gpuNative');
				}
				renderTimings.renderSurfaceMeshMs = performance.now() - surfaceMeshStartedAt;
			}

			if (!utciSurface) {
				throw new Error('Compute-buffer UTCI surface was not created.');
			}

			activeGpuResidentSyncKey = syncKey;
			setComputeBufferSurfacePendingStorageInit(utciSurface);
			setGpuResidentCopyDiagnostics('pending', {
				requestId: acceptedOutput.requestId
			});
			await copyComputeBufferIntoRenderOwnedStorage({
				mesh: utciSurface,
				acceptedOutput,
				copyRunToken,
				syncKey,
				syncStartedAt,
				renderTimings
			});
		} catch (error) {
			if (
				copyRunToken !== gpuResidentCopyRunToken ||
				activeGpuResidentSyncKey !== syncKey ||
				acceptedGpuResidentOutput?.requestId !== acceptedOutput.requestId
			) {
				return;
			}

			if (utciSurface) {
				setComputeBufferSurfacePublicationVisibility(utciSurface, false);
			}
			setGpuResidentCopyDiagnostics('failed', {
				error: error instanceof Error ? error.message : String(error),
				requestId: acceptedOutput.requestId
			});
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
		const useGpuResidentComputeSurface =
			Boolean(renderState?.analysis) &&
			utciSurfaceBackend === 'gpuNative' &&
			acceptedGpuResidentOutput != null;

		if (!renderState) {
			disposeUtciSurface();
			lastUpdateState = null;
		} else if (useGpuResidentComputeSurface && acceptedGpuResidentOutput && acceptedKey) {
			lastUpdateState = null;
			if (
				activeGpuResidentSyncKey !== acceptedKey ||
				renderState.analysis !== lastAnalysis ||
				!isComputeBufferUtciSurface(utciSurface)
			) {
				void syncAcceptedGpuResidentSurface(
					renderState.analysis,
					acceptedGpuResidentOutput
				);
			}
		} else {
			activeGpuResidentSyncKey = null;
			gpuResidentCopyStatus = 'idle';
			gpuResidentCopyError = undefined;
			gpuResidentCopyRequestId = undefined;

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

