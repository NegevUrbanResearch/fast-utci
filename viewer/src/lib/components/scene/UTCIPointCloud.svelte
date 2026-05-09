<script lang="ts">
	import { useThrelte } from '@threlte/core';
	import { onDestroy } from 'svelte';
	import type { OnDemandUtciOutput } from '$lib/compute/gpu-pipeline';
	import type { Analysis } from '$lib/types/analysis';
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
		getGpuNativeUtciSurfaceSource,
		updateComputeBufferUtciSurfaceMesh
	} from '$lib/services/gpuUtciRenderBridge';
	import { getEffectiveHourIndex } from '$lib/utils/effectiveHourIndex';
	import { viewerStore } from '$lib/stores/viewerStore';
	import { unifiedUtciRange } from '$lib/stores/comparisonStore';
	import type { Group, Mesh } from 'three';

	type AcceptedGpuResidentUtciOutput = {
		requestId: number;
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
		output: OnDemandUtciOutput;
		utciRange: { min: number; max: number };
	};

	type UtciSurfaceDiagnostics = {
		utciSurfaceSource?: string;
		selectedHourTransferCount?: number;
		dataTextureBuildCount?: number;
		gpuResidentCopyStatus?: 'idle' | 'pending' | 'complete' | 'failed';
		gpuResidentCopyError?: string;
		gpuResidentCopyRequestId?: number;
	};

	export let analysis: Analysis | null = null;
	export let model: Group | null = null;
	export let utciSurfaceBackend: UtciSurfaceBackendType = 'dataTexture';
	export let acceptedGpuResidentOutput: AcceptedGpuResidentUtciOutput | null = null;
	export let onUtciSurfaceDiagnostics:
		| ((diagnostics: UtciSurfaceDiagnostics) => void)
		| undefined = undefined;

	export let utciSurface: Mesh | null = null;
	let lastAnalysis: Analysis | null = null;
	let lastBackend: UtciSurfaceBackendType | null = null;
	let activeGpuResidentSyncKey: string | null = null;
	let gpuResidentCopyStatus: 'idle' | 'pending' | 'complete' | 'failed' = 'idle';
	let gpuResidentCopyError: string | undefined = undefined;
	let gpuResidentCopyRequestId: number | undefined = undefined;
	let gpuResidentCopyRunToken = 0;
	const { renderer, scene, invalidate } = useThrelte();

	function getAcceptedGpuResidentKey(
		value: AcceptedGpuResidentUtciOutput | null
	): string | null {
		if (!value) return null;
		return `${value.requestId}:${value.monthIndex}:${value.timeIndex}:${value.utciRange.min}:${value.utciRange.max}`;
	}

	function isComputeBufferSurface(mesh: Mesh | null): boolean {
		return mesh != null && getGpuNativeUtciSurfaceSource(mesh) === 'compute-buffer-selected-hour';
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
		onUtciSurfaceDiagnostics?.({});
	}

	function publishUtciSurfaceDiagnostics(): void {
		onUtciSurfaceDiagnostics?.({
			utciSurfaceSource: utciSurface?.userData.utciSurfaceSource as string | undefined,
			selectedHourTransferCount: utciSurface?.userData.selectedHourTransferCount as
				| number
				| undefined,
			dataTextureBuildCount: utciSurface?.userData.dataTextureBuildCount as
				| number
				| undefined,
			gpuResidentCopyStatus,
			gpuResidentCopyError,
			gpuResidentCopyRequestId
		});
	}

	function setGpuResidentCopyDiagnostics(
		status: 'idle' | 'pending' | 'complete' | 'failed',
		options?: { error?: string; requestId?: number }
	): void {
		gpuResidentCopyStatus = status;
		gpuResidentCopyError = options?.error;
		gpuResidentCopyRequestId = options?.requestId;
		publishUtciSurfaceDiagnostics();
	}

	function waitForNextFrame(): Promise<void> {
		return new Promise((resolve) => requestAnimationFrame(() => resolve()));
	}

	function buildSurfaceOptions(
		activeAnalysis: Analysis,
		viewerState: typeof $viewerStore,
		rangeOverride: typeof $unifiedUtciRange | undefined
	) {
		return {
			analysis: activeAnalysis,
			hourIndex: getEffectiveHourIndex(
				activeAnalysis,
				viewerState?.currentHour ?? 0,
				viewerState?.currentMonth ?? 7
			),
			colorMode: viewerState?.colorMode ?? 'normalized',
			metricType: viewerState?.metricType ?? 'utci',
			rangeOverride: rangeOverride ?? undefined,
			monthIndex: viewerState?.currentMonth ?? 7,
			backend: utciSurfaceBackend
		} as const;
	}

	function recreateUtciSurface(
		activeAnalysis: Analysis,
		viewerState: typeof $viewerStore,
		rangeOverride: typeof $unifiedUtciRange | undefined
	): void {
		disposeUtciSurface();
		utciSurface = createUtciSurfaceMesh(buildSurfaceOptions(activeAnalysis, viewerState, rangeOverride));
		scene.add(utciSurface);
		lastAnalysis = activeAnalysis;
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
		acceptedOutput: AcceptedGpuResidentUtciOutput
	): void {
		const sourceBuffer = acceptedOutput.output.gpuBuffer as GPUBuffer | undefined;
		if (!sourceBuffer) {
			throw new Error('Accepted GPU-resident UTCI output is missing its GPUBuffer handle.');
		}

		disposeUtciSurface({ invalidateGpuResidentCopies: false });
		const layout = extractUtciLayout(activeAnalysis);
		utciSurface = createComputeBufferUtciSurfaceMesh({
			layout,
			utciBuffer: sourceBuffer,
			utciRange: acceptedOutput.utciRange
		});
		applySurfaceMeshState(utciSurface, layout, 'gpuNative');
		utciSurface.visible = false;
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
		acceptedOutput: AcceptedGpuResidentUtciOutput;
		copyRunToken: number;
		syncKey: string;
	}): Promise<void> {
		const { mesh, acceptedOutput, copyRunToken, syncKey } = params;
		const sourceBuffer = acceptedOutput.output.gpuBuffer as GPUBuffer | undefined;
		if (!sourceBuffer) {
			throw new Error('Accepted GPU-resident UTCI output is missing its GPUBuffer handle.');
		}

		const storageAttribute = getComputeBufferUtciStorageAttribute(mesh);
		if (!storageAttribute) {
			throw new Error('Compute-buffer UTCI storage attribute was not available.');
		}
		const { device, targetBuffer } = await waitForRenderStorageBuffer({
			storageAttribute,
			copyRunToken,
			syncKey,
			requestId: acceptedOutput.requestId
		});
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

		const encoder = device.createCommandEncoder();
		encoder.copyBufferToBuffer(sourceBuffer, 0, targetBuffer, 0, sourceBuffer.size);
		device.queue.submit([encoder.finish()]);
		await device.queue.onSubmittedWorkDone();
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
		mesh.visible = Boolean(analysis && $viewerStore?.utciVisible);
		setGpuResidentCopyDiagnostics('complete', {
			requestId: acceptedOutput.requestId
		});
		invalidate();
	}

	async function syncAcceptedGpuResidentSurface(
		activeAnalysis: Analysis,
		acceptedOutput: AcceptedGpuResidentUtciOutput
	): Promise<void> {
		const syncKey = getAcceptedGpuResidentKey(acceptedOutput);
		if (!syncKey) return;

		const copyRunToken = ++gpuResidentCopyRunToken;
		activeGpuResidentSyncKey = syncKey;
		setGpuResidentCopyDiagnostics('pending', {
			requestId: acceptedOutput.requestId
		});

		try {
			if (!utciSurface || !isComputeBufferSurface(utciSurface) || activeAnalysis !== lastAnalysis) {
				recreateComputeBufferSurface(activeAnalysis, acceptedOutput);
			} else {
				const sourceBuffer = acceptedOutput.output.gpuBuffer as GPUBuffer | undefined;
				if (!sourceBuffer) {
					throw new Error('Accepted GPU-resident UTCI output is missing its GPUBuffer handle.');
				}
				const layout = extractUtciLayout(activeAnalysis);
				const updated = updateComputeBufferUtciSurfaceMesh(utciSurface, {
					layout,
					utciBuffer: sourceBuffer,
					utciRange: acceptedOutput.utciRange
				});
				if (!updated) {
					recreateComputeBufferSurface(activeAnalysis, acceptedOutput);
				} else {
					applySurfaceMeshState(utciSurface, layout, 'gpuNative');
				}
			}

			if (!utciSurface) {
				throw new Error('Compute-buffer UTCI surface was not created.');
			}

			activeGpuResidentSyncKey = syncKey;
			utciSurface.visible = false;
			setGpuResidentCopyDiagnostics('pending', {
				requestId: acceptedOutput.requestId
			});
			await copyComputeBufferIntoRenderOwnedStorage({
				mesh: utciSurface,
				acceptedOutput,
				copyRunToken,
				syncKey
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
				utciSurface.visible = false;
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
		viewerState: typeof $viewerStore,
		unifiedRange: typeof $unifiedUtciRange
	): boolean {
		const currentState = {
			hour: viewerState.currentHour,
			month: viewerState.currentMonth ?? 7,
			colorMode: viewerState.colorMode,
			metricType: viewerState.metricType ?? 'utci',
			unifiedRangeMin: unifiedRange?.utciMin ?? null,
			unifiedRangeMax: unifiedRange?.utciMax ?? null
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
		const currentUnifiedRange = $unifiedUtciRange;
		const acceptedKey =
			analysis && utciSurfaceBackend === 'gpuNative'
				? getAcceptedGpuResidentKey(acceptedGpuResidentOutput)
				: null;
		const useGpuResidentComputeSurface =
			Boolean(analysis) &&
			utciSurfaceBackend === 'gpuNative' &&
			acceptedGpuResidentOutput != null;

		if (!analysis) {
			disposeUtciSurface();
			lastUpdateState = null;
		} else if (useGpuResidentComputeSurface && acceptedGpuResidentOutput && acceptedKey) {
			lastUpdateState = null;
			if (activeGpuResidentSyncKey !== acceptedKey || analysis !== lastAnalysis || !isComputeBufferSurface(utciSurface)) {
				void syncAcceptedGpuResidentSurface(analysis, acceptedGpuResidentOutput);
			}
		} else {
			activeGpuResidentSyncKey = null;
			gpuResidentCopyStatus = 'idle';
			gpuResidentCopyError = undefined;
			gpuResidentCopyRequestId = undefined;

			if (utciSurface && isComputeBufferSurface(utciSurface)) {
				disposeUtciSurface();
			}

			if (!utciSurface || analysis !== lastAnalysis || utciSurfaceBackend !== lastBackend) {
				recreateUtciSurface(analysis, viewerState, currentUnifiedRange);
				lastUpdateState = {
					hour: viewerState?.currentHour ?? 0,
					month: viewerState?.currentMonth ?? 7,
					colorMode: viewerState?.colorMode ?? 'normalized',
					metricType: viewerState?.metricType ?? 'utci',
					unifiedRangeMin: currentUnifiedRange?.utciMin ?? null,
					unifiedRangeMax: currentUnifiedRange?.utciMax ?? null
				};
				invalidate();
			} else if (utciSurface && viewerState && hasStateChanged(viewerState, currentUnifiedRange)) {
				const updated = updateUtciSurfaceMesh(
					utciSurface,
					buildSurfaceOptions(analysis, viewerState, currentUnifiedRange)
				);
				if (!updated) {
					recreateUtciSurface(analysis, viewerState, currentUnifiedRange);
				} else {
					publishUtciSurfaceDiagnostics();
				}
				invalidate();
			}
		}
	}

	$: {
		if (utciSurface) {
			const shouldBeVisible =
				Boolean(analysis && $viewerStore?.utciVisible) &&
				(!isComputeBufferSurface(utciSurface) || gpuResidentCopyStatus === 'complete');
			utciSurface.visible = shouldBeVisible;
			if (utciSurface.visible) {
				invalidate();
			}
		}
	}

	onDestroy(() => {
		disposeUtciSurface();
	});
</script>

