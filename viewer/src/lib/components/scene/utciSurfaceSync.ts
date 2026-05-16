import type { Mesh } from 'three';
import type { SelectedHourGpuResidentOutput } from '$lib/compute/selected-hour/liveUtciSelectedHourSession';
import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/selected-hour/liveSelectedHourSurfaceIdentity';
import type { SelectedHourRenderTimingSubsteps } from '$lib/compute/on-demand/onDemandDiagnostics';
import {
	copyRenderPublicationDiagnostics,
	type SelectedHourRenderSurfaceMeshRecreateDecision
} from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';
import { getGpuNativeUtciSurfaceSource } from '$lib/services/gpuUtciRenderBridge';

export type GpuResidentCopyStatus = 'idle' | 'pending' | 'complete' | 'failed';

export type UtciSurfaceDiagnostics = {
	utciSurfaceSource?: string;
	selectedHourTransferCount?: number;
	dataTextureBuildCount?: number;
	renderOwnedSelectedHourBytes?: number;
	cpuPublishRequestId?: number;
	cpuPublishMonthIndex?: number;
	cpuPublishHourIndex?: number;
	cpuPublishTimeIndex?: number;
	cpuPublishSelectionKey?: string;
	gpuResidentCopyStatus?: GpuResidentCopyStatus;
	gpuResidentCopyError?: string;
	gpuResidentCopyRequestId?: number;
} & SelectedHourRenderTimingSubsteps;

export function getAcceptedGpuResidentKey(
	value: SelectedHourGpuResidentOutput | null
): string | null {
	if (!value) return null;
	return `${value.requestId}:${value.monthIndex}:${value.timeIndex}:${value.utciRange.min}:${value.utciRange.max}`;
}

export function isComputeBufferUtciSurface(mesh: Mesh | null): boolean {
	return mesh != null && getGpuNativeUtciSurfaceSource(mesh) === 'compute-buffer-selected-hour';
}

export function shouldRecreateComputeBufferUtciSurface(
	recreateDecision: SelectedHourRenderSurfaceMeshRecreateDecision
): {
	shouldRecreate: boolean;
	recreateDecision: SelectedHourRenderSurfaceMeshRecreateDecision;
} {
	return {
		shouldRecreate:
			recreateDecision.missingSurface ||
			recreateDecision.notComputeBufferSurface ||
			!recreateDecision.layoutCompatible,
		recreateDecision
	};
}

export function buildCpuPublicationDiagnostics(params: {
	mesh: Mesh | null;
	liveSelectedHourSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
}): Partial<UtciSurfaceDiagnostics> {
	if (
		params.mesh == null ||
		isComputeBufferUtciSurface(params.mesh) ||
		params.liveSelectedHourSurfaceIdentity == null
	) {
		return {};
	}

	return {
		utciSurfaceSource: 'cpu-uploaded-selected-hour',
		cpuPublishRequestId: params.liveSelectedHourSurfaceIdentity.requestId,
		cpuPublishMonthIndex: params.liveSelectedHourSurfaceIdentity.monthIndex,
		cpuPublishHourIndex: params.liveSelectedHourSurfaceIdentity.hourIndex,
		cpuPublishTimeIndex: params.liveSelectedHourSurfaceIdentity.timeIndex,
		cpuPublishSelectionKey: params.liveSelectedHourSurfaceIdentity.selectionKey
	};
}

export function buildUtciSurfaceDiagnostics(params: {
	mesh: Mesh | null;
	cpuPublicationDiagnostics?: Partial<UtciSurfaceDiagnostics>;
	gpuResidentCopyStatus: GpuResidentCopyStatus;
	gpuResidentCopyError?: string;
	gpuResidentCopyRequestId?: number;
	gpuResidentRenderTimings?: SelectedHourRenderTimingSubsteps;
}): UtciSurfaceDiagnostics {
	const cpuPublicationDiagnostics = params.cpuPublicationDiagnostics ?? {};
	return {
		utciSurfaceSource:
			cpuPublicationDiagnostics.utciSurfaceSource ??
			(params.mesh?.userData.utciSurfaceSource as string | undefined),
		selectedHourTransferCount: params.mesh?.userData.selectedHourTransferCount as
			| number
			| undefined,
		dataTextureBuildCount: params.mesh?.userData.dataTextureBuildCount as number | undefined,
		renderOwnedSelectedHourBytes: params.mesh?.userData.renderOwnedSelectedHourBytes as
			| number
			| undefined,
		cpuPublishRequestId: cpuPublicationDiagnostics.cpuPublishRequestId,
		cpuPublishMonthIndex: cpuPublicationDiagnostics.cpuPublishMonthIndex,
		cpuPublishHourIndex: cpuPublicationDiagnostics.cpuPublishHourIndex,
		cpuPublishTimeIndex: cpuPublicationDiagnostics.cpuPublishTimeIndex,
		cpuPublishSelectionKey: cpuPublicationDiagnostics.cpuPublishSelectionKey,
		gpuResidentCopyStatus: params.gpuResidentCopyStatus,
		gpuResidentCopyError: params.gpuResidentCopyError,
		gpuResidentCopyRequestId: params.gpuResidentCopyRequestId,
		...params.gpuResidentRenderTimings,
		renderPublication: copyRenderPublicationDiagnostics(
			params.gpuResidentRenderTimings?.renderPublication
		)
	};
}
