import type {
	F32MetricPeriod,
	F32MetricType,
	F32MetricValueLayout
} from '$lib/compute/on-demand/onDemandOutputFormat';

export type SelectedHourOutputSource = 'webgpu-on-demand-snapshot';

export interface SelectedHourOutputHandle {
	readonly buffer: GPUBuffer;
	readonly byteLength: number;
	requestId?: number;
	timeIndex?: number;
	readonly source: SelectedHourOutputSource;
	readonly ownerId: string;
	readonly metricType: F32MetricType;
	readonly valueLayout: F32MetricValueLayout;
	readonly period: F32MetricPeriod;
	disposed: boolean;
	dispose(): void;
}

export interface SelectedHourOutputHandleParams {
	buffer: GPUBuffer;
	byteLength: number;
	requestId?: number;
	timeIndex?: number;
	source: SelectedHourOutputSource;
	ownerId?: string;
	metricType?: F32MetricType;
	valueLayout?: F32MetricValueLayout;
	period?: F32MetricPeriod;
}

export interface F32MetricOutputHandleCandidate {
	source: SelectedHourOutputSource;
	ownerId: string;
	metricType: F32MetricType;
	valueLayout: F32MetricValueLayout;
	period: F32MetricPeriod;
	numPoints: number;
	gpuOutputHandle?: SelectedHourOutputHandle;
	outputBytes?: number;
	gpuBuffer?: unknown;
}

export interface ResolveOwnedF32MetricOutputHandleParams {
	output: F32MetricOutputHandleCandidate;
	metricType: F32MetricType;
	numPoints: number;
	source: SelectedHourOutputSource;
	ownerId?: string;
}

export function createSelectedHourOutputHandle(
	params: SelectedHourOutputHandleParams
): SelectedHourOutputHandle {
	const handle: SelectedHourOutputHandle = {
		buffer: params.buffer,
		byteLength: params.byteLength,
		requestId: params.requestId,
		timeIndex: params.timeIndex,
		source: params.source,
		ownerId: params.ownerId ?? params.source,
		metricType: params.metricType ?? 'utci',
		valueLayout: params.valueLayout ?? 'one-f32-per-point',
		period: params.period ?? { kind: 'time-index', index: params.timeIndex ?? -1 },
		disposed: false,
		dispose() {
			if (handle.disposed) return;
			handle.buffer.destroy();
			handle.disposed = true;
		}
	};
	return handle;
}

function periodsMatch(left: F32MetricPeriod, right: F32MetricPeriod): boolean {
	if (left.kind !== right.kind || left.index !== right.index) return false;
	if (left.kind === 'time-index' && right.kind === 'time-index') return true;
	if (left.kind === 'month-index' && right.kind === 'month-index') {
		return left.startTimeIndex === right.startTimeIndex && left.timeCount === right.timeCount;
	}
	return false;
}

export function resolveOwnedF32MetricOutputHandle(
	params: ResolveOwnedF32MetricOutputHandleParams
): SelectedHourOutputHandle {
	const { output } = params;
	const handle = output.gpuOutputHandle;
	if (!handle) {
		throw new Error('F32 metric output summary requires a GPU output handle');
	}
	if (output.metricType !== params.metricType) {
		throw new Error(
			`F32 metric output summary metric type mismatch output=${output.metricType} requested=${params.metricType}`
		);
	}
	if (handle.metricType !== params.metricType) {
		throw new Error(
			`F32 metric output summary handle metric type mismatch handle=${handle.metricType} requested=${params.metricType}`
		);
	}
	if (output.valueLayout !== 'one-f32-per-point' || handle.valueLayout !== 'one-f32-per-point') {
		throw new Error('F32 metric output summary requires one-f32-per-point value layout');
	}
	if (!periodsMatch(output.period, handle.period)) {
		throw new Error('F32 metric output summary period mismatch between output and handle');
	}
	if (output.source !== handle.source) {
		throw new Error(
			`F32 metric output summary source mismatch between output and handle output=${output.source} handle=${handle.source}`
		);
	}
	if (output.ownerId !== handle.ownerId) {
		throw new Error(
			`F32 metric output summary owner mismatch between output and handle output=${output.ownerId} handle=${handle.ownerId}`
		);
	}
	if (output.numPoints !== params.numPoints) {
		throw new Error(
			`F32 metric output summary numPoints mismatch output=${output.numPoints} requested=${params.numPoints}`
		);
	}
	if (handle.source !== params.source) {
		throw new Error(
			`F32 metric output summary source mismatch handle=${handle.source} requested=${params.source}`
		);
	}
	if (params.ownerId !== undefined && handle.ownerId !== params.ownerId) {
		throw new Error(
			`F32 metric output summary owner mismatch handle=${handle.ownerId} requested=${params.ownerId}`
		);
	}
	if (handle.disposed) {
		throw new Error('F32 metric output summary cannot use a disposed GPU output handle');
	}
	const requiredBytes = params.numPoints * 4;
	if (handle.byteLength < requiredBytes) {
		throw new Error(
			`F32 metric output summary GPU output handle is too small (required=${requiredBytes}, actual=${handle.byteLength})`
		);
	}
	if (typeof output.outputBytes === 'number' && output.outputBytes < requiredBytes) {
		throw new Error(
			`F32 metric output summary outputBytes is too small (required=${requiredBytes}, actual=${output.outputBytes})`
		);
	}
	if (output.gpuBuffer !== undefined && output.gpuBuffer !== handle.buffer) {
		throw new Error(
			'F32 metric output summary raw GPU buffer does not match the GPU output handle buffer'
		);
	}
	return handle;
}

export function disposeSelectedHourOutputHandle(
	handle: SelectedHourOutputHandle | null | undefined
): void {
	handle?.dispose();
}
