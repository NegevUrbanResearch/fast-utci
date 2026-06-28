import { buildActiveCellSpatialArrays } from '$lib/gis/innovationDistrictExport';
import type { Analysis } from '$lib/types/analysis';
import type { SelectedHourGpuResidentOutput } from '$lib/compute/selected-hour/liveUtciSelectedHourSession';

export const FAST_UTCI_COLLECTOR_EXPORT_QUERY_FLAG = 'fastUtciCollectorExport';

export type FastUtciCollectorExportWindow = Window & {
	__fastUtciCollectorExport?: (() => Promise<FastUtciCollectorExportResult>) | undefined;
};

export interface FastUtciCollectorCurrentState {
	analysis: Analysis | null;
	output: SelectedHourGpuResidentOutput | null;
	device?: GPUDevice;
}

export interface FastUtciCollectorExportResult {
	metadata: {
		analysisId: string;
		metricType: 'utci' | 'shading_index';
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
		valueLayout: string;
		period: SelectedHourGpuResidentOutput['output']['period'];
		activeMask: {
			source: 'base+road';
			activeCanonicalIndices: Uint32Array;
			canonicalPointCount: number;
			activePointCount: number;
			inactivePointCount: number;
			activePointRatio: number;
			checksum: string;
			signature: string;
		};
	};
	canonicalIndices: Uint32Array;
	positions: Float32Array;
	values: Float32Array;
	surfaceFlags: Uint8Array;
}

export type ReadF32MetricOutput = (params: {
	device: GPUDevice;
	buffer: GPUBuffer;
	numPoints: number;
	byteLength?: number;
}) => Promise<Float32Array>;

export async function readF32MetricOutput(params: {
	device: GPUDevice;
	buffer: GPUBuffer;
	numPoints: number;
	byteLength?: number;
}): Promise<Float32Array> {
	const bytes = params.numPoints * Float32Array.BYTES_PER_ELEMENT;
	if ((params.byteLength ?? bytes) < bytes) {
		throw new Error(
			`Collector metric output is too small: expected ${bytes} bytes, got ${params.byteLength ?? 0}.`
		);
	}

	const readback = params.device.createBuffer({
		size: bytes,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
	});
	try {
		const encoder = params.device.createCommandEncoder();
		encoder.copyBufferToBuffer(params.buffer, 0, readback, 0, bytes);
		params.device.queue.submit([encoder.finish()]);
		await readback.mapAsync(GPUMapMode.READ);
		const mapped = new Float32Array(readback.getMappedRange(0, bytes));
		const values = new Float32Array(mapped.length);
		values.set(mapped);
		readback.unmap();
		return values;
	} finally {
		readback.destroy();
	}
}

export async function buildFastUtciCollectorExport(params: {
	analysis: Analysis | null;
	output: SelectedHourGpuResidentOutput | null;
	device?: GPUDevice;
	readF32MetricOutput?: ReadF32MetricOutput;
}): Promise<FastUtciCollectorExportResult> {
	if (params.analysis == null) {
		throw new Error('Collector export requires an active analysis');
	}
	if (params.output == null) {
		throw new Error('Collector export requires an active GPU-resident selected-hour output');
	}
	if (params.device == null) {
		throw new Error('Collector export requires the route WebGPU device');
	}

	const metricType = params.output.metricType;
	if (metricType !== 'utci' && metricType !== 'shading_index') {
		throw new Error('Collector export requires a UTCI or Shading Index output');
	}
	if (params.output.output.metricType !== metricType) {
		throw new Error('Collector export output metric type mismatch');
	}
	if (params.output.output.valueLayout !== 'one-f32-per-point') {
		throw new Error('Collector export requires one-f32-per-point output layout');
	}
	if (!params.output.gpuOutputHandle || params.output.gpuOutputHandle.disposed) {
		throw new Error('Collector export requires a live GPU output handle');
	}

	const spatial = buildActiveCellSpatialArrays(params.analysis);
	if (params.output.output.numPoints !== spatial.activeMask.activePointCount) {
		throw new Error('Collector export output row count does not match active mask row count');
	}

	const reader = params.readF32MetricOutput ?? readF32MetricOutput;
	const values = await reader({
		device: params.device,
		buffer: params.output.gpuOutputHandle.buffer,
		numPoints: spatial.activeMask.activePointCount,
		byteLength: params.output.output.outputBytes
	});
	if (values.length !== spatial.activeMask.activePointCount) {
		throw new Error('Collector export value row count does not match active mask row count');
	}

	return {
		metadata: {
			analysisId:
				params.analysis.metadata.source_analysis_id ??
				'Innovation-District/innovation_district_webgpu',
			metricType,
			monthIndex: params.output.monthIndex,
			hourIndex: params.output.hourIndex,
			timeIndex: params.output.timeIndex,
			valueLayout: params.output.output.valueLayout,
			period: params.output.output.period,
			activeMask: {
				...spatial.activeMask,
				activeCanonicalIndices: new Uint32Array(spatial.canonicalIndices)
			}
		},
		canonicalIndices: spatial.canonicalIndices,
		positions: spatial.positions,
		values,
		surfaceFlags: spatial.surfaceFlags
	};
}

export function installFastUtciCollectorExport(params: {
	win: FastUtciCollectorExportWindow;
	searchParams: URLSearchParams;
	getCurrent: () => FastUtciCollectorCurrentState | null;
	readF32MetricOutput?: ReadF32MetricOutput;
}): () => void {
	if (params.searchParams.get(FAST_UTCI_COLLECTOR_EXPORT_QUERY_FLAG) !== '1') {
		params.win.__fastUtciCollectorExport = undefined;
		return () => {
			params.win.__fastUtciCollectorExport = undefined;
		};
	}

	const exportFn = async () => {
		const current = params.getCurrent();
		return buildFastUtciCollectorExport({
			analysis: current?.analysis ?? null,
			output: current?.output ?? null,
			device: current?.device,
			readF32MetricOutput: params.readF32MetricOutput
		});
	};
	params.win.__fastUtciCollectorExport = exportFn;

	return () => {
		if (params.win.__fastUtciCollectorExport === exportFn) {
			params.win.__fastUtciCollectorExport = undefined;
		}
	};
}
