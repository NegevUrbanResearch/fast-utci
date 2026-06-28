import type { Analysis } from '$lib/types/analysis';
import type { Group } from 'three';
import type {
	LiveSelectedHourController,
	LiveSelectedHourControllerRequest,
	LiveSelectedHourControllerState
} from '$lib/compute/selected-hour/liveSelectedHourController';
import type { LiveSelectedHourRouteInputs } from '$lib/compute/selected-hour/liveSelectedHourRouteHost';

export function createFullDayAnalysis(params: {
	label: string;
	sourceAnalysisId?: string;
	baseMin?: number;
	baseMax?: number;
}): Analysis {
	const baseMin = params.baseMin ?? 10;
	const baseMax = params.baseMax ?? 30;

	return {
		metadata: {
			analysis_type: 'full_day',
			num_positions: 2,
			hours: Array.from({ length: 24 }, (_, hour) => `${String(hour).padStart(2, '0')}:00`),
			utci_range: { min: baseMin, max: baseMax },
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: `${params.label}.glb`,
			source_analysis_id: params.sourceAnalysisId ?? params.label,
			num_months: 12,
			hour_statistics: Array.from({ length: 288 }, (_, index) => ({
				min: baseMin + (index % 24),
				max: baseMax + (index % 24),
				mean: (baseMin + baseMax) / 2 + (index % 24)
			}))
		},
		data: {
			numPositions: 2,
			numHours: 24,
			positions: new Float32Array([0, 0, 0, 1, 0, 0]),
			utciByHour: Array.from({ length: 288 }, () => new Float32Array([baseMin, baseMax]))
		}
	};
}

export function createInitialLiveControllerState(): LiveSelectedHourControllerState {
	return {
		analysis: null,
		acceptedGpuResidentOutput: null,
		surfaceIdentity: null,
		acceptedVisibleSurface: null,
		acceptedRequestId: undefined,
		acceptedSelectionKey: undefined,
		acceptedVisibleAtMs: undefined,
		visibleSelectedHourReadbackCount: undefined,
		readbackInstrumentation: 'not-instrumented',
		selectedHourReadbackReasons: [],
		selectedHourReadbackReasonCounts: {},
		loading: false,
		error: null,
		renderTransport: 'idle',
		sameDeviceForComputeAndRender: null,
		pendingRenderUpdateStartedAt: undefined,
		renderSurfaceDiagnostics: {},
		ready: false,
		renderReady: false,
		awaitingGpuSurface: false
	};
}

export function cloneLiveControllerState(
	state: LiveSelectedHourControllerState
): LiveSelectedHourControllerState {
	return {
		...state,
		surfaceIdentity: state.surfaceIdentity ? { ...state.surfaceIdentity } : null,
		acceptedVisibleSurface: state.acceptedVisibleSurface
			? { ...state.acceptedVisibleSurface }
			: null,
		selectedHourReadbackReasons: [...state.selectedHourReadbackReasons],
		selectedHourReadbackReasonCounts: { ...state.selectedHourReadbackReasonCounts },
		renderSurfaceDiagnostics: { ...state.renderSurfaceDiagnostics }
	};
}

export function createMetricRecordingControllerFactory() {
	const records: Array<{ requests: LiveSelectedHourControllerRequest[] }> = [];

	return {
		records,
		createController(): LiveSelectedHourController {
			let state = createInitialLiveControllerState();
			const listeners = new Set<(state: LiveSelectedHourControllerState) => void>();
			const record = { requests: [] as LiveSelectedHourControllerRequest[] };
			records.push(record);

			function emit(): void {
				const snapshot = cloneLiveControllerState(state);
				for (const listener of listeners) {
					listener(snapshot);
				}
			}

			return {
				getState() {
					return cloneLiveControllerState(state);
				},
				subscribe(listener) {
					listeners.add(listener);
					return () => {
						listeners.delete(listener);
					};
				},
				async requestSelection(request) {
					record.requests.push(request);
					const requestId = record.requests.length;
					const selectionKey =
						request.selectionKey ??
						`${requestId}:${request.monthIndex}:${request.hourIndex}:${request.timeIndex}`;
					state = {
						...state,
						analysis: request.sessionConfig.base,
						surfaceIdentity: {
							controllerIdentity: 'controller',
							controllerInstanceId: 0,
							requestId,
							monthIndex: request.monthIndex,
							hourIndex: request.hourIndex,
							timeIndex: request.timeIndex,
							selectionKey,
							pendingRenderUpdateStartedAt: undefined,
							acceptedGpuResidentOutput: null
						},
						acceptedVisibleSurface: {
							requestId,
							selectionKey,
							visibleAtMs: requestId * 100
						},
						acceptedRequestId: requestId,
						acceptedSelectionKey: selectionKey,
						acceptedVisibleAtMs: requestId * 100,
						renderTransport:
							request.metricType === 'shading_index'
								? 'compute-buffer-selected-hour'
								: 'cpu-uploaded-selected-hour',
						sameDeviceForComputeAndRender: request.metricType === 'shading_index',
						ready: true,
						renderReady: true
					};
					emit();
					return {
						accepted: true,
						state: cloneLiveControllerState(state)
					};
				},
				async handleRenderSurfaceDiagnostics() {
					return;
				},
				releaseAcceptedGpuResidentOutput() {
					return;
				},
				dispose() {
					listeners.clear();
					state = createInitialLiveControllerState();
				}
			};
		}
	};
}

export function createLiveRouteInputs(params: {
	metricType: LiveSelectedHourRouteInputs['metricType'];
	baseAnalysis?: Analysis;
	baseModel?: Group;
	comparisonAnalysis?: Analysis | null;
	comparisonModel?: Group | null;
	rendererDevice?: GPUDevice;
	selectionKey?: string;
	monthIndex?: number;
	hourIndex?: number;
	timeIndex?: number;
}): LiveSelectedHourRouteInputs {
	const monthIndex = params.monthIndex ?? 7;
	const hourIndex = params.hourIndex ?? 12;
	const timeIndex = params.timeIndex ?? monthIndex * 24 + hourIndex;
	const baseAnalysis =
		params.baseAnalysis ??
		createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
	const comparisonAnalysis = params.comparisonAnalysis ?? null;

	return {
		enabled: true,
		analysisId: 'Ben-Gurion/base',
		baseAnalysis,
		baseModel: params.baseModel ?? ({} as Group),
		metricType: params.metricType,
		selection: {
			monthIndex,
			hourIndex,
			timeIndex,
			selectionKey:
				params.selectionKey ?? `Ben-Gurion/base|${params.metricType}|${monthIndex}|${hourIndex}`
		},
		colorMode: 'normalized',
		utciRenderMode: 'auto',
		rendererBackend: 'webgpu',
		rendererDevice:
			params.rendererDevice ?? ({ label: 'renderer' } as unknown as GPUDevice),
		utciSurfaceBackend: 'gpuNative',
		comparison: {
			active: comparisonAnalysis != null,
			analysisId: comparisonAnalysis ? 'Ben-Gurion/comparison' : null,
			sourceAnalysis: comparisonAnalysis,
			model: comparisonAnalysis ? (params.comparisonModel ?? ({} as Group)) : null,
			rendererDevice: comparisonAnalysis
				? ({ label: 'comparison-renderer' } as unknown as GPUDevice)
				: undefined
		}
	};
}
