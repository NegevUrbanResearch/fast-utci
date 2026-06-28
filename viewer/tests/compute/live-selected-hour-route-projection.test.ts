import { describe, expect, it, vi } from 'vitest';
import type { Analysis } from '$lib/types/analysis';
import type { LiveSelectedHourControllerState } from '$lib/compute/selected-hour/liveSelectedHourController';
import type { LiveSelectedHourRouteState } from '$lib/compute/selected-hour/liveSelectedHourRouteHost';
import { projectMainRouteLiveSceneState } from '$lib/compute/selected-hour/liveSelectedHourRouteProjection';
import type { SelectedHourGpuResidentOutput } from '$lib/compute/selected-hour/liveUtciSelectedHourSession';
import { createMainRouteRenderPublicationProjectionTracker } from '../../src/routes/main/liveSelectedHour';

function createFullDayAnalysis(label: string): Analysis {
	return {
		metadata: {
			analysis_type: 'full_day',
			model_file: `${label}.glb`,
			num_positions: 2,
			hours: Array.from({ length: 24 }, (_, hour) => `${hour}:00`),
			grid_size: 1,
			coordinate_system: 'xy_ground',
			num_months: 12,
			source_analysis_id: label,
			utci_range: { min: 10, max: 40 }
		},
		data: {
			numPositions: 2,
			numHours: 24,
			positions: new Float32Array([0, 0, 0, 1, 0, 0]),
			utciByHour: Array.from({ length: 24 }, () => new Float32Array([10, 40]))
		}
	};
}

function createControllerState(): LiveSelectedHourControllerState {
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

function createState(): LiveSelectedHourRouteState {
	return {
		base: createControllerState(),
		comparison: createControllerState(),
		baseDisplayAnalysis: null,
		comparisonDisplayAnalysis: undefined,
		primaryAcceptedVisibleSurface: null,
		baseAcceptedVisibleSurface: null,
		comparisonAcceptedVisibleSurface: null,
		acceptedRequestId: undefined,
		acceptedSelectionKey: undefined,
		acceptedVisibleAtMs: undefined,
		baseHasVisibleLiveSurface: false,
		comparisonHasVisibleLiveSurface: false,
		baseSceneSurfaceIdentity: null,
		comparisonSceneSurfaceIdentity: undefined,
		baseSurfaceIdentity: null,
		comparisonSurfaceIdentity: null,
		baseRenderContext: null,
		comparisonRenderContext: undefined,
		baseReady: false,
		comparisonReady: true,
		comparisonSourceAnalysisId: null,
		liveUnifiedRange: null
	};
}

describe('projectMainRouteLiveSceneState', () => {
	it('preserves the non-live route projection', () => {
		const baseAnalysis = createFullDayAnalysis('base');
		const projection = projectMainRouteLiveSceneState({
			useLiveUtciOnMainRoute: false,
			isComparing: false,
			baseAnalysis,
			comparisonAnalysis: null,
			liveRouteState: createState()
		});

		expect(projection.baseDisplayedAnalysis).toBe(baseAnalysis);
		expect(projection.baseSceneAnalysis).toBe(baseAnalysis);
		expect(projection.baseSceneRenderContext).toBeNull();
		expect(projection.basePendingGpuResidentOutput).toBeNull();
		expect(projection.comparisonSceneAnalysis).toBeUndefined();
	});

	it('preserves the non-live comparison projection for ComparisonRenderer', () => {
		const baseAnalysis = createFullDayAnalysis('base');
		const comparisonAnalysis = createFullDayAnalysis('winter');
		const projection = projectMainRouteLiveSceneState({
			useLiveUtciOnMainRoute: false,
			isComparing: true,
			baseAnalysis,
			comparisonAnalysis,
			liveRouteState: createState()
		});

		expect(projection.comparisonRendererDisplayAnalysis).toBe(comparisonAnalysis);
		expect(projection.comparisonSceneAnalysis).toBe(comparisonAnalysis);
		expect(projection.comparisonSceneRenderContext).toBeNull();
		expect(projection.comparisonSceneSurfaceIdentity).toBeNull();
		expect(projection.comparisonLiveReady).toBe(true);
	});

	it('passes bootstrap GPU selected-hour props to the base scene before visible publication', () => {
		const baseAnalysis = createFullDayAnalysis('base');
		const bootstrapAnalysis = createFullDayAnalysis('bootstrap-selected');
		const acceptedGpuResidentOutput = {
			requestId: 7,
			monthIndex: 7,
			hourIndex: 12,
			timeIndex: 180,
			utciRange: { min: 10, max: 40 },
			output: {
				format: 'f32-utci',
				numPoints: 2,
				timeIndex: 180,
				gpuBuffer: { label: 'gpu-buffer' } as unknown as GPUBuffer
			}
		} satisfies SelectedHourGpuResidentOutput;
		const liveRouteState = createState();
		liveRouteState.baseDisplayAnalysis = bootstrapAnalysis;
		liveRouteState.baseHasVisibleLiveSurface = false;
		liveRouteState.baseSurfaceIdentity = null;
		liveRouteState.baseSceneSurfaceIdentity = {
			controllerIdentity: 'base-controller',
			controllerInstanceId: 1,
			requestId: 7,
			monthIndex: 7,
			hourIndex: 12,
			timeIndex: 180,
			selectionKey: 'Ben-Gurion/20250815_grid_2m_fullday|7|12',
			pendingRenderUpdateStartedAt: 100,
			acceptedGpuResidentOutput
		};
		liveRouteState.baseRenderContext = {
			analysis: bootstrapAnalysis,
			monthIndex: 7,
			hourIndex: 12,
			timeIndex: 180,
			selectionKey: 'Ben-Gurion/20250815_grid_2m_fullday|7|12',
			publicationPhase: 'initial',
			colorMode: 'discrete',
			metricType: 'utci',
			rangeOverride: null
		};

		const projected = projectMainRouteLiveSceneState({
			useLiveUtciOnMainRoute: true,
			isComparing: false,
			baseAnalysis,
			comparisonAnalysis: null,
			liveRouteState
		});

		expect(projected.baseSceneAnalysis).toBe(bootstrapAnalysis);
		expect(projected.basePendingGpuResidentOutput).toBe(acceptedGpuResidentOutput);
		expect(projected.baseSceneSurfaceIdentity?.requestId).toBe(7);
		expect(projected.baseSceneRenderContext?.timeIndex).toBe(180);
		expect(projected.baseDisplayedAnalysis).toBe(bootstrapAnalysis);
		expect(projected.baseSceneRenderContext?.analysis).toBe(bootstrapAnalysis);
		expect(projected.baseSceneAnalysis?.metadata.grid_size).toBe(
			bootstrapAnalysis.metadata.grid_size
		);
		expect(projected.baseSceneAnalysis?.data.numPositions).toBe(
			bootstrapAnalysis.data.numPositions
		);
	});

	it('passes bootstrap GPU selected-hour props to the comparison scene before visible publication', () => {
		const baseAnalysis = createFullDayAnalysis('base');
		const comparisonAnalysis = createFullDayAnalysis('winter');
		const acceptedGpuResidentOutput = {
			requestId: 11,
			monthIndex: 1,
			hourIndex: 9,
			timeIndex: 42,
			utciRange: { min: 8, max: 35 },
			output: {
				format: 'f32-utci',
				numPoints: 2,
				timeIndex: 42,
				gpuBuffer: { label: 'comparison-gpu-buffer' } as unknown as GPUBuffer
			}
		} satisfies SelectedHourGpuResidentOutput;
		const liveRouteState = createState();
		liveRouteState.comparisonDisplayAnalysis = comparisonAnalysis;
		liveRouteState.comparisonHasVisibleLiveSurface = false;
		liveRouteState.comparisonSurfaceIdentity = null;
		liveRouteState.comparisonSceneSurfaceIdentity = {
			controllerIdentity: 'comparison-controller',
			controllerInstanceId: 2,
			requestId: 11,
			monthIndex: 1,
			hourIndex: 9,
			timeIndex: 42,
			selectionKey: 'winter|1|9',
			pendingRenderUpdateStartedAt: 120,
			acceptedGpuResidentOutput
		};
		liveRouteState.comparisonRenderContext = {
			analysis: comparisonAnalysis,
			monthIndex: 1,
			hourIndex: 9,
			timeIndex: 42,
			selectionKey: 'winter|1|9',
			publicationPhase: 'initial',
			colorMode: 'discrete',
			metricType: 'utci',
			rangeOverride: null
		};

		const projected = projectMainRouteLiveSceneState({
			useLiveUtciOnMainRoute: true,
			isComparing: true,
			baseAnalysis,
			comparisonAnalysis,
			liveRouteState
		});

		expect(projected.comparisonSceneAnalysis).toBe(comparisonAnalysis);
		expect(projected.comparisonPendingGpuResidentOutput).toBe(acceptedGpuResidentOutput);
		expect(projected.comparisonSceneSurfaceIdentity?.requestId).toBe(11);
		expect(projected.comparisonSceneRenderContext?.timeIndex).toBe(42);
	});

	it('passes CPU fallback bootstrap props to the base scene before visible publication', () => {
		const baseAnalysis = createFullDayAnalysis('base');
		const bootstrapAnalysis = createFullDayAnalysis('base-cpu-selected');
		const liveRouteState = createState();
		liveRouteState.baseDisplayAnalysis = bootstrapAnalysis;
		liveRouteState.baseHasVisibleLiveSurface = false;
		liveRouteState.baseSurfaceIdentity = null;
		liveRouteState.baseSceneSurfaceIdentity = {
			controllerIdentity: 'base-controller',
			controllerInstanceId: 3,
			requestId: 13,
			monthIndex: 7,
			hourIndex: 15,
			timeIndex: 183,
			selectionKey: 'base-cpu|7|15',
			pendingRenderUpdateStartedAt: 140,
			acceptedGpuResidentOutput: null
		};
		liveRouteState.baseRenderContext = {
			analysis: bootstrapAnalysis,
			monthIndex: 7,
			hourIndex: 15,
			timeIndex: 183,
			selectionKey: 'base-cpu|7|15',
			publicationPhase: 'initial',
			colorMode: 'discrete',
			metricType: 'utci',
			rangeOverride: null
		};

		const projected = projectMainRouteLiveSceneState({
			useLiveUtciOnMainRoute: true,
			isComparing: false,
			baseAnalysis,
			comparisonAnalysis: null,
			liveRouteState
		});

		expect(projected.baseSceneAnalysis).toBe(bootstrapAnalysis);
		expect(projected.baseSceneRenderContext).toBe(liveRouteState.baseRenderContext);
		expect(projected.baseSceneSurfaceIdentity).toBe(liveRouteState.baseSceneSurfaceIdentity);
		expect(projected.basePendingGpuResidentOutput).toBeNull();
		expect(projected.comparisonSceneAnalysis).toBeUndefined();
		expect(projected.comparisonSceneRenderContext).toBeUndefined();
		expect(projected.comparisonSceneSurfaceIdentity).toBeUndefined();
	});

	it('passes CPU fallback bootstrap props to the active comparison scene before visible publication', () => {
		const baseAnalysis = createFullDayAnalysis('base');
		const comparisonAnalysis = createFullDayAnalysis('winter-cpu-selected');
		const liveRouteState = createState();
		liveRouteState.comparisonDisplayAnalysis = comparisonAnalysis;
		liveRouteState.comparisonHasVisibleLiveSurface = false;
		liveRouteState.comparisonSurfaceIdentity = null;
		liveRouteState.comparisonSceneSurfaceIdentity = {
			controllerIdentity: 'comparison-controller',
			controllerInstanceId: 4,
			requestId: 17,
			monthIndex: 1,
			hourIndex: 9,
			timeIndex: 42,
			selectionKey: 'winter-cpu|1|9',
			pendingRenderUpdateStartedAt: 160,
			acceptedGpuResidentOutput: null
		};
		liveRouteState.comparisonRenderContext = {
			analysis: comparisonAnalysis,
			monthIndex: 1,
			hourIndex: 9,
			timeIndex: 42,
			selectionKey: 'winter-cpu|1|9',
			publicationPhase: 'initial',
			colorMode: 'discrete',
			metricType: 'utci',
			rangeOverride: null
		};

		const projected = projectMainRouteLiveSceneState({
			useLiveUtciOnMainRoute: true,
			isComparing: true,
			baseAnalysis,
			comparisonAnalysis,
			liveRouteState
		});

		expect(projected.comparisonSceneAnalysis).toBe(comparisonAnalysis);
		expect(projected.comparisonSceneRenderContext).toBe(liveRouteState.comparisonRenderContext);
		expect(projected.comparisonSceneSurfaceIdentity).toBe(
			liveRouteState.comparisonSceneSurfaceIdentity
		);
		expect(projected.comparisonPendingGpuResidentOutput).toBeNull();
	});

	it('projects the published live surface once visible state is current', () => {
		const liveAnalysis = createFullDayAnalysis('selected');
		const liveRouteState = createState();
		liveRouteState.baseDisplayAnalysis = liveAnalysis;
		liveRouteState.baseHasVisibleLiveSurface = true;
		liveRouteState.baseSurfaceIdentity = {
			controllerIdentity: 'base-controller',
			controllerInstanceId: 5,
			requestId: 2,
			monthIndex: 7,
			hourIndex: 14,
			timeIndex: 182,
			selectionKey: 'base|7|14',
			pendingRenderUpdateStartedAt: undefined,
			acceptedGpuResidentOutput: null
		};
		liveRouteState.baseRenderContext = {
			analysis: liveAnalysis,
			monthIndex: 7,
			hourIndex: 14,
			timeIndex: 182,
			selectionKey: 'base|7|14',
			publicationPhase: 'scrub',
			colorMode: 'discrete',
			metricType: 'utci',
			rangeOverride: null
		};
		liveRouteState.baseReady = true;

		const projection = projectMainRouteLiveSceneState({
			useLiveUtciOnMainRoute: true,
			isComparing: false,
			baseAnalysis: createFullDayAnalysis('base'),
			comparisonAnalysis: null,
			liveRouteState
		});

		expect(projection.baseLiveReady).toBe(true);
		expect(projection.baseHasVisibleLiveSurface).toBe(true);
		expect(projection.baseSceneAnalysis).toBe(liveAnalysis);
		expect(projection.baseSceneRenderContext).toBe(liveRouteState.baseRenderContext);
		expect(projection.baseSceneSurfaceIdentity).toBe(liveRouteState.baseSurfaceIdentity);
	});

	it('passes the pending GPU selection to the base scene while replacing a visible surface', () => {
		const liveAnalysis = createFullDayAnalysis('selected');
		const pendingAnalysis = createFullDayAnalysis('selected-pending');
		const pendingGpuResidentOutput = {
			requestId: 3,
			monthIndex: 7,
			hourIndex: 1,
			timeIndex: 181,
			utciRange: { min: 11, max: 39 },
			output: {
				format: 'f32-utci',
				numPoints: 2,
				timeIndex: 181,
				gpuBuffer: { label: 'pending-gpu-buffer' } as unknown as GPUBuffer
			}
		} satisfies SelectedHourGpuResidentOutput;
		const liveRouteState = createState();
		liveRouteState.baseDisplayAnalysis = liveAnalysis;
		liveRouteState.baseHasVisibleLiveSurface = true;
		liveRouteState.baseSurfaceIdentity = {
			controllerIdentity: 'base-controller',
			controllerInstanceId: 6,
			requestId: 2,
			monthIndex: 7,
			hourIndex: 0,
			timeIndex: 180,
			selectionKey: 'base|7|0',
			pendingRenderUpdateStartedAt: undefined,
			acceptedGpuResidentOutput: null
		};
		liveRouteState.baseSceneSurfaceIdentity = {
			controllerIdentity: 'base-controller',
			controllerInstanceId: 7,
			requestId: 3,
			monthIndex: 7,
			hourIndex: 1,
			timeIndex: 181,
			selectionKey: 'base|7|1',
			pendingRenderUpdateStartedAt: 200,
			acceptedGpuResidentOutput: pendingGpuResidentOutput
		};
		liveRouteState.baseRenderContext = {
			analysis: pendingAnalysis,
			monthIndex: 7,
			hourIndex: 1,
			timeIndex: 181,
			selectionKey: 'base|7|1',
			publicationPhase: 'scrub',
			colorMode: 'discrete',
			metricType: 'utci',
			rangeOverride: null
		};
		liveRouteState.baseReady = false;

		const projection = projectMainRouteLiveSceneState({
			useLiveUtciOnMainRoute: true,
			isComparing: false,
			baseAnalysis: createFullDayAnalysis('base'),
			comparisonAnalysis: null,
			liveRouteState
		});

		expect(projection.baseHasVisibleLiveSurface).toBe(true);
		expect(projection.baseSceneSurfaceIdentity?.requestId).toBe(3);
		expect(projection.basePendingGpuResidentOutput).toBe(pendingGpuResidentOutput);
		expect(projection.baseSceneRenderContext).toBe(liveRouteState.baseRenderContext);
	});

	it('keeps repeated render-publication projections passive for unchanged route identities', () => {
		const tracker = createMainRouteRenderPublicationProjectionTracker();
		const surfaceIdentity = {
			controllerIdentity: 'base-controller',
			controllerInstanceId: 9,
			requestId: 21,
			monthIndex: 7,
			hourIndex: 14,
			timeIndex: 182,
			selectionKey: 'base|7|14',
			pendingRenderUpdateStartedAt: undefined,
			acceptedGpuResidentOutput: null
		};
		const timings = {
			renderPublication: {
				renderPublicationVersion: 1,
				renderPublicationPath: 'compute-buffer-selected-hour',
				renderPublicationPhase: 'scrub',
				renderPublicationMeshAction: 'reused'
			}
		} as const;
		const nowSpy = vi
			.spyOn(performance, 'now')
			.mockReturnValueOnce(101)
			.mockReturnValueOnce(501)
			.mockReturnValueOnce(601)
			.mockReturnValueOnce(701)
			.mockReturnValueOnce(801)
			.mockReturnValueOnce(901);

		try {
			const first = tracker.apply({
				enabled: true,
				timings,
				projectedSceneSurfaceIdentity: surfaceIdentity,
				publishedSurfaceIdentity: surfaceIdentity,
				sceneRenderContextTimeIndex: 182,
				selectedTimeIndex: 182
			});
			const second = tracker.apply({
				enabled: true,
				timings,
				projectedSceneSurfaceIdentity: surfaceIdentity,
				publishedSurfaceIdentity: surfaceIdentity,
				sceneRenderContextTimeIndex: 182,
				selectedTimeIndex: 182
			});

			expect(second).toBe(first);
			expect(second?.renderPublication).toBe(first?.renderPublication);
			expect(second?.renderPublication?.renderPublicationTimeline).toMatchObject({
				routePendingSurfaceExposedAtMs: 501,
				routeProjectedAtMs: 501,
				routeProjectionEvaluationStartedAtMs: 101,
				routeProjectionEvaluationCompletedAtMs: 601
			});
			expect(performance.now).toHaveBeenCalledTimes(3);
		} finally {
			nowSpy.mockRestore();
		}
	});
});
