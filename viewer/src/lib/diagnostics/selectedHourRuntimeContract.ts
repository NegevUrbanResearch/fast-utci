export type SelectedHourRouteRole = 'main' | 'debug';
export type SelectedHourEngine = 'legacy-debug' | 'shared-host';
export type SelectedHourRenderTransport =
	| 'none'
	| 'cpu-uploaded-selected-hour'
	| 'compute-buffer-selected-hour';
export type SelectedHourReadbackReason = 'visible-fallback' | 'range' | 'tooltip' | 'comparison' | 'debug';
export type SelectedHourReadbackInstrumentation = 'instrumented' | 'not-instrumented';

export interface SelectedHourRuntimeContractInputs {
	route: SelectedHourRouteRole;
	selectedHourEngine: SelectedHourEngine;
	renderTransport?: SelectedHourRenderTransport;
	utciSurfaceSource?: SelectedHourRenderTransport;
	sameDeviceForComputeAndRender?: boolean;
	dataTextureBuildCount?: number;
	visibleSelectedHourReadbackCount?: number;
	readbackInstrumentation: SelectedHourReadbackInstrumentation;
	legacySelectedHourDispatchCount?: number;
	legacyScrubScheduleCount?: number;
	requestId?: number;
	sceneRequestId?: number;
	selectionKey?: string;
	sceneSelectionKey?: string;
	readbackReasons?: readonly SelectedHourReadbackReason[];
	readbackReasonCounts?: Partial<Record<SelectedHourReadbackReason, number>>;
}

export interface SelectedHourRuntimeContract {
	route: SelectedHourRouteRole;
	selectedHourEngine: SelectedHourEngine;
	renderTransport: SelectedHourRenderTransport;
	utciSurfaceSource: SelectedHourRenderTransport;
	sameDeviceForComputeAndRender: boolean;
	dataTextureBuildCount: number;
	visibleSelectedHourReadbackCount?: number;
	visibleSelectedHourReadbackCountInstrumented: boolean;
	readbackInstrumentation: SelectedHourReadbackInstrumentation;
	legacySelectedHourDispatchCount: number;
	legacyScrubScheduleCount: number;
	acceptedRequestId?: number;
	sceneRequestId?: number;
	selectionKey?: string;
	sceneSelectionKey?: string;
	readbackReasons: SelectedHourReadbackReason[];
	readbackReasonCounts: Partial<Record<SelectedHourReadbackReason, number>>;
	totalSelectedHourReadbackReasonCount: number;
	hasLegacyDebugOverlap: boolean;
	selectionMatchesScene: boolean;
	requestMatchesScene: boolean;
	strongVisibleGpuPath: boolean;
	visibleRenderPathAvoidsCpuReadback: boolean;
}

export function buildSelectedHourRuntimeContract(
	inputs: SelectedHourRuntimeContractInputs
): SelectedHourRuntimeContract {
	const renderTransport = inputs.renderTransport ?? 'none';
	const utciSurfaceSource = inputs.utciSurfaceSource ?? 'none';
	const dataTextureBuildCount = inputs.dataTextureBuildCount ?? 0;
	const visibleSelectedHourReadbackCount = inputs.visibleSelectedHourReadbackCount;
	const legacySelectedHourDispatchCount = inputs.legacySelectedHourDispatchCount ?? 0;
	const legacyScrubScheduleCount = inputs.legacyScrubScheduleCount ?? 0;
	const hasLegacyDebugOverlap =
		inputs.selectedHourEngine === 'shared-host' &&
		(legacySelectedHourDispatchCount > 0 || legacyScrubScheduleCount > 0);
	const selectionMatchesScene =
		inputs.selectionKey !== undefined &&
		inputs.sceneSelectionKey !== undefined &&
		inputs.selectionKey === inputs.sceneSelectionKey;
	const requestMatchesScene =
		typeof inputs.requestId === 'number' &&
		typeof inputs.sceneRequestId === 'number' &&
		inputs.requestId === inputs.sceneRequestId;
	const hasExplicitVisibleReadbackCount = typeof inputs.visibleSelectedHourReadbackCount === 'number';
	const hasExplicitDataTextureBuildCount = typeof inputs.dataTextureBuildCount === 'number';
	const readbackReasonCounts = { ...(inputs.readbackReasonCounts ?? {}) };
	const totalSelectedHourReadbackReasonCount =
		Object.values(readbackReasonCounts).reduce((sum, value) => sum + (value ?? 0), 0) ||
		inputs.readbackReasons?.length ||
		0;
	const visibleRenderPathAvoidsCpuReadback =
		inputs.readbackInstrumentation === 'instrumented' &&
		hasExplicitVisibleReadbackCount &&
		hasExplicitDataTextureBuildCount &&
		renderTransport === 'compute-buffer-selected-hour' &&
		utciSurfaceSource === 'compute-buffer-selected-hour' &&
		visibleSelectedHourReadbackCount === 0 &&
		dataTextureBuildCount === 0;
	const strongVisibleGpuPath =
		inputs.selectedHourEngine === 'shared-host' &&
		visibleRenderPathAvoidsCpuReadback &&
		inputs.sameDeviceForComputeAndRender === true &&
		selectionMatchesScene &&
		requestMatchesScene &&
		!hasLegacyDebugOverlap;

	return {
		route: inputs.route,
		selectedHourEngine: inputs.selectedHourEngine,
		renderTransport,
		utciSurfaceSource,
		sameDeviceForComputeAndRender: inputs.sameDeviceForComputeAndRender === true,
		dataTextureBuildCount,
		visibleSelectedHourReadbackCount,
		visibleSelectedHourReadbackCountInstrumented: hasExplicitVisibleReadbackCount,
		readbackInstrumentation: inputs.readbackInstrumentation,
		legacySelectedHourDispatchCount,
		legacyScrubScheduleCount,
		acceptedRequestId: inputs.requestId,
		sceneRequestId: inputs.sceneRequestId,
		selectionKey: inputs.selectionKey,
		sceneSelectionKey: inputs.sceneSelectionKey,
		readbackReasons: [...(inputs.readbackReasons ?? [])],
		readbackReasonCounts,
		totalSelectedHourReadbackReasonCount,
		hasLegacyDebugOverlap,
		selectionMatchesScene,
		requestMatchesScene,
		strongVisibleGpuPath,
		visibleRenderPathAvoidsCpuReadback
	};
}
