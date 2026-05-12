import type { SelectedHourGpuResidentOutput } from '$lib/compute/selected-hour/liveUtciSelectedHourSession';

export type LiveSelectedHourSurfaceIdentity = {
	controllerIdentity: string;
	controllerInstanceId: number;
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey: string;
	pendingRenderUpdateStartedAt: number | undefined;
	acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null;
};
