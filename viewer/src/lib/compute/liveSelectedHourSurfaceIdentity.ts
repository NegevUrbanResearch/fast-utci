import type { SelectedHourGpuResidentOutput } from '$lib/compute/liveUtciSelectedHourSession';

export type LiveSelectedHourSurfaceIdentity = {
	requestId: number;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey: string;
	pendingRenderUpdateStartedAt: number | undefined;
	acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null;
};
