export interface OnDemandSelection {
	monthIndex: number;
	timeIndex: number;
}

export interface OnDemandScrubRequest extends OnDemandSelection {
	requestId: number;
}

export interface OnDemandScrubState {
	nextRequestId: number;
	activeRequestId: number | null;
	completedRequestId: number | null;
	scrubSampleCount: number;
	selectedMonthIndex: number | null;
	selectedTimeIndex: number | null;
	completedMonthIndex: number | null;
	completedTimeIndex: number | null;
	staleResultDiscardCount: number;
	inFlightCount: number;
}

export function createOnDemandScrubState(): OnDemandScrubState {
	return {
		nextRequestId: 1,
		activeRequestId: null,
		completedRequestId: null,
		scrubSampleCount: 0,
		selectedMonthIndex: null,
		selectedTimeIndex: null,
		completedMonthIndex: null,
		completedTimeIndex: null,
		staleResultDiscardCount: 0,
		inFlightCount: 0
	};
}

export function startOnDemandRequest(
	state: OnDemandScrubState,
	selection: OnDemandSelection
): { state: OnDemandScrubState; request: OnDemandScrubRequest } {
	const request: OnDemandScrubRequest = {
		requestId: state.nextRequestId,
		...selection
	};

	return {
		request,
		state: {
			...state,
			nextRequestId: state.nextRequestId + 1,
			activeRequestId: request.requestId,
			scrubSampleCount: state.scrubSampleCount + 1,
			selectedMonthIndex: selection.monthIndex,
			selectedTimeIndex: selection.timeIndex,
			inFlightCount: state.inFlightCount + 1
		}
	};
}

export function markOnDemandRequestCompleted(
	state: OnDemandScrubState,
	request: OnDemandScrubRequest
): { state: OnDemandScrubState; accepted: boolean } {
	const inFlightCount = Math.max(0, state.inFlightCount - 1);

	if (request.requestId !== state.activeRequestId) {
		return {
			accepted: false,
			state: {
				...state,
				inFlightCount,
				staleResultDiscardCount: state.staleResultDiscardCount + 1
			}
		};
	}

	return {
		accepted: true,
		state: {
			...state,
			activeRequestId: null,
			inFlightCount,
			completedRequestId: request.requestId,
			completedMonthIndex: request.monthIndex,
			completedTimeIndex: request.timeIndex
		}
	};
}
