import { describe, expect, it } from 'vitest';
import {
	createOnDemandScrubState,
	markOnDemandRequestCompleted,
	startOnDemandRequest
} from '$lib/compute/on-demand/onDemandScrubState';

describe('on-demand scrub state', () => {
	it('accepts the newest request and discards stale completions', () => {
		let state = createOnDemandScrubState();

		const first = startOnDemandRequest(state, { monthIndex: 0, timeIndex: 12 });
		state = first.state;
		expect(state.scrubSampleCount).toBe(1);
		const second = startOnDemandRequest(state, { monthIndex: 0, timeIndex: 17 });
		state = second.state;
		expect(state.scrubSampleCount).toBe(2);

		const stale = markOnDemandRequestCompleted(state, first.request);
		expect(stale.accepted).toBe(false);
		expect(stale.state.staleResultDiscardCount).toBe(1);
		expect(stale.state.scrubSampleCount).toBe(2);
		expect(stale.state.completedTimeIndex).toBeNull();

		const fresh = markOnDemandRequestCompleted(stale.state, second.request);
		expect(fresh.accepted).toBe(true);
		expect(fresh.state.activeRequestId).toBeNull();
		expect(fresh.state.inFlightCount).toBe(0);
		expect(fresh.state.scrubSampleCount).toBe(2);
		expect(fresh.state.completedTimeIndex).toBe(17);
		expect(fresh.state.completedMonthIndex).toBe(0);
	});

	it('rejects a duplicate completion for the newest request after it was already accepted', () => {
		let state = createOnDemandScrubState();

		const started = startOnDemandRequest(state, { monthIndex: 0, timeIndex: 17 });
		state = started.state;

		const completed = markOnDemandRequestCompleted(state, started.request);
		expect(completed.accepted).toBe(true);

		const duplicate = markOnDemandRequestCompleted(completed.state, started.request);
		expect(duplicate.accepted).toBe(false);
		expect(duplicate.state.staleResultDiscardCount).toBe(1);
		expect(duplicate.state.scrubSampleCount).toBe(1);
		expect(duplicate.state.completedRequestId).toBe(started.request.requestId);
		expect(duplicate.state.completedTimeIndex).toBe(17);
	});
});
