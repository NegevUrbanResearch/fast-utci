import { describe, expect, it, vi } from 'vitest';

import {
	createAcceptedGpuResidentOutputReleaseNotifier,
	type AcceptedGpuResidentOutputReleaseCallback
} from '$lib/components/scene/acceptedGpuResidentOutputRelease';

describe('acceptedGpuResidentOutputRelease', () => {
	it('notifies release exactly once with controller identity and instance id', () => {
		const callback = vi.fn<AcceptedGpuResidentOutputReleaseCallback>();
		const notify = createAcceptedGpuResidentOutputReleaseNotifier({
			callback,
			componentName: 'test',
			controllerIdentity: 'controller-a',
			controllerInstanceId: 7,
			requestId: 11,
			monthIndex: 2,
			timeIndex: 51
		});

		expect(notify('copy-complete')).toBe(true);
		expect(notify('copy-failed')).toBe(false);
		expect(notify('superseded')).toBe(false);

		expect(callback).toHaveBeenCalledTimes(1);
		expect(callback).toHaveBeenCalledWith({
			controllerIdentity: 'controller-a',
			controllerInstanceId: 7,
			requestId: 11,
			monthIndex: 2,
			timeIndex: 51,
			reason: 'copy-complete'
		});
	});

	it('suppresses release when controller identity is missing', () => {
		const callback = vi.fn<AcceptedGpuResidentOutputReleaseCallback>();
		const notify = createAcceptedGpuResidentOutputReleaseNotifier({
			callback,
			componentName: 'test',
			controllerIdentity: undefined,
			controllerInstanceId: 7,
			requestId: 12,
			monthIndex: 3,
			timeIndex: 60
		});

		expect(notify('superseded')).toBe(false);
		expect(callback).not.toHaveBeenCalled();
	});

	it('suppresses release when controller instance id is missing', () => {
		const callback = vi.fn<AcceptedGpuResidentOutputReleaseCallback>();
		const notify = createAcceptedGpuResidentOutputReleaseNotifier({
			callback,
			componentName: 'test',
			controllerIdentity: 'controller-b',
			controllerInstanceId: undefined,
			requestId: 13,
			monthIndex: 4,
			timeIndex: 72
		});

		expect(notify('copy-failed')).toBe(false);
		expect(callback).not.toHaveBeenCalled();
	});
});
