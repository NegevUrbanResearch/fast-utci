import { describe, expect, it } from 'vitest';
import {
	TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS,
	armTooltipMotionSuppression,
	createTooltipMotionSuppressionState,
	releaseTooltipMotionPointer,
	setTooltipMotionPointerDown,
	shouldSuppressTooltipMotion
} from '$lib/services/tooltipMotionSuppression';

describe('tooltip motion suppression', () => {
	it('suppresses hover while the pointer is down and through the shared settle window', () => {
		const armedAtMs = 100;
		const releasedAtMs = 160;

		const pointerDownState = setTooltipMotionPointerDown(
			createTooltipMotionSuppressionState(),
			true,
			armedAtMs
		);
		expect(shouldSuppressTooltipMotion(pointerDownState, armedAtMs)).toBe(true);

		const releasedState = releaseTooltipMotionPointer(pointerDownState, releasedAtMs);
		expect(shouldSuppressTooltipMotion(releasedState, releasedAtMs + 1)).toBe(true);
		expect(
			shouldSuppressTooltipMotion(
				releasedState,
				releasedAtMs + TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS - 1
			)
		).toBe(true);
		expect(
			shouldSuppressTooltipMotion(
				releasedState,
				releasedAtMs + TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS + 1
			)
		).toBe(false);
	});

	it('arms suppression for wheel-style interactions without requiring a pointer hold', () => {
		const armedAtMs = 250;
		const state = armTooltipMotionSuppression(
			createTooltipMotionSuppressionState(),
			armedAtMs
		);

		expect(shouldSuppressTooltipMotion(state, armedAtMs)).toBe(true);
		expect(
			shouldSuppressTooltipMotion(state, armedAtMs + TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS + 1)
		).toBe(false);
	});

	it('ignores pointer release events that did not end an active canvas-originated pointer interaction', () => {
		const releasedAtMs = 600;
		const initialState = createTooltipMotionSuppressionState();

		const releasedState = releaseTooltipMotionPointer(initialState, releasedAtMs);

		expect(releasedState).toEqual(initialState);
		expect(shouldSuppressTooltipMotion(releasedState, releasedAtMs + 1)).toBe(false);
	});
});
