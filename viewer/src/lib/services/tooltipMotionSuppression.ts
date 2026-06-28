export const TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS = 400;

export type TooltipMotionSuppressionState = {
	pointerDown: boolean;
	suppressUntilMs: number;
};

export function createTooltipMotionSuppressionState(): TooltipMotionSuppressionState {
	return {
		pointerDown: false,
		suppressUntilMs: 0,
	};
}

export function armTooltipMotionSuppression(
	state: TooltipMotionSuppressionState,
	armedAtMs: number,
	windowMs = TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS,
): TooltipMotionSuppressionState {
	return {
		...state,
		suppressUntilMs: Math.max(state.suppressUntilMs, armedAtMs + windowMs),
	};
}

export function setTooltipMotionPointerDown(
	state: TooltipMotionSuppressionState,
	pointerDown: boolean,
	armedAtMs: number,
	windowMs = TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS,
): TooltipMotionSuppressionState {
	return armTooltipMotionSuppression(
		{
			...state,
			pointerDown,
		},
		armedAtMs,
		windowMs,
	);
}

export function releaseTooltipMotionPointer(
	state: TooltipMotionSuppressionState,
	releasedAtMs: number,
	windowMs = TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS,
): TooltipMotionSuppressionState {
	if (!state.pointerDown) {
		return state;
	}

	return armTooltipMotionSuppression(
		{
			...state,
			pointerDown: false,
		},
		releasedAtMs,
		windowMs,
	);
}

export function shouldSuppressTooltipMotion(
	state: TooltipMotionSuppressionState,
	nowMs: number,
): boolean {
	return state.pointerDown || nowMs <= state.suppressUntilMs;
}
