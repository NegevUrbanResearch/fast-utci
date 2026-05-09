export interface CameraInteractionDiagnostics {
	slowThresholdMs: number;
	sampleCount: number;
	overBudgetCount: number;
	lastFrameMs: number | null;
	maxFrameMs: number;
	p95FrameMs: number | null;
}

export interface CameraInteractionTelemetry {
	diagnostics: CameraInteractionDiagnostics;
	recentFrameMs: number[];
	maxSamples: number;
}

export const CAMERA_INTERACTION_SLOW_THRESHOLD_MS = 20;
export const CAMERA_INTERACTION_SAMPLE_BUFFER_SIZE = 120;

export function createEmptyCameraInteractionTelemetry(options?: {
	maxSamples?: number;
	slowThresholdMs?: number;
}): CameraInteractionTelemetry {
	const slowThresholdMs = options?.slowThresholdMs ?? CAMERA_INTERACTION_SLOW_THRESHOLD_MS;
	return {
		diagnostics: {
			slowThresholdMs,
			sampleCount: 0,
			overBudgetCount: 0,
			lastFrameMs: null,
			maxFrameMs: 0,
			p95FrameMs: null
		},
		recentFrameMs: [],
		maxSamples: options?.maxSamples ?? CAMERA_INTERACTION_SAMPLE_BUFFER_SIZE
	};
}

export function recordCameraInteractionFrame(
	telemetry: CameraInteractionTelemetry,
	frameMs: number
): CameraInteractionTelemetry {
	if (!Number.isFinite(frameMs) || frameMs <= 0) {
		return telemetry;
	}

	const recentFrameMs =
		telemetry.recentFrameMs.length >= telemetry.maxSamples
			? [...telemetry.recentFrameMs.slice(1), frameMs]
			: [...telemetry.recentFrameMs, frameMs];
	const overBudget = frameMs > telemetry.diagnostics.slowThresholdMs;

	return {
		...telemetry,
		recentFrameMs,
		diagnostics: {
			...telemetry.diagnostics,
			sampleCount: telemetry.diagnostics.sampleCount + 1,
			overBudgetCount: telemetry.diagnostics.overBudgetCount + (overBudget ? 1 : 0),
			lastFrameMs: frameMs,
			maxFrameMs: Math.max(telemetry.diagnostics.maxFrameMs, frameMs),
			p95FrameMs: getPercentile(recentFrameMs, 95)
		}
	};
}

function getPercentile(values: number[], percentile: number): number | null {
	if (values.length === 0) return null;

	const sorted = [...values].sort((a, b) => a - b);
	const index = Math.min(
		sorted.length - 1,
		Math.max(0, Math.ceil((percentile / 100) * sorted.length) - 1)
	);
	return sorted[index] ?? null;
}
