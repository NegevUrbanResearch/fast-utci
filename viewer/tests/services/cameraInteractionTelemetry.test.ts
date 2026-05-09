import { describe, expect, it } from 'vitest';
import {
	CAMERA_INTERACTION_SAMPLE_BUFFER_SIZE,
	CAMERA_INTERACTION_SLOW_THRESHOLD_MS,
	createEmptyCameraInteractionTelemetry,
	recordCameraInteractionFrame
} from '$lib/services/cameraInteractionTelemetry';

describe('camera interaction telemetry', () => {
	it('starts with conservative empty diagnostics', () => {
		const telemetry = createEmptyCameraInteractionTelemetry();

		expect(telemetry.diagnostics.slowThresholdMs).toBe(CAMERA_INTERACTION_SLOW_THRESHOLD_MS);
		expect(telemetry.diagnostics.sampleCount).toBe(0);
		expect(telemetry.diagnostics.overBudgetCount).toBe(0);
		expect(telemetry.diagnostics.lastFrameMs).toBeNull();
		expect(telemetry.diagnostics.maxFrameMs).toBe(0);
		expect(telemetry.diagnostics.p95FrameMs).toBeNull();
		expect(telemetry.recentFrameMs).toEqual([]);
		expect(telemetry.maxSamples).toBe(CAMERA_INTERACTION_SAMPLE_BUFFER_SIZE);
	});

	it('tracks summary stats while keeping only a rolling frame buffer for percentile math', () => {
		const telemetry = [12, 20, 18, 40].reduce(
			(current, frameMs) => recordCameraInteractionFrame(current, frameMs),
			createEmptyCameraInteractionTelemetry({ maxSamples: 3 })
		);

		expect(telemetry.diagnostics.sampleCount).toBe(4);
		expect(telemetry.diagnostics.overBudgetCount).toBe(1);
		expect(telemetry.diagnostics.lastFrameMs).toBe(40);
		expect(telemetry.diagnostics.maxFrameMs).toBe(40);
		expect(telemetry.recentFrameMs).toEqual([20, 18, 40]);
		expect(telemetry.diagnostics.p95FrameMs).toBe(40);
	});
});
