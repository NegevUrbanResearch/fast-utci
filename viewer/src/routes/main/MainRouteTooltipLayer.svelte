<script lang="ts">
	import { onDestroy } from 'svelte';
	import * as THREE from 'three';
	import type { Analysis } from '$lib/types/analysis';
	import type { MetricType } from '$lib/types/viewer';
	import MetricTooltip from '$lib/components/ui/MetricTooltip.svelte';
	import {
		createCanvasInteractionController,
		type CanvasInteractionController
	} from '$lib/components/viewer/canvasInteractionController';
	import {
		getTooltipData,
		getTooltipDataAsync,
		getTooltipProbeData
	} from '$lib/services/tooltipService';
	import {
		readF32MetricPointValue,
		sharedMetricPointReadbackCache
	} from '$lib/compute/gpu/metricPointReadback';
	import type { SelectedHourGpuResidentOutput } from '$lib/compute/selected-hour/liveUtciSelectedHourSession';
	import {
		createEmptyTooltipInteractionDiagnostics,
		recordTooltipInteractionMeasurement,
		type TooltipInteractionDiagnostics,
		type TooltipInteractionMeasurement,
		type TooltipMetricPointReadbackMeasurement
	} from '$lib/services/tooltipService';
	import {
		createEmptyCameraInteractionTelemetry,
		recordCameraInteractionFrame,
		type CameraInteractionDiagnostics,
		type CameraInteractionTelemetry
	} from '$lib/services/cameraInteractionTelemetry';
	import {
		armTooltipMotionSuppression,
		createTooltipMotionSuppressionState,
		releaseTooltipMotionPointer,
		setTooltipMotionPointerDown
	} from '$lib/services/tooltipMotionSuppression';
	import { getMainRouteTooltipHoverPolicy, resolveMainRouteTooltipTarget } from './tooltip';
	import type { Mesh, PerspectiveCamera } from 'three';

	export let canvasElement: HTMLCanvasElement | null;
	export let cameraRef: PerspectiveCamera | undefined;
	export let baseMesh: Mesh | null;
	export let baseDisplayedAnalysis: Analysis | null;
	export let baseSceneTimeIndex: number | undefined;
	export let basePendingGpuResidentOutput: SelectedHourGpuResidentOutput | null = null;
	export let comparisonDisplayedAnalysis: Analysis | null | undefined;
	export let comparisonSceneTimeIndex: number | undefined;
	export let comparisonPendingGpuResidentOutput: SelectedHourGpuResidentOutput | null = null;
	export let rendererDevice: GPUDevice | undefined;
	export let getComparisonUtciMesh: () => Mesh | null = () => null;
	export let useLiveUtciOnMainRoute: boolean;
	export let isComparing: boolean;
	export let mainViewportElement: HTMLElement | null;
	export let curtainPosition: number;
	export let viewerCurrentHour: number;
	export let metricType: MetricType;
	export let utciVisible: boolean;
	export let diagnosticsEnabled = false;
	export let tooltipHoverSampleCount = 0;
	export let cameraWheelEventCount = 0;
	export let tooltipInteractionDiagnostics: TooltipInteractionDiagnostics & {
		hoverSampleCount: number;
	} = {
		...createEmptyTooltipInteractionDiagnostics(false),
		hoverSampleCount: 0
	};
	export let cameraInteractionDiagnostics: CameraInteractionDiagnostics =
		createEmptyCameraInteractionTelemetry().diagnostics;

	let tooltipVisible = false;
	let tooltipX = 0;
	let tooltipY = 0;
	let tooltipValue: number | null = null;
	let tooltipPosition: { x: number; y: number; z: number } | null = null;
	let lastTooltipUpdate = 0;
	const TOOLTIP_THROTTLE_MS = 16;
	let tooltipMotionSuppression = createTooltipMotionSuppressionState();
	let cameraInteractionTelemetry: CameraInteractionTelemetry =
		createEmptyCameraInteractionTelemetry();
	let cameraInteractionFrameId: number | null = null;
	let cameraInteractionLastFrameTimeMs: number | null = null;
	let cameraInteractionPointerDown = false;
	let cameraInteractionSequenceActive = false;
	let cameraInteractionArmedUntilMs = 0;
	let hasCameraInteractionSnapshot = false;
	let hoverRequestToken = 0;
	let previousDiagnosticsEnabled = diagnosticsEnabled;
	const lastCameraInteractionPosition = new THREE.Vector3();
	const lastCameraInteractionQuaternion = new THREE.Quaternion();

	let hoverInteractionController: CanvasInteractionController | null = null;
	let hoverInteractionCanvas: HTMLCanvasElement | null = null;
	let tooltipMotionInteractionController: CanvasInteractionController | null = null;
	let tooltipMotionInteractionCanvas: HTMLCanvasElement | null = null;

	type MainRouteTooltipSnapshot = {
		clientX: number;
		clientY: number;
		positionIndex: number;
		value: number;
		position: { x: number; y: number; z: number };
		tooltipHourIndex: number;
	};

	type MainRouteTooltipProbeSnapshot = Omit<MainRouteTooltipSnapshot, 'value'>;

	type MainRouteTooltipProbeWindow = Window & {
		__mainRouteTooltipProbe__?: (() => MainRouteTooltipSnapshot | null) | undefined;
		__mainRouteTooltipProbePosition__?:
			| (() => MainRouteTooltipProbeSnapshot | null)
			| undefined;
		__mainRouteTooltipAt__?:
			| ((clientX: number, clientY: number) => MainRouteTooltipSnapshot | null)
			| undefined;
		__mainRouteLastTooltip__?: MainRouteTooltipSnapshot | null | undefined;
	};

	const CAMERA_INTERACTION_ARM_WINDOW_MS = 500;
	const CAMERA_INTERACTION_POSITION_EPSILON_SQ = 1e-8;
	const CAMERA_INTERACTION_QUATERNION_EPSILON = 1e-8;
	const MAIN_ROUTE_TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS = 900;
	const METRIC_POINT_READBACK_BYTES = 4;

	function publishTooltipDiagnostics(
		diagnostics: TooltipInteractionDiagnostics = tooltipInteractionDiagnostics
	): void {
		if (!diagnosticsEnabled) return;
		tooltipInteractionDiagnostics = {
			...diagnostics,
			hoverSampleCount: tooltipHoverSampleCount,
			metricPointReadbackCacheEntries: sharedMetricPointReadbackCache.size
		};
	}

	function recordMetricPointReadback(measurement: TooltipMetricPointReadbackMeasurement): void {
		if (!diagnosticsEnabled) return;
		const nextReadbackCount =
			tooltipInteractionDiagnostics.metricPointReadbackCount +
			(!measurement.cacheHit && measurement.success ? 1 : 0);
		const nextReadbackBytes =
			tooltipInteractionDiagnostics.metricPointReadbackBytes +
			(!measurement.cacheHit && measurement.success ? measurement.byteLength : 0);
		publishTooltipDiagnostics({
			...tooltipInteractionDiagnostics,
			metricPointReadbackCount: nextReadbackCount,
			metricPointReadbackBytes: nextReadbackBytes,
			metricPointReadbackLastBytes:
				!measurement.cacheHit && measurement.success
					? measurement.byteLength
					: tooltipInteractionDiagnostics.metricPointReadbackLastBytes,
			metricPointReadbackCacheHitCount:
				tooltipInteractionDiagnostics.metricPointReadbackCacheHitCount +
				(measurement.cacheHit ? 1 : 0),
			metricPointReadbackCacheMissCount:
				tooltipInteractionDiagnostics.metricPointReadbackCacheMissCount +
				(measurement.cacheHit ? 0 : 1),
			metricPointReadbackLastLatencyMs: measurement.latencyMs,
			metricPointReadbackMaxLatencyMs: Math.max(
				tooltipInteractionDiagnostics.metricPointReadbackMaxLatencyMs,
				measurement.latencyMs
			)
		});
	}

	function publishCameraDiagnostics(): void {
		if (!diagnosticsEnabled) return;
		cameraInteractionDiagnostics = cameraInteractionTelemetry.diagnostics;
	}

	function resetInteractionDiagnostics(): void {
		tooltipHoverSampleCount = 0;
		cameraWheelEventCount = 0;
		tooltipInteractionDiagnostics = {
			...createEmptyTooltipInteractionDiagnostics(false),
			hoverSampleCount: 0
		};
		cameraInteractionTelemetry = createEmptyCameraInteractionTelemetry();
		cameraInteractionDiagnostics = cameraInteractionTelemetry.diagnostics;
		cameraInteractionSequenceActive = false;
		cameraInteractionArmedUntilMs = 0;
		cameraInteractionLastFrameTimeMs = null;
		hasCameraInteractionSnapshot = false;
	}

	function armCameraInteractionTelemetry(
		windowMs = CAMERA_INTERACTION_ARM_WINDOW_MS
	): void {
		cameraInteractionArmedUntilMs = Math.max(
			cameraInteractionArmedUntilMs,
			performance.now() + windowMs
		);
	}

	function snapshotCameraInteractionTransform(camera: PerspectiveCamera): void {
		lastCameraInteractionPosition.copy(camera.position);
		lastCameraInteractionQuaternion.copy(camera.quaternion);
		hasCameraInteractionSnapshot = true;
	}

	function hasCameraInteractionTransformChanged(camera: PerspectiveCamera): boolean {
		return (
			camera.position.distanceToSquared(lastCameraInteractionPosition) >
				CAMERA_INTERACTION_POSITION_EPSILON_SQ ||
			1 - Math.abs(camera.quaternion.dot(lastCameraInteractionQuaternion)) >
				CAMERA_INTERACTION_QUATERNION_EPSILON
		);
	}

	function stopCameraInteractionTelemetryLoop(): void {
		if (cameraInteractionFrameId !== null) {
			cancelAnimationFrame(cameraInteractionFrameId);
			cameraInteractionFrameId = null;
		}
		cameraInteractionLastFrameTimeMs = null;
		hasCameraInteractionSnapshot = false;
	}

	function startCameraInteractionTelemetryLoop(): void {
		if (
			typeof window === 'undefined' ||
			!diagnosticsEnabled ||
			cameraInteractionFrameId !== null
		) {
			return;
		}

		const tick = (frameTimeMs: number) => {
			cameraInteractionFrameId = requestAnimationFrame(tick);
			const activeCamera = cameraRef;
			if (!activeCamera) {
				cameraInteractionLastFrameTimeMs = frameTimeMs;
				hasCameraInteractionSnapshot = false;
				return;
			}

			if (!hasCameraInteractionSnapshot) {
				snapshotCameraInteractionTransform(activeCamera);
				cameraInteractionLastFrameTimeMs = frameTimeMs;
				return;
			}

			const frameMs =
				cameraInteractionLastFrameTimeMs == null
					? 0
					: frameTimeMs - cameraInteractionLastFrameTimeMs;
			const cameraChanged = hasCameraInteractionTransformChanged(activeCamera);
			const interactionArmed =
				cameraInteractionPointerDown ||
				frameTimeMs <= cameraInteractionArmedUntilMs ||
				cameraInteractionSequenceActive;

			if (cameraChanged && interactionArmed) {
				cameraInteractionTelemetry = recordCameraInteractionFrame(
					cameraInteractionTelemetry,
					frameMs
				);
				cameraInteractionSequenceActive = true;
				armCameraInteractionTelemetry();
				tooltipMotionSuppression = armTooltipMotionSuppression(
					tooltipMotionSuppression,
					frameTimeMs,
					MAIN_ROUTE_TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS
				);
				hideTooltipForCameraMotion();
				publishCameraDiagnostics();
			} else if (!cameraChanged) {
				cameraInteractionSequenceActive = false;
			}

			snapshotCameraInteractionTransform(activeCamera);
			cameraInteractionLastFrameTimeMs = frameTimeMs;
		};

		cameraInteractionFrameId = requestAnimationFrame(tick);
	}

	async function getTooltipSnapshotAtPoint(params: {
		canvas: HTMLCanvasElement | null;
		camera: PerspectiveCamera | undefined;
		mesh: Mesh | null;
		analysis: Analysis | null;
		acceptedGpuResidentOutput: SelectedHourGpuResidentOutput | null;
		metricType: MetricType;
		hourIndex: number;
		clientX: number;
		clientY: number;
	}): Promise<MainRouteTooltipSnapshot | null> {
		const { canvas, camera, mesh, analysis, acceptedGpuResidentOutput, metricType, hourIndex, clientX, clientY } = params;
		if (!canvas || !camera || !mesh || !analysis) return null;
		const metricPointValueReader =
			metricType === 'shading_index' &&
			acceptedGpuResidentOutput?.metricType === 'shading_index' &&
			acceptedGpuResidentOutput.gpuOutputHandle &&
			rendererDevice
				? {
						monthIndex: acceptedGpuResidentOutput.monthIndex,
						requestId: acceptedGpuResidentOutput.requestId,
						ownerId: acceptedGpuResidentOutput.gpuOutputHandle.ownerId,
						readbackByteLength: METRIC_POINT_READBACK_BYTES,
						readMetricPointValue: async ({ positionIndex }: { positionIndex: number }) => {
							const value = await readF32MetricPointValue({
								device: rendererDevice as GPUDevice,
								sourceBuffer: acceptedGpuResidentOutput.gpuOutputHandle!.buffer,
								pointIndex: positionIndex,
								numPoints: analysis.data.numPositions
							});
							return value;
						}
				  }
				: undefined;
		const tooltipOptions =
			diagnosticsEnabled || metricPointValueReader
				? {
						onDiagnosticsSample: diagnosticsEnabled
							? (measurement: TooltipInteractionMeasurement) => {
									publishTooltipDiagnostics(
										recordTooltipInteractionMeasurement(
											tooltipInteractionDiagnostics,
											measurement
										)
									);
							  }
							: undefined,
						metricPointValueReader,
						onMetricPointReadbackSample: diagnosticsEnabled
							? recordMetricPointReadback
							: undefined
				  }
				: undefined;
		const tooltipData = await getTooltipDataAsync(
			{ clientX, clientY } as MouseEvent,
			camera,
			mesh,
			analysis,
			metricType,
			hourIndex,
			canvas.getBoundingClientRect(),
			tooltipOptions
		);
		return tooltipData
			? {
					clientX,
					clientY,
					positionIndex: tooltipData.positionIndex,
					value: tooltipData.value,
					position: tooltipData.position,
					tooltipHourIndex: hourIndex
			  }
			: null;
	}

	async function getTooltipSnapshotForClientPoint(
		clientX: number,
		clientY: number,
		comparisonMesh: Mesh | null = getComparisonUtciMesh()
	): Promise<MainRouteTooltipSnapshot | null> {
		if (
			!baseMesh ||
			!baseDisplayedAnalysis ||
			!utciVisible ||
			!canvasElement ||
			!cameraRef
		) {
			return null;
		}

		const tooltipTarget = resolveMainRouteTooltipTarget({
			baseMesh,
			baseAnalysis: baseDisplayedAnalysis,
			baseSceneTimeIndex,
			comparisonMesh,
			comparisonAnalysis: comparisonDisplayedAnalysis,
			comparisonSceneTimeIndex,
			useLiveUtciOnMainRoute,
			isComparing,
			mouseClientX: clientX,
			mainViewportRect: mainViewportElement
				? mainViewportElement.getBoundingClientRect()
				: null,
			curtainPosition,
			viewerCurrentHour
		});

		const usesComparison =
			tooltipTarget.meshToRaycast === comparisonMesh &&
			tooltipTarget.analysisToUse === comparisonDisplayedAnalysis;
		return getTooltipSnapshotAtPoint({
			canvas: canvasElement,
			camera: cameraRef,
			mesh: tooltipTarget.meshToRaycast,
			analysis: tooltipTarget.analysisToUse,
			acceptedGpuResidentOutput: usesComparison
				? comparisonPendingGpuResidentOutput
				: basePendingGpuResidentOutput,
			metricType,
			hourIndex: tooltipTarget.tooltipHourIndex,
			clientX,
			clientY
		});
	}

	function getTooltipSnapshotForClientPointSync(
		clientX: number,
		clientY: number,
		comparisonMesh: Mesh | null = getComparisonUtciMesh()
	): MainRouteTooltipSnapshot | null {
		if (
			!baseMesh ||
			!baseDisplayedAnalysis ||
			!utciVisible ||
			!canvasElement ||
			!cameraRef
		) {
			return null;
		}

		const tooltipTarget = resolveMainRouteTooltipTarget({
			baseMesh,
			baseAnalysis: baseDisplayedAnalysis,
			baseSceneTimeIndex,
			comparisonMesh,
			comparisonAnalysis: comparisonDisplayedAnalysis,
			comparisonSceneTimeIndex,
			useLiveUtciOnMainRoute,
			isComparing,
			mouseClientX: clientX,
			mainViewportRect: mainViewportElement
				? mainViewportElement.getBoundingClientRect()
				: null,
			curtainPosition,
			viewerCurrentHour
		});
		const tooltipData = getTooltipData(
			{ clientX, clientY } as MouseEvent,
			cameraRef,
			tooltipTarget.meshToRaycast,
			tooltipTarget.analysisToUse,
			metricType,
			tooltipTarget.tooltipHourIndex,
			canvasElement.getBoundingClientRect()
		);

		return tooltipData
			? {
					clientX,
					clientY,
					positionIndex: tooltipData.positionIndex,
					value: tooltipData.value,
					position: tooltipData.position,
					tooltipHourIndex: tooltipTarget.tooltipHourIndex
			  }
			: null;
	}

	function getTooltipProbeSnapshotForClientPointSync(
		clientX: number,
		clientY: number,
		comparisonMesh: Mesh | null = getComparisonUtciMesh()
	): MainRouteTooltipProbeSnapshot | null {
		if (
			!baseMesh ||
			!baseDisplayedAnalysis ||
			!utciVisible ||
			!canvasElement ||
			!cameraRef
		) {
			return null;
		}

		const tooltipTarget = resolveMainRouteTooltipTarget({
			baseMesh,
			baseAnalysis: baseDisplayedAnalysis,
			baseSceneTimeIndex,
			comparisonMesh,
			comparisonAnalysis: comparisonDisplayedAnalysis,
			comparisonSceneTimeIndex,
			useLiveUtciOnMainRoute,
			isComparing,
			mouseClientX: clientX,
			mainViewportRect: mainViewportElement
				? mainViewportElement.getBoundingClientRect()
				: null,
			curtainPosition,
			viewerCurrentHour
		});
		const tooltipProbe = getTooltipProbeData(
			{ clientX, clientY } as MouseEvent,
			cameraRef,
			tooltipTarget.meshToRaycast,
			tooltipTarget.analysisToUse,
			canvasElement.getBoundingClientRect()
		);

		return tooltipProbe
			? {
					clientX,
					clientY,
					positionIndex: tooltipProbe.positionIndex,
					position: tooltipProbe.position,
					tooltipHourIndex: tooltipTarget.tooltipHourIndex
			  }
			: null;
	}

	function computeTooltipProbe<T extends MainRouteTooltipProbeSnapshot>(params: {
		canvas: HTMLCanvasElement | null;
		camera: PerspectiveCamera | undefined;
		mesh: Mesh | null;
		snapshotAtClientPoint: (
			clientX: number,
			clientY: number
		) => T | null;
	}): T | null {
		const { canvas, camera, mesh, snapshotAtClientPoint } = params;
		if (!canvas || !camera || !mesh) return null;
		const positionAttribute = mesh.geometry?.getAttribute?.('position');
		if (!positionAttribute || positionAttribute.count <= 0) return null;

		const rect = canvas.getBoundingClientRect();
		const sampleStep = Math.max(1, Math.floor(positionAttribute.count / 256));
		const localPoint = new THREE.Vector3();
		const worldPoint = new THREE.Vector3();
		const projectedPoint = new THREE.Vector3();
		let bestProbe: T | null = null;
		let bestProbeDistanceToCenter = Number.POSITIVE_INFINITY;

		for (let index = 0; index < positionAttribute.count; index += sampleStep) {
			localPoint.fromBufferAttribute(positionAttribute, index);
			worldPoint.copy(localPoint);
			mesh.localToWorld(worldPoint);
			projectedPoint.copy(worldPoint).project(camera);
			if (
				!Number.isFinite(projectedPoint.x) ||
				!Number.isFinite(projectedPoint.y) ||
				!Number.isFinite(projectedPoint.z) ||
				projectedPoint.z < -1 ||
				projectedPoint.z > 1 ||
				projectedPoint.x < -1 ||
				projectedPoint.x > 1 ||
				projectedPoint.y < -1 ||
				projectedPoint.y > 1
			) {
				continue;
			}

			const clientX = rect.left + ((projectedPoint.x + 1) * 0.5) * rect.width;
			const clientY = rect.top + ((1 - projectedPoint.y) * 0.5) * rect.height;
			if (canvas.ownerDocument.elementFromPoint(clientX, clientY) !== canvas) {
				continue;
			}

			const tooltipData = snapshotAtClientPoint(clientX, clientY);
			if (!tooltipData) {
				continue;
			}

			const distanceToCenter = Math.abs(projectedPoint.x) + Math.abs(projectedPoint.y);
			if (!bestProbe || distanceToCenter < bestProbeDistanceToCenter) {
				bestProbe = {
					...tooltipData,
					clientX,
					clientY
				};
				bestProbeDistanceToCenter = distanceToCenter;
			}
		}

		if (!bestProbe) {
			return null;
		}

		return bestProbe;
	}

	function hideTooltip() {
		tooltipVisible = false;
		tooltipPosition = null;
		if (typeof window !== 'undefined' && diagnosticsEnabled) {
			(window as MainRouteTooltipProbeWindow).__mainRouteLastTooltip__ = null;
		}
	}

	function cancelPendingHover() {
		hoverRequestToken += 1;
	}

	function hideTooltipForCameraMotion() {
		cancelPendingHover();
		hideTooltip();
	}

	function clearTooltipProbeGlobals() {
		if (typeof window === 'undefined') return;
		const probeWindow = window as MainRouteTooltipProbeWindow;
		probeWindow.__mainRouteTooltipProbe__ = undefined;
		probeWindow.__mainRouteTooltipProbePosition__ = undefined;
		probeWindow.__mainRouteTooltipAt__ = undefined;
		probeWindow.__mainRouteLastTooltip__ = undefined;
	}

	function handleTooltipMotionPointerDown() {
		cancelPendingHover();
		cameraInteractionPointerDown = true;
		armCameraInteractionTelemetry();
		tooltipMotionSuppression = setTooltipMotionPointerDown(
			tooltipMotionSuppression,
			true,
			performance.now(),
			MAIN_ROUTE_TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS
		);
		hideTooltip();
	}

	function handleTooltipMotionPointerRelease() {
		cancelPendingHover();
		const hadCanvasPointerInteraction = tooltipMotionSuppression.pointerDown;
		tooltipMotionSuppression = releaseTooltipMotionPointer(
			tooltipMotionSuppression,
			performance.now(),
			MAIN_ROUTE_TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS
		);
		cameraInteractionPointerDown = false;
		if (hadCanvasPointerInteraction) {
			armCameraInteractionTelemetry();
			hideTooltip();
		}
	}

	function handleTooltipMotionWheel() {
		cancelPendingHover();
		cameraWheelEventCount += 1;
		if (diagnosticsEnabled) {
			cameraInteractionTelemetry = {
				...cameraInteractionTelemetry,
				diagnostics: {
					...cameraInteractionTelemetry.diagnostics,
					wheelEventCount: cameraWheelEventCount
				}
			};
			publishCameraDiagnostics();
		}
		armCameraInteractionTelemetry();
		tooltipMotionSuppression = armTooltipMotionSuppression(
			tooltipMotionSuppression,
			performance.now(),
			MAIN_ROUTE_TOOLTIP_MOTION_SUPPRESSION_WINDOW_MS
		);
		hideTooltip();
	}

	async function handleMouseMove(event: MouseEvent) {
		const requestToken = ++hoverRequestToken;
		const now = performance.now();
		if (diagnosticsEnabled) {
			publishTooltipDiagnostics({
				...tooltipInteractionDiagnostics,
				hoverAttemptCount: tooltipInteractionDiagnostics.hoverAttemptCount + 1
			});
		}
		const hoverPolicy = getMainRouteTooltipHoverPolicy({
			tooltipMotionSuppression,
			now,
			lastTooltipUpdate,
			throttleMs: TOOLTIP_THROTTLE_MS
		});
		if (hoverPolicy.shouldSuppress) {
			if (diagnosticsEnabled) {
				publishTooltipDiagnostics({
					...tooltipInteractionDiagnostics,
					suppressedHoverCount:
						tooltipInteractionDiagnostics.suppressedHoverCount + 1
				});
			}
			hideTooltip();
			return;
		}
		if (hoverPolicy.shouldThrottle) {
			if (diagnosticsEnabled) {
				publishTooltipDiagnostics({
					...tooltipInteractionDiagnostics,
					throttledHoverCount:
						tooltipInteractionDiagnostics.throttledHoverCount + 1
				});
			}
			return;
		}
		tooltipHoverSampleCount += 1;
		if (diagnosticsEnabled) {
			publishTooltipDiagnostics();
		}
		lastTooltipUpdate = hoverPolicy.nextTooltipUpdate;

		const comparisonMesh = getComparisonUtciMesh();
		const tooltipData = await getTooltipSnapshotForClientPoint(
			event.clientX,
			event.clientY,
			comparisonMesh
		);
		if (requestToken !== hoverRequestToken) {
			return;
		}

		if (tooltipData) {
			tooltipVisible = true;
			tooltipX = tooltipData.clientX;
			tooltipY = tooltipData.clientY;
			tooltipValue = tooltipData.value;
			tooltipPosition = tooltipData.position;
			if (typeof window !== 'undefined' && diagnosticsEnabled) {
				(window as MainRouteTooltipProbeWindow).__mainRouteLastTooltip__ =
					tooltipData;
			}
		} else {
			hideTooltip();
		}
	}

	function handleMouseLeave() {
		cancelPendingHover();
		hideTooltip();
	}

	function detachHoverListeners() {
		hoverInteractionController?.dispose();
		hoverInteractionController = null;
		hoverInteractionCanvas = null;
	}

	function detachTooltipMotionListeners() {
		tooltipMotionInteractionController?.dispose();
		tooltipMotionInteractionController = null;
		tooltipMotionInteractionCanvas = null;
	}

	$: if (typeof window !== 'undefined') {
		if (previousDiagnosticsEnabled !== diagnosticsEnabled) {
			previousDiagnosticsEnabled = diagnosticsEnabled;
			resetInteractionDiagnostics();
		}

		if (diagnosticsEnabled) {
			(window as MainRouteTooltipProbeWindow).__mainRouteTooltipProbe__ = () =>
				computeTooltipProbe({
					canvas: canvasElement,
					camera: cameraRef,
					mesh: baseMesh,
					snapshotAtClientPoint: getTooltipSnapshotForClientPointSync
				});
			(window as MainRouteTooltipProbeWindow).__mainRouteTooltipProbePosition__ = () =>
				computeTooltipProbe({
					canvas: canvasElement,
					camera: cameraRef,
					mesh: baseMesh,
					snapshotAtClientPoint: getTooltipProbeSnapshotForClientPointSync
				});
			(window as MainRouteTooltipProbeWindow).__mainRouteTooltipAt__ = (
				clientX: number,
				clientY: number
			) => getTooltipSnapshotForClientPointSync(clientX, clientY);
		} else {
			clearTooltipProbeGlobals();
		}

		if (hoverInteractionCanvas && hoverInteractionCanvas !== canvasElement) {
			detachHoverListeners();
		}
		if (canvasElement && hoverInteractionCanvas !== canvasElement) {
			hoverInteractionController = createCanvasInteractionController({
				canvas: canvasElement,
				onPointerMove: handleMouseMove,
				onPointerLeave: handleMouseLeave
			});
			hoverInteractionCanvas = canvasElement;
		}

		if (
			tooltipMotionInteractionCanvas &&
			tooltipMotionInteractionCanvas !== canvasElement
		) {
			detachTooltipMotionListeners();
		}
		if (canvasElement && tooltipMotionInteractionCanvas !== canvasElement) {
			tooltipMotionInteractionController = createCanvasInteractionController({
				canvas: canvasElement,
				windowTarget: window,
				onPointerDown: handleTooltipMotionPointerDown,
				onWheel: handleTooltipMotionWheel,
				onWindowPointerUp: handleTooltipMotionPointerRelease,
				onWindowPointerCancel: handleTooltipMotionPointerRelease
			});
			tooltipMotionInteractionCanvas = canvasElement;
		}
		if (diagnosticsEnabled) {
			startCameraInteractionTelemetryLoop();
		} else {
			stopCameraInteractionTelemetryLoop();
		}
	}

	onDestroy(() => {
		cancelPendingHover();
		clearTooltipProbeGlobals();
		detachHoverListeners();
		detachTooltipMotionListeners();
		stopCameraInteractionTelemetryLoop();
	});
</script>

<MetricTooltip
	visible={tooltipVisible}
	x={tooltipX}
	y={tooltipY}
	value={tooltipValue}
	position={tooltipPosition}
	{metricType}
/>
