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
	import { getTooltipData } from '$lib/services/tooltipService';
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
	export let comparisonDisplayedAnalysis: Analysis | null | undefined;
	export let comparisonSceneTimeIndex: number | undefined;
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

	let tooltipVisible = false;
	let tooltipX = 0;
	let tooltipY = 0;
	let tooltipValue: number | null = null;
	let tooltipPosition: { x: number; y: number; z: number } | null = null;
	let lastTooltipUpdate = 0;
	const TOOLTIP_THROTTLE_MS = 16;
	let tooltipMotionSuppression = createTooltipMotionSuppressionState();

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

	type MainRouteTooltipProbeWindow = Window & {
		__mainRouteTooltipProbe__?: (() => MainRouteTooltipSnapshot | null) | undefined;
		__mainRouteTooltipAt__?:
			| ((clientX: number, clientY: number) => MainRouteTooltipSnapshot | null)
			| undefined;
		__mainRouteLastTooltip__?: MainRouteTooltipSnapshot | null | undefined;
	};

	function getTooltipSnapshotAtPoint(params: {
		canvas: HTMLCanvasElement | null;
		camera: PerspectiveCamera | undefined;
		mesh: Mesh | null;
		analysis: Analysis | null;
		metricType: MetricType;
		hourIndex: number;
		clientX: number;
		clientY: number;
	}): MainRouteTooltipSnapshot | null {
		const { canvas, camera, mesh, analysis, metricType, hourIndex, clientX, clientY } = params;
		if (!canvas || !camera || !mesh || !analysis) return null;
		const tooltipData = getTooltipData(
			{ clientX, clientY } as MouseEvent,
			camera,
			mesh,
			analysis,
			metricType,
			hourIndex,
			canvas.getBoundingClientRect()
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

	function getTooltipSnapshotForClientPoint(
		clientX: number,
		clientY: number
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
			comparisonMesh: getComparisonUtciMesh(),
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

		return getTooltipSnapshotAtPoint({
			canvas: canvasElement,
			camera: cameraRef,
			mesh: tooltipTarget.meshToRaycast,
			analysis: tooltipTarget.analysisToUse,
			metricType,
			hourIndex: tooltipTarget.tooltipHourIndex,
			clientX,
			clientY
		});
	}

	function computeTooltipProbe(params: {
		canvas: HTMLCanvasElement | null;
		camera: PerspectiveCamera | undefined;
		mesh: Mesh | null;
		snapshotAtClientPoint: (
			clientX: number,
			clientY: number
		) => MainRouteTooltipSnapshot | null;
	}): MainRouteTooltipSnapshot | null {
		const { canvas, camera, mesh, snapshotAtClientPoint } = params;
		if (!canvas || !camera || !mesh) return null;
		const positionAttribute = mesh.geometry?.getAttribute?.('position');
		if (!positionAttribute || positionAttribute.count <= 0) return null;

		const rect = canvas.getBoundingClientRect();
		const sampleStep = Math.max(1, Math.floor(positionAttribute.count / 256));
		const localPoint = new THREE.Vector3();
		const worldPoint = new THREE.Vector3();
		const projectedPoint = new THREE.Vector3();
		let bestProbe:
			| {
					clientX: number;
					clientY: number;
					positionIndex: number;
					value: number;
					position: { x: number; y: number; z: number };
					tooltipHourIndex: number;
					distanceToCenter: number;
			  }
			| null = null;

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
			if (!bestProbe || distanceToCenter < bestProbe.distanceToCenter) {
				bestProbe = {
					clientX,
					clientY,
					positionIndex: tooltipData.positionIndex,
					value: tooltipData.value,
					position: tooltipData.position,
					tooltipHourIndex: tooltipData.tooltipHourIndex,
					distanceToCenter
				};
			}
		}

		if (!bestProbe) {
			return null;
		}

		return {
			clientX: bestProbe.clientX,
			clientY: bestProbe.clientY,
			positionIndex: bestProbe.positionIndex,
			value: bestProbe.value,
			position: bestProbe.position,
			tooltipHourIndex: bestProbe.tooltipHourIndex
		};
	}

	function hideTooltip() {
		tooltipVisible = false;
		tooltipPosition = null;
		if (typeof window !== 'undefined' && diagnosticsEnabled) {
			(window as MainRouteTooltipProbeWindow).__mainRouteLastTooltip__ = null;
		}
	}

	function clearTooltipProbeGlobals() {
		if (typeof window === 'undefined') return;
		const probeWindow = window as MainRouteTooltipProbeWindow;
		probeWindow.__mainRouteTooltipProbe__ = undefined;
		probeWindow.__mainRouteTooltipAt__ = undefined;
		probeWindow.__mainRouteLastTooltip__ = undefined;
	}

	function handleTooltipMotionPointerDown() {
		tooltipMotionSuppression = setTooltipMotionPointerDown(
			tooltipMotionSuppression,
			true,
			performance.now()
		);
		hideTooltip();
	}

	function handleTooltipMotionPointerRelease() {
		const hadCanvasPointerInteraction = tooltipMotionSuppression.pointerDown;
		tooltipMotionSuppression = releaseTooltipMotionPointer(
			tooltipMotionSuppression,
			performance.now()
		);
		if (hadCanvasPointerInteraction) {
			hideTooltip();
		}
	}

	function handleTooltipMotionWheel() {
		cameraWheelEventCount += 1;
		tooltipMotionSuppression = armTooltipMotionSuppression(
			tooltipMotionSuppression,
			performance.now()
		);
		hideTooltip();
	}

	function handleMouseMove(event: MouseEvent) {
		const now = performance.now();
		const hoverPolicy = getMainRouteTooltipHoverPolicy({
			tooltipMotionSuppression,
			now,
			lastTooltipUpdate,
			throttleMs: TOOLTIP_THROTTLE_MS
		});
		if (hoverPolicy.shouldSuppress) {
			hideTooltip();
			return;
		}
		if (hoverPolicy.shouldThrottle) {
			return;
		}
		tooltipHoverSampleCount += 1;
		lastTooltipUpdate = hoverPolicy.nextTooltipUpdate;

		const tooltipData = getTooltipSnapshotForClientPoint(event.clientX, event.clientY);

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
		if (diagnosticsEnabled) {
			(window as MainRouteTooltipProbeWindow).__mainRouteTooltipProbe__ = () =>
				computeTooltipProbe({
					canvas: canvasElement,
					camera: cameraRef,
					mesh: baseMesh,
					snapshotAtClientPoint: getTooltipSnapshotForClientPoint
				});
			(window as MainRouteTooltipProbeWindow).__mainRouteTooltipAt__ = (
				clientX: number,
				clientY: number
			) => getTooltipSnapshotForClientPoint(clientX, clientY);
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
	}

	onDestroy(() => {
		clearTooltipProbeGlobals();
		detachHoverListeners();
		detachTooltipMotionListeners();
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
