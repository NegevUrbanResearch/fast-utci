<script lang="ts">
	import { onDestroy } from 'svelte';
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

	function hideTooltip() {
		tooltipVisible = false;
		tooltipPosition = null;
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

		if (
			!baseMesh ||
			!baseDisplayedAnalysis ||
			!utciVisible ||
			!canvasElement ||
			!cameraRef
		) {
			hideTooltip();
			return;
		}

		const canvasRect = canvasElement.getBoundingClientRect();
		const tooltipTarget = resolveMainRouteTooltipTarget({
			baseMesh,
			baseAnalysis: baseDisplayedAnalysis,
			baseSceneTimeIndex,
			comparisonMesh: getComparisonUtciMesh(),
			comparisonAnalysis: comparisonDisplayedAnalysis,
			comparisonSceneTimeIndex,
			useLiveUtciOnMainRoute,
			isComparing,
			mouseClientX: event.clientX,
			mainViewportRect: mainViewportElement
				? mainViewportElement.getBoundingClientRect()
				: null,
			curtainPosition,
			viewerCurrentHour
		});

		const tooltipData = getTooltipData(
			event,
			cameraRef,
			tooltipTarget.meshToRaycast,
			tooltipTarget.analysisToUse,
			metricType,
			tooltipTarget.tooltipHourIndex,
			canvasRect
		);

		if (tooltipData) {
			tooltipVisible = true;
			tooltipX = event.clientX;
			tooltipY = event.clientY;
			tooltipValue = tooltipData.value;
			tooltipPosition = tooltipData.position;
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
