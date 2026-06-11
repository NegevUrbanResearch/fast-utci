<svelte:options immutable={false} />

<script lang="ts">
	import { analysisStore } from "$lib/stores/analysisStore";
	import { viewerStore, setCurrentHour, setCurrentMonth } from "$lib/stores/viewerStore";
	import { getDayRingConicGradient } from "$lib/utils/dayGradient";
	import { getYearRingConicGradient } from "$lib/utils/yearGradient";
	import {
		normalizeHourEntries,
		derivePhaseMarkers,
		getDayStateForIndex,
		getPositionForIndex,
		getIndexFromPointerEvent,
		DIAL_SIZE_PX,
		DIAL_CENTER,
		HANDLE_RADIUS,
		LABEL_RADIUS,
	} from "$lib/utils/radialTimePickerState";
	import ColorModeToggle from "./ColorModeToggle.svelte";

	const dayRingGradient = getDayRingConicGradient();
	const yearRingGradient = getYearRingConicGradient();
	const circumference = 2 * Math.PI * HANDLE_RADIUS;
	export let fixedMode: "day" | "month" | null = null;
	let mode: "day" | "month" = fixedMode ?? "day";
	/** Handle stops short of the hour labels (at 110) so it doesn’t sit on the text */
	let dialEl: HTMLDivElement | null = null;
	let isDragging = false;
	let isHoveringDial = false;

	$: showLabels = isDragging || isHoveringDial;

	const MONTH_LABELS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];
	const MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

	$: analysis = $analysisStore;
	$: hours = analysis?.metadata.hours ?? [];
	$: normalizedHours = normalizeHourEntries(hours);
	$: totalHours = normalizedHours.length;
	$: hourValues = normalizedHours.map((e) => e.value);
	$: maxHourIndex = Math.max(totalHours - 1, 0);
	$: currentHourIndex = Math.min($viewerStore.currentHour, maxHourIndex);
	$: currentMonthIndex = Math.min(Math.max(0, $viewerStore.currentMonth ?? 7), 11);
	$: totalSegments = mode === "day" ? totalHours : 12;
	$: currentIndex = mode === "day" ? currentHourIndex : currentMonthIndex;
	$: currentHourLabel =
		totalHours > 0
			? (normalizedHours[currentHourIndex]?.label ??
				`${currentHourIndex.toString().padStart(2, "0")}:00`)
			: "--:--";
	$: currentMonthLabel = MONTH_NAMES[currentMonthIndex];
	$: centerLabel = mode === "day" ? currentHourLabel : currentMonthLabel;
	$: phaseMarkers = derivePhaseMarkers(analysis, totalHours, hourValues);
	$: currentState = getDayStateForIndex(
		currentHourIndex,
		phaseMarkers,
		hourValues,
		totalHours,
	);
	$: progressDash =
		totalSegments > 0 ? (currentIndex / totalSegments) * circumference : 0;
	$: ringGradient = mode === "day" ? dayRingGradient : yearRingGradient;

	function updateFromPointer(event: PointerEvent) {
		if (!dialEl || totalSegments === 0) return;
		const rect = dialEl.getBoundingClientRect();
		const targetIndex = getIndexFromPointerEvent(
			rect,
			event.clientX,
			event.clientY,
			totalSegments,
		);
		if (mode === "day") {
			if (targetIndex !== $viewerStore.currentHour) setCurrentHour(targetIndex);
		} else {
			if (targetIndex !== ($viewerStore.currentMonth ?? 7)) setCurrentMonth(targetIndex);
		}
	}

	function handlePointerDown(event: PointerEvent) {
		if (!dialEl) return;
		isDragging = true;
		dialEl.setPointerCapture(event.pointerId);
		event.preventDefault();
		updateFromPointer(event);
	}

	function handlePointerMove(event: PointerEvent) {
		if (!isDragging) return;
		updateFromPointer(event);
	}

		function handlePointerUp(event: PointerEvent) {
		if (!dialEl) return;
		isDragging = false;
		if (dialEl.hasPointerCapture(event.pointerId)) {
			dialEl.releasePointerCapture(event.pointerId);
		}
		// Hide labels when drag ends and pointer is outside the dial
		const rect = dialEl.getBoundingClientRect();
		const inside =
			event.clientX >= rect.left &&
			event.clientX <= rect.right &&
			event.clientY >= rect.top &&
			event.clientY <= rect.bottom;
		if (!inside) isHoveringDial = false;
	}

	function handlePointerEnter() {
		isHoveringDial = true;
	}
	function handlePointerLeave() {
		if (!isDragging) isHoveringDial = false;
	}

	function handleKeyDown(event: KeyboardEvent) {
		if (totalSegments === 0) return;
		let nextIndex: number | null = null;

		if (event.key === "ArrowRight" || event.key === "ArrowUp") {
			nextIndex = (currentIndex + 1) % totalSegments;
		} else if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
			nextIndex = (currentIndex - 1 + totalSegments) % totalSegments;
		} else if (event.key === "Home") {
			nextIndex = 0;
		} else if (event.key === "End") {
			nextIndex = totalSegments - 1;
		}

		if (nextIndex !== null) {
			event.preventDefault();
			if (mode === "day") setCurrentHour(nextIndex);
			else setCurrentMonth(nextIndex);
		}
	}

	$: if (mode === "day" && totalHours > 0 && $viewerStore.currentHour !== currentHourIndex) {
		setCurrentHour(currentHourIndex);
	}
	$: if (mode === "month" && ($viewerStore.currentMonth ?? 7) !== currentMonthIndex) {
		setCurrentMonth(currentMonthIndex);
	}
	$: if (fixedMode && mode !== fixedMode) {
		mode = fixedMode;
	}
</script>

<div class="radial-wrapper">
	<div class="radial-panel">
		<div
			class="radial-dial"
			style="--dial-size-px: {DIAL_SIZE_PX}px; --dial-center-px: {DIAL_CENTER}px;"
			role="slider"
			tabindex={totalHours > 0 ? 0 : -1}
			aria-valuemin="0"
			aria-valuemax={Math.max(totalSegments - 1, 0)}
			aria-valuenow={currentIndex}
			aria-valuetext={mode === "day" ? `Time ${currentHourLabel}` : `Month ${currentMonthLabel}`}
			aria-label={mode === "day" ? "Select analysis hour" : "Select month"}
			bind:this={dialEl}
			on:pointerdown={handlePointerDown}
			on:pointermove={handlePointerMove}
			on:pointerup={handlePointerUp}
			on:pointerenter={handlePointerEnter}
			on:pointerleave={handlePointerLeave}
			on:keydown={handleKeyDown}
		>
			<!-- Full dial face: one conic gradient from center to edge (inner and outer as one) -->
			<div
				class="day-ring"
				style="width: {DIAL_SIZE_PX}px; height: {DIAL_SIZE_PX}px; background: {ringGradient};"
				aria-hidden="true"
			></div>

			<svg
				width={DIAL_SIZE_PX}
				height={DIAL_SIZE_PX}
				class="dial-viewport"
				viewBox="0 0 {DIAL_SIZE_PX} {DIAL_SIZE_PX}"
			>
				<defs>
					<filter id="label-drop-shadow" x="-30%" y="-30%" width="160%" height="160%">
						<feDropShadow
							dx="0"
							dy="1"
							stdDeviation="2.5"
							flood-color="#000"
							flood-opacity="0.5"
						/>
					</filter>
					{#if totalSegments > 0}
						{@const linePos = getPositionForIndex(
							currentIndex,
							totalSegments,
							HANDLE_RADIUS,
						)}
						<linearGradient
							id="handle-line-gradient"
							gradientUnits="userSpaceOnUse"
							x1={DIAL_CENTER}
							y1={DIAL_CENTER}
							x2={linePos.x}
							y2={linePos.y}
						>
							<!-- Fade in from center, stay bright through mid, fade out as it reaches the knob/labels -->
							<stop offset="0%" stop-color="rgba(255, 255, 255, 0)" />
							<stop offset="20%" stop-color="rgba(255, 255, 255, 1)" />
							<stop offset="65%" stop-color="rgba(255, 255, 255, 0.7)" />
							<stop offset="100%" stop-color="rgba(255, 255, 255, 0)" />
						</linearGradient>
					{/if}
				</defs>

				{#if totalSegments > 0}
					<!-- Labels: hours or months, visible on hover or while dragging -->
					<g
						class="hour-labels"
						class:visible={showLabels}
						filter="url(#label-drop-shadow)"
						aria-hidden="true"
					>
						{#each Array(mode === "day" ? 24 : 12) as _, i (i)}
							{@const pos = getPositionForIndex(i, mode === "day" ? 24 : 12, LABEL_RADIUS)}
							<text
								x={pos.x}
								y={pos.y}
								text-anchor="middle"
								dominant-baseline="middle"
								class="hour-label"
							>
								{mode === "day" ? String(i).padStart(2, "0") : MONTH_LABELS[i]}
							</text>
						{/each}
					</g>

					{@const knobPos = getPositionForIndex(
						currentIndex,
						totalSegments,
						HANDLE_RADIUS,
					)}
					{@const linePos = getPositionForIndex(
						currentIndex,
						totalSegments,
						HANDLE_RADIUS,
					)}
					<!-- Progress trail: single soft arc that blends with the dial (refined, minimal) -->
					<g class="progress-trail">
					<circle
						cx={DIAL_CENTER}
						cy={DIAL_CENTER}
						r={HANDLE_RADIUS}
						fill="none"
						stroke="rgba(255, 255, 255, 0.35)"
						stroke-width="1.5"
						stroke-dasharray={`${progressDash} ${circumference}`}
							stroke-linecap="round"
							transform="rotate(-90 {DIAL_CENTER} {DIAL_CENTER})"
							class="progress-arc"
							class:animate-transition={!isDragging}
						/>
					</g>
					<!-- Figma-style hand: line fades in from center so it's less pronounced over ":" -->
					<line
						x1={DIAL_CENTER}
						y1={DIAL_CENTER}
						x2={linePos.x}
						y2={linePos.y}
						stroke="url(#handle-line-gradient)"
						stroke-width="2.5"
						stroke-linecap="round"
						class="dial-handle-line"
						class:animate-transition={!isDragging}
					/>
					<circle
						cx={knobPos.x}
						cy={knobPos.y}
						r="6"
						fill="#ffffff"
						stroke="rgba(0, 0, 0, 0.12)"
						stroke-width="1"
						class="dial-handle"
					/>
				{/if}
			</svg>

			<div class="center-display">
				<div class="current-time">{centerLabel}</div>
				{#if mode === "day"}
					<div class="state-label">{currentState.label}</div>
				{/if}
			</div>
		</div>

		{#if fixedMode == null}
			<div class="mode-toggle">
				<button
					class="mode-pill"
					class:active={mode === "day"}
					on:click={() => (mode = "day")}
					type="button"
				>
					Day
				</button>
				<button
					class="mode-pill"
					class:active={mode === "month"}
					on:click={() => (mode = "month")}
					type="button"
				>
					Month
				</button>
			</div>
		{/if}
		<ColorModeToggle />
	</div>
</div>

<svelte:window on:pointerup={() => (isDragging = false)} />

<style>
	.radial-wrapper {
		width: 100%;
	}

	/* Task 5: panel stays distinct; dial face is the gradient ring */
	.radial-panel {
		background: var(--color-bg-panel-soft);
		padding: var(--spacing-lg);
		border-radius: var(--radius-panel);
		box-shadow: var(--shadow-panel);
		-webkit-backdrop-filter: blur(12px);
		backdrop-filter: blur(12px);
	}

	.mode-toggle {
		display: flex;
		gap: 4px;
		margin-top: var(--spacing-md);
		justify-content: flex-end;
	}
	.mode-pill {
		font-family: var(--font-family);
		font-size: 12px;
		font-weight: 500;
		padding: 4px 10px;
		border-radius: 9999px;
		border: 1px solid rgba(255, 255, 255, 0.2);
		background: rgba(255, 255, 255, 0.08);
		color: var(--color-text-secondary, #94a3b8);
		cursor: pointer;
		transition: background 0.2s, color 0.2s, border-color 0.2s;
	}
	.mode-pill:hover {
		background: rgba(255, 255, 255, 0.12);
		color: var(--color-text-primary, #f8fafc);
	}
	.mode-pill.active {
		background: rgba(255, 255, 255, 0.2);
		color: #fff;
		border-color: rgba(255, 255, 255, 0.35);
	}

	.radial-dial {
		position: relative;
		width: var(--dial-size-px);
		height: var(--dial-size-px);
		border-radius: 50%;
		cursor: pointer;
		outline: none;
		touch-action: none;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.radial-dial:focus-visible {
		box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.35);
	}

	.day-ring {
		position: absolute;
		left: 50%;
		top: 50%;
		transform: translate(-50%, -50%);
		border-radius: 50%;
		pointer-events: none;
	}

	.dial-viewport {
		position: absolute;
		top: 0;
		right: 0;
		bottom: 0;
		left: 0;
		pointer-events: none;
	}

	/* Hour labels: show on hover/drag, smooth fade + scale (200–300ms per ui-ux-pro-max) */
	.hour-labels {
		opacity: 0;
		transform: scale(0.92);
		transform-origin: var(--dial-center-px) var(--dial-center-px);
		pointer-events: none;
		transition:
			opacity 0.28s cubic-bezier(0.4, 0, 0.2, 1),
			transform 0.28s cubic-bezier(0.4, 0, 0.2, 1);
	}
	.hour-labels.visible {
		opacity: 1;
		transform: scale(1);
	}
	.hour-label {
		font-family: var(--font-family);
		font-size: 9px;
		font-weight: 300;
		fill: #ffffff;
		letter-spacing: 0.02em;
		user-select: none;
	}

	.center-display {
		position: relative;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
		color: var(--color-text-primary);
		font-family: var(--font-family);
		pointer-events: none;
	}

	/* Fixed-width time so digits (e.g. 9 vs 10) don’t shift layout when handle crosses text */
	.current-time {
		font-size: 32px;
		font-weight: 600;
		font-variant-numeric: tabular-nums;
		letter-spacing: 0.02em;
		min-width: 5ch;
		text-align: center;
		color: var(--color-text-primary, #1e293b);
		text-shadow:
			0 0 12px rgba(255, 255, 255, 0.9),
			0 1px 3px rgba(0, 0, 0, 0.2);
	}

	/* White state label with strong drop shadow so it reads on any gradient and stays elegant */
	.state-label {
		font-size: 15px;
		font-weight: 600;
		color: #ffffff;
		text-shadow:
			0 2px 4px rgba(0, 0, 0, 0.45),
			0 4px 12px rgba(0, 0, 0, 0.35),
			0 0 24px rgba(0, 0, 0, 0.25),
			0 0 1px rgba(0, 0, 0, 0.5);
	}

	/* Figma-style hand: line + circle at end */
	.dial-handle-line {
		transition: none;
	}
	.dial-handle-line.animate-transition {
		transition: stroke 0.2s ease;
	}
	.dial-handle {
		transition: transform 0.15s ease;
		filter: drop-shadow(0 1px 3px rgba(0, 0, 0, 0.2));
	}
	.radial-dial:focus-visible .dial-handle {
		filter: drop-shadow(0 0 6px rgba(255, 255, 255, 0.8))
			drop-shadow(0 1px 3px rgba(0, 0, 0, 0.2));
	}

	/* Progress trail: blends with gradient (overlay), 200ms per ui-ux-pro-max */
	.progress-trail {
		mix-blend-mode: overlay;
		isolation: isolate;
	}
	.animate-transition {
		transition: stroke-dasharray 0.2s ease, stroke 0.2s ease;
	}
</style>
