<script lang="ts">
	/**
	 * ComparisonCurtain Component
	 *
	 * ABOUTME: A draggable vertical divider overlay for comparing base and comparison scenes.
	 * Features:
	 * - Draggable handle to adjust curtain position
	 * - Anchor snapping at 0%, 50%, 100%
	 * - Keyboard support (left/right arrows to nudge)
	 * - Double-click to snap to center
	 * - Labels showing base vs comparison scenario names
	 */
	import { onMount, onDestroy } from "svelte";
	import {
		comparisonStore,
		setCurtainPosition,
		snapCurtainToAnchor,
		nudgeCurtain,
		stopComparison,
	} from "$lib/stores/comparisonStore";

	// Props
	export let containerElement: HTMLElement | null = null;
	export let comparisonScenarioName: string = "Comparison";

	// Local state
	let isDragging = false;
	let handleElement: HTMLElement | null = null;

	// Snap threshold (in percentage)
	const SNAP_THRESHOLD = 0.03;

	// Anchor positions
	const ANCHORS = [0, 0.5, 1];

	/**
	 * Get the position as a percentage of container width
	 */
	function getPositionFromEvent(event: MouseEvent | TouchEvent): number {
		const doc = globalThis.document;
		const container = containerElement || (doc ? doc.body : null);
		if (!container) {
			// Reasonable fallback during non-DOM environments
			return 0.5;
		}
		const rect = container.getBoundingClientRect();

		let clientX: number;
		if ("touches" in event) {
			clientX = event.touches[0].clientX;
		} else {
			clientX = event.clientX;
		}

		const relativeX = clientX - rect.left;
		const position = relativeX / rect.width;

		return Math.max(0, Math.min(1, position));
	}

	/**
	 * Snap to anchor if close enough
	 */
	function snapToAnchorIfClose(position: number): number {
		for (const anchor of ANCHORS) {
			if (Math.abs(position - anchor) < SNAP_THRESHOLD) {
				return anchor;
			}
		}
		return position;
	}

	/**
	 * Handle mouse/touch down on handle
	 */
	function handlePointerDown(event: MouseEvent | TouchEvent): void {
		event.preventDefault();
		isDragging = true;

		// Add global listeners
		if ("touches" in event) {
			globalThis.document?.addEventListener("touchmove", handlePointerMove, {
				passive: false,
			});
			globalThis.document?.addEventListener("touchend", handlePointerUp);
			globalThis.document?.addEventListener("touchcancel", handlePointerUp);
		} else {
			globalThis.document?.addEventListener("mousemove", handlePointerMove);
			globalThis.document?.addEventListener("mouseup", handlePointerUp);
		}
	}

	/**
	 * Handle mouse/touch move
	 */
	function handlePointerMove(event: MouseEvent | TouchEvent): void {
		if (!isDragging) return;
		event.preventDefault();

		const position = getPositionFromEvent(event);
		const snappedPosition = snapToAnchorIfClose(position);
		setCurtainPosition(snappedPosition);
	}

	/**
	 * Handle mouse/touch up
	 */
	function handlePointerUp(): void {
		isDragging = false;

		// Remove global listeners
		globalThis.document?.removeEventListener("mousemove", handlePointerMove);
		globalThis.document?.removeEventListener("mouseup", handlePointerUp);
		globalThis.document?.removeEventListener("touchmove", handlePointerMove);
		globalThis.document?.removeEventListener("touchend", handlePointerUp);
		globalThis.document?.removeEventListener("touchcancel", handlePointerUp);
	}

	/**
	 * Handle double-click on handle to snap to center
	 */
	function handleDoubleClick(): void {
		snapCurtainToAnchor("center");
	}

	/**
	 * Handle keyboard navigation
	 */
	function handleKeyDown(event: KeyboardEvent): void {
		const activeElement = globalThis.document?.activeElement;
		if (!activeElement || !handleElement?.contains(activeElement)) return;

		switch (event.key) {
			case "ArrowLeft":
				event.preventDefault();
				nudgeCurtain("left");
				break;
			case "ArrowRight":
				event.preventDefault();
				nudgeCurtain("right");
				break;
			case "Home":
				event.preventDefault();
				snapCurtainToAnchor("left");
				break;
			case "End":
				event.preventDefault();
				snapCurtainToAnchor("right");
				break;
			case "Escape":
				event.preventDefault();
				stopComparison();
				break;
		}
	}

	/**
	 * Handle anchor click
	 */
	function handleAnchorClick(anchor: "left" | "right"): void {
		snapCurtainToAnchor(anchor);
	}

	// Add keyboard listener
	onMount(() => {
		globalThis.document?.addEventListener("keydown", handleKeyDown);
	});

	onDestroy(() => {
		globalThis.document?.removeEventListener("keydown", handleKeyDown);
		handlePointerUp(); // Clean up any active drag
	});

	// Reactive curtain position
	$: curtainPercent = $comparisonStore.curtainPosition * 100;
</script>

<div
	class="comparison-curtain"
	style="--curtain-position: {$comparisonStore.curtainPosition}"
>
	<!-- Left anchor (Base label) -->
	<button
		type="button"
		class="anchor anchor-left"
		class:active={$comparisonStore.curtainPosition === 0}
		on:click={() => handleAnchorClick("left")}
		aria-label="Show base only"
	>
		<span class="anchor-label">Base</span>
	</button>

	<!-- Divider line -->
	<div class="curtain-line" aria-hidden="true"></div>

	<!-- Draggable handle -->
	<div
		bind:this={handleElement}
		class="curtain-handle"
		class:dragging={isDragging}
		role="slider"
		tabindex="0"
		aria-label="Comparison curtain position"
		aria-valuemin="0"
		aria-valuemax="100"
		aria-valuenow={Math.round(curtainPercent)}
		aria-valuetext="{Math.round(curtainPercent)}% comparison visible"
		on:mousedown={handlePointerDown}
		on:touchstart={handlePointerDown}
		on:dblclick={handleDoubleClick}
	>
		<div class="handle-grip">
			<div class="grip-line"></div>
			<div class="grip-line"></div>
			<div class="grip-line"></div>
		</div>
	</div>

	<!-- Right anchor (Comparison label) -->
	<button
		type="button"
		class="anchor anchor-right"
		class:active={$comparisonStore.curtainPosition === 1}
		on:click={() => handleAnchorClick("right")}
		aria-label="Show comparison only"
	>
		<span class="anchor-label">{comparisonScenarioName}</span>
	</button>

	<!-- Exit comparison button -->
	<button
		type="button"
		class="exit-button"
		on:click={stopComparison}
		aria-label="Exit comparison mode"
	>
		<span class="exit-icon" aria-hidden="true">x</span>
		<span class="exit-text">Exit</span>
	</button>
</div>

<style>
	.comparison-curtain {
		position: absolute;
		top: 0;
		bottom: 0;
		left: calc(var(--curtain-position) * 100%);
		z-index: var(--z-panel);
		pointer-events: none;
		pointer-events: none;
		transform: translateX(-50%);
		font-family: var(--font-family);
	}

	.curtain-line {
		position: absolute;
		top: 0;
		bottom: 0;
		left: 50%;
		width: 2px;
		background: var(--color-border-strong);
		transform: translateX(-50%);
		pointer-events: none;
		opacity: 0.8;
	}

	.curtain-handle {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		width: 24px;
		height: 48px;
		background: var(--color-bg-panel);
		border: 1px solid var(--color-border-subtle);
		border-radius: 12px;
		cursor: ew-resize;
		pointer-events: auto;
		display: flex;
		align-items: center;
		justify-content: center;
		box-shadow: var(--shadow-panel);
		transition:
			transform 0.1s ease,
			box-shadow 0.15s ease;
	}

	.curtain-handle:hover {
		box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
	}

	.curtain-handle:focus {
		outline: none;
		box-shadow: 0 0 0 2px var(--color-accent);
	}

	.curtain-handle.dragging {
		transform: translate(-50%, -50%) scale(1.05);
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
	}

	.handle-grip {
		display: flex;
		flex-direction: column;
		gap: 3px;
		align-items: center;
	}

	.grip-line {
		width: 10px;
		height: 2px;
		background: var(--color-text-muted);
		border-radius: 1px;
	}

	.anchor {
		position: absolute;
		top: 12px;
		background: var(--color-bg-panel);
		border: 1px solid var(--color-border-subtle);
		border-radius: var(--radius-control);
		padding: 4px 8px;
		cursor: pointer;
		pointer-events: auto;
		transition:
			background 0.15s ease,
			transform 0.1s ease;
		box-shadow: var(--shadow-panel);
		font-family: var(--font-family);
	}

	.anchor:hover {
		background: var(--color-accent-soft);
	}

	.anchor:active {
		transform: scale(0.98);
	}

	.anchor.active {
		background: var(--color-accent-soft);
		border-color: var(--color-accent);
	}

	.anchor-left {
		right: calc(100% + 8px);
		transform: translateX(0);
	}

	.anchor-right {
		left: calc(100% + 8px);
		transform: translateX(0);
	}

	.anchor-label {
		font-size: var(--font-xxs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--color-text-secondary);
		white-space: nowrap;
	}

	.exit-button {
		position: absolute;
		bottom: 12px;
		left: 50%;
		transform: translateX(-50%);
		display: flex;
		align-items: center;
		gap: 4px;
		background: var(--color-bg-panel);
		border: 1px solid var(--color-border-subtle);
		border-radius: var(--radius-control);
		padding: 6px 10px;
		cursor: pointer;
		pointer-events: auto;
		transition:
			background 0.15s ease,
			border-color 0.15s ease;
		box-shadow: var(--shadow-panel);
		font-family: var(--font-family);
	}

	.exit-button:hover {
		background: rgba(251, 113, 133, 0.15);
		border-color: var(--color-danger);
	}

	.exit-icon {
		font-size: var(--font-xs);
		font-weight: 700;
		color: var(--color-danger);
		line-height: 1;
	}

	.exit-text {
		font-size: var(--font-xxs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		letter-spacing: 0.06em;
		color: var(--color-text-secondary);
		font-family: var(--font-family);
	}
</style>
