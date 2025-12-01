<script lang="ts">
	import { analysisStore } from '$lib/stores/analysisStore';
	import { viewerStore, setCurrentHour } from '$lib/stores/viewerStore';
	import type { Analysis } from '$lib/types/analysis';

	const BASE_RADIUS = 100;
	const TICK_RADIUS_INNER_MAJOR = 78;
	const TICK_RADIUS_INNER_MINOR = 88;
	const TICK_RADIUS_OUTER = 96;

	let dialEl: HTMLDivElement | null = null;
	let isDragging = false;
	let analysis: Analysis | null = null;
	let hours: unknown[] = [];

	type HourEntry = {
		value: number;
		label: string;
	};

let normalizedHours: HourEntry[] = [];
let totalHours = 0;
let hourValues: number[] = [];
let maxIndex = 0;
let currentHourIndex = 0;
let currentHourLabel = '--:--';
let phaseMarkers: PhaseMarker[] = [];

	$: analysis = $analysisStore;
	$: hours = analysis?.metadata.hours ?? [];
	$: normalizedHours = normalizeHourEntries(hours);
	$: totalHours = normalizedHours.length;
	$: hourValues = normalizedHours.map((entry) => entry.value);

	$: maxIndex = Math.max(totalHours - 1, 0);
	$: currentHourIndex = Math.min($viewerStore.currentHour, maxIndex);
	$: currentHourLabel =
		totalHours > 0
			? normalizedHours[currentHourIndex]?.label ??
				`${currentHourIndex.toString().padStart(2, '0')}:00`
			: '--:--';

	$: currentState = getDayStateForIndex(currentHourIndex, phaseMarkers);
	$: phaseMarkers = derivePhaseMarkers();
	$: circumference = 2 * Math.PI * BASE_RADIUS;
	$: progressDash = totalHours > 0 ? (currentHourIndex / totalHours) * circumference : 0;

type DayStateKey = 'night' | 'morning' | 'afternoon' | 'evening';

type PhaseMarker = {
	key: DayStateKey;
	hourValue: number;
	label: string;
	color: string;
	icon: 'moon' | 'sunrise' | 'sun' | 'sunset';
};

const DAY_STATE_META: Record<
	DayStateKey,
	{
		label: string;
		color: string;
		icon: 'moon' | 'sunrise' | 'sun' | 'sunset';
	}
> = {
	night: { label: 'Night', color: '#6366f1', icon: 'moon' },
	morning: { label: 'Morning', color: '#f59e0b', icon: 'sunrise' },
	afternoon: { label: 'Afternoon', color: '#facc15', icon: 'sun' },
	evening: { label: 'Evening', color: '#f54900', icon: 'sunset' }
};

const ICON_MAP = {
	sun: '#radial-sun-icon',
	sunrise: '#radial-sunrise-icon',
	sunset: '#radial-sunset-icon',
	moon: '#radial-moon-icon'
} as const;

function buildFallbackPhaseMarkers(): PhaseMarker[] {
	const fallbackTargets: Array<{ key: DayStateKey; hour: number }> = [
		{ key: 'night', hour: 0 },
		{ key: 'morning', hour: 6 },
		{ key: 'afternoon', hour: 12 },
		{ key: 'evening', hour: 18 }
	];

	return fallbackTargets
		.map((target) => createPhaseMarker(target.key, target.hour))
		.filter((marker): marker is PhaseMarker => Boolean(marker));
}

const FALLBACK_PHASE_MARKERS = buildFallbackPhaseMarkers();

function derivePhaseMarkers(): PhaseMarker[] {
	if (totalHours === 0) return [];

	const fallback = FALLBACK_PHASE_MARKERS;
	const sunPositions = analysis?.metadata.sun_positions;

	if (!sunPositions || sunPositions.length !== totalHours) {
		return fallback;
	}

	const hourValueForIndex = (index: number) => normalizeHourNumber(hourValues[index] ?? index);

	const markers: PhaseMarker[] = [];
	const nightMarker = createPhaseMarker('night', hourValueForIndex(0));
	if (nightMarker) markers.push(nightMarker);

	const sunriseIndex = sunPositions.findIndex((position) => position.is_up);
	if (sunriseIndex !== -1) {
		const sunriseMarker = createPhaseMarker('morning', hourValueForIndex(sunriseIndex));
		if (sunriseMarker) markers.push(sunriseMarker);
	}

	let middayIndex = 0;
	let maxAltitude = -Infinity;
	sunPositions.forEach((position, index) => {
		if (position.altitude > maxAltitude) {
			maxAltitude = position.altitude;
			middayIndex = index;
		}
	});
	const middayMarker = createPhaseMarker('afternoon', hourValueForIndex(middayIndex));
	if (middayMarker) markers.push(middayMarker);

	let sunsetIndex = -1;
	if (sunriseIndex !== -1) {
		for (let i = sunriseIndex + 1; i < sunPositions.length; i += 1) {
			if (!sunPositions[i].is_up) {
				sunsetIndex = i;
				break;
			}
		}
		if (sunsetIndex === -1) {
			sunsetIndex = sunPositions.length - 1;
		}
	} else {
		// If the sun never rises, treat the highest altitude as sunset start
		sunsetIndex = middayIndex;
	}

	if (sunsetIndex !== -1) {
		const sunsetMarker = createPhaseMarker('evening', hourValueForIndex(sunsetIndex));
		if (sunsetMarker) markers.push(sunsetMarker);
	}

	return mergeAndSortPhaseMarkers(markers, fallback);
}

function getDayStateForIndex(index: number, markers: PhaseMarker[]) {
	if (totalHours === 0 || markers.length === 0) {
		return { key: 'night' as DayStateKey, ...DAY_STATE_META.night };
	}

	const hourValue = normalizeHourNumber(hourValues[index] ?? index);

	// Default to the last marker to handle wrap-around before the first transition
	let activeMarker = markers[markers.length - 1];

	for (const marker of markers) {
		if (hourValue >= marker.hourValue) {
			activeMarker = marker;
		} else {
			break;
		}
	}

	return {
		key: activeMarker.key,
		...DAY_STATE_META[activeMarker.key]
	};
	}

	function normalizeHourEntries(entries: unknown[]): HourEntry[] {
		return entries.map((entry, index) => {
			const fallbackValue = index % 24;
			const fallbackLabel = `${String(fallbackValue).padStart(2, '0')}:00`;

			if (typeof entry === 'number' && Number.isFinite(entry)) {
				const numeric = ((entry % 24) + 24) % 24;
				return {
					value: numeric,
					label: `${String(entry).padStart(2, '0')}:00`
				};
			}

			if (typeof entry === 'string') {
				const trimmed = entry.trim();
				const match = trimmed.match(/(\d{1,2})/);
				let numeric = Number(match?.[1]);
				if (!Number.isFinite(numeric)) {
					numeric = fallbackValue;
				}
				numeric = ((numeric % 24) + 24) % 24;
				const label = trimmed.includes(':')
					? trimmed
					: `${String(numeric).padStart(2, '0')}:00`;

				return {
					value: numeric,
					label
				};
			}

			if (typeof entry === 'object' && entry !== null) {
				const candidateHour =
					(typeof (entry as Record<string, unknown>).hour === 'number'
						? (entry as Record<string, number>).hour
						: typeof (entry as Record<string, unknown>).hour === 'string'
							? Number((entry as Record<string, string>).hour)
							: typeof (entry as Record<string, unknown>).value === 'number'
								? (entry as Record<string, number>).value
								: typeof (entry as Record<string, unknown>).index === 'number'
									? (entry as Record<string, number>).index
									: null);

				let numeric = Number(candidateHour);
				if (!Number.isFinite(numeric)) {
					numeric = fallbackValue;
				}
				numeric = ((numeric % 24) + 24) % 24;

				const candidateLabel =
					typeof (entry as Record<string, unknown>).label === 'string'
						? ((entry as Record<string, string>).label ?? '')
						: typeof (entry as Record<string, unknown>).time === 'string'
							? ((entry as Record<string, string>).time ?? '')
							: '';

				const label = candidateLabel && candidateLabel.trim().length > 0
					? candidateLabel.trim()
					: `${String(numeric).padStart(2, '0')}:00`;

				return {
					value: numeric,
					label
				};
			}

			return {
				value: fallbackValue,
				label: fallbackLabel
			};
		});
	}

	function getAngleForIndex(index: number) {
		if (totalHours === 0) return -90;
		return (index / totalHours) * 360 - 90;
	}

	function getPositionForIndex(index: number, radius: number) {
		const angle = (getAngleForIndex(index) * Math.PI) / 180;
		return {
			x: 120 + Math.cos(angle) * radius,
			y: 120 + Math.sin(angle) * radius
		};
	}

function getPositionForHourValue(hour: number, radius: number) {
	const angle = (getAngleForHour(hour) * Math.PI) / 180;
	return {
		x: 120 + Math.cos(angle) * radius,
		y: 120 + Math.sin(angle) * radius
	};
}

function getAngleForHour(hour: number) {
	return normalizeHourNumber(hour) * (360 / 24) - 90;
}

function normalizeHourNumber(hour: number) {
	return ((hour % 24) + 24) % 24;
}

function createPhaseMarker(key: DayStateKey, hourValue?: number | null): PhaseMarker | null {
	if (hourValue === undefined || hourValue === null || Number.isNaN(hourValue)) {
		return null;
	}
	const normalized = normalizeHourNumber(hourValue);
	return {
		key,
		hourValue: normalized,
		label: DAY_STATE_META[key].label,
		color: DAY_STATE_META[key].color,
		icon: DAY_STATE_META[key].icon
	};
}

function mergeAndSortPhaseMarkers(customMarkers: PhaseMarker[], fallbackMarkers: PhaseMarker[]): PhaseMarker[] {
	const markerMap = new Map<DayStateKey, PhaseMarker>();

	// Start with fallback to guarantee each key exists
	fallbackMarkers.forEach((marker) => {
		markerMap.set(marker.key, marker);
	});

	customMarkers.forEach((marker) => {
		if (marker) {
			markerMap.set(marker.key, marker);
		}
	});

	return [...markerMap.values()].sort((a, b) => a.hourValue - b.hourValue);
}
	function updateHourFromPointer(event: PointerEvent) {
		if (!dialEl || totalHours === 0) return;

		const rect = dialEl.getBoundingClientRect();
		const centerX = rect.left + rect.width / 2;
		const centerY = rect.top + rect.height / 2;
		const x = event.clientX - centerX;
		const y = event.clientY - centerY;

		let angle = (Math.atan2(y, x) * 180) / Math.PI;
		angle = (angle + 450) % 360; // align 0 at top

		let targetIndex = Math.round((angle / 360) * totalHours);
		if (targetIndex >= totalHours) targetIndex = 0;

		if (targetIndex !== $viewerStore.currentHour) {
			setCurrentHour(targetIndex);
		}
	}

	function handlePointerDown(event: PointerEvent) {
		if (!dialEl) return;
		isDragging = true;
		dialEl.setPointerCapture(event.pointerId);
		event.preventDefault();
		updateHourFromPointer(event);
	}

	function handlePointerMove(event: PointerEvent) {
		if (!isDragging) return;
		updateHourFromPointer(event);
	}

	function handlePointerUp(event: PointerEvent) {
		if (!dialEl) return;
		isDragging = false;
		if (dialEl.hasPointerCapture(event.pointerId)) {
			dialEl.releasePointerCapture(event.pointerId);
		}
	}

	function handleKeyDown(event: KeyboardEvent) {
		if (totalHours === 0) return;
		let nextIndex: number | null = null;

		if (event.key === 'ArrowRight' || event.key === 'ArrowUp') {
			nextIndex = (currentHourIndex + 1) % totalHours;
		} else if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') {
			nextIndex = (currentHourIndex - 1 + totalHours) % totalHours;
		} else if (event.key === 'Home') {
			nextIndex = 0;
		} else if (event.key === 'End') {
			nextIndex = totalHours - 1;
		}

		if (nextIndex !== null) {
			event.preventDefault();
			setCurrentHour(nextIndex);
		}
	}

	$: if (totalHours > 0 && $viewerStore.currentHour !== currentHourIndex) {
		setCurrentHour(currentHourIndex);
	}
</script>

<div class="radial-wrapper">
	<div class="radial-panel">
		<div
			class="radial-dial"
			role="slider"
			tabindex={totalHours > 0 ? 0 : -1}
			aria-valuemin="0"
			aria-valuemax={Math.max(totalHours - 1, 0)}
			aria-valuenow={currentHourIndex}
			aria-valuetext={`Time ${currentHourLabel}`}
			aria-label="Select analysis hour"
			bind:this={dialEl}
			on:pointerdown={handlePointerDown}
			on:pointermove={handlePointerMove}
			on:pointerup={handlePointerUp}
			on:keydown={handleKeyDown}
		>
			<svg width="240" height="240" class="dial-viewport" viewBox="0 0 240 240">
				<circle cx="120" cy="120" r="110" fill="none" stroke="rgba(148, 163, 184, 0.08)" stroke-width="2" />

				{#if totalHours > 0}
					{#each Array(totalHours) as _, index}
						{@const angle = (index / totalHours) * 360 - 90}
						{@const rad = (angle * Math.PI) / 180}
						{@const isMajor = index % Math.max(Math.round(totalHours / 6) || 1, 1) === 0}
						<line
							x1={120 + Math.cos(rad) * (isMajor ? TICK_RADIUS_INNER_MAJOR : TICK_RADIUS_INNER_MINOR)}
							y1={120 + Math.sin(rad) * (isMajor ? TICK_RADIUS_INNER_MAJOR : TICK_RADIUS_INNER_MINOR)}
							x2={120 + Math.cos(rad) * TICK_RADIUS_OUTER}
							y2={120 + Math.sin(rad) * TICK_RADIUS_OUTER}
							stroke={isMajor ? 'rgba(148, 163, 184, 0.6)' : 'rgba(148, 163, 184, 0.25)'}
							stroke-width={isMajor ? 2 : 1}
						/>
					{/each}

					{#each phaseMarkers as marker (marker.key)}
						{@const pos = getPositionForHourValue(marker.hourValue, 62)}
						<g
							class="phase-marker"
							transform={`translate(${pos.x}, ${pos.y})`}
							style={`color: ${marker.color};`}
							aria-hidden="true"
						>
							<circle r="9" class="phase-ring" />
							<circle r="5" class="phase-ring-inner" />
						</g>
					{/each}

					<circle
						cx="120"
						cy="120"
						r={BASE_RADIUS}
						fill="none"
						stroke={currentState.color}
						stroke-width="8"
						stroke-dasharray={`${progressDash} ${circumference}`}
						stroke-linecap="round"
						transform="rotate(-90 120 120)"
						class:animate-transition={!isDragging}
					/>

					<circle
						cx={getPositionForIndex(currentHourIndex, BASE_RADIUS).x}
						cy={getPositionForIndex(currentHourIndex, BASE_RADIUS).y}
						r="9"
						fill={currentState.color}
						stroke="white"
						stroke-width="2"
						class="dial-handle"
					/>
				{/if}
			</svg>

			<div class="center-display">
				<div class="state-icon" style={`color: ${currentState.color};`}>
					<svg viewBox="0 0 24 24" width="32" height="32">
						<use href={ICON_MAP[currentState.icon]} />
					</svg>
				</div>
				<div class="current-time">{currentHourLabel}</div>
				<div class="state-label">{currentState.label}</div>
			</div>
		</div>
	</div>
</div>

<svelte:window on:pointerup={() => (isDragging = false)} />

<svelte:options immutable={false} />

<style>
	.radial-wrapper {
		width: 100%;
	}

	.radial-panel {
		background: var(--color-bg-panel-soft);
		padding: var(--spacing-lg);
		border-radius: var(--radius-panel);
		box-shadow: var(--shadow-panel);
		backdrop-filter: blur(12px);
	}

	.radial-dial {
		position: relative;
		width: 240px;
		height: 240px;
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

	.dial-viewport {
		position: absolute;
		inset: 0;
		pointer-events: none;
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

	.current-time {
		font-size: 28px;
		font-weight: 600;
		letter-spacing: 0.02em;
	}

	.state-label {
		font-size: 13px;
		color: var(--color-text-secondary, rgba(148, 163, 184, 0.9));
	}

	.state-icon svg {
		width: 32px;
		height: 32px;
		fill: none;
		stroke: currentColor;
		stroke-linecap: round;
		stroke-linejoin: round;
		stroke-width: 1.8;
	}

	.phase-marker {
		transition: transform 0.2s ease, opacity 0.2s ease;
		opacity: 0.7;
	}

	.phase-marker .phase-ring {
		fill: currentColor;
		opacity: 0.18;
	}

	.phase-marker .phase-ring-inner {
		fill: currentColor;
		opacity: 0.28;
	}

	.radial-dial:focus-visible .phase-marker,
	.radial-dial:hover .phase-marker {
		opacity: 1;
	}

	.dial-handle {
		transition: transform 0.15s ease, fill 0.2s ease;
	}

	.animate-transition {
		transition: stroke-dasharray 0.25s ease, stroke 0.25s ease;
	}

	.icon-defs {
		position: absolute;
		width: 0;
		height: 0;
		overflow: hidden;
		pointer-events: none;
	}
</style>

<svg
	class="icon-defs"
	aria-hidden="true"
	focusable="false"
	width="0"
	height="0"
	style="position:absolute;width:0;height:0;overflow:hidden;opacity:0;pointer-events:none;display:none;"
>
	<defs>
		<symbol id="radial-sun-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
			<circle cx="12" cy="12" r="4" />
			<path d="M12 2v2" />
			<path d="M12 20v2" />
			<path d="m4.93 4.93 1.41 1.41" />
			<path d="m17.66 17.66 1.41 1.41" />
			<path d="M2 12h2" />
			<path d="M20 12h2" />
			<path d="m6.34 17.66-1.41 1.41" />
			<path d="m19.07 4.93-1.41 1.41" />
		</symbol>
		<symbol id="radial-sunrise-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
			<path d="M12 2v8" />
			<path d="m4.93 10.93 1.41 1.41" />
			<path d="M2 18h2" />
			<path d="M20 18h2" />
			<path d="m19.07 10.93-1.41 1.41" />
			<path d="M22 22H2" />
			<path d="m8 6 4-4 4 4" />
			<path d="M16 18a4 4 0 0 0-8 0" />
		</symbol>
		<symbol id="radial-sunset-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
			<path d="M12 10V2" />
			<path d="m4.93 10.93 1.41 1.41" />
			<path d="M2 18h2" />
			<path d="M20 18h2" />
			<path d="m19.07 10.93-1.41 1.41" />
			<path d="M22 22H2" />
			<path d="m16 6-4 4-4-4" />
			<path d="M16 18a4 4 0 0 0-8 0" />
		</symbol>
		<symbol id="radial-moon-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
			<path d="M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.344-.215.825-.004.803.401" />
		</symbol>
	</defs>
</svg>

