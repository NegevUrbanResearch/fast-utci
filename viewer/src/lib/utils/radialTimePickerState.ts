/**
 * Radial time picker: hour normalization, phase markers, day state, and geometry.
 * Keeps the main component thin and testable.
 */

import type { Analysis } from "$lib/types/analysis";

export type DayStateKey = "night" | "morning" | "afternoon" | "evening";

export interface HourEntry {
	value: number;
	label: string;
}

export interface PhaseMarker {
	key: DayStateKey;
	hourValue: number;
	label: string;
	color: string;
	icon: "moon" | "sunrise" | "sun" | "sunset";
}

export const DAY_STATE_META: Record<
	DayStateKey,
	{ label: string; color: string; icon: "moon" | "sunrise" | "sun" | "sunset" }
> = {
	night: { label: "Night", color: "#6366f1", icon: "moon" },
	morning: { label: "Morning", color: "#f59e0b", icon: "sunrise" },
	afternoon: { label: "Afternoon", color: "#facc15", icon: "sun" },
	evening: { label: "Evening", color: "#f54900", icon: "sunset" },
};

/** Dial layout: size, center, handle/line/label radii (single source of truth for SVG/layout). */
export const DIAL_SIZE_PX = 240;
export const DIAL_CENTER = 120;
/** Handle knob traces near the outer rim of the clock. */
export const HANDLE_RADIUS = 114;
/** Handle line stops clearly before labels so it doesn't run through the text. */
export const HANDLE_LINE_RADIUS = 96;
/** Hour labels sit slightly inside the rim, below the handle knob. */
export const LABEL_RADIUS = 102;

const FALLBACK_PHASE_HOURS: Array<{ key: DayStateKey; hour: number }> = [
	{ key: "night", hour: 0 },
	{ key: "morning", hour: 6 },
	{ key: "afternoon", hour: 12 },
	{ key: "evening", hour: 18 },
];

export function normalizeHourNumber(hour: number): number {
	return ((hour % 24) + 24) % 24;
}

export function normalizeHourEntries(entries: unknown[]): HourEntry[] {
	return entries.map((entry, index) => {
		const fallbackValue = index % 24;
		const fallbackLabel = `${String(fallbackValue).padStart(2, "0")}:00`;

		if (typeof entry === "number" && Number.isFinite(entry)) {
			const numeric = ((entry % 24) + 24) % 24;
			return {
				value: numeric,
				label: `${String(entry).padStart(2, "0")}:00`,
			};
		}

		if (typeof entry === "string") {
			const trimmed = entry.trim();
			const match = trimmed.match(/(\d{1,2})/);
			let numeric = Number(match?.[1]);
			if (!Number.isFinite(numeric)) numeric = fallbackValue;
			numeric = ((numeric % 24) + 24) % 24;
			const label = trimmed.includes(":")
				? trimmed
				: `${String(numeric).padStart(2, "0")}:00`;
			return { value: numeric, label };
		}

		if (typeof entry === "object" && entry !== null) {
			const rec = entry as Record<string, unknown>;
			const candidateHour =
				typeof rec.hour === "number"
					? rec.hour
					: typeof rec.hour === "string"
						? Number(rec.hour)
						: typeof rec.value === "number"
							? rec.value
							: typeof rec.index === "number"
								? rec.index
								: null;
			let numeric = Number(candidateHour);
			if (!Number.isFinite(numeric)) numeric = fallbackValue;
			numeric = ((numeric % 24) + 24) % 24;
			const candidateLabel =
				typeof rec.label === "string"
					? (rec.label as string)
					: typeof rec.time === "string"
						? (rec.time as string)
						: "";
			const label =
				candidateLabel?.trim().length > 0
					? candidateLabel.trim()
					: `${String(numeric).padStart(2, "0")}:00`;
			return { value: numeric, label };
		}

		return { value: fallbackValue, label: fallbackLabel };
	});
}

function createPhaseMarker(
	key: DayStateKey,
	hourValue?: number | null,
): PhaseMarker | null {
	if (
		hourValue === undefined ||
		hourValue === null ||
		Number.isNaN(hourValue)
	) {
		return null;
	}
	const normalized = normalizeHourNumber(hourValue);
	return {
		key,
		hourValue: normalized,
		label: DAY_STATE_META[key].label,
		color: DAY_STATE_META[key].color,
		icon: DAY_STATE_META[key].icon,
	};
}

function mergeAndSortPhaseMarkers(
	custom: PhaseMarker[],
	fallback: PhaseMarker[],
): PhaseMarker[] {
	const map = new Map<DayStateKey, PhaseMarker>();
	fallback.forEach((m) => map.set(m.key, m));
	custom.forEach((m) => m && map.set(m.key, m));
	return [...map.values()].sort((a, b) => a.hourValue - b.hourValue);
}

export function derivePhaseMarkers(
	analysis: Analysis | null,
	totalHours: number,
	hourValues: number[],
): PhaseMarker[] {
	if (totalHours === 0) return [];
	const fallback = FALLBACK_PHASE_HOURS.map((t) =>
		createPhaseMarker(t.key, t.hour),
	).filter((m): m is PhaseMarker => Boolean(m));
	const sunPositions = analysis?.metadata.sun_positions;
	if (!sunPositions || sunPositions.length !== totalHours) return fallback;

	const hourValueForIndex = (index: number) =>
		normalizeHourNumber(hourValues[index] ?? index);
	const markers: PhaseMarker[] = [];
	const nightMarker = createPhaseMarker("night", hourValueForIndex(0));
	if (nightMarker) markers.push(nightMarker);

	const sunriseIndex = sunPositions.findIndex((p) => p.is_up);
	if (sunriseIndex !== -1) {
		const m = createPhaseMarker("morning", hourValueForIndex(sunriseIndex));
		if (m) markers.push(m);
	}

	let middayIndex = 0;
	let maxAltitude = -Infinity;
	sunPositions.forEach((p, i) => {
		if (p.altitude > maxAltitude) {
			maxAltitude = p.altitude;
			middayIndex = i;
		}
	});
	const middayMarker = createPhaseMarker(
		"afternoon",
		hourValueForIndex(middayIndex),
	);
	if (middayMarker) markers.push(middayMarker);

	let sunsetIndex = -1;
	if (sunriseIndex !== -1) {
		for (let i = sunriseIndex + 1; i < sunPositions.length; i++) {
			if (!sunPositions[i].is_up) {
				sunsetIndex = i;
				break;
			}
		}
		if (sunsetIndex === -1) sunsetIndex = sunPositions.length - 1;
	} else {
		sunsetIndex = middayIndex;
	}
	if (sunsetIndex !== -1) {
		const m = createPhaseMarker("evening", hourValueForIndex(sunsetIndex));
		if (m) markers.push(m);
	}

	return mergeAndSortPhaseMarkers(markers, fallback);
}

/** Hour value from which we treat as night instead of evening (22 = 10pm). */
const NIGHT_START_HOUR = 22;

export function getDayStateForIndex(
	index: number,
	phaseMarkers: PhaseMarker[],
	hourValues: number[],
	totalHours: number,
): { key: DayStateKey; label: string; color: string; icon: PhaseMarker["icon"] } {
	if (totalHours === 0 || phaseMarkers.length === 0) {
		return { key: "night", ...DAY_STATE_META.night };
	}
	const hourValue = normalizeHourNumber(hourValues[index] ?? index);
	let active = phaseMarkers[phaseMarkers.length - 1];
	for (const marker of phaseMarkers) {
		if (hourValue >= marker.hourValue) active = marker;
		else break;
	}
	// Hours 22–23 (and 0–5) are night, not evening
	if (active.key === "evening" && hourValue >= NIGHT_START_HOUR) {
		return { key: "night", ...DAY_STATE_META.night };
	}
	return { key: active.key, ...DAY_STATE_META[active.key] };
}

export function getAngleForIndex(index: number, totalHours: number): number {
	if (totalHours === 0) return -90;
	return (index / totalHours) * 360 - 90;
}

export function getPositionForIndex(
	index: number,
	totalHours: number,
	radius: number,
): { x: number; y: number } {
	const angleDeg = getAngleForIndex(index, totalHours);
	const angleRad = (angleDeg * Math.PI) / 180;
	return {
		x: DIAL_CENTER + Math.cos(angleRad) * radius,
		y: DIAL_CENTER + Math.sin(angleRad) * radius,
	};
}

export function getAngleForHour(hour: number): number {
	return normalizeHourNumber(hour) * (360 / 24) - 90;
}

export function getPositionForHourValue(
	hour: number,
	radius: number,
): { x: number; y: number } {
	const angleRad = (getAngleForHour(hour) * Math.PI) / 180;
	return {
		x: DIAL_CENTER + Math.cos(angleRad) * radius,
		y: DIAL_CENTER + Math.sin(angleRad) * radius,
	};
}

/**
 * Map a pointer position (client coordinates) to an hour index.
 * Uses same angle convention as the dial: 0 at top, clockwise.
 */
export function getIndexFromPointerEvent(
	rect: DOMRect,
	clientX: number,
	clientY: number,
	totalHours: number,
): number {
	if (totalHours === 0) return 0;
	const centerX = rect.left + rect.width / 2;
	const centerY = rect.top + rect.height / 2;
	const x = clientX - centerX;
	const y = clientY - centerY;
	let angle = (Math.atan2(y, x) * 180) / Math.PI;
	angle = (angle + 450) % 360; // align 0 at top
	let targetIndex = Math.round((angle / 360) * totalHours);
	if (targetIndex >= totalHours) targetIndex = 0;
	return targetIndex;
}
