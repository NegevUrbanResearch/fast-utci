/**
 * Day-clock gradient (Figma-style). 0 = noon, 0.5 = midnight.
 * Used to build a single conic-gradient so the ring has no segment seams.
 */

export const DAY_GRADIENT_STOPS: Array<{ stop: number; color: string }> = [
	{ stop: 0, color: "#F8E8C0" },
	{ stop: 0.08, color: "#F0D070" },
	{ stop: 0.18, color: "#E8A050" },
	{ stop: 0.25, color: "#D86848" },
	{ stop: 0.32, color: "#A04880" },
	{ stop: 0.42, color: "#5040A0" },
	{ stop: 0.5, color: "#282868" },
	{ stop: 0.58, color: "#3848A0" },
	{ stop: 0.68, color: "#6858B0" },
	{ stop: 0.75, color: "#D07050" },
	{ stop: 0.82, color: "#E8A050" },
	{ stop: 0.9, color: "#F0C860" },
	{ stop: 1, color: "#F8E8C0" },
];

/** Ring radii (px) for layout. */
export const DAY_RING_INNER = 84;
export const DAY_RING_OUTER = 110;

/**
 * Returns a conic-gradient() CSS value with midnight at top (12 o'clock).
 * Use as background on a full circle; pair with an inner circle for the ring effect.
 */
export function getDayRingConicGradient(): string {
	// Rotate so Figma 0.5 (midnight) is at 0deg (top). Order: 0.5..1, then 0..0.5.
	const reordered: Array<{ pct: number; color: string }> = [];
	for (let i = 0; i < DAY_GRADIENT_STOPS.length; i++) {
		const stop = DAY_GRADIENT_STOPS[i];
		const rotated = (stop.stop + 0.5) % 1;
		reordered.push({ pct: rotated * 100, color: stop.color });
	}
	reordered.sort((a, b) => a.pct - b.pct);
	// Close the loop: explicit 100% with same color as 0% (midnight) so no sharp cut at wrap.
	const midnightColor = reordered[0].color;
	if (reordered[reordered.length - 1].pct < 100) {
		reordered.push({ pct: 100, color: midnightColor });
	}
	const parts = reordered.map(({ pct, color }) => `${color} ${pct}%`);
	return `conic-gradient(from 0deg, ${parts.join(", ")})`;
}
