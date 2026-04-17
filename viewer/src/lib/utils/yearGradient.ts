// viewer/src/lib/utils/yearGradient.ts

export const YEAR_GRADIENT_STOPS: Array<{ stop: number; color: string }> = [
	{ stop: 0, color: '#8B6CC7' },   // Winter solstice - violet/purple
	{ stop: 0.06, color: '#6E8ED8' },
	{ stop: 0.12, color: '#5AA0E8' },
	{ stop: 0.18, color: '#48B8D8' },
	{ stop: 0.25, color: '#40C8A0' },  // March equinox - teal
	{ stop: 0.32, color: '#50D878' },
	{ stop: 0.38, color: '#78E060' },
	{ stop: 0.44, color: '#B8E848' },
	{ stop: 0.5, color: '#E8E040' },   // June solstice - bright yellow
	{ stop: 0.56, color: '#F0C840' },
	{ stop: 0.62, color: '#F0A840' },
	{ stop: 0.68, color: '#E88840' },
	{ stop: 0.75, color: '#E06848' },  // September equinox - coral-orange
	{ stop: 0.82, color: '#D84858' },
	{ stop: 0.88, color: '#C84070' },
	{ stop: 0.94, color: '#A850A0' },
	{ stop: 1, color: '#8B6CC7' }
];

export const YEAR_RING_INNER = 84;
export const YEAR_RING_OUTER = 110;

/**
 * Returns a conic-gradient() CSS value for the year ring.
 * Uses Figma gradient as-is: 0 = Jan (12 o'clock), 0.5 = Jul, 1 = Jan again.
 * No rotation is applied (unlike the day ring, which rotates for midnight-at-top).
 */
export function getYearRingConicGradient(): string {
	const parts = YEAR_GRADIENT_STOPS.map(
		({ stop, color }) => `${color} ${(stop * 100).toFixed(2)}%`
	);
	return `conic-gradient(from 0deg, ${parts.join(', ')})`;
}
