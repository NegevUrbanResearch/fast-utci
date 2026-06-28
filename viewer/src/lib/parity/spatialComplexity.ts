export interface SpatialComplexityMetrics {
	gradientEnergy: number;
	variance: number;
	entropy: number;
}

export interface RectGridShape {
	width: number;
	height: number;
}

function toFiniteValues(values: readonly number[]): number[] {
	return values.filter((v) => Number.isFinite(v));
}

export function computeSpatialComplexity(
	field: readonly number[],
	width: number,
	height: number
): SpatialComplexityMetrics {
	if (width <= 0 || height <= 0) {
		throw new Error('width and height must be positive');
	}
	if (field.length !== width * height) {
		throw new Error(`Field length mismatch: got ${field.length}, expected ${width * height}`);
	}

	let gradientEnergySum = 0;
	let gradientSamples = 0;
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const idx = y * width + x;
			const v = field[idx];
			if (!Number.isFinite(v)) continue;
			if (x + 1 < width) {
				const right = field[idx + 1];
				if (Number.isFinite(right)) {
					const dx = right - v;
					gradientEnergySum += dx * dx;
					gradientSamples++;
				}
			}
			if (y + 1 < height) {
				const down = field[idx + width];
				if (Number.isFinite(down)) {
					const dy = down - v;
					gradientEnergySum += dy * dy;
					gradientSamples++;
				}
			}
		}
	}
	const gradientEnergy = gradientSamples > 0 ? gradientEnergySum / gradientSamples : 0;

	const finite = toFiniteValues(field);
	if (finite.length === 0) {
		return { gradientEnergy, variance: 0, entropy: 0 };
	}
	const mean = finite.reduce((acc, v) => acc + v, 0) / finite.length;
	const variance = finite.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / finite.length;

	let min = finite[0];
	let max = finite[0];
	for (const v of finite) {
		if (v < min) min = v;
		if (v > max) max = v;
	}
	if (max <= min) {
		return { gradientEnergy, variance, entropy: 0 };
	}
	const bins = 64;
	const hist = new Array<number>(bins).fill(0);
	const invRange = 1 / (max - min);
	for (const v of finite) {
		const normalized = (v - min) * invRange;
		const idx = Math.min(bins - 1, Math.max(0, Math.floor(normalized * bins)));
		hist[idx]++;
	}
	let entropy = 0;
	for (const count of hist) {
		if (count <= 0) continue;
		const p = count / finite.length;
		entropy -= p * Math.log2(p);
	}
	return { gradientEnergy, variance, entropy };
}

function quantize(value: number, epsilon: number): string {
	return (Math.round(value / epsilon) * epsilon).toFixed(6);
}

export function inferRectGridShapeFromPositions(
	positions: readonly number[],
	epsilon = 1e-6
): RectGridShape | null {
	if (positions.length === 0 || positions.length % 3 !== 0) return null;
	const numPoints = positions.length / 3;
	const xSet = new Set<string>();
	const zSet = new Set<string>();
	for (let i = 0; i < numPoints; i++) {
		xSet.add(quantize(positions[i * 3] ?? 0, epsilon));
		zSet.add(quantize(positions[i * 3 + 2] ?? 0, epsilon));
	}
	const width = zSet.size;
	const height = xSet.size;
	if (width <= 0 || height <= 0 || width * height !== numPoints) {
		return null;
	}
	return { width, height };
}
