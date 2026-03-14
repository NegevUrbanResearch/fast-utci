/**
 * Total Tregenza dome weight used in MRT shader (mrt_utci.wgsl).
 * WebGPU exposure shader writes raw weight sum; divide by this to get 0–1 sky view factor for parity.
 */
export const TOTAL_TREGENZA_WEIGHT = 145.2488;

export function normalizeSkyExposureToViewFactor(rawSky: number[] | Float32Array): number[] {
	const out: number[] = [];
	for (let i = 0; i < rawSky.length; i++) {
		out.push(Math.max(0, Math.min(1, rawSky[i] / TOTAL_TREGENZA_WEIGHT)));
	}
	return out;
}
