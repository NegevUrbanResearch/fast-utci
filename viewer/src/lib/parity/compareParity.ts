export interface ParityOptions {
	toleranceC?: number;
}

export interface ParityResult {
	pass: boolean;
	rmse: number;
	maxError: number;
	withinTolerancePct: number;
	numPoints: number;
}

/**
 * Compare reference UTCI (e.g. from .bin) with WebGPU result for one hour.
 * Pure function; safe to run in Node or browser.
 */
export function compareParity(params: {
	utciRef: Float32Array;
	utciWebgpu: Float32Array;
	toleranceC?: number;
}): ParityResult {
	const { utciRef, utciWebgpu, toleranceC = 1 } = params;
	if (utciRef.length !== utciWebgpu.length) {
		throw new Error(`Length mismatch: ref ${utciRef.length} vs webgpu ${utciWebgpu.length}`);
	}
	const n = utciRef.length;
	if (n === 0) {
		return { pass: true, rmse: 0, maxError: 0, withinTolerancePct: 100, numPoints: 0 };
	}
	let sumSq = 0;
	let maxError = 0;
	let within = 0;
	for (let i = 0; i < n; i++) {
		const d = utciWebgpu[i] - utciRef[i];
		sumSq += d * d;
		const absD = Math.abs(d);
		if (absD > maxError) maxError = absD;
		if (absD <= toleranceC) within++;
	}
	const rmse = Math.sqrt(sumSq / n);
	const withinTolerancePct = (100 * within) / n;
	const pass = maxError <= toleranceC;
	return {
		pass,
		rmse,
		maxError,
		withinTolerancePct,
		numPoints: n
	};
}

/**
 * Compare full-day reference vs WebGPU (all hours). Returns one result per hour and an overall pass.
 */
export function compareParityFullDay(params: {
	utciRefByHour: Float32Array[];
	utciWebgpuByHour: Float32Array[];
	toleranceC?: number;
}): { byHour: ParityResult[]; overallPass: boolean; worstHour: number } {
	const { utciRefByHour, utciWebgpuByHour, toleranceC = 1 } = params;
	if (utciRefByHour.length !== utciWebgpuByHour.length) {
		throw new Error(
			`Hour count mismatch: ref ${utciRefByHour.length} vs webgpu ${utciWebgpuByHour.length}`
		);
	}
	const byHour: ParityResult[] = [];
	let worstHour = 0;
	let worstMax = -1;
	for (let h = 0; h < utciRefByHour.length; h++) {
		const r = compareParity({
			utciRef: utciRefByHour[h],
			utciWebgpu: utciWebgpuByHour[h],
			toleranceC
		});
		byHour.push(r);
		if (r.maxError > worstMax) {
			worstMax = r.maxError;
			worstHour = h;
		}
	}
	const overallPass = byHour.every((r) => r.pass);
	return { byHour, overallPass, worstHour };
}
