export interface TimeGridSizeParams {
	numPoints: number;
	numHours: number;
	numMonths: number;
}

export interface SolarBitmaskSizeParams {
	numPoints: number;
	totalTimeSteps: number;
}

export interface AllHoursBufferSizes {
	totalTimeSteps: number;
	solarExposureBytes: number;
	skyExposureBytes: number;
	utciAllHoursBytes: number;
	mrtAllHoursBytes: number;
	cpuInt16UtciBytes: number;
}

export interface OneHourOutputSizes {
	utciF32Bytes: number;
	mrtF32Bytes: number;
	combinedF32Bytes: number;
	packedMrtUtciBytes: number;
}

function assertPositiveInteger(value: number, name: string): void {
	if (!Number.isInteger(value) || value <= 0) {
		throw new Error(`${name} must be a positive integer`);
	}
}

export function calculateSolarBitmaskBytes(params: SolarBitmaskSizeParams): number {
	const { numPoints, totalTimeSteps } = params;
	assertPositiveInteger(numPoints, 'numPoints');
	assertPositiveInteger(totalTimeSteps, 'totalTimeSteps');
	return Math.ceil((numPoints * totalTimeSteps) / 32) * 4;
}

export function calculateAllHoursBufferSizes(params: TimeGridSizeParams): AllHoursBufferSizes {
	const { numPoints, numHours, numMonths } = params;
	assertPositiveInteger(numPoints, 'numPoints');
	assertPositiveInteger(numHours, 'numHours');
	assertPositiveInteger(numMonths, 'numMonths');

	const totalTimeSteps = numHours * numMonths;
	const solarExposureBytes = calculateSolarBitmaskBytes({ numPoints, totalTimeSteps });
	const skyExposureBytes = numPoints * 4;
	const utciAllHoursBytes = numPoints * totalTimeSteps * 4;
	const mrtAllHoursBytes = utciAllHoursBytes;
	const cpuInt16UtciBytes = numPoints * totalTimeSteps * 2;

	return {
		totalTimeSteps,
		solarExposureBytes,
		skyExposureBytes,
		utciAllHoursBytes,
		mrtAllHoursBytes,
		cpuInt16UtciBytes
	};
}

export function calculateOneHourOutputSizes(params: { numPoints: number }): OneHourOutputSizes {
	const { numPoints } = params;
	assertPositiveInteger(numPoints, 'numPoints');

	const utciF32Bytes = numPoints * 4;
	const mrtF32Bytes = numPoints * 4;
	const combinedF32Bytes = utciF32Bytes + mrtF32Bytes;
	const packedMrtUtciBytes = numPoints * 4;

	return {
		utciF32Bytes,
		mrtF32Bytes,
		combinedF32Bytes,
		packedMrtUtciBytes
	};
}
