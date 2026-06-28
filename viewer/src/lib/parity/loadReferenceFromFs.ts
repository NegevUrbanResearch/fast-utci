/** Parse full-day .bin format (mirrors dataLoader.parseFullDayBinary) so this module is Node-safe (no $app/paths). */
function parseFullDayBinaryNode(buffer: ArrayBuffer): {
	numPositions: number;
	numHours: number;
	positions: Float32Array;
	utciByHour: Float32Array[];
} {
	const dataView = new DataView(buffer);
	let offset = 0;
	const numPositions = dataView.getUint32(offset, true);
	offset += 4;
	const numHours = dataView.getUint32(offset, true);
	offset += 4;
	const positions = new Float32Array(numPositions * 3);
	for (let i = 0; i < numPositions * 3; i++) {
		positions[i] = dataView.getFloat32(offset, true);
		offset += 4;
	}
	const oldFormatSize = 8 + numPositions * 12 + numPositions * 4 * numHours;
	const isNewFormat = buffer.byteLength > oldFormatSize;
	if (isNewFormat) {
		const hasShadingIndex = dataView.getUint32(offset, true) === 1;
		offset += 4;
		if (hasShadingIndex) offset += numPositions * 4;
	}
	const utciByHour: Float32Array[] = [];
	for (let hour = 0; hour < numHours; hour++) {
		const hourValues = new Float32Array(numPositions);
		for (let i = 0; i < numPositions; i++) {
			hourValues[i] = dataView.getFloat32(offset, true);
			offset += 4;
		}
		utciByHour.push(hourValues);
	}
	return { numPositions, numHours, positions, utciByHour };
}

export interface ReferenceMetadata {
	num_positions: number;
	hours: number[];
	analysis_type: string;
	coordinate_system?: string;
	[key: string]: unknown;
}

export interface ReferenceData {
	metadata: ReferenceMetadata;
	data: {
		numPositions: number;
		numHours: number;
		positions: Float32Array;
		utciByHour: Float32Array[];
	};
}

/**
 * Load reference analysis from filesystem (.bin + .json). Node only.
 * @param basePath - Full path without extension, e.g. .../Ben-Gurion/20250815_grid_2m_fullday
 */
export async function loadReferenceFromFs(basePath: string): Promise<ReferenceData> {
	const { readFileSync } = await import('node:fs');
	const metadataPath = `${basePath}.json`;
	const binaryPath = `${basePath}.bin`;
	let metadata: ReferenceMetadata;
	try {
		metadata = JSON.parse(readFileSync(metadataPath, 'utf8'));
	} catch (e) {
		throw new Error(`Failed to load metadata from ${metadataPath}: ${e}`);
	}
	let buffer: Buffer;
	try {
		buffer = readFileSync(binaryPath);
	} catch (e) {
		throw new Error(`Failed to load binary from ${binaryPath}: ${e}`);
	}
	const ownedBuffer = new Uint8Array(buffer.byteLength);
	ownedBuffer.set(buffer);
	const data = parseFullDayBinaryNode(ownedBuffer.buffer);
	return {
		metadata,
		data: {
			numPositions: data.numPositions,
			numHours: data.numHours,
			positions: data.positions,
			utciByHour: data.utciByHour
		}
	};
}
