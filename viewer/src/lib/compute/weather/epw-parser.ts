export interface EPWLocation {
	city: string;
	country: string;
	lat: number;
	lon: number;
	timezone: number;
	elevation: number;
}

export interface HourData {
	dryBulb: number;
	relHumidity: number;
	directNormal: number;
	diffuseHoriz: number;
	windSpeed: number;
	horizIR: number;
}

export interface EPWData {
	location: EPWLocation;
	dryBulbTemp: Float32Array;
	relativeHumidity: Float32Array;
	directNormalRad: Float32Array;
	diffuseHorizRad: Float32Array;
	windSpeed: Float32Array;
	horizInfrared: Float32Array;
	getHourData(month: number, day: number, hour: number): HourData | undefined;
}

const DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

function getHourIndex(month: number, day: number, hour: number): number {
	// month is 1-12, day is 1-31, hour is 1-24
	let dayOfYear = 0;
	for (let i = 0; i < month - 1; i++) {
		dayOfYear += DAYS_IN_MONTH[i];
	}
	dayOfYear += (day - 1);
	return dayOfYear * 24 + (hour - 1);
}

export function parseEPW(content: string): EPWData {
	const lines = content.split(/\r?\n/);
	
	// Parse location (line 0)
	// LOCATION,City,State,Country,Source,WMO,Lat,Lon,Timezone,Elevation
	const locCols = lines[0].split(',');
	const location: EPWLocation = {
		city: locCols[1],
		country: locCols[3],
		lat: parseFloat(locCols[6]),
		lon: parseFloat(locCols[7]),
		timezone: parseFloat(locCols[8]),
		elevation: parseFloat(locCols[9])
	};

	let numDataLines = 0;
	// Count data lines
	for (let i = 8; i < lines.length; i++) {
		if (lines[i].trim().length > 0) numDataLines++;
	}

	const dryBulbTemp = new Float32Array(numDataLines);
	const relativeHumidity = new Float32Array(numDataLines);
	const directNormalRad = new Float32Array(numDataLines);
	const diffuseHorizRad = new Float32Array(numDataLines);
	const windSpeed = new Float32Array(numDataLines);
	const horizInfrared = new Float32Array(numDataLines);

	let index = 0;
	for (let i = 8; i < lines.length; i++) {
		const line = lines[i].trim();
		if (line.length === 0) continue;

		const cols = line.split(',');
		dryBulbTemp[index] = parseFloat(cols[6]);
		relativeHumidity[index] = parseFloat(cols[8]);
		horizInfrared[index] = parseFloat(cols[12]);
		directNormalRad[index] = parseFloat(cols[14]);
		diffuseHorizRad[index] = parseFloat(cols[15]);
		windSpeed[index] = parseFloat(cols[21]);
		index++;
	}

	return {
		location,
		dryBulbTemp,
		relativeHumidity,
		directNormalRad,
		diffuseHorizRad,
		windSpeed,
		horizInfrared,
		getHourData: (month: number, day: number, hour: number) => {
			const idx = getHourIndex(month, day, hour);
			if (idx >= 0 && idx < numDataLines) {
				return {
					dryBulb: dryBulbTemp[idx],
					relHumidity: relativeHumidity[idx],
					directNormal: directNormalRad[idx],
					diffuseHoriz: diffuseHorizRad[idx],
					windSpeed: windSpeed[idx],
					horizIR: horizInfrared[idx]
				};
			}
			return undefined;
		}
	};
}
