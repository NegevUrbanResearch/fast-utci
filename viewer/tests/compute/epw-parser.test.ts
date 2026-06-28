import { describe, it, expect } from 'vitest';
import { parseEPW } from '$lib/compute/weather/epw-parser';

// Minimal EPW content for testing (8 header lines + a few data lines)
const MINIMAL_EPW = `LOCATION,Beer Sheva,ISR,source,wmo,31.25,34.79,2,300
DESIGN CONDITIONS,dummy
TYPICAL/EXTREME,dummy
GROUND TEMPERATURES,dummy
HOLIDAYS/DAYLIGHT,dummy
COMMENTS 1,dummy
COMMENTS 2,dummy
DATA PERIODS,dummy
2020,1,1,1,0,?,25.0,20.0,50,99999,0,0,800,0,200,150,0,0,0,0,0,3.5
2020,1,1,2,0,?,24.5,19.0,48,99999,0,0,810,0,250,180,0,0,0,0,0,4.0`;

describe('EPW Parser', () => {
	it('should parse header and extract location', () => {
		const data = parseEPW(MINIMAL_EPW);
		expect(data.location.city).toBe('Beer Sheva');
	});

	it('should extract hourly dry bulb temperature', () => {
		const data = parseEPW(MINIMAL_EPW);
		expect(data.dryBulbTemp[0]).toBeCloseTo(25.0, 1);
	});

	it('should extract hourly relative humidity', () => {
		const data = parseEPW(MINIMAL_EPW);
		expect(data.relativeHumidity[0]).toBeCloseTo(50, 1);
	});

	it('should extract direct normal radiation', () => {
		const data = parseEPW(MINIMAL_EPW);
		expect(data.directNormalRad[0]).toBeCloseTo(200, 1);
	});

	it('should extract diffuse horizontal radiation', () => {
		const data = parseEPW(MINIMAL_EPW);
		expect(data.diffuseHorizRad[0]).toBeCloseTo(150, 1);
	});

	it('should extract wind speed', () => {
		const data = parseEPW(MINIMAL_EPW);
		expect(data.windSpeed[0]).toBeCloseTo(3.5, 1);
	});

	it('should extract horizontal IR', () => {
		const data = parseEPW(MINIMAL_EPW);
		expect(data.horizInfrared[0]).toBeCloseTo(800, 1);
	});

	it('should extract data for specific month/day/hour', () => {
		const data = parseEPW(MINIMAL_EPW);
		const hourData = data.getHourData(1, 1, 1); // Jan 1, 1 AM (index 0)
		expect(hourData).toBeDefined();
		if (hourData) {
			expect(hourData.dryBulb).toBeCloseTo(25.0, 1);
		}
	});

	it('should handle hour-of-year index correctly', () => {
		// Just a pure function test to verify day-of-year calculation
		// We'll test this via getHourData
		const data = parseEPW(MINIMAL_EPW);
		const hourData = data.getHourData(1, 1, 2); // Jan 1, 2 AM (index 1)
		expect(hourData).toBeDefined();
		if (hourData) {
			expect(hourData.dryBulb).toBeCloseTo(24.5, 1);
		}
	});
});
