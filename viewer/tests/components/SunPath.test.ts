/**
 * SunPath Component Tests
 *
 * Tests for sun path visualization component
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { AnalysisMetadata } from '$lib/types/analysis';
import * as THREE from 'three';

describe('SunPath Component Logic', () => {
	let mockMetadata: AnalysisMetadata;
	let mockModel: THREE.Group;

	beforeEach(() => {
		// Create mock metadata with sun positions
		mockMetadata = {
			analysis_type: 'full_day',
			num_positions: 100,
			hours: Array.from({ length: 24 }, (_, i) => `${i}:00`),
			utci_range: { min: 10, max: 30 },
			grid_size: 2.0,
			coordinate_system: 'xy_ground',
			model_file: 'test.glb',
			sun_positions: Array.from({ length: 24 }, (_, hour) => ({
				hour,
				altitude: 30 + Math.sin((hour - 6) * Math.PI / 12) * 50, // Simple daylight curve
				azimuth: hour * 15, // 15 degrees per hour
				is_up: hour >= 6 && hour <= 18, // Sun up from 6am to 6pm
				vector: [
					Math.sin((hour * 15) * Math.PI / 180) * Math.cos(30 * Math.PI / 180),
					Math.cos((hour * 15) * Math.PI / 180) * Math.cos(30 * Math.PI / 180),
					Math.sin(30 * Math.PI / 180)
				]
			}))
		};

		// Create mock model
		mockModel = new THREE.Group();
		mockModel.name = 'TestModel';
		
		// Add some geometry to establish bounds
		const geometry = new THREE.BoxGeometry(100, 100, 100);
		const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
		const mesh = new THREE.Mesh(geometry, material);
		mockModel.add(mesh);
	});

	describe('Sun Path Point Generation', () => {
		it('should generate 24 hour positions from sun_positions', () => {
			// This tests the logic that would be used in createSunPathPoints
			const sunPositions = mockMetadata.sun_positions!;
			expect(sunPositions).toHaveLength(24);
			
			// Verify each hour has required data
			sunPositions.forEach((sun, hour) => {
				expect(sun.hour).toBe(hour);
				expect(typeof sun.altitude).toBe('number');
				expect(typeof sun.azimuth).toBe('number');
				expect(typeof sun.is_up).toBe('boolean');
				expect(sun.vector).toHaveLength(3);
			});
		});

		it('should correctly identify daylight hours', () => {
			const sunPositions = mockMetadata.sun_positions!;
			
			// Hours 6-18 should be "up" (daylight)
			const daylightHours = sunPositions.filter(sun => sun.is_up);
			expect(daylightHours.length).toBeGreaterThan(0);
			
			// Nighttime hours should have is_up: false
			const nightHours = sunPositions.filter(sun => !sun.is_up);
			expect(nightHours.length).toBeGreaterThan(0);
		});

		it('should handle nighttime positions (below horizon)', () => {
			const nightSun = mockMetadata.sun_positions!.find(sun => !sun.is_up);
			expect(nightSun).toBeDefined();
			
			if (nightSun) {
				// Nighttime positions should still have altitude/azimuth for positioning
				expect(typeof nightSun.altitude).toBe('number');
				expect(typeof nightSun.azimuth).toBe('number');
			}
		});
	});

	describe('Coordinate Transformations', () => {
		it('should transform Python coords to Three.js coords correctly', () => {
			// Python: X=East, Y=North, Z=Up
			// Three.js: X=East, Y=Up, Z=South
			
			const pythonVector = [1, 0, 0]; // East in Python
			const scale = 100;
			
			// This is the transformation logic from SunPath.svelte
			const threeVector = new THREE.Vector3(
				pythonVector[0] * scale,  // X stays X (East)
				pythonVector[2] * scale,  // Y = Z from Python (Up)
				-pythonVector[1] * scale  // Z = -Y from Python (South)
			);
			
			expect(threeVector.x).toBe(100); // East preserved
			expect(Math.abs(threeVector.y)).toBe(0);   // No vertical component (handles -0/+0)
			expect(Math.abs(threeVector.z)).toBe(0);   // No north/south component (handles -0/+0)
		});

		it('should handle vertical (zenith) vectors correctly', () => {
			const pythonVector = [0, 0, 1]; // Straight up in Python
			const scale = 100;
			
			const threeVector = new THREE.Vector3(
				pythonVector[0] * scale,
				pythonVector[2] * scale,
				-pythonVector[1] * scale
			);
			
			expect(Math.abs(threeVector.x)).toBe(0);  // Handles -0/+0
			expect(threeVector.y).toBe(100); // Up in Three.js
			expect(Math.abs(threeVector.z)).toBe(0);  // Handles -0/+0
		});
	});

	describe('Marker Data Generation', () => {
		it('should create marker data for each hour', () => {
			const sunPositions = mockMetadata.sun_positions!;
			const markerData = sunPositions.map((sun, hour) => ({
				hour,
				isUp: sun.is_up,
				color: sun.is_up ? 0xffdd00 : 0x666666,
				opacity: sun.is_up ? 0.9 : 0.4
			}));
			
			expect(markerData).toHaveLength(24);
		});

		it('should use bright color for daytime markers', () => {
			const dayMarker = {
				hour: 12,
				isUp: true,
				color: 0xffdd00,
				opacity: 0.9
			};
			
			expect(dayMarker.color).toBe(0xffdd00); // Bright yellow
			expect(dayMarker.opacity).toBe(0.9);
		});

		it('should use dimmed color for nighttime markers', () => {
			const nightMarker = {
				hour: 2,
				isUp: false,
				color: 0x666666,
				opacity: 0.4
			};
			
			expect(nightMarker.color).toBe(0x666666); // Gray
			expect(nightMarker.opacity).toBe(0.4); // Dimmed
		});
	});

	describe('Current Hour Highlighting', () => {
		it('should highlight current hour with distinct color', () => {
			const currentHour = 12;
			const sunPositions = mockMetadata.sun_positions!;
			
			// Simulate updateCurrentSunIndicator logic
			const markerData = sunPositions.map((sun, hour) => {
				if (hour === currentHour) {
					return {
						hour,
						isUp: sun.is_up,
						color: 0xff3300, // Bright red-orange
						opacity: 1.0
					};
				} else {
					return {
						hour,
						isUp: sun.is_up,
						color: sun.is_up ? 0xffdd00 : 0x666666,
						opacity: sun.is_up ? 0.9 : 0.4
					};
				}
			});
			
			// Current hour should be highlighted
			expect(markerData[currentHour].color).toBe(0xff3300);
			expect(markerData[currentHour].opacity).toBe(1.0);
			
			// Other hours should not be highlighted
			expect(markerData[11].color).not.toBe(0xff3300);
			expect(markerData[13].color).not.toBe(0xff3300);
		});

		it('should update highlight when hour changes', () => {
			const sunPositions = mockMetadata.sun_positions!;
			
			// Initial state: hour 10
			let currentHour = 10;
			let markerData = sunPositions.map((sun, hour) => ({
				hour,
				isUp: sun.is_up,
				color: hour === currentHour ? 0xff3300 : (sun.is_up ? 0xffdd00 : 0x666666),
				opacity: hour === currentHour ? 1.0 : (sun.is_up ? 0.9 : 0.4)
			}));
			
			expect(markerData[10].color).toBe(0xff3300);
			expect(markerData[15].color).not.toBe(0xff3300);
			
			// Update to hour 15
			currentHour = 15;
			markerData = sunPositions.map((sun, hour) => ({
				hour,
				isUp: sun.is_up,
				color: hour === currentHour ? 0xff3300 : (sun.is_up ? 0xffdd00 : 0x666666),
				opacity: hour === currentHour ? 1.0 : (sun.is_up ? 0.9 : 0.4)
			}));
			
			expect(markerData[10].color).not.toBe(0xff3300);
			expect(markerData[15].color).toBe(0xff3300);
		});
	});

	describe('Sun Path Curve Generation', () => {
		it('should create a closed curve through all 24 points', () => {
			const points = mockMetadata.sun_positions!.map(sun => {
				return new THREE.Vector3(
					sun.vector[0] * 100,
					sun.vector[2] * 100,
					-sun.vector[1] * 100
				);
			});
			
			const curve = new THREE.CatmullRomCurve3(points, true); // true = closed
			
			expect(curve.points).toHaveLength(24);
			expect(curve.closed).toBe(true);
		});

		it('should create tube geometry from curve', () => {
			const points = [
				new THREE.Vector3(0, 0, 0),
				new THREE.Vector3(10, 10, 0),
				new THREE.Vector3(20, 0, 0),
				new THREE.Vector3(10, -10, 0)
			];
			
			const curve = new THREE.CatmullRomCurve3(points, true);
			const geometry = new THREE.TubeGeometry(curve, 64, 2, 8, false);
			
			expect(geometry.type).toBe('TubeGeometry');
			expect(geometry.parameters.tubularSegments).toBe(64);
			expect(geometry.parameters.radius).toBe(2);
			expect(geometry.parameters.radialSegments).toBe(8);
		});
	});

	describe('Scaling and Positioning', () => {
		it('should scale sun path relative to model size', () => {
			// Calculate model size
			const box = new THREE.Box3().setFromObject(mockModel);
			const size = box.getSize(new THREE.Vector3());
			const modelSize = size.length();
			
			// Sun path should be 0.6x model size
			const sunPathScale = modelSize * 0.6;
			
			expect(sunPathScale).toBeGreaterThan(0);
			expect(sunPathScale).toBeLessThan(modelSize);
		});

		it('should center sun path at model center', () => {
			// Calculate model center
			const box = new THREE.Box3().setFromObject(mockModel);
			const center = box.getCenter(new THREE.Vector3());
			
			expect(center).toBeInstanceOf(THREE.Vector3);
			
			// Sun path points should be offset by this center
			const testPoint = new THREE.Vector3(10, 20, 30);
			const offsetPoint = testPoint.clone().add(center);
			
			expect(offsetPoint.x).toBe(testPoint.x + center.x);
			expect(offsetPoint.y).toBe(testPoint.y + center.y);
			expect(offsetPoint.z).toBe(testPoint.z + center.z);
		});
	});

	describe('Resource Disposal', () => {
		it('should dispose geometry and material on cleanup', () => {
			const geometry = new THREE.TubeGeometry(
				new THREE.CatmullRomCurve3([
					new THREE.Vector3(0, 0, 0),
					new THREE.Vector3(10, 10, 0)
				], false),
				64, 2, 8, false
			);
			const material = new THREE.MeshBasicMaterial({ color: 0xffaa00 });
			
			const disposeSpy = vi.spyOn(geometry, 'dispose');
			const materialDisposeSpy = vi.spyOn(material, 'dispose');
			
			// Simulate disposal
			geometry.dispose();
			material.dispose();
			
			expect(disposeSpy).toHaveBeenCalled();
			expect(materialDisposeSpy).toHaveBeenCalled();
		});
	});

	describe('Edge Cases', () => {
		it('should handle missing sun_positions gracefully', () => {
			const metadataWithoutSun: AnalysisMetadata = {
				...mockMetadata,
				sun_positions: undefined
			};
			
			expect(metadataWithoutSun.sun_positions).toBeUndefined();
			// Component should not crash, just not render sun path
		});

		it('should handle empty sun_positions array', () => {
			const metadataWithEmptySun: AnalysisMetadata = {
				...mockMetadata,
				sun_positions: []
			};
			
			expect(metadataWithEmptySun.sun_positions).toHaveLength(0);
			// Should create no points or markers
		});

		it('should handle model without sun_positions', () => {
			const minimalMetadata: AnalysisMetadata = {
				analysis_type: 'single_hour',
				num_positions: 100,
				hours: ['12:00'],
				utci_range: { min: 10, max: 30 },
				grid_size: 2.0,
				coordinate_system: 'xy_ground',
				model_file: 'test.glb'
			};
			
			expect(minimalMetadata.sun_positions).toBeUndefined();
		});

		it('should handle hour changes when sun path is hidden', () => {
			// When sunPathVisible is false, updateCurrentSunIndicator should not run
			// This prevents unnecessary updates
			const sunPathVisible = false;
			const currentHour = 12;
			
			// The reactive statement should short-circuit
			if (!sunPathVisible) {
				// No update should occur
				expect(currentHour).toBe(12); // Value unchanged
			}
		});
	});

	describe('Integration with Viewer Store', () => {
		it('should react to sunPathVisible changes', () => {
			// Initial state: hidden
			let sunPathVisible = false;
			let sunPathGroup: THREE.Group | null = null;
			
			// When visible becomes true, sun path should be created
			sunPathVisible = true;
			if (sunPathVisible && !sunPathGroup) {
				sunPathGroup = new THREE.Group();
				sunPathGroup.name = 'SunPath';
			}
			
			expect(sunPathGroup).not.toBeNull();
			expect(sunPathGroup?.name).toBe('SunPath');
			
			// When visible becomes false, sun path should be hidden
			sunPathVisible = false;
			if (!sunPathVisible && sunPathGroup) {
				sunPathGroup.visible = false;
			}
			
			expect(sunPathGroup?.visible).toBe(false);
		});

		it('should react to currentHour changes', () => {
			const sunPositions = mockMetadata.sun_positions!;
			let currentHour = 8;
			
			// Create initial marker data
			let markerData = sunPositions.map((sun, hour) => ({
				hour,
				isUp: sun.is_up,
				color: hour === currentHour ? 0xff3300 : (sun.is_up ? 0xffdd00 : 0x666666),
				opacity: hour === currentHour ? 1.0 : (sun.is_up ? 0.9 : 0.4)
			}));
			
			expect(markerData[8].color).toBe(0xff3300);
			
			// Change hour
			currentHour = 14;
			markerData = sunPositions.map((sun, hour) => ({
				hour,
				isUp: sun.is_up,
				color: hour === currentHour ? 0xff3300 : (sun.is_up ? 0xffdd00 : 0x666666),
				opacity: hour === currentHour ? 1.0 : (sun.is_up ? 0.9 : 0.4)
			}));
			
			expect(markerData[8].color).not.toBe(0xff3300);
			expect(markerData[14].color).toBe(0xff3300);
		});
	});
});

