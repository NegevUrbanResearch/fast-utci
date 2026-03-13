import { describe, it, expect } from 'vitest';
import { calculateUTCI } from '$lib/compute/utci';

describe('UTCI Polynomial', () => {
  it('should return comfortable UTCI for mild conditions', () => {
    const utci = calculateUTCI(22, 22, 1.0, 50); // 22°C, no rad, 1m/s, 50%RH
    expect(utci).toBeGreaterThan(18);
    expect(utci).toBeLessThan(26);
  });

  it('should return heat stress for extreme conditions', () => {
    const utci = calculateUTCI(40, 60, 0.5, 30);
    expect(utci).toBeGreaterThan(40);
  });

  it('should return cold stress for cold conditions', () => {
    const utci = calculateUTCI(-10, -10, 5.0, 80);
    expect(utci).toBeLessThan(-5);
  });

  it('should clamp wind speed to minimum 0.5 m/s', () => {
    const utci = calculateUTCI(22, 22, 0, 50);
    const utci05 = calculateUTCI(22, 22, 0.5, 50);
    expect(utci).toBeCloseTo(utci05, 2);
  });

  // Example cases from pythermalcomfort docs:
  // tdb=25, tr=25, v=1.0, rh=50 -> 24.6
  it('should match pythermalcomfort reference case 1', () => {
    const utci = calculateUTCI(25, 25, 1.0, 50);
    expect(utci).toBeCloseTo(24.6, 1);
  });
});
