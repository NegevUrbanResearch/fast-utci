/**
 * Convert positions from analysis (xy_ground) to Three.js world (Y-up).
 * xy_ground: (x, y, z) → world: (x, z, -y). In-place safe if out === positions.
 */
export function analysisPositionsToWorld(
	positions: Float32Array,
	coordinateSystem: 'xy_ground' | 'xz_ground'
): Float32Array {
	const n = positions.length / 3;
	const out = new Float32Array(positions.length);
	if (coordinateSystem === 'xy_ground') {
		for (let i = 0; i < n; i++) {
			const x = positions[i * 3];
			const y = positions[i * 3 + 1];
			const z = positions[i * 3 + 2];
			out[i * 3] = x;
			out[i * 3 + 1] = z;
			out[i * 3 + 2] = -y;
		}
	} else {
		out.set(positions);
	}
	return out;
}
