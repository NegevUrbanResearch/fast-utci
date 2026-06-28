import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const sceneRoot = resolve(__dirname, '../../src/lib/components/scene');

function readSceneComponent(fileName: string): string {
	return readFileSync(resolve(sceneRoot, fileName), 'utf8');
}

describe('accepted GPU resident output release call sites', () => {
	it.each(['UTCIPointCloud.svelte', 'ComparisonRenderer.svelte'])(
		'%s uses the shared surface sync helper and exactly-once release notifier',
		(fileName) => {
			const source = readSceneComponent(fileName);

			expect(source).toContain(
				'createAcceptedGpuResidentSurfaceSync'
			);
			expect(source).toContain(
				"from '$lib/components/scene/acceptedGpuResidentSurfaceSync'"
			);
			expect(source).toContain(
				'type AcceptedGpuResidentOutputReleaseCallback'
			);
			expect(source).toContain(
				"from '$lib/components/scene/acceptedGpuResidentOutputRelease'"
			);
			expect(source).not.toContain(
				'function invokeAcceptedGpuResidentOutputRelease'
			);
		}
	);

	it.each(['UTCIPointCloud.svelte', 'ComparisonRenderer.svelte'])(
		'%s delegates GPU-resident supersession to the shared helper',
		(fileName) => {
			const source = readSceneComponent(fileName);

			expect(source).toContain(
				'acceptedGpuResidentSurfaceSync.isSuperseded'
			);
			expect(source).toContain(
				'acceptedGpuResidentSurfaceSync.startSync'
			);
			expect(source).toContain(
				'acceptedGpuResidentSurfaceSync.completeSync'
			);
			expect(source).toContain(
				'acceptedGpuResidentSurfaceSync.failSync'
			);
		}
	);
});
