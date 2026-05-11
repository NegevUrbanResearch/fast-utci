import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const sceneRoot = resolve(__dirname, '../../src/lib/components/scene');

function readSceneComponent(fileName: string): string {
	return readFileSync(resolve(sceneRoot, fileName), 'utf8');
}

describe('accepted GPU resident output release call sites', () => {
	it.each(['UTCIPointCloud.svelte', 'ComparisonRenderer.svelte'])(
		'%s uses the shared exactly-once release notifier',
		(fileName) => {
			const source = readSceneComponent(fileName);

			expect(source).toContain(
				'createAcceptedGpuResidentOutputReleaseNotifier'
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
		'%s gates GPU-resident supersession with controller instance id',
		(fileName) => {
			const source = readSceneComponent(fileName);

			expect(source).toContain(
				'const controllerInstanceId = liveSelectedHourSurfaceIdentity.controllerInstanceId;'
			);
			expect(source).toContain(
				'controllerInstanceId: number;'
			);
			expect(source).toContain(
				'controllerInstanceId,'
			);
			expect(source).toContain(
				'liveSelectedHourSurfaceIdentity?.controllerInstanceId !== controllerInstanceId'
			);
		}
	);
});
