import { describe, expect, it } from 'vitest';
import { Mesh } from 'three';
import {
	buildUtciSurfaceDiagnostics,
	buildCpuPublicationDiagnostics,
	getAcceptedGpuResidentKey,
	isComputeBufferUtciSurface
} from '../../src/lib/components/scene/utciSurfaceSync';

describe('utciSurfaceSync', () => {
	it('builds a stable accepted GPU resident key from request and range', () => {
		const key = getAcceptedGpuResidentKey({
			requestId: 5,
			monthIndex: 7,
			hourIndex: 12,
			timeIndex: 180,
			utciRange: { min: 18.5, max: 41.25 },
			output: {
				format: 'f32-utci',
				numPoints: 2,
				timeIndex: 180,
				gpuBuffer: {} as GPUBuffer
			}
		});

		expect(key).toBe('5:7:180:18.5:41.25');
	});

	it('does not treat userData alone as authoritative compute-buffer proof', () => {
		const mesh = new Mesh();
		mesh.userData.utciSurfaceSource = 'compute-buffer-selected-hour';
		expect(isComputeBufferUtciSurface(mesh)).toBe(false);
	});

	it('builds request-scoped CPU publication diagnostics for non-compute surfaces', () => {
		const mesh = new Mesh();
		const diagnostics = buildCpuPublicationDiagnostics({
			mesh,
			liveSelectedHourSurfaceIdentity: {
				controllerIdentity: 'test-controller',
				controllerInstanceId: 0,
				requestId: 9,
				monthIndex: 1,
				hourIndex: 8,
				timeIndex: 32,
				selectionKey: 'selection',
				pendingRenderUpdateStartedAt: undefined,
				acceptedGpuResidentOutput: null
			}
		});

		expect(diagnostics).toMatchObject({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			cpuPublishRequestId: 9,
			cpuPublishSelectionKey: 'selection'
		});
	});

	it('assembles common surface diagnostics from mesh and copy state', () => {
		const mesh = new Mesh();
		mesh.userData.utciSurfaceSource = 'data-texture';
		mesh.userData.selectedHourTransferCount = 3;
		mesh.userData.dataTextureBuildCount = 2;
		mesh.userData.renderOwnedSelectedHourBytes = 1024;

		const diagnostics = buildUtciSurfaceDiagnostics({
			mesh,
			cpuPublicationDiagnostics: {
				utciSurfaceSource: 'cpu-uploaded-selected-hour',
				cpuPublishRequestId: 9
			},
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 9,
			gpuResidentRenderTimings: {
				renderBufferCopyMs: 1.5
			}
		});

		expect(diagnostics).toMatchObject({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			selectedHourTransferCount: 3,
			dataTextureBuildCount: 2,
			renderOwnedSelectedHourBytes: 1024,
			cpuPublishRequestId: 9,
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 9,
			renderBufferCopyMs: 1.5
		});
	});

	it('includes render publication diagnostics in UTCI surface diagnostics', () => {
		const renderPublication = {
			renderPublicationVersion: 1 as const,
			renderPublicationPath: 'compute-buffer-selected-hour' as const,
			renderPublicationPhase: 'scrub' as const,
			renderPublicationMeshAction: 'reused' as const,
			renderPublicationPointCount: 100,
			renderPublicationTargetByteLength: 400,
			renderPublicationTimeline: {
				routeProjectedAtMs: 101,
				sceneSurfaceReceivedAtMs: 103,
				sceneSyncCompletedAtMs: 107
			}
		};
		const diagnostics = buildUtciSurfaceDiagnostics({
			mesh: {
				userData: {
					utciSurfaceSource: 'compute-buffer-selected-hour',
					renderOwnedSelectedHourBytes: 2048
				}
			} as any,
			gpuResidentCopyStatus: 'complete',
			gpuResidentRenderTimings: {
				renderSceneSyncTotalMs: 10,
				renderPublication
			}
		});

		renderPublication.renderPublicationTimeline.sceneSyncCompletedAtMs = 999;

		expect(diagnostics.renderPublication).toMatchObject({
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: 'scrub',
			renderPublicationMeshAction: 'reused',
			renderPublicationTimeline: {
				routeProjectedAtMs: 101,
				sceneSurfaceReceivedAtMs: 103,
				sceneSyncCompletedAtMs: 107
			}
		});
	});
});
