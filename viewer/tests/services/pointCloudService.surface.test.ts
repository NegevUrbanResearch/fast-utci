import * as THREE from 'three';
import { beforeEach, describe, expect, it } from 'vitest';
import {
	buildUtciGridLayoutReuseProofDiagnostics,
	buildUtciGridLayout,
	createUtciLayoutReusePublicationState,
	createUtciLayoutReuseKeyForAnalysis,
	createUtciLayoutReuseKey,
	createColors,
	createUtciSurfaceMesh,
	deriveUtciLayoutFrameForTest,
	getUtciLayoutFrameCacheDiagnosticsForTest,
	getUtciLayoutReuseSignatureDiagnosticsForTest,
	isUtciLayoutReuseProofSafe,
	planUtciLayoutPublication,
	planUtciLayoutReuseCandidate,
	resetUtciLayoutFrameCachesForTest,
	resolveUtciLayoutReusePublicationStateAfterSync,
	type UtciLayoutReuseKeyDiagnostics,
	type UtciSurfaceMeshOptions,
	updateUtciSurfaceMesh
} from '$lib/services/pointCloudService';
import {
	evaluateComputeBufferUtciSurfaceLayoutCompatibility,
	createCellToPointIndexArray,
	createVertexToPointIndexArray,
	createComputeBufferUtciSurfaceMesh,
	getGpuNativeUtciSurfaceSource,
	isComputeBufferUtciSurfaceLayoutCompatible,
	updateComputeBufferUtciSurfaceMesh
} from '$lib/services/gpuUtciRenderBridge';
import type { SelectedHourRenderLayoutNormalizationSignature } from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';
import type { Analysis, SingleHourData } from '$lib/types/analysis';

function createAnalysis(params?: {
	positions?: number[];
	positionsArray?: Float32Array;
	utciValues?: number[];
	gridSize?: number;
	coordinateSystem?: 'xy_ground' | 'xz_ground';
	bounds?: { x_min: number; x_max: number; y_min: number; y_max: number; z?: number };
	source?: string;
	sourceAnalysisId?: string;
	modelFile?: string;
}): Analysis {
	const analysis = {
		metadata: {
			analysis_type: 'single_hour',
			num_positions: 4,
			hours: ['00:00'],
			utci_range: { min: 10, max: 40 },
			grid_size: params?.gridSize ?? 1,
			coordinate_system: params?.coordinateSystem ?? 'xz_ground',
			model_file: params?.modelFile ?? 'test.obj',
			source_analysis_id: params?.sourceAnalysisId,
			bounds: params?.bounds
		},
		data: {
			numPositions: 4,
			numHours: 1 as const,
			positions:
				params?.positionsArray ??
				new Float32Array(
					params?.positions ?? [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]
				),
			utciValues: new Float32Array(params?.utciValues ?? [10, 20, 30, 40])
		}
	} as Analysis & { __source?: string };

	if (params?.source) {
		analysis.__source = params.source;
	}

	return analysis;
}

function createReuseKey(analysis: Analysis) {
	return createUtciLayoutReuseKey({
		analysis,
		layout: buildUtciGridLayout(analysis),
		utciSurfaceSource: 'compute-buffer-selected-hour',
		rendererBackend: 'webgpu'
	});
}

function createPublicationReuseKey(analysis: Analysis) {
	return createUtciLayoutReuseKeyForAnalysis({
		analysis,
		utciSurfaceSource: 'compute-buffer-selected-hour',
		rendererBackend: 'webgpu'
	});
}

function cloneAnalysisWithSelectedHourValues(
	analysis: Analysis,
	values: number[]
): Analysis {
	return {
		...analysis,
		data: {
			...analysis.data,
			positions: analysis.data.positions,
			utciValues: new Float32Array(values)
		} as SingleHourData,
		metadata: {
			...analysis.metadata
		}
	};
}

function serializeExpectedNormalizationSignature(
	signature: SelectedHourRenderLayoutNormalizationSignature
): string {
	return [
		signature.enabled ? '1' : '0',
		signature.provenance,
		signature.offset.x,
		signature.offset.y,
		signature.offset.z
	].join('|');
}

function getGpuNativeColorArray(mesh: THREE.Mesh): Float32Array {
	return (
		mesh.userData.gpuNativeUtciSurfaceState.colorStorageAttribute.array as Float32Array
	);
}

function toLinearColor(r: number, g: number, b: number): THREE.Color {
	return new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
}

describe('pointCloudService UTCI surface seam', () => {
	beforeEach(() => {
		resetUtciLayoutFrameCachesForTest();
	});

	it('paints the same logical cells across DataTexture and gpuNative backends', () => {
		const analysis = createAnalysis();
		const colors = createColors(analysis, 0, 'normalized', 'utci');

		const dataTextureMesh = createUtciSurfaceMesh(analysis);
		const dataTextureLayout = dataTextureMesh.userData.utciLayout;

		expect(Array.from(dataTextureLayout.indexToRow)).toEqual([0, 1, 0, 1]);
		expect(Array.from(dataTextureLayout.indexToColumn)).toEqual([0, 0, 1, 1]);
		expect(Array.from(dataTextureLayout.indexToTexel)).toEqual([2, 0, 3, 1]);

		for (let index = 0; index < analysis.data.numPositions; index += 1) {
			const texelOffset = dataTextureLayout.indexToTexel[index] * 4;
			expect(dataTextureLayout.colorBuffer[texelOffset]).toBe(
				Math.floor(colors[index * 3] * 255)
			);
			expect(dataTextureLayout.colorBuffer[texelOffset + 1]).toBe(
				Math.floor(colors[index * 3 + 1] * 255)
			);
			expect(dataTextureLayout.colorBuffer[texelOffset + 2]).toBe(
				Math.floor(colors[index * 3 + 2] * 255)
			);
			expect(dataTextureLayout.colorBuffer[texelOffset + 3]).toBe(255);
		}

		const gpuNativeMesh = createUtciSurfaceMesh({
			analysis,
			backend: 'gpuNative'
		});
		const gpuNativeLayout = gpuNativeMesh.userData.utciLayout;
		const gpuNativeColors = getGpuNativeColorArray(gpuNativeMesh);

		expect(Array.from(gpuNativeLayout.indexToRow)).toEqual([0, 1, 0, 1]);
		expect(Array.from(gpuNativeLayout.indexToColumn)).toEqual([0, 0, 1, 1]);

		for (let index = 0; index < analysis.data.numPositions; index += 1) {
			const cellIndex =
				gpuNativeLayout.indexToRow[index] * gpuNativeLayout.width +
				gpuNativeLayout.indexToColumn[index];
			const cellOffset = cellIndex * 6 * 4;
			const expected = toLinearColor(
				colors[index * 3],
				colors[index * 3 + 1],
				colors[index * 3 + 2]
			);

			expect(gpuNativeColors[cellOffset]).toBeCloseTo(expected.r, 6);
			expect(gpuNativeColors[cellOffset + 1]).toBeCloseTo(expected.g, 6);
			expect(gpuNativeColors[cellOffset + 2]).toBeCloseTo(expected.b, 6);
			expect(gpuNativeColors[cellOffset + 3]).toBeCloseTo(0.9, 6);
		}
	});

	it('rebuilds logical fallback cell mapping from metadata.bounds for live webgpu analyses', () => {
		const analysis = createAnalysis({
			positions: [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
			coordinateSystem: 'xy_ground',
			bounds: { x_min: 0, x_max: 1, y_min: 0, y_max: 1, z: 2 },
			source: 'webgpu'
		});

		const mesh = createUtciSurfaceMesh(analysis);
		const layout = mesh.userData.utciLayout;

		expect(layout.width).toBe(2);
		expect(layout.height).toBe(2);
		expect(layout.minX).toBe(0);
		expect(layout.minZ).toBe(-1);
		expect(layout.centerX).toBe(0.5);
		expect(layout.centerZ).toBe(-0.5);
		expect(layout.baseY).toBeCloseTo(1.95, 6);
		expect(Array.from(layout.indexToRow)).toEqual([0, 1, 0, 1]);
		expect(Array.from(layout.indexToColumn)).toEqual([0, 0, 1, 1]);
		expect(Array.from(layout.indexToTexel)).toEqual([2, 0, 3, 1]);
	});

	it('records layout-build diagnostic substeps without changing the layout output', () => {
		const analysis = createAnalysis();
		const diagnostics: {
			totalMs?: number;
			arrayAllocationMs?: number;
			transformBoundsPassMs?: number;
			coordinateAssignmentMs?: number;
			indexToTexelFillMs?: number;
			cellToPointIndexBuildMs?: number;
			colorBufferAllocationMs?: number;
		} = {};

		const layout = buildUtciGridLayout(analysis, { diagnostics });

		expect(layout.width).toBe(2);
		expect(layout.height).toBe(2);
		expect(Array.from(layout.indexToRow)).toEqual([0, 1, 0, 1]);
		expect(Array.from(layout.indexToColumn)).toEqual([0, 0, 1, 1]);
		expect(Array.from(layout.indexToTexel)).toEqual([2, 0, 3, 1]);
		expect(diagnostics).toEqual({
			totalMs: expect.any(Number),
			arrayAllocationMs: expect.any(Number),
			transformBoundsPassMs: expect.any(Number),
			coordinateAssignmentMs: expect.any(Number),
			indexToTexelFillMs: expect.any(Number),
			cellToPointIndexBuildMs: expect.any(Number),
			colorBufferAllocationMs: expect.any(Number)
		});
	});

	it('reports reuse-safe proof diagnostics for the same runtime-relevant layout', () => {
		const analysis = createAnalysis();
		const previousLayout = buildUtciGridLayout(analysis);
		const nextLayout = buildUtciGridLayout(analysis);

		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout,
			canonicalRuntimeCompatibilityWouldReuse: true
		});

		expect(proof).toMatchObject({
			decision: 'reuse-safe',
			hoverCellLookupProofStatus: 'same-point-confirmed',
			previousLayoutPresent: true,
			canonicalRuntimeCompatibilityWouldReuse: true,
			proofMatchesCanonicalRuntimeCompatibility: true,
			positionsReferenceMatch: true,
			pointCountMatch: true,
			gridSizeMatch: true,
			coordinateSystemMatch: true,
			normalizationSignatureMatch: true,
			constructionModeMatch: true,
			dimensionsMatch: true,
			placementMatch: true,
			cellToPointMappingMatch: true
		});
		expect(proof.proofCostMs).toEqual(expect.any(Number));
		expect(proof.estimatedRetainedCpuLayoutBytes).toBeGreaterThan(0);
	});

	it('reports rebuild-required when the effective cell mapping changes', () => {
		const analysis = createAnalysis();
		const previousLayout = buildUtciGridLayout(analysis);
		const nextLayout = {
			...buildUtciGridLayout(analysis),
			cellToPointIndex: new Int32Array([1, 0, 3, 2])
		};

		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout,
			canonicalRuntimeCompatibilityWouldReuse: false
		});

		expect(proof.decision).toBe('rebuild-required');
		expect(proof.hoverCellLookupProofStatus).toBe('not-compatible');
		expect(proof.canonicalRuntimeCompatibilityWouldReuse).toBe(false);
		expect(proof.cellToPointMappingMatch).toBe(false);
		expect(proof.proofMatchesCanonicalRuntimeCompatibility).toBe(true);
	});

	it('uses the full compute-buffer compatibility predicate for canonical runtime reuse', () => {
		const analysis = createAnalysis();
		const previousLayout = buildUtciGridLayout(analysis);
		const nextLayout = {
			...buildUtciGridLayout(analysis),
			width: previousLayout.width + 1
		};

		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout,
			canonicalRuntimeCompatibilityWouldReuse: false
		});

		expect(proof.decision).toBe('rebuild-required');
		expect(proof.canonicalRuntimeCompatibilityWouldReuse).toBe(false);
		expect(proof.dimensionsMatch).toBe(false);
	});

	it('reports rebuild-required when placement changes in centerX, centerZ, or baseY', () => {
		const analysis = createAnalysis();
		const previousLayout = buildUtciGridLayout(analysis);

		for (const nextLayout of [
			{ ...buildUtciGridLayout(analysis), centerX: previousLayout.centerX + 1 },
			{ ...buildUtciGridLayout(analysis), centerZ: previousLayout.centerZ + 1 },
			{ ...buildUtciGridLayout(analysis), baseY: previousLayout.baseY + 1 }
		]) {
			const proof = buildUtciGridLayoutReuseProofDiagnostics({
				previousLayout,
				nextLayout,
				canonicalRuntimeCompatibilityWouldReuse: false
			});

			expect(proof.decision).toBe('rebuild-required');
			expect(proof.placementMatch).toBe(false);
		}
	});

	it('reports rebuild-required when construction mode changes', () => {
		const previousLayout = buildUtciGridLayout(createAnalysis());
		const nextLayout = buildUtciGridLayout(
			createAnalysis({
				positions: [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
				coordinateSystem: 'xy_ground',
				bounds: { x_min: 0, x_max: 1, y_min: 0, y_max: 1, z: 2 },
				source: 'webgpu'
			})
		);

		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout,
			canonicalRuntimeCompatibilityWouldReuse: false
		});

		expect(proof.decision).toBe('rebuild-required');
		expect(proof.hoverCellLookupProofStatus).toBe('not-compatible');
		expect(proof.constructionModeMatch).toBe(false);
		expect(proof.canonicalRuntimeCompatibilityWouldReuse).toBe(false);
	});

	it('reports rebuild-required when normalization signature changes', () => {
		const analysis = createAnalysis();
		const previousLayout = buildUtciGridLayout(analysis);
		const nextLayout = {
			...buildUtciGridLayout(analysis),
			normalizationSignature: {
				enabled: true,
				offset: { x: 10, y: 0, z: 0 },
				provenance: 'anchor-offset-minus-origin' as const
			}
		};

		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout,
			canonicalRuntimeCompatibilityWouldReuse: false
		});

		expect(proof.decision).toBe('rebuild-required');
		expect(proof.hoverCellLookupProofStatus).toBe('not-compatible');
		expect(proof.normalizationSignatureMatch).toBe(false);
		expect(proof.proofMatchesCanonicalRuntimeCompatibility).toBe(true);
	});

	it('reports proof-inconclusive when expensive mapping comparison is skipped', () => {
		const layout = {
			width: 1,
			height: 1,
			gridSize: 1,
			numPositions: 2,
			centerX: 0.5,
			centerZ: 0.5,
			minX: 0,
			minZ: 0,
			baseY: 0,
			coordinateSystem: 'xy_ground' as const,
			minY: 0,
			maxY: 0,
			indexToRow: new Uint32Array([0, 0]),
			indexToColumn: new Uint32Array([0, 0]),
			cellToPointIndex: new Int32Array([-2]),
			indexToTexel: new Uint32Array([0, 0]),
			colorBuffer: new Uint8Array(4)
		};
		const nextLayout = {
			...layout,
			indexToColumn: new Uint32Array([0, 1])
		};

		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout: layout,
			nextLayout,
			skipExpensiveMappingComparison: true
		});

		expect(proof.decision).toBe('proof-inconclusive');
		expect(proof.hoverCellLookupProofStatus).toBe('proof-inconclusive');
		expect(proof.canonicalRuntimeCompatibilityWouldReuse).toBeNull();
		expect(proof.cellToPointMappingMatch).toBeNull();
		expect(proof.proofMatchesCanonicalRuntimeCompatibility).toBeNull();
	});

	it('never reports reuse-safe when the proof and canonical runtime compatibility disagree', () => {
		const analysis = createAnalysis();
		const previousLayout = buildUtciGridLayout(analysis);
		const nextLayout = {
			...buildUtciGridLayout(analysis),
			baseY: previousLayout.baseY + 1
		};

		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout,
			canonicalRuntimeCompatibilityWouldReuse: true
		});

		expect(proof.decision).not.toBe('reuse-safe');
		expect(proof.hoverCellLookupProofStatus).toBe('not-compatible');
		expect(proof.proofMatchesCanonicalRuntimeCompatibility).toBe(false);
	});

	it('returns reuse-candidate when the proof is safe and the stable key matches', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: buildUtciGridLayout(analysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});

		expect(isUtciLayoutReuseProofSafe(proof)).toBe(true);
		expect(
			planUtciLayoutReuseCandidate({
				previousLayout,
				proof,
				previousKey: createReuseKey(analysis),
				currentKey: createReuseKey(analysis)
			})
		).toEqual({ action: 'reuse-candidate', reason: 'reuse-safe', keyMatch: true });
	});

	it('returns build-required when previous layout is missing', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		expect(
			planUtciLayoutReuseCandidate({
				previousLayout: null,
				proof: null,
				previousKey: null,
				currentKey: createReuseKey(analysis)
			})
		).toEqual({ action: 'build-required', reason: 'missing-previous-layout', keyMatch: false });
	});

	it('returns build-required when proof or prior key diagnostics are missing', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const currentKey = createReuseKey(analysis);

		expect(
			planUtciLayoutReuseCandidate({
				previousLayout,
				proof: null,
				previousKey: currentKey,
				currentKey
			})
		).toEqual({ action: 'build-required', reason: 'diagnostics-missing', keyMatch: false });

		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: buildUtciGridLayout(analysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});
		expect(
			planUtciLayoutReuseCandidate({
				previousLayout,
				proof,
				previousKey: null,
				currentKey
			})
		).toEqual({ action: 'build-required', reason: 'diagnostics-missing', keyMatch: false });
	});

	it('returns build-required when canonical reuse proof disagrees with runtime compatibility', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: { ...buildUtciGridLayout(analysis), baseY: previousLayout.baseY + 1 },
			canonicalRuntimeCompatibilityWouldReuse: true
		});

		expect(
			planUtciLayoutReuseCandidate({
				previousLayout,
				proof,
				previousKey: createReuseKey(analysis),
				currentKey: createReuseKey(analysis)
			})
		).toEqual({ action: 'build-required', reason: 'canonical-mismatch', keyMatch: true });
	});

	it('returns build-required when runtime compatibility says not to reuse', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: { ...buildUtciGridLayout(analysis), width: previousLayout.width + 1 },
			canonicalRuntimeCompatibilityWouldReuse: false
		});

		expect(
			planUtciLayoutReuseCandidate({
				previousLayout,
				proof,
				previousKey: createReuseKey(analysis),
				currentKey: createReuseKey(analysis)
			})
		).toEqual({ action: 'build-required', reason: 'canonical-mismatch', keyMatch: true });
	});

	it('returns build-required for mapping, hover, construction, dimensions, and placement failures', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const currentKey = createReuseKey(analysis);

		const mappingProof = {
			...buildUtciGridLayoutReuseProofDiagnostics({
				previousLayout,
				nextLayout: buildUtciGridLayout(analysis),
				canonicalRuntimeCompatibilityWouldReuse: true
			}),
			decision: 'rebuild-required' as const,
			cellToPointMappingMatch: false
		};
		expect(
			planUtciLayoutReuseCandidate({
				previousLayout,
				proof: mappingProof,
				previousKey: currentKey,
				currentKey
			})
		).toEqual({ action: 'build-required', reason: 'mapping-unsafe', keyMatch: true });

		const hoverProof = {
			...buildUtciGridLayoutReuseProofDiagnostics({
				previousLayout,
				nextLayout: buildUtciGridLayout(analysis),
				canonicalRuntimeCompatibilityWouldReuse: true
			}),
			hoverCellLookupProofStatus: 'proof-inconclusive' as const
		};
		expect(
			planUtciLayoutReuseCandidate({
				previousLayout,
				proof: hoverProof,
				previousKey: currentKey,
				currentKey
			})
		).toEqual({ action: 'build-required', reason: 'hover-proof-missing', keyMatch: true });

		for (const proof of [
			{
				...buildUtciGridLayoutReuseProofDiagnostics({
					previousLayout,
					nextLayout: buildUtciGridLayout(analysis),
					canonicalRuntimeCompatibilityWouldReuse: true
				}),
				decision: 'rebuild-required' as const,
				constructionModeMatch: false
			},
			{
				...buildUtciGridLayoutReuseProofDiagnostics({
					previousLayout,
					nextLayout: buildUtciGridLayout(analysis),
					canonicalRuntimeCompatibilityWouldReuse: true
				}),
				decision: 'rebuild-required' as const,
				dimensionsMatch: false
			},
			{
				...buildUtciGridLayoutReuseProofDiagnostics({
					previousLayout,
					nextLayout: buildUtciGridLayout(analysis),
					canonicalRuntimeCompatibilityWouldReuse: true
				}),
				decision: 'rebuild-required' as const,
				placementMatch: false
			}
		]) {
			expect(
				planUtciLayoutReuseCandidate({
					previousLayout,
					proof,
					previousKey: currentKey,
					currentKey
				})
			).toEqual({ action: 'build-required', reason: 'proof-not-safe', keyMatch: true });
		}
	});

	it('returns build-required for backend/source mismatch', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: buildUtciGridLayout(analysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});
		const currentKey = createReuseKey(analysis);

		expect(
			planUtciLayoutReuseCandidate({
				previousLayout,
				proof,
				previousKey: {
					...currentKey,
					utciSurfaceSource: 'cpu-uploaded-selected-hour'
				} as any,
				currentKey
			})
		).toEqual({
			action: 'build-required',
			reason: 'backend-or-source-mismatch',
			keyMatch: false
		});

		expect(
			planUtciLayoutReuseCandidate({
				previousLayout,
				proof,
				previousKey: {
					...currentKey,
					rendererBackend: 'cpu'
				} as any,
				currentKey
			})
		).toEqual({
			action: 'build-required',
			reason: 'backend-or-source-mismatch',
			keyMatch: false
		});
	});

	it('returns build-required when the stable layout key does not match', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: buildUtciGridLayout(analysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});
		const currentKey = createReuseKey(analysis);

		expect(
			planUtciLayoutReuseCandidate({
				previousLayout,
				proof,
				previousKey: { ...currentKey, width: currentKey.width + 1 },
				currentKey
			})
		).toEqual({ action: 'build-required', reason: 'layout-key-mismatch', keyMatch: false });
	});

	it('changes the key and returns build-required when point order changes under the same analysis id', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const reorderedAnalysis = createAnalysis({
			sourceAnalysisId: 'Ben-Gurion/base',
			positions: [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
		});
		const previousLayout = buildUtciGridLayout(analysis);
		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: buildUtciGridLayout(analysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});
		const originalKey = createReuseKey(analysis);
		const reorderedKey = createReuseKey(reorderedAnalysis);

		expect(reorderedKey.analysisId).toBe(originalKey.analysisId);
		expect(reorderedKey.layoutSourceSignature).not.toBe(
			originalKey.layoutSourceSignature
		);

		expect(
			planUtciLayoutReuseCandidate({
				previousLayout,
				proof,
				previousKey: reorderedKey,
				currentKey: originalKey
			})
		).toEqual({ action: 'build-required', reason: 'layout-key-mismatch', keyMatch: false });
	});

	it('changes the key when source positions change even if bounds and mapping stay the same', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const shiftedAnalysis = createAnalysis({
			sourceAnalysisId: 'Ben-Gurion/base',
			positions: [0, 0, 0, 0, 0.25, 1, 1, 0, 0, 1, 0, 1]
		});

		const originalLayout = buildUtciGridLayout(analysis);
		const shiftedLayout = buildUtciGridLayout(shiftedAnalysis);
		const originalKey = createReuseKey(analysis);
		const shiftedKey = createReuseKey(shiftedAnalysis);
		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout: originalLayout,
			nextLayout: originalLayout,
			canonicalRuntimeCompatibilityWouldReuse: true
		});

		expect(shiftedLayout.width).toBe(originalLayout.width);
		expect(shiftedLayout.height).toBe(originalLayout.height);
		expect(shiftedLayout.centerX).toBe(originalLayout.centerX);
		expect(shiftedLayout.centerZ).toBe(originalLayout.centerZ);
		expect(Array.from(shiftedLayout.indexToRow)).toEqual(
			Array.from(originalLayout.indexToRow)
		);
		expect(Array.from(shiftedLayout.indexToColumn)).toEqual(
			Array.from(originalLayout.indexToColumn)
		);
		expect(Array.from(shiftedLayout.indexToTexel)).toEqual(
			Array.from(originalLayout.indexToTexel)
		);
		expect(shiftedKey.layoutSourceSignature).not.toBe(
			originalKey.layoutSourceSignature
		);
		expect(
			planUtciLayoutReuseCandidate({
				previousLayout: originalLayout,
				proof,
				previousKey: shiftedKey,
				currentKey: originalKey
			})
		).toEqual({ action: 'build-required', reason: 'layout-key-mismatch', keyMatch: false });
	});

	it('reuses the cached positions signature instead of recomputing it on repeated key builds', () => {
		const sharedPositions = new Float32Array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]);
		const analysis = createAnalysis({
			sourceAnalysisId: 'Ben-Gurion/base',
			positionsArray: sharedPositions
		});
		const before =
			getUtciLayoutReuseSignatureDiagnosticsForTest()
				.positionsSourceSignatureComputationCount;

		createReuseKey(analysis);
		createReuseKey(analysis);

		const after =
			getUtciLayoutReuseSignatureDiagnosticsForTest()
				.positionsSourceSignatureComputationCount;
		expect(after - before).toBe(1);
	});

	it('reuses the cached analysis frame and reports key timing diagnostics on repeated scrub key builds', () => {
		const analysis = createAnalysis({
			sourceAnalysisId: 'Ben-Gurion/base'
		});
		const firstDiagnostics: UtciLayoutReuseKeyDiagnostics = {};
		const secondDiagnostics: UtciLayoutReuseKeyDiagnostics = {};

		const firstKey = createUtciLayoutReuseKeyForAnalysis({
			analysis,
			utciSurfaceSource: 'compute-buffer-selected-hour',
			rendererBackend: 'webgpu',
			diagnostics: firstDiagnostics
		});
		const secondKey = createUtciLayoutReuseKeyForAnalysis({
			analysis,
			utciSurfaceSource: 'compute-buffer-selected-hour',
			rendererBackend: 'webgpu',
			diagnostics: secondDiagnostics
		});

		expect(secondKey).toEqual(firstKey);
		expect(firstDiagnostics.keyBuildMs).toEqual(expect.any(Number));
		expect(firstDiagnostics.layoutSourceSignatureMs).toEqual(expect.any(Number));
		expect(firstDiagnostics.positionsSourceSignatureMs).toEqual(expect.any(Number));
		expect(firstDiagnostics.positionsSourceSignatureCacheHit).toBe(false);
		expect(firstDiagnostics.frameCacheLookupMs).toEqual(expect.any(Number));
		expect(firstDiagnostics.frameDerivationMs).toEqual(expect.any(Number));
		expect(firstDiagnostics.frameCacheHit).toBe(false);
		expect(secondDiagnostics.keyBuildMs).toEqual(expect.any(Number));
		expect(secondDiagnostics.layoutSourceSignatureMs).toEqual(expect.any(Number));
		expect(secondDiagnostics.positionsSourceSignatureMs).toEqual(expect.any(Number));
		expect(secondDiagnostics.positionsSourceSignatureCacheHit).toBe(true);
		expect(secondDiagnostics.frameCacheLookupMs).toEqual(expect.any(Number));
		expect(secondDiagnostics.frameDerivationMs).toBe(0);
		expect(secondDiagnostics.frameCacheHit).toBe(true);
	});

	it('reuses derived layout frames across structurally equivalent selected-hour analyses', () => {
		const base = createAnalysis({
			gridSize: 0.5,
			bounds: { x_min: 0, x_max: 0.5, y_min: 0, y_max: 0.5, z: 0 },
			positionsArray: new Float32Array([0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0.5, 0, 0.5])
		});
		const first = cloneAnalysisWithSelectedHourValues(base, [20, 21, 22, 23]);
		const second = cloneAnalysisWithSelectedHourValues(base, [24, 25, 26, 27]);
		const firstDiagnostics: UtciLayoutReuseKeyDiagnostics = {};
		const secondDiagnostics: UtciLayoutReuseKeyDiagnostics = {};

		const firstKey = createUtciLayoutReuseKeyForAnalysis({
			analysis: first,
			utciSurfaceSource: 'compute-buffer-selected-hour',
			rendererBackend: 'webgpu',
			diagnostics: firstDiagnostics
		});
		const secondKey = createUtciLayoutReuseKeyForAnalysis({
			analysis: second,
			utciSurfaceSource: 'compute-buffer-selected-hour',
			rendererBackend: 'webgpu',
			diagnostics: secondDiagnostics
		});

		expect(firstKey).toEqual(secondKey);
		expect(firstDiagnostics.frameCacheHit).toBe(false);
		expect(secondDiagnostics.frameCacheHit).toBe(true);
		expect(secondDiagnostics.frameDerivationMs ?? 0).toBe(0);
		const freshFrame = deriveUtciLayoutFrameForTest(second);
		expect(secondKey).toMatchObject({
			gridSize: freshFrame.gridSize,
			pointCount: freshFrame.pointCount,
			coordinateSystem: freshFrame.coordinateSystem,
			normalizationSignature: serializeExpectedNormalizationSignature(
				freshFrame.normalizationSignature
			),
			constructionMode: freshFrame.constructionMode,
			width: freshFrame.width,
			height: freshFrame.height,
			centerX: freshFrame.centerX,
			centerZ: freshFrame.centerZ,
			baseY: freshFrame.baseY
		});
	});

	it.each([
		[
			'grid size',
			(analysis: Analysis): Analysis => ({
				...analysis,
				metadata: { ...analysis.metadata, grid_size: 1 }
			})
		],
		[
			'coordinate system',
			(analysis: Analysis): Analysis => ({
				...analysis,
				metadata: { ...analysis.metadata, coordinate_system: 'xy_ground' }
			})
		],
		[
			'bounds placement',
			(analysis: Analysis): Analysis => ({
				...analysis,
				metadata: {
					...analysis.metadata,
					bounds: {
						...analysis.metadata.bounds!,
						x_min: analysis.metadata.bounds!.x_min + 1,
						x_max: analysis.metadata.bounds!.x_max + 1
					}
				}
			})
		]
	])('does not reuse structural frame cache when %s differs', (_label, mutate) => {
		const base = createAnalysis({
			sourceAnalysisId: `structural-mismatch/${_label}`,
			gridSize: 0.5,
			bounds: { x_min: 0, x_max: 0.5, y_min: 0, y_max: 0.5, z: 0 },
			positionsArray: new Float32Array([0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0.5, 0, 0.5])
		});
		const seededMissDiagnostics: UtciLayoutReuseKeyDiagnostics = {};
		const seededHitDiagnostics: UtciLayoutReuseKeyDiagnostics = {};
		const mismatchDiagnostics: UtciLayoutReuseKeyDiagnostics = {};

		createUtciLayoutReuseKeyForAnalysis({
			analysis: cloneAnalysisWithSelectedHourValues(base, [20, 21, 22, 23]),
			utciSurfaceSource: 'compute-buffer-selected-hour',
			rendererBackend: 'webgpu',
			diagnostics: seededMissDiagnostics
		});

		createUtciLayoutReuseKeyForAnalysis({
			analysis: cloneAnalysisWithSelectedHourValues(base, [24, 25, 26, 27]),
			utciSurfaceSource: 'compute-buffer-selected-hour',
			rendererBackend: 'webgpu',
			diagnostics: seededHitDiagnostics
		});
		createUtciLayoutReuseKeyForAnalysis({
			analysis: mutate(cloneAnalysisWithSelectedHourValues(base, [28, 29, 30, 31])),
			utciSurfaceSource: 'compute-buffer-selected-hour',
			rendererBackend: 'webgpu',
			diagnostics: mismatchDiagnostics
		});

		expect(seededMissDiagnostics.frameCacheHit).toBe(false);
		expect(seededHitDiagnostics.frameCacheHit).toBe(true);
		expect(mismatchDiagnostics.frameCacheHit).toBe(false);
	});

	it('keeps the structural frame cache bounded', () => {
		for (let index = 0; index < 12; index += 1) {
			createUtciLayoutReuseKeyForAnalysis({
				analysis: createAnalysis({
					sourceAnalysisId: `structural-cache/${index}`,
					bounds: { x_min: index, x_max: index + 1, y_min: 0, y_max: 1, z: 0 },
					positionsArray: new Float32Array([
						index,
						0,
						0,
						index + 1,
						0,
						0,
						index,
						0,
						1,
						index + 1,
						0,
						1
					])
				}),
				utciSurfaceSource: 'compute-buffer-selected-hour',
				rendererBackend: 'webgpu'
			});
		}

		const diagnostics = getUtciLayoutFrameCacheDiagnosticsForTest();
		expect(diagnostics.structuralFrameCacheSize).toBeLessThanOrEqual(
			diagnostics.structuralFrameCacheLimit
		);
	});

	it('returns reuse-existing for safe scrub publications with the same stable layout key', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const previousProof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: buildUtciGridLayout(analysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});
		const currentKey = createPublicationReuseKey(analysis);

		expect(
			planUtciLayoutPublication({
				previousLayout,
				previousProof,
				previousKey: currentKey,
				currentKey,
				currentSurfaceSource: 'compute-buffer-selected-hour',
				currentRendererBackend: 'webgpu',
				publicationPhase: 'scrub'
			})
		).toEqual({
			action: 'reuse-existing',
			layout: previousLayout,
			reason: 'reuse-safe',
			keyMatch: true
		});
	});

	it('allows a runtime-compatible existing mesh to refresh stale initial proof for the first scrub', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ness-Tziona/hot-path' });
		const previousLayout = buildUtciGridLayout(analysis);
		const currentKey = createPublicationReuseKey(analysis);
		const staleInitialProof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout: null,
			nextLayout: previousLayout,
			canonicalRuntimeCompatibilityWouldReuse: null
		});
		const refreshedProof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: previousLayout,
			canonicalRuntimeCompatibilityWouldReuse: true,
			canonicalPointCompatibility: {
				compatible: true,
				cellToPointMappingMatch: true,
				requiredExpensiveMappingComparison: false,
				performedExpensiveMappingComparison: false
			}
		});

		expect(isUtciLayoutReuseProofSafe(refreshedProof)).toBe(true);
		expect(
			planUtciLayoutPublication({
				previousLayout,
				previousProof: staleInitialProof,
				previousKey: currentKey,
				currentKey,
				currentSurfaceSource: 'compute-buffer-selected-hour',
				currentRendererBackend: 'webgpu',
				publicationPhase: 'scrub',
				refreshedProof
			})
		).toEqual({
			action: 'reuse-existing',
			layout: previousLayout,
			reason: 'refreshed-proof-safe',
			keyMatch: true
		});
	});

	it('does not reuse from refreshed proof when hover/cell safety is inconclusive', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ness-Tziona/hot-path' });
		const previousLayout = buildUtciGridLayout(analysis);
		const currentKey = createPublicationReuseKey(analysis);
		const refreshedProof = {
			...buildUtciGridLayoutReuseProofDiagnostics({
				previousLayout,
				nextLayout: previousLayout,
				canonicalRuntimeCompatibilityWouldReuse: true,
				canonicalPointCompatibility: {
					compatible: true,
					cellToPointMappingMatch: true,
					requiredExpensiveMappingComparison: false,
					performedExpensiveMappingComparison: false
				}
			}),
			hoverCellLookupProofStatus: 'proof-inconclusive' as const
		};

		expect(isUtciLayoutReuseProofSafe(refreshedProof)).toBe(false);
		expect(
			planUtciLayoutPublication({
				previousLayout,
				previousProof: null,
				previousKey: currentKey,
				currentKey,
				currentSurfaceSource: 'compute-buffer-selected-hour',
				currentRendererBackend: 'webgpu',
				publicationPhase: 'scrub',
				refreshedProof
			})
		).toEqual({ action: 'build-new', reason: 'proof-not-safe', keyMatch: true });
	});

	it('commits pending reuse metadata only after sync completion succeeds', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout: buildUtciGridLayout(analysis),
			nextLayout: buildUtciGridLayout(analysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});
		const previousState = createUtciLayoutReusePublicationState({
			proof,
			key: createPublicationReuseKey(analysis),
			requestId: 7,
			selectionKey: 'previous-selection'
		});
		const pendingState = createUtciLayoutReusePublicationState({
			proof,
			key: createPublicationReuseKey(analysis),
			requestId: 8,
			selectionKey: 'pending-selection'
		});

		expect(
			resolveUtciLayoutReusePublicationStateAfterSync({
				currentState: previousState,
				pendingState,
				syncResult: 'complete'
			})
		).toEqual(pendingState);
	});

	it('preserves the last completed reuse metadata when the next sync fails or is superseded', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const proof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout: buildUtciGridLayout(analysis),
			nextLayout: buildUtciGridLayout(analysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});
		const previousState = createUtciLayoutReusePublicationState({
			proof,
			key: createPublicationReuseKey(analysis),
			requestId: 7,
			selectionKey: 'previous-selection'
		});
		const pendingState = createUtciLayoutReusePublicationState({
			proof,
			key: createPublicationReuseKey(analysis),
			requestId: 9,
			selectionKey: 'failed-selection'
		});

		for (const syncResult of ['failed', 'superseded', 'already-released'] as const) {
			expect(
				resolveUtciLayoutReusePublicationStateAfterSync({
					currentState: previousState,
					pendingState,
					syncResult
				})
			).toEqual(previousState);
		}
	});

	it('returns build-new when previous layout or proof evidence is missing', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const currentKey = createPublicationReuseKey(analysis);

		expect(
			planUtciLayoutPublication({
				previousLayout: null,
				previousProof: null,
				previousKey: null,
				currentKey,
				currentSurfaceSource: 'compute-buffer-selected-hour',
				currentRendererBackend: 'webgpu',
				publicationPhase: 'scrub'
			})
		).toMatchObject({ action: 'build-new', reason: 'missing-previous-layout' });

		expect(
			planUtciLayoutPublication({
				previousLayout,
				previousProof: null,
				previousKey: currentKey,
				currentKey,
				currentSurfaceSource: 'compute-buffer-selected-hour',
				currentRendererBackend: 'webgpu',
				publicationPhase: 'scrub'
			})
		).toMatchObject({ action: 'build-new', reason: 'diagnostics-missing' });
	});

	it('returns build-new when the proof is inconclusive or unsafe', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const currentKey = createPublicationReuseKey(analysis);

		for (const previousProof of [
			{
				...buildUtciGridLayoutReuseProofDiagnostics({
					previousLayout,
					nextLayout: buildUtciGridLayout(analysis),
					canonicalRuntimeCompatibilityWouldReuse: true
				}),
				decision: 'proof-inconclusive' as const
			},
			{
				...buildUtciGridLayoutReuseProofDiagnostics({
					previousLayout,
					nextLayout: buildUtciGridLayout(analysis),
					canonicalRuntimeCompatibilityWouldReuse: true
				}),
				hoverCellLookupProofStatus: 'proof-inconclusive' as const
			}
		]) {
			expect(
				planUtciLayoutPublication({
					previousLayout,
					previousProof,
					previousKey: currentKey,
					currentKey,
					currentSurfaceSource: 'compute-buffer-selected-hour',
					currentRendererBackend: 'webgpu',
					publicationPhase: 'scrub'
				})
			).toMatchObject({ action: 'build-new' });
		}
	});

	it('returns build-new when the stable layout key changes across analysis, grid, normalization, construction, coordinate, or point-count seams', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const previousProof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: buildUtciGridLayout(analysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});
		const currentKey = createPublicationReuseKey(analysis);
		const mismatchedKeys = [
			{ ...currentKey, analysisId: 'Ness-Tziona/base' },
			{ ...currentKey, gridSize: currentKey.gridSize + 0.5 },
			{ ...currentKey, normalizationSignature: `${currentKey.normalizationSignature}|shifted` },
			{ ...currentKey, constructionMode: 'metadata-bounds-fallback' as const },
			{ ...currentKey, coordinateSystem: 'xy_ground' },
			{ ...currentKey, pointCount: currentKey.pointCount + 1 }
		];

		for (const mismatchedKey of mismatchedKeys) {
			expect(
				planUtciLayoutPublication({
					previousLayout,
					previousProof,
					previousKey: mismatchedKey,
					currentKey,
					currentSurfaceSource: 'compute-buffer-selected-hour',
					currentRendererBackend: 'webgpu',
					publicationPhase: 'scrub'
				})
			).toMatchObject({ action: 'build-new', reason: 'layout-key-mismatch' });
		}
	});

	it('returns build-new when backend or selected-hour source changes', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const previousProof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: buildUtciGridLayout(analysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});
		const currentKey = createPublicationReuseKey(analysis);

		expect(
			planUtciLayoutPublication({
				previousLayout,
				previousProof,
				previousKey: currentKey,
				currentKey,
				currentSurfaceSource: 'cpu-uploaded-selected-hour',
				currentRendererBackend: 'webgpu',
				publicationPhase: 'scrub'
			})
		).toMatchObject({ action: 'build-new', reason: 'backend-or-source-mismatch' });

		expect(
			planUtciLayoutPublication({
				previousLayout,
				previousProof,
				previousKey: currentKey,
				currentKey,
				currentSurfaceSource: 'compute-buffer-selected-hour',
				currentRendererBackend: 'webgl',
				publicationPhase: 'scrub'
			})
		).toMatchObject({ action: 'build-new', reason: 'backend-or-source-mismatch' });
	});

	it('returns build-new when layoutSourceSignature changes under the same analysis id', () => {
		const previousAnalysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const currentAnalysis = createAnalysis({
			sourceAnalysisId: 'Ben-Gurion/base',
			positions: [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
		});
		const previousLayout = buildUtciGridLayout(previousAnalysis);
		const previousProof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: buildUtciGridLayout(previousAnalysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});
		const previousKey = createPublicationReuseKey(previousAnalysis);
		const currentKey = createPublicationReuseKey(currentAnalysis);

		expect(previousKey.analysisId).toBe(currentKey.analysisId);
		expect(previousKey.layoutSourceSignature).not.toBe(
			currentKey.layoutSourceSignature
		);
		expect(
			planUtciLayoutPublication({
				previousLayout,
				previousProof,
				previousKey,
				currentKey,
				currentSurfaceSource: 'compute-buffer-selected-hour',
				currentRendererBackend: 'webgpu',
				publicationPhase: 'scrub'
			})
		).toMatchObject({ action: 'build-new', reason: 'layout-key-mismatch' });
	});

	it('returns build-new when a reloaded source changes signature under the same analysis id', () => {
		const previousAnalysis = createAnalysis({
			sourceAnalysisId: 'Ben-Gurion/base',
			source: 'loaded'
		});
		const reloadedAnalysis = createAnalysis({
			sourceAnalysisId: 'Ben-Gurion/base',
			source: 'webgpu'
		});
		const previousLayout = buildUtciGridLayout(previousAnalysis);
		const previousProof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: buildUtciGridLayout(previousAnalysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});
		const previousKey = createPublicationReuseKey(previousAnalysis);
		const currentKey = createPublicationReuseKey(reloadedAnalysis);

		expect(previousKey.analysisId).toBe(currentKey.analysisId);
		expect(previousKey.layoutSourceSignature).not.toBe(
			currentKey.layoutSourceSignature
		);
		expect(
			planUtciLayoutPublication({
				previousLayout,
				previousProof,
				previousKey,
				currentKey,
				currentSurfaceSource: 'compute-buffer-selected-hour',
				currentRendererBackend: 'webgpu',
				publicationPhase: 'scrub'
			})
		).toMatchObject({ action: 'build-new', reason: 'layout-key-mismatch' });
	});

	it('returns build-new for initial publications even when a reusable layout exists', () => {
		const analysis = createAnalysis({ sourceAnalysisId: 'Ben-Gurion/base' });
		const previousLayout = buildUtciGridLayout(analysis);
		const previousProof = buildUtciGridLayoutReuseProofDiagnostics({
			previousLayout,
			nextLayout: buildUtciGridLayout(analysis),
			canonicalRuntimeCompatibilityWouldReuse: true
		});
		const currentKey = createPublicationReuseKey(analysis);

		expect(
			planUtciLayoutPublication({
				previousLayout,
				previousProof,
				previousKey: currentKey,
				currentKey,
				currentSurfaceSource: 'compute-buffer-selected-hour',
				currentRendererBackend: 'webgpu',
				publicationPhase: 'initial'
			})
		).toMatchObject({ action: 'build-new', reason: 'initial-publication' });
	});

	it('returns an observable recreate-required signal from updateUtciSurfaceMesh', () => {
		const analysis = createAnalysis();
		const mesh = createUtciSurfaceMesh(analysis);

		expect(
			updateUtciSurfaceMesh(mesh, {
				analysis,
				backend: 'gpuNative'
			})
		).toBe(false);

		expect(
			updateUtciSurfaceMesh(mesh, {
				analysis
			})
		).toBe(true);
	});

	it('labels gpuNative selected-hour surfaces as cpu-uploaded without a texture map', () => {
		const analysis = createAnalysis();
		const mesh = createUtciSurfaceMesh({
			analysis,
			backend: 'gpuNative'
		});
		const material = mesh.material as THREE.Material & { map?: THREE.Texture | null };
		const layout = mesh.userData.utciLayout;
		const expectedCpuSurfaceVertexCount = layout.width * layout.height * 6;

		expect(mesh.userData.utciSurfaceBackend).toBe('gpuNative');
		expect(mesh.userData.utciSurfaceSource).toBe('cpu-uploaded-selected-hour');
		expect(getGpuNativeUtciSurfaceSource(mesh)).toBe('cpu-uploaded-selected-hour');
		expect(getGpuNativeUtciSurfaceSource(mesh)).not.toBe('compute-buffer-selected-hour');
		expect(material.map ?? null).toBeNull();
		expect(mesh.geometry.getAttribute('position').count).toBe(expectedCpuSurfaceVertexCount);
		expect(mesh.userData.gpuNativeUtciSurfaceState.vertexCount).toBe(
			expectedCpuSurfaceVertexCount
		);
	});

	it('keeps CPU-uploaded gpuNative surfaces front-sided and normally frustum culled', () => {
		const analysis = createAnalysis();
		const mesh = createUtciSurfaceMesh({
			analysis,
			backend: 'gpuNative'
		});
		const material = mesh.material as THREE.Material;

		expect(material.side).toBe(THREE.FrontSide);
		expect(mesh.frustumCulled).toBe(true);
	});

	it('maps each surface vertex back to the shuffled source point index', () => {
		const cellToPoint = createCellToPointIndexArray({
			width: 2,
			height: 2,
			gridSize: 1,
			numPositions: 4,
			centerX: 1,
			centerZ: 1,
			minX: 0,
			minZ: 0,
			baseY: 0,
			coordinateSystem: 'xy_ground',
			minY: 0,
			maxY: 0,
			indexToRow: new Uint32Array([1, 0, 1, 0]),
			indexToColumn: new Uint32Array([1, 0, 0, 1]),
			indexToTexel: new Uint32Array([0, 1, 2, 3]),
			colorBuffer: new Uint8Array(2 * 2 * 4)
		});
		const vertexToPoint = createVertexToPointIndexArray({
			width: 2,
			height: 2,
			gridSize: 1,
			numPositions: 4,
			centerX: 0.5,
			centerZ: 0.5,
			minX: 0,
			minZ: 0,
			baseY: 0,
			coordinateSystem: 'xy_ground' as const,
			minY: 0,
			maxY: 0,
			indexToRow: new Uint32Array([1, 0, 1, 0]),
			indexToColumn: new Uint32Array([1, 0, 0, 1]),
			indexToTexel: new Uint32Array([1, 2, 0, 3]),
			colorBuffer: new Uint8Array(2 * 2 * 4)
		});

		expect(Array.from(cellToPoint)).toEqual([1, 3, 2, 0]);
		expect(Array.from(vertexToPoint)).toEqual([
			1, 1, 1, 1, 1, 1,
			3, 3, 3, 3, 3, 3,
			2, 2, 2, 2, 2, 2,
			0, 0, 0, 0, 0, 0
		]);
	});

	it('creates compute-buffer surfaces without uploading selected-hour UTCI from CPU readback', () => {
		const computeBuffer = {} as GPUBuffer;
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout: {
				width: 1,
				height: 1,
				gridSize: 1,
				numPositions: 1,
				centerX: 0.5,
				centerZ: 0.5,
				minX: 0,
				minZ: 0,
				baseY: 0,
				coordinateSystem: 'xy_ground' as const,
				minY: 0,
				maxY: 0,
				indexToRow: new Uint32Array([0]),
				indexToColumn: new Uint32Array([0]),
				indexToTexel: new Uint32Array([0]),
				colorBuffer: new Uint8Array(4)
			},
			utciBuffer: computeBuffer,
			utciRange: { min: 10, max: 40 }
		});
		const material = mesh.material as THREE.Material;

		expect(getGpuNativeUtciSurfaceSource(mesh)).toBe('compute-buffer-selected-hour');
		expect(material.side).toBe(THREE.FrontSide);
		expect(mesh.frustumCulled).toBe(true);
		expect(mesh.userData.utciSurfaceSource).toBeUndefined();
		expect(mesh.userData.pendingComputeBufferUtciSource).toBe(computeBuffer);
		expect(Array.from(mesh.userData.gpuNativeUtciSurfaceState.utciStorageAttribute.array)).toEqual([0]);
		expect(
			Array.from(mesh.userData.gpuNativeUtciSurfaceState.cellToPointStorageAttribute.array)
		).toEqual([0]);
		expect(mesh.userData.gpuNativeUtciSurfaceState.utciRange).toEqual({ min: 10, max: 40 });
		expect(mesh.userData.renderOwnedSelectedHourBytes).toBe(1104);
	});

	it('uses indexed shared-grid geometry for compute-buffer surfaces', () => {
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout: {
				width: 2,
				height: 1,
				gridSize: 1,
				numPositions: 2,
				centerX: 1,
				centerZ: 0.5,
				minX: 0,
				minZ: 0,
				baseY: 0,
				coordinateSystem: 'xy_ground' as const,
				minY: 0,
				maxY: 0,
				indexToRow: new Uint32Array([0, 0]),
				indexToColumn: new Uint32Array([0, 1]),
				indexToTexel: new Uint32Array([0, 1]),
				colorBuffer: new Uint8Array(8)
			},
			utciBuffer: {} as GPUBuffer,
			utciRange: { min: 10, max: 40 }
		});

		expect(mesh.geometry.index?.array).toBeInstanceOf(Uint32Array);
		expect(mesh.geometry.index?.count).toBe(12);
		expect(mesh.geometry.getAttribute('position').count).toBe(6);
		expect(mesh.userData.gpuNativeUtciSurfaceState.vertexCount).toBe(6);
		expect(mesh.userData.renderOwnedSelectedHourBytes).toBe(1160);
	});

	it('updates compute-buffer surfaces by storing pending GPU source and refreshing uniforms only', () => {
		const layout = {
			width: 2,
			height: 1,
			gridSize: 1,
			numPositions: 2,
			centerX: 1,
			centerZ: 0.5,
			minX: 0,
			minZ: 0,
			baseY: 0,
			coordinateSystem: 'xy_ground' as const,
			minY: 0,
			maxY: 0,
			indexToRow: new Uint32Array([0, 0]),
			indexToColumn: new Uint32Array([0, 1]),
			indexToTexel: new Uint32Array([0, 1]),
			colorBuffer: new Uint8Array(2 * 4)
		};
		const initialBuffer = {} as GPUBuffer;
		const nextBuffer = {} as GPUBuffer;
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout,
			utciBuffer: initialBuffer,
			utciRange: { min: 10, max: 40 }
		});

		expect(
			updateComputeBufferUtciSurfaceMesh(mesh, {
				layout,
				utciBuffer: nextBuffer,
				utciRange: { min: 5, max: 55 }
			})
		).toBe(true);

		expect(mesh.userData.utciSurfaceSource).toBeUndefined();
		expect(mesh.userData.pendingComputeBufferUtciSource).toBe(nextBuffer);
		expect(Array.from(mesh.userData.gpuNativeUtciSurfaceState.utciStorageAttribute.array)).toEqual([
			0, 0
		]);
		expect(mesh.userData.gpuNativeUtciSurfaceState.utciRange).toEqual({ min: 5, max: 55 });
		expect(mesh.userData.gpuNativeUtciSurfaceState.minUniform.value).toBe(5);
		expect(mesh.userData.gpuNativeUtciSurfaceState.maxUniform.value).toBe(55);
		expect(mesh.userData.renderOwnedSelectedHourBytes).toBe(1160);
	});

	it('expects compute-buffer compatibility to use shared-grid position vertices', () => {
		const layout = {
			width: 2,
			height: 1,
			gridSize: 1,
			numPositions: 2,
			centerX: 1,
			centerZ: 0.5,
			minX: 0,
			minZ: 0,
			baseY: 0,
			coordinateSystem: 'xy_ground' as const,
			minY: 0,
			maxY: 0,
			indexToRow: new Uint32Array([0, 0]),
			indexToColumn: new Uint32Array([0, 1]),
			cellToPointIndex: new Int32Array([0, 1]),
			indexToTexel: new Uint32Array([0, 1]),
			colorBuffer: new Uint8Array(2 * 4)
		};
		const sharedGridVertexCount = (layout.width + 1) * (layout.height + 1);
		const nonIndexedVertexCount = layout.width * layout.height * 6;

		expect(
			evaluateComputeBufferUtciSurfaceLayoutCompatibility({
				state: {
					source: 'compute-buffer-selected-hour',
					width: layout.width,
					height: layout.height,
					gridSize: layout.gridSize,
					vertexCount: sharedGridVertexCount,
					storageCount: layout.numPositions
				},
				previousLayout: layout,
				nextLayout: layout,
				allowExpensiveMappingComparison: true
			})
		).toMatchObject({
			compatible: true,
			vertexCountMatch: true
		});
		expect(
			evaluateComputeBufferUtciSurfaceLayoutCompatibility({
				state: {
					source: 'compute-buffer-selected-hour',
					width: layout.width,
					height: layout.height,
					gridSize: layout.gridSize,
					vertexCount: nonIndexedVertexCount,
					storageCount: layout.numPositions
				},
				previousLayout: layout,
				nextLayout: layout,
				allowExpensiveMappingComparison: true
			})
		).toMatchObject({
			compatible: false,
			vertexCountMatch: false
		});
	});

	it('respects a precomputed incompatible update result without recomputing the live predicate', () => {
		const layout = {
			width: 2,
			height: 1,
			gridSize: 1,
			numPositions: 2,
			centerX: 1,
			centerZ: 0.5,
			minX: 0,
			minZ: 0,
			baseY: 0,
			coordinateSystem: 'xy_ground' as const,
			minY: 0,
			maxY: 0,
			indexToRow: new Uint32Array([0, 0]),
			indexToColumn: new Uint32Array([0, 1]),
			cellToPointIndex: new Int32Array([0, 1]),
			indexToTexel: new Uint32Array([0, 1]),
			colorBuffer: new Uint8Array(2 * 4)
		};
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout,
			utciBuffer: {} as GPUBuffer,
			utciRange: { min: 10, max: 40 }
		});
		const compatibilityEvaluation = evaluateComputeBufferUtciSurfaceLayoutCompatibility({
			state: {
				source: 'compute-buffer-selected-hour',
				width: layout.width,
				height: layout.height + 1,
				gridSize: layout.gridSize,
				vertexCount: (layout.width + 1) * (layout.height + 1),
				storageCount: layout.numPositions
			},
			previousLayout: layout,
			nextLayout: layout,
			allowExpensiveMappingComparison: true
		});

		expect(
			updateComputeBufferUtciSurfaceMesh(mesh, {
				layout,
				utciBuffer: {} as GPUBuffer,
				utciRange: { min: 5, max: 55 },
				compatibilityEvaluation
			})
		).toBe(false);
		expect(mesh.userData.gpuNativeUtciSurfaceState.utciRange).toEqual({ min: 10, max: 40 });
	});

	it('reports compute-buffer layout compatibility without mutating surface state', () => {
		const layout = {
			width: 2,
			height: 1,
			gridSize: 1,
			numPositions: 2,
			centerX: 1,
			centerZ: 0.5,
			minX: 0,
			minZ: 0,
			baseY: 0,
			coordinateSystem: 'xy_ground' as const,
			minY: 0,
			maxY: 0,
			indexToRow: new Uint32Array([0, 0]),
			indexToColumn: new Uint32Array([0, 1]),
			cellToPointIndex: new Int32Array([0, 1]),
			indexToTexel: new Uint32Array([0, 1]),
			colorBuffer: new Uint8Array(2 * 4)
		};
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout,
			utciBuffer: {} as GPUBuffer,
			utciRange: { min: 10, max: 40 }
		});
		const originalPendingSource = mesh.userData.pendingComputeBufferUtciSource;
		const originalRange = { ...mesh.userData.gpuNativeUtciSurfaceState.utciRange };

		expect(isComputeBufferUtciSurfaceLayoutCompatible(mesh, layout)).toBe(true);
		expect(
			isComputeBufferUtciSurfaceLayoutCompatible(mesh, {
				...layout,
				width: 1
			})
		).toBe(false);
		expect(
			isComputeBufferUtciSurfaceLayoutCompatible(mesh, {
				...layout,
				numPositions: 1,
				indexToRow: new Uint32Array([0]),
				indexToColumn: new Uint32Array([0]),
				indexToTexel: new Uint32Array([0]),
				colorBuffer: new Uint8Array(4)
			})
		).toBe(false);
		expect(
			isComputeBufferUtciSurfaceLayoutCompatible(mesh, {
				...layout,
				cellToPointIndex: new Int32Array([1, 0])
			})
		).toBe(false);
		expect(isComputeBufferUtciSurfaceLayoutCompatible(null, layout)).toBe(false);
		expect(mesh.userData.pendingComputeBufferUtciSource).toBe(originalPendingSource);
		expect(mesh.userData.gpuNativeUtciSurfaceState.utciRange).toEqual(originalRange);
	});

	it('rejects compute-buffer layout reuse when ambiguous cells rebuild to a different effective mapping', () => {
		const layout = {
			width: 1,
			height: 1,
			gridSize: 1,
			numPositions: 2,
			centerX: 0.5,
			centerZ: 0.5,
			minX: 0,
			minZ: 0,
			baseY: 0,
			coordinateSystem: 'xy_ground' as const,
			minY: 0,
			maxY: 0,
			indexToRow: new Uint32Array([0, 0]),
			indexToColumn: new Uint32Array([0, 0]),
			cellToPointIndex: new Int32Array([-2]),
			indexToTexel: new Uint32Array([0, 0]),
			colorBuffer: new Uint8Array(4)
		};
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout,
			utciBuffer: {} as GPUBuffer,
			utciRange: { min: 10, max: 40 }
		});

		expect(isComputeBufferUtciSurfaceLayoutCompatible(mesh, layout)).toBe(true);
		expect(
			isComputeBufferUtciSurfaceLayoutCompatible(mesh, {
				...layout,
				indexToRow: new Uint32Array([0, 0]),
				indexToColumn: new Uint32Array([0, 0])
			})
		).toBe(true);
		expect(
			isComputeBufferUtciSurfaceLayoutCompatible(mesh, {
				...layout,
				indexToRow: new Uint32Array([0, 0]),
				indexToColumn: new Uint32Array([0, 1])
			})
		).toBe(false);
	});

	it('refuses generic gpuNative CPU updates on compute-buffer backed meshes', () => {
		const analysis = createAnalysis();
		const cpuUploadedMesh = createUtciSurfaceMesh({
			analysis,
			backend: 'gpuNative'
		});
		const layout = cpuUploadedMesh.userData.utciLayout;
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout,
			utciBuffer: {} as GPUBuffer,
			utciRange: { min: 10, max: 40 }
		});
		mesh.userData.utciLayout = layout;
		mesh.userData.utciSurfaceBackend = 'gpuNative';

		expect(getGpuNativeUtciSurfaceSource(mesh)).toBe('compute-buffer-selected-hour');

		expect(
			updateUtciSurfaceMesh(mesh, {
				analysis,
				backend: 'gpuNative'
			})
		).toBe(false);

		expect(getGpuNativeUtciSurfaceSource(mesh)).toBe('compute-buffer-selected-hour');
		expect(mesh.userData.utciSurfaceSource).toBeUndefined();
	});

	it('rejects compute-buffer surface updates when the layout changes', () => {
		const mesh = createComputeBufferUtciSurfaceMesh({
			layout: {
				width: 1,
				height: 1,
				gridSize: 1,
				numPositions: 1,
				centerX: 0.5,
				centerZ: 0.5,
				minX: 0,
				minZ: 0,
				baseY: 0,
				coordinateSystem: 'xy_ground' as const,
				minY: 0,
				maxY: 0,
				indexToRow: new Uint32Array([0]),
				indexToColumn: new Uint32Array([0]),
				indexToTexel: new Uint32Array([0]),
				colorBuffer: new Uint8Array(4)
			},
			utciBuffer: {} as GPUBuffer,
			utciRange: { min: 10, max: 40 }
		});

		expect(
			updateComputeBufferUtciSurfaceMesh(mesh, {
				layout: {
					width: 2,
					height: 1,
					gridSize: 1,
					numPositions: 2,
					centerX: 1,
					centerZ: 0.5,
					minX: 0,
					minZ: 0,
					baseY: 0,
					coordinateSystem: 'xy_ground' as const,
					minY: 0,
					maxY: 0,
					indexToRow: new Uint32Array([0, 0]),
					indexToColumn: new Uint32Array([0, 1]),
					indexToTexel: new Uint32Array([0, 1]),
					colorBuffer: new Uint8Array(8)
				},
				utciBuffer: {} as GPUBuffer,
				utciRange: { min: 5, max: 55 }
			})
		).toBe(false);
	});

	it('stores real per-backend surface diagnostics on mesh userData', () => {
		const analysis = createAnalysis();
		const dataTextureMesh = createUtciSurfaceMesh({ analysis, backend: 'dataTexture' });

		expect(dataTextureMesh.userData.dataTextureBuildCount).toBe(1);
		expect(dataTextureMesh.userData.selectedHourTransferCount).toBeUndefined();
		expect(dataTextureMesh.userData.utciSurfaceSource).toBeUndefined();

		expect(
			updateUtciSurfaceMesh(dataTextureMesh, {
				analysis,
				backend: 'dataTexture'
			})
		).toBe(true);
		expect(dataTextureMesh.userData.dataTextureBuildCount).toBe(2);
		expect(dataTextureMesh.userData.selectedHourTransferCount).toBeUndefined();

		const gpuNativeMesh = createUtciSurfaceMesh({
			analysis,
			backend: 'gpuNative'
		});

		expect(gpuNativeMesh.userData.dataTextureBuildCount).toBe(0);
		expect(gpuNativeMesh.userData.selectedHourTransferCount).toBe(1);
		expect(
			updateUtciSurfaceMesh(gpuNativeMesh, {
				analysis,
				backend: 'gpuNative'
			})
		).toBe(true);
		expect(gpuNativeMesh.userData.dataTextureBuildCount).toBe(0);
		expect(gpuNativeMesh.userData.selectedHourTransferCount).toBe(2);
	});
});
