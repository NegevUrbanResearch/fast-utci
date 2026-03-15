import { expect, test } from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const DEFAULT_BASE_PATH = 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday';
const PARITY_BASE_PATH = process.env.PARITY_BASE_PATH || DEFAULT_BASE_PATH;
const basePath = resolve(REPO_ROOT, PARITY_BASE_PATH);
const COLLECT_WAIT_MS = 180_000;

type FlipCell = {
	flatIndex: number;
	pointIndex: number;
	hourIndex: number;
	refSolar: number;
	webgpuSolar: number;
};

test.describe('Diagnose solar flip rays with CPU oracle', () => {
	test('raycast top flip cells against loaded model', async ({ page }) => {
		test.setTimeout(240_000);
		const flipPath = `${basePath}_solar_flip_diagnostics.json`;
		const flipJson = JSON.parse(readFileSync(flipPath, 'utf8')) as {
			topAffectedCells?: FlipCell[];
		};
		const topCells = (flipJson.topAffectedCells ?? []).slice(0, 10);
		expect(topCells.length, `No flip cells found in ${flipPath}`).toBeGreaterThan(0);

		const analysisSlug = PARITY_BASE_PATH.replace(/^data[/\\]analyses[/\\]/, '').replace(/\\/g, '/');
		await page.goto(`/debug-webgpu-utci?analysis=${encodeURIComponent(analysisSlug)}`);

		await page.waitForFunction(
			() => {
				const w = window as unknown as {
					__parityCollectionStatus__?: { state: 'running' | 'success' | 'error' | 'timeout' };
					__parityResults__?: unknown;
					__parityIntermediates__?: unknown;
					__parityCollectionError__?: string;
					__parityIntermediatesError__?: string;
				};
				if (w.__parityCollectionError__ || w.__parityIntermediatesError__) return true;
				if (w.__parityCollectionStatus__?.state === 'error' || w.__parityCollectionStatus__?.state === 'timeout') return true;
				if (w.__parityCollectionStatus__?.state === 'success') {
					return w.__parityResults__ != null && w.__parityIntermediates__ != null;
				}
				return false;
			},
			{ timeout: COLLECT_WAIT_MS, polling: 1000 }
		);

		const oracleRows = await page.evaluate((cells) => {
			const w = window as unknown as {
				__parityModel__?: {
					matrixWorld?: { elements: number[] };
				};
				__parityThree__?: {
					Vector3: new (x?: number, y?: number, z?: number) => {
						x: number;
						y: number;
						z: number;
						length: () => number;
						normalize: () => unknown;
					};
					Raycaster: new (origin: unknown, direction: unknown, near?: number, far?: number) => {
						intersectObject: (obj: unknown, recursive?: boolean) => Array<{
							distance: number;
							point: { x: number; y: number; z: number };
						}>;
					};
				};
				__parityMetadata__?: {
					sun_positions?: Array<{ vector?: [number, number, number] }>;
				};
				__parityResults__?: {
					positions?: number[];
					computeGridPointsWorld?: number[] | null;
				};
			};
			const model = w.__parityModel__;
			const THREE = w.__parityThree__;
			const metadata = w.__parityMetadata__;
			const positions = w.__parityResults__?.positions;
			const computeGridPointsWorld = w.__parityResults__?.computeGridPointsWorld ?? null;
			if (!model || !THREE || !metadata?.sun_positions || !positions) {
				throw new Error('Missing parity model/THREE/metadata/positions for ray oracle.');
			}
			const touchedMaterials: Array<{ material: { side: number }; prevSide: number }> = [];
			const touchedMaterialArrays: Array<{ materials: Array<{ side: number }>; prevSides: number[] }> = [];
			(model as unknown as {
				traverse: (cb: (obj: unknown) => void) => void;
			}).traverse((obj) => {
				const mesh = obj as { isMesh?: boolean; material?: unknown };
				if (!mesh.isMesh || !mesh.material) return;
				if (Array.isArray(mesh.material)) {
					const mats = mesh.material as Array<{ side: number }>;
					touchedMaterialArrays.push({ materials: mats, prevSides: mats.map((m) => m.side) });
					for (const m of mats) m.side = (THREE as unknown as { DoubleSide: number }).DoubleSide;
					return;
				}
				const mat = mesh.material as { side: number };
				touchedMaterials.push({ material: mat, prevSide: mat.side });
				mat.side = (THREE as unknown as { DoubleSide: number }).DoubleSide;
			});
			const box = new (THREE as unknown as { Box3: new () => { setFromObject: (obj: unknown) => unknown; containsPoint: (v: unknown) => boolean; min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } } }).Box3();
			box.setFromObject(model);
			const tmpA = new THREE.Vector3();
			const tmpB = new THREE.Vector3();
			const tmpC = new THREE.Vector3();
			const tmpHit = new THREE.Vector3();
			const tmpBox = new (THREE as unknown as { Box3: new () => { copy: (b: unknown) => unknown; applyMatrix4: (m: unknown) => unknown } }).Box3();
			const rayCtor = (THREE as unknown as { Ray: new (origin: unknown, direction: unknown) => { intersectTriangle: (a: unknown, b: unknown, c: unknown, backfaceCulling: boolean, target: { x: number; y: number; z: number }) => unknown } }).Ray;

			function bruteForceOcclusion(origin: { x: number; y: number; z: number }, direction: { x: number; y: number; z: number }): { hit: boolean; distance: number | null } {
				const ray = new rayCtor(origin, direction);
				let bestDist = Number.POSITIVE_INFINITY;
				(model as unknown as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
					const mesh = obj as {
						isMesh?: boolean;
						geometry?: {
							getAttribute: (name: string) => { array: ArrayLike<number>; count: number; getX?: (i: number) => number; getY?: (i: number) => number; getZ?: (i: number) => number } | undefined;
							getIndex: () => { array: ArrayLike<number>; count: number } | null;
							boundingBox?: unknown;
							computeBoundingBox?: () => void;
						};
						matrixWorld?: unknown;
					};
					if (!mesh.isMesh || !mesh.geometry || !mesh.matrixWorld) return;
					const position = mesh.geometry.getAttribute('position');
					if (!position) return;
					if (!mesh.geometry.boundingBox && mesh.geometry.computeBoundingBox) {
						mesh.geometry.computeBoundingBox();
					}
					if (mesh.geometry.boundingBox) {
						tmpBox.copy(mesh.geometry.boundingBox);
						tmpBox.applyMatrix4(mesh.matrixWorld);
						const rc = new THREE.Raycaster(origin as unknown as never, direction as unknown as never, 0, 1_000_000);
						if (rc.ray.intersectsBox(tmpBox as unknown as never) === false) return;
					}
					const index = mesh.geometry.getIndex();
					const readPos = (vertexIndex: number, out: { set: (x: number, y: number, z: number) => unknown; applyMatrix4: (m: unknown) => unknown }) => {
						const base = vertexIndex * 3;
						out.set(
							position.array[base] ?? 0,
							position.array[base + 1] ?? 0,
							position.array[base + 2] ?? 0
						);
						out.applyMatrix4(mesh.matrixWorld);
					};
					const triCount = index ? Math.floor(index.count / 3) : Math.floor(position.count / 3);
					for (let tri = 0; tri < triCount; tri++) {
						const i0 = index ? Number(index.array[tri * 3]) : tri * 3;
						const i1 = index ? Number(index.array[tri * 3 + 1]) : tri * 3 + 1;
						const i2 = index ? Number(index.array[tri * 3 + 2]) : tri * 3 + 2;
						readPos(i0, tmpA);
						readPos(i1, tmpB);
						readPos(i2, tmpC);
						const isect = ray.intersectTriangle(tmpA as unknown as never, tmpB as unknown as never, tmpC as unknown as never, false, tmpHit as unknown as never);
						if (!isect) continue;
						const dx = (tmpHit.x ?? 0) - origin.x;
						const dy = (tmpHit.y ?? 0) - origin.y;
						const dz = (tmpHit.z ?? 0) - origin.z;
						const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
						if (Number.isFinite(dist) && dist < bestDist) {
							bestDist = dist;
						}
					}
				});
				return { hit: Number.isFinite(bestDist), distance: Number.isFinite(bestDist) ? bestDist : null };
			}

			const rows = cells.map((cell) => {
				const pointOffset = cell.pointIndex * 3;
				const origin = computeGridPointsWorld && computeGridPointsWorld.length >= pointOffset + 3
					? new THREE.Vector3(
							computeGridPointsWorld[pointOffset] ?? 0,
							computeGridPointsWorld[pointOffset + 1] ?? 0,
							computeGridPointsWorld[pointOffset + 2] ?? 0
					  )
					: new THREE.Vector3(
							positions[pointOffset] ?? 0,
							positions[pointOffset + 1] ?? 0,
							positions[pointOffset + 2] ?? 0
					  );
				const vec = metadata.sun_positions?.[cell.hourIndex]?.vector ?? [0, 0, 0];
				const variants = {
					x_z_negY: [vec[0] ?? 0, vec[2] ?? 0, -(vec[1] ?? 0)],
					x_z_posY: [vec[0] ?? 0, vec[2] ?? 0, vec[1] ?? 0],
					xyz_raw: [vec[0] ?? 0, vec[1] ?? 0, vec[2] ?? 0],
					x_negY_z: [vec[0] ?? 0, -(vec[1] ?? 0), vec[2] ?? 0]
				} as const;
				const primaryDir = new THREE.Vector3(...variants.x_z_negY);
				const dirLen = primaryDir.length();
				if (!Number.isFinite(dirLen) || dirLen <= 1e-10) {
					return {
						...cell,
						cpuHit: false,
						cpuFirstDistance: null,
						cpuFirstPoint: null,
						note: 'sun vector is zero'
					};
				}
				primaryDir.normalize();
				const raycaster = new THREE.Raycaster(origin, primaryDir, 0, 1_000_000);
				const hits = raycaster.intersectObject(model, true);
				const first = hits[0];
				const cpuHitByVariant: Record<string, boolean> = {};
				const bruteByVariant: Record<string, { hit: boolean; distance: number | null }> = {};
				for (const [name, raw] of Object.entries(variants)) {
					const v = new THREE.Vector3(raw[0], raw[1], raw[2]);
					if (v.length() <= 1e-10) {
						cpuHitByVariant[name] = false;
						bruteByVariant[name] = { hit: false, distance: null };
						continue;
					}
					v.normalize();
					const rc = new THREE.Raycaster(origin, v, 0, 1_000_000);
					cpuHitByVariant[name] = rc.intersectObject(model, true).length > 0;
					bruteByVariant[name] = bruteForceOcclusion(origin, v);
				}
				return {
					...cell,
					modelBounds: {
						min: { x: box.min.x, y: box.min.y, z: box.min.z },
						max: { x: box.max.x, y: box.max.y, z: box.max.z }
					},
					pointInsideModelBounds: box.containsPoint(origin),
					cpuHit: hits.length > 0,
					cpuFirstDistance: first?.distance ?? null,
					cpuFirstPoint: first?.point
						? { x: first.point.x, y: first.point.y, z: first.point.z }
						: null,
					cpuHitByVariant,
					cpuBruteByVariant: bruteByVariant
				};
			});
			for (const entry of touchedMaterials) {
				entry.material.side = entry.prevSide;
			}
			for (const entry of touchedMaterialArrays) {
				for (let i = 0; i < entry.materials.length; i++) {
					entry.materials[i].side = entry.prevSides[i];
				}
			}
			return rows;
		}, topCells);

		const outPath = `${basePath}_solar_ray_oracle.json`;
		writeFileSync(
			outPath,
			JSON.stringify(
				{
					basePath,
					count: oracleRows.length,
					rows: oracleRows
				},
				null,
				2
			),
			'utf8'
		);

		console.log(`ray oracle rows: ${oracleRows.length}`);
		console.log(`report written: ${outPath}`);
	});
});
