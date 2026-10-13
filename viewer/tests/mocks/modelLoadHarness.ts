import type { ThrelteGltf } from '@threlte/extras';
export const modelLoadHandlers: Array<{ url: string; onload: (gltf: ThrelteGltf) => void }> = [];
