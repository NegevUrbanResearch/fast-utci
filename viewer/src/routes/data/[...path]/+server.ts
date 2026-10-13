import { readFile, realpath } from 'node:fs/promises';
import { isAbsolute, relative, resolve, sep, win32 } from 'node:path';
import type { RequestHandler } from './$types';

function isContainedFile(root: string, candidate: string): boolean {
	const path = relative(root, candidate);
	return path !== '' && path !== '..' && !path.startsWith(`..${sep}`) && !isAbsolute(path);
}

/** Serve local data files from ../data relative to the viewer working directory. */
export const GET: RequestHandler = async ({ params }) => {
	try {
		// Svelte already decodes route params. Preserve literal percent-encoded filenames.
		const requestedPath = params.path?.replace(/\\/g, '/') || '';
		const pathParts = requestedPath.split('/');
		const filename = pathParts[pathParts.length - 1];
		if (!filename || requestedPath.includes('\0') || isAbsolute(requestedPath) ||
			win32.isAbsolute(requestedPath) || /^[a-z]:/i.test(requestedPath) ||
			pathParts.includes('..')) {
			return new Response('Invalid path', { status: 400 });
		}

		const dataRoot = resolve(process.cwd(), '..', 'data');
		const dataPath = resolve(dataRoot, requestedPath);
		if (!isContainedFile(dataRoot, dataPath)) {
			return new Response('Invalid path', { status: 400 });
		}
		const realRoot = await realpath(dataRoot);
		const realFile = await realpath(dataPath);
		if (!isContainedFile(realRoot, realFile)) {
			return new Response('Invalid path', { status: 400 });
		}
		const file = await readFile(realFile);
		
		// Determine content type based on file extension
		const extension = filename.split('.').pop()?.toLowerCase();
		let contentType = 'application/octet-stream';
		
		if (extension === 'json') {
			contentType = 'application/json';
		} else if (extension === 'bin') {
			contentType = 'application/octet-stream';
		} else if (extension === 'glb') {
			contentType = 'model/gltf-binary';
		} else if (extension === 'gltf') {
			contentType = 'model/gltf+json';
		}
		
		return new Response(file, {
			headers: {
				'Content-Type': contentType,
				'Access-Control-Allow-Origin': '*'
			}
		});
	} catch (error) {
		console.error('[ERROR] Failed to serve data file:', error);
		return new Response('File not found', { status: 404 });
	}
};

