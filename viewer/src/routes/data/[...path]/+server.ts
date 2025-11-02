import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { RequestHandler } from './$types';

/**
 * Proxy route handler to serve data files from outside the project directory
 * Handles requests to /data/* and proxies them to ../data/* in the file system
 */
export const GET: RequestHandler = async ({ params }) => {
	try {
		const pathParts = params.path?.split('/') || [];
		const filename = pathParts[pathParts.length - 1];
		
		if (!filename) {
			return new Response('Invalid path', { status: 400 });
		}
		
		// Build path to data folder (outside viewer directory)
		// process.cwd() is the viewer directory, so go up one level to access ../data
		const dataPath = join(process.cwd(), '..', 'data', ...pathParts);
		
		// Read file
		const file = await readFile(dataPath);
		
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

