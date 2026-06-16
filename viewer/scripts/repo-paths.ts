import { basename, isAbsolute, resolve } from 'node:path';

export function resolveRepoRoot(cwd: string): string {
	return basename(cwd).toLowerCase() === 'viewer' ? resolve(cwd, '..') : cwd;
}

export function resolveRepoRelativePath(cwd: string, inputPath: string): string {
	if (isAbsolute(inputPath)) return inputPath;

	const repoRoot = resolveRepoRoot(cwd);
	if (/^data(?:[\\/]|$)/i.test(inputPath)) {
		return resolve(repoRoot, inputPath);
	}

	if (basename(cwd).toLowerCase() === 'viewer' && /^\.\.[\\/]+data(?:[\\/]|$)/i.test(inputPath)) {
		return resolve(repoRoot, inputPath.replace(/^\.\.[\\/]+/i, ''));
	}

	return resolve(cwd, inputPath);
}
