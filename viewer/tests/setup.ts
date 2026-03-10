import '@testing-library/jest-dom';
import { expect, afterEach } from 'vitest';
import { cleanup } from '@testing-library/svelte/svelte5';

// Cleanup after each test
afterEach(() => {
	cleanup();
});


