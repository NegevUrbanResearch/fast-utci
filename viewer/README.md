# UTCI Analysis Viewer

Interactive 3D viewer for UTCI (Universal Thermal Climate Index) analysis results, built with SvelteKit and Threlte (Three.js).

## Features

- **3D Model Visualization**: Load and explore 3D GLB models with layer-based visibility controls
- **UTCI Point Cloud**: Visualize thermal comfort data as interactive point clouds
- **Time-based Analysis**: Scrub through hourly UTCI data with an interactive timeline
- **Scenario Comparison**: Switch between different urban design scenarios
- **Real-time Analytics**: View statistics and heatmaps for selected hours
- **Responsive Controls**: Orbit, zoom, and pan with smooth camera controls

## Architecture

The viewer uses a clean separation between data and application:
- **Data Location**: All analysis data (models, analyses, validation) resides in `../data/` (project root)
- **GitHub Pages**: Data is served from `/fast-utci/data/`, app from `/fast-utci/viewer/build/`

This eliminates redundancy and keeps the repository size manageable.

## Development

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev

# or open in browser automatically
npm run dev -- --open
```

The app will be available at `http://localhost:5173`.

## Building for Production

### Standard Build

```bash
npm run build
```

### GitHub Pages Build

For deployment to GitHub Pages with the correct base path:

```bash
npm run build:gh
```

This sets `NODE_ENV=production` to configure the base path as `/fast-utci/viewer/` and creates a `.nojekyll` file to prevent Jekyll processing.

## Preview Production Build

After building, preview the production version locally:

```bash
npm run preview
```

**Note**: The preview server simulates the GitHub Pages environment with the base path `/fast-utci/viewer/`.

## Testing

Run unit tests:

```bash
npm test

# or with UI
npm run test:ui

# or with coverage
npm run test:coverage
```

## Deployment to GitHub Pages

1. **Build the production version**:
   ```bash
   cd viewer
   npm run build:gh
   ```

2. **Commit all changes** (including the `build/` folder):
   ```bash
   git add .
   git commit -m "Update viewer for GitHub Pages"
   git push
   ```

3. **Ensure GitHub Pages is configured** to serve from the root directory.

The viewer will be accessible at: `https://[username].github.io/fast-utci/viewer/build/`

The root redirect at `https://[username].github.io/fast-utci/` will automatically forward to `viewer/build/`.

## Project Structure

```
viewer/
├── src/
│   ├── lib/
│   │   ├── components/     # Svelte components (Scene, UI, etc.)
│   │   ├── services/       # Business logic (data loading, caching)
│   │   ├── stores/         # Svelte stores (state management)
│   │   ├── utils/          # Utility functions
│   │   └── types/          # TypeScript type definitions
│   └── routes/             # SvelteKit routes
├── tests/                  # Unit tests
├── static/                 # Static assets (robots.txt)
└── build/                  # Production build output (commit this!)
```

## Key Technologies

- **SvelteKit**: Full-stack framework with static adapter for GitHub Pages
- **Threlte**: Svelte wrapper for Three.js, providing reactive 3D rendering
- **Three.js**: 3D graphics library for WebGL rendering
- **TypeScript**: Type-safe development
- **Vitest**: Unit testing framework
