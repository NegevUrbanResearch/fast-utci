from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from scripts.export_for_viewer import export_utci_for_viewer


def test_export_utci_for_viewer_project_paths(tmp_path: Path) -> None:
	utci_results = {
		'position_0': {
			'position': (0.0, 0.0, 0.0),
			'utci': [25.0],
			'datetime': None
		}
	}

	output_dir = tmp_path / 'out'

	binary_path, metadata_path = export_utci_for_viewer(
		utci_results=utci_results,
		analysis_type='single_hour',
		grid_size=2.0,
		model_file='data/3d_models/Ben-Gurion/original_with_layers.glb',
		epw_file='data/weather/test.epw',
		runtime_seconds=1.0,
		output_dir=str(output_dir),
		project='Ben-Gurion',
		category='existing_buildings'
	)

	assert Path(binary_path).exists()
	assert Path(metadata_path).exists()
	assert str(output_dir / 'Ben-Gurion' / 'existing_buildings') in binary_path
	assert str(output_dir / 'Ben-Gurion' / 'existing_buildings') in metadata_path
