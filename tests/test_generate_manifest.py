import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_manifest import generate_manifest


def write_json(path: Path, data: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(data), encoding='utf-8')


def test_generate_manifest_per_project_layout(tmp_path: Path) -> None:
	analyses_dir = tmp_path / 'data' / 'analyses'

	write_json(
		analyses_dir / 'Ben-Gurion' / '20250815_grid_2m_fullday.json',
		{
			'analysis_id': '20250815_grid_2m_fullday',
			'analysis_type': 'full_day',
			'grid_size': 2.0,
			'date': '20250815',
			'num_positions': 10,
			'hours': list(range(24)),
			'model_file': 'data/3d_models/Ben-Gurion/original_with_layers.glb'
		}
	)

	write_json(
		analyses_dir / 'Ben-Gurion' / 'existing_buildings' / 'existing_buildings_01.json',
		{
			'analysis_id': 'existing_buildings_01',
			'analysis_type': 'full_day',
			'grid_size': 2.0,
			'date': '20250815',
			'num_positions': 10,
			'hours': list(range(24)),
			'model_file': 'data/3d_models/Ben-Gurion/scenarios/existing_buildings/existing_buildings_01.glb'
		}
	)

	write_json(
		analyses_dir / 'Ness-Tziona' / 'original' / '20250815_grid_2m_fullday.json',
		{
			'analysis_id': '20250815_grid_2m_fullday',
			'analysis_type': 'full_day',
			'grid_size': 2.0,
			'date': '20250815',
			'num_positions': 10,
			'hours': list(range(24)),
			'model_file': 'data/3d_models/Ness-Tziona/nes_tziona_1.gltf'
		}
	)

	output_path = tmp_path / 'manifest.json'

	assert generate_manifest(str(analyses_dir), str(output_path)) is True

	manifest = json.loads(output_path.read_text(encoding='utf-8'))
	ids = {entry['id'] for entry in manifest['analyses']}

	assert 'Ben-Gurion/20250815_grid_2m_fullday' in ids
	assert 'Ben-Gurion/existing_buildings/existing_buildings_01' in ids
	assert 'Ness-Tziona/original/20250815_grid_2m_fullday' in ids

	projects = {entry['project'] for entry in manifest['analyses']}
	assert 'Ben-Gurion' in projects
	assert 'Ness-Tziona' in projects
