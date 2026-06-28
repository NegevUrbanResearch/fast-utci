import ast
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_ben_gurion_intermediates.py"
ROOT = Path(__file__).resolve().parents[1]
SOURCE = SCRIPT_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _get_assignment_list(name: str) -> list[str]:
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(node.value, ast.List):
                        return [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
    raise AssertionError(f"Could not find list assignment for {name}")


def _get_mrt_payload_keys() -> set[str]:
    for node in ast.walk(TREE):
        if isinstance(node, ast.Dict):
            keys = set()
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
            if {"mrt", "short_erf", "long_erf", "short_dmrt", "long_dmrt"}.issubset(keys):
                return keys
    raise AssertionError("Could not find MRT payload dict with expected keys")


def test_default_stages_include_mrt_by_default() -> None:
    assert _get_assignment_list("DEFAULT_STAGES") == ["solar", "sky", "mrt"]


def test_stage_help_text_mentions_mrt_default() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "default: solar, sky, mrt" in result.stdout.lower()


def test_mrt_json_schema_includes_expected_arrays() -> None:
    keys = _get_mrt_payload_keys()
    assert {"mrt", "short_erf", "long_erf", "short_dmrt", "long_dmrt"}.issubset(keys)
    assert "numPositions" in keys
    assert "numHours" in keys
