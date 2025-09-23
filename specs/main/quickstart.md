# Quickstart (Phase 1)

## Prereqs
- Python 3.x, Windows
- Install deps: `pip install -r requirements.txt`

## Run Baseline vs Fast
1. Baseline (current workflow):
   - `python demo_utci_workflow.py --scene data/3D_Models/100.gltf --grid 10`
2. Fast mode (prototype path):
   - `python demo_utci_workflow_simplified_model.py --fast --scene data/3D_Models/100.gltf --grid 10`

## Validate
- Compare generated CSV/HTML outputs.
- Target: ≤ 60 s first update; ≤ 2.0 °C UTCI RMSE.

## Troubleshooting
- If Embree unavailable, ensure fallback BVH path is used.
- Large scenes: increase tile size or reduce rays per point.
