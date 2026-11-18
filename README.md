# SSG Core — executable reference implementation

This package is a **revised extension of `cluster_repro/core.py`**, not identical to it. It separates semantic and structural layers, uses connected components of the retained graph, includes nearest-cluster links for isolated nodes, supports same-code inter-cluster links, and exposes a CLI for registered metric-depth inputs, graph matching and PnP.





## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Real input format

Create an NPZ containing arrays registered to the same undistorted image:

- `depth`: HxW metric optical-axis depth in metres
- `K`: 3x3 camera intrinsic matrix
- `instances`: HxW instance IDs; 0 is background
- `categories`: HxW integer semantic class IDs
- `patches`: HxW continuous surface-patch IDs; 0 is background

The instance map must truly be instance-level. A semantic-only ADE20K label map is not an instance map. `patches` must come from a documented depth/geometry surface segmentation algorithm; this package estimates patch planes and boundary tangents, it does not create the patches.

```bash
python cli.py extract frame.npz query_nodes.json
```

For a reference map, prepare JSON containing a list of Node objects with fields `xyz`, `uv`, `layer`, `category`, `sign`, `node_id`. `xyz` must be in the common reference-map frame; query `xyz` must be in the query-camera frame and `uv` the observed region centroid in pixels. The reference map must already be multi-view fused into a common coordinate frame.

```bash
python cli.py match query_nodes.json reference_nodes.json --dc 3 --T 0.85 --h 12 --threshold 0.5 --out matches.json
python cli.py localize query_nodes.json reference_nodes.json K.txt --matches matches.json --out pose.json
```

`pose.json` reports `T_cw` (world/map to camera). This is centroid-based PnP. Since region centroids need not be identical physical points, inspect inlier count and reprojection residuals on real data. It is not ORB-SLAM3 BA.

## Equation mapping

- Eq. (3): `signed_dihedral`; `extract_structural_nodes` estimates patch normals and shared-boundary tangent from supplied patch labels. Normal/tangent sign convention must be calibrated consistently.
- Eq. (6): `triangle_descriptor`, `triangle_discrepancy`, `triangle_similarity`; descriptor stores three normalized edge lengths followed by three internal angles (radians).
- Algorithm 1: `hierarchical_match`; semantic and structural layers are matched separately; compatible triangles are exhaustively searched within clusters; top-h candidates are retained; one-to-one node assignment is applied; adjacent-cluster local refinement uses cross-cluster distance consistency.




