# Semantic–structural graph relocalization: reference implementation



## Quick start
Python 3.10+; from this folder:
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest -v
python run.py demo
```

## Real input
`python run.py extract frame.npz query.json`
NPZ keys: depth (H,W, metric optical-axis depth in meters), K (3,3), instances (H,W nonnegative integer IDs, zero background), categories (H,W nonnegative integer class IDs), patches (H,W nonnegative integer surface region IDs, zero background). All registered to undistorted camera image. Patch IDs must come from a real depth-based surface segmentation pipeline; this package fits their normals and extracts adjacent boundary nodes, it does not implement patch segmentation or junction merging. Use consistent class mapping across frames. Camera-facing normal and tangent sign conventions require convex/concave calibration with real geometry.

`python run.py localize query.json reference.json K.txt pose.json --dc 3 --T 1 --h 5`
Reference JSON is an array of nodes with xyz in the world map and code `["s",category]` or `["g",category,sign]`. Query xyz is in camera coordinates and uv is in pixels. K.txt is whitespace-separated 3x3 intrinsic matrix. Output T_cw maps world to camera; camera center is -R.T @ t. Input maps must already be fused with reference poses. Run independently per layer or provide mixed nodes (mixed-layer triangle search is a prototype deviation; use separate layers for manuscript fidelity). Reject failed PnP rather than injecting an invalid pose.

`python evaluate.py pose_errors.csv` requires te_m,re_deg columns and all queries, with blank cells for failures. TE/RE means conditional on valid outputs, recall denominator all queries.

## Equation mapping and deviations
- Eq. 3: `dihedral`, `extract`; atan2 signed angle and threshold. Boundary tangent estimated by PCA; normal and tangent orientation choice is provisional. No connected junction merging yet.
- Eq. 6: `descriptor`, `similarity`; angle discrepancy and normalized edge-ratio discrepancy are L1 sums, radians, normalization by maximum edge.
- Eq. 4–5 / Algorithm 1: `topology`, `match`; distance and nearest-neighbor graph, greedy complete-link clustering, triangle enumeration, top-h per candidate cluster, greedy one-to-one selection, adjacent-cluster distance-based refinement. Cluster pairing via summed scores, ceil rounding, adjacency 2*dc and distance weighting are explicit prototype choices;
- Section III-F: `pose`; centroid 2D–3D PnP-RANSAC plus LM pose-only refinement, NOT bundle adjustment.

## External dependencies and integration plan
Keep upstream code separate and retain licenses/attribution. Record actual commit IDs and model checkpoint hashes. RTAB-Map: https://github.com/introlab/rtabmap ; ROS wrapper: https://github.com/introlab/rtabmap_ros ; ORB-SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3 ; 

Indoor: registered measured RGB-D plus actual instance segmentation model. Outdoor: metric depth estimation or reference reconstruction projection, according to actual experiment; query projection must not depend on unknown ground-truth query pose. SegFormer ADE20K gives semantic classes, not instance IDs; use a documented region/instance method. 

ORB-SLAM3: integrate into the actual Tracking::Relocalization path (version-specific). Export candidate reference keyframe IDs, retrieve their map points, establish ORB correspondences, use existing geometric pose validation and local-map tracking. Preserve original failure handling. Convert T_cw and scale to the map conventions. Log accepted/rejected hypotheses, map ID, before/after pose, inlier counts, tracking state and time. Existing ORB-SLAM3 local BA remains upstream. This package has no live ORB-SLAM3 binding, ROS launch or BA implementation.
