"""PnP-RANSAC pose estimation from matched 3-D node coordinates and 2-D observations."""
from ssg_core import estimate_pose_pnp

def estimate_pose(query_nodes, reference_nodes, matches, K, reprojection_error=4.0,
                  confidence=0.99, iterations=2000):
    return estimate_pose_pnp(query_nodes, reference_nodes, matches, K,
        reprojection_error=reprojection_error, confidence=confidence,
        iterations=iterations)
