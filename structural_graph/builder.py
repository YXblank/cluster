"""Estimate structural nodes from supplied surface-patch labels."""
from ssg_core import extract_structural_nodes, signed_dihedral

def build_structural_nodes(depth, K, patch_map, category_map, dihedral_threshold=0.15, min_pixels=8, merge_regions=True):
    """Return structural Node objects. patch_map must be produced upstream."""
    return extract_structural_nodes(depth, K, patch_map, category_map,
        threshold=dihedral_threshold, min_pixels=min_pixels, merge_regions=merge_regions)
