"""Build semantic graph nodes from registered metric depth and instance maps."""
from ssg_core import extract_semantic_nodes

def build_semantic_nodes(depth, K, instance_map, category_map, min_pixels=8):
    """Return semantic Node objects. Inputs must be registered to the same image."""
    return extract_semantic_nodes(depth, K, instance_map, category_map, min_pixels=min_pixels)
