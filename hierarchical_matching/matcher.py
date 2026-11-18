"""Layer-wise graph clustering, triangle matching and adaptive candidate selection."""
from ssg_core import hierarchical_match, hierarchical_match_adaptive

def match_graphs(query_nodes, reference_nodes, dc=3.0, neighbor_ratio=0.15,
                 top_h=12, min_similarity=0.5, lambda_angle=1.0, lambda_ratio=1.0):
    return hierarchical_match(query_nodes, reference_nodes, dc=dc,
        neighbor_ratio=neighbor_ratio, top_h=top_h,
        min_similarity=min_similarity, lambda_angle=lambda_angle,
        lambda_ratio=lambda_ratio)

def match_graphs_adaptive(query_nodes, reference_nodes, config):
    return hierarchical_match_adaptive(query_nodes, reference_nodes, config)
