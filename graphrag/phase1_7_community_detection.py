"""
PHASE 1.7: COMMUNITY DETECTION (GRAPHRAG CORE)

Apply hierarchical clustering (Leiden).

Produces:
- C0: entire novel
- C1: major arcs
- C2: subplots
- C3: scene-level clusters
"""

import numpy as np
import networkx as nx
import leidenalg
import igraph as ig
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from schemas import KnowledgeGraph, KnowledgeGraphNode, CommunitySummary


class CommunityDetection:
    """
    Community detection layer using Leiden algorithm.
    
    Applies hierarchical clustering to detect communities at multiple levels.
    """
    
    def __init__(self):
        self.communities: Dict[int, Dict[str, int]] = {}  # level -> {node_id: community_id}
        self.community_nodes: Dict[Tuple[int, int], List[str]] = {}  # (level, community_id) -> [node_ids]
    
    def detect_communities(
        self,
        graph: KnowledgeGraph,
        levels: List[int] = None
    ) -> Dict[int, Dict[str, int]]:
        """
        Detect communities at multiple hierarchical levels.
        
        Process:
        1. Convert KnowledgeGraph to igraph format
        2. Apply Leiden algorithm at different resolutions
        3. Create hierarchical community structure
        
        Args:
            graph: KnowledgeGraph object
            levels: List of levels to detect (default: [0, 1, 2, 3])
            
        Returns:
            Dictionary mapping level to node->community mapping
        """
        if levels is None:
            levels = [0, 1, 2, 3]
        
        # Convert to igraph
        ig_graph = self._knowledge_graph_to_igraph(graph)
        
        # Level 0: entire novel (single community)
        all_nodes = list(graph.nodes.keys())
        level_0_communities = {node_id: 0 for node_id in all_nodes}
        self.communities[0] = level_0_communities
        self.community_nodes[(0, 0)] = all_nodes
        
        # Levels 1-3: hierarchical Leiden clustering
        resolutions = [0.5, 1.0, 1.5]  # Different resolutions for different levels
        
        for level, resolution in zip(levels[1:], resolutions):
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                resolution_parameter=resolution
            )
            
            # Map nodes to communities
            node_community_map = {}
            for i, community_id in enumerate(partition.membership):
                node_id = all_nodes[i]
                node_community_map[node_id] = community_id
            
            self.communities[level] = node_community_map
            
            # Build community -> nodes mapping
            community_nodes_map = defaultdict(list)
            for node_id, community_id in node_community_map.items():
                community_nodes_map[community_id].append(node_id)
            
            for community_id, nodes in community_nodes_map.items():
                self.community_nodes[(level, community_id)] = nodes
        
        return self.communities
    
    def _knowledge_graph_to_igraph(self, graph: KnowledgeGraph) -> ig.Graph:
        """
        Convert KnowledgeGraph to igraph format.
        
        Args:
            graph: KnowledgeGraph object
            
        Returns:
            igraph Graph object
        """
        # Create mapping from node_id to index
        node_ids = list(graph.nodes.keys())
        node_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # Create edges list with weights
        edges = []
        weights = []
        
        for edge in graph.edges:
            if edge.source_id in node_to_index and edge.target_id in node_to_index:
                source_idx = node_to_index[edge.source_id]
                target_idx = node_to_index[edge.target_id]
                edges.append((source_idx, target_idx))
                weights.append(edge.weight)
        
        # Create igraph
        ig_graph = ig.Graph(edges=edges, directed=False)
        ig_graph.es['weight'] = weights
        ig_graph.vs['node_id'] = node_ids
        
        return ig_graph
    
    def get_community_nodes(self, level: int, community_id: int) -> List[str]:
        """
        Get nodes in a specific community.
        
        Args:
            level: Community level
            community_id: Community ID
            
        Returns:
            List of node IDs in the community
        """
        return self.community_nodes.get((level, community_id), [])
    
    def get_node_community(self, node_id: str, level: int) -> int:
        """
        Get community ID for a node at a specific level.
        
        Args:
            node_id: Node ID
            level: Community level
            
        Returns:
            Community ID
        """
        if level not in self.communities:
            return -1
        
        return self.communities[level].get(node_id, -1)
    
    def get_communities_at_level(self, level: int) -> Dict[int, List[str]]:
        """
        Get all communities at a specific level.
        
        Args:
            level: Community level
            
        Returns:
            Dictionary mapping community_id to list of node IDs
        """
        if level not in self.communities:
            return {}
        
        communities = defaultdict(list)
        for node_id, community_id in self.communities[level].items():
            communities[community_id].append(node_id)
        
        return dict(communities)
    
    def get_hierarchical_communities(self) -> Dict[int, Dict[int, List[str]]]:
        """
        Get complete hierarchical community structure.
        
        Returns:
            Dictionary mapping level to community->nodes mapping
        """
        result = {}
        for level in self.communities.keys():
            result[level] = self.get_communities_at_level(level)
        return result


if __name__ == "__main__":
    # Test example
    from phase1_6_knowledge_graph_assembly import KnowledgeGraphAssembly
    from schemas import KnowledgeGraph
    
    # Create a simple test graph
    from schemas import KnowledgeGraphNode, KnowledgeGraphEdge, NodeType, EdgeType
    
    graph = KnowledgeGraph()
    graph.nodes["e_1"] = KnowledgeGraphNode(
        node_id="e_1",
        node_type=NodeType.CHARACTER,
        properties={"name": "John"}
    )
    graph.nodes["e_2"] = KnowledgeGraphNode(
        node_id="e_2",
        node_type=NodeType.CHARACTER,
        properties={"name": "Mary"}
    )
    graph.nodes["ev_1"] = KnowledgeGraphNode(
        node_id="ev_1",
        node_type=NodeType.EVENT,
        properties={"description": "Meeting"}
    )
    
    graph.edges.append(KnowledgeGraphEdge(
        edge_id="e1",
        source_id="e_1",
        target_id="ev_1",
        edge_type=EdgeType.PARTICIPATES_IN,
        weight=1.0
    ))
    graph.edges.append(KnowledgeGraphEdge(
        edge_id="e2",
        source_id="e_2",
        target_id="ev_1",
        edge_type=EdgeType.PARTICIPATES_IN,
        weight=1.0
    ))
    
    detector = CommunityDetection()
    communities = detector.detect_communities(graph)
    
    print(f"Detected communities at {len(communities)} levels")
    for level, comms in communities.items():
        print(f"Level {level}: {len(set(comms.values()))} communities")

