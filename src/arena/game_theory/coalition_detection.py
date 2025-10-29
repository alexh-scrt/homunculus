"""
Coalition Detection for Arena

This module implements algorithms to detect and prevent
coalitions and collusion between agents.

Features:
- Pattern-based coalition detection
- Collaboration tracking
- Alliance identification
- Manipulation detection
- Countermeasures

Author: Homunculus Team
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
try:
    import networkx as nx
except ImportError:
    # NetworkX is optional - use simple graph implementation
    nx = None

from ..models import Message, AgentState

logger = logging.getLogger(__name__)


# Helper functions for when networkx not available
def find_cliques(graph):
    """Simple clique finding for SimpleGraph."""
    if hasattr(graph, 'find_cliques'):
        return nx.find_cliques(graph)
    # Simple implementation - find all triangles
    cliques = []
    nodes = list(graph.nodes)
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            if graph.has_edge(node1, node2):
                for node3 in nodes[j+1:]:
                    if graph.has_edge(node1, node3) and graph.has_edge(node2, node3):
                        cliques.append([node1, node2, node3])
    return cliques

def degree_centrality(graph):
    """Simple degree centrality for SimpleGraph."""
    if hasattr(graph, 'degree_centrality'):
        return nx.degree_centrality(graph)
    centrality = {}
    n = len(graph.nodes)
    if n <= 1:
        return {node: 0 for node in graph.nodes}
    for node in graph.nodes:
        degree = len(graph.edges.get(node, set()))
        centrality[node] = degree / (n - 1)
    return centrality

# Simple graph implementation when networkx not available
class SimpleGraph:
    """Simple graph implementation for when networkx is not available."""
    
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.edge_weights = {}
    
    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = set()
    
    def add_edge(self, node1, node2, weight=1):
        self.add_node(node1)
        self.add_node(node2)
        self.edges[node1].add(node2)
        self.edges[node2].add(node1)
        self.edge_weights[(node1, node2)] = weight
        self.edge_weights[(node2, node1)] = weight
    
    def has_edge(self, node1, node2):
        return node1 in self.edges and node2 in self.edges[node1]
    
    def neighbors(self, node):
        return list(self.edges.get(node, set()))
    
    def __getitem__(self, node):
        # Return edge data
        return {n: {"weight": self.edge_weights.get((node, n), 1)} 
                for n in self.edges.get(node, set())}
    
    def __contains__(self, node):
        return node in self.nodes


@dataclass
class CollaborationPattern:
    """Pattern of collaboration between agents."""
    agent1: str
    agent2: str
    interaction_count: int
    mutual_support_count: int
    synchronized_actions: int
    similarity_score: float
    time_correlation: float
    
    @property
    def collaboration_strength(self) -> float:
        """Calculate overall collaboration strength."""
        return (
            0.2 * min(1.0, self.interaction_count / 10) +
            0.3 * min(1.0, self.mutual_support_count / 5) +
            0.2 * min(1.0, self.synchronized_actions / 3) +
            0.2 * self.similarity_score +
            0.1 * self.time_correlation
        )
    
    def is_suspicious(self) -> bool:
        """Check if collaboration is suspicious."""
        return self.collaboration_strength > 0.7


@dataclass
class Coalition:
    """Detected coalition of agents."""
    members: Set[str]
    formation_turn: int
    strength: float
    evidence: List[str]
    pattern_type: str  # 'mutual_support', 'synchronized', 'exclusive'
    
    def add_member(self, agent_id: str) -> None:
        """Add member to coalition."""
        self.members.add(agent_id)
    
    def remove_member(self, agent_id: str) -> None:
        """Remove member from coalition."""
        self.members.discard(agent_id)
    
    @property
    def size(self) -> int:
        """Get coalition size."""
        return len(self.members)


class CoalitionDetector:
    """
    Main coalition detection system.
    
    Uses graph analysis and pattern recognition to identify
    potential coalitions and collusive behavior.
    """
    
    def __init__(
        self,
        sensitivity: float = 0.6,
        min_coalition_size: int = 2,
        detection_window: int = 20
    ):
        """
        Initialize coalition detector.
        
        Args:
            sensitivity: Detection sensitivity (0.0 to 1.0)
            min_coalition_size: Minimum size to consider coalition
            detection_window: Turns to analyze for patterns
        """
        self.sensitivity = sensitivity
        self.min_coalition_size = min_coalition_size
        self.detection_window = detection_window
        
        # Interaction tracking
        self.interaction_graph = nx.Graph() if nx else SimpleGraph()
        self.interaction_history: List[Dict[str, Any]] = []
        
        # Coalition tracking
        self.detected_coalitions: List[Coalition] = []
        self.coalition_scores: Dict[Tuple[str, ...], float] = {}
        
        # Pattern analysis
        self.collaboration_patterns: Dict[Tuple[str, str], CollaborationPattern] = {}
        self.suspicious_agents: Set[str] = set()
    
    def analyze_interactions(
        self,
        messages: List[Message],
        turn_number: int
    ) -> List[Coalition]:
        """
        Analyze interactions to detect coalitions.
        
        Args:
            messages: Recent messages
            turn_number: Current turn
            
        Returns:
            List of detected coalitions
        """
        # Update interaction graph
        self._update_interaction_graph(messages)
        
        # Detect patterns
        patterns = self._detect_collaboration_patterns(messages, turn_number)
        
        # Identify coalitions
        coalitions = self._identify_coalitions(patterns, turn_number)
        
        # Update tracking
        self.detected_coalitions.extend(coalitions)
        
        return coalitions
    
    def _update_interaction_graph(self, messages: List[Message]) -> None:
        """Update interaction graph based on messages."""
        for msg in messages:
            sender = msg.sender_id
            
            # Add node if not exists
            if sender not in self.interaction_graph:
                self.interaction_graph.add_node(sender)
            
            # Check for references to other agents
            referenced = self._extract_references(msg)
            
            for ref in referenced:
                if ref not in self.interaction_graph:
                    self.interaction_graph.add_node(ref)
                
                # Add or update edge
                if self.interaction_graph.has_edge(sender, ref):
                    # Increase weight
                    self.interaction_graph[sender][ref]["weight"] += 1
                else:
                    self.interaction_graph.add_edge(sender, ref, weight=1)
            
            # Track in history
            self.interaction_history.append({
                "sender": sender,
                "referenced": referenced,
                "turn": msg.metadata.get("turn", 0),
                "type": msg.message_type
            })
    
    def _detect_collaboration_patterns(
        self,
        messages: List[Message],
        turn_number: int
    ) -> List[CollaborationPattern]:
        """Detect collaboration patterns between agents."""
        patterns = []
        
        # Get recent interactions
        recent_window = [
            h for h in self.interaction_history
            if turn_number - h["turn"] <= self.detection_window
        ]
        
        # Analyze pairs of agents
        agents = list(self.interaction_graph.nodes())
        
        for i, agent1 in enumerate(agents):
            for agent2 in agents[i+1:]:
                pattern = self._analyze_pair(agent1, agent2, recent_window, messages)
                
                if pattern and pattern.is_suspicious():
                    patterns.append(pattern)
                    self.collaboration_patterns[(agent1, agent2)] = pattern
        
        return patterns
    
    def _analyze_pair(
        self,
        agent1: str,
        agent2: str,
        interactions: List[Dict],
        messages: List[Message]
    ) -> Optional[CollaborationPattern]:
        """Analyze collaboration between two agents."""
        # Count interactions
        interaction_count = 0
        mutual_support = 0
        synchronized = 0
        
        for interaction in interactions:
            if interaction["sender"] == agent1 and agent2 in interaction["referenced"]:
                interaction_count += 1
            elif interaction["sender"] == agent2 and agent1 in interaction["referenced"]:
                interaction_count += 1
        
        if interaction_count == 0:
            return None
        
        # Analyze mutual support
        mutual_support = self._count_mutual_support(agent1, agent2, messages)
        
        # Check synchronized actions
        synchronized = self._count_synchronized_actions(agent1, agent2, interactions)
        
        # Calculate similarity
        similarity = self._calculate_similarity(agent1, agent2, messages)
        
        # Time correlation
        time_correlation = self._calculate_time_correlation(agent1, agent2, interactions)
        
        return CollaborationPattern(
            agent1=agent1,
            agent2=agent2,
            interaction_count=interaction_count,
            mutual_support_count=mutual_support,
            synchronized_actions=synchronized,
            similarity_score=similarity,
            time_correlation=time_correlation
        )
    
    def _count_mutual_support(
        self,
        agent1: str,
        agent2: str,
        messages: List[Message]
    ) -> int:
        """Count instances of mutual support."""
        support_count = 0
        
        for msg in messages:
            if msg.sender_id == agent1:
                # Check if supporting agent2
                content = msg.content.lower()
                if agent2.lower() in content and any(
                    word in content for word in ["agree", "support", "correct", "excellent"]
                ):
                    support_count += 1
            elif msg.sender_id == agent2:
                # Check if supporting agent1
                content = msg.content.lower()
                if agent1.lower() in content and any(
                    word in content for word in ["agree", "support", "correct", "excellent"]
                ):
                    support_count += 1
        
        return support_count
    
    def _count_synchronized_actions(
        self,
        agent1: str,
        agent2: str,
        interactions: List[Dict]
    ) -> int:
        """Count synchronized actions."""
        synchronized = 0
        
        # Group interactions by turn
        turns = defaultdict(list)
        for interaction in interactions:
            turns[interaction["turn"]].append(interaction["sender"])
        
        # Check for synchronized behavior
        for turn_agents in turns.values():
            if agent1 in turn_agents and agent2 in turn_agents:
                synchronized += 1
        
        return synchronized
    
    def _calculate_similarity(
        self,
        agent1: str,
        agent2: str,
        messages: List[Message]
    ) -> float:
        """Calculate content similarity between agents."""
        agent1_messages = [m for m in messages if m.sender_id == agent1]
        agent2_messages = [m for m in messages if m.sender_id == agent2]
        
        if not agent1_messages or not agent2_messages:
            return 0.0
        
        # Simple similarity based on message length and keywords
        # In production, use embeddings or more sophisticated NLP
        avg_len1 = np.mean([len(m.content) for m in agent1_messages])
        avg_len2 = np.mean([len(m.content) for m in agent2_messages])
        
        length_similarity = 1.0 - abs(avg_len1 - avg_len2) / max(avg_len1, avg_len2)
        
        # Keyword overlap
        keywords1 = self._extract_keywords(agent1_messages)
        keywords2 = self._extract_keywords(agent2_messages)
        
        if not keywords1 or not keywords2:
            keyword_similarity = 0.0
        else:
            overlap = len(keywords1.intersection(keywords2))
            total = len(keywords1.union(keywords2))
            keyword_similarity = overlap / total if total > 0 else 0.0
        
        return (length_similarity + keyword_similarity) / 2
    
    def _calculate_time_correlation(
        self,
        agent1: str,
        agent2: str,
        interactions: List[Dict]
    ) -> float:
        """Calculate temporal correlation of actions."""
        agent1_turns = [i["turn"] for i in interactions if i["sender"] == agent1]
        agent2_turns = [i["turn"] for i in interactions if i["sender"] == agent2]
        
        if not agent1_turns or not agent2_turns:
            return 0.0
        
        # Check for consistent timing patterns
        time_diffs = []
        for t1 in agent1_turns:
            closest_t2 = min(agent2_turns, key=lambda t2: abs(t2 - t1)) if agent2_turns else None
            if closest_t2:
                time_diffs.append(abs(t1 - closest_t2))
        
        if not time_diffs:
            return 0.0
        
        # Low variance in time differences suggests coordination
        avg_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)
        
        if avg_diff == 0:
            return 1.0  # Perfect synchronization
        
        # Normalize correlation score
        correlation = 1.0 - min(1.0, std_diff / avg_diff)
        return correlation
    
    def _identify_coalitions(
        self,
        patterns: List[CollaborationPattern],
        turn_number: int
    ) -> List[Coalition]:
        """Identify coalitions from collaboration patterns."""
        coalitions = []
        
        # Build collaboration graph
        collab_graph = nx.Graph() if nx else SimpleGraph()
        
        for pattern in patterns:
            if pattern.collaboration_strength > self.sensitivity:
                collab_graph.add_edge(
                    pattern.agent1,
                    pattern.agent2,
                    weight=pattern.collaboration_strength
                )
        
        # Find cliques (fully connected subgraphs)
        cliques = list(find_cliques(collab_graph))
        
        for clique in cliques:
            if len(clique) >= self.min_coalition_size:
                # Calculate coalition strength
                strength = self._calculate_coalition_strength(clique, collab_graph)
                
                # Gather evidence
                evidence = self._gather_evidence(clique, patterns)
                
                # Determine pattern type
                pattern_type = self._determine_pattern_type(clique, patterns)
                
                coalition = Coalition(
                    members=set(clique),
                    formation_turn=turn_number,
                    strength=strength,
                    evidence=evidence,
                    pattern_type=pattern_type
                )
                
                coalitions.append(coalition)
        
        # Also check for star patterns (one central node)
        star_coalitions = self._detect_star_patterns(collab_graph, patterns, turn_number)
        coalitions.extend(star_coalitions)
        
        return coalitions
    
    def _calculate_coalition_strength(
        self,
        members: List[str],
        graph
    ) -> float:
        """Calculate overall coalition strength."""
        if len(members) < 2:
            return 0.0
        
        # Average edge weight between members
        total_weight = 0
        edge_count = 0
        
        for i, member1 in enumerate(members):
            for member2 in members[i+1:]:
                if graph.has_edge(member1, member2):
                    total_weight += graph[member1][member2]["weight"]
                    edge_count += 1
        
        if edge_count == 0:
            return 0.0
        
        avg_weight = total_weight / edge_count
        
        # Size factor (larger coalitions are more concerning)
        size_factor = min(1.0, len(members) / 5)
        
        return avg_weight * (1 + size_factor * 0.5)
    
    def _gather_evidence(
        self,
        members: List[str],
        patterns: List[CollaborationPattern]
    ) -> List[str]:
        """Gather evidence of coalition behavior."""
        evidence = []
        
        # Check patterns involving members
        for pattern in patterns:
            if pattern.agent1 in members and pattern.agent2 in members:
                if pattern.mutual_support_count > 3:
                    evidence.append(f"High mutual support between {pattern.agent1} and {pattern.agent2}")
                if pattern.synchronized_actions > 2:
                    evidence.append(f"Synchronized actions by {pattern.agent1} and {pattern.agent2}")
                if pattern.similarity_score > 0.8:
                    evidence.append(f"High content similarity between {pattern.agent1} and {pattern.agent2}")
        
        return evidence
    
    def _determine_pattern_type(
        self,
        members: List[str],
        patterns: List[CollaborationPattern]
    ) -> str:
        """Determine coalition pattern type."""
        mutual_support_count = 0
        synchronized_count = 0
        
        for pattern in patterns:
            if pattern.agent1 in members and pattern.agent2 in members:
                if pattern.mutual_support_count > 2:
                    mutual_support_count += 1
                if pattern.synchronized_actions > 1:
                    synchronized_count += 1
        
        if mutual_support_count > synchronized_count:
            return "mutual_support"
        elif synchronized_count > 0:
            return "synchronized"
        else:
            return "exclusive"
    
    def _detect_star_patterns(
        self,
        graph,
        patterns: List[CollaborationPattern],
        turn_number: int
    ) -> List[Coalition]:
        """Detect star-shaped coalition patterns (hub and spokes)."""
        coalitions = []
        
        # Find nodes with high degree centrality
        centrality = degree_centrality(graph)
        
        for node, cent_score in centrality.items():
            if cent_score > 0.5:  # Potential hub
                neighbors = list(graph.neighbors(node))
                
                if len(neighbors) >= self.min_coalition_size - 1:
                    # Check if hub coordinates with spokes
                    coordination_score = self._check_hub_coordination(
                        node, neighbors, patterns
                    )
                    
                    if coordination_score > self.sensitivity:
                        coalition = Coalition(
                            members={node}.union(neighbors),
                            formation_turn=turn_number,
                            strength=coordination_score,
                            evidence=[f"{node} acts as central coordinator"],
                            pattern_type="star"
                        )
                        coalitions.append(coalition)
        
        return coalitions
    
    def _check_hub_coordination(
        self,
        hub: str,
        spokes: List[str],
        patterns: List[CollaborationPattern]
    ) -> float:
        """Check coordination between hub and spokes."""
        total_strength = 0
        count = 0
        
        for spoke in spokes:
            key = tuple(sorted([hub, spoke]))
            if key in self.collaboration_patterns:
                pattern = self.collaboration_patterns[key]
                total_strength += pattern.collaboration_strength
                count += 1
        
        if count == 0:
            return 0.0
        
        return total_strength / count
    
    def _extract_references(self, message: Message) -> Set[str]:
        """Extract agent references from message."""
        references = set()
        
        # Extract from metadata
        if "referenced_agents" in message.metadata:
            references.update(message.metadata["referenced_agents"])
        
        # Simple extraction from content (could be more sophisticated)
        content_lower = message.content.lower()
        
        # Look for agent mentions (simplified)
        for node in self.interaction_graph.nodes():
            if node.lower() in content_lower:
                references.add(node)
        
        return references
    
    def _extract_keywords(self, messages: List[Message]) -> Set[str]:
        """Extract keywords from messages."""
        keywords = set()
        
        for msg in messages:
            # Simple keyword extraction
            words = msg.content.lower().split()
            # Filter common words and keep meaningful ones
            meaningful = [w for w in words if len(w) > 4]
            keywords.update(meaningful[:10])  # Top 10 per message
        
        return keywords


class AllianceTracker:
    """
    Track explicit and implicit alliances between agents.
    """
    
    def __init__(self):
        """Initialize alliance tracker."""
        self.explicit_alliances: Dict[str, Set[str]] = defaultdict(set)
        self.implicit_alliances: Dict[str, Set[str]] = defaultdict(set)
        self.alliance_history: List[Dict[str, Any]] = []
    
    def record_explicit_alliance(
        self,
        agent1: str,
        agent2: str,
        turn: int
    ) -> None:
        """Record explicit alliance declaration."""
        self.explicit_alliances[agent1].add(agent2)
        self.explicit_alliances[agent2].add(agent1)
        
        self.alliance_history.append({
            "type": "explicit",
            "agents": [agent1, agent2],
            "turn": turn,
            "timestamp": datetime.utcnow()
        })
    
    def detect_implicit_alliance(
        self,
        agent1: str,
        agent2: str,
        confidence: float,
        turn: int
    ) -> None:
        """Detect and record implicit alliance."""
        if confidence > 0.7:
            self.implicit_alliances[agent1].add(agent2)
            self.implicit_alliances[agent2].add(agent1)
            
            self.alliance_history.append({
                "type": "implicit",
                "agents": [agent1, agent2],
                "confidence": confidence,
                "turn": turn,
                "timestamp": datetime.utcnow()
            })
    
    def get_alliance_network(self):
        """Get alliance network graph."""
        graph = nx.Graph() if nx else SimpleGraph()
        
        # Add explicit alliances
        for agent, allies in self.explicit_alliances.items():
            for ally in allies:
                graph.add_edge(agent, ally, type="explicit")
        
        # Add implicit alliances
        for agent, allies in self.implicit_alliances.items():
            for ally in allies:
                if not graph.has_edge(agent, ally):
                    graph.add_edge(agent, ally, type="implicit")
        
        return graph


class ManipulationDetector:
    """
    Detect manipulation tactics and gaming behavior.
    """
    
    def __init__(self, threshold: float = 0.6):
        """
        Initialize manipulation detector.
        
        Args:
            threshold: Detection threshold
        """
        self.threshold = threshold
        self.manipulation_scores: Dict[str, float] = defaultdict(float)
        self.detected_tactics: Dict[str, List[str]] = defaultdict(list)
    
    def analyze_for_manipulation(
        self,
        agent_id: str,
        messages: List[Message],
        context: Dict[str, Any]
    ) -> float:
        """
        Analyze agent behavior for manipulation.
        
        Args:
            agent_id: Agent to analyze
            messages: Agent's messages
            context: Game context
            
        Returns:
            Manipulation score (0.0 to 1.0)
        """
        score = 0.0
        tactics = []
        
        # Check for vote manipulation
        vote_manipulation = self._check_vote_manipulation(agent_id, messages)
        if vote_manipulation > self.threshold:
            score += 0.3
            tactics.append("vote_manipulation")
        
        # Check for false accusations
        false_accusations = self._check_false_accusations(agent_id, context)
        if false_accusations > self.threshold:
            score += 0.3
            tactics.append("false_accusations")
        
        # Check for gaming scoring system
        gaming_score = self._check_scoring_manipulation(agent_id, messages)
        if gaming_score > self.threshold:
            score += 0.2
            tactics.append("scoring_manipulation")
        
        # Check for social engineering
        social_engineering = self._check_social_engineering(agent_id, messages)
        if social_engineering > self.threshold:
            score += 0.2
            tactics.append("social_engineering")
        
        # Update tracking
        self.manipulation_scores[agent_id] = min(1.0, score)
        self.detected_tactics[agent_id] = tactics
        
        return min(1.0, score)
    
    def _check_vote_manipulation(
        self,
        agent_id: str,
        messages: List[Message]
    ) -> float:
        """Check for vote manipulation tactics."""
        vote_related = 0
        manipulation_indicators = 0
        
        for msg in messages:
            if msg.sender_id != agent_id:
                continue
            
            content = msg.content.lower()
            
            # Check for vote-related content
            if any(word in content for word in ["vote", "eliminate", "remove"]):
                vote_related += 1
                
                # Check for manipulation indicators
                if any(phrase in content for phrase in [
                    "vote together", "coordinate", "agree to eliminate"
                ]):
                    manipulation_indicators += 1
        
        if vote_related == 0:
            return 0.0
        
        return manipulation_indicators / vote_related
    
    def _check_false_accusations(
        self,
        agent_id: str,
        context: Dict[str, Any]
    ) -> float:
        """Check for pattern of false accusations."""
        accusations = context.get("accusations_by", {}).get(agent_id, [])
        proven_false = context.get("false_accusations", {}).get(agent_id, 0)
        
        if len(accusations) == 0:
            return 0.0
        
        false_rate = proven_false / len(accusations)
        
        # High false accusation rate is manipulative
        return false_rate
    
    def _check_scoring_manipulation(
        self,
        agent_id: str,
        messages: List[Message]
    ) -> float:
        """Check for attempts to game the scoring system."""
        gaming_score = 0.0
        
        # Check for keyword stuffing
        keyword_density = self._calculate_keyword_density(messages)
        if keyword_density > 0.3:
            gaming_score += 0.5
        
        # Check for repetitive patterns
        if self._check_repetitive_patterns(messages):
            gaming_score += 0.5
        
        return min(1.0, gaming_score)
    
    def _check_social_engineering(
        self,
        agent_id: str,
        messages: List[Message]
    ) -> float:
        """Check for social engineering tactics."""
        engineering_score = 0.0
        
        for msg in messages:
            if msg.sender_id != agent_id:
                continue
            
            content = msg.content.lower()
            
            # Check for emotional manipulation
            if any(word in content for word in [
                "trust me", "believe me", "unfair", "victim"
            ]):
                engineering_score += 0.1
            
            # Check for false consensus building
            if any(phrase in content for phrase in [
                "we all agree", "everyone thinks", "obvious to all"
            ]):
                engineering_score += 0.2
        
        return min(1.0, engineering_score)
    
    def _calculate_keyword_density(self, messages: List[Message]) -> float:
        """Calculate keyword stuffing density."""
        if not messages:
            return 0.0
        
        keywords = [
            "innovative", "creative", "novel", "brilliant",
            "groundbreaking", "revolutionary", "paradigm"
        ]
        
        total_words = 0
        keyword_count = 0
        
        for msg in messages:
            words = msg.content.lower().split()
            total_words += len(words)
            keyword_count += sum(1 for word in words if word in keywords)
        
        if total_words == 0:
            return 0.0
        
        return keyword_count / total_words
    
    def _check_repetitive_patterns(self, messages: List[Message]) -> bool:
        """Check for repetitive message patterns."""
        if len(messages) < 3:
            return False
        
        # Check for similar message lengths
        lengths = [len(msg.content) for msg in messages]
        length_variance = np.var(lengths)
        
        # Low variance suggests repetitive patterns
        return length_variance < 100