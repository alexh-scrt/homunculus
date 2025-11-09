"""
Topic Extraction for Arena Conversations

Adapted from the talks project to extract topics and detect when agents
are cycling through the same topics without meaningful progression.
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TopicExtractor:
    """Extracts topics from arena discussions and detects topic tensions"""
    
    def __init__(self):
        """Initialize with arena-specific topic lexicons"""
        
        # Business and startup topic lexicons
        self.topic_lexicons = {
            'ai_technology': [
                'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning',
                'neural networks', 'algorithms', 'automation', 'data', 'analytics',
                'llm', 'language models', 'computer vision', 'nlp', 'robotics'
            ],
            
            'energy': [
                'renewable energy', 'solar', 'wind', 'nuclear', 'clean energy',
                'oil', 'gas', 'fossil fuels', 'energy storage', 'grid',
                'power', 'electricity', 'battery', 'hydrogen', 'coal'
            ],
            
            'business_model': [
                'revenue', 'subscription', 'saas', 'platform', 'marketplace',
                'freemium', 'business model', 'monetization', 'pricing',
                'profit', 'margin', 'scalability', 'economics'
            ],
            
            'market': [
                'market', 'customers', 'demand', 'competition', 'industry',
                'sector', 'segment', 'target market', 'addressable market',
                'growth', 'expansion', 'penetration', 'adoption'
            ],
            
            'technology_infrastructure': [
                'infrastructure', 'cloud', 'servers', 'data centers', 'computing',
                'storage', 'bandwidth', 'network', 'blockchain', 'cyber security',
                'architecture', 'platform', 'api', 'software', 'hardware'
            ],
            
            'finance': [
                'investment', 'funding', 'venture capital', 'valuation',
                'ipo', 'acquisition', 'merger', 'capital', 'financing',
                'trillion dollar', 'billion', 'million', 'fundraising'
            ],
            
            'innovation': [
                'innovation', 'disruption', 'transformation', 'breakthrough',
                'revolutionary', 'novel', 'cutting edge', 'pioneering',
                'invention', 'discovery', 'advancement', 'progress'
            ],
            
            'execution': [
                'execution', 'implementation', 'strategy', 'operations',
                'scaling', 'growth', 'team', 'hiring', 'management',
                'leadership', 'culture', 'talent', 'resources'
            ],
            
            'risk': [
                'risk', 'challenge', 'threat', 'uncertainty', 'failure',
                'regulation', 'compliance', 'legal', 'policy', 'government',
                'regulatory', 'barriers', 'obstacles', 'downside'
            ],
            
            'sustainability': [
                'sustainable', 'sustainability', 'environmental', 'carbon',
                'emissions', 'climate', 'green', 'eco-friendly',
                'responsible', 'ethical', 'impact', 'esg'
            ]
        }
        
        # Common topic tensions in business discussions
        self.topic_tensions = [
            ('ai_technology', 'energy'),
            ('innovation', 'execution'),
            ('market', 'technology_infrastructure'),
            ('finance', 'sustainability'),
            ('business_model', 'market'),
            ('execution', 'risk'),
            ('innovation', 'regulation'),
            ('scalability', 'sustainability')
        ]
        
        # Compile regex patterns for efficiency
        self.topic_patterns = {}
        for topic, keywords in self.topic_lexicons.items():
            # Create pattern that matches any of the keywords
            pattern_str = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.topic_patterns[topic] = re.compile(pattern_str, re.IGNORECASE)
    
    def extract_topics(self, text: str) -> Set[str]:
        """
        Extract topics mentioned in the text
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of topic names found in the text
        """
        found_topics = set()
        
        for topic, pattern in self.topic_patterns.items():
            if pattern.search(text):
                found_topics.add(topic)
        
        logger.debug(f"Extracted topics: {found_topics}")
        return found_topics
    
    def get_topic_density(self, text: str) -> Dict[str, int]:
        """
        Get the density of topics (how many mentions per topic)
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping topic names to mention counts
        """
        topic_counts = defaultdict(int)
        
        for topic, pattern in self.topic_patterns.items():
            matches = pattern.findall(text)
            topic_counts[topic] = len(matches)
        
        return dict(topic_counts)
    
    def detect_tensions(self, current_text: str, recent_topics_history: List[Set[str]], window_size: int = 3) -> List[Tuple[str, str]]:
        """
        Detect topic tensions in the current discussion
        
        Args:
            current_text: Current message text
            recent_topics_history: List of recent topic sets
            window_size: How many recent turns to consider
            
        Returns:
            List of detected topic tensions (pairs)
        """
        current_topics = self.extract_topics(current_text)
        
        # Combine current with recent topics within window
        all_recent_topics = current_topics.copy()
        for topic_set in recent_topics_history[-window_size:]:
            all_recent_topics.update(topic_set)
        
        detected_tensions = []
        
        # Check for any tension pairs
        for topic1, topic2 in self.topic_tensions:
            if topic1 in all_recent_topics and topic2 in all_recent_topics:
                detected_tensions.append((topic1, topic2))
        
        # Also detect custom tensions based on current discussion
        detected_tensions.extend(self._detect_dynamic_tensions(all_recent_topics))
        
        logger.debug(f"Detected tensions: {detected_tensions}")
        return detected_tensions
    
    def _detect_dynamic_tensions(self, topics: Set[str]) -> List[Tuple[str, str]]:
        """Detect tensions that emerge dynamically from the current topic mix"""
        tensions = []
        topic_list = list(topics)
        
        # Look for potentially conflicting topic combinations
        conflict_indicators = [
            ('ai_technology', 'risk'),  # AI + risk concerns
            ('finance', 'execution'),   # Financial vs operational focus
            ('innovation', 'market'),   # Innovation vs market realities
            ('sustainability', 'finance')  # Sustainability vs profit
        ]
        
        for topic1, topic2 in conflict_indicators:
            if topic1 in topics and topic2 in topics:
                tensions.append((topic1, topic2))
        
        return tensions
    
    def analyze_topic_evolution(self, messages_history: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Analyze how topics have evolved over the conversation
        
        Args:
            messages_history: List of message dictionaries with 'content' and 'turn' keys
            
        Returns:
            Dictionary with topic evolution analysis
        """
        topic_timeline = []
        topic_frequency = defaultdict(int)
        topic_first_mentioned = {}
        topic_last_mentioned = {}
        
        for i, msg in enumerate(messages_history):
            content = msg.get('content', '')
            turn = msg.get('turn', i)
            
            topics = self.extract_topics(content)
            topic_timeline.append({
                'turn': turn,
                'topics': topics,
                'topic_count': len(topics)
            })
            
            for topic in topics:
                topic_frequency[topic] += 1
                if topic not in topic_first_mentioned:
                    topic_first_mentioned[topic] = turn
                topic_last_mentioned[topic] = turn
        
        # Identify recurring topics (mentioned across multiple turns)
        recurring_topics = {
            topic: count for topic, count in topic_frequency.items() 
            if count > 1
        }
        
        # Calculate topic persistence (how long topics stay active)
        topic_persistence = {
            topic: topic_last_mentioned[topic] - topic_first_mentioned[topic] + 1
            for topic in topic_frequency.keys()
        }
        
        return {
            'timeline': topic_timeline,
            'frequency': dict(topic_frequency),
            'recurring_topics': recurring_topics,
            'topic_persistence': topic_persistence,
            'total_unique_topics': len(topic_frequency),
            'dominant_topics': self._get_dominant_topics(topic_frequency, top_n=3)
        }
    
    def _get_dominant_topics(self, topic_frequency: Dict[str, int], top_n: int = 3) -> List[Tuple[str, int]]:
        """Get the most frequently mentioned topics"""
        return sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def detect_topic_cycles(self, recent_messages: List[str], cycle_threshold: int = 2) -> Dict[str, any]:
        """
        Detect if agents are cycling through the same topics repeatedly
        
        Args:
            recent_messages: List of recent message contents
            cycle_threshold: Number of repetitions to consider a cycle
            
        Returns:
            Dictionary with cycle detection results
        """
        if len(recent_messages) < cycle_threshold:
            return {'has_cycles': False, 'cycle_topics': [], 'cycle_count': 0}
        
        # Extract topics from each message
        message_topics = [self.extract_topics(msg) for msg in recent_messages]
        
        # Count topic appearances
        topic_appearances = defaultdict(int)
        for topics in message_topics:
            for topic in topics:
                topic_appearances[topic] += 1
        
        # Identify cycling topics
        cycling_topics = [
            topic for topic, count in topic_appearances.items() 
            if count >= cycle_threshold
        ]
        
        # Check for same topic combinations appearing repeatedly
        topic_combinations = []
        for topics in message_topics:
            if len(topics) > 1:
                sorted_topics = tuple(sorted(topics))
                topic_combinations.append(sorted_topics)
        
        combination_counts = defaultdict(int)
        for combo in topic_combinations:
            combination_counts[combo] += 1
        
        cycling_combinations = [
            combo for combo, count in combination_counts.items() 
            if count >= cycle_threshold
        ]
        
        max_cycle_count = max(topic_appearances.values()) if topic_appearances else 0
        
        return {
            'has_cycles': len(cycling_topics) > 0,
            'cycle_topics': cycling_topics,
            'cycle_count': max_cycle_count,
            'cycling_combinations': cycling_combinations,
            'analysis': {
                'total_cycling_topics': len(cycling_topics),
                'max_repetitions': max_cycle_count,
                'diverse_topics': len([t for t, c in topic_appearances.items() if c == 1])
            }
        }
    
    def get_topic_context(self, topics: Set[str]) -> Dict[str, any]:
        """
        Get contextual information about a set of topics
        
        Args:
            topics: Set of topics to analyze
            
        Returns:
            Dictionary with topic context information
        """
        # Categorize topics by domain
        domain_mapping = {
            'technology': ['ai_technology', 'technology_infrastructure'],
            'business': ['business_model', 'market', 'finance'],
            'operations': ['execution', 'innovation'],
            'external': ['risk', 'sustainability', 'energy']
        }
        
        topic_domains = defaultdict(list)
        for topic in topics:
            for domain, domain_topics in domain_mapping.items():
                if topic in domain_topics:
                    topic_domains[domain].append(topic)
        
        # Assess topic coherence (are topics related or scattered?)
        coherence_score = self._calculate_topic_coherence(topics)
        
        return {
            'domains': dict(topic_domains),
            'domain_count': len(topic_domains),
            'coherence_score': coherence_score,
            'focus_assessment': self._assess_focus(topics, topic_domains),
            'suggested_directions': self._suggest_topic_directions(topics)
        }
    
    def _calculate_topic_coherence(self, topics: Set[str]) -> float:
        """Calculate how coherent/related the topics are"""
        if len(topics) <= 1:
            return 1.0
        
        # Check how many known tensions exist in the topic set
        tension_count = 0
        topic_list = list(topics)
        
        for i, topic1 in enumerate(topic_list):
            for topic2 in topic_list[i+1:]:
                if (topic1, topic2) in self.topic_tensions or (topic2, topic1) in self.topic_tensions:
                    tension_count += 1
        
        # More tensions = less coherence
        max_possible_tensions = len(topics) * (len(topics) - 1) // 2
        coherence = 1.0 - (tension_count / max_possible_tensions if max_possible_tensions > 0 else 0)
        
        return coherence
    
    def _assess_focus(self, topics: Set[str], topic_domains: Dict[str, List[str]]) -> str:
        """Assess the focus level of the current topics"""
        if len(topics) <= 2:
            return "highly_focused"
        elif len(topic_domains) <= 2:
            return "moderately_focused"
        elif len(topic_domains) <= 3:
            return "somewhat_scattered"
        else:
            return "highly_scattered"
    
    def _suggest_topic_directions(self, topics: Set[str]) -> List[str]:
        """Suggest directions for topic development"""
        suggestions = []
        
        # If too many topics, suggest consolidation
        if len(topics) > 4:
            suggestions.append("consolidate_focus")
        
        # If topics show tensions, suggest resolution
        for topic1, topic2 in self.topic_tensions:
            if topic1 in topics and topic2 in topics:
                suggestions.append(f"resolve_tension_{topic1}_{topic2}")
        
        # If missing key business elements, suggest addition
        business_essentials = ['market', 'business_model', 'execution']
        missing_essentials = [topic for topic in business_essentials if topic not in topics]
        if missing_essentials:
            suggestions.extend([f"add_{topic}" for topic in missing_essentials])
        
        return suggestions