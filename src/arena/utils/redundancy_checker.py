"""
Redundancy Checker for Arena Conversations

Adapted from the talks project to detect semantic similarity and prevent 
agents from repeating the same content in arena discussions.
"""

from typing import List, Optional
import logging
import re

logger = logging.getLogger(__name__)


class RedundancyChecker:
    """Checks for semantic similarity to detect redundant content in arena discussions"""
    
    def __init__(self, similarity_threshold: float = 0.82, lookback_window: int = 5):
        """
        Initialize redundancy checker
        
        Args:
            similarity_threshold: Threshold above which content is considered redundant (lowered from talks project to be more aggressive)
            lookback_window: Number of recent messages to compare against
        """
        self.threshold = similarity_threshold
        self.lookback_window = lookback_window
        self.model = None
        self.np = None
        self._initialized = False
    
    def _initialize_model(self):
        """Lazy initialization of sentence transformer model"""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.np = np
            self._initialized = True
            logger.debug("Sentence transformer model initialized successfully")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback similarity check")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize sentence transformer: {e}")
            self._initialized = True
    
    def is_redundant(self, candidate: str, recent_texts: List[str], window_size: Optional[int] = None) -> bool:
        """
        Check if candidate response is too similar to recent texts
        
        Args:
            candidate: New response to check
            recent_texts: List of recent responses to compare against
            window_size: How many recent texts to consider (uses instance default if None)
            
        Returns:
            True if candidate is redundant, False otherwise
        """
        if not recent_texts:
            return False
        
        # Use only the most recent texts within window
        window = window_size or self.lookback_window
        comparison_texts = recent_texts[-window:] if len(recent_texts) > window else recent_texts
        
        self._initialize_model()
        
        if self.model is None:
            return self._fallback_similarity_check(candidate, comparison_texts)
        
        try:
            # Encode all texts
            embeddings = self.model.encode([candidate] + comparison_texts)
            candidate_emb = embeddings[0]
            recent_embs = embeddings[1:]
            
            # Compute cosine similarities
            similarities = self.np.dot(recent_embs, candidate_emb) / (
                self.np.linalg.norm(recent_embs, axis=1) * self.np.linalg.norm(candidate_emb)
            )
            
            max_similarity = float(self.np.max(similarities))
            is_redundant = max_similarity > self.threshold
            
            logger.debug(f"Max similarity: {max_similarity:.3f}, threshold: {self.threshold}, redundant: {is_redundant}")
            return is_redundant
            
        except Exception as e:
            logger.error(f"Error in semantic similarity check: {e}")
            return self._fallback_similarity_check(candidate, comparison_texts)
    
    def _fallback_similarity_check(self, candidate: str, recent_texts: List[str]) -> bool:
        """
        Fallback similarity check using word overlap when embeddings unavailable
        
        Args:
            candidate: Text to check
            recent_texts: Recent texts to compare against
            
        Returns:
            True if similarity above threshold based on word overlap
        """
        candidate_words = set(self._normalize_text(candidate).split())
        
        max_similarity = 0.0
        for text in recent_texts:
            text_words = set(self._normalize_text(text).split())
            
            if not text_words:
                continue
                
            # Jaccard similarity
            intersection = len(candidate_words & text_words)
            union = len(candidate_words | text_words)
            similarity = intersection / union if union > 0 else 0.0
            max_similarity = max(max_similarity, similarity)
        
        # Use lower threshold for word-based similarity
        fallback_threshold = self.threshold * 0.7
        is_redundant = max_similarity > fallback_threshold
        
        logger.debug(f"Fallback word overlap similarity: {max_similarity:.3f}, redundant: {is_redundant}")
        return is_redundant
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison - enhanced to catch paraphrasing patterns"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove common conversation starters and filler phrases that agents use to paraphrase
        filler_patterns = [
            # Repetitive conversation openers
            r'^(as we continue to explore the concept of)',
            r'^(as we (continue to )?explore|as we (continue to )?delve|as we (continue to )?discuss)',
            r'^(i\'d like to delve deeper into|let me delve deeper into)',
            r'^(building on the points? made by|building on.*insights?)',
            r'^(to move (the|this) discussion forward)',
            r'^(i\'d like to (build|add|emphasize|highlight))',
            r'^(let me (build|add|emphasize|highlight))',
            r'^(i propose (that )?we)',
            
            # Formulaic reference patterns
            r'(building on (jobs|gates|musk|bezos))',
            r'(building on the points made by)',
            r'(as.*noted by.*(jobs|gates|musk|bezos))',
            r'(jobs|gates|musk|bezos)\'?s?\s*(insight|idea|point|suggestion|perspective)',
            r'(the points? made by (jobs|gates|musk|bezos))',
            
            # Research citation fillers
            r'^(the research (suggests|highlights|shows|indicates))',
            r'^(according to recent (research|studies|reports))',
            r'^(recent research (suggests|shows|indicates))',
            r'^(studies (show|indicate|suggest))',
            
            # Business conversation fillers
            r'^(notably,|furthermore,|moreover,|additionally,)',
            r'^(it\'s (clear|crucial|essential|important) that)',
            r'^(this (approach|strategy|concept) not only.*but also)',
            r'(by (exploring|combining|leveraging|utilizing))',
            r'(with (companies like|leaders like|investors like))',
            
            # Repetitive trillion-dollar company references
            r'(creating the next trillion[- ]dollar company)',
            r'(concept of.*trillion[- ]dollar)',
            r'(potential.*trillion[- ]dollar)',
            
            # Generic bridging phrases
            r'(it\'s clear that)',
            r'(this clearly shows)',
            r'(we can see that)',
            r'(this demonstrates)',
            r'(this indicates)',
        ]
        
        for pattern in filler_patterns:
            text = re.sub(pattern, '', text)
        
        # Remove agent names
        text = re.sub(r'\b(jobs|gates|musk|bezos|steve|bill|elon|jeff)\b', '', text)
        
        # Remove common business buzzwords that don't add semantic meaning
        buzzwords = [
            r'\b(significant|substantial|tremendous|compelling)\s*(growth|potential|opportunity|prospects)\b',
            r'\b(emerging\s*sectors?)\b',
            r'\b(innovative\s*thinking)\b',
            r'\b(strategic\s*execution)\b',
            r'\b(prime\s*targets?)\b',
            r'\b(long[- ]term\s*returns?)\b',
            
            # Additional filler business terms
            r'\b(leveraging ai to optimize)\b',
            r'\b(significant impact)\b',
            r'\b(lasting impact on our planet)\b',
            r'\b(unlock the full potential)\b',
            r'\b(scalability,?\s*sustainability,?\s*(and\s*)?social responsibility)\b',
            r'\b(truly trillion[- ]dollar company)\b',
            r'\b(can have a significant impact)\b',
            r'\b(improve efficiency)\b',
            r'\b(transform these industries)\b',
        ]
        
        for pattern in buzzwords:
            text = re.sub(pattern, '', text)
        
        # Remove punctuation and normalize whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def get_similarity_details(self, candidate: str, recent_texts: List[str]) -> dict:
        """
        Get detailed similarity information for debugging
        
        Args:
            candidate: Text to check
            recent_texts: Recent texts to compare against
            
        Returns:
            Dictionary with similarity details
        """
        if not recent_texts:
            return {"max_similarity": 0.0, "similarities": [], "redundant": False}
        
        self._initialize_model()
        
        try:
            if self.model is not None:
                embeddings = self.model.encode([candidate] + recent_texts)
                candidate_emb = embeddings[0]
                recent_embs = embeddings[1:]
                
                similarities = self.np.dot(recent_embs, candidate_emb) / (
                    self.np.linalg.norm(recent_embs, axis=1) * self.np.linalg.norm(candidate_emb)
                )
                similarities_list = [float(sim) for sim in similarities]
            else:
                # Fallback to word overlap
                similarities_list = []
                candidate_words = set(self._normalize_text(candidate).split())
                
                for text in recent_texts:
                    text_words = set(self._normalize_text(text).split())
                    if not text_words:
                        similarities_list.append(0.0)
                        continue
                    
                    intersection = len(candidate_words & text_words)
                    union = len(candidate_words | text_words)
                    similarity = intersection / union if union > 0 else 0.0
                    similarities_list.append(similarity)
            
            max_similarity = max(similarities_list) if similarities_list else 0.0
            is_redundant = max_similarity > self.threshold
            
            return {
                "max_similarity": max_similarity,
                "similarities": similarities_list,
                "redundant": is_redundant,
                "threshold": self.threshold,
                "method": "embeddings" if self.model is not None else "word_overlap"
            }
            
        except Exception as e:
            logger.error(f"Error getting similarity details: {e}")
            return {"max_similarity": 0.0, "similarities": [], "redundant": False, "error": str(e)}


class ConversationHistoryTracker:
    """Tracks conversation history for redundancy checking"""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize history tracker
        
        Args:
            max_history: Maximum number of messages to keep in history
        """
        self.max_history = max_history
        self.history = []
    
    def add_message(self, speaker: str, content: str, turn: int):
        """Add a new message to history"""
        self.history.append({
            "speaker": speaker,
            "content": content,
            "turn": turn
        })
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_recent_messages(self, exclude_speaker: Optional[str] = None, count: int = 5) -> List[str]:
        """
        Get recent message contents for comparison
        
        Args:
            exclude_speaker: Don't include messages from this speaker
            count: Number of recent messages to return
            
        Returns:
            List of recent message contents
        """
        recent = []
        for msg in reversed(self.history):
            if exclude_speaker and msg["speaker"] == exclude_speaker:
                continue
            recent.append(msg["content"])
            if len(recent) >= count:
                break
        
        return recent
    
    def get_speaker_recent_messages(self, speaker: str, count: int = 3) -> List[str]:
        """Get recent messages from a specific speaker"""
        speaker_messages = []
        for msg in reversed(self.history):
            if msg["speaker"] == speaker:
                speaker_messages.append(msg["content"])
                if len(speaker_messages) >= count:
                    break
        
        return speaker_messages
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()