"""Web Search Knowledge Cache for storing and retrieving web search results."""

import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import asyncio
from dataclasses import dataclass

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.api.types import EmbeddingFunction, Embeddings, Documents
except ImportError as e:
    raise ImportError(f"ChromaDB not installed: {e}")

try:
    from ..config.settings import get_settings
except ImportError:
    from config.settings import get_settings

try:
    from ..memory.knowledge_graph_module import KnowledgeGraphModule
except ImportError:
    from memory.knowledge_graph_module import KnowledgeGraphModule


@dataclass
class CachedWebResult:
    """Represents a cached web search result."""
    query: str
    answer: str
    query_type: str  # 'static', 'time_sensitive', 'current'
    domain: str  # 'weather', 'sports', 'facts', 'news', etc.
    timestamp: datetime
    expiry_hours: int
    confidence: float
    source_urls: List[str]
    character_id: str
    
    def is_expired(self) -> bool:
        """Check if this cached result has expired."""
        if self.query_type == 'static':
            return False  # Static information never expires
        
        expiry_time = self.timestamp + timedelta(hours=self.expiry_hours)
        return datetime.now() > expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'query': self.query,
            'answer': self.answer,
            'query_type': self.query_type,
            'domain': self.domain,
            'timestamp': self.timestamp.isoformat(),
            'expiry_hours': self.expiry_hours,
            'confidence': self.confidence,
            'source_urls': ','.join(self.source_urls),  # Convert list to string for ChromaDB
            'character_id': self.character_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedWebResult':
        """Create from dictionary."""
        # Convert source_urls back from string to list
        source_urls = data['source_urls']
        if isinstance(source_urls, str):
            source_urls = source_urls.split(',') if source_urls else []
        
        return cls(
            query=data['query'],
            answer=data['answer'],
            query_type=data['query_type'],
            domain=data['domain'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            expiry_hours=data['expiry_hours'],
            confidence=data['confidence'],
            source_urls=source_urls,
            character_id=data['character_id']
        )


class QueryEmbeddingFunction(EmbeddingFunction):
    """Embedding function optimized for web search queries."""
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for web search queries."""
        embeddings = []
        
        for query in input:
            # Extract semantic features from queries
            query_lower = query.lower()
            
            # Basic semantic features
            features = [
                len(query) / 100,  # Query length
                query_lower.count('what') * 2,  # Question type
                query_lower.count('how') * 2,   # How-to questions
                query_lower.count('when') * 1.5, # Time-based questions
                query_lower.count('where') * 1.5, # Location questions
                query_lower.count('why') * 1.5,  # Causal questions
                
                # Time sensitivity indicators
                query_lower.count('current') * 3,  # Current information
                query_lower.count('latest') * 3,   # Latest information
                query_lower.count('recent') * 2,   # Recent information
                query_lower.count('today') * 3,    # Today's information
                query_lower.count('now') * 3,      # Real-time information
                
                # Domain indicators
                query_lower.count('weather') * 2,   # Weather domain
                query_lower.count('temperature') * 2, # Weather domain
                query_lower.count('score') * 2,     # Sports domain
                query_lower.count('game') * 1.5,    # Sports/Entertainment
                query_lower.count('news') * 2,      # News domain
                query_lower.count('stock') * 2,     # Financial domain
                query_lower.count('price') * 1.5,   # Financial/Commerce
                
                # Static knowledge indicators
                query_lower.count('definition') * -2,  # Less time-sensitive
                query_lower.count('explain') * -1,     # Usually static
                query_lower.count('history') * -2,     # Historical facts
                query_lower.count('meaning') * -2,     # Definitions
            ]
            
            # Add word count and question mark features
            features.extend([
                len(query.split()),  # Word count
                1 if '?' in query else 0,  # Question mark presence
            ])
            
            # Pad to 384 dimensions for consistency
            embedding = features + [0.0] * (384 - len(features))
            embedding = embedding[:384]
            embeddings.append(embedding)
        
        return embeddings


class WebSearchCache:
    """
    Manages caching and retrieval of web search results using ChromaDB for similarity search.
    
    Provides intelligent caching that considers query similarity, time sensitivity,
    and domain-specific expiration rules.
    """
    
    # Time sensitivity patterns
    TIME_SENSITIVE_PATTERNS = {
        'current': ['current', 'now', 'right now', 'at the moment'],
        'latest': ['latest', 'most recent', 'newest', 'most up-to-date'],
        'today': ['today', 'this morning', 'this afternoon', 'this evening'],
        'real_time': ['live', 'real-time', 'real time', 'streaming'],
    }
    
    # Domain-specific expiration rules (in hours)
    DOMAIN_EXPIRY_RULES = {
        'weather': 6,      # Weather changes throughout the day
        'sports_live': 1,  # Live game scores change rapidly
        'sports_final': 168, # Final scores are permanent (1 week to be safe)
        'news': 24,        # News becomes outdated daily
        'stocks': 1,       # Stock prices change rapidly during market hours
        'traffic': 1,      # Traffic conditions change hourly
        'events': 48,      # Event information changes less frequently
        'facts': 8760,     # General facts last much longer (1 year)
        'definitions': 8760, # Definitions rarely change
        'procedures': 4320,  # How-to information (6 months)
        'history': 8760,     # Historical facts don't change
    }
    
    def __init__(self, character_id: str):
        """Initialize web search cache for a specific character."""
        self.character_id = character_id
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB for query similarity matching
        self.client = self._init_chromadb()
        self.collection_name = f"web_cache_{character_id}"
        self.collection = self._get_or_create_collection()
        
        # Initialize knowledge graph for additional storage
        self.knowledge_graph = KnowledgeGraphModule(character_id)
        
        self.logger.info(f"WebSearchCache initialized for character {character_id}")
    
    def _init_chromadb(self) -> chromadb.Client:
        """Initialize ChromaDB client using the same pattern as ExperienceModule."""
        try:
            persist_dir = self.settings.chroma_persist_directory
            
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir, mode=0o755, exist_ok=True)
                self.logger.info(f"Created ChromaDB directory: {persist_dir}")
            
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.logger.info(f"ChromaDB client initialized for web search cache")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB for web cache: {e}")
            self.logger.warning("Falling back to in-memory ChromaDB client")
            return chromadb.Client()
    
    def _get_or_create_collection(self):
        """Get or create the web search cache collection."""
        try:
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=QueryEmbeddingFunction()
            )
            self.logger.info(f"Retrieved existing web cache collection: {self.collection_name}")
            
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=QueryEmbeddingFunction(),
                metadata={"description": f"Web search cache for character {self.character_id}"}
            )
            self.logger.info(f"Created new web cache collection: {self.collection_name}")
        
        return collection
    
    def classify_query(self, query: str) -> Tuple[str, str, int]:
        """
        Classify a query to determine its type, domain, and expiration.
        
        Returns:
            Tuple of (query_type, domain, expiry_hours)
        """
        query_lower = query.lower()
        
        # Check for time sensitivity patterns
        query_type = 'static'  # Default to static
        for pattern_type, patterns in self.TIME_SENSITIVE_PATTERNS.items():
            if any(pattern in query_lower for pattern in patterns):
                query_type = 'time_sensitive'
                break
        
        # Determine domain
        domain = 'general'
        expiry_hours = self.DOMAIN_EXPIRY_RULES['facts']  # Default
        
        # Weather domain
        if any(term in query_lower for term in ['weather', 'temperature', 'rain', 'snow', 'sunny', 'cloudy']):
            domain = 'weather'
            expiry_hours = self.DOMAIN_EXPIRY_RULES['weather']
            query_type = 'time_sensitive'
        
        # Sports domain
        elif any(term in query_lower for term in ['score', 'game', 'match', 'sports', 'football', 'basketball', 'baseball']):
            if any(term in query_lower for term in ['final', 'ended', 'finished']):
                domain = 'sports_final'
                expiry_hours = self.DOMAIN_EXPIRY_RULES['sports_final']
            elif any(term in query_lower for term in ['live', 'current', 'now']):
                domain = 'sports_live'
                expiry_hours = self.DOMAIN_EXPIRY_RULES['sports_live']
                query_type = 'time_sensitive'
            else:
                domain = 'sports_final'
                expiry_hours = self.DOMAIN_EXPIRY_RULES['sports_final']
        
        # News domain
        elif any(term in query_lower for term in ['news', 'breaking', 'report', 'announcement']):
            domain = 'news'
            expiry_hours = self.DOMAIN_EXPIRY_RULES['news']
            query_type = 'time_sensitive'
        
        # Financial domain
        elif any(term in query_lower for term in ['stock', 'price', 'market', 'trading']):
            domain = 'stocks'
            expiry_hours = self.DOMAIN_EXPIRY_RULES['stocks']
            query_type = 'time_sensitive'
        
        # Definitions and facts
        elif any(term in query_lower for term in ['what is', 'define', 'definition', 'meaning', 'explain']):
            domain = 'definitions'
            expiry_hours = self.DOMAIN_EXPIRY_RULES['definitions']
            query_type = 'static'
        
        # How-to questions
        elif 'how to' in query_lower or 'how do' in query_lower:
            domain = 'procedures'
            expiry_hours = self.DOMAIN_EXPIRY_RULES['procedures']
            query_type = 'static'
        
        # Historical information
        elif any(term in query_lower for term in ['history', 'historical', 'past', 'ancient', 'old']):
            domain = 'history'
            expiry_hours = self.DOMAIN_EXPIRY_RULES['history']
            query_type = 'static'
        
        return query_type, domain, expiry_hours
    
    async def search_cache(self, query: str, similarity_threshold: float = 0.7) -> Optional[CachedWebResult]:
        """
        Search the cache for similar queries with valid (non-expired) results.
        
        Args:
            query: The search query
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            CachedWebResult if found and valid, None otherwise
        """
        try:
            # Query the collection for similar queries
            results = self.collection.query(
                query_texts=[query],
                n_results=5,  # Get top 5 similar queries
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return None
            
            # Check each result for validity
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                # Convert distance to similarity (ChromaDB uses distance, lower is better)
                similarity = 1.0 - distance
                
                if similarity < similarity_threshold:
                    continue
                
                # Reconstruct cached result from metadata
                try:
                    cached_result = CachedWebResult.from_dict(metadata)
                    
                    # Check if result is expired
                    if cached_result.is_expired():
                        self.logger.debug(f"Cached result expired for query: {cached_result.query}")
                        continue
                    
                    self.logger.info(f"Cache hit for query '{query}' -> '{cached_result.query}' (similarity: {similarity:.3f})")
                    return cached_result
                    
                except Exception as e:
                    self.logger.error(f"Error reconstructing cached result: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error searching web search cache: {e}")
            return None
    
    async def store_result(self, query: str, answer: str, source_urls: List[str], confidence: float = 0.8) -> bool:
        """
        Store a web search result in the cache.
        
        Args:
            query: The search query
            answer: The search result/answer
            source_urls: URLs of the sources
            confidence: Confidence in the result (0-1)
            
        Returns:
            True if successfully stored
        """
        try:
            # Classify the query
            query_type, domain, expiry_hours = self.classify_query(query)
            
            # Create cached result
            cached_result = CachedWebResult(
                query=query,
                answer=answer,
                query_type=query_type,
                domain=domain,
                timestamp=datetime.now(),
                expiry_hours=expiry_hours,
                confidence=confidence,
                source_urls=source_urls,
                character_id=self.character_id
            )
            
            # Generate unique ID
            result_id = str(uuid.uuid4())
            
            # Store in ChromaDB for similarity search
            self.collection.add(
                documents=[query],  # The query text for similarity matching
                metadatas=[cached_result.to_dict()],  # Full result data
                ids=[result_id]
            )
            
            # Also store in knowledge graph as a fact
            await self.knowledge_graph.store_fact(
                fact_text=f"Query: {query}. Answer: {answer}",
                source="web_search_cache",
                confidence=confidence,
                domain=domain,
                web_search_context={
                    'query': query,
                    'query_type': query_type,
                    'expiry_hours': expiry_hours,
                    'source_urls': source_urls
                }
            )
            
            self.logger.info(f"Stored web search result: {query} -> {domain} ({query_type}, expires in {expiry_hours}h)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing web search result: {e}")
            return False
    
    async def clean_expired_results(self) -> int:
        """
        Remove expired results from the cache.
        
        Returns:
            Number of results removed
        """
        try:
            # Get all results
            all_results = self.collection.get()
            
            if not all_results['ids']:
                return 0
            
            expired_ids = []
            metadatas = all_results.get('metadatas', [])
            
            for result_id, metadata in zip(all_results['ids'], metadatas):
                try:
                    cached_result = CachedWebResult.from_dict(metadata)
                    if cached_result.is_expired():
                        expired_ids.append(result_id)
                except Exception:
                    # If we can't parse it, remove it
                    expired_ids.append(result_id)
            
            if expired_ids:
                self.collection.delete(ids=expired_ids)
                self.logger.info(f"Cleaned {len(expired_ids)} expired web search cache entries")
            
            return len(expired_ids)
            
        except Exception as e:
            self.logger.error(f"Error cleaning expired cache results: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        try:
            all_results = self.collection.get()
            
            metadatas = all_results.get('metadatas', [])
            if not metadatas:
                return {
                    'total_entries': 0,
                    'expired_entries': 0,
                    'valid_entries': 0,
                    'domains': {},
                    'query_types': {}
                }
            
            domains = {}
            query_types = {}
            expired_count = 0
            
            for metadata in metadatas:
                try:
                    cached_result = CachedWebResult.from_dict(metadata)
                    
                    # Count domains
                    domains[cached_result.domain] = domains.get(cached_result.domain, 0) + 1
                    
                    # Count query types
                    query_types[cached_result.query_type] = query_types.get(cached_result.query_type, 0) + 1
                    
                    # Count expired
                    if cached_result.is_expired():
                        expired_count += 1
                        
                except Exception:
                    expired_count += 1  # If we can't parse it, consider it expired
            
            total_entries = len(metadatas)
            valid_entries = total_entries - expired_count
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_count,
                'valid_entries': valid_entries,
                'domains': domains,
                'query_types': query_types
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}