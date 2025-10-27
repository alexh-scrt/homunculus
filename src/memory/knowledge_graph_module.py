"""Knowledge graph module for relationship and fact storage using Neo4j."""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import asyncio
import json

try:
    from neo4j import GraphDatabase, basic_auth
    from neo4j.exceptions import ServiceUnavailable, AuthError
except ImportError as e:
    raise ImportError(f"Neo4j driver not installed: {e}")

try:
    from ..config.settings import get_settings
except ImportError:
    from config.settings import get_settings


class KnowledgeGraphModule:
    """
    Manages knowledge graph storage and retrieval using Neo4j.
    
    Stores character knowledge as interconnected facts, relationships,
    and web search results that can be queried for relevant context.
    """
    
    def __init__(self, character_id: str):
        """Initialize knowledge graph module for a specific character."""
        self.character_id = character_id
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Neo4j driver
        self.driver = self._init_neo4j()
        
        # Create constraints and indexes
        self._setup_database()
        
        self.logger.info(f"KnowledgeGraphModule initialized for character {character_id}")
    
    def _init_neo4j(self):
        """Initialize Neo4j driver with appropriate settings."""
        try:
            auth = self.settings.get_neo4j_auth()
            driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=auth,
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,
                connection_acquisition_timeout=2 * 60,  # 2 minutes
            )
            
            # Test connection
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            self.logger.info(f"Neo4j driver initialized: {self.settings.neo4j_uri}")
            return driver
            
        except (ServiceUnavailable, AuthError) as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            self.logger.warning("Knowledge graph functionality will be limited")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error initializing Neo4j: {e}")
            return None
    
    def _setup_database(self):
        """Create necessary constraints and indexes."""
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT character_id_unique IF NOT EXISTS FOR (c:Character) REQUIRE c.character_id IS UNIQUE",
                    "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE",
                    "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
                    "CREATE CONSTRAINT goal_id_unique IF NOT EXISTS FOR (g:Goal) REQUIRE g.goal_id IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        # Constraint might already exist
                        self.logger.debug(f"Constraint creation skipped: {e}")
                
                # Create indexes for better query performance
                indexes = [
                    "CREATE INDEX fact_timestamp IF NOT EXISTS FOR (f:Fact) ON (f.timestamp)",
                    "CREATE INDEX fact_character IF NOT EXISTS FOR (f:Fact) ON (f.character_id)",
                    "CREATE INDEX concept_domain IF NOT EXISTS FOR (c:Concept) ON (c.domain)",
                    "CREATE INDEX web_search_query IF NOT EXISTS FOR (w:WebSearchResult) ON (w.query)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        self.logger.debug(f"Index creation skipped: {e}")
                
                self.logger.debug("Database setup completed")
                
        except Exception as e:
            self.logger.error(f"Failed to setup database: {e}")
    
    async def store_fact(
        self,
        fact_text: str,
        source: str,
        confidence: float = 0.8,
        domain: Optional[str] = None,
        related_concepts: Optional[List[str]] = None,
        web_search_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a fact in the knowledge graph.
        
        Args:
            fact_text: The factual information
            source: Where this fact came from ('conversation', 'web_search', 'inference')
            confidence: Confidence level in this fact (0.0 to 1.0)
            domain: Domain/category of the fact
            related_concepts: List of concepts this fact relates to
            web_search_context: Context from web search if applicable
            
        Returns:
            str: Fact ID if successful, empty string if failed
        """
        if not self.driver:
            return ""
        
        try:
            fact_id = f"fact_{self.character_id}_{int(datetime.now().timestamp() * 1000)}"
            
            with self.driver.session() as session:
                # Create the fact node
                fact_query = """
                MERGE (c:Character {character_id: $character_id})
                CREATE (f:Fact {
                    fact_id: $fact_id,
                    character_id: $character_id,
                    text: $fact_text,
                    source: $source,
                    confidence: $confidence,
                    domain: $domain,
                    timestamp: $timestamp,
                    retrieval_count: 0,
                    last_retrieved: null
                })
                CREATE (c)-[:KNOWS]->(f)
                RETURN f.fact_id as fact_id
                """
                
                result = session.run(fact_query, {
                    "character_id": self.character_id,
                    "fact_id": fact_id,
                    "fact_text": fact_text,
                    "source": source,
                    "confidence": confidence,
                    "domain": domain or "general",
                    "timestamp": datetime.now().isoformat()
                })
                
                created_fact_id = result.single()["fact_id"]
                
                # Link to related concepts
                if related_concepts:
                    for concept in related_concepts:
                        await self._link_fact_to_concept(fact_id, concept, domain)
                
                # Store web search context if provided
                if web_search_context:
                    await self._store_web_search_context(fact_id, web_search_context)
                
                self.logger.debug(f"Stored fact {fact_id}: {fact_text[:50]}...")
                return created_fact_id
            
        except Exception as e:
            self.logger.error(f"Failed to store fact: {e}")
            return ""
    
    async def _link_fact_to_concept(self, fact_id: str, concept_name: str, domain: Optional[str] = None):
        """Link a fact to a concept, creating the concept if it doesn't exist."""
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (f:Fact {fact_id: $fact_id})
                MERGE (c:Concept {name: $concept_name})
                ON CREATE SET c.domain = $domain, c.created_at = $timestamp
                MERGE (f)-[:RELATES_TO]->(c)
                """
                
                session.run(query, {
                    "fact_id": fact_id,
                    "concept_name": concept_name.lower(),
                    "domain": domain or "general",
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Failed to link fact to concept: {e}")
    
    async def _store_web_search_context(self, fact_id: str, web_context: Dict[str, Any]):
        """Store web search context for a fact."""
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (f:Fact {fact_id: $fact_id})
                CREATE (w:WebSearchResult {
                    search_id: $search_id,
                    query: $query,
                    url: $url,
                    title: $title,
                    snippet: $snippet,
                    timestamp: $timestamp
                })
                CREATE (f)-[:SOURCED_FROM]->(w)
                """
                
                for i, result in enumerate(web_context.get("results", [])):
                    search_id = f"search_{fact_id}_{i}"
                    session.run(query, {
                        "fact_id": fact_id,
                        "search_id": search_id,
                        "query": web_context.get("query", ""),
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "snippet": result.get("content", "")[:500],  # Limit snippet length
                        "timestamp": datetime.now().isoformat()
                    })
                
        except Exception as e:
            self.logger.error(f"Failed to store web search context: {e}")
    
    async def retrieve_related_facts(
        self,
        concepts: List[str],
        limit: int = 10,
        min_confidence: float = 0.5,
        domains: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve facts related to given concepts.
        
        Args:
            concepts: List of concept names to search for
            limit: Maximum number of facts to return
            min_confidence: Minimum confidence threshold
            domains: Optional list of domains to filter by
            
        Returns:
            List of fact dictionaries with metadata
        """
        if not self.driver or not concepts:
            return []
        
        try:
            with self.driver.session() as session:
                # Validate concepts list
                valid_concepts = [concept.strip() for concept in concepts if concept and concept.strip()]
                if not valid_concepts:
                    self.logger.debug("No valid concepts provided for fact retrieval")
                    return []
                
                # Build the query dynamically with proper escaping
                # Use parameterized queries to prevent injection
                concept_params = {}
                concept_conditions = []
                for i, concept in enumerate(valid_concepts):
                    param_name = f"concept_{i}"
                    concept_params[param_name] = concept.lower()
                    concept_conditions.append(f"c.name CONTAINS ${param_name}")
                
                concept_condition_str = " OR ".join(concept_conditions)
                
                domain_condition = ""
                domain_params = {}
                if domains:
                    domain_params["domains"] = domains
                    domain_condition = "AND f.domain IN $domains"
                
                # Build the query string avoiding variable name conflicts
                query = f"""
                MATCH (f:Fact)-[:RELATES_TO]->(c:Concept)
                WHERE f.character_id = $character_id
                AND f.confidence >= $min_confidence
                AND ({concept_condition_str})
                {domain_condition}
                WITH f, COUNT(DISTINCT c) AS concept_matches
                ORDER BY concept_matches DESC, f.confidence DESC, f.timestamp DESC
                LIMIT $limit
                
                OPTIONAL MATCH (f)-[:SOURCED_FROM]->(w:WebSearchResult)
                RETURN f.fact_id as fact_id, f.text as text, f.source as source,
                       f.confidence as confidence, f.domain as domain,
                       f.timestamp as timestamp, f.retrieval_count as retrieval_count,
                       concept_matches,
                       COLLECT(DISTINCT {{query: w.query, url: w.url, title: w.title}}) as web_sources
                """
                
                # Merge all parameters
                query_params = {
                    "character_id": self.character_id,
                    "min_confidence": min_confidence,
                    "limit": limit
                }
                query_params.update(concept_params)
                query_params.update(domain_params)
                
                result = session.run(query, query_params)
                
                facts = []
                for record in result:
                    fact = {
                        "fact_id": record["fact_id"],
                        "text": record["text"],
                        "source": record["source"],
                        "confidence": record["confidence"],
                        "domain": record["domain"],
                        "timestamp": record["timestamp"],
                        "retrieval_count": record["retrieval_count"],
                        "concept_matches": record["concept_matches"],
                        "web_sources": [ws for ws in record["web_sources"] if ws["url"]]
                    }
                    facts.append(fact)
                
                # Update retrieval statistics
                for fact in facts:
                    await self._update_fact_retrieval_stats(fact["fact_id"])
                
                self.logger.debug(f"Retrieved {len(facts)} facts for concepts: {concepts}")
                return facts
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve related facts: {e}")
            import traceback
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            return []
    
    async def _update_fact_retrieval_stats(self, fact_id: str):
        """Update retrieval statistics for a fact."""
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (f:Fact {fact_id: $fact_id})
                SET f.retrieval_count = COALESCE(f.retrieval_count, 0) + 1,
                    f.last_retrieved = $timestamp
                """
                
                session.run(query, {
                    "fact_id": fact_id,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Failed to update fact retrieval stats: {e}")
    
    async def store_goal_progress(
        self,
        goal_id: str,
        goal_description: str,
        progress: float,
        related_facts: Optional[List[str]] = None
    ) -> bool:
        """
        Store or update goal progress in the knowledge graph.
        
        Args:
            goal_id: Unique identifier for the goal
            goal_description: Description of the goal
            progress: Progress towards goal (0.0 to 1.0)
            related_facts: List of fact IDs that relate to this goal
            
        Returns:
            bool: Success status
        """
        if not self.driver:
            return False
        
        try:
            with self.driver.session() as session:
                # Create or update goal
                goal_query = """
                MERGE (c:Character {character_id: $character_id})
                MERGE (g:Goal {goal_id: $goal_id})
                ON CREATE SET g.description = $description,
                              g.created_at = $timestamp,
                              g.character_id = $character_id
                SET g.progress = $progress,
                    g.updated_at = $timestamp
                MERGE (c)-[:HAS_GOAL]->(g)
                """
                
                session.run(goal_query, {
                    "character_id": self.character_id,
                    "goal_id": goal_id,
                    "description": goal_description,
                    "progress": progress,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Link related facts
                if related_facts:
                    for fact_id in related_facts:
                        link_query = """
                        MATCH (g:Goal {goal_id: $goal_id})
                        MATCH (f:Fact {fact_id: $fact_id})
                        MERGE (g)-[:INFORMED_BY]->(f)
                        """
                        
                        session.run(link_query, {
                            "goal_id": goal_id,
                            "fact_id": fact_id
                        })
                
                self.logger.debug(f"Stored goal progress: {goal_id} at {progress:.2f}")
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to store goal progress: {e}")
            return False
    
    async def get_goal_related_knowledge(self, goal_id: str) -> Dict[str, Any]:
        """Get all knowledge related to a specific goal."""
        if not self.driver:
            return {}
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (g:Goal {goal_id: $goal_id})-[:INFORMED_BY]->(f:Fact)
                OPTIONAL MATCH (f)-[:RELATES_TO]->(c:Concept)
                OPTIONAL MATCH (f)-[:SOURCED_FROM]->(w:WebSearchResult)
                
                RETURN g.description as goal_description, g.progress as progress,
                       COLLECT(DISTINCT {
                           fact_id: f.fact_id,
                           text: f.text,
                           confidence: f.confidence,
                           domain: f.domain
                       }) as facts,
                       COLLECT(DISTINCT c.name) as concepts,
                       COLLECT(DISTINCT {query: w.query, title: w.title, url: w.url}) as web_sources
                """
                
                result = session.run(query, {"goal_id": goal_id})
                record = result.single()
                
                if record:
                    return {
                        "goal_id": goal_id,
                        "description": record["goal_description"],
                        "progress": record["progress"],
                        "related_facts": record["facts"],
                        "related_concepts": [c for c in record["concepts"] if c],
                        "web_sources": [ws for ws in record["web_sources"] if ws["url"]]
                    }
                else:
                    return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get goal-related knowledge: {e}")
            return {}
    
    async def find_concept_connections(
        self,
        concept_name: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Find concepts connected to the given concept.
        
        Args:
            concept_name: Name of the concept to start from
            max_depth: Maximum relationship depth to explore
            
        Returns:
            Dictionary with connected concepts and their relationships
        """
        if not self.driver:
            return {}
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH path = (start:Concept {name: $concept_name})-[:RELATES_TO*1..%d]-(connected:Concept)
                WHERE ALL(node IN nodes(path) WHERE node.name IS NOT NULL)
                WITH connected, LENGTH(path) as distance
                ORDER BY distance, connected.name
                RETURN DISTINCT connected.name as concept, distance
                LIMIT 20
                """ % max_depth
                
                result = session.run(query, {"concept_name": concept_name.lower()})
                
                connections = {}
                for record in result:
                    distance = record["distance"]
                    if distance not in connections:
                        connections[distance] = []
                    connections[distance].append(record["concept"])
                
                return {
                    "source_concept": concept_name,
                    "connections": connections,
                    "total_connected": sum(len(concepts) for concepts in connections.values())
                }
            
        except Exception as e:
            self.logger.error(f"Failed to find concept connections: {e}")
            return {}
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the character's knowledge graph."""
        if not self.driver:
            return {"error": "Neo4j not available"}
        
        try:
            with self.driver.session() as session:
                stats_query = """
                MATCH (f:Fact {character_id: $character_id})
                OPTIONAL MATCH (f)-[:RELATES_TO]->(c:Concept)
                OPTIONAL MATCH (g:Goal {character_id: $character_id})
                OPTIONAL MATCH (w:WebSearchResult)<-[:SOURCED_FROM]-(f)
                
                RETURN 
                    COUNT(DISTINCT f) as total_facts,
                    COUNT(DISTINCT c) as total_concepts,
                    COUNT(DISTINCT g) as total_goals,
                    COUNT(DISTINCT w) as web_sources,
                    AVG(f.confidence) as avg_confidence,
                    COLLECT(DISTINCT f.domain) as domains,
                    COLLECT(DISTINCT f.source) as sources
                """
                
                result = session.run(stats_query, {"character_id": self.character_id})
                record = result.single()
                
                if record:
                    return {
                        "character_id": self.character_id,
                        "total_facts": record["total_facts"],
                        "total_concepts": record["total_concepts"],
                        "total_goals": record["total_goals"],
                        "web_sources": record["web_sources"],
                        "average_confidence": round(record["avg_confidence"] or 0, 2),
                        "domains": [d for d in record["domains"] if d],
                        "sources": [s for s in record["sources"] if s]
                    }
                else:
                    return {
                        "character_id": self.character_id,
                        "total_facts": 0,
                        "total_concepts": 0,
                        "total_goals": 0,
                        "web_sources": 0,
                        "average_confidence": 0,
                        "domains": [],
                        "sources": []
                    }
            
        except Exception as e:
            self.logger.error(f"Failed to get knowledge stats: {e}")
            return {"error": str(e)}
    
    async def search_cached_web_knowledge(
        self,
        query: str,
        max_age_hours: int = 24,
        min_confidence: float = 0.6,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for cached web search knowledge that might answer the query.
        
        Args:
            query: The search query
            max_age_hours: Maximum age of cached results in hours
            min_confidence: Minimum confidence level for results
            limit: Maximum number of results to return
            
        Returns:
            List of relevant cached web search results
        """
        if not self.driver:
            return []
        
        try:
            # Calculate cutoff timestamp
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cutoff_iso = cutoff_time.isoformat()
            
            with self.driver.session() as session:
                query_cypher = """
                MATCH (f:Fact)-[:SOURCED_FROM]->(w:WebSearchResult)
                WHERE f.character_id = $character_id
                AND f.confidence >= $min_confidence
                AND w.timestamp >= $cutoff_time
                AND (
                    toLower(w.query) CONTAINS toLower($query)
                    OR toLower(f.text) CONTAINS toLower($query)
                    OR toLower(w.title) CONTAINS toLower($query)
                    OR toLower(w.snippet) CONTAINS toLower($query)
                )
                RETURN f.fact_id as fact_id, f.text as fact_text, f.confidence as confidence,
                       w.query as original_query, w.title as title, w.snippet as snippet,
                       w.url as url, w.timestamp as timestamp
                ORDER BY f.confidence DESC, w.timestamp DESC
                LIMIT $limit
                """
                
                result = session.run(query_cypher, {
                    "character_id": self.character_id,
                    "query": query,
                    "min_confidence": min_confidence,
                    "cutoff_time": cutoff_iso,
                    "limit": limit
                })
                
                cached_results = []
                for record in result:
                    cached_results.append({
                        "fact_id": record["fact_id"],
                        "fact_text": record["fact_text"],
                        "confidence": record["confidence"],
                        "original_query": record["original_query"],
                        "title": record["title"],
                        "snippet": record["snippet"],
                        "url": record["url"],
                        "timestamp": record["timestamp"]
                    })
                
                if cached_results:
                    self.logger.info(f"Found {len(cached_results)} cached web knowledge results for query: {query}")
                
                return cached_results
                
        except Exception as e:
            self.logger.error(f"Error searching cached web knowledge: {e}")
            return []
    
    async def store_web_search_knowledge(
        self,
        query: str,
        answer: str,
        source_urls: List[str],
        query_type: str = "static",
        domain: str = "general",
        expiry_hours: int = 8760,
        confidence: float = 0.8
    ) -> str:
        """
        Store web search knowledge with enhanced metadata for caching.
        
        Args:
            query: The search query
            answer: The extracted answer/knowledge
            source_urls: URLs that provided this information
            query_type: Type of query ('static', 'time_sensitive', 'current')
            domain: Domain category ('weather', 'sports', 'facts', etc.)
            expiry_hours: Hours until this knowledge expires
            confidence: Confidence in this knowledge (0-1)
            
        Returns:
            Fact ID if successful, empty string if failed
        """
        try:
            # Create a comprehensive fact text
            fact_text = f"Query: '{query}' Answer: {answer}"
            
            # Create the fact with enhanced web search context
            web_search_context = {
                "query": query,
                "query_type": query_type,
                "domain": domain,
                "expiry_hours": expiry_hours,
                "results": [{"url": url, "title": "", "content": answer} for url in source_urls]
            }
            
            fact_id = await self.store_fact(
                fact_text=fact_text,
                source="web_search_cache",
                confidence=confidence,
                domain=domain,
                related_concepts=await extract_concepts_from_text(f"{query} {answer}"),
                web_search_context=web_search_context
            )
            
            if fact_id:
                # Also create a special cached knowledge node for faster retrieval
                await self._create_cached_knowledge_node(
                    fact_id, query, answer, query_type, domain, expiry_hours, source_urls
                )
            
            return fact_id
            
        except Exception as e:
            self.logger.error(f"Error storing web search knowledge: {e}")
            return ""
    
    async def _create_cached_knowledge_node(
        self,
        fact_id: str,
        query: str,
        answer: str,
        query_type: str,
        domain: str,
        expiry_hours: int,
        source_urls: List[str]
    ):
        """Create a specialized cached knowledge node for faster retrieval."""
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                cache_query = """
                MATCH (f:Fact {fact_id: $fact_id})
                CREATE (c:CachedKnowledge {
                    cache_id: $cache_id,
                    character_id: $character_id,
                    query: $query,
                    answer: $answer,
                    query_type: $query_type,
                    domain: $domain,
                    created_at: $timestamp,
                    expires_at: $expires_at,
                    source_urls: $source_urls,
                    access_count: 0,
                    last_accessed: null
                })
                CREATE (f)-[:CACHED_AS]->(c)
                """
                
                cache_id = f"cache_{fact_id}_{int(datetime.now().timestamp())}"
                expires_at = (datetime.now() + timedelta(hours=expiry_hours)).isoformat()
                
                session.run(cache_query, {
                    "fact_id": fact_id,
                    "cache_id": cache_id,
                    "character_id": self.character_id,
                    "query": query,
                    "answer": answer,
                    "query_type": query_type,
                    "domain": domain,
                    "timestamp": datetime.now().isoformat(),
                    "expires_at": expires_at,
                    "source_urls": source_urls
                })
                
                self.logger.debug(f"Created cached knowledge node: {cache_id}")
                
        except Exception as e:
            self.logger.error(f"Error creating cached knowledge node: {e}")
    
    async def find_similar_cached_queries(
        self,
        query: str,
        similarity_threshold: float = 0.7,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find cached queries that are similar to the given query.
        
        This uses simple text similarity - in production you might want
        to use semantic similarity with embeddings.
        
        Args:
            query: The query to find similar cached queries for
            similarity_threshold: Minimum similarity score (0-1)
            limit: Maximum number of results
            
        Returns:
            List of similar cached queries with their answers
        """
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                # Get all non-expired cached knowledge
                similarity_query = """
                MATCH (c:CachedKnowledge)
                WHERE c.character_id = $character_id
                AND datetime(c.expires_at) > datetime()
                RETURN c.cache_id as cache_id, c.query as cached_query, 
                       c.answer as answer, c.domain as domain,
                       c.query_type as query_type, c.created_at as created_at,
                       c.access_count as access_count
                """
                
                result = session.run(similarity_query, {
                    "character_id": self.character_id
                })
                
                similar_queries = []
                query_lower = query.lower()
                
                for record in result:
                    cached_query = record["cached_query"]
                    cached_query_lower = cached_query.lower()
                    
                    # Simple similarity calculation
                    # This could be improved with proper NLP similarity measures
                    similarity = self._calculate_text_similarity(query_lower, cached_query_lower)
                    
                    if similarity >= similarity_threshold:
                        similar_queries.append({
                            "cache_id": record["cache_id"],
                            "cached_query": cached_query,
                            "answer": record["answer"],
                            "domain": record["domain"],
                            "query_type": record["query_type"],
                            "created_at": record["created_at"],
                            "access_count": record["access_count"],
                            "similarity": similarity
                        })
                
                # Sort by similarity and limit results
                similar_queries.sort(key=lambda x: x["similarity"], reverse=True)
                return similar_queries[:limit]
                
        except Exception as e:
            self.logger.error(f"Error finding similar cached queries: {e}")
            return []
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity between two strings.
        
        This is a basic implementation using word overlap.
        In production, you might want to use more sophisticated methods.
        """
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def update_cache_access(self, cache_id: str):
        """Update access statistics for a cached knowledge item."""
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                update_query = """
                MATCH (c:CachedKnowledge {cache_id: $cache_id})
                SET c.access_count = coalesce(c.access_count, 0) + 1,
                    c.last_accessed = $timestamp
                """
                
                session.run(update_query, {
                    "cache_id": cache_id,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Error updating cache access: {e}")
    
    async def clean_expired_cache(self) -> int:
        """
        Remove expired cached knowledge entries.
        
        Returns:
            Number of entries removed
        """
        if not self.driver:
            return 0
        
        try:
            with self.driver.session() as session:
                # Delete expired cached knowledge nodes
                delete_query = """
                MATCH (c:CachedKnowledge)
                WHERE c.character_id = $character_id
                AND datetime(c.expires_at) <= datetime()
                DETACH DELETE c
                RETURN count(c) as deleted_count
                """
                
                result = session.run(delete_query, {
                    "character_id": self.character_id
                })
                
                deleted_count = result.single()["deleted_count"]
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned {deleted_count} expired cache entries for character {self.character_id}")
                
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error cleaning expired cache: {e}")
            return 0

    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            try:
                self.driver.close()
                self.logger.info(f"KnowledgeGraphModule closed for character {self.character_id}")
            except Exception as e:
                self.logger.error(f"Error closing KnowledgeGraphModule: {e}")


# Utility functions for knowledge management

async def extract_concepts_from_text(text: str) -> List[str]:
    """
    Extract potential concepts from text.
    
    This is a simple implementation - in production you might want to use
    NLP libraries like spaCy or transformers for better concept extraction.
    """
    import re
    
    # Simple concept extraction based on keywords and noun phrases
    concepts = []
    
    # Extract capitalized words (potential proper nouns)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
    concepts.extend(proper_nouns)
    
    # Extract quoted terms
    quoted_terms = re.findall(r'"([^"]*)"', text)
    concepts.extend(quoted_terms)
    
    # Extract common conceptual keywords
    conceptual_patterns = [
        r'\b(technology|science|art|music|literature|history|philosophy)\b',
        r'\b(method|technique|approach|strategy|system|process)\b',
        r'\b(theory|concept|idea|principle|law|rule)\b'
    ]
    
    for pattern in conceptual_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        concepts.extend(matches)
    
    # Clean and deduplicate
    concepts = [c.lower().strip() for c in concepts if len(c) > 2]
    concepts = list(set(concepts))  # Remove duplicates
    
    return concepts[:10]  # Limit to top 10 concepts


async def create_knowledge_from_web_search(
    character_id: str,
    query: str,
    results: List[Dict[str, Any]],
    confidence: float = 0.7
) -> List[str]:
    """
    Create knowledge graph entries from web search results.
    
    Args:
        character_id: ID of the character
        query: The search query
        results: List of search result dictionaries
        confidence: Confidence level for extracted facts
        
    Returns:
        List of created fact IDs
    """
    if not results:
        return []
    
    kg_module = KnowledgeGraphModule(character_id)
    created_fact_ids = []
    
    try:
        for i, result in enumerate(results[:3]):  # Limit to top 3 results
            # Extract key information from search result
            title = result.get("title", "")
            content = result.get("content", "")[:500]  # Limit content length
            url = result.get("url", "")
            
            if not content:
                continue
            
            # Create fact from search result
            fact_text = f"From search '{query}': {title}. {content}"
            
            # Extract concepts from the content
            concepts = await extract_concepts_from_text(f"{title} {content}")
            
            # Determine domain based on query and content
            domain = "web_knowledge"
            if any(term in query.lower() for term in ["science", "technology", "research"]):
                domain = "science_technology"
            elif any(term in query.lower() for term in ["history", "historical", "past"]):
                domain = "history"
            elif any(term in query.lower() for term in ["news", "current", "recent"]):
                domain = "current_events"
            
            # Store the fact
            fact_id = await kg_module.store_fact(
                fact_text=fact_text,
                source="web_search",
                confidence=confidence,
                domain=domain,
                related_concepts=concepts,
                web_search_context={
                    "query": query,
                    "results": [{"title": title, "url": url, "content": content}]
                }
            )
            
            if fact_id:
                created_fact_ids.append(fact_id)
        
        return created_fact_ids
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to create knowledge from web search: {e}")
        return created_fact_ids
    
    finally:
        kg_module.close()