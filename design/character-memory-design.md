# Adding Memory & Experience System to Character Agent

We need a **persistent, queryable memory system** that allows characters to:
1. **Remember every interaction** (with humans and other agents)
2. **Build knowledge about the world** through experiences
3. **Learn relationships** between concepts, people, events
4. **Recall relevant past experiences** when making decisions
5. **Associate emotions with memories** (how experiences made them feel)

Currently, our `CharacterState` only has a simple list-based `conversation_history` (last 20 messages). This is **insufficient** for:
- Long-term memory across sessions
- Semantic search ("remember when we talked about...")
- Relationship tracking
- Knowledge accumulation
- Emotional associations

---

## Enhanced Memory Architecture

### Storage Strategy

We'll use **three complementary storage systems**:

```
┌─────────────────────────────────────────────────────────┐
│         CHARACTER MEMORY SYSTEM                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. EPISODIC MEMORY (ChromaDB)                         │
│     - Vector embeddings of experiences                  │
│     - Semantic search: "remember when..."              │
│     - Includes emotional context                        │
│                                                         │
│  2. KNOWLEDGE GRAPH (Neo4j)                            │
│     - Entities (people, places, concepts)              │
│     - Relationships (knows, likes, learned_from)       │
│     - Temporal evolution                                │
│                                                         │
│  3. WORKING MEMORY (Redis)                             │
│     - Recent conversation (fast access)                │
│     - Current session context                           │
│     - Active emotional state                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## New Classes to Add

### 1. ExperienceModule (New)

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

@dataclass
class Experience:
    """
    A single memorable experience for the character.
    Stored in ChromaDB with vector embedding for semantic search.
    """
    experience_id: str
    character_id: str
    timestamp: datetime
    
    # Content
    experience_type: str  # 'conversation', 'interaction', 'learned_fact', 'event'
    description: str      # Natural language description
    participants: List[str]  # Who was involved (character_ids or 'human')
    location: Optional[str]
    
    # Emotional context
    emotional_state: str  # mood at the time
    emotional_impact: Dict[str, float]  # hormone changes caused
    emotional_valence: float  # -1 (negative) to 1 (positive)
    intensity: float  # 0-1, how significant/memorable
    
    # Cognitive context
    related_goals: List[str]  # goal_ids affected
    knowledge_gained: List[str]  # new facts learned
    relationship_changes: Dict[str, float]  # character_id -> trust delta
    
    # Metadata
    tags: List[str]  # searchable tags
    embedding: Optional[List[float]]  # Vector embedding (generated)
    retrieval_count: int = 0  # How often recalled
    last_retrieved: Optional[datetime] = None
    
    def to_searchable_text(self) -> str:
        """Convert to text for embedding"""
        return f"""
Experience at {self.timestamp.isoformat()}
Participants: {', '.join(self.participants)}
Description: {self.description}
Emotional state: {self.emotional_state}
Impact: {self.emotional_valence:.2f} intensity
Knowledge: {', '.join(self.knowledge_gained) if self.knowledge_gained else 'none'}
Tags: {', '.join(self.tags)}
        """.strip()
    
    @staticmethod
    def generate_id(character_id: str, timestamp: datetime) -> str:
        """Generate unique experience ID"""
        content = f"{character_id}_{timestamp.isoformat()}"
        return f"exp_{hashlib.md5(content.encode()).hexdigest()[:12]}"


class ExperienceModule:
    """
    Manages character's episodic memory.
    Stores, retrieves, and reasons about past experiences.
    """
    
    def __init__(
        self,
        character_id: str,
        chroma_client: Any,  # ChromaDB client
        collection_name: str = None
    ):
        self.character_id = character_id
        self.chroma_client = chroma_client
        
        # Create/get collection for this character
        collection_name = collection_name or f"character_{character_id}_experiences"
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"character_id": character_id}
        )
    
    def store_experience(
        self,
        experience: Experience
    ):
        """
        Store a new experience in episodic memory.
        Generates embedding and stores in ChromaDB.
        """
        
        # Convert experience to searchable text
        searchable_text = experience.to_searchable_text()
        
        # Store in ChromaDB
        self.collection.add(
            ids=[experience.experience_id],
            documents=[searchable_text],
            metadatas=[{
                'character_id': self.character_id,
                'timestamp': experience.timestamp.isoformat(),
                'experience_type': experience.experience_type,
                'emotional_state': experience.emotional_state,
                'emotional_valence': experience.emotional_valence,
                'intensity': experience.intensity,
                'participants': ','.join(experience.participants),
                'tags': ','.join(experience.tags)
            }]
        )
    
    def recall_similar_experiences(
        self,
        query: str,
        n_results: int = 5,
        min_intensity: float = 0.0
    ) -> List[Experience]:
        """
        Semantic search for similar past experiences.
        
        Example: query="feeling stressed at work" → returns similar stressful work experiences
        """
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={
                "intensity": {"$gte": min_intensity}
            } if min_intensity > 0 else None
        )
        
        # Convert results back to Experience objects
        experiences = []
        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            
            # Reconstruct Experience (simplified - in production, store full object)
            exp = Experience(
                experience_id=doc_id,
                character_id=self.character_id,
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                experience_type=metadata['experience_type'],
                description=results['documents'][0][i],
                participants=metadata['participants'].split(','),
                location=None,
                emotional_state=metadata['emotional_state'],
                emotional_impact={},
                emotional_valence=metadata['emotional_valence'],
                intensity=metadata['intensity'],
                related_goals=[],
                knowledge_gained=[],
                relationship_changes={},
                tags=metadata['tags'].split(',')
            )
            experiences.append(exp)
        
        return experiences
    
    def recall_by_participant(
        self,
        participant_id: str,
        n_results: int = 10
    ) -> List[Experience]:
        """
        Recall all experiences involving a specific person/agent.
        """
        
        results = self.collection.query(
            query_texts=[f"interactions with {participant_id}"],
            n_results=n_results,
            where={
                "participants": {"$contains": participant_id}
            }
        )
        
        return self._results_to_experiences(results)
    
    def recall_by_emotion(
        self,
        emotional_state: str,
        n_results: int = 5
    ) -> List[Experience]:
        """
        Recall experiences where character felt a specific emotion.
        
        Example: emotional_state="anxious" → returns times they felt anxious
        """
        
        results = self.collection.query(
            query_texts=[f"when I felt {emotional_state}"],
            n_results=n_results,
            where={
                "emotional_state": emotional_state
            }
        )
        
        return self._results_to_experiences(results)
    
    def recall_by_time_period(
        self,
        start_time: datetime,
        end_time: datetime,
        n_results: int = 20
    ) -> List[Experience]:
        """
        Recall experiences within a time range.
        """
        
        results = self.collection.query(
            query_texts=["experiences during this period"],
            n_results=n_results,
            where={
                "$and": [
                    {"timestamp": {"$gte": start_time.isoformat()}},
                    {"timestamp": {"$lte": end_time.isoformat()}}
                ]
            }
        )
        
        return self._results_to_experiences(results)
    
    def get_most_significant_experiences(
        self,
        n_results: int = 10,
        min_intensity: float = 0.7
    ) -> List[Experience]:
        """
        Get the most impactful/memorable experiences.
        """
        
        results = self.collection.query(
            query_texts=["most significant and memorable experiences"],
            n_results=n_results,
            where={
                "intensity": {"$gte": min_intensity}
            }
        )
        
        return self._results_to_experiences(results)
    
    def update_retrieval_stats(self, experience_id: str):
        """
        Update stats when an experience is recalled.
        More frequently recalled memories become stronger.
        """
        
        # In production, update the experience record
        # For now, just a placeholder
        pass
    
    def _results_to_experiences(self, results: Dict) -> List[Experience]:
        """Helper to convert ChromaDB results to Experience objects"""
        experiences = []
        
        if not results['ids'] or not results['ids'][0]:
            return experiences
        
        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            
            exp = Experience(
                experience_id=doc_id,
                character_id=self.character_id,
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                experience_type=metadata['experience_type'],
                description=results['documents'][0][i],
                participants=metadata['participants'].split(','),
                location=None,
                emotional_state=metadata['emotional_state'],
                emotional_impact={},
                emotional_valence=float(metadata['emotional_valence']),
                intensity=float(metadata['intensity']),
                related_goals=[],
                knowledge_gained=[],
                relationship_changes={},
                tags=metadata['tags'].split(',') if metadata.get('tags') else []
            )
            experiences.append(exp)
        
        return experiences
```

---

### 2. KnowledgeGraphModule (New)

```python
from typing import Dict, List, Any, Optional
from datetime import datetime

class KnowledgeGraphModule:
    """
    Manages character's semantic knowledge graph in Neo4j.
    Tracks entities, relationships, and how knowledge evolves over time.
    """
    
    def __init__(
        self,
        character_id: str,
        neo4j_driver: Any  # Neo4j driver
    ):
        self.character_id = character_id
        self.driver = neo4j_driver
    
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,  # 'person', 'place', 'concept', 'event', 'fact'
        properties: Dict[str, Any],
        learned_at: datetime
    ):
        """
        Add a new entity to the character's knowledge graph.
        
        Example: entity_type='person', properties={'name': 'Alice', 'occupation': 'doctor'}
        """
        
        with self.driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {entity_id: $entity_id, character_id: $character_id})
                SET e.entity_type = $entity_type,
                    e.learned_at = $learned_at,
                    e += $properties
                """,
                entity_id=entity_id,
                character_id=self.character_id,
                entity_type=entity_type,
                learned_at=learned_at.isoformat(),
                properties=properties
            )
    
    def add_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,  # 'KNOWS', 'LIKES', 'WORKS_AT', 'LEARNED_FROM', etc.
        properties: Dict[str, Any] = None,
        learned_at: datetime = None
    ):
        """
        Add a relationship between two entities.
        
        Example: ('Alice', 'Bob', 'KNOWS', {'since': '2024', 'context': 'work'})
        """
        
        if properties is None:
            properties = {}
        
        if learned_at:
            properties['learned_at'] = learned_at.isoformat()
        
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (from:Entity {{entity_id: $from_id, character_id: $character_id}})
                MATCH (to:Entity {{entity_id: $to_id, character_id: $character_id}})
                MERGE (from)-[r:{relationship_type}]->(to)
                SET r += $properties
                """,
                from_id=from_entity_id,
                to_id=to_entity_id,
                character_id=self.character_id,
                properties=properties
            )
    
    def query_entity(
        self,
        entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve all known information about an entity.
        """
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id, character_id: $character_id})
                RETURN e
                """,
                entity_id=entity_id,
                character_id=self.character_id
            )
            
            record = result.single()
            if record:
                return dict(record['e'])
            return None
    
    def query_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity.
        
        Example: query_relationships('Alice') → [{'type': 'KNOWS', 'target': 'Bob'}, ...]
        """
        
        query = """
        MATCH (e:Entity {entity_id: $entity_id, character_id: $character_id})-[r]->(target)
        """
        
        if relationship_type:
            query = f"""
            MATCH (e:Entity {{entity_id: $entity_id, character_id: $character_id}})-[r:{relationship_type}]->(target)
            """
        
        query += " RETURN type(r) as rel_type, target, properties(r) as props"
        
        with self.driver.session() as session:
            results = session.run(
                query,
                entity_id=entity_id,
                character_id=self.character_id
            )
            
            relationships = []
            for record in results:
                relationships.append({
                    'relationship_type': record['rel_type'],
                    'target_entity': dict(record['target']),
                    'properties': record['props']
                })
            
            return relationships
    
    def query_path(
        self,
        from_entity_id: str,
        to_entity_id: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """
        Find connection paths between two entities.
        
        Example: How does character know about X through Y?
        """
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (from:Entity {entity_id: $from_id, character_id: $character_id})
                             -[*1..%d]-
                             (to:Entity {entity_id: $to_id, character_id: $character_id})
                RETURN [node in nodes(path) | node.entity_id] as path_ids
                LIMIT 5
                """ % max_depth,
                from_id=from_entity_id,
                to_id=to_entity_id,
                character_id=self.character_id
            )
            
            paths = []
            for record in result:
                paths.append(record['path_ids'])
            
            return paths
    
    def search_entities_by_property(
        self,
        property_name: str,
        property_value: Any
    ) -> List[Dict[str, Any]]:
        """
        Search for entities matching a property.
        
        Example: search_entities_by_property('occupation', 'doctor')
        """
        
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity {{character_id: $character_id}})
                WHERE e.{property_name} = $property_value
                RETURN e
                """,
                character_id=self.character_id,
                property_value=property_value
            )
            
            entities = []
            for record in result:
                entities.append(dict(record['e']))
            
            return entities
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        Get a summary of what the character knows.
        """
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {character_id: $character_id})
                RETURN e.entity_type as type, count(e) as count
                """,
                character_id=self.character_id
            )
            
            summary = {}
            for record in result:
                summary[record['type']] = record['count']
            
            return summary
    
    def integrate_external_knowledge(
        self,
        search_query: str,
        tavily_client: Any
    ) -> List[Dict[str, Any]]:
        """
        Use Tavily to search the web and integrate new knowledge.
        
        This expands the character's world knowledge based on conversation needs.
        """
        
        # Search web via Tavily
        search_results = tavily_client.search(query=search_query, max_results=5)
        
        new_entities = []
        
        for result in search_results.get('results', []):
            # Extract entities from search result
            entity_id = f"web_{result['url'].split('//')[-1][:30]}"
            
            # Add to knowledge graph
            self.add_entity(
                entity_id=entity_id,
                entity_type='web_knowledge',
                properties={
                    'title': result.get('title', ''),
                    'content_snippet': result.get('content', '')[:500],
                    'source_url': result.get('url', ''),
                    'relevance_score': result.get('score', 0.0)
                },
                learned_at=datetime.now()
            )
            
            new_entities.append({
                'entity_id': entity_id,
                'title': result.get('title', ''),
                'url': result.get('url', '')
            })
        
        return new_entities
```

---

### 3. MemoryAgent (New Agent)

```python
class MemoryAgent(BaseAgent):
    """
    Agent that retrieves relevant memories during conversation.
    Provides context from past experiences to inform current response.
    """
    
    def __init__(
        self,
        agent_id: str,
        character_id: str,
        llm_client: Any,
        experience_module: ExperienceModule,
        knowledge_graph: KnowledgeGraphModule
    ):
        super().__init__(agent_id, "memory", character_id, llm_client)
        self.experience_module = experience_module
        self.knowledge_graph = knowledge_graph
    
    def consult(
        self,
        context: Dict[str, Any],
        character_state: 'CharacterState',
        user_message: str
    ) -> AgentInput:
        """
        Retrieves relevant memories and provides them as context.
        """
        
        # Semantic search for similar past experiences
        relevant_experiences = self.experience_module.recall_similar_experiences(
            query=user_message,
            n_results=3,
            min_intensity=0.3
        )
        
        # Check if user mentions any known entities
        known_entities = self._extract_known_entities(user_message)
        
        # Build memory context
        memory_context = self._build_memory_context(
            relevant_experiences,
            known_entities
        )
        
        # Use LLM to synthesize memory relevance
        prompt = self.get_prompt_template().format(
            user_message=user_message,
            memory_context=memory_context,
            recent_mood=character_state.agent_states['mood']['current_state']
        )
        
        response = self._call_llm(prompt, temperature=0.6)
        
        return AgentInput(
            agent_type="memory",
            content=response,
            confidence=0.8,
            priority=0.75,  # Memories are important but not absolute
            emotional_tone="reflective",
            metadata={
                'relevant_experiences': [exp.experience_id for exp in relevant_experiences],
                'known_entities': known_entities
            }
        )
    
    def get_prompt_template(self) -> str:
        return """You are analyzing how past experiences and memories inform the current response.

USER MESSAGE: "{user_message}"

RELEVANT PAST EXPERIENCES:
{memory_context}

CURRENT MOOD: {recent_mood}

Based on these memories, consider:
1. Do any past experiences make this conversation familiar/unfamiliar?
2. Has the character discussed this topic before? What was learned?
3. Do memories trigger emotional associations (good/bad past experiences)?
4. Should the character reference these memories explicitly or just be influenced by them?

Provide 2-3 sentences on how memories should shape the response."""
    
    def _extract_known_entities(self, message: str) -> List[Dict[str, Any]]:
        """
        Extract entities from message that are in the knowledge graph.
        Simple keyword matching for POC.
        """
        
        # In production, use NER or more sophisticated extraction
        # For now, search for common entity types
        
        entities = []
        
        # Search knowledge graph for entities mentioned
        # This is a simplified version - real implementation would be more sophisticated
        
        return entities
    
    def _build_memory_context(
        self,
        experiences: List[Experience],
        entities: List[Dict[str, Any]]
    ) -> str:
        """Format memories into readable context"""
        
        if not experiences and not entities:
            return "No directly relevant past experiences or known entities."
        
        context_parts = []
        
        if experiences:
            context_parts.append("PAST EXPERIENCES:")
            for exp in experiences:
                context_parts.append(
                    f"- {exp.timestamp.strftime('%Y-%m-%d')}: {exp.description[:200]}... "
                    f"(felt: {exp.emotional_state}, impact: {exp.emotional_valence:.2f})"
                )
        
        if entities:
            context_parts.append("\nKNOWN ENTITIES:")
            for entity in entities:
                context_parts.append(f"- {entity}")
        
        return "\n".join(context_parts)
```

---

### 4. Enhanced StateUpdater (Modified)

Add memory creation to existing `StateUpdater`:

```python
class StateUpdater:
    """
    Enhanced version that also creates memories and updates knowledge graph.
    """
    
    def __init__(
        self,
        neurochemical_agent: NeurochemicalAgent,
        experience_module: ExperienceModule,
        knowledge_graph: KnowledgeGraphModule
    ):
        self.neurochemical_agent = neurochemical_agent
        self.experience_module = experience_module
        self.knowledge_graph = knowledge_graph
    
    def update_after_response(
        self,
        character_state: CharacterState,
        user_message: str,
        character_response: str,
        agent_inputs: Dict[str, AgentInput]
    ) -> CharacterState:
        """
        Enhanced update that creates memories and updates knowledge.
        """
        
        # [Previous hormone decay and mood update code remains...]
        
        # NEW: Create experience/memory from this interaction
        experience = self._create_experience_from_interaction(
            character_state,
            user_message,
            character_response,
            agent_inputs
        )
        
        # Store in episodic memory
        self.experience_module.store_experience(experience)
        
        # NEW: Extract and store knowledge
        self._extract_and_store_knowledge(
            user_message,
            character_response,
            character_state
        )
        
        # [Rest of update logic...]
        
        return character_state
    
    def _create_experience_from_interaction(
        self,
        character_state: CharacterState,
        user_message: str,
        character_response: str,
        agent_inputs: Dict[str, AgentInput]
    ) -> Experience:
        """
        Convert this interaction into a memorable experience.
        """
        
        # Determine if this interaction is memorable enough to store
        mood = character_state.agent_states['mood']
        intensity = self._calculate_memory_intensity(
            user_message,
            character_response,
            mood
        )
        
        # Create experience record
        experience = Experience(
            experience_id=Experience.generate_id(
                character_state.character_id,
                datetime.now()
            ),
            character_id=character_state.character_id,
            timestamp=datetime.now(),
            experience_type='conversation',
            description=f"User said: {user_message[:200]}... I responded: {character_response[:200]}...",
            participants=['human_user'],
            location=None,
            emotional_state=mood['current_state'],
            emotional_impact=self._calculate_emotional_impact(agent_inputs),
            emotional_valence=self._calculate_valence(mood, character_response),
            intensity=intensity,
            related_goals=self._extract_related_goals(agent_inputs),
            knowledge_gained=self._extract_knowledge_gained(user_message, character_response),
            relationship_changes={'human_user': character_state.relationship_state['trust_level']},
            tags=self._generate_tags(user_message, character_response)
        )
        
        return experience
    
    def _calculate_memory_intensity(
        self,
        user_message: str,
        character_response: str,
        mood: Dict[str, Any]
    ) -> float:
        """
        Calculate how memorable this interaction is.
        More intense emotions = stronger memories.
        """
        
        intensity = 0.5  # baseline
        
        # Strong emotions increase memorability
        if mood['intensity'] > 0.7:
            intensity += 0.3
        
        # Longer interactions are more memorable
        if len(user_message) + len(character_response) > 200:
            intensity += 0.1
        
        # First-time topics are memorable
        # (would check against knowledge graph in production)
        
        return min(1.0, intensity)
    
    def _calculate_emotional_impact(
        self,
        agent_inputs: Dict[str, AgentInput]
    ) -> Dict[str, float]:
        """Extract emotional impact from neurochemical agent"""
        
        neuro_input = agent_inputs.get('neurochemical')
        if neuro_input and neuro_input.metadata:
            return {
                'dopamine': neuro_input.metadata.get('reward_seeking', 0),
                'cortisol': neuro_input.metadata.get('stress_level', 0),
                'oxytocin': neuro_input.metadata.get('connection_desire', 0)
            }
        return {}
    
    def _calculate_valence(
        self,
        mood: Dict[str, Any],
        response: str
    ) -> float:
        """Calculate emotional valence of experience"""
        
        positive_words = ['happy', 'great', 'love', 'wonderful', 'excited']
        negative_words = ['sad', 'angry', 'frustrated', 'annoyed', 'terrible']
        
        response_lower = response.lower()
        
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        if positive_count > negative_count:
            return 0.5 + (positive_count * 0.1)
        elif negative_count > positive_count:
            return -0.5 - (negative_count * 0.1)
        else:
            return 0.0
    
    def _extract_related_goals(
        self,
        agent_inputs: Dict[str, AgentInput]
    ) -> List[str]:
        """Extract which goals were relevant to this interaction"""
        
        goals_input = agent_inputs.get('goals')
        if goals_input and goals_input.metadata:
            return [g['goal_id'] for g in goals_input.metadata.get('active_goals', [])]
        return []
    
    def _extract_knowledge_gained(
        self,
        user_message: str,
        character_response: str
    ) -> List[str]:
        """
        Extract new facts/knowledge from this interaction.
        In production, use NER and fact extraction.
        """
        
        knowledge = []
        
        # Simple pattern matching for POC
        if "i am" in user_message.lower():
            knowledge.append(f"User revealed: {user_message}")
        
        if "my" in user_message.lower():
            knowledge.append(f"Learned about user: {user_message}")
        
        return knowledge
    
    def _generate_tags(
        self,
        user_message: str,
        character_response: str
    ) -> List[str]:
        """Generate searchable tags for this experience"""
        
        tags = ['conversation']
        
        # Topic detection (simple keyword matching for POC)
        topics = {
            'work': ['job', 'work', 'career', 'office', 'project'],
            'family': ['family', 'parent', 'child', 'sibling', 'mother', 'father'],
            'health': ['health', 'sick', 'doctor', 'medical', 'pain'],
            'emotion': ['feel', 'happy', 'sad', 'angry', 'anxious', 'excited']
        }
        
        combined_text = (user_message + " " + character_response).lower()
        
        for topic, keywords in topics.items():
            if any(keyword in combined_text for keyword in keywords):
                tags.append(topic)
        
        return tags
    
    def _extract_and_store_knowledge(
        self,
        user_message: str,
        character_response: str,
        character_state: CharacterState
    ):
        """
        Extract entities and relationships from interaction.
        Store in knowledge graph.
        """
        
        # Simple entity extraction for POC
        # In production, use NER (spaCy, etc.)
        
        # Example: if user says "I work at Google", extract:
        # - Entity: "Google" (type: company)
        # - Relationship: user WORKS_AT Google
        
        # This is a placeholder for more sophisticated NER
        pass
```

---

## Updated CharacterAgent Integration

```python
class CharacterAgent:
    """
    Enhanced with memory and knowledge systems.
    """
    
    def __init__(
        self,
        character_config: Dict[str, Any],
        llm_client: Any,
        chroma_client: Any,  # NEW
        neo4j_driver: Any,   # NEW
        tavily_client: Any = None  # NEW (optional)
    ):
        self.config = character_config
        self.character_id = character_config['character_id']
        self.character_name = character_config['name']
        self.llm_client = llm_client
        
        # Initialize memory systems
        self.experience_module = ExperienceModule(
            self.character_id,
            chroma_client
        )
        
        self.knowledge_graph = KnowledgeGraphModule(
            self.character_id,
            neo4j_driver
        )
        
        self.tavily_client = tavily_client
        
        # Initialize character state
        self.state = self._initialize_state(character_config)
        
        # Initialize agents (including new MemoryAgent)
        self.agents = self._initialize_agents(
            character_config,
            llm_client,
            self.experience_module,
            self.knowledge_graph
        )
        
        # Initialize orchestrator
        self.orchestrator = AgentOrchestrator(self.agents, self.state)
        
        # Initialize modules
        self.cognitive_module = CognitiveModule(self.character_id, llm_client)
        self.response_generator = ResponseGenerator(self.character_name, llm_client)
        self.state_updater = StateUpdater(
            self._get_neurochemical_agent(),
            self.experience_module,  # NEW
            self.knowledge_graph      # NEW
        )
        
        self.logger = logging.getLogger(f"Character.{self.character_name}")
    
    def _initialize_agents(
        self,
        config: Dict[str, Any],
        llm_client: Any,
        experience_module: ExperienceModule,
        knowledge_graph: KnowledgeGraphModule
    ) -> List[BaseAgent]:
        """Initialize all agents including MemoryAgent"""
        
        agents = [
            PersonalityAgent(...),
            MoodAgent(...),
            NeurochemicalAgent(...),
            GoalsAgent(...),
            CommunicationStyleAgent(...),
            # NEW:
            MemoryAgent(
                f"{self.character_id}_memory",
                self.character_id,
                llm_client,
                experience_module,
                knowledge_graph
            )
        ]
        
        return agents
    
    def search_and_learn(
        self,
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Use Tavily to search web and integrate new knowledge.
        Expands character's world knowledge.
        """
        
        if not self.tavily_client:
            self.logger.warning("Tavily client not configured")
            return []
        
        new_entities = self.knowledge_graph.integrate_external_knowledge(
            query,
            self.tavily_client
        )
        
        self.logger.info(f"Learned {len(new_entities)} new entities from web search")
        
        return new_entities
    
    def recall_past_conversations(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Experience]:
        """
        Manually trigger memory recall (for debugging/testing).
        """
        return self.experience_module.recall_similar_experiences(query, n_results)
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of what character knows"""
        return self.knowledge_graph.get_knowledge_summary()
```

---

## Summary of Memory System

### What We Added:

1. **ExperienceModule** - Episodic memory in ChromaDB
   - Stores every meaningful interaction
   - Semantic search for similar experiences
   - Emotional associations with memories
   - Retrieval by participant, emotion, time, significance

2. **KnowledgeGraphModule** - Semantic knowledge in Neo4j
   - Entities (people, places, concepts)
   - Relationships between entities
   - Path queries (how does character know X?)
   - Integration with Tavily for web knowledge

3. **MemoryAgent** - New agent that retrieves memories
   - Consults episodic memory during conversations
   - Provides relevant past experiences to other agents
   - Influences responses based on history

4. **Enhanced StateUpdater** - Creates memories automatically
   - Converts interactions into Experience objects
   - Stores in ChromaDB with embeddings
   - Extracts knowledge for graph storage
   - Tracks emotional associations

### Key Benefits:

✅ **Characters remember everything** - No information loss
✅ **Semantic search** - "Remember when we talked about X?"
✅ **Emotional associations** - Memories carry feelings
✅ **Knowledge accumulation** - Characters learn from every interaction
✅ **Web integration** - Can expand knowledge via Tavily
✅ **Relationship tracking** - Knows history with each person
✅ **Cross-session persistence** - Memory survives restarts

---

## Updated Class Hierarchy

```
BaseAgent
├── PersonalityAgent
├── MoodAgent
├── NeurochemicalAgent
├── GoalsAgent
├── CommunicationStyleAgent
└── MemoryAgent (NEW)

CharacterState (existing)

ExperienceModule (NEW)
└── Experience (dataclass)

KnowledgeGraphModule (NEW)

StateUpdater (enhanced with memory creation)

CharacterAgent (enhanced with memory systems)
```

