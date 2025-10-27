"""Experience module for episodic memory storage and retrieval using ChromaDB."""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import asyncio
from dataclasses import asdict

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.api.types import EmbeddingFunction, Embeddings, Documents
except ImportError as e:
    raise ImportError(f"ChromaDB not installed: {e}")

try:
    from ..core.experience import Experience
except ImportError:
    from core.experience import Experience

try:
    from ..config.settings import get_settings
except ImportError:
    from config.settings import get_settings


class SimpleEmbeddingFunction(EmbeddingFunction):
    """Simple embedding function using basic text features."""
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate simple embeddings based on text features."""
        embeddings = []
        
        for doc in input:
            # Simple feature-based embedding (in production, use proper embedding model)
            features = [
                len(doc) / 1000,  # Document length
                doc.lower().count('happy') * 2,  # Positive sentiment
                doc.lower().count('sad') * -2,   # Negative sentiment
                doc.lower().count('learn') * 1.5,  # Learning content
                doc.lower().count('goal') * 1.2,   # Goal-related
                doc.lower().count('search') * 0.8, # Web search content
            ]
            
            # Pad or truncate to fixed size (384 dimensions)
            embedding = features + [0.0] * (384 - len(features))
            embedding = embedding[:384]
            embeddings.append(embedding)
        
        return embeddings


class ExperienceModule:
    """
    Manages episodic memory storage and retrieval using ChromaDB.
    
    Stores character experiences as vector embeddings for semantic search
    and retrieval based on similarity to current context.
    """
    
    def __init__(self, character_id: str):
        """Initialize experience module for a specific character."""
        self.character_id = character_id
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB client
        self.client = self._init_chromadb()
        self.collection_name = f"experiences_{character_id}"
        self.collection = self._get_or_create_collection()
        
        self.logger.info(f"ExperienceModule initialized for character {character_id}")
    
    def _init_chromadb(self) -> chromadb.Client:
        """Initialize ChromaDB client with appropriate settings."""
        try:
            # Ensure the persist directory exists and has proper permissions
            persist_dir = self.settings.chroma_persist_directory
            
            # Create directory if it doesn't exist
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir, mode=0o755, exist_ok=True)
                self.logger.info(f"Created ChromaDB directory: {persist_dir}")
            
            # Check if directory is writable
            if not os.access(persist_dir, os.W_OK):
                raise PermissionError(f"ChromaDB directory not writable: {persist_dir}")
            
            # Use persistent client for data persistence
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.logger.info(f"ChromaDB client initialized at {persist_dir}")
            return client
            
        except PermissionError as e:
            self.logger.error(f"ChromaDB permission error: {e}")
            self.logger.warning("Falling back to in-memory ChromaDB client")
            return chromadb.Client()
        except OSError as e:
            if "Permission denied" in str(e):
                self.logger.error(f"ChromaDB permission denied (os error 13): {persist_dir}")
                self.logger.info("Try running with appropriate permissions or changing CHROMA_PERSIST_DIRECTORY")
            else:
                self.logger.error(f"ChromaDB OS error: {e}")
            self.logger.warning("Falling back to in-memory ChromaDB client")
            return chromadb.Client()
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            # Fallback to in-memory client
            self.logger.warning("Falling back to in-memory ChromaDB client")
            return chromadb.Client()
    
    def _get_or_create_collection(self):
        """Get or create the experiences collection for this character."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=SimpleEmbeddingFunction()
            )
            self.logger.info(f"Retrieved existing collection: {self.collection_name}")
            
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=SimpleEmbeddingFunction(),
                metadata={"description": f"Episodic memories for character {self.character_id}"}
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    async def store_experience(self, experience: Experience) -> bool:
        """
        Store an experience in the episodic memory.
        
        Args:
            experience: The Experience object to store
            
        Returns:
            bool: True if successfully stored, False otherwise
        """
        try:
            # Convert experience to searchable text
            searchable_text = experience.to_searchable_text()
            
            # Prepare metadata (ChromaDB requires string values)
            metadata = {
                "character_id": experience.character_id,
                "timestamp": experience.timestamp.isoformat(),
                "experience_type": experience.experience_type,
                "emotional_state": experience.emotional_state,
                "emotional_valence": str(experience.emotional_valence),
                "intensity": str(experience.intensity),
                "participants": ",".join(experience.participants),
                "web_search_triggered": str(experience.web_search_triggered),
                "tags": ",".join(experience.tags) if experience.tags else ""
            }
            
            # Add to collection
            self.collection.add(
                documents=[searchable_text],
                metadatas=[metadata],
                ids=[experience.experience_id]
            )
            
            self.logger.debug(f"Stored experience {experience.experience_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store experience {experience.experience_id}: {e}")
            return False
    
    async def retrieve_similar_experiences(
        self,
        query_text: str,
        n_results: int = 5,
        time_window_days: Optional[int] = None,
        experience_types: Optional[List[str]] = None,
        min_intensity: Optional[float] = None
    ) -> List[Experience]:
        """
        Retrieve experiences similar to the query text.
        
        Args:
            query_text: Text to search for similar experiences
            n_results: Maximum number of results to return
            time_window_days: Only return experiences from last N days
            experience_types: Filter by experience types
            min_intensity: Minimum intensity threshold
            
        Returns:
            List of Experience objects ordered by similarity
        """
        try:
            # Build where clause for filtering - ChromaDB expects simple equality filters
            where_clause = {"character_id": self.character_id}
            
            # For ChromaDB, we need to handle time filtering differently
            # We'll filter the results after querying instead of in the where clause
            
            if experience_types:
                # Only add simple equality filters to ChromaDB where clause
                if len(experience_types) == 1:
                    where_clause["experience_type"] = experience_types[0]
                # If multiple types, we'll filter after querying
            
            # Query the collection (ChromaDB doesn't support complex operators in where)
            try:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results * 2,  # Get more results to allow for filtering
                    where=where_clause if len(where_clause) > 1 else {"character_id": self.character_id}
                )
            except Exception as query_error:
                self.logger.warning(f"ChromaDB query with where clause failed: {query_error}, trying without where clause")
                # Fallback: query without where clause and filter manually
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results * 3  # Get even more to filter manually
                )
            
            # Convert results back to Experience objects with post-query filtering
            experiences = []
            if results["ids"] and results["ids"][0]:
                for i, experience_id in enumerate(results["ids"][0]):
                    try:
                        experience = await self.get_experience_by_id(experience_id)
                        if experience:
                            # Apply post-query filters
                            
                            # Time window filtering
                            if time_window_days:
                                cutoff_date = datetime.now() - timedelta(days=time_window_days)
                                if experience.timestamp < cutoff_date:
                                    continue
                            
                            # Experience type filtering (if multiple types were specified)
                            if experience_types and len(experience_types) > 1:
                                if experience.experience_type not in experience_types:
                                    continue
                            
                            # Intensity filtering
                            if min_intensity and experience.intensity < min_intensity:
                                continue
                            
                            # Add similarity score from ChromaDB
                            if results["distances"] and results["distances"][0] and i < len(results["distances"][0]):
                                distance = results["distances"][0][i]
                                similarity = 1.0 - distance  # Convert distance to similarity
                                # Store similarity in a way that won't interfere with the dataclass
                                experience._similarity_score = similarity
                            
                            experiences.append(experience)
                            
                            # Stop when we have enough results
                            if len(experiences) >= n_results:
                                break
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to reconstruct experience {experience_id}: {e}")
                        continue
            
            self.logger.debug(f"Retrieved {len(experiences)} similar experiences for query: {query_text[:50]}...")
            return experiences
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve similar experiences: {e}")
            return []
    
    async def get_experience_by_id(self, experience_id: str) -> Optional[Experience]:
        """Retrieve a specific experience by ID."""
        try:
            results = self.collection.get(ids=[experience_id])
            
            if not results["ids"] or not results["ids"]:
                return None
            
            metadata = results["metadatas"][0]
            
            # Reconstruct Experience object from metadata
            # Note: This is a simplified reconstruction - in production,
            # you might want to store the full object as JSON in the document
            experience = Experience(
                experience_id=experience_id,
                character_id=metadata["character_id"],
                timestamp=datetime.fromisoformat(metadata["timestamp"]),
                experience_type=metadata["experience_type"],
                description=results["documents"][0],  # Using searchable text as description
                participants=metadata["participants"].split(",") if metadata["participants"] else [],
                emotional_state=metadata["emotional_state"],
                emotional_valence=float(metadata["emotional_valence"]),
                intensity=float(metadata["intensity"]),
                web_search_triggered=metadata["web_search_triggered"].lower() == "true",
                tags=metadata["tags"].split(",") if metadata["tags"] else []
            )
            
            return experience
            
        except Exception as e:
            self.logger.error(f"Failed to get experience {experience_id}: {e}")
            return None
    
    async def get_recent_experiences(
        self,
        n_results: int = 10,
        days_back: int = 7
    ) -> List[Experience]:
        """Get recent experiences within the specified time window."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            # Use simple where clause for ChromaDB - filter by time after retrieval
            try:
                results = self.collection.get(
                    where={"character_id": self.character_id},
                    limit=n_results * 3  # Get more to filter by time
                )
            except Exception as get_error:
                self.logger.warning(f"ChromaDB get with where clause failed: {get_error}, trying without where clause")
                # Fallback: get all and filter manually
                results = self.collection.get(limit=n_results * 5)
            
            experiences = []
            if results["ids"]:
                for experience_id in results["ids"]:
                    experience = await self.get_experience_by_id(experience_id)
                    if experience:
                        # Apply time filtering manually
                        if experience.timestamp >= cutoff_date:
                            experiences.append(experience)
                        
                        # Stop when we have enough results
                        if len(experiences) >= n_results:
                            break
            
            # Sort by timestamp (most recent first)
            experiences.sort(key=lambda x: x.timestamp, reverse=True)
            
            return experiences[:n_results]
            
        except Exception as e:
            self.logger.error(f"Failed to get recent experiences: {e}")
            return []
    
    async def get_experiences_by_type(
        self,
        experience_type: str,
        n_results: int = 10
    ) -> List[Experience]:
        """Get experiences of a specific type."""
        try:
            results = self.collection.get(
                where={
                    "character_id": self.character_id,
                    "experience_type": experience_type
                },
                limit=n_results
            )
            
            experiences = []
            if results["ids"]:
                for experience_id in results["ids"]:
                    experience = await self.get_experience_by_id(experience_id)
                    if experience:
                        experiences.append(experience)
            
            return experiences
            
        except Exception as e:
            self.logger.error(f"Failed to get experiences by type {experience_type}: {e}")
            return []
    
    async def update_experience_retrieval_stats(self, experience_id: str) -> bool:
        """Update retrieval statistics for an experience."""
        try:
            experience = await self.get_experience_by_id(experience_id)
            if not experience:
                return False
            
            # Update retrieval stats
            experience.increment_retrieval()
            
            # Re-store the updated experience
            return await self.store_experience(experience)
            
        except Exception as e:
            self.logger.error(f"Failed to update retrieval stats for {experience_id}: {e}")
            return False
    
    async def delete_experience(self, experience_id: str) -> bool:
        """Delete an experience from memory."""
        try:
            self.collection.delete(ids=[experience_id])
            self.logger.debug(f"Deleted experience {experience_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete experience {experience_id}: {e}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the character's episodic memory."""
        try:
            # Get collection info
            collection_count = self.collection.count()
            
            # Get recent activity
            recent_experiences = await self.get_recent_experiences(n_results=100, days_back=30)
            
            # Analyze experience types
            type_counts = {}
            emotional_distribution = {"positive": 0, "neutral": 0, "negative": 0}
            web_search_count = 0
            
            for exp in recent_experiences:
                # Count types
                type_counts[exp.experience_type] = type_counts.get(exp.experience_type, 0) + 1
                
                # Emotional distribution
                if exp.emotional_valence > 0.3:
                    emotional_distribution["positive"] += 1
                elif exp.emotional_valence < -0.3:
                    emotional_distribution["negative"] += 1
                else:
                    emotional_distribution["neutral"] += 1
                
                # Web search usage
                if exp.web_search_triggered:
                    web_search_count += 1
            
            return {
                "character_id": self.character_id,
                "total_experiences": collection_count,
                "recent_experiences_30d": len(recent_experiences),
                "experience_types": type_counts,
                "emotional_distribution": emotional_distribution,
                "web_search_experiences": web_search_count,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Clean up resources."""
        try:
            # ChromaDB client doesn't need explicit closing
            self.logger.info(f"ExperienceModule closed for character {self.character_id}")
        except Exception as e:
            self.logger.error(f"Error closing ExperienceModule: {e}")


# Utility functions for experience management

async def create_experience_from_interaction(
    character_id: str,
    user_message: str,
    character_response: str,
    emotional_state: str,
    web_search_data: Optional[Dict[str, Any]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Experience:
    """
    Create an Experience object from a character interaction.
    
    Args:
        character_id: ID of the character
        user_message: What the user said
        character_response: How the character responded
        emotional_state: Character's emotional state during interaction
        web_search_data: Any web search results that were part of this interaction
        additional_metadata: Extra metadata about the interaction
        
    Returns:
        Experience object ready for storage
    """
    experience_id = Experience.generate_id(character_id, datetime.now())
    
    # Build description from interaction
    description = f"Conversation with human: User said '{user_message}', character responded '{character_response}'"
    
    # Create base experience
    experience = Experience(
        experience_id=experience_id,
        character_id=character_id,
        timestamp=datetime.now(),
        experience_type="conversation",
        description=description,
        participants=["human"],
        emotional_state=emotional_state,
        tags=["conversation", "interaction"]
    )
    
    # Add web search data if present
    if web_search_data:
        experience.add_web_search_results(
            query=web_search_data.get("query", ""),
            results=web_search_data.get("results", []),
            knowledge_extracted=web_search_data.get("knowledge", [])
        )
    
    # Add any additional metadata
    if additional_metadata:
        if "emotional_valence" in additional_metadata:
            experience.emotional_valence = additional_metadata["emotional_valence"]
        if "intensity" in additional_metadata:
            experience.intensity = additional_metadata["intensity"]
        if "related_goals" in additional_metadata:
            experience.related_goals = additional_metadata["related_goals"]
        if "additional_tags" in additional_metadata:
            experience.tags.extend(additional_metadata["additional_tags"])
    
    return experience