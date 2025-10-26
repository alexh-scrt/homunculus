"""Test memory system components."""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def character_id():
    """Test character ID."""
    return "test_character_001"


@pytest.fixture
def sample_experience():
    """Sample experience for testing."""
    from core.experience import Experience
    
    return Experience(
        experience_id="test_exp_001",
        character_id="test_character_001",
        timestamp=datetime.now(),
        experience_type="conversation",
        description="Test conversation about programming",
        participants=["human"],
        emotional_state="curious",
        emotional_valence=0.5,
        intensity=0.7,
        tags=["programming", "learning"]
    )


class TestExperienceModule:
    """Test ExperienceModule functionality."""
    
    @pytest.mark.asyncio
    async def test_experience_module_init(self, character_id):
        """Test ExperienceModule initialization."""
        from memory.experience_module import ExperienceModule
        
        # Mock ChromaDB to avoid requiring actual database
        with patch('memory.experience_module.chromadb') as mock_chromadb:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.side_effect = Exception("Not found")
            mock_client.create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_chromadb.Client.return_value = mock_client
            
            exp_module = ExperienceModule(character_id)
            
            assert exp_module.character_id == character_id
            assert exp_module.collection_name == f"experiences_{character_id}"
    
    @pytest.mark.asyncio
    async def test_store_experience(self, character_id, sample_experience):
        """Test storing an experience."""
        from memory.experience_module import ExperienceModule
        
        with patch('memory.experience_module.chromadb') as mock_chromadb:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.side_effect = Exception("Not found")
            mock_client.create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_chromadb.Client.return_value = mock_client
            
            exp_module = ExperienceModule(character_id)
            
            # Mock successful storage
            mock_collection.add.return_value = None
            
            result = await exp_module.store_experience(sample_experience)
            
            assert result is True
            mock_collection.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_similar_experiences(self, character_id):
        """Test retrieving similar experiences."""
        from memory.experience_module import ExperienceModule
        
        with patch('memory.experience_module.chromadb') as mock_chromadb:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.side_effect = Exception("Not found")
            mock_client.create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_chromadb.Client.return_value = mock_client
            
            # Mock query results
            mock_collection.query.return_value = {
                "ids": [["test_exp_001"]],
                "distances": [[0.3]],
                "metadatas": [[{
                    "character_id": character_id,
                    "timestamp": datetime.now().isoformat(),
                    "experience_type": "conversation",
                    "emotional_state": "curious",
                    "emotional_valence": "0.5",
                    "intensity": "0.7",
                    "participants": "human",
                    "web_search_triggered": "false",
                    "tags": "programming,learning"
                }]],
                "documents": [["Test conversation about programming"]]
            }
            
            mock_collection.get.return_value = {
                "ids": ["test_exp_001"],
                "metadatas": [{
                    "character_id": character_id,
                    "timestamp": datetime.now().isoformat(),
                    "experience_type": "conversation",
                    "emotional_state": "curious",
                    "emotional_valence": "0.5",
                    "intensity": "0.7",
                    "participants": "human",
                    "web_search_triggered": "false",
                    "tags": "programming,learning"
                }],
                "documents": ["Test conversation about programming"]
            }
            
            exp_module = ExperienceModule(character_id)
            
            results = await exp_module.retrieve_similar_experiences(
                query_text="Tell me about programming",
                n_results=5
            )
            
            assert len(results) == 1
            assert results[0].experience_id == "test_exp_001"
            assert results[0].experience_type == "conversation"
    
    @pytest.mark.asyncio
    async def test_get_memory_stats(self, character_id):
        """Test getting memory statistics."""
        from memory.experience_module import ExperienceModule
        
        with patch('memory.experience_module.chromadb') as mock_chromadb:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.side_effect = Exception("Not found")
            mock_client.create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_chromadb.Client.return_value = mock_client
            
            # Mock collection count
            mock_collection.count.return_value = 42
            
            exp_module = ExperienceModule(character_id)
            
            # Mock get_recent_experiences
            async def mock_get_recent():
                return []
            exp_module.get_recent_experiences = Mock(return_value=mock_get_recent())
            
            stats = await exp_module.get_memory_stats()
            
            assert stats["character_id"] == character_id
            assert stats["total_experiences"] == 42


class TestKnowledgeGraphModule:
    """Test KnowledgeGraphModule functionality."""
    
    def test_knowledge_graph_init(self, character_id):
        """Test KnowledgeGraphModule initialization."""
        from memory.knowledge_graph_module import KnowledgeGraphModule
        
        # Mock Neo4j to avoid requiring actual database
        with patch('memory.knowledge_graph_module.GraphDatabase') as mock_graphdb:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_session.run.return_value.single.return_value = None
            mock_driver.session.return_value = mock_session
            mock_graphdb.driver.return_value = mock_driver
            
            kg_module = KnowledgeGraphModule(character_id)
            
            assert kg_module.character_id == character_id
            assert kg_module.driver is not None
    
    @pytest.mark.asyncio
    async def test_store_fact(self, character_id):
        """Test storing a fact."""
        from memory.knowledge_graph_module import KnowledgeGraphModule
        
        with patch('memory.knowledge_graph_module.GraphDatabase') as mock_graphdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_result = MagicMock()
            mock_result.single.return_value = {"fact_id": "test_fact_001"}
            mock_session.run.return_value = mock_result
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None
            mock_graphdb.driver.return_value = mock_driver
            
            kg_module = KnowledgeGraphModule(character_id)
            
            fact_id = await kg_module.store_fact(
                fact_text="Python is a programming language",
                source="conversation",
                confidence=0.9,
                domain="programming",
                related_concepts=["python", "programming"]
            )
            
            assert fact_id == "test_fact_001"
            assert mock_session.run.called
    
    @pytest.mark.asyncio
    async def test_retrieve_related_facts(self, character_id):
        """Test retrieving related facts."""
        from memory.knowledge_graph_module import KnowledgeGraphModule
        
        with patch('memory.knowledge_graph_module.GraphDatabase') as mock_graphdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_result = MagicMock()
            
            # Mock query result
            mock_record = MagicMock()
            mock_record.__getitem__ = lambda self, key: {
                "fact_id": "test_fact_001",
                "text": "Python is a programming language",
                "source": "conversation",
                "confidence": 0.9,
                "domain": "programming",
                "timestamp": datetime.now().isoformat(),
                "retrieval_count": 0,
                "concept_matches": 1,
                "web_sources": []
            }[key]
            
            mock_result.__iter__ = lambda: iter([mock_record])
            mock_session.run.return_value = mock_result
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None
            mock_graphdb.driver.return_value = mock_driver
            
            kg_module = KnowledgeGraphModule(character_id)
            
            facts = await kg_module.retrieve_related_facts(
                concepts=["python", "programming"],
                limit=5
            )
            
            assert len(facts) == 1
            assert facts[0]["fact_id"] == "test_fact_001"
            assert facts[0]["text"] == "Python is a programming language"
    
    @pytest.mark.asyncio
    async def test_get_knowledge_stats(self, character_id):
        """Test getting knowledge statistics."""
        from memory.knowledge_graph_module import KnowledgeGraphModule
        
        with patch('memory.knowledge_graph_module.GraphDatabase') as mock_graphdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_result = MagicMock()
            mock_record = MagicMock()
            mock_record.__getitem__ = lambda self, key: {
                "total_facts": 15,
                "total_concepts": 8,
                "total_goals": 3,
                "web_sources": 5,
                "avg_confidence": 0.85,
                "domains": ["programming", "science"],
                "sources": ["conversation", "web_search"]
            }[key]
            mock_result.single.return_value = mock_record
            mock_session.run.return_value = mock_result
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None
            mock_graphdb.driver.return_value = mock_driver
            
            kg_module = KnowledgeGraphModule(character_id)
            
            stats = await kg_module.get_knowledge_stats()
            
            assert stats["character_id"] == character_id
            assert stats["total_facts"] == 15
            assert stats["total_concepts"] == 8


class TestMemoryAgent:
    """Test MemoryAgent functionality."""
    
    def test_memory_agent_init(self, character_id):
        """Test MemoryAgent initialization."""
        from agents.memory_agent import MemoryAgent
        from llm.ollama_client import OllamaClient
        from config.settings import get_settings
        
        settings = get_settings()
        llm_client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            tavily_api_key=settings.tavily_api_key
        )
        
        with patch('memory.experience_module.chromadb'), \
             patch('memory.knowledge_graph_module.GraphDatabase'):
            
            memory_agent = MemoryAgent(
                agent_id="test_memory",
                character_id=character_id,
                llm_client=llm_client
            )
            
            assert memory_agent.agent_type == "memory"
            assert memory_agent.character_id == character_id
            assert memory_agent.web_search_enabled is False
    
    @pytest.mark.asyncio
    async def test_store_interaction_memory(self, character_id):
        """Test storing interaction memory."""
        from agents.memory_agent import MemoryAgent
        from core.character_state import CharacterState
        from llm.ollama_client import OllamaClient
        from config.settings import get_settings
        
        settings = get_settings()
        llm_client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            tavily_api_key=settings.tavily_api_key
        )
        
        char_state = CharacterState(
            character_id=character_id,
            last_updated=datetime.now(),
            name="Test Character",
            archetype="test"
        )
        
        with patch('memory.experience_module.chromadb'), \
             patch('memory.knowledge_graph_module.GraphDatabase'):
            
            memory_agent = MemoryAgent(
                agent_id="test_memory",
                character_id=character_id,
                llm_client=llm_client
            )
            
            # Mock the store_experience method
            async def mock_store_experience(*args, **kwargs):
                return True
            memory_agent.experience_module.store_experience = Mock(
                return_value=mock_store_experience()
            )
            
            result = await memory_agent.store_interaction_memory(
                user_message="Tell me about Python",
                character_response="Python is a great programming language!",
                character_state=char_state
            )
            
            assert result is True


class TestMemoryUtilities:
    """Test memory utility functions."""
    
    @pytest.mark.asyncio
    async def test_extract_concepts_from_text(self):
        """Test concept extraction from text."""
        from memory.knowledge_graph_module import extract_concepts_from_text
        
        text = "Python is a programming language used for machine learning and data science."
        concepts = await extract_concepts_from_text(text)
        
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        # Should extract concepts like 'python', 'programming', 'science', etc.
        assert any(concept in ['python', 'programming', 'science'] for concept in concepts)
    
    @pytest.mark.asyncio
    async def test_create_experience_from_interaction(self):
        """Test creating experience from interaction."""
        from memory.experience_module import create_experience_from_interaction
        
        experience = await create_experience_from_interaction(
            character_id="test_char",
            user_message="Hello, how are you?",
            character_response="I'm doing great, thanks for asking!",
            emotional_state="happy",
            web_search_data={
                "query": "how to be happy",
                "results": [{"title": "Happiness Guide", "url": "http://example.com"}],
                "knowledge": ["Happiness comes from positive interactions"]
            }
        )
        
        assert experience.character_id == "test_char"
        assert experience.experience_type == "conversation"
        assert "Hello, how are you?" in experience.description
        assert experience.emotional_state == "happy"
        assert experience.web_search_triggered is True
        assert experience.web_search_query == "how to be happy"


def test_memory_imports():
    """Test that all memory modules can be imported."""
    from memory.experience_module import ExperienceModule
    from memory.knowledge_graph_module import KnowledgeGraphModule
    from agents.memory_agent import MemoryAgent
    
    # If we get here without ImportError, test passes
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])