"""
Example Agent Integration with Docker Compose Infrastructure

This script demonstrates how to interact with all services:
- Ollama (LLM)
- ChromaDB (Vector Store)
- Neo4j (Graph Database)
- Redis (Cache)

Configuration is loaded from .env file.
"""

import json
import os
from typing import List, Dict, Any, Optional

import requests
import redis
import chromadb
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AgentInfrastructure:
    """Manages connections to all infrastructure services."""
    
    def __init__(
        self,
        ollama_host: str|None = None,
        chroma_host: str|None = None,
        chroma_port: int|None = None,
        neo4j_uri: str|None = None,
        neo4j_user: str|None = None,
        neo4j_password: str|None = None,
        redis_host: str|None = None,
        redis_port: int|None = None,
    ):
        # Load from environment variables with fallbacks
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Initialize ChromaDB client
        chroma_host = chroma_host or os.getenv("CHROMA_HOST", "localhost")
        chroma_port = chroma_port or int(os.getenv("CHROMA_PORT", "8000"))
        self.chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port
        )
        
        # Initialize Neo4j driver
        neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password123")
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        
        # Initialize Redis client
        redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password if redis_password else None,
            decode_responses=True
        )
        
        # Default model from env
        self.default_model = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2")
    
    def close(self):
        """Close all connections."""
        self.neo4j_driver.close()
        self.redis_client.close()
    
    # Ollama Methods
    def query_llm(self, prompt: str, model: Optional[str] = None) -> str:
        """Query the LLM via Ollama."""
        model = model or self.default_model
        response = requests.post(
            f"{self.ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def get_embeddings(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get embeddings from Ollama."""
        model = model or self.default_model
        response = requests.post(
            f"{self.ollama_host}/api/embeddings",
            json={
                "model": model,
                "prompt": text
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]
    
    # ChromaDB Methods
    def store_embeddings(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Store documents and their embeddings in ChromaDB."""
        collection = self.chroma_client.get_or_create_collection(collection_name)
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def similarity_search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Perform similarity search in ChromaDB."""
        collection = self.chroma_client.get_collection(collection_name)
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    
    # Neo4j Methods
    def create_knowledge_node(
        self,
        label: str,
        properties: Dict[str, Any]
    ) -> int:
        """Create a node in Neo4j knowledge graph."""
        with self.neo4j_driver.session() as session:
            result = session.run(
                f"CREATE (n:{label} $props) RETURN id(n) as node_id",
                props=properties
            )
            return result.single()["node_id"]
    
    def create_relationship(
        self,
        from_id: int,
        to_id: int,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Create a relationship between two nodes."""
        with self.neo4j_driver.session() as session:
            props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()]) if properties else ""
            query = f"""
                MATCH (a), (b)
                WHERE id(a) = $from_id AND id(b) = $to_id
                CREATE (a)-[r:{relationship_type} {{{props_str}}}]->(b)
                RETURN r
            """
            params = {"from_id": from_id, "to_id": to_id}
            if properties:
                params.update(properties)
            session.run(query, **params)
    
    def query_graph(self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None):
        """Execute a Cypher query on Neo4j."""
        with self.neo4j_driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [record.data() for record in result]
    
    # Redis Methods
    def cache_set(self, key: str, value: Any, expire: Optional[int] = None):
        """Set a value in Redis cache."""
        self.redis_client.set(key, json.dumps(value), ex=expire)
    
    def cache_get(self, key: str) -> Any:
        """Get a value from Redis cache."""
        value = self.redis_client.get(key)
        return json.loads(value) if value else None
    
    def cache_exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.redis_client.exists(key) > 0


class SimpleAgent:
    """A simple agent that uses all infrastructure components."""
    
    def __init__(self, infrastructure: AgentInfrastructure):
        self.infra = infrastructure
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query using all infrastructure components."""
        
        # 1. Check cache first
        cache_key = f"query:{hash(query)}"
        cached_result = self.infra.cache_get(cache_key)
        if cached_result:
            print("✓ Retrieved from cache")
            return cached_result
        
        # 2. Perform similarity search in ChromaDB
        print("✓ Searching similar documents...")
        try:
            similar_docs = self.infra.similarity_search(
                "knowledge_base",
                query,
                n_results=3
            )
            context = "\n".join(similar_docs.get("documents", [[]])[0])
        except Exception as e:
            print(f"  No existing collection found: {e}")
            context = ""
        
        # 3. Query Neo4j for related knowledge
        print("✓ Querying knowledge graph...")
        try:
            graph_results = self.infra.query_graph(
                "MATCH (n) WHERE n.content CONTAINS $keyword RETURN n LIMIT 3",
                {"keyword": query.split()[0] if query else ""}
            )
            graph_context = "\n".join([str(r) for r in graph_results])
        except Exception as e:
            print(f"  Graph query failed: {e}")
            graph_context = ""
        
        # 4. Generate response with LLM
        print("✓ Generating LLM response...")
        augmented_prompt = f"""
        Context from similar documents:
        {context}
        
        Context from knowledge graph:
        {graph_context}
        
        User query: {query}
        
        Please provide a helpful response based on the context above.
        """
        
        response = self.infra.query_llm(augmented_prompt)
        
        # 5. Store result in cache
        result = {
            "query": query,
            "response": response,
            "context_used": bool(context or graph_context)
        }
        self.infra.cache_set(cache_key, result, expire=3600)  # 1 hour
        
        # 6. Store query-response pair in ChromaDB for future reference
        print("✓ Storing for future reference...")
        try:
            self.infra.store_embeddings(
                "knowledge_base",
                [f"Q: {query}\nA: {response}"],
                metadatas=[{"type": "qa_pair"}],
                ids=[f"qa_{hash(query)}"]
            )
        except Exception as e:
            print(f"  Storage failed: {e}")
        
        return result


def main():
    """Example usage of the agent infrastructure."""
    
    print("🚀 Initializing Agent Infrastructure...")
    infra = AgentInfrastructure()
    agent = SimpleAgent(infra)
    
    try:
        # Test connection to all services
        print("\n📡 Testing connections...")
        
        # Test Ollama
        print("  • Ollama: ", end="")
        response = requests.get(f"{infra.ollama_host}/api/tags")
        print(f"✓ Connected ({len(response.json().get('models', []))} models available)")
        
        # Test ChromaDB
        print("  • ChromaDB: ", end="")
        infra.chroma_client.heartbeat()
        print("✓ Connected")
        
        # Test Neo4j
        print("  • Neo4j: ", end="")
        with infra.neo4j_driver.session() as session:
            session.run("RETURN 1")
        print("✓ Connected")
        
        # Test Redis
        print("  • Redis: ", end="")
        infra.redis_client.ping()
        print("✓ Connected")
        
        # Example agent workflow
        print("\n🤖 Running example agent workflow...")
        query = "What is machine learning?"
        print(f"\nQuery: {query}\n")
        
        result = agent.process_query(query)
        
        print(f"\n📝 Response:")
        print(f"{result['response'][:500]}...")  # First 500 chars
        
        print("\n✨ Done!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        infra.close()


if __name__ == "__main__":
    # Install required packages:
    # pip install requests redis chromadb-client neo4j
    main()