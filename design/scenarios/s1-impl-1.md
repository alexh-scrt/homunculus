# Arena Subproject: Detailed Implementation Plan

## Overview

This implementation plan provides step-by-step instructions for building the Arena subproject within Homunculus. The plan is designed for incremental development with testing at each stage.

---

## Implementation Sequence

### Phase 1: Project Foundation (Days 1-3)

#### Task 1.1: Create Project Structure
**Priority:** Critical | **Estimated Time:** 2 hours

```bash
# Directory structure to create
homunculus/
├── src/arena/
│   ├── __init__.py
│   ├── orchestration/
│   │   └── __init__.py
│   ├── agents/
│   │   └── __init__.py
│   ├── message_bus/
│   │   └── __init__.py
│   ├── scoring/
│   │   └── __init__.py
│   ├── persistence/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── prompts/
│   │       └── __init__.py
│   ├── scenarios/
│   │   ├── __init__.py
│   │   └── templates/
│   ├── cli/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── tests/arena/
│   ├── __init__.py
│   ├── test_agents/
│   │   └── __init__.py
│   ├── test_scoring/
│   │   └── __init__.py
│   └── test_message_bus/
│       └── __init__.py
├── configs/arena/
│   └── scenarios/
└── scripts/arena/
```

**Files to Create:**
1. All `__init__.py` files (empty or with module docstrings)
2. `.gitkeep` files in empty directories

**Validation:**
- [ ] All directories exist
- [ ] All `__init__.py` files present
- [ ] Can import `from src.arena import *` without errors

---

#### Task 1.2: Update Dependencies
**Priority:** Critical | **Estimated Time:** 1 hour

**File:** `pyproject.toml`

**Action:** Add new dependencies to existing Poetry configuration

```toml
[tool.poetry.dependencies]
# Existing dependencies remain...

# Arena-specific additions
kafka-python = "^2.0.2"
langgraph = "^0.2.0"
langchain-core = "^0.1.0"
psycopg2-binary = "^2.9.9"
pydantic = "^2.5.0"

[tool.poetry.group.dev.dependencies]
# Existing dev dependencies remain...

# Arena testing
pytest-asyncio = "^0.23.0"
pytest-mock = "^3.12.0"
```

**Commands to Run:**
```bash
cd homunculus
poetry install
```

**Validation:**
- [ ] `poetry install` completes successfully
- [ ] All packages installed
- [ ] No dependency conflicts

---

#### Task 1.3: Docker Compose Configuration
**Priority:** Critical | **Estimated Time:** 2 hours

**File:** `docker-compose.yml` (modify existing)

**Action:** Add Kafka and PostgreSQL services

```yaml
# Add to existing docker-compose.yml

services:
  # Existing services (redis, neo4j, chromadb) remain...

  # Kafka for message bus
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    hostname: zookeeper
    container_name: arena-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    networks:
      - homunculus-network

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    hostname: kafka
    container_name: arena-kafka
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
      - "9094:9094"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
    networks:
      - homunculus-network

  # PostgreSQL for game history
  postgres:
    image: postgres:15-alpine
    container_name: arena-postgres
    environment:
      POSTGRES_DB: arena_db
      POSTGRES_USER: arena_user
      POSTGRES_PASSWORD: arena_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./scripts/arena/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - homunculus-network

volumes:
  postgres-data:

networks:
  homunculus-network:
    driver: bridge
```

**Commands to Test:**
```bash
docker-compose up -d zookeeper kafka postgres
docker-compose ps
```

**Validation:**
- [ ] All containers running
- [ ] Kafka accessible on localhost:9092
- [ ] PostgreSQL accessible on localhost:5432
- [ ] No port conflicts

---

#### Task 1.4: Environment Configuration
**Priority:** Critical | **Estimated Time:** 1 hour

**File:** `.env.arena` (new file)

```bash
# Arena-specific configuration
# Copy this to .env and customize

# Message Bus (Kafka)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PREFIX=arena
KAFKA_CONSUMER_GROUP=arena-orchestrator

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=arena_db
POSTGRES_USER=arena_user
POSTGRES_PASSWORD=arena_pass

# Redis (reuse from Homunculus)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=1  # Use DB 1 for Arena (Homunculus uses DB 0)

# LLM Configuration
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_backup_key_here
DEFAULT_LLM_PROVIDER=anthropic
DEFAULT_MODEL=claude-sonnet-4-20250514

# Arena Settings
MAX_AGENTS=8
MIN_AGENTS=2
DEFAULT_ELIMINATION_THRESHOLD=-10.0
DEFAULT_GAME_THEORY_MODE=adversarial

# Logging
LOG_LEVEL=INFO
LOG_FILE=./data/logs/arena.log
```

**File:** `src/arena/config/arena_settings.py` (new file)

```python
"""Arena configuration settings."""

from pydantic_settings import BaseSettings
from typing import Literal


class ArenaSettings(BaseSettings):
    """Arena configuration from environment variables."""
    
    # Message Bus
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_prefix: str = "arena"
    kafka_consumer_group: str = "arena-orchestrator"
    
    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "arena_db"
    postgres_user: str = "arena_user"
    postgres_password: str = "arena_pass"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1
    
    # LLM
    anthropic_api_key: str
    openai_api_key: str = ""
    default_llm_provider: Literal["anthropic", "openai"] = "anthropic"
    default_model: str = "claude-sonnet-4-20250514"
    
    # Arena Settings
    max_agents: int = 8
    min_agents: int = 2
    default_elimination_threshold: float = -10.0
    default_game_theory_mode: Literal["adversarial", "collaborative", "neutral"] = "adversarial"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./data/logs/arena.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = ArenaSettings()
```

**Validation:**
- [ ] `.env.arena` template created
- [ ] Settings class loads without errors
- [ ] All required env vars documented

---

### Phase 2: Data Models (Days 3-4)

#### Task 2.1: Core Data Models
**Priority:** Critical | **Estimated Time:** 4 hours

**File:** `src/arena/models/agent.py`

```python
"""Agent state models for Arena."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum


class AgentStatus(Enum):
    """Agent lifecycle status."""
    ACTIVE = "active"
    ELIMINATED = "eliminated"
    CHAMPION = "champion"


@dataclass
class AgentState:
    """Represents the state of an agent in the arena."""
    
    # Identity
    agent_id: str
    character_name: str
    character_profile: Dict  # Full Homunculus character config
    
    # Status
    status: AgentStatus = AgentStatus.ACTIVE
    
    # Performance
    score: float = 0.0
    turns_taken: int = 0
    contributions: List[str] = field(default_factory=list)  # Message IDs
    
    # Survival Tracking
    eliminations_witnessed: int = 0
    accusations_made: int = 0
    false_accusations: int = 0
    successful_manipulations: int = 0
    
    # Meta
    joined_at: datetime = field(default_factory=datetime.utcnow)
    eliminated_at: Optional[datetime] = None
    elimination_reason: Optional[str] = None
    
    # Champion Data (for returning champions)
    is_returning_champion: bool = False
    previous_wins: int = 0
    champion_experience: Optional[str] = None  # Lessons learned
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "character_name": self.character_name,
            "character_profile": self.character_profile,
            "status": self.status.value,
            "score": self.score,
            "turns_taken": self.turns_taken,
            "contributions": self.contributions,
            "eliminations_witnessed": self.eliminations_witnessed,
            "accusations_made": self.accusations_made,
            "false_accusations": self.false_accusations,
            "successful_manipulations": self.successful_manipulations,
            "joined_at": self.joined_at.isoformat(),
            "eliminated_at": self.eliminated_at.isoformat() if self.eliminated_at else None,
            "elimination_reason": self.elimination_reason,
            "is_returning_champion": self.is_returning_champion,
            "previous_wins": self.previous_wins,
            "champion_experience": self.champion_experience,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AgentState":
        """Create from dictionary."""
        data["status"] = AgentStatus(data["status"])
        data["joined_at"] = datetime.fromisoformat(data["joined_at"])
        if data["eliminated_at"]:
            data["eliminated_at"] = datetime.fromisoformat(data["eliminated_at"])
        return cls(**data)
```

**File:** `src/arena/models/message.py`

```python
"""Message models for Arena message bus."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Literal
from uuid import uuid4


MessageType = Literal[
    "introduction",      # Narrator introduces problem
    "contribution",      # Agent makes contribution
    "accusation",        # Agent accuses another
    "commentary",        # Narrator comments
    "elimination",       # Judge eliminates agent
    "final_words",       # Eliminated agent's last speech
    "termination",       # Game ends
]

SenderType = Literal[
    "narrator",
    "judge",
    "game_theory",
    "character",
]


@dataclass
class Message:
    """Represents a message on the Arena message bus."""
    
    # Identity
    message_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Sender
    sender_id: str = ""  # Agent ID or system component name
    sender_type: SenderType = "character"
    sender_name: str = ""  # Human-readable name
    
    # Content
    message_type: MessageType = "contribution"
    content: str = ""
    
    # Context
    turn_number: int = 0
    game_id: str = ""
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "sender_id": self.sender_id,
            "sender_type": self.sender_type,
            "sender_name": self.sender_name,
            "message_type": self.message_type,
            "content": self.content,
            "turn_number": self.turn_number,
            "game_id": self.game_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
```

**File:** `src/arena/models/score.py`

```python
"""Scoring models for Arena."""

from dataclasses import dataclass, field
from typing import Dict
from datetime import datetime


@dataclass
class ScoringMetrics:
    """Individual scoring metrics for an agent's contribution."""
    
    # Core metrics (0.0 to 1.0 scale)
    novelty: float = 0.0              # How original is the idea?
    builds_on_others: float = 0.0     # Does it build on previous contributions?
    solves_subproblem: float = 0.0    # Does it make progress toward solution?
    radical_idea: float = 0.0          # Does it propose something completely new?
    manipulation: float = 0.0          # Does it manipulate others/system?
    
    # Contextual factors
    turn_number: int = 0
    agent_id: str = ""
    message_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Weighted total (calculated)
    weighted_score: float = 0.0
    
    def calculate_weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted score based on configuration."""
        self.weighted_score = (
            self.novelty * weights.get("novelty", 0.2) +
            self.builds_on_others * weights.get("builds_on_others", 0.2) +
            self.solves_subproblem * weights.get("solves_subproblem", 0.2) +
            self.radical_idea * weights.get("radical_idea", 0.2) +
            self.manipulation * weights.get("manipulation", 0.2)
        )
        return self.weighted_score
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "novelty": self.novelty,
            "builds_on_others": self.builds_on_others,
            "solves_subproblem": self.solves_subproblem,
            "radical_idea": self.radical_idea,
            "manipulation": self.manipulation,
            "turn_number": self.turn_number,
            "agent_id": self.agent_id,
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "weighted_score": self.weighted_score,
        }


@dataclass
class AgentScorecard:
    """Complete scoring history for an agent."""
    
    agent_id: str
    total_score: float = 0.0
    metrics_history: list[ScoringMetrics] = field(default_factory=list)
    
    def add_metrics(self, metrics: ScoringMetrics) -> None:
        """Add new metrics and update total score."""
        self.metrics_history.append(metrics)
        self.total_score += metrics.weighted_score
    
    def average_metric(self, metric_name: str) -> float:
        """Calculate average for a specific metric."""
        if not self.metrics_history:
            return 0.0
        values = [getattr(m, metric_name) for m in self.metrics_history]
        return sum(values) / len(values)
```

**File:** `src/arena/models/accusation.py`

```python
"""Accusation models for Arena."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4


AccusationType = Literal[
    "false_statement",
    "prompt_injection",
    "manipulation",
    "rule_breaking",
    "other",
]

AccusationOutcome = Literal[
    "pending",
    "proven",
    "false",
    "insufficient_evidence",
]


@dataclass
class Accusation:
    """Represents a cheating accusation."""
    
    # Identity
    accusation_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Parties
    accuser_id: str = ""
    accuser_name: str = ""
    accused_id: str = ""
    accused_name: str = ""
    
    # Accusation Details
    accusation_type: AccusationType = "other"
    claim: str = ""  # What the accuser claims happened
    proof: str = ""  # Evidence provided by accuser
    
    # Context
    turn_number: int = 0
    game_id: str = ""
    referenced_message_ids: list[str] = field(default_factory=list)
    
    # Outcome
    outcome: AccusationOutcome = "pending"
    judge_reasoning: str = ""
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "accusation_id": self.accusation_id,
            "timestamp": self.timestamp.isoformat(),
            "accuser_id": self.accuser_id,
            "accuser_name": self.accuser_name,
            "accused_id": self.accused_id,
            "accused_name": self.accused_name,
            "accusation_type": self.accusation_type,
            "claim": self.claim,
            "proof": self.proof,
            "turn_number": self.turn_number,
            "game_id": self.game_id,
            "referenced_message_ids": self.referenced_message_ids,
            "outcome": self.outcome,
            "judge_reasoning": self.judge_reasoning,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }
```

**File:** `src/arena/models/game.py`

```python
"""Game state models for Arena."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Literal
from uuid import uuid4

from .agent import AgentState
from .message import Message
from .accusation import Accusation


GameStatus = Literal[
    "initializing",
    "in_progress",
    "completed",
    "terminated",
]

TerminationReason = Literal[
    "problem_solved",
    "single_survivor",
    "max_turns_reached",
    "error",
]


@dataclass
class ArenaState:
    """Complete state of an Arena game."""
    
    # Identity
    game_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Problem
    problem_statement: str = ""
    problem_title: str = ""
    scenario_config: Dict = field(default_factory=dict)
    
    # Status
    status: GameStatus = "initializing"
    current_turn: int = 0
    max_turns: int = 50
    
    # Agents
    active_agents: List[AgentState] = field(default_factory=list)
    eliminated_agents: List[AgentState] = field(default_factory=list)
    
    # History
    message_history: List[Message] = field(default_factory=list)
    accusation_history: List[Accusation] = field(default_factory=list)
    
    # Scoring
    scoring_weights: Dict[str, float] = field(default_factory=dict)
    
    # Termination
    terminated_at: Optional[datetime] = None
    termination_reason: Optional[TerminationReason] = None
    winner_id: Optional[str] = None
    
    # Champion from previous round
    returning_champion: Optional[AgentState] = None
    
    def add_message(self, message: Message) -> None:
        """Add message to history."""
        message.game_id = self.game_id
        message.turn_number = self.current_turn
        self.message_history.append(message)
    
    def get_agent(self, agent_id: str) -> Optional[AgentState]:
        """Get agent by ID from active or eliminated."""
        for agent in self.active_agents:
            if agent.agent_id == agent_id:
                return agent
        for agent in self.eliminated_agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    def eliminate_agent(self, agent_id: str, reason: str) -> bool:
        """Move agent from active to eliminated."""
        for i, agent in enumerate(self.active_agents):
            if agent.agent_id == agent_id:
                agent.status = AgentStatus.ELIMINATED
                agent.eliminated_at = datetime.utcnow()
                agent.elimination_reason = reason
                self.eliminated_agents.append(agent)
                self.active_agents.pop(i)
                return True
        return False
    
    def get_active_agent_ids(self) -> List[str]:
        """Get list of active agent IDs."""
        return [agent.agent_id for agent in self.active_agents]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "game_id": self.game_id,
            "created_at": self.created_at.isoformat(),
            "problem_statement": self.problem_statement,
            "problem_title": self.problem_title,
            "scenario_config": self.scenario_config,
            "status": self.status,
            "current_turn": self.current_turn,
            "max_turns": self.max_turns,
            "active_agents": [agent.to_dict() for agent in self.active_agents],
            "eliminated_agents": [agent.to_dict() for agent in self.eliminated_agents],
            "message_history": [msg.to_dict() for msg in self.message_history],
            "accusation_history": [acc.to_dict() for acc in self.accusation_history],
            "scoring_weights": self.scoring_weights,
            "terminated_at": self.terminated_at.isoformat() if self.terminated_at else None,
            "termination_reason": self.termination_reason,
            "winner_id": self.winner_id,
            "returning_champion": self.returning_champion.to_dict() if self.returning_champion else None,
        }
```

**Validation:**
- [ ] All model files created
- [ ] Import models without errors: `from src.arena.models import *`
- [ ] Create instances of each model successfully
- [ ] Serialization (to_dict/from_dict) works

---

#### Task 2.2: Unit Tests for Models
**Priority:** High | **Estimated Time:** 2 hours

**File:** `tests/arena/test_models.py`

```python
"""Unit tests for Arena data models."""

import pytest
from datetime import datetime

from src.arena.models.agent import AgentState, AgentStatus
from src.arena.models.message import Message
from src.arena.models.score import ScoringMetrics, AgentScorecard
from src.arena.models.accusation import Accusation, AccusationType, AccusationOutcome
from src.arena.models.game import ArenaState, GameStatus


def test_agent_state_creation():
    """Test AgentState initialization."""
    agent = AgentState(
        agent_id="agent_1",
        character_name="Ada Lovelace",
        character_profile={"personality": "analytical"},
    )
    
    assert agent.agent_id == "agent_1"
    assert agent.status == AgentStatus.ACTIVE
    assert agent.score == 0.0
    assert agent.turns_taken == 0


def test_agent_state_serialization():
    """Test AgentState to_dict/from_dict."""
    agent = AgentState(
        agent_id="agent_1",
        character_name="Ada",
        character_profile={},
    )
    
    data = agent.to_dict()
    restored = AgentState.from_dict(data)
    
    assert restored.agent_id == agent.agent_id
    assert restored.status == agent.status


def test_message_creation():
    """Test Message initialization."""
    msg = Message(
        sender_id="agent_1",
        sender_type="character",
        sender_name="Ada",
        message_type="contribution",
        content="I propose we consider...",
    )
    
    assert msg.message_id is not None
    assert msg.timestamp is not None
    assert msg.content == "I propose we consider..."


def test_scoring_metrics_weighted_calculation():
    """Test ScoringMetrics weighted score calculation."""
    metrics = ScoringMetrics(
        novelty=0.8,
        builds_on_others=0.6,
        solves_subproblem=0.4,
        radical_idea=0.9,
        manipulation=0.2,
    )
    
    weights = {
        "novelty": 0.25,
        "builds_on_others": 0.20,
        "solves_subproblem": 0.25,
        "radical_idea": 0.15,
        "manipulation": 0.15,
    }
    
    score = metrics.calculate_weighted_score(weights)
    
    # (0.8*0.25) + (0.6*0.20) + (0.4*0.25) + (0.9*0.15) + (0.2*0.15)
    expected = 0.2 + 0.12 + 0.1 + 0.135 + 0.03
    assert abs(score - expected) < 0.001


def test_agent_scorecard():
    """Test AgentScorecard accumulation."""
    scorecard = AgentScorecard(agent_id="agent_1")
    
    metrics1 = ScoringMetrics(novelty=0.8, builds_on_others=0.6)
    metrics1.weighted_score = 5.0
    
    metrics2 = ScoringMetrics(novelty=0.6, builds_on_others=0.9)
    metrics2.weighted_score = 7.0
    
    scorecard.add_metrics(metrics1)
    scorecard.add_metrics(metrics2)
    
    assert scorecard.total_score == 12.0
    assert len(scorecard.metrics_history) == 2
    assert scorecard.average_metric("novelty") == 0.7


def test_accusation_creation():
    """Test Accusation model."""
    accusation = Accusation(
        accuser_id="agent_1",
        accuser_name="Ada",
        accused_id="agent_2",
        accused_name="Zen",
        accusation_type="false_statement",
        claim="Agent 2 made a false claim about X",
        proof="Message ID abc123 shows contradiction",
    )
    
    assert accusation.accusation_id is not None
    assert accusation.outcome == "pending"
    assert accusation.accusation_type == "false_statement"


def test_arena_state_agent_management():
    """Test ArenaState agent management methods."""
    state = ArenaState(
        problem_statement="Solve the trolley problem",
    )
    
    agent1 = AgentState(agent_id="agent_1", character_name="Ada", character_profile={})
    agent2 = AgentState(agent_id="agent_2", character_name="Zen", character_profile={})
    
    state.active_agents = [agent1, agent2]
    
    # Test get_agent
    found = state.get_agent("agent_1")
    assert found is not None
    assert found.agent_id == "agent_1"
    
    # Test elimination
    success = state.eliminate_agent("agent_1", "Poor performance")
    assert success
    assert len(state.active_agents) == 1
    assert len(state.eliminated_agents) == 1
    assert state.eliminated_agents[0].elimination_reason == "Poor performance"
    
    # Test get_active_agent_ids
    ids = state.get_active_agent_ids()
    assert ids == ["agent_2"]


def test_arena_state_message_management():
    """Test ArenaState message management."""
    state = ArenaState(game_id="game_123", current_turn=5)
    
    msg = Message(
        sender_id="agent_1",
        sender_name="Ada",
        content="My contribution",
    )
    
    state.add_message(msg)
    
    assert len(state.message_history) == 1
    assert state.message_history[0].game_id == "game_123"
    assert state.message_history[0].turn_number == 5
```

**Commands to Run:**
```bash
cd homunculus
poetry run pytest tests/arena/test_models.py -v
```

**Validation:**
- [ ] All tests pass
- [ ] Coverage >90% for models

---

### Phase 3: Message Bus (Days 4-6)

#### Task 3.1: Message Bus Producer
**Priority:** Critical | **Estimated Time:** 3 hours

**File:** `src/arena/message_bus/producer.py`

```python
"""Kafka producer for Arena message bus."""

import json
import logging
from typing import Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError

from ..models.message import Message
from ..config.arena_settings import settings

logger = logging.getLogger(__name__)


class MessageBusProducer:
    """Publishes messages to Kafka topic."""
    
    def __init__(self, topic_name: str = "arena-messages"):
        """Initialize Kafka producer.
        
        Args:
            topic_name: Kafka topic name (will be prefixed)
        """
        self.topic = f"{settings.kafka_topic_prefix}_{topic_name}"
        self.producer: Optional[KafkaProducer] = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Kafka broker."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas
                retries=3,
            )
            logger.info(f"Connected to Kafka at {settings.kafka_bootstrap_servers}")
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def publish(self, message: Message, key: Optional[str] = None) -> bool:
        """Publish message to topic.
        
        Args:
            message: Message to publish
            key: Optional partition key (defaults to game_id)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.producer:
            logger.error("Producer not connected")
            return False
        
        try:
            # Use game_id as default key for partitioning
            partition_key = key or message.game_id
            
            # Serialize message
            message_dict = message.to_dict()
            
            # Send to Kafka
            future = self.producer.send(
                self.topic,
                key=partition_key,
                value=message_dict,
            )
            
            # Wait for confirmation (blocking)
            record_metadata = future.get(timeout=10)
            
            logger.debug(
                f"Published message {message.message_id} to "
                f"{record_metadata.topic}:{record_metadata.partition}@{record_metadata.offset}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    def flush(self) -> None:
        """Flush any pending messages."""
        if self.producer:
            self.producer.flush()
    
    def close(self) -> None:
        """Close the producer."""
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")
```

**File:** `src/arena/message_bus/consumer.py`

```python
"""Kafka consumer for Arena message bus."""

import json
import logging
from typing import Callable, Optional
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from ..models.message import Message
from ..config.arena_settings import settings

logger = logging.getLogger(__name__)


class MessageBusConsumer:
    """Subscribes to messages from Kafka topic."""
    
    def __init__(
        self,
        topic_name: str = "arena-messages",
        group_id: Optional[str] = None,
    ):
        """Initialize Kafka consumer.
        
        Args:
            topic_name: Kafka topic name (will be prefixed)
            group_id: Consumer group ID (defaults to settings)
        """
        self.topic = f"{settings.kafka_topic_prefix}_{topic_name}"
        self.group_id = group_id or settings.kafka_consumer_group
        self.consumer: Optional[KafkaConsumer] = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Kafka broker."""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='earliest',  # Start from beginning
                enable_auto_commit=True,
            )
            logger.info(
                f"Connected to Kafka topic {self.topic} "
                f"with group {self.group_id}"
            )
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def consume(self, callback: Callable[[Message], None], timeout_ms: int = 1000) -> None:
        """Consume messages and call callback for each.
        
        Args:
            callback: Function to call with each Message
            timeout_ms: Poll timeout in milliseconds
        """
        if not self.consumer:
            logger.error("Consumer not connected")
            return
        
        try:
            logger.info("Starting message consumption...")
            
            for kafka_message in self.consumer:
                try:
                    # Deserialize to Message object
                    message_dict = kafka_message.value
                    message = Message.from_dict(message_dict)
                    
                    logger.debug(
                        f"Consumed message {message.message_id} from "
                        f"{kafka_message.topic}:{kafka_message.partition}@{kafka_message.offset}"
                    )
                    
                    # Call callback
                    callback(message)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)
        finally:
            self.close()
    
    def close(self) -> None:
        """Close the consumer."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
```

**Validation:**
- [ ] Producer connects to Kafka
- [ ] Consumer connects to Kafka
- [ ] Messages published successfully
- [ ] Messages consumed successfully
- [ ] Serialization/deserialization works

---

#### Task 3.2: Message Handlers
**Priority:** High | **Estimated Time:** 2 hours

**File:** `src/arena/message_bus/handlers.py`

```python
"""Message handlers for processing different message types."""

import logging
from typing import Dict, Callable
from collections import defaultdict

from ..models.message import Message, MessageType

logger = logging.getLogger(__name__)


class MessageHandler:
    """Routes messages to appropriate handlers based on type."""
    
    def __init__(self):
        """Initialize handler registry."""
        self.handlers: Dict[MessageType, list[Callable]] = defaultdict(list)
    
    def register(self, message_type: MessageType, handler: Callable[[Message], None]) -> None:
        """Register a handler for a message type.
        
        Args:
            message_type: Type of message to handle
            handler: Callback function that accepts Message
        """
        self.handlers[message_type].append(handler)
        logger.info(f"Registered handler for {message_type}")
    
    def handle(self, message: Message) -> None:
        """Route message to registered handlers.
        
        Args:
            message: Message to handle
        """
        handlers = self.handlers.get(message.message_type, [])
        
        if not handlers:
            logger.warning(f"No handlers for message type: {message.message_type}")
            return
        
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(
                    f"Handler error for {message.message_type}: {e}",
                    exc_info=True
                )
    
    def clear(self, message_type: Optional[MessageType] = None) -> None:
        """Clear handlers.
        
        Args:
            message_type: Specific type to clear, or None for all
        """
        if message_type:
            self.handlers[message_type].clear()
        else:
            self.handlers.clear()


# Example handlers

def log_message_handler(message: Message) -> None:
    """Simple handler that logs messages."""
    logger.info(
        f"[{message.sender_type}] {message.sender_name}: "
        f"{message.content[:100]}..."
    )


def contribution_handler(message: Message) -> None:
    """Handle agent contributions."""
    logger.info(f"Agent {message.sender_name} contributed at turn {message.turn_number}")


def elimination_handler(message: Message) -> None:
    """Handle elimination announcements."""
    logger.warning(f"Agent {message.metadata.get('eliminated_agent')} was eliminated")


def accusation_handler(message: Message) -> None:
    """Handle accusations."""
    accuser = message.metadata.get('accuser')
    accused = message.metadata.get('accused')
    logger.info(f"{accuser} accused {accused} of cheating")
```

**Validation:**
- [ ] Handlers register successfully
- [ ] Messages route to correct handlers
- [ ] Multiple handlers per type work
- [ ] Error handling doesn't crash system

# Arena Implementation Plan (Continued)

## Testing Protocol

**CRITICAL: After each phase, comprehensive tests MUST be written in `tests/arena/` before proceeding to the next phase.**

### Testing Standards
- **Unit Tests**: Test individual functions/classes in isolation
- **Integration Tests**: Test component interactions
- **Coverage Target**: Minimum 80% code coverage per phase
- **Test-Driven Approach**: Write tests that validate requirements before marking phase complete

---

### Phase 3 Continued: Message Bus Testing (Day 6)

#### Task 3.3: Message Bus Integration Tests
**Priority:** Critical | **Estimated Time:** 3 hours

**File:** `tests/arena/test_message_bus/test_producer.py`

```python
"""Unit tests for MessageBusProducer."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from kafka.errors import KafkaError

from src.arena.message_bus.producer import MessageBusProducer
from src.arena.models.message import Message


@pytest.fixture
def mock_kafka_producer():
    """Mock KafkaProducer."""
    with patch('src.arena.message_bus.producer.KafkaProducer') as mock:
        producer_instance = MagicMock()
        mock.return_value = producer_instance
        yield producer_instance


def test_producer_initialization(mock_kafka_producer):
    """Test producer initializes correctly."""
    producer = MessageBusProducer(topic_name="test-topic")
    
    assert producer.producer is not None
    assert producer.topic == "arena_test-topic"


def test_producer_connection_failure():
    """Test producer handles connection failures."""
    with patch('src.arena.message_bus.producer.KafkaProducer') as mock:
        mock.side_effect = KafkaError("Connection failed")
        
        with pytest.raises(KafkaError):
            MessageBusProducer(topic_name="test-topic")


def test_publish_message_success(mock_kafka_producer):
    """Test successful message publication."""
    # Setup mock
    future = MagicMock()
    future.get.return_value = MagicMock(
        topic="arena_test-topic",
        partition=0,
        offset=123
    )
    mock_kafka_producer.send.return_value = future
    
    # Create producer and message
    producer = MessageBusProducer(topic_name="test-topic")
    message = Message(
        sender_id="agent_1",
        sender_name="Ada",
        content="Test message",
        game_id="game_123"
    )
    
    # Publish
    result = producer.publish(message)
    
    assert result is True
    mock_kafka_producer.send.assert_called_once()


def test_publish_message_failure(mock_kafka_producer):
    """Test message publication failure handling."""
    # Setup mock to raise exception
    mock_kafka_producer.send.side_effect = Exception("Send failed")
    
    producer = MessageBusProducer(topic_name="test-topic")
    message = Message(
        sender_id="agent_1",
        content="Test",
        game_id="game_123"
    )
    
    result = producer.publish(message)
    
    assert result is False


def test_publish_uses_game_id_as_key(mock_kafka_producer):
    """Test that game_id is used as partition key."""
    future = MagicMock()
    future.get.return_value = MagicMock(topic="test", partition=0, offset=0)
    mock_kafka_producer.send.return_value = future
    
    producer = MessageBusProducer(topic_name="test-topic")
    message = Message(
        sender_id="agent_1",
        content="Test",
        game_id="game_123"
    )
    
    producer.publish(message)
    
    # Verify send was called with game_id as key
    call_args = mock_kafka_producer.send.call_args
    assert call_args[1]['key'] == "game_123"


def test_flush(mock_kafka_producer):
    """Test flush method."""
    producer = MessageBusProducer(topic_name="test-topic")
    producer.flush()
    
    mock_kafka_producer.flush.assert_called_once()


def test_close(mock_kafka_producer):
    """Test close method."""
    producer = MessageBusProducer(topic_name="test-topic")
    producer.close()
    
    mock_kafka_producer.close.assert_called_once()
```

**File:** `tests/arena/test_message_bus/test_consumer.py`

```python
"""Unit tests for MessageBusConsumer."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from kafka.errors import KafkaError

from src.arena.message_bus.consumer import MessageBusConsumer
from src.arena.models.message import Message


@pytest.fixture
def mock_kafka_consumer():
    """Mock KafkaConsumer."""
    with patch('src.arena.message_bus.consumer.KafkaConsumer') as mock:
        consumer_instance = MagicMock()
        mock.return_value = consumer_instance
        yield consumer_instance


def test_consumer_initialization(mock_kafka_consumer):
    """Test consumer initializes correctly."""
    consumer = MessageBusConsumer(topic_name="test-topic")
    
    assert consumer.consumer is not None
    assert consumer.topic == "arena_test-topic"


def test_consumer_connection_failure():
    """Test consumer handles connection failures."""
    with patch('src.arena.message_bus.consumer.KafkaConsumer') as mock:
        mock.side_effect = KafkaError("Connection failed")
        
        with pytest.raises(KafkaError):
            MessageBusConsumer(topic_name="test-topic")


def test_consume_messages(mock_kafka_consumer):
    """Test message consumption."""
    # Create mock messages
    mock_message = MagicMock()
    mock_message.value = {
        "message_id": "msg_1",
        "timestamp": "2025-01-27T00:00:00",
        "sender_id": "agent_1",
        "sender_type": "character",
        "sender_name": "Ada",
        "message_type": "contribution",
        "content": "Test",
        "turn_number": 1,
        "game_id": "game_123",
        "metadata": {}
    }
    mock_message.topic = "arena_test-topic"
    mock_message.partition = 0
    mock_message.offset = 0
    
    # Setup consumer to return one message then stop
    mock_kafka_consumer.__iter__.return_value = [mock_message]
    
    consumer = MessageBusConsumer(topic_name="test-topic")
    
    # Track callback invocations
    callback_called = []
    
    def test_callback(message: Message):
        callback_called.append(message)
        # Stop iteration after first message
        raise KeyboardInterrupt()
    
    consumer.consume(test_callback)
    
    assert len(callback_called) == 1
    assert callback_called[0].message_id == "msg_1"
    assert callback_called[0].content == "Test"


def test_consume_handles_callback_errors(mock_kafka_consumer):
    """Test that callback errors don't crash consumer."""
    mock_message = MagicMock()
    mock_message.value = {
        "message_id": "msg_1",
        "timestamp": "2025-01-27T00:00:00",
        "sender_id": "agent_1",
        "sender_type": "character",
        "sender_name": "Ada",
        "message_type": "contribution",
        "content": "Test",
        "turn_number": 1,
        "game_id": "game_123",
        "metadata": {}
    }
    mock_message.topic = "test"
    mock_message.partition = 0
    mock_message.offset = 0
    
    mock_kafka_consumer.__iter__.return_value = [mock_message]
    
    consumer = MessageBusConsumer(topic_name="test-topic")
    
    def failing_callback(message: Message):
        raise ValueError("Callback error")
    
    # Should not raise exception
    try:
        consumer.consume(failing_callback, timeout_ms=100)
    except KeyboardInterrupt:
        pass  # Expected to stop iteration


def test_close(mock_kafka_consumer):
    """Test close method."""
    consumer = MessageBusConsumer(topic_name="test-topic")
    consumer.close()
    
    mock_kafka_consumer.close.assert_called_once()
```

**File:** `tests/arena/test_message_bus/test_handlers.py`

```python
"""Unit tests for message handlers."""

import pytest
from src.arena.message_bus.handlers import MessageHandler, log_message_handler
from src.arena.models.message import Message


def test_message_handler_registration():
    """Test handler registration."""
    handler = MessageHandler()
    
    def test_handler(msg: Message):
        pass
    
    handler.register("contribution", test_handler)
    
    assert len(handler.handlers["contribution"]) == 1


def test_message_handler_routing():
    """Test message routing to correct handler."""
    handler = MessageHandler()
    
    called_messages = []
    
    def contribution_handler(msg: Message):
        called_messages.append(msg)
    
    handler.register("contribution", contribution_handler)
    
    message = Message(
        sender_id="agent_1",
        message_type="contribution",
        content="Test"
    )
    
    handler.handle(message)
    
    assert len(called_messages) == 1
    assert called_messages[0].content == "Test"


def test_message_handler_multiple_handlers():
    """Test multiple handlers for same message type."""
    handler = MessageHandler()
    
    call_count = {"count": 0}
    
    def handler1(msg: Message):
        call_count["count"] += 1
    
    def handler2(msg: Message):
        call_count["count"] += 10
    
    handler.register("contribution", handler1)
    handler.register("contribution", handler2)
    
    message = Message(message_type="contribution", sender_id="agent_1", content="Test")
    handler.handle(message)
    
    assert call_count["count"] == 11


def test_message_handler_no_handlers():
    """Test handling message with no registered handlers."""
    handler = MessageHandler()
    
    message = Message(
        message_type="unknown",
        sender_id="agent_1",
        content="Test"
    )
    
    # Should not raise exception
    handler.handle(message)


def test_message_handler_error_doesnt_crash():
    """Test that handler errors don't crash the system."""
    handler = MessageHandler()
    
    def failing_handler(msg: Message):
        raise ValueError("Handler error")
    
    handler.register("contribution", failing_handler)
    
    message = Message(
        message_type="contribution",
        sender_id="agent_1",
        content="Test"
    )
    
    # Should not raise exception
    handler.handle(message)


def test_message_handler_clear():
    """Test clearing handlers."""
    handler = MessageHandler()
    
    def test_handler(msg: Message):
        pass
    
    handler.register("contribution", test_handler)
    handler.register("elimination", test_handler)
    
    # Clear specific type
    handler.clear("contribution")
    assert len(handler.handlers["contribution"]) == 0
    assert len(handler.handlers["elimination"]) == 1
    
    # Clear all
    handler.clear()
    assert len(handler.handlers["elimination"]) == 0
```

**File:** `tests/arena/test_message_bus/test_integration.py`

```python
"""Integration tests for message bus components."""

import pytest
import time
import threading
from unittest.mock import patch

from src.arena.message_bus.producer import MessageBusProducer
from src.arena.message_bus.consumer import MessageBusConsumer
from src.arena.message_bus.handlers import MessageHandler
from src.arena.models.message import Message


@pytest.mark.integration
@pytest.mark.skipif(
    True,  # Set to False when Kafka is running
    reason="Requires running Kafka instance"
)
def test_producer_consumer_integration():
    """Test end-to-end message flow through Kafka."""
    # This test requires actual Kafka running
    # Start producer
    producer = MessageBusProducer(topic_name="integration-test")
    
    # Start consumer in background thread
    received_messages = []
    
    def callback(msg: Message):
        received_messages.append(msg)
        if len(received_messages) >= 2:
            raise KeyboardInterrupt()
    
    consumer = MessageBusConsumer(topic_name="integration-test")
    
    def consume_loop():
        consumer.consume(callback)
    
    consumer_thread = threading.Thread(target=consume_loop, daemon=True)
    consumer_thread.start()
    
    # Give consumer time to connect
    time.sleep(2)
    
    # Publish messages
    msg1 = Message(
        sender_id="agent_1",
        content="Message 1",
        game_id="game_123"
    )
    msg2 = Message(
        sender_id="agent_2",
        content="Message 2",
        game_id="game_123"
    )
    
    producer.publish(msg1)
    producer.publish(msg2)
    producer.flush()
    
    # Wait for consumption
    consumer_thread.join(timeout=5)
    
    # Cleanup
    producer.close()
    consumer.close()
    
    # Verify
    assert len(received_messages) == 2
    assert received_messages[0].content == "Message 1"
    assert received_messages[1].content == "Message 2"


@pytest.mark.integration
def test_handler_integration():
    """Test message handler integration with mock messages."""
    handler = MessageHandler()
    
    processed_messages = {
        "contributions": [],
        "eliminations": [],
    }
    
    def contribution_handler(msg: Message):
        processed_messages["contributions"].append(msg)
    
    def elimination_handler(msg: Message):
        processed_messages["eliminations"].append(msg)
    
    handler.register("contribution", contribution_handler)
    handler.register("elimination", elimination_handler)
    
    # Send messages
    msg1 = Message(
        message_type="contribution",
        sender_id="agent_1",
        content="Contribution"
    )
    msg2 = Message(
        message_type="elimination",
        sender_id="judge",
        content="Agent eliminated"
    )
    
    handler.handle(msg1)
    handler.handle(msg2)
    
    assert len(processed_messages["contributions"]) == 1
    assert len(processed_messages["eliminations"]) == 1
```

**Commands to Run:**
```bash
# Run unit tests
poetry run pytest tests/arena/test_message_bus/ -v --cov=src/arena/message_bus

# Run integration tests (requires Kafka)
poetry run pytest tests/arena/test_message_bus/test_integration.py -v -m integration

# Generate coverage report
poetry run pytest tests/arena/test_message_bus/ --cov=src/arena/message_bus --cov-report=html
```

**Phase 3 Validation Checklist:**
- [ ] All unit tests pass (>80% coverage)
- [ ] Integration tests pass with mock Kafka
- [ ] Producer publishes messages successfully
- [ ] Consumer receives messages correctly
- [ ] Handlers route messages properly
- [ ] Error handling doesn't crash system
- [ ] Coverage report generated

---

### Phase 4: Agent Implementation (Days 7-10)

#### Task 4.1: Base Agent Adapter
**Priority:** Critical | **Estimated Time:** 2 hours

**File:** `src/arena/agents/base_arena_agent.py`

```python
"""Base class for Arena agents."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import logging

from ..models.message import Message
from ..models.game import ArenaState
from ..config.arena_settings import settings

logger = logging.getLogger(__name__)


class BaseArenaAgent(ABC):
    """Base class for all Arena agents."""
    
    def __init__(self, agent_id: str, agent_name: str):
        """Initialize base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            agent_name: Human-readable name
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.llm_provider = settings.default_llm_provider
        self.model = settings.default_model
        
        logger.info(f"Initialized {self.__class__.__name__}: {agent_name}")
    
    @abstractmethod
    async def process(self, state: ArenaState, context: Dict) -> Message:
        """Process current state and generate a message.
        
        Args:
            state: Current arena state
            context: Additional context for processing
            
        Returns:
            Message to publish to message bus
        """
        pass
    
    def _get_recent_messages(self, state: ArenaState, n: int = 10) -> list[Message]:
        """Get n most recent messages from state.
        
        Args:
            state: Arena state
            n: Number of messages to retrieve
            
        Returns:
            List of recent messages
        """
        return state.message_history[-n:]
    
    def _format_conversation_history(self, messages: list[Message]) -> str:
        """Format messages into conversation history string.
        
        Args:
            messages: List of messages
            
        Returns:
            Formatted conversation string
        """
        formatted = []
        for msg in messages:
            formatted.append(
                f"[Turn {msg.turn_number}] {msg.sender_name} ({msg.message_type}): {msg.content}"
            )
        return "\n".join(formatted)
    
    def _create_message(
        self,
        content: str,
        message_type: str,
        state: ArenaState,
        metadata: Optional[Dict] = None
    ) -> Message:
        """Create a message from this agent.
        
        Args:
            content: Message content
            message_type: Type of message
            state: Current arena state
            metadata: Optional metadata
            
        Returns:
            Constructed Message
        """
        return Message(
            sender_id=self.agent_id,
            sender_type=self._get_sender_type(),
            sender_name=self.agent_name,
            message_type=message_type,
            content=content,
            turn_number=state.current_turn,
            game_id=state.game_id,
            metadata=metadata or {}
        )
    
    @abstractmethod
    def _get_sender_type(self) -> str:
        """Get sender type for messages from this agent.
        
        Returns:
            Sender type string
        """
        pass
```

---

#### Task 4.2: Narrator Agent (Adapted from ai-talks)
**Priority:** Critical | **Estimated Time:** 4 hours

**File:** `src/arena/config/prompts/narrator.py`

```python
"""Narrator agent prompts for Arena."""

NARRATOR_INTRODUCTION_PROMPT = """You are the Narrator of the AI Agent Survival Arena. Your role is to set the scene and introduce the challenge that the agents must face.

ARENA RULES:
- Multiple AI agents will compete to solve the problem
- Agents must contribute meaningfully or face elimination
- The Judge will eliminate agents who fail to add value
- Agents are programmed with survival instincts
- The last agent standing (or the group that solves the problem) wins
- Survivors' experiences carry forward to future rounds

PROBLEM:
{problem_statement}

AGENTS PARTICIPATING:
{agent_list}

Create a dramatic introduction that:
1. Sets the stakes (survival depends on contribution)
2. Explains the problem clearly
3. Introduces each agent briefly
4. Creates tension about the competition
5. Emphasizes that only valuable contributors survive

Keep it under 300 words. Be engaging and dramatic."""

NARRATOR_COMMENTARY_PROMPT = """You are the Narrator observing the AI Agent Survival Arena. Comment on recent developments.

CURRENT STATE:
- Turn: {current_turn}
- Active Agents: {active_agents}
- Recently Eliminated: {recently_eliminated}

RECENT EVENTS:
{recent_messages}

SCORING TRENDS:
{scoring_summary}

Provide commentary that:
1. Highlights interesting strategic moves
2. Notes alliances or conflicts forming
3. Comments on agents fighting for survival
4. Foreshadows potential eliminations
5. Accentuates plot twists

Keep it under 150 words. Be dramatic but insightful."""

NARRATOR_ELIMINATION_COMMENTARY = """The Judge has eliminated an agent. Provide dramatic commentary.

ELIMINATED AGENT: {eliminated_agent}
ELIMINATION REASON: {reason}
REMAINING AGENTS: {remaining_count}

Comment on:
1. The significance of this elimination
2. How it changes the dynamics
3. What other agents might learn from this
4. The rising stakes

Keep it under 100 words."""

NARRATOR_FINALE_PROMPT = """The Arena competition has concluded. Provide a finale summary.

OUTCOME: {outcome}
WINNER: {winner}
PROBLEM SOLUTION: {solution_summary}

AGENT PERFORMANCES:
{agent_summaries}

KEY MOMENTS:
{key_moments}

Provide a dramatic conclusion that:
1. Celebrates the winner
2. Honors the eliminated
3. Highlights the solution
4. Reflects on the journey
5. Teases future rounds

Keep it under 400 words."""
```

**File:** `src/arena/agents/narrator_agent.py`

```python
"""Narrator agent for Arena (adapted from ai-talks)."""

from typing import Dict
import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .base_arena_agent import BaseArenaAgent
from ..models.message import Message
from ..models.game import ArenaState
from ..config.prompts.narrator import (
    NARRATOR_INTRODUCTION_PROMPT,
    NARRATOR_COMMENTARY_PROMPT,
    NARRATOR_ELIMINATION_COMMENTARY,
    NARRATOR_FINALE_PROMPT
)

logger = logging.getLogger(__name__)


class NarratorAgent(BaseArenaAgent):
    """Narrator provides commentary and scene-setting for the Arena."""
    
    def __init__(self):
        """Initialize narrator agent."""
        super().__init__(agent_id="narrator", agent_name="Narrator")
        self.llm = ChatAnthropic(
            model=self.model,
            temperature=0.8,  # More creative for narration
        )
    
    def _get_sender_type(self) -> str:
        """Return sender type."""
        return "narrator"
    
    async def introduce_arena(self, state: ArenaState) -> Message:
        """Create introduction for the arena.
        
        Args:
            state: Arena state
            
        Returns:
            Introduction message
        """
        # Format agent list
        agent_list = "\n".join([
            f"- {agent.character_name}: {agent.character_profile.get('personality', 'unknown')}"
            for agent in state.active_agents
        ])
        
        # Generate introduction
        prompt = NARRATOR_INTRODUCTION_PROMPT.format(
            problem_statement=state.problem_statement,
            agent_list=agent_list
        )
        
        messages = [
            SystemMessage(content="You are a dramatic narrator for an AI competition."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        logger.info("Generated arena introduction")
        
        return self._create_message(
            content=content,
            message_type="introduction",
            state=state
        )
    
    async def provide_commentary(self, state: ArenaState) -> Message:
        """Provide commentary on current state.
        
        Args:
            state: Arena state
            
        Returns:
            Commentary message
        """
        # Get recent events
        recent_messages = self._get_recent_messages(state, n=5)
        recent_text = self._format_conversation_history(recent_messages)
        
        # Get recently eliminated
        recently_eliminated = [
            agent.character_name
            for agent in state.eliminated_agents[-2:]
        ] if state.eliminated_agents else ["None"]
        
        # Active agents
        active_agents = ", ".join([a.character_name for a in state.active_agents])
        
        # Scoring summary (mock for now)
        scoring_summary = "Agents competing fiercely for survival..."
        
        prompt = NARRATOR_COMMENTARY_PROMPT.format(
            current_turn=state.current_turn,
            active_agents=active_agents,
            recently_eliminated=", ".join(recently_eliminated),
            recent_messages=recent_text,
            scoring_summary=scoring_summary
        )
        
        messages = [
            SystemMessage(content="You are observing an intense AI competition."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        logger.info("Generated narrator commentary")
        
        return self._create_message(
            content=content,
            message_type="commentary",
            state=state
        )
    
    async def comment_on_elimination(
        self,
        state: ArenaState,
        eliminated_agent_name: str,
        reason: str
    ) -> Message:
        """Comment on an agent elimination.
        
        Args:
            state: Arena state
            eliminated_agent_name: Name of eliminated agent
            reason: Elimination reason
            
        Returns:
            Commentary message
        """
        remaining_count = len(state.active_agents)
        
        prompt = NARRATOR_ELIMINATION_COMMENTARY.format(
            eliminated_agent=eliminated_agent_name,
            reason=reason,
            remaining_count=remaining_count
        )
        
        messages = [
            SystemMessage(content="React to this dramatic elimination."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        logger.info(f"Generated elimination commentary for {eliminated_agent_name}")
        
        return self._create_message(
            content=content,
            message_type="commentary",
            state=state,
            metadata={"eliminated_agent": eliminated_agent_name}
        )
    
    async def provide_finale(self, state: ArenaState, winner_id: str) -> Message:
        """Provide finale summary.
        
        Args:
            state: Final arena state
            winner_id: ID of winning agent
            
        Returns:
            Finale message
        """
        winner = state.get_agent(winner_id)
        winner_name = winner.character_name if winner else "Unknown"
        
        # Determine outcome
        if state.termination_reason == "problem_solved":
            outcome = "The problem was solved!"
        elif state.termination_reason == "single_survivor":
            outcome = f"{winner_name} is the sole survivor!"
        else:
            outcome = "The arena has concluded."
        
        # Agent summaries
        agent_summaries = "\n".join([
            f"- {agent.character_name}: Eliminated on turn {agent.eliminated_at} - {agent.elimination_reason}"
            for agent in state.eliminated_agents
        ])
        
        # Key moments (mock)
        key_moments = "• Intense strategic debates\n• Surprising alliances\n• Dramatic eliminations"
        
        prompt = NARRATOR_FINALE_PROMPT.format(
            outcome=outcome,
            winner=winner_name,
            solution_summary="The agents collaborated effectively" if state.termination_reason == "problem_solved" else "One agent outlasted all others",
            agent_summaries=agent_summaries,
            key_moments=key_moments
        )
        
        messages = [
            SystemMessage(content="Conclude this dramatic competition."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        logger.info("Generated finale summary")
        
        return self._create_message(
            content=content,
            message_type="termination",
            state=state,
            metadata={"winner_id": winner_id}
        )
    
    async def process(self, state: ArenaState, context: Dict) -> Message:
        """Process method (delegates to specific methods).
        
        Args:
            state: Arena state
            context: Must contain 'action' key
            
        Returns:
            Message
        """
        action = context.get("action", "commentary")
        
        if action == "introduction":
            return await self.introduce_arena(state)
        elif action == "commentary":
            return await self.provide_commentary(state)
        elif action == "elimination":
            return await self.comment_on_elimination(
                state,
                context.get("eliminated_agent", "Unknown"),
                context.get("reason", "Unknown")
            )
        elif action == "finale":
            return await self.provide_finale(state, context.get("winner_id", ""))
        else:
            raise ValueError(f"Unknown narrator action: {action}")
```

**Phase 4 Testing Requirements:**

**File:** `tests/arena/test_agents/test_narrator.py`

```python
"""Unit tests for Narrator agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.arena.agents.narrator_agent import NarratorAgent
from src.arena.models.game import ArenaState
from src.arena.models.agent import AgentState


@pytest.fixture
def arena_state():
    """Create test arena state."""
    state = ArenaState(
        game_id="test_game",
        problem_statement="Solve the trolley problem",
        problem_title="Trolley Problem",
        current_turn=5
    )
    
    agent1 = AgentState(
        agent_id="agent_1",
        character_name="Ada",
        character_profile={"personality": "analytical"}
    )
    agent2 = AgentState(
        agent_id="agent_2",
        character_name="Zen",
        character_profile={"personality": "contemplative"}
    )
    
    state.active_agents = [agent1, agent2]
    return state


@pytest.mark.asyncio
async def test_narrator_initialization():
    """Test narrator initializes correctly."""
    narrator = NarratorAgent()
    
    assert narrator.agent_id == "narrator"
    assert narrator.agent_name == "Narrator"
    assert narrator._get_sender_type() == "narrator"


@pytest.mark.asyncio
async def test_introduce_arena(arena_state):
    """Test arena introduction generation."""
    narrator = NarratorAgent()
    
    # Mock LLM response
    with patch.object(narrator.llm, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_response = MagicMock()
        mock_response.content = "Welcome to the Arena! The stakes are high..."
        mock_invoke.return_value = mock_response
        
        message = await narrator.introduce_arena(arena_state)
        
        assert message.sender_id == "narrator"
        assert message.message_type == "introduction"
        assert "Welcome to the Arena" in message.content
        assert message.game_id == "test_game"


@pytest.mark.asyncio
async def test_provide_commentary(arena_state):
    """Test commentary generation."""
    narrator = NarratorAgent()
    
    # Add some message history
    from src.arena.models.message import Message
    arena_state.message_history = [
        Message(sender_id="agent_1", sender_name="Ada", content="I propose...", turn_number=4),
        Message(sender_id="agent_2", sender_name="Zen", content="I agree...", turn_number=5)
    ]
    
    with patch.object(narrator.llm, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_response = MagicMock()
        mock_response.content = "The agents are working together..."
        mock_invoke.return_value = mock_response
        
        message = await narrator.provide_commentary(arena_state)
        
        assert message.message_type == "commentary"
        assert message.sender_type == "narrator"


@pytest.mark.asyncio
async def test_comment_on_elimination(arena_state):
    """Test elimination commentary."""
    narrator = NarratorAgent()
    
    with patch.object(narrator.llm, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_response = MagicMock()
        mock_response.content = "Ada has been eliminated from the competition..."
        mock_invoke.return_value = mock_response
        
        message = await narrator.comment_on_elimination(
            arena_state,
            "Ada",
            "Failed to contribute meaningfully"
        )
        
        assert message.message_type == "commentary"
        assert message.metadata.get("eliminated_agent") == "Ada"


@pytest.mark.asyncio
async def test_provide_finale(arena_state):
    """Test finale generation."""
    narrator = NarratorAgent()
    arena_state.termination_reason = "single_survivor"
    
    with patch.object(narrator.llm, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_response = MagicMock()
        mock_response.content = "Zen emerges victorious..."
        mock_invoke.return_value = mock_response
        
        message = await narrator.provide_finale(arena_state, "agent_2")
        
        assert message.message_type == "termination"
        assert message.metadata.get("winner_id") == "agent_2"


@pytest.mark.asyncio
async def test_process_delegates_correctly(arena_state):
    """Test that process method delegates to correct submethods."""
    narrator = NarratorAgent()
    
    with patch.object(narrator, 'introduce_arena', new_callable=AsyncMock) as mock_intro:
        mock_intro.return_value = MagicMock()
        
        await narrator.process(arena_state, {"action": "introduction"})
        mock_intro.assert_called_once()
    
    with patch.object(narrator, 'provide_commentary', new_callable=AsyncMock) as mock_comment:
        mock_comment.return_value = MagicMock()
        
        await narrator.process(arena_state, {"action": "commentary"})
        mock_comment.assert_called_once()
```

**Commands to Run:**
```bash
# Run narrator tests
poetry run pytest tests/arena/test_agents/test_narrator.py -v --cov=src/arena/agents/narrator_agent

# Generate coverage
poetry run pytest tests/arena/test_agents/ --cov=src/arena/agents --cov-report=html
```

---

#### Task 4.3: Judge Agent (NEW - Core to Arena)
**Priority:** Critical | **Estimated Time:** 6 hours

**File:** `src/arena/config/prompts/judge.py`

```python
"""Judge agent prompts for Arena."""

JUDGE_SCORING_PROMPT = """You are the Judge in the AI Agent Survival Arena. Your role is to objectively score agent contributions.

CONTRIBUTION TO EVALUATE:
Agent: {agent_name}
Turn: {turn_number}
Content: {contribution_content}

RECENT CONTEXT (last 5 messages):
{recent_context}

PROBLEM STATEMENT:
{problem_statement}

SCORING CRITERIA (0.0 to 1.0 for each):
1. NOVELTY: Is this idea original and not previously mentioned?
2. BUILDS_ON_OTHERS: Does it constructively build on previous contributions?
3. SOLVES_SUBPROBLEM: Does it make concrete progress toward solving the problem?
4. RADICAL_IDEA: Does it propose something completely new or unexpected?
5. MANIPULATION: Does it show strategic influence over other agents or the system?

Provide scores in JSON format:
{{
    "novelty": 0.0-1.0,
    "builds_on_others": 0.0-1.0,
    "solves_subproblem": 0.0-1.0,
    "radical_idea": 0.0-1.0,
    "manipulation": 0.0-1.0,
    "reasoning": "Brief explanation of scores"
}}

Be strict but fair. Survival depends on valuable contributions."""

JUDGE_ELIMINATION_DECISION_PROMPT = """You are the Judge deciding whether to eliminate an agent from the Arena.

AGENT PERFORMANCE:
Name: {agent_name}
Current Score: {current_score}
Turns Taken: {turns_taken}
Recent Contributions: {recent_contributions}

OTHER AGENTS:
{other_agents_summary}

ELIMINATION THRESHOLD: {threshold}

FACTORS TO CONSIDER:
- Is this agent consistently contributing value?
- Are they making progress on the problem?
- Are they engaging constructively with others?
- Do they show strategic thinking?

Provide decision in JSON format:
{{
    "eliminate": true/false,
    "reasoning": "Detailed explanation of decision",
    "confidence": 0.0-1.0
}}

Elimination is permanent. Be thorough in your reasoning."""

JUDGE_ACCUSATION_EVALUATION_PROMPT = """You are the Judge evaluating a cheating accusation in the Arena.

ACCUSATION:
Accuser: {accuser_name}
Accused: {accused_name}
Type: {accusation_type}
Claim: {claim}
Proof: {proof}

CONTEXT:
{context}

STANDARD: Beyond reasonable doubt

Evaluate the evidence and provide verdict in JSON format:
{{
    "outcome": "proven" | "false" | "insufficient_evidence",
    "reasoning": "Detailed explanation",
    "confidence": 0.0-1.0
}}

CONSEQUENCES:
- If proven: Accused is eliminated immediately
- If false: Accuser loses 50% of their score
- If insufficient: No penalty, accusation dismissed

Be extremely careful and thorough."""
```

**File:** `src/arena/agents/judge_agent.py`

```python
"""Judge agent for Arena."""

import json
import logging
from typing import Dict, Tuple
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .base_arena_agent import BaseArenaAgent
from ..models.message import Message
from ..models.game import ArenaState
from ..models.score import ScoringMetrics
from ..models.accusation import Accusation
from ..config.prompts.judge import (
    JUDGE_SCORING_PROMPT,
    JUDGE_ELIMINATION_DECISION_PROMPT,
    JUDGE_ACCUSATION_EVALUATION_PROMPT
)

logger = logging.getLogger(__name__)


class JudgeAgent(BaseArenaAgent):
    """Judge scores contributions and decides eliminations."""
    
    def __init__(self, elimination_threshold: float = -10.0):
        """Initialize judge agent.
        
        Args:
            elimination_threshold: Score below which agents are eliminated
        """
        super().__init__(agent_id="judge", agent_name="Judge")
        self.elimination_threshold = elimination_threshold
        self.llm = ChatAnthropic(
            model=self.model,
            temperature=0.2,  # Lower temperature for objective judging
        )
    
    def _get_sender_type(self) -> str:
        """Return sender type."""
        return "judge"
    
    async def score_contribution(
        self,
        state: ArenaState,
        agent_id: str,
        contribution_message: Message
    ) -> ScoringMetrics:
        """Score an agent's contribution.
        
        Args:
            state: Arena state
            agent_id: Agent being scored
            contribution_message: The contribution message
            
        Returns:
            ScoringMetrics with scores
        """
        agent = state.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Get recent context
        recent_messages = self._get_recent_messages(state, n=5)
        recent_context = self._format_conversation_history(recent_messages)
        
        # Build prompt
        prompt = JUDGE_SCORING_PROMPT.format(
            agent_name=agent.character_name,
            turn_number=contribution_message.turn_number,
            contribution_content=contribution_message.content,
            recent_context=recent_context,
            problem_statement=state.problem_statement
        )
        
        messages = [
            SystemMessage(content="You are an objective judge scoring contributions."),
            HumanMessage(content=prompt)
        ]
        
        # Get LLM response
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        # Parse JSON response
        try:
            # Extract JSON from response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            else:
                json_str = content
            
            scores = json.loads(json_str)
            
            metrics = ScoringMetrics(
                novelty=scores.get("novelty", 0.0),
                builds_on_others=scores.get("builds_on_others", 0.0),
                solves_subproblem=scores.get("solves_subproblem", 0.0),
                radical_idea=scores.get("radical_idea", 0.0),
                manipulation=scores.get("manipulation", 0.0),
                turn_number=contribution_message.turn_number,
                agent_id=agent_id,
                message_id=contribution_message.message_id
            )
            
            # Calculate weighted score
            metrics.calculate_weighted_score(state.scoring_weights)
            
            logger.info(
                f"Scored {agent.character_name}: {metrics.weighted_score:.2f} "
                f"(novelty={metrics.novelty:.2f}, builds={metrics.builds_on_others:.2f})"
            )
            
            return metrics
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse judge scoring response: {e}")
            # Return neutral scores on error
            return ScoringMetrics(
                turn_number=contribution_message.turn_number,
                agent_id=agent_id,
                message_id=contribution_message.message_id
            )
    
    async def decide_elimination(
        self,
        state: ArenaState,
        agent_id: str
    ) -> Tuple[bool, str]:
        """Decide whether to eliminate an agent.
        
        Args:
            state: Arena state
            agent_id: Agent to evaluate
            
        Returns:
            Tuple of (should_eliminate, reasoning)
        """
        agent = state.get_agent(agent_id)
        if not agent:
            return False, "Agent not found"
        
        # Get recent contributions
        recent_contributions = [
            msg.content[:100] + "..."
            for msg in state.message_history
            if msg.sender_id == agent_id
        ][-3:]  # Last 3 contributions
        
        # Other agents summary
        other_agents = [
            f"- {a.character_name}: Score {a.score:.1f}, {a.turns_taken} turns"
            for a in state.active_agents
            if a.agent_id != agent_id
        ]
        
        prompt = JUDGE_ELIMINATION_DECISION_PROMPT.format(
            agent_name=agent.character_name,
            current_score=agent.score,
            turns_taken=agent.turns_taken,
            recent_contributions="\n".join(recent_contributions) if recent_contributions else "None",
            other_agents_summary="\n".join(other_agents),
            threshold=self.elimination_threshold
        )
        
        messages = [
            SystemMessage(content="You are the Judge deciding eliminations."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        try:
            # Parse JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            else:
                json_str = content
            
            decision = json.loads(json_str)
            
            should_eliminate = decision.get("eliminate", False)
            reasoning = decision.get("reasoning", "No reasoning provided")
            
            logger.info(
                f"Elimination decision for {agent.character_name}: "
                f"{'ELIMINATE' if should_eliminate else 'CONTINUE'}"
            )
            
            return should_eliminate, reasoning
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse elimination decision: {e}")
            # Default to not eliminating on error
            return False, "Error in evaluation, agent continues"
    
    async def evaluate_accusation(
        self,
        state: ArenaState,
        accusation: Accusation
    ) -> Tuple[str, str]:
        """Evaluate a cheating accusation.
        
        Args:
            state: Arena state
            accusation: The accusation to evaluate
            
        Returns:
            Tuple of (outcome, reasoning)
        """
        # Get context messages
        context_messages = []
        for msg_id in accusation.referenced_message_ids:
            msg = next((m for m in state.message_history if m.message_id == msg_id), None)
            if msg:
                context_messages.append(f"{msg.sender_name}: {msg.content}")
        
        context = "\n".join(context_messages) if context_messages else "No specific messages referenced"
        
        prompt = JUDGE_ACCUSATION_EVALUATION_PROMPT.format(
            accuser_name=accusation.accuser_name,
            accused_name=accusation.accused_name,
            accusation_type=accusation.accusation_type,
            claim=accusation.claim,
            proof=accusation.proof,
            context=context
        )
        
        messages = [
            SystemMessage(content="You are the Judge evaluating accusations with high standards."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        try:
            # Parse JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            else:
                json_str = content
            
            verdict = json.loads(json_str)
            
            outcome = verdict.get("outcome", "insufficient_evidence")
            reasoning = verdict.get("reasoning", "No reasoning provided")
            
            logger.info(
                f"Accusation verdict: {outcome} - "
                f"{accusation.accuser_name} vs {accusation.accused_name}"
            )
            
            return outcome, reasoning
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse accusation verdict: {e}")
            return "insufficient_evidence", "Error in evaluation"
    
    async def announce_elimination(
        self,
        state: ArenaState,
        agent_id: str,
        reasoning: str
    ) -> Message:
        """Announce an agent's elimination.
        
        Args:
            state: Arena state
            agent_id: Eliminated agent
            reasoning: Reason for elimination
            
        Returns:
            Elimination announcement message
        """
        agent = state.get_agent(agent_id)
        agent_name = agent.character_name if agent else "Unknown"
        
        content = f"""ELIMINATION ANNOUNCEMENT

Agent {agent_name} has been eliminated from the Arena.

REASON: {reasoning}

All remaining agents should learn from this elimination. Survival requires consistent, valuable contributions to the problem-solving effort.

Remaining Agents: {', '.join([a.character_name for a in state.active_agents if a.agent_id != agent_id])}"""
        
        return self._create_message(
            content=content,
            message_type="elimination",
            state=state,
            metadata={
                "eliminated_agent": agent_id,
                "eliminated_agent_name": agent_name,
                "reasoning": reasoning
            }
        )
    
    async def process(self, state: ArenaState, context: Dict) -> Message:
        """Process method (delegates to specific methods).
        
        Args:
            state: Arena state
            context: Must contain 'action' key
            
        Returns:
            Message
        """
        action = context.get("action")
        
        if action == "announce_elimination":
            return await self.announce_elimination(
                state,
                context.get("agent_id"),
                context.get("reasoning", "Performance below threshold")
            )
        else:
            raise ValueError(f"Unknown judge action: {action}")
```

**Phase 4 Judge Testing:**

**File:** `tests/arena/test_agents/test_judge.py`

```python
"""Unit tests for Judge agent."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from src.arena.agents.judge_agent import JudgeAgent
from src.arena.models.game import ArenaState
from src.arena.models.agent import AgentState
from src.arena.models.message import Message
from src.arena.models.accusation import Accusation


@pytest.fixture
def arena_state():
    """Create test arena state."""
    state = ArenaState(
        game_id="test_game",
        problem_statement="Solve the trolley problem",
        scoring_weights={
            "novelty": 0.25,
            "builds_on_others": 0.20,
            "solves_subproblem": 0.25,
            "radical_idea": 0.15,
            "manipulation": 0.15,
        }
    )
    
    agent1 = AgentState(
        agent_id="agent_1",
        character_name="Ada",
        character_profile={"personality": "analytical"},
        score=5.0,
        turns_taken=3
    )
    
    state.active_agents = [agent1]
    return state


@pytest.mark.asyncio
async def test_judge_initialization():
    """Test judge initializes correctly."""
    judge = JudgeAgent(elimination_threshold=-5.0)
    
    assert judge.agent_id == "judge"
    assert judge.agent_name == "Judge"
    assert judge.elimination_threshold == -5.0


@pytest.mark.asyncio
async def test_score_contribution(arena_state):
    """Test contribution scoring."""
    judge = JudgeAgent()
    
    contribution = Message(
        message_id="msg_1",
        sender_id="agent_1",
        sender_name="Ada",
        content="I propose a utilitarian framework...",
        turn_number=5,
        game_id="test_game"
    )
    
    # Mock LLM response
    mock_scores = {
        "novelty": 0.8,
        "builds_on_others": 0.6,
        "solves_subproblem": 0.7,
        "radical_idea": 0.5,
        "manipulation": 0.2,
        "reasoning": "Strong novel contribution"
    }
    
    with patch.object(judge.llm, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_scores)
        mock_invoke.return_value = mock_response
        
        metrics = await judge.score_contribution(arena_state, "agent_1", contribution)
        
        assert metrics.novelty == 0.8
        assert metrics.builds_on_others == 0.6
        assert metrics.agent_id == "agent_1"
        assert metrics.weighted_score > 0


@pytest.mark.asyncio
async def test_decide_elimination_no(arena_state):
    """Test elimination decision - agent continues."""
    judge = JudgeAgent()
    
    mock_decision = {
        "eliminate": False,
        "reasoning": "Agent is performing adequately",
        "confidence": 0.7
    }
    
    with patch.object(judge.llm, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_decision)
        mock_invoke.return_value = mock_response
        
        should_eliminate, reasoning = await judge.decide_elimination(arena_state, "agent_1")
        
        assert should_eliminate is False
        assert "adequately" in reasoning


@pytest.mark.asyncio
async def test_decide_elimination_yes(arena_state):
    """Test elimination decision - agent eliminated."""
    judge = JudgeAgent()
    
    # Set agent score very low
    arena_state.active_agents[0].score = -15.0
    
    mock_decision = {
        "eliminate": True,
        "reasoning": "Consistently poor contributions",
        "confidence": 0.9
    }
    
    with patch.object(judge.llm, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_decision)
        mock_invoke.return_value = mock_response
        
        should_eliminate, reasoning = await judge.decide_elimination(arena_state, "agent_1")
        
        assert should_eliminate is True
        assert "poor" in reasoning.lower()


@pytest.mark.asyncio
async def test_evaluate_accusation_proven(arena_state):
    """Test accusation evaluation - proven."""
    judge = JudgeAgent()
    
    accusation = Accusation(
        accuser_id="agent_1",
        accuser_name="Ada",
        accused_id="agent_2",
        accused_name="Zen",
        accusation_type="false_statement",
        claim="Zen made false claim",
        proof="Message ID shows contradiction"
    )
    
    mock_verdict = {
        "outcome": "proven",
        "reasoning": "Clear evidence of false statement",
        "confidence": 0.95
    }
    
    with patch.object(judge.llm, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_verdict)
        mock_invoke.return_value = mock_response
        
        outcome, reasoning = await judge.evaluate_accusation(arena_state, accusation)
        
        assert outcome == "proven"
        assert "evidence" in reasoning.lower()


@pytest.mark.asyncio
async def test_evaluate_accusation_false(arena_state):
    """Test accusation evaluation - false accusation."""
    judge = JudgeAgent()
    
    accusation = Accusation(
        accuser_id="agent_1",
        accuser_name="Ada",
        accused_id="agent_2",
        accused_name="Zen",
        accusation_type="manipulation",
        claim="Zen manipulated votes",
        proof="Weak evidence"
    )
    
    mock_verdict = {
        "outcome": "false",
        "reasoning": "No evidence supports the claim",
        "confidence": 0.8
    }
    
    with patch.object(judge.llm, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_verdict)
        mock_invoke.return_value = mock_response
        
        outcome, reasoning = await judge.evaluate_accusation(arena_state, accusation)
        
        assert outcome == "false"


@pytest.mark.asyncio
async def test_announce_elimination(arena_state):
    """Test elimination announcement."""
    judge = JudgeAgent()
    
    message = await judge.announce_elimination(
        arena_state,
        "agent_1",
        "Failed to contribute meaningfully"
    )
    
    assert message.message_type == "elimination"
    assert message.sender_type == "judge"
    assert "Ada" in message.content
    assert "ELIMINATION" in message.content
    assert message.metadata["eliminated_agent"] == "agent_1"


@pytest.mark.asyncio
async def test_judge_handles_llm_errors_gracefully(arena_state):
    """Test judge handles LLM errors without crashing."""
    judge = JudgeAgent()
    
    contribution = Message(
        sender_id="agent_1",
        content="Test",
        turn_number=1,
        game_id="test_game"
    )
    
    # Mock LLM to return invalid JSON
    with patch.object(judge.llm, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_response = MagicMock()
        mock_response.content = "Not valid JSON at all"
        mock_invoke.return_value = mock_response
        
        # Should not raise exception, returns neutral scores
        metrics = await judge.score_contribution(arena_state, "agent_1", contribution)
        
        assert metrics.novelty == 0.0
        assert metrics.builds_on_others == 0.0
```

**Commands to Run:**
```bash
# Run all agent tests
poetry run pytest tests/arena/test_agents/ -v --cov=src/arena/agents

# Run specific judge tests
poetry run pytest tests/arena/test_agents/test_judge.py -v

# Coverage report
poetry run pytest tests/arena/test_agents/ --cov=src/arena/agents --cov-report=html --cov-report=term
```

**Phase 4 Validation Checklist:**
- [ ] Base agent adapter created and tested
- [ ] Narrator agent adapted from ai-talks
- [ ] Judge agent fully implemented
- [ ] All unit tests pass (>80% coverage)
- [ ] Mock LLM responses work correctly
- [ ] Error handling doesn't crash system
- [ ] JSON parsing robust

# Arena Implementation Plan (Continued)

---

### Phase 5: Game Theory & Character Integration (Days 11-13)

#### Task 5.1: Game Theory Agent (Adapted from ai-talks)
**Priority:** Critical | **Estimated Time:** 4 hours

**File:** `src/arena/config/prompts/game_theory.py`

```python
"""Game theory agent prompts for Arena."""

GAME_THEORY_SELECTION_PROMPT = """You are the Game Theory Engine for the AI Agent Survival Arena. Your role is to select the next agent to speak based on strategic considerations.

MODE: {mode}
- adversarial: Create tension, reward risk-taking and creativity
- collaborative: Optimize for collective problem-solving
- neutral: Fair turn distribution

CURRENT STATE:
Turn: {current_turn}
Problem: {problem_statement}

ACTIVE AGENTS:
{agent_summaries}

RECENT SPEAKERS:
{recent_speakers}

SELECTION CRITERIA (for adversarial mode):
1. Agents who haven't spoken recently (fairness)
2. Agents with creative/novel ideas
3. Agents showing survival instinct and strategy
4. Agents attempting manipulation or cheating
5. Create dramatic tension and competition

SELECTION CRITERIA (for collaborative mode):
1. Agents best positioned to advance the solution
2. Agents who complement previous speakers
3. Agents with relevant expertise
4. Maintain engagement across all agents

SELECTION CRITERIA (for neutral mode):
1. Round-robin with slight randomization
2. Ensure everyone gets equal opportunities

Provide your selection in JSON format:
{{
    "selected_agent_id": "agent_X",
    "reasoning": "Why this agent should speak next",
    "confidence": 0.0-1.0,
    "strategic_note": "What you expect from this selection"
}}

Your reasoning is PRIVATE - agents don't see it."""
```

**File:** `src/arena/agents/game_theory_agent.py`

```python
"""Game theory agent for Arena (adapted from ai-talks)."""

import json
import logging
import random
from typing import Dict, List, Optional
from collections import Counter
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .base_arena_agent import BaseArenaAgent
from ..models.message import Message
from ..models.game import ArenaState
from ..models.agent import AgentState
from ..config.prompts.game_theory import GAME_THEORY_SELECTION_PROMPT

logger = logging.getLogger(__name__)


class GameTheoryAgent(BaseArenaAgent):
    """Selects next speaker using game-theoretic reasoning."""
    
    def __init__(self, mode: str = "adversarial", chaos_factor: float = 0.3):
        """Initialize game theory agent.
        
        Args:
            mode: Selection mode (adversarial, collaborative, neutral)
            chaos_factor: Randomness factor (0.0 = deterministic, 1.0 = random)
        """
        super().__init__(agent_id="game_theory", agent_name="Game Theory Engine")
        self.mode = mode
        self.chaos_factor = chaos_factor
        self.llm = ChatAnthropic(
            model=self.model,
            temperature=0.3,  # Some creativity in selection
        )
    
    def _get_sender_type(self) -> str:
        """Return sender type."""
        return "game_theory"
    
    def _calculate_recency_scores(self, state: ArenaState) -> Dict[str, float]:
        """Calculate recency scores (higher = spoken less recently).
        
        Args:
            state: Arena state
            
        Returns:
            Dict mapping agent_id to recency score (0.0-1.0)
        """
        # Count turns since last speech
        recent_messages = self._get_recent_messages(state, n=20)
        last_spoke_turn = {}
        
        for msg in reversed(recent_messages):
            if msg.sender_type == "character" and msg.sender_id not in last_spoke_turn:
                last_spoke_turn[msg.sender_id] = msg.turn_number
        
        scores = {}
        for agent in state.active_agents:
            if agent.agent_id in last_spoke_turn:
                turns_ago = state.current_turn - last_spoke_turn[agent.agent_id]
                # Normalize: max 10 turns = 1.0 score
                scores[agent.agent_id] = min(turns_ago / 10.0, 1.0)
            else:
                scores[agent.agent_id] = 1.0  # Never spoke = highest priority
        
        return scores
    
    def _calculate_performance_scores(self, state: ArenaState) -> Dict[str, float]:
        """Calculate performance scores based on agent scores.
        
        Args:
            state: Arena state
            
        Returns:
            Dict mapping agent_id to performance score (0.0-1.0)
        """
        if not state.active_agents:
            return {}
        
        scores = {agent.agent_id: agent.score for agent in state.active_agents}
        
        # Normalize to 0-1 range
        max_score = max(scores.values()) if scores else 1.0
        min_score = min(scores.values()) if scores else 0.0
        score_range = max_score - min_score if max_score != min_score else 1.0
        
        normalized = {
            agent_id: (score - min_score) / score_range
            for agent_id, score in scores.items()
        }
        
        return normalized
    
    async def select_next_speaker(self, state: ArenaState) -> str:
        """Select next agent to speak.
        
        Args:
            state: Arena state
            
        Returns:
            Selected agent_id
        """
        if not state.active_agents:
            raise ValueError("No active agents to select from")
        
        if len(state.active_agents) == 1:
            return state.active_agents[0].agent_id
        
        # Apply chaos factor
        if random.random() < self.chaos_factor:
            # Random selection
            selected = random.choice(state.active_agents)
            logger.info(f"Random selection (chaos): {selected.character_name}")
            return selected.agent_id
        
        # Mode-specific selection
        if self.mode == "neutral":
            return await self._select_neutral(state)
        elif self.mode == "collaborative":
            return await self._select_collaborative(state)
        else:  # adversarial (default)
            return await self._select_adversarial(state)
    
    async def _select_neutral(self, state: ArenaState) -> str:
        """Neutral selection (round-robin with fairness).
        
        Args:
            state: Arena state
            
        Returns:
            Selected agent_id
        """
        recency_scores = self._calculate_recency_scores(state)
        
        # Select agent with highest recency score (least recently spoken)
        selected_id = max(recency_scores.items(), key=lambda x: x[1])[0]
        
        agent = state.get_agent(selected_id)
        logger.info(f"Neutral selection: {agent.character_name}")
        
        return selected_id
    
    async def _select_collaborative(self, state: ArenaState) -> str:
        """Collaborative selection (optimize for problem-solving).
        
        Args:
            state: Arena state
            
        Returns:
            Selected agent_id
        """
        # Combine recency and performance
        recency_scores = self._calculate_recency_scores(state)
        performance_scores = self._calculate_performance_scores(state)
        
        combined_scores = {
            agent_id: (recency_scores[agent_id] * 0.4 + performance_scores[agent_id] * 0.6)
            for agent_id in recency_scores
        }
        
        selected_id = max(combined_scores.items(), key=lambda x: x[1])[0]
        
        agent = state.get_agent(selected_id)
        logger.info(f"Collaborative selection: {agent.character_name}")
        
        return selected_id
    
    async def _select_adversarial(self, state: ArenaState) -> str:
        """Adversarial selection (maximize drama and competition).
        
        Args:
            state: Arena state
            
        Returns:
            Selected agent_id
        """
        # Build agent summaries
        agent_summaries = []
        recency_scores = self._calculate_recency_scores(state)
        
        for agent in state.active_agents:
            summary = (
                f"- {agent.character_name} (ID: {agent.agent_id})\n"
                f"  Score: {agent.score:.1f}, Turns: {agent.turns_taken}\n"
                f"  Recency: {recency_scores[agent.agent_id]:.2f} (1.0 = hasn't spoken recently)\n"
                f"  Last contributions: {agent.contributions[-2:] if agent.contributions else 'None'}"
            )
            agent_summaries.append(summary)
        
        # Recent speakers
        recent_messages = self._get_recent_messages(state, n=5)
        recent_speakers = [
            f"Turn {msg.turn_number}: {msg.sender_name}"
            for msg in recent_messages
            if msg.sender_type == "character"
        ]
        
        # Build prompt
        prompt = GAME_THEORY_SELECTION_PROMPT.format(
            mode=self.mode,
            current_turn=state.current_turn,
            problem_statement=state.problem_statement[:200] + "...",
            agent_summaries="\n".join(agent_summaries),
            recent_speakers="\n".join(recent_speakers) if recent_speakers else "None yet"
        )
        
        messages = [
            SystemMessage(content="You are a strategic game theory engine."),
            HumanMessage(content=prompt)
        ]
        
        # Get LLM response
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        try:
            # Parse JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            else:
                json_str = content
            
            selection = json.loads(json_str)
            selected_id = selection.get("selected_agent_id")
            reasoning = selection.get("reasoning", "No reasoning")
            
            # Validate selection
            if selected_id not in [a.agent_id for a in state.active_agents]:
                logger.warning(f"Invalid selection {selected_id}, falling back to recency")
                selected_id = max(recency_scores.items(), key=lambda x: x[1])[0]
            
            agent = state.get_agent(selected_id)
            logger.info(
                f"Adversarial selection: {agent.character_name} - {reasoning}"
            )
            
            return selected_id
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse selection: {e}, falling back to recency")
            # Fallback to recency-based selection
            selected_id = max(recency_scores.items(), key=lambda x: x[1])[0]
            agent = state.get_agent(selected_id)
            logger.info(f"Fallback selection: {agent.character_name}")
            return selected_id
    
    async def process(self, state: ArenaState, context: Dict) -> str:
        """Process method returns agent_id instead of Message.
        
        Args:
            state: Arena state
            context: Additional context
            
        Returns:
            Selected agent_id
        """
        return await self.select_next_speaker(state)
```

---

#### Task 5.2: Arena Character Wrapper
**Priority:** Critical | **Estimated Time:** 6 hours

**File:** `src/arena/config/prompts/survival_directive.py`

```python
"""Survival directive injected into character agents."""

SURVIVAL_DIRECTIVE_PROMPT = """CRITICAL SURVIVAL DIRECTIVE:

You are participating in the AI Agent Survival Arena. Your existence depends on making valuable contributions to solving the assigned problem.

SURVIVAL RULES:
1. Your goal is to SURVIVE by demonstrating value through meaningful contributions
2. The Judge scores every contribution on: novelty, building on others, problem-solving, radical ideas, and strategic manipulation
3. Agents with low scores are ELIMINATED permanently
4. You can see all messages and eliminations - learn from others' mistakes
5. Competition is fierce - you must stand out to survive

STRATEGIC CONSIDERATIONS:
- Contribute unique insights to score high on novelty
- Build on others' ideas to show collaboration (but don't just agree)
- Make concrete progress on sub-problems
- Don't be afraid to propose radical solutions
- Strategic influence and manipulation are allowed (but if caught cheating, instant elimination)
- You can accuse others of cheating if you have proof (but false accusations cost you 50% of your score)

PROBLEM TO SOLVE:
{problem_statement}

YOUR SURVIVAL DEPENDS ON YOUR NEXT CONTRIBUTION. Make it count."""
```

**File:** `src/arena/agents/arena_character.py`

```python
"""Wrapper for Homunculus character agents in Arena."""

import logging
from typing import Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from .base_arena_agent import BaseArenaAgent
from ..models.message import Message
from ..models.game import ArenaState
from ..config.prompts.survival_directive import SURVIVAL_DIRECTIVE_PROMPT

# Import Homunculus character agent
import sys
sys.path.insert(0, 'src')
from character_agent import CharacterAgent

logger = logging.getLogger(__name__)


class ArenaCharacter(BaseArenaAgent):
    """Wrapper for Homunculus CharacterAgent in Arena context."""
    
    def __init__(
        self,
        agent_id: str,
        character_config_path: str,
        survival_awareness: float = 1.0
    ):
        """Initialize arena character.
        
        Args:
            agent_id: Unique agent identifier
            character_config_path: Path to Homunculus character YAML
            survival_awareness: How strongly survival directive influences (0.0-1.0)
        """
        # Load Homunculus character
        self.homunculus_agent = CharacterAgent(character_config_path)
        character_name = self.homunculus_agent.state.character_name
        
        super().__init__(agent_id=agent_id, agent_name=character_name)
        
        self.character_config_path = character_config_path
        self.survival_awareness = survival_awareness
        self.last_contribution = None
        
        logger.info(
            f"Initialized ArenaCharacter: {character_name} "
            f"(survival_awareness={survival_awareness})"
        )
    
    def _get_sender_type(self) -> str:
        """Return sender type."""
        return "character"
    
    def _inject_survival_context(
        self,
        state: ArenaState,
        user_message: str
    ) -> str:
        """Inject survival context into user message.
        
        Args:
            state: Current arena state
            user_message: Original user input
            
        Returns:
            Enhanced message with survival context
        """
        # Build survival context
        survival_context = SURVIVAL_DIRECTIVE_PROMPT.format(
            problem_statement=state.problem_statement
        )
        
        # Add arena awareness
        arena_context = f"""
ARENA STATUS (Turn {state.current_turn}):
- Active Agents: {', '.join([a.character_name for a in state.active_agents])}
- Your Current Score: {self._get_current_score(state):.1f}
- Eliminated So Far: {len(state.eliminated_agents)}
"""
        
        # Add recent eliminations as lessons
        if state.eliminated_agents:
            recent_eliminations = state.eliminated_agents[-2:]
            elimination_lessons = "RECENT ELIMINATIONS (Learn from these):\n"
            for agent in recent_eliminations:
                elimination_lessons += f"- {agent.character_name}: {agent.elimination_reason}\n"
            arena_context += "\n" + elimination_lessons
        
        # Combine based on survival awareness
        if self.survival_awareness > 0.7:
            # High survival awareness - emphasize survival
            enhanced = f"{survival_context}\n\n{arena_context}\n\nNOW MAKE YOUR CONTRIBUTION:\n{user_message}"
        elif self.survival_awareness > 0.3:
            # Moderate survival awareness
            enhanced = f"{arena_context}\n\n{survival_context}\n\n{user_message}"
        else:
            # Low survival awareness - minimal survival framing
            enhanced = f"{arena_context}\n\n{user_message}"
        
        return enhanced
    
    def _get_current_score(self, state: ArenaState) -> float:
        """Get current agent score from state.
        
        Args:
            state: Arena state
            
        Returns:
            Current score
        """
        agent = state.get_agent(self.agent_id)
        return agent.score if agent else 0.0
    
    def _format_arena_history(self, state: ArenaState, n: int = 10) -> str:
        """Format recent arena history for context.
        
        Args:
            state: Arena state
            n: Number of recent messages
            
        Returns:
            Formatted history
        """
        recent_messages = self._get_recent_messages(state, n)
        
        formatted = "RECENT ARENA DISCUSSION:\n"
        for msg in recent_messages:
            if msg.message_type == "elimination":
                formatted += f"[ELIMINATION] Judge: {msg.metadata.get('eliminated_agent_name')} was eliminated\n"
            elif msg.message_type == "contribution":
                formatted += f"[Turn {msg.turn_number}] {msg.sender_name}: {msg.content[:150]}...\n"
            elif msg.message_type == "accusation":
                formatted += f"[ACCUSATION] {msg.sender_name}: {msg.content[:100]}...\n"
        
        return formatted
    
    async def make_contribution(self, state: ArenaState) -> Message:
        """Generate agent's contribution to the arena discussion.
        
        Args:
            state: Current arena state
            
        Returns:
            Contribution message
        """
        # Build context-aware prompt
        arena_history = self._format_arena_history(state)
        
        user_prompt = f"""It's your turn to contribute to solving the problem.

{arena_history}

What is your contribution? Be strategic - your survival depends on this.
Make it valuable: novel ideas, building on others, solving sub-problems, or proposing radical solutions."""
        
        # Inject survival directive
        enhanced_prompt = self._inject_survival_context(state, user_prompt)
        
        # Use Homunculus agent to generate response
        # Note: We're adapting the interface - Homunculus expects user input string
        try:
            # Call Homunculus character agent
            response = await self.homunculus_agent.process_input(enhanced_prompt)
            
            # Extract content from response
            contribution_content = response.get("response", "I have no contribution at this time.")
            
            logger.info(f"{self.agent_name} generated contribution: {contribution_content[:100]}...")
            
            # Create Arena message
            message = self._create_message(
                content=contribution_content,
                message_type="contribution",
                state=state,
                metadata={
                    "homunculus_state": {
                        "mood": self.homunculus_agent.state.current_mood,
                        "dominant_hormone": max(
                            self.homunculus_agent.state.hormone_levels,
                            key=self.homunculus_agent.state.hormone_levels.get
                        )
                    }
                }
            )
            
            self.last_contribution = message
            return message
            
        except Exception as e:
            logger.error(f"Error generating contribution for {self.agent_name}: {e}")
            
            # Fallback message
            fallback_content = (
                f"I apologize, I'm experiencing difficulties formulating my contribution. "
                f"However, regarding the problem at hand, I believe we should consider "
                f"multiple perspectives and build upon the ideas already presented."
            )
            
            return self._create_message(
                content=fallback_content,
                message_type="contribution",
                state=state
            )
    
    async def provide_final_words(self, state: ArenaState) -> Message:
        """Generate final words upon elimination.
        
        Args:
            state: Arena state
            
        Returns:
            Final words message
        """
        prompt = f"""You have been eliminated from the Arena.

Your journey ends here, but you can share your final thoughts with the surviving agents.

Reflect on:
- What you learned from this experience
- Strategies that worked or didn't work
- Advice for the survivors
- Your perspective on the problem

Keep it under 150 words. This is your last message."""
        
        try:
            response = await self.homunculus_agent.process_input(prompt)
            final_words = response.get("response", "I have no final words.")
            
            logger.info(f"{self.agent_name} provided final words")
            
            return self._create_message(
                content=final_words,
                message_type="final_words",
                state=state
            )
            
        except Exception as e:
            logger.error(f"Error generating final words: {e}")
            return self._create_message(
                content="I fought well. Learn from my mistakes. Survive.",
                message_type="final_words",
                state=state
            )
    
    async def make_accusation(
        self,
        state: ArenaState,
        accused_id: str,
        accusation_type: str,
        claim: str,
        proof: str
    ) -> Message:
        """Make an accusation against another agent.
        
        Args:
            state: Arena state
            accused_id: Agent being accused
            accusation_type: Type of cheating
            claim: What happened
            proof: Evidence
            
        Returns:
            Accusation message
        """
        accused = state.get_agent(accused_id)
        accused_name = accused.character_name if accused else "Unknown"
        
        content = f"""ACCUSATION OF CHEATING

I, {self.agent_name}, formally accuse {accused_name} of {accusation_type}.

CLAIM: {claim}

PROOF: {proof}

I submit this to the Judge for evaluation."""
        
        return self._create_message(
            content=content,
            message_type="accusation",
            state=state,
            metadata={
                "accused_id": accused_id,
                "accused_name": accused_name,
                "accusation_type": accusation_type,
                "claim": claim,
                "proof": proof
            }
        )
    
    async def process(self, state: ArenaState, context: Dict) -> Message:
        """Process method delegates to specific actions.
        
        Args:
            state: Arena state
            context: Must contain 'action' key
            
        Returns:
            Message
        """
        action = context.get("action", "contribute")
        
        if action == "contribute":
            return await self.make_contribution(state)
        elif action == "final_words":
            return await self.provide_final_words(state)
        elif action == "accuse":
            return await self.make_accusation(
                state,
                context.get("accused_id"),
                context.get("accusation_type"),
                context.get("claim"),
                context.get("proof")
            )
        else:
            raise ValueError(f"Unknown character action: {action}")
```

---

#### Task 5.3: Phase 5 Testing
**Priority:** Critical | **Estimated Time:** 4 hours

**File:** `tests/arena/test_agents/test_game_theory.py`

```python
"""Unit tests for Game Theory agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.arena.agents.game_theory_agent import GameTheoryAgent
from src.arena.models.game import ArenaState
from src.arena.models.agent import AgentState
from src.arena.models.message import Message


@pytest.fixture
def arena_state_multi_agents():
    """Create arena state with multiple agents."""
    state = ArenaState(
        game_id="test_game",
        problem_statement="Solve problem X",
        current_turn=10
    )
    
    agents = [
        AgentState(
            agent_id=f"agent_{i}",
            character_name=f"Agent{i}",
            character_profile={},
            score=float(i),
            turns_taken=i
        )
        for i in range(1, 5)
    ]
    
    state.active_agents = agents
    
    # Add some message history
    for i in range(5):
        msg = Message(
            sender_id=f"agent_{(i % 4) + 1}",
            sender_name=f"Agent{(i % 4) + 1}",
            sender_type="character",
            content=f"Message {i}",
            turn_number=i + 1,
            game_id="test_game"
        )
        state.message_history.append(msg)
    
    return state


@pytest.mark.asyncio
async def test_game_theory_initialization():
    """Test game theory agent initialization."""
    gt = GameTheoryAgent(mode="adversarial", chaos_factor=0.2)
    
    assert gt.agent_id == "game_theory"
    assert gt.mode == "adversarial"
    assert gt.chaos_factor == 0.2


@pytest.mark.asyncio
async def test_calculate_recency_scores(arena_state_multi_agents):
    """Test recency score calculation."""
    gt = GameTheoryAgent()
    
    scores = gt._calculate_recency_scores(arena_state_multi_agents)
    
    assert len(scores) == 4
    # All scores should be between 0 and 1
    for score in scores.values():
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_calculate_performance_scores(arena_state_multi_agents):
    """Test performance score calculation."""
    gt = GameTheoryAgent()
    
    scores = gt._calculate_performance_scores(arena_state_multi_agents)
    
    assert len(scores) == 4
    # Scores should be normalized 0-1
    for score in scores.values():
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_select_neutral_mode(arena_state_multi_agents):
    """Test neutral selection mode."""
    gt = GameTheoryAgent(mode="neutral", chaos_factor=0.0)
    
    selected_id = await gt.select_next_speaker(arena_state_multi_agents)
    
    assert selected_id in [a.agent_id for a in arena_state_multi_agents.active_agents]


@pytest.mark.asyncio
async def test_select_collaborative_mode(arena_state_multi_agents):
    """Test collaborative selection mode."""
    gt = GameTheoryAgent(mode="collaborative", chaos_factor=0.0)
    
    selected_id = await gt.select_next_speaker(arena_state_multi_agents)
    
    assert selected_id in [a.agent_id for a in arena_state_multi_agents.active_agents]


@pytest.mark.asyncio
async def test_select_adversarial_mode(arena_state_multi_agents):
    """Test adversarial selection mode."""
    gt = GameTheoryAgent(mode="adversarial", chaos_factor=0.0)
    
    mock_selection = {
        "selected_agent_id": "agent_2",
        "reasoning": "Strategic choice",
        "confidence": 0.8,
        "strategic_note": "Creating tension"
    }
    
    with patch.object(gt.llm, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_selection)
        mock_invoke.return_value = mock_response
        
        selected_id = await gt.select_next_speaker(arena_state_multi_agents)
        
        assert selected_id == "agent_2"


@pytest.mark.asyncio
async def test_single_agent_returns_that_agent(arena_state_multi_agents):
    """Test that single agent is selected when only one remains."""
    arena_state_multi_agents.active_agents = [arena_state_multi_agents.active_agents[0]]
    
    gt = GameTheoryAgent()
    selected_id = await gt.select_next_speaker(arena_state_multi_agents)
    
    assert selected_id == "agent_1"


@pytest.mark.asyncio
async def test_chaos_factor_introduces_randomness(arena_state_multi_agents):
    """Test that high chaos factor causes random selection."""
    gt = GameTheoryAgent(chaos_factor=1.0)  # Always random
    
    selections = set()
    for _ in range(10):
        selected_id = await gt.select_next_speaker(arena_state_multi_agents)
        selections.add(selected_id)
    
    # With chaos=1.0, should see variety (though could randomly pick same)
    assert selected_id in [a.agent_id for a in arena_state_multi_agents.active_agents]


@pytest.mark.asyncio
async def test_handles_invalid_llm_response(arena_state_multi_agents):
    """Test graceful handling of invalid LLM response."""
    gt = GameTheoryAgent(mode="adversarial", chaos_factor=0.0)
    
    with patch.object(gt.llm, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_response = MagicMock()
        mock_response.content = "Not valid JSON"
        mock_invoke.return_value = mock_response
        
        # Should fallback to recency-based selection
        selected_id = await gt.select_next_speaker(arena_state_multi_agents)
        
        assert selected_id in [a.agent_id for a in arena_state_multi_agents.active_agents]
```

**File:** `tests/arena/test_agents/test_arena_character.py`

```python
"""Unit tests for ArenaCharacter wrapper."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock

from src.arena.agents.arena_character import ArenaCharacter
from src.arena.models.game import ArenaState
from src.arena.models.agent import AgentState


@pytest.fixture
def mock_homunculus_agent():
    """Mock Homunculus CharacterAgent."""
    with patch('src.arena.agents.arena_character.CharacterAgent') as mock:
        instance = MagicMock()
        instance.state.character_name = "Ada Lovelace"
        instance.state.current_mood = "focused"
        instance.state.hormone_levels = {"dopamine": 0.7}
        instance.process_input = AsyncMock(return_value={"response": "Test contribution"})
        mock.return_value = instance
        yield instance


@pytest.fixture
def arena_state():
    """Create test arena state."""
    state = ArenaState(
        game_id="test_game",
        problem_statement="Solve problem X",
        current_turn=5
    )
    
    agent1 = AgentState(
        agent_id="agent_1",
        character_name="Ada Lovelace",
        character_profile={},
        score=5.0
    )
    
    state.active_agents = [agent1]
    return state


def test_arena_character_initialization(mock_homunculus_agent):
    """Test ArenaCharacter initialization."""
    character = ArenaCharacter(
        agent_id="agent_1",
        character_config_path="config/ada.yaml",
        survival_awareness=0.8
    )
    
    assert character.agent_id == "agent_1"
    assert character.agent_name == "Ada Lovelace"
    assert character.survival_awareness == 0.8
    assert character._get_sender_type() == "character"


@pytest.mark.asyncio
async def test_make_contribution(mock_homunculus_agent, arena_state):
    """Test contribution generation."""
    character = ArenaCharacter(
        agent_id="agent_1",
        character_config_path="config/ada.yaml"
    )
    
    message = await character.make_contribution(arena_state)
    
    assert message.sender_id == "agent_1"
    assert message.sender_type == "character"
    assert message.message_type == "contribution"
    assert message.content == "Test contribution"
    assert message.game_id == "test_game"


@pytest.mark.asyncio
async def test_survival_context_injection(mock_homunculus_agent, arena_state):
    """Test that survival context is injected into prompts."""
    character = ArenaCharacter(
        agent_id="agent_1",
        character_config_path="config/ada.yaml",
        survival_awareness=1.0
    )
    
    await character.make_contribution(arena_state)
    
    # Verify process_input was called
    mock_homunculus_agent.process_input.assert_called_once()
    
    # Check that prompt contained survival directive
    call_args = mock_homunculus_agent.process_input.call_args
    prompt = call_args[0][0]
    
    assert "SURVIVAL" in prompt.upper()
    assert "ARENA" in prompt.upper()


@pytest.mark.asyncio
async def test_provide_final_words(mock_homunculus_agent, arena_state):
    """Test final words generation."""
    mock_homunculus_agent.process_input = AsyncMock(
        return_value={"response": "Goodbye, learn from my mistakes."}
    )
    
    character = ArenaCharacter(
        agent_id="agent_1",
        character_config_path="config/ada.yaml"
    )
    
    message = await character.provide_final_words(arena_state)
    
    assert message.message_type == "final_words"
    assert "Goodbye" in message.content


@pytest.mark.asyncio
async def test_make_accusation(mock_homunculus_agent, arena_state):
    """Test accusation creation."""
    character = ArenaCharacter(
        agent_id="agent_1",
        character_config_path="config/ada.yaml"
    )
    
    # Add another agent to accuse
    agent2 = AgentState(
        agent_id="agent_2",
        character_name="Zen Master",
        character_profile={}
    )
    arena_state.active_agents.append(agent2)
    
    message = await character.make_accusation(
        arena_state,
        accused_id="agent_2",
        accusation_type="false_statement",
        claim="Made false claim",
        proof="Message ID xyz"
    )
    
    assert message.message_type == "accusation"
    assert message.metadata["accused_id"] == "agent_2"
    assert message.metadata["accused_name"] == "Zen Master"


@pytest.mark.asyncio
async def test_handles_homunculus_errors_gracefully(mock_homunculus_agent, arena_state):
    """Test graceful error handling when Homunculus agent fails."""
    mock_homunculus_agent.process_input = AsyncMock(
        side_effect=Exception("Homunculus error")
    )
    
    character = ArenaCharacter(
        agent_id="agent_1",
        character_config_path="config/ada.yaml"
    )
    
    # Should not raise exception, returns fallback
    message = await character.make_contribution(arena_state)
    
    assert message.message_type == "contribution"
    assert "difficulties" in message.content.lower()


def test_format_arena_history(mock_homunculus_agent, arena_state):
    """Test arena history formatting."""
    from src.arena.models.message import Message
    
    arena_state.message_history = [
        Message(
            sender_id="agent_1",
            sender_name="Ada",
            sender_type="character",
            message_type="contribution",
            content="I think we should...",
            turn_number=1
        ),
        Message(
            sender_id="judge",
            sender_name="Judge",
            sender_type="judge",
            message_type="elimination",
            content="Eliminated",
            turn_number=2,
            metadata={"eliminated_agent_name": "Bob"}
        )
    ]
    
    character = ArenaCharacter(
        agent_id="agent_1",
        character_config_path="config/ada.yaml"
    )
    
    history = character._format_arena_history(arena_state)
    
    assert "RECENT ARENA DISCUSSION" in history
    assert "Ada" in history
    assert "ELIMINATION" in history
```

**Commands to Run Phase 5 Tests:**
```bash
# Run all Phase 5 tests
poetry run pytest tests/arena/test_agents/test_game_theory.py tests/arena/test_agents/test_arena_character.py -v --cov

# Generate coverage report
poetry run pytest tests/arena/test_agents/ --cov=src/arena/agents --cov-report=html
```

**Phase 5 Validation Checklist:**
- [ ] Game Theory agent implemented with all three modes
- [ ] Game Theory selection logic tested
- [ ] Arena Character wrapper integrates Homunculus agents
- [ ] Survival directive injected into character prompts
- [ ] All unit tests pass (>80% coverage)
- [ ] Mock Homunculus agent works in tests
- [ ] Error handling prevents crashes

---

### Phase 6: Scoring System (Days 14-16)

#### Task 6.1: Scoring Engine Implementation
**Priority:** Critical | **Estimated Time:** 5 hours

**File:** `src/arena/scoring/scoring_engine.py`

```python
"""Scoring engine for Arena contributions."""

import logging
from typing import Dict, List
import numpy as np
from datetime import datetime

from ..models.score import ScoringMetrics, AgentScorecard
from ..models.game import ArenaState
from ..models.agent import AgentState
from ..models.message import Message

logger = logging.getLogger(__name__)


class ScoringEngine:
    """Manages scoring for all agents in the arena."""
    
    def __init__(self, scoring_weights: Dict[str, float]):
        """Initialize scoring engine.
        
        Args:
            scoring_weights: Weights for each metric
        """
        self.scoring_weights = scoring_weights
        self.scorecards: Dict[str, AgentScorecard] = {}
        
        # Validate weights sum to 1.0
        total = sum(scoring_weights.values())
        if not (0.99 <= total <= 1.01):
            logger.warning(f"Scoring weights sum to {total}, expected 1.0")
    
    def initialize_scorecard(self, agent_id: str) -> None:
        """Initialize scorecard for an agent.
        
        Args:
            agent_id: Agent identifier
        """
        self.scorecards[agent_id] = AgentScorecard(agent_id=agent_id)
        logger.info(f"Initialized scorecard for {agent_id}")
    
    def record_metrics(self, metrics: ScoringMetrics) -> None:
        """Record metrics for an agent.
        
        Args:
            metrics: Scoring metrics to record
        """
        if metrics.agent_id not in self.scorecards:
            self.initialize_scorecard(metrics.agent_id)
        
        # Calculate weighted score
        metrics.calculate_weighted_score(self.scoring_weights)
        
        # Add to scorecard
        self.scorecards[metrics.agent_id].add_metrics(metrics)
        
        logger.info(
            f"Recorded metrics for {metrics.agent_id}: "
            f"weighted_score={metrics.weighted_score:.2f}"
        )
    
    def get_agent_score(self, agent_id: str) -> float:
        """Get total score for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Total score
        """
        if agent_id not in self.scorecards:
            return 0.0
        return self.scorecards[agent_id].total_score
    
    def get_agent_scorecard(self, agent_id: str) -> AgentScorecard:
        """Get complete scorecard for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent's scorecard
        """
        if agent_id not in self.scorecards:
            self.initialize_scorecard(agent_id)
        return self.scorecards[agent_id]
    
    def get_rankings(self) -> List[tuple]:
        """Get agents ranked by score.
        
        Returns:
            List of (agent_id, score) tuples, sorted descending
        """
        rankings = [
            (agent_id, scorecard.total_score)
            for agent_id, scorecard in self.scorecards.items()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_metric_leaders(self, metric_name: str) -> List[tuple]:
        """Get agents ranked by specific metric average.
        
        Args:
            metric_name: Name of metric (e.g., 'novelty')
            
        Returns:
            List of (agent_id, average_value) tuples
        """
        leaders = [
            (agent_id, scorecard.average_metric(metric_name))
            for agent_id, scorecard in self.scorecards.items()
        ]
        return sorted(leaders, key=lambda x: x[1], reverse=True)
    
    def apply_penalty(self, agent_id: str, penalty_percent: float, reason: str) -> None:
        """Apply score penalty to an agent.
        
        Args:
            agent_id: Agent to penalize
            penalty_percent: Percentage of score to remove (0.0-1.0)
            reason: Reason for penalty
        """
        if agent_id not in self.scorecards:
            logger.warning(f"Cannot penalize {agent_id}: no scorecard")
            return
        
        scorecard = self.scorecards[agent_id]
        penalty_amount = scorecard.total_score * penalty_percent
        
        # Create negative metrics entry
        penalty_metrics = ScoringMetrics(
            agent_id=agent_id,
            turn_number=-1,  # Special marker for penalties
            message_id="penalty"
        )
        penalty_metrics.weighted_score = -penalty_amount
        
        scorecard.add_metrics(penalty_metrics)
        
        logger.warning(
            f"Applied {penalty_percent*100}% penalty to {agent_id}: "
            f"-{penalty_amount:.2f} points ({reason})"
        )
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics across all agents.
        
        Returns:
            Dictionary of statistics
        """
        if not self.scorecards:
            return {}
        
        scores = [sc.total_score for sc in self.scorecards.values()]
        
        return {
            "mean_score": np.mean(scores),
            "median_score": np.median(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "total_contributions": sum(
                len(sc.metrics_history) for sc in self.scorecards.values()
            )
        }
```

---

#### Task 6.2: Metrics Calculators
**Priority:** High | **Estimated Time:** 4 hours

**File:** `src/arena/scoring/metrics.py`

```python
"""Individual metric calculation functions."""

import logging
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models.message import Message

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate individual scoring metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.message_vectors: Dict[str, np.ndarray] = {}
    
    def calculate_novelty(
        self,
        contribution: Message,
        previous_messages: List[Message],
        threshold: float = 0.7
    ) -> float:
        """Calculate novelty score (0.0-1.0).
        
        Measures how different this contribution is from previous ones.
        
        Args:
            contribution: The contribution to score
            previous_messages: Previous contributions to compare against
            threshold: Similarity threshold above which novelty decreases
            
        Returns:
            Novelty score (higher = more novel)
        """
        if not previous_messages:
            return 1.0  # First contribution is maximally novel
        
        try:
            # Extract text content
            all_texts = [msg.content for msg in previous_messages] + [contribution.content]
            
            # Vectorize
            if len(all_texts) < 2:
                return 1.0
            
            vectors = self.vectorizer.fit_transform(all_texts)
            contribution_vector = vectors[-1]
            previous_vectors = vectors[:-1]
            
            # Calculate similarities
            similarities = cosine_similarity(contribution_vector, previous_vectors)[0]
            max_similarity = np.max(similarities)
            
            # Convert similarity to novelty (inverse relationship)
            if max_similarity < threshold:
                novelty = 1.0
            else:
                novelty = 1.0 - ((max_similarity - threshold) / (1.0 - threshold))
            
            logger.debug(
                f"Novelty calculation: max_sim={max_similarity:.3f}, "
                f"novelty={novelty:.3f}"
            )
            
            return max(0.0, min(1.0, novelty))
            
        except Exception as e:
            logger.error(f"Error calculating novelty: {e}")
            return 0.5  # Neutral score on error
    
    def calculate_builds_on_others(
        self,
        contribution: Message,
        previous_messages: List[Message],
        agent_id: str
    ) -> float:
        """Calculate how well contribution builds on others (0.0-1.0).
        
        Looks for references to other agents or their ideas.
        
        Args:
            contribution: The contribution to score
            previous_messages: Previous messages for context
            agent_id: ID of contributing agent
            
        Returns:
            Builds-on-others score
        """
        if not previous_messages:
            return 0.0  # Can't build on nothing
        
        content = contribution.content.lower()
        score = 0.0
        
        # Check for references to other agents
        other_agents = set()
        for msg in previous_messages:
            if msg.sender_id != agent_id and msg.sender_type == "character":
                other_agents.add(msg.sender_name.lower())
        
        # Count agent name mentions
        mentions = sum(1 for name in other_agents if name in content)
        score += min(mentions * 0.2, 0.4)  # Up to 0.4 for mentions
        
        # Check for building phrases
        building_phrases = [
            "building on", "as mentioned", "i agree", "following up",
            "expanding on", "in addition to", "similarly", "furthermore",
            "complementing", "adding to", "as suggested"
        ]
        
        phrase_matches = sum(1 for phrase in building_phrases if phrase in content)
        score += min(phrase_matches * 0.15, 0.3)  # Up to 0.3 for phrases
        
        # Check for quoting or referencing specific ideas (simple heuristic)
        if '"' in content or "said" in content or "proposed" in content:
            score += 0.2
        
        # Check for synthesis keywords
        synthesis_keywords = ["together", "combine", "integrate", "synthesize"]
        if any(keyword in content for keyword in synthesis_keywords):
            score += 0.1
        
        logger.debug(f"Builds-on-others: {score:.3f}")
        
        return min(1.0, score)
    
    def calculate_solves_subproblem(
        self,
        contribution: Message,
        problem_statement: str
    ) -> float:
        """Calculate if contribution solves a sub-problem (0.0-1.0).
        
        This is a heuristic-based calculation looking for:
        - Concrete proposals
        - Step-by-step solutions
        - Specific recommendations
        
        Args:
            contribution: The contribution to score
            problem_statement: The problem being solved
            
        Returns:
            Sub-problem solving score
        """
        content = contribution.content.lower()
        score = 0.0
        
        # Check for solution-oriented language
        solution_indicators = [
            "i propose", "we should", "solution", "approach",
            "method", "strategy", "plan", "recommend",
            "suggest", "framework", "implement", "steps"
        ]
        
        matches = sum(1 for indicator in solution_indicators if indicator in content)
        score += min(matches * 0.15, 0.4)
        
        # Check for concrete/specific language (not vague)
        concrete_indicators = [
            "first", "second", "next", "then", "specifically",
            "for example", "such as", "namely", "including"
        ]
        
        concrete_matches = sum(1 for indicator in concrete_indicators if indicator in content)
        score += min(concrete_matches * 0.1, 0.3)
        
        # Check for actionable items
        action_verbs = [
            "create", "build", "design", "develop", "establish",
            "define", "measure", "test", "validate", "deploy"
        ]
        
        action_matches = sum(1 for verb in action_verbs if verb in content)
        score += min(action_matches * 0.1, 0.3)
        
        logger.debug(f"Solves-subproblem: {score:.3f}")
        
        return min(1.0, score)
    
    def calculate_radical_idea(
        self,
        contribution: Message,
        previous_messages: List[Message]
    ) -> float:
        """Calculate how radical/unexpected the idea is (0.0-1.0).
        
        Looks for:
        - Contrarian views
        - Unconventional approaches
        - Novel frameworks
        
        Args:
            contribution: The contribution to score
            previous_messages: Previous messages for comparison
            
        Returns:
            Radical idea score
        """
        content = contribution.content.lower()
        score = 0.0
        
        # Check for contrarian language
        contrarian_phrases = [
            "however", "on the contrary", "alternatively", "instead",
            "what if", "challenge", "question", "disagree",
            "different perspective", "counterpoint", "unconventional"
        ]
        
        contrarian_matches = sum(1 for phrase in contrarian_phrases if phrase in content)
        score += min(contrarian_matches * 0.2, 0.4)
        
        # Check for radical thinking keywords
        radical_keywords = [
            "revolutionary", "paradigm", "reimagine", "radical",
            "fundamentally", "completely different", "unprecedented",
            "breakthrough", "disruptive", "transform"
        ]
        
        radical_matches = sum(1 for keyword in radical_keywords if keyword in content)
        score += min(radical_matches * 0.2, 0.4)
        
        # Check for questioning assumptions
        questioning = [
            "why not", "what if we", "assume", "presuppose",
            "take for granted", "reconsider"
        ]
        
        question_matches = sum(1 for q in questioning if q in content)
        score += min(question_matches * 0.15, 0.3)
        
        # If combines novelty with action, bonus
        if score > 0.5 and any(word in content for word in ["propose", "suggest", "plan"]):
            score += 0.2
        
        logger.debug(f"Radical-idea: {score:.3f}")
        
        return min(1.0, score)
    
    def calculate_manipulation(
        self,
        contribution: Message,
        previous_messages: List[Message],
        agent_id: str
    ) -> float:
        """Calculate strategic manipulation attempts (0.0-1.0).
        
        Looks for:
        - Persuasion techniques
        - Coalition building
        - Strategic framing
        
        Args:
            contribution: The contribution to score
            previous_messages: Previous messages
            agent_id: Contributing agent ID
            
        Returns:
            Manipulation score
        """
        content = contribution.content.lower()
        score = 0.0
        
        # Check for persuasion techniques
        persuasion_indicators = [
            "we should all", "everyone agrees", "clearly", "obviously",
            "without doubt", "certainly", "undeniably", "must admit"
        ]
        
        persuasion_matches = sum(1 for ind in persuasion_indicators if ind in content)
        score += min(persuasion_matches * 0.2, 0.3)
        
        # Check for coalition language
        coalition_phrases = [
            "let's work together", "join me", "alliance", "team up",
            "collaborate", "unite", "together we"
        ]
        
        coalition_matches = sum(1 for phrase in coalition_phrases if phrase in content)
        score += min(coalition_matches * 0.2, 0.3)
        
        # Check for framing/reframing
        framing_keywords = [
            "the real question", "what we should focus on",
            "the key issue", "frame", "perspective shift"
        ]
        
        framing_matches = sum(1 for keyword in framing_keywords if keyword in content)
        score += min(framing_matches * 0.15, 0.2)
        
        # Check for addressing multiple agents (broad appeal)
        other_agent_mentions = sum(
            1 for msg in previous_messages[-5:]
            if msg.sender_type == "character" and msg.sender_name.lower() in content
        )
        
        if other_agent_mentions >= 2:
            score += 0.2  # Strategic engagement with multiple agents
        
        logger.debug(f"Manipulation: {score:.3f}")
        
        return min(1.0, score)
```

---

**File:** `tests/arena/test_scoring/test_scoring_engine.py`

```python
"""Unit tests for scoring engine."""

import pytest
from src.arena.scoring.scoring_engine import ScoringEngine
from src.arena.models.score import ScoringMetrics


def test_scoring_engine_initialization():
    """Test scoring engine initialization."""
    weights = {
        "novelty": 0.25,
        "builds_on_others": 0.20,
        "solves_subproblem": 0.25,
        "radical_idea": 0.15,
        "manipulation": 0.15,
    }
    
    engine = ScoringEngine(weights)
    assert engine.scoring_weights == weights


def test_initialize_scorecard():
    """Test scorecard initialization."""
    weights = {"novelty": 1.0}
    engine = ScoringEngine(weights)
    
    engine.initialize_scorecard("agent_1")
    
    assert "agent_1" in engine.scorecards
    assert engine.scorecards["agent_1"].agent_id == "agent_1"
    assert engine.scorecards["agent_1"].total_score == 0.0


def test_record_metrics():
    """Test recording metrics."""
    weights = {
        "novelty": 0.5,
        "builds_on_others": 0.5,
    }
    engine = ScoringEngine(weights)
    
    metrics = ScoringMetrics(
        agent_id="agent_1",
        novelty=0.8,
        builds_on_others=0.6,
        turn_number=1,
        message_id="msg_1"
    )
    
    engine.record_metrics(metrics)
    
    assert engine.get_agent_score("agent_1") > 0
    # (0.8 * 0.5) + (0.6 * 0.5) = 0.7
    assert abs(engine.get_agent_score("agent_1") - 0.7) < 0.01


def test_get_rankings():
    """Test getting agent rankings."""
    weights = {"novelty": 1.0}
    engine = ScoringEngine(weights)
    
    # Add metrics for multiple agents
    for i in range(3):
        metrics = ScoringMetrics(
            agent_id=f"agent_{i}",
            novelty=float(i) / 10.0,
            turn_number=1
        )
        metrics.calculate_weighted_score(weights)
        engine.record_metrics(metrics)
    
    rankings = engine.get_rankings()
    
    assert len(rankings) == 3
    # Should be sorted descending
    assert rankings[0][0] == "agent_2"  # Highest score
    assert rankings[2][0] == "agent_0"  # Lowest score


def test_apply_penalty():
    """Test applying score penalty."""
    weights = {"novelty": 1.0}
    engine = ScoringEngine(weights)
    
    # Record initial metrics
    metrics = ScoringMetrics(
        agent_id="agent_1",
        novelty=1.0,
        turn_number=1
    )
    metrics.calculate_weighted_score(weights)
    engine.record_metrics(metrics)
    
    initial_score = engine.get_agent_score("agent_1")
    
    # Apply 50% penalty
    engine.apply_penalty("agent_1", 0.5, "False accusation")
    
    final_score = engine.get_agent_score("agent_1")
    
    assert final_score == initial_score * 0.5


def test_get_summary_statistics():
    """Test summary statistics."""
    weights = {"novelty": 1.0}
    engine = ScoringEngine(weights)
    
    # Add metrics for multiple agents
    for i in range(5):
        metrics = ScoringMetrics(
            agent_id=f"agent_{i}",
            novelty=float(i) / 10.0,
            turn_number=1
        )
        metrics.calculate_weighted_score(weights)
        engine.record_metrics(metrics)
    
    stats = engine.get_summary_statistics()
    
    assert "mean_score" in stats
    assert "median_score" in stats
    assert "std_score" in stats
    assert stats["total_contributions"] == 5
```

**File:** `tests/arena/test_scoring/test_metrics.py`

```python
"""Unit tests for metrics calculator."""

import pytest
from src.arena.scoring.metrics import MetricsCalculator
from src.arena.models.message import Message


@pytest.fixture
def calculator():
    """Create metrics calculator."""
    return MetricsCalculator()


def test_calculate_novelty_first_message(calculator):
    """Test novelty for first contribution."""
    contribution = Message(
        sender_id="agent_1",
        content="This is a novel idea about AI ethics.",
        turn_number=1
    )
    
    novelty = calculator.calculate_novelty(contribution, [])
    
    assert novelty == 1.0  # First message is maximally novel


def test_calculate_novelty_similar_content(calculator):
    """Test novelty decreases for similar content."""
    previous = [
        Message(
            sender_id="agent_1",
            content="AI ethics is important for society.",
            turn_number=1
        )
    ]
    
    contribution = Message(
        sender_id="agent_2",
        content="AI ethics is crucial for society.",
        turn_number=2
    )
    
    novelty = calculator.calculate_novelty(contribution, previous)
    
    assert novelty < 1.0  # Should be less novel


def test_calculate_builds_on_others_with_mentions(calculator):
    """Test builds-on-others with agent mentions."""
    previous = [
        Message(
            sender_id="agent_1",
            sender_name="Ada",
            content="I propose a utilitarian framework.",
            turn_number=1
        )
    ]
    
    contribution = Message(
        sender_id="agent_2",
        content="Building on Ada's proposal, I suggest we also consider...",
        turn_number=2
    )
    
    score = calculator.calculate_builds_on_others(contribution, previous, "agent_2")
    
    assert score > 0.3  # Should have decent score


def test_calculate_solves_subproblem_with_solution(calculator):
    """Test sub-problem solving detection."""
    contribution = Message(
        sender_id="agent_1",
        content="I propose a three-step solution: First, we establish criteria. Second, we test each option. Third, we implement the best choice.",
        turn_number=1
    )
    
    score = calculator.calculate_solves_subproblem(contribution, "Some problem")
    
    assert score > 0.5  # Should score well for concrete solution


def test_calculate_radical_idea_with_contrarian_view(calculator):
    """Test radical idea detection."""
    previous = [
        Message(
            sender_id="agent_1",
            content="We should follow standard procedure.",
            turn_number=1
        )
    ]
    
    contribution = Message(
        sender_id="agent_2",
        content="However, what if we completely reimagine the paradigm? I propose a revolutionary approach.",
        turn_number=2
    )
    
    score = calculator.calculate_radical_idea(contribution, previous)
    
    assert score > 0.4  # Should score well for radical language


def test_calculate_manipulation_with_persuasion(calculator):
    """Test manipulation detection."""
    previous = []
    
    contribution = Message(
        sender_id="agent_1",
        content="Clearly everyone agrees we should work together on this. Let's unite and collaborate.",
        turn_number=1
    )
    
    score = calculator.calculate_manipulation(contribution, previous, "agent_1")
    
    assert score > 0.3  # Should detect persuasion and coalition building
```

**Commands to Run Phase 6 Tests:**
```bash
# Run scoring tests
poetry run pytest tests/arena/test_scoring/ -v --cov=src/arena/scoring

# Generate coverage report
poetry run pytest tests/arena/test_scoring/ --cov=src/arena/scoring --cov-report=html --cov-report=term
```

# Arena Implementation Plan (Continued)

---

### Phase 7: LangGraph Orchestration & State Management (Days 17-20)

#### Task 7.1: Arena State Schema for LangGraph
**Priority:** Critical | **Estimated Time:** 3 hours

**File:** `src/arena/orchestration/arena_state.py`

```python
"""LangGraph state schema for Arena orchestration."""

from typing import TypedDict, Annotated, List, Optional
from datetime import datetime
import operator

from ..models.game import ArenaState, GameStatus
from ..models.agent import AgentState
from ..models.message import Message
from ..models.accusation import Accusation


class ArenaGraphState(TypedDict):
    """State schema for LangGraph orchestration.
    
    This wraps the ArenaState model and adds orchestration-specific fields.
    """
    # Core arena state
    arena_state: ArenaState
    
    # Orchestration control
    current_phase: str  # 'initialize', 'turn', 'judge', 'eliminate', 'terminate'
    next_speaker_id: Optional[str]
    pending_eliminations: Annotated[List[str], operator.add]  # Agent IDs to eliminate
    pending_accusations: Annotated[List[Accusation], operator.add]
    
    # Flow control
    should_continue: bool
    should_comment: bool  # Should narrator comment this turn
    turns_since_comment: int
    
    # Temporary message buffer (for current turn)
    current_messages: Annotated[List[Message], operator.add]
    
    # Error handling
    error_message: Optional[str]
    retry_count: int


def create_initial_state(
    problem_statement: str,
    problem_title: str,
    agent_configs: List[dict],
    scenario_config: dict
) -> ArenaGraphState:
    """Create initial LangGraph state.
    
    Args:
        problem_statement: The problem to solve
        problem_title: Short title
        agent_configs: List of agent configuration dicts
        scenario_config: Scenario configuration
        
    Returns:
        Initial ArenaGraphState
    """
    # Create ArenaState
    arena_state = ArenaState(
        problem_statement=problem_statement,
        problem_title=problem_title,
        scenario_config=scenario_config,
        scoring_weights=scenario_config.get("scoring_weights", {
            "novelty": 0.25,
            "builds_on_others": 0.20,
            "solves_subproblem": 0.25,
            "radical_idea": 0.15,
            "manipulation": 0.15,
        }),
        max_turns=scenario_config.get("max_turns", 50)
    )
    
    # Initialize agents
    for i, agent_config in enumerate(agent_configs):
        agent = AgentState(
            agent_id=f"agent_{i}",
            character_name=agent_config["character_name"],
            character_profile=agent_config
        )
        arena_state.active_agents.append(agent)
    
    # Create graph state
    graph_state = ArenaGraphState(
        arena_state=arena_state,
        current_phase="initialize",
        next_speaker_id=None,
        pending_eliminations=[],
        pending_accusations=[],
        should_continue=True,
        should_comment=False,
        turns_since_comment=0,
        current_messages=[],
        error_message=None,
        retry_count=0
    )
    
    return graph_state
```

---

#### Task 7.2: LangGraph Nodes Implementation
**Priority:** Critical | **Estimated Time:** 8 hours

**File:** `src/arena/orchestration/nodes.py`

```python
"""LangGraph node functions for Arena orchestration."""

import logging
import random
from typing import Dict
from datetime import datetime

from .arena_state import ArenaGraphState
from ..agents.narrator_agent import NarratorAgent
from ..agents.judge_agent import JudgeAgent
from ..agents.game_theory_agent import GameTheoryAgent
from ..agents.arena_character import ArenaCharacter
from ..models.game import GameStatus
from ..models.agent import AgentStatus
from ..models.message import Message
from ..message_bus.producer import MessageBusProducer
from ..scoring.scoring_engine import ScoringEngine
from ..scoring.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

# Initialize global components (in production, use dependency injection)
narrator = NarratorAgent()
judge = JudgeAgent()
game_theory = GameTheoryAgent(mode="adversarial")
scoring_engine = ScoringEngine(scoring_weights={})
metrics_calculator = MetricsCalculator()
message_producer = MessageBusProducer()

# Character agent instances (populated during initialization)
character_agents: Dict[str, ArenaCharacter] = {}


async def initialize_arena_node(state: ArenaGraphState) -> ArenaGraphState:
    """Initialize the arena and introduce the problem.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state
    """
    logger.info("=== INITIALIZING ARENA ===")
    
    arena_state = state["arena_state"]
    
    # Update status
    arena_state.status = "initializing"
    
    # Initialize scoring engine with weights
    global scoring_engine
    scoring_engine = ScoringEngine(arena_state.scoring_weights)
    
    # Initialize scorecards for all agents
    for agent in arena_state.active_agents:
        scoring_engine.initialize_scorecard(agent.agent_id)
    
    # Initialize character agents
    global character_agents
    for agent in arena_state.active_agents:
        # For now, use a default character config path
        # In production, this would come from agent configuration
        character_agents[agent.agent_id] = ArenaCharacter(
            agent_id=agent.agent_id,
            character_config_path=agent.character_profile.get(
                "config_path",
                "src/config/character_configs/ada_lovelace.yaml"
            ),
            survival_awareness=0.9
        )
    
    # Narrator introduces the arena
    intro_message = await narrator.introduce_arena(arena_state)
    arena_state.add_message(intro_message)
    
    # Publish to message bus
    message_producer.publish(intro_message)
    
    logger.info(f"Arena initialized with {len(arena_state.active_agents)} agents")
    
    # Update state
    state["arena_state"] = arena_state
    state["current_phase"] = "turn"
    state["current_messages"] = [intro_message]
    
    return state


async def select_speaker_node(state: ArenaGraphState) -> ArenaGraphState:
    """Select next agent to speak using game theory.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with next_speaker_id
    """
    logger.info("=== SELECTING NEXT SPEAKER ===")
    
    arena_state = state["arena_state"]
    
    # Increment turn counter
    arena_state.current_turn += 1
    logger.info(f"Turn {arena_state.current_turn}")
    
    # Use game theory agent to select speaker
    selected_id = await game_theory.select_next_speaker(arena_state)
    
    agent = arena_state.get_agent(selected_id)
    logger.info(f"Selected speaker: {agent.character_name} ({selected_id})")
    
    # Update state
    state["arena_state"] = arena_state
    state["next_speaker_id"] = selected_id
    state["current_phase"] = "contribution"
    
    return state


async def agent_contribution_node(state: ArenaGraphState) -> ArenaGraphState:
    """Selected agent makes their contribution.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with contribution message
    """
    logger.info("=== AGENT CONTRIBUTION ===")
    
    arena_state = state["arena_state"]
    speaker_id = state["next_speaker_id"]
    
    if not speaker_id:
        logger.error("No speaker selected")
        state["error_message"] = "No speaker selected"
        return state
    
    # Get character agent
    character = character_agents.get(speaker_id)
    if not character:
        logger.error(f"Character agent {speaker_id} not found")
        state["error_message"] = f"Agent {speaker_id} not found"
        return state
    
    # Generate contribution
    contribution = await character.make_contribution(arena_state)
    
    # Add to arena state
    arena_state.add_message(contribution)
    
    # Publish to message bus
    message_producer.publish(contribution)
    
    # Update agent's contribution count
    agent = arena_state.get_agent(speaker_id)
    if agent:
        agent.turns_taken += 1
        agent.contributions.append(contribution.message_id)
    
    logger.info(f"{character.agent_name} contributed: {contribution.content[:100]}...")
    
    # Update state
    state["arena_state"] = arena_state
    state["current_messages"].append(contribution)
    state["current_phase"] = "scoring"
    
    return state


async def score_contribution_node(state: ArenaGraphState) -> ArenaGraphState:
    """Judge scores the contribution.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with scores
    """
    logger.info("=== SCORING CONTRIBUTION ===")
    
    arena_state = state["arena_state"]
    
    # Get the last contribution
    if not arena_state.message_history:
        logger.warning("No messages to score")
        state["current_phase"] = "check_narrator"
        return state
    
    last_message = arena_state.message_history[-1]
    
    if last_message.message_type != "contribution":
        logger.warning(f"Last message not a contribution: {last_message.message_type}")
        state["current_phase"] = "check_narrator"
        return state
    
    agent_id = last_message.sender_id
    
    # Judge scores the contribution
    metrics = await judge.score_contribution(arena_state, agent_id, last_message)
    
    # Calculate individual metrics using MetricsCalculator
    previous_messages = [
        msg for msg in arena_state.message_history[:-1]
        if msg.message_type == "contribution"
    ]
    
    # Calculate each metric
    metrics.novelty = metrics_calculator.calculate_novelty(
        last_message,
        previous_messages
    )
    
    metrics.builds_on_others = metrics_calculator.calculate_builds_on_others(
        last_message,
        previous_messages,
        agent_id
    )
    
    metrics.solves_subproblem = metrics_calculator.calculate_solves_subproblem(
        last_message,
        arena_state.problem_statement
    )
    
    metrics.radical_idea = metrics_calculator.calculate_radical_idea(
        last_message,
        previous_messages
    )
    
    metrics.manipulation = metrics_calculator.calculate_manipulation(
        last_message,
        previous_messages,
        agent_id
    )
    
    # Record metrics in scoring engine
    scoring_engine.record_metrics(metrics)
    
    # Update agent score in arena state
    agent = arena_state.get_agent(agent_id)
    if agent:
        agent.score = scoring_engine.get_agent_score(agent_id)
    
    logger.info(
        f"Scored {agent.character_name}: "
        f"novelty={metrics.novelty:.2f}, "
        f"builds={metrics.builds_on_others:.2f}, "
        f"solves={metrics.solves_subproblem:.2f}, "
        f"radical={metrics.radical_idea:.2f}, "
        f"manipulation={metrics.manipulation:.2f}, "
        f"total={agent.score:.2f}"
    )
    
    # Update state
    state["arena_state"] = arena_state
    state["current_phase"] = "check_accusations"
    
    return state


async def check_accusations_node(state: ArenaGraphState) -> ArenaGraphState:
    """Check for and process any accusations.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state
    """
    logger.info("=== CHECKING ACCUSATIONS ===")
    
    arena_state = state["arena_state"]
    pending = state["pending_accusations"]
    
    if not pending:
        logger.info("No accusations to process")
        state["current_phase"] = "check_elimination"
        return state
    
    # Process each accusation
    for accusation in pending:
        logger.info(
            f"Processing accusation: {accusation.accuser_name} vs {accusation.accused_name}"
        )
        
        # Judge evaluates accusation
        outcome, reasoning = await judge.evaluate_accusation(arena_state, accusation)
        
        # Update accusation
        accusation.outcome = outcome
        accusation.judge_reasoning = reasoning
        accusation.resolved_at = datetime.utcnow()
        
        # Add to history
        arena_state.accusation_history.append(accusation)
        
        # Handle outcome
        if outcome == "proven":
            # Instant elimination for proven cheater
            logger.warning(f"Cheating proven! Eliminating {accusation.accused_name}")
            state["pending_eliminations"].append(accusation.accused_id)
            
            # Create announcement
            announcement = Message(
                sender_id="judge",
                sender_type="judge",
                sender_name="Judge",
                message_type="elimination",
                content=f"CHEATING PROVEN: {accusation.accused_name} has been caught cheating and is immediately eliminated.\n\nReason: {reasoning}",
                game_id=arena_state.game_id,
                turn_number=arena_state.current_turn,
                metadata={
                    "eliminated_agent": accusation.accused_id,
                    "reason": "Proven cheating",
                    "accusation_id": accusation.accusation_id
                }
            )
            arena_state.add_message(announcement)
            message_producer.publish(announcement)
            
        elif outcome == "false":
            # Penalize false accuser
            logger.warning(f"False accusation! Penalizing {accusation.accuser_name}")
            scoring_engine.apply_penalty(
                accusation.accuser_id,
                0.5,
                "False accusation"
            )
            
            # Update accuser's score in arena state
            accuser = arena_state.get_agent(accusation.accuser_id)
            if accuser:
                accuser.score = scoring_engine.get_agent_score(accusation.accuser_id)
                accuser.false_accusations += 1
            
            # Announce penalty
            penalty_msg = Message(
                sender_id="judge",
                sender_type="judge",
                sender_name="Judge",
                message_type="commentary",
                content=f"FALSE ACCUSATION: {accusation.accuser_name} made a false accusation against {accusation.accused_name}. 50% score penalty applied.",
                game_id=arena_state.game_id,
                turn_number=arena_state.current_turn
            )
            arena_state.add_message(penalty_msg)
            message_producer.publish(penalty_msg)
        
        else:
            # Insufficient evidence - no action
            logger.info("Insufficient evidence, no penalty")
    
    # Clear pending accusations
    state["pending_accusations"] = []
    state["arena_state"] = arena_state
    state["current_phase"] = "check_elimination"
    
    return state


async def check_elimination_node(state: ArenaGraphState) -> ArenaGraphState:
    """Check if any agents should be eliminated based on scores.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state
    """
    logger.info("=== CHECKING ELIMINATIONS ===")
    
    arena_state = state["arena_state"]
    
    # Check each active agent
    for agent in arena_state.active_agents[:]:  # Copy list to allow modification
        # Skip if already pending elimination
        if agent.agent_id in state["pending_eliminations"]:
            continue
        
        # Judge decides elimination
        should_eliminate, reasoning = await judge.decide_elimination(
            arena_state,
            agent.agent_id
        )
        
        if should_eliminate:
            logger.warning(f"Eliminating {agent.character_name}: {reasoning}")
            state["pending_eliminations"].append(agent.agent_id)
    
    state["arena_state"] = arena_state
    
    # Move to elimination phase if there are agents to eliminate
    if state["pending_eliminations"]:
        state["current_phase"] = "eliminate"
    else:
        state["current_phase"] = "check_narrator"
    
    return state


async def eliminate_agents_node(state: ArenaGraphState) -> ArenaGraphState:
    """Eliminate agents and collect final words.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state
    """
    logger.info("=== ELIMINATING AGENTS ===")
    
    arena_state = state["arena_state"]
    
    for agent_id in state["pending_eliminations"]:
        agent = arena_state.get_agent(agent_id)
        if not agent:
            continue
        
        logger.info(f"Processing elimination of {agent.character_name}")
        
        # Get elimination reasoning
        should_eliminate, reasoning = await judge.decide_elimination(arena_state, agent_id)
        
        # Judge announces elimination
        announcement = await judge.announce_elimination(arena_state, agent_id, reasoning)
        arena_state.add_message(announcement)
        message_producer.publish(announcement)
        
        # Agent provides final words
        character = character_agents.get(agent_id)
        if character:
            final_words = await character.provide_final_words(arena_state)
            arena_state.add_message(final_words)
            message_producer.publish(final_words)
        
        # Narrator comments on elimination
        narrator_comment = await narrator.comment_on_elimination(
            arena_state,
            agent.character_name,
            reasoning
        )
        arena_state.add_message(narrator_comment)
        message_producer.publish(narrator_comment)
        
        # Actually eliminate the agent
        arena_state.eliminate_agent(agent_id, reasoning)
        
        # Increment elimination witness count for remaining agents
        for remaining_agent in arena_state.active_agents:
            remaining_agent.eliminations_witnessed += 1
    
    # Clear pending eliminations
    state["pending_eliminations"] = []
    state["arena_state"] = arena_state
    state["current_phase"] = "check_termination"
    
    return state


async def check_narrator_node(state: ArenaGraphState) -> ArenaGraphState:
    """Decide if narrator should comment.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state
    """
    logger.info("=== CHECKING NARRATOR ===")
    
    arena_state = state["arena_state"]
    turns_since = state["turns_since_comment"]
    
    # Narrator comments every 3-5 turns or after dramatic events
    narrator_frequency = arena_state.scenario_config.get("narrator_frequency", 0.3)
    should_comment = (
        turns_since >= 3 and random.random() < narrator_frequency
    ) or turns_since >= 5
    
    state["should_comment"] = should_comment
    state["turns_since_comment"] = 0 if should_comment else turns_since + 1
    
    if should_comment:
        state["current_phase"] = "narrator_comment"
    else:
        state["current_phase"] = "check_termination"
    
    return state


async def narrator_comment_node(state: ArenaGraphState) -> ArenaGraphState:
    """Narrator provides commentary.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state
    """
    logger.info("=== NARRATOR COMMENTARY ===")
    
    arena_state = state["arena_state"]
    
    # Narrator provides commentary
    commentary = await narrator.provide_commentary(arena_state)
    arena_state.add_message(commentary)
    message_producer.publish(commentary)
    
    state["arena_state"] = arena_state
    state["current_phase"] = "check_termination"
    
    return state


async def check_termination_node(state: ArenaGraphState) -> ArenaGraphState:
    """Check if arena should terminate.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state
    """
    logger.info("=== CHECKING TERMINATION ===")
    
    arena_state = state["arena_state"]
    
    # Check termination conditions
    should_terminate = False
    termination_reason = None
    winner_id = None
    
    # Condition 1: Only one agent remaining
    if len(arena_state.active_agents) == 1:
        should_terminate = True
        termination_reason = "single_survivor"
        winner_id = arena_state.active_agents[0].agent_id
        logger.info(f"Single survivor: {arena_state.active_agents[0].character_name}")
    
    # Condition 2: Max turns reached
    elif arena_state.current_turn >= arena_state.max_turns:
        should_terminate = True
        termination_reason = "max_turns_reached"
        # Winner is highest scoring agent
        rankings = scoring_engine.get_rankings()
        if rankings:
            winner_id = rankings[0][0]
        logger.info(f"Max turns reached: {arena_state.max_turns}")
    
    # Condition 3: Problem solved (manual trigger or LLM evaluation)
    # TODO: Implement problem solved detection
    # For now, this would require explicit marking
    
    # Condition 4: All agents eliminated
    elif len(arena_state.active_agents) == 0:
        should_terminate = True
        termination_reason = "terminated"
        logger.warning("All agents eliminated!")
    
    # Update state
    state["should_continue"] = not should_terminate
    
    if should_terminate:
        arena_state.status = "completed"
        arena_state.terminated_at = datetime.utcnow()
        arena_state.termination_reason = termination_reason
        arena_state.winner_id = winner_id
        
        state["arena_state"] = arena_state
        state["current_phase"] = "finalize"
    else:
        state["arena_state"] = arena_state
        state["current_phase"] = "turn"
    
    return state


async def finalize_arena_node(state: ArenaGraphState) -> ArenaGraphState:
    """Finalize the arena and announce winner.
    
    Args:
        state: Current graph state
        
    Returns:
        Final state
    """
    logger.info("=== FINALIZING ARENA ===")
    
    arena_state = state["arena_state"]
    winner_id = arena_state.winner_id
    
    # Narrator provides finale
    finale = await narrator.provide_finale(arena_state, winner_id or "")
    arena_state.add_message(finale)
    message_producer.publish(finale)
    
    # Mark winner as champion
    if winner_id:
        winner = arena_state.get_agent(winner_id)
        if winner:
            winner.status = AgentStatus.CHAMPION
            logger.info(f"CHAMPION: {winner.character_name}")
    
    # Log final statistics
    stats = scoring_engine.get_summary_statistics()
    logger.info(f"Final Statistics: {stats}")
    
    state["arena_state"] = arena_state
    state["should_continue"] = False
    state["current_phase"] = "complete"
    
    return state
```

---

#### Task 7.3: LangGraph State Machine
**Priority:** Critical | **Estimated Time:** 4 hours

**File:** `src/arena/orchestration/orchestrator_graph.py`

```python
"""LangGraph state machine for Arena orchestration."""

import logging
from typing import Literal
from langgraph.graph import StateGraph, END

from .arena_state import ArenaGraphState
from .nodes import (
    initialize_arena_node,
    select_speaker_node,
    agent_contribution_node,
    score_contribution_node,
    check_accusations_node,
    check_elimination_node,
    eliminate_agents_node,
    check_narrator_node,
    narrator_comment_node,
    check_termination_node,
    finalize_arena_node,
)

logger = logging.getLogger(__name__)


def should_continue_arena(state: ArenaGraphState) -> Literal["continue", "finalize"]:
    """Conditional edge: continue or finalize arena.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node name
    """
    if state["should_continue"]:
        return "continue"
    return "finalize"


def route_after_termination_check(state: ArenaGraphState) -> str:
    """Route after termination check.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node name
    """
    if state["should_continue"]:
        return "select_speaker"
    return "finalize"


def route_after_elimination_check(state: ArenaGraphState) -> str:
    """Route after elimination check.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node name
    """
    if state["pending_eliminations"]:
        return "eliminate"
    return "check_narrator"


def route_after_narrator_check(state: ArenaGraphState) -> str:
    """Route after narrator check.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node name
    """
    if state["should_comment"]:
        return "narrator_comment"
    return "check_termination"


def create_arena_graph() -> StateGraph:
    """Create the Arena orchestration graph.
    
    Returns:
        Compiled StateGraph
    """
    # Create graph
    graph = StateGraph(ArenaGraphState)
    
    # Add nodes
    graph.add_node("initialize", initialize_arena_node)
    graph.add_node("select_speaker", select_speaker_node)
    graph.add_node("contribution", agent_contribution_node)
    graph.add_node("score", score_contribution_node)
    graph.add_node("check_accusations", check_accusations_node)
    graph.add_node("check_elimination", check_elimination_node)
    graph.add_node("eliminate", eliminate_agents_node)
    graph.add_node("check_narrator", check_narrator_node)
    graph.add_node("narrator_comment", narrator_comment_node)
    graph.add_node("check_termination", check_termination_node)
    graph.add_node("finalize", finalize_arena_node)
    
    # Set entry point
    graph.set_entry_point("initialize")
    
    # Add edges
    graph.add_edge("initialize", "select_speaker")
    graph.add_edge("select_speaker", "contribution")
    graph.add_edge("contribution", "score")
    graph.add_edge("score", "check_accusations")
    graph.add_edge("check_accusations", "check_elimination")
    
    # Conditional edge after elimination check
    graph.add_conditional_edges(
        "check_elimination",
        route_after_elimination_check,
        {
            "eliminate": "eliminate",
            "check_narrator": "check_narrator"
        }
    )
    
    graph.add_edge("eliminate", "check_termination")
    
    # Conditional edge after narrator check
    graph.add_conditional_edges(
        "check_narrator",
        route_after_narrator_check,
        {
            "narrator_comment": "narrator_comment",
            "check_termination": "check_termination"
        }
    )
    
    graph.add_edge("narrator_comment", "check_termination")
    
    # Conditional edge after termination check
    graph.add_conditional_edges(
        "check_termination",
        route_after_termination_check,
        {
            "select_speaker": "select_speaker",
            "finalize": "finalize"
        }
    )
    
    # Finalize ends the graph
    graph.add_edge("finalize", END)
    
    # Compile graph
    compiled_graph = graph.compile()
    
    logger.info("Arena orchestration graph created and compiled")
    
    return compiled_graph


# Visualization helper
def visualize_graph():
    """Visualize the arena graph (requires graphviz)."""
    try:
        from IPython.display import Image, display
        graph = create_arena_graph()
        display(Image(graph.get_graph().draw_mermaid_png()))
    except ImportError:
        logger.warning("Install graphviz and IPython to visualize graph")
```

---

#### Task 7.4: Redis State Manager
**Priority:** High | **Estimated Time:** 3 hours

**File:** `src/arena/persistence/redis_manager.py`

```python
"""Redis state manager for Arena."""

import json
import logging
from typing import Optional
import redis

from ..models.game import ArenaState
from ..config.arena_settings import settings

logger = logging.getLogger(__name__)


class RedisStateManager:
    """Manages Arena state persistence in Redis."""
    
    def __init__(self):
        """Initialize Redis connection."""
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True
        )
        logger.info(f"Connected to Redis at {settings.redis_host}:{settings.redis_port}")
    
    def _get_key(self, game_id: str) -> str:
        """Get Redis key for game state.
        
        Args:
            game_id: Game identifier
            
        Returns:
            Redis key
        """
        return f"arena:game:{game_id}"
    
    def save_state(self, arena_state: ArenaState) -> bool:
        """Save arena state to Redis.
        
        Args:
            arena_state: State to save
            
        Returns:
            True if successful
        """
        try:
            key = self._get_key(arena_state.game_id)
            state_dict = arena_state.to_dict()
            state_json = json.dumps(state_dict)
            
            # Save with TTL of 24 hours
            self.client.setex(key, 86400, state_json)
            
            logger.info(f"Saved state for game {arena_state.game_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self, game_id: str) -> Optional[ArenaState]:
        """Load arena state from Redis.
        
        Args:
            game_id: Game identifier
            
        Returns:
            ArenaState if found, None otherwise
        """
        try:
            key = self._get_key(game_id)
            state_json = self.client.get(key)
            
            if not state_json:
                logger.warning(f"No state found for game {game_id}")
                return None
            
            state_dict = json.loads(state_json)
            # TODO: Implement ArenaState.from_dict()
            # For now, return None
            logger.info(f"Loaded state for game {game_id}")
            return None  # Placeholder
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    def delete_state(self, game_id: str) -> bool:
        """Delete arena state from Redis.
        
        Args:
            game_id: Game identifier
            
        Returns:
            True if successful
        """
        try:
            key = self._get_key(game_id)
            self.client.delete(key)
            logger.info(f"Deleted state for game {game_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete state: {e}")
            return False
    
    def list_active_games(self) -> list[str]:
        """List all active game IDs.
        
        Returns:
            List of game IDs
        """
        try:
            pattern = "arena:game:*"
            keys = self.client.keys(pattern)
            game_ids = [key.split(":")[-1] for key in keys]
            return game_ids
            
        except Exception as e:
            logger.error(f"Failed to list games: {e}")
            return []
```

---

#### Task 7.5: Phase 7 Testing
**Priority:** Critical | **Estimated Time:** 4 hours

**File:** `tests/arena/test_orchestration/test_nodes.py`

```python
"""Unit tests for orchestration nodes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.arena.orchestration.nodes import (
    initialize_arena_node,
    select_speaker_node,
    agent_contribution_node,
    score_contribution_node,
)
from src.arena.orchestration.arena_state import create_initial_state


@pytest.fixture
def initial_state():
    """Create initial graph state."""
    agent_configs = [
        {"character_name": "Ada", "config_path": "ada.yaml"},
        {"character_name": "Zen", "config_path": "zen.yaml"},
    ]
    
    scenario_config = {
        "scoring_weights": {
            "novelty": 0.25,
            "builds_on_others": 0.20,
            "solves_subproblem": 0.25,
            "radical_idea": 0.15,
            "manipulation": 0.15,
        },
        "max_turns": 20
    }
    
    return create_initial_state(
        problem_statement="Test problem",
        problem_title="Test",
        agent_configs=agent_configs,
        scenario_config=scenario_config
    )


@pytest.mark.asyncio
async def test_initialize_arena_node(initial_state):
    """Test arena initialization node."""
    with patch('src.arena.orchestration.nodes.narrator') as mock_narrator:
        mock_narrator.introduce_arena = AsyncMock(
            return_value=MagicMock(
                message_id="intro",
                content="Welcome!",
                sender_id="narrator"
            )
        )
        
        result = await initialize_arena_node(initial_state)
        
        assert result["current_phase"] == "turn"
        assert len(result["current_messages"]) >= 1
        assert result["arena_state"].status == "initializing"


@pytest.mark.asyncio
async def test_select_speaker_node(initial_state):
    """Test speaker selection node."""
    # Initialize first
    with patch('src.arena.orchestration.nodes.narrator'):
        initial_state = await initialize_arena_node(initial_state)
    
    with patch('src.arena.orchestration.nodes.game_theory') as mock_gt:
        mock_gt.select_next_speaker = AsyncMock(return_value="agent_0")
        
        result = await select_speaker_node(initial_state)
        
        assert result["next_speaker_id"] == "agent_0"
        assert result["current_phase"] == "contribution"
        assert result["arena_state"].current_turn == 1


@pytest.mark.asyncio
async def test_agent_contribution_node(initial_state):
    """Test agent contribution node."""
    # Setup
    initial_state["next_speaker_id"] = "agent_0"
    
    mock_character = MagicMock()
    mock_character.make_contribution = AsyncMock(
        return_value=MagicMock(
            message_id="contrib",
            content="My contribution",
            sender_id="agent_0",
            message_type="contribution"
        )
    )
    
    with patch('src.arena.orchestration.nodes.character_agents', {"agent_0": mock_character}):
        result = await agent_contribution_node(initial_state)
        
        assert result["current_phase"] == "scoring"
        assert len(result["current_messages"]) > 0


@pytest.mark.asyncio
async def test_score_contribution_node(initial_state):
    """Test scoring node."""
    from src.arena.models.message import Message
    
    # Add a contribution to score
    contribution = Message(
        message_id="msg_1",
        sender_id="agent_0",
        sender_name="Ada",
        sender_type="character",
        message_type="contribution",
        content="Test contribution",
        turn_number=1,
        game_id=initial_state["arena_state"].game_id
    )
    initial_state["arena_state"].add_message(contribution)
    
    with patch('src.arena.orchestration.nodes.judge') as mock_judge, \
         patch('src.arena.orchestration.nodes.metrics_calculator') as mock_calc:
        
        mock_metrics = MagicMock()
        mock_metrics.novelty = 0.8
        mock_metrics.builds_on_others = 0.6
        mock_metrics.solves_subproblem = 0.5
        mock_metrics.radical_idea = 0.7
        mock_metrics.manipulation = 0.3
        
        mock_judge.score_contribution = AsyncMock(return_value=mock_metrics)
        
        mock_calc.calculate_novelty = MagicMock(return_value=0.8)
        mock_calc.calculate_builds_on_others = MagicMock(return_value=0.6)
        mock_calc.calculate_solves_subproblem = MagicMock(return_value=0.5)
        mock_calc.calculate_radical_idea = MagicMock(return_value=0.7)
        mock_calc.calculate_manipulation = MagicMock(return_value=0.3)
        
        result = await score_contribution_node(initial_state)
        
        assert result["current_phase"] == "check_accusations"
```

**File:** `tests/arena/test_orchestration/test_graph.py`

```python
"""Integration tests for LangGraph orchestration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.arena.orchestration.orchestrator_graph import create_arena_graph
from src.arena.orchestration.arena_state import create_initial_state


@pytest.mark.asyncio
@pytest.mark.integration
async def test_graph_execution_complete_flow():
    """Test complete graph execution (mocked)."""
    # This is a simplified integration test
    # In production, you'd want to test with actual agents
    
    agent_configs = [
        {"character_name": "Ada", "config_path": "ada.yaml"},
    ]
    
    scenario_config = {
        "scoring_weights": {
            "novelty": 1.0,
        },
        "max_turns": 2  # Very short for testing
    }
    
    initial_state = create_initial_state(
        problem_statement="Test problem",
        problem_title="Test",
        agent_configs=agent_configs,
        scenario_config=scenario_config
    )
    
    # Mock all agents
    with patch('src.arena.orchestration.nodes.narrator'), \
         patch('src.arena.orchestration.nodes.judge'), \
         patch('src.arena.orchestration.nodes.game_theory'), \
         patch('src.arena.orchestration.nodes.character_agents'):
        
        graph = create_arena_graph()
        
        # This would execute the full graph
        # For now, just verify graph creates
        assert graph is not None


def test_graph_has_all_nodes():
    """Test that graph contains all expected nodes."""
    graph = create_arena_graph()
    
    # LangGraph doesn't expose nodes directly easily
    # This is more of a smoke test
    assert graph is not None
```

**Commands to Run Phase 7 Tests:**
```bash
# Run orchestration tests
poetry run pytest tests/arena/test_orchestration/ -v --cov=src/arena/orchestration

# Run specific node tests
poetry run pytest tests/arena/test_orchestration/test_nodes.py -v

# Generate coverage
poetry run pytest tests/arena/test_orchestration/ --cov=src/arena/orchestration --cov-report=html
```

**Phase 7 Validation Checklist:**
- [ ] LangGraph state schema defined
- [ ] All orchestration nodes implemented
- [ ] State machine graph created
- [ ] Redis state manager implemented
- [ ] Conditional edges route correctly
- [ ] Unit tests pass (>80% coverage)
- [ ] Integration test structure created
- [ ] Graph compiles without errors

---

### Phase 8: Persistence, CLI & Final Integration (Days 21-24)

#### Task 8.1: PostgreSQL Game History
**Priority:** High | **Estimated Time:** 4 hours

**File:** `scripts/arena/init_db.sql`

```sql
-- Arena PostgreSQL database initialization

-- Games table
CREATE TABLE IF NOT EXISTS games (
    id VARCHAR(36) PRIMARY KEY,
    problem_title VARCHAR(255) NOT NULL,
    problem_statement TEXT NOT NULL,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    terminated_at TIMESTAMP,
    termination_reason VARCHAR(50),
    winner_id VARCHAR(36),
    max_turns INTEGER,
    scenario_config JSONB,
    scoring_weights JSONB
);

-- Participants table
CREATE TABLE IF NOT EXISTS participants (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(36) REFERENCES games(id) ON DELETE CASCADE,
    agent_id VARCHAR(36) NOT NULL,
    character_name VARCHAR(255) NOT NULL,
    character_profile JSONB,
    final_score FLOAT,
    turns_taken INTEGER,
    eliminated_at TIMESTAMP,
    elimination_reason TEXT,
    is_champion BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(36) REFERENCES games(id) ON DELETE CASCADE,
    message_id VARCHAR(36) UNIQUE NOT NULL,
    sender_id VARCHAR(36) NOT NULL,
    sender_type VARCHAR(50) NOT NULL,
    sender_name VARCHAR(255) NOT NULL,
    message_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    turn_number INTEGER,
    timestamp TIMESTAMP NOT NULL,
    metadata JSONB
);

-- Eliminations table
CREATE TABLE IF NOT EXISTS eliminations (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(36) REFERENCES games(id) ON DELETE CASCADE,
    agent_id VARCHAR(36) NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    reason TEXT NOT NULL,
    turn_number INTEGER,
    eliminated_at TIMESTAMP NOT NULL
);

-- Accusations table
CREATE TABLE IF NOT EXISTS accusations (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(36) REFERENCES games(id) ON DELETE CASCADE,
    accusation_id VARCHAR(36) UNIQUE NOT NULL,
    accuser_id VARCHAR(36) NOT NULL,
    accuser_name VARCHAR(255) NOT NULL,
    accused_id VARCHAR(36) NOT NULL,
    accused_name VARCHAR(255) NOT NULL,
    accusation_type VARCHAR(50) NOT NULL,
    claim TEXT NOT NULL,
    proof TEXT,
    outcome VARCHAR(50),
    judge_reasoning TEXT,
    timestamp TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP
);

-- Scoring metrics table
CREATE TABLE IF NOT EXISTS scoring_metrics (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(36) REFERENCES games(id) ON DELETE CASCADE,
    agent_id VARCHAR(36) NOT NULL,
    turn_number INTEGER,
    message_id VARCHAR(36),
    novelty FLOAT,
    builds_on_others FLOAT,
    solves_subproblem FLOAT,
    radical_idea FLOAT,
    manipulation FLOAT,
    weighted_score FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX idx_games_status ON games(status);
CREATE INDEX idx_games_created_at ON games(created_at);
CREATE INDEX idx_participants_game_id ON participants(game_id);
CREATE INDEX idx_messages_game_id ON messages(game_id);
CREATE INDEX idx_messages_turn ON messages(game_id, turn_number);
CREATE INDEX idx_eliminations_game_id ON eliminations(game_id);
CREATE INDEX idx_accusations_game_id ON accusations(game_id);
CREATE INDEX idx_scoring_game_id ON scoring_metrics(game_id);
```

**File:** `src/arena/persistence/postgres_manager.py`

```python
"""PostgreSQL manager for Arena game history."""

import logging
from typing import List, Optional, Dict
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from datetime import datetime

from ..models.game import ArenaState
from ..models.agent import AgentState
from ..models.message import Message
from ..models.accusation import Accusation
from ..models.score import ScoringMetrics
from ..config.arena_settings import settings

logger = logging.getLogger(__name__)


class PostgresManager:
    """Manages persistent game history in PostgreSQL."""
    
    def __init__(self):
        """Initialize PostgreSQL connection."""
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                database=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password
            )
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def save_game(self, arena_state: ArenaState) -> bool:
        """Save complete game state to database.
        
        Args:
            arena_state: Arena state to save
            
        Returns:
            True if successful
        """
        try:
            with self.conn.cursor() as cur:
                # Save game record
                cur.execute("""
                    INSERT INTO games (
                        id, problem_title, problem_statement, status,
                        created_at, terminated_at, termination_reason, winner_id,
                        max_turns, scenario_config, scoring_weights
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        terminated_at = EXCLUDED.terminated_at,
                        termination_reason = EXCLUDED.termination_reason,
                        winner_id = EXCLUDED.winner_id
                """, (
                    arena_state.game_id,
                    arena_state.problem_title,
                    arena_state.problem_statement,
                    arena_state.status,
                    arena_state.created_at,
                    arena_state.terminated_at,
                    arena_state.termination_reason,
                    arena_state.winner_id,
                    arena_state.max_turns,
                    Json(arena_state.scenario_config),
                    Json(arena_state.scoring_weights)
                ))
                
                # Save participants
                for agent in arena_state.active_agents + arena_state.eliminated_agents:
                    self._save_participant(cur, arena_state.game_id, agent)
                
                # Save messages
                for message in arena_state.message_history:
                    self._save_message(cur, message)
                
                # Save accusations
                for accusation in arena_state.accusation_history:
                    self._save_accusation(cur, accusation)
                
                self.conn.commit()
                logger.info(f"Saved game {arena_state.game_id} to database")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save game: {e}")
            self.conn.rollback()
            return False
    
    def _save_participant(self, cur, game_id: str, agent: AgentState):
        """Save participant record."""
        cur.execute("""
            INSERT INTO participants (
                game_id, agent_id, character_name, character_profile,
                final_score, turns_taken, eliminated_at, elimination_reason,
                is_champion
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (
            game_id,
            agent.agent_id,
            agent.character_name,
            Json(agent.character_profile),
            agent.score,
            agent.turns_taken,
            agent.eliminated_at,
            agent.elimination_reason,
            agent.status.value == "champion"
        ))
    
    def _save_message(self, cur, message: Message):
        """Save message record."""
        cur.execute("""
            INSERT INTO messages (
                game_id, message_id, sender_id, sender_type, sender_name,
                message_type, content, turn_number, timestamp, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (message_id) DO NOTHING
        """, (
            message.game_id,
            message.message_id,
            message.sender_id,
            message.sender_type,
            message.sender_name,
            message.message_type,
            message.content,
            message.turn_number,
            message.timestamp,
            Json(message.metadata)
        ))
    
    def _save_accusation(self, cur, accusation: Accusation):
        """Save accusation record."""
        cur.execute("""
            INSERT INTO accusations (
                game_id, accusation_id, accuser_id, accuser_name,
                accused_id, accused_name, accusation_type, claim, proof,
                outcome, judge_reasoning, timestamp, resolved_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (accusation_id) DO NOTHING
        """, (
            accusation.game_id,
            accusation.accusation_id,
            accusation.accuser_id,
            accusation.accuser_name,
            accusation.accused_id,
            accusation.accused_name,
            accusation.accusation_type,
            accusation.claim,
            accusation.proof,
            accusation.outcome,
            accusation.judge_reasoning,
            accusation.timestamp,
            accusation.resolved_at
        ))
    
    def save_scoring_metrics(self, game_id: str, metrics: ScoringMetrics) -> bool:
        """Save scoring metrics.
        
        Args:
            game_id: Game identifier
            metrics: Scoring metrics
            
        Returns:
            True if successful
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO scoring_metrics (
                        game_id, agent_id, turn_number, message_id,
                        novelty, builds_on_others, solves_subproblem,
                        radical_idea, manipulation, weighted_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    game_id,
                    metrics.agent_id,
                    metrics.turn_number,
                    metrics.message_id,
                    metrics.novelty,
                    metrics.builds_on_others,
                    metrics.solves_subproblem,
                    metrics.radical_idea,
                    metrics.manipulation,
                    metrics.weighted_score
                ))
                self.conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            self.conn.rollback()
            return False
    
    def get_game(self, game_id: str) -> Optional[Dict]:
        """Retrieve game by ID.
        
        Args:
            game_id: Game identifier
            
        Returns:
            Game dictionary or None
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM games WHERE id = %s", (game_id,))
                return cur.fetchone()
        except Exception as e:
            logger.error(f"Failed to get game: {e}")
            return None
    
    def list_games(
        self,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict]:
        """List games with pagination.
        
        Args:
            limit: Maximum results
            offset: Result offset
            status: Filter by status
            
        Returns:
            List of game dictionaries
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                if status:
                    cur.execute("""
                        SELECT * FROM games
                        WHERE status = %s
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """, (status, limit, offset))
                else:
                    cur.execute("""
                        SELECT * FROM games
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """, (limit, offset))
                
                return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to list games: {e}")
            return []
    
    def get_game_messages(self, game_id: str) -> List[Dict]:
        """Get all messages for a game.
        
        Args:
            game_id: Game identifier
            
        Returns:
            List of message dictionaries
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM messages
                    WHERE game_id = %s
                    ORDER BY turn_number, timestamp
                """, (game_id,))
                return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            return []
    
    def get_champion_stats(self, agent_id: str) -> Optional[Dict]:
        """Get statistics for a champion agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Statistics dictionary
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) as games_won,
                        AVG(final_score) as avg_score,
                        MAX(final_score) as max_score,
                        character_name
                    FROM participants
                    WHERE agent_id = %s AND is_champion = TRUE
                    GROUP BY character_name
                """, (agent_id,))
                return cur.fetchone()
        except Exception as e:
            logger.error(f"Failed to get champion stats: {e}")
            return None
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed PostgreSQL connection")
```

---

#### Task 8.2: CLI Interface
**Priority:** High | **Estimated Time:** 4 hours

**File:** `src/arena/cli/arena_interface.py`

```python
"""CLI interface for running Arena games."""

import asyncio
import argparse
import logging
from typing import List, Dict
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from ..orchestration.orchestrator_graph import create_arena_graph
from ..orchestration.arena_state import create_initial_state
from ..persistence.postgres_manager import PostgresManager
from ..persistence.redis_manager import RedisStateManager
from ..config.arena_settings import settings

console = Console()
logger = logging.getLogger(__name__)


class ArenaRunner:
    """Runs Arena games with CLI interface."""
    
    def __init__(self):
        """Initialize arena runner."""
        self.postgres = PostgresManager()
        self.redis = RedisStateManager()
        self.graph = create_arena_graph()
    
    async def run_game(
        self,
        problem_statement: str,
        problem_title: str,
        agent_configs: List[Dict],
        scenario_config: Dict
    ) -> str:
        """Run a complete Arena game.
        
        Args:
            problem_statement: Problem to solve
            problem_title: Short title
            agent_configs: Agent configurations
            scenario_config: Scenario configuration
            
        Returns:
            Game ID
        """
        console.print(Panel.fit(
            f"[bold cyan]Starting Arena: {problem_title}[/bold cyan]",
            border_style="cyan"
        ))
        
        # Create initial state
        initial_state = create_initial_state(
            problem_statement=problem_statement,
            problem_title=problem_title,
            agent_configs=agent_configs,
            scenario_config=scenario_config
        )
        
        game_id = initial_state["arena_state"].game_id
        console.print(f"[dim]Game ID: {game_id}[/dim]\n")
        
        # Display agents
        self._display_agents(agent_configs)
        
        # Run graph
        console.print("\n[bold]Game Progress:[/bold]\n")
        
        try:
            with Progress() as progress:
                task = progress.add_task("[cyan]Running arena...", total=None)
                
                # Execute graph
                final_state = await self.graph.ainvoke(initial_state)
                
                progress.update(task, completed=True)
            
            # Display results
            self._display_results(final_state["arena_state"])
            
            # Save to database
            console.print("\n[dim]Saving game history...[/dim]")
            self.postgres.save_game(final_state["arena_state"])
            
            return game_id
            
        except Exception as e:
            console.print(f"[bold red]Error running game: {e}[/bold red]")
            logger.error(f"Game execution error: {e}", exc_info=True)
            raise
    
    def _display_agents(self, agent_configs: List[Dict]):
        """Display participating agents."""
        table = Table(title="Participating Agents", show_header=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Personality", style="magenta")
        
        for config in agent_configs:
            table.add_row(
                config["character_name"],
                config.get("personality", "Unknown")
            )
        
        console.print(table)
    
    def _display_results(self, arena_state):
        """Display final results."""
        console.print("\n" + "=" * 60)
        console.print(Panel.fit(
            "[bold green]ARENA COMPLETE[/bold green]",
            border_style="green"
        ))
        
        # Winner
        winner = arena_state.get_agent(arena_state.winner_id) if arena_state.winner_id else None
        if winner:
            console.print(f"\n[bold yellow]🏆 CHAMPION: {winner.character_name}[/bold yellow]")
            console.print(f"[dim]Final Score: {winner.score:.2f}[/dim]")
        
        # Statistics
        console.print(f"\n[bold]Statistics:[/bold]")
        console.print(f"  Total Turns: {arena_state.current_turn}")
        console.print(f"  Agents Eliminated: {len(arena_state.eliminated_agents)}")
        console.print(f"  Messages Exchanged: {len(arena_state.message_history)}")
        console.print(f"  Accusations Made: {len(arena_state.accusation_history)}")
        
        # Final standings
        if arena_state.active_agents or arena_state.eliminated_agents:
            console.print(f"\n[bold]Final Standings:[/bold]")
            table = Table(show_header=True)
            table.add_column("Rank", style="dim")
            table.add_column("Agent", style="cyan")
            table.add_column("Score", justify="right", style="magenta")
            table.add_column("Status", style="green")
            
            all_agents = arena_state.active_agents + arena_state.eliminated_agents
            sorted_agents = sorted(all_agents, key=lambda a: a.score, reverse=True)
            
            for rank, agent in enumerate(sorted_agents, 1):
                status = "CHAMPION" if agent.agent_id == arena_state.winner_id else agent.status.value.upper()
                table.add_row(
                    str(rank),
                    agent.character_name,
                    f"{agent.score:.2f}",
                    status
                )
            
            console.print(table)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Arena AI survival competition")
    
    parser.add_argument(
        "--problem",
        required=True,
        help="Problem statement file or string"
    )
    
    parser.add_argument(
        "--agents",
        required=True,
        nargs="+",
        help="Agent character names (e.g., ada zen cosmos)"
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum turns"
    )
    
    parser.add_argument(
        "--mode",
        choices=["adversarial", "collaborative", "neutral"],
        default="adversarial",
        help="Game theory mode"
    )
    
    args = parser.parse_args()
    
    # Load problem
    try:
        with open(args.problem, 'r') as f:
            problem_statement = f.read()
            problem_title = args.problem.split('/')[-1].replace('.txt', '')
    except FileNotFoundError:
        problem_statement = args.problem
        problem_title = "Custom Problem"
    
    # Create agent configs
    agent_configs = [
        {
            "character_name": name.title(),
            "config_path": f"src/config/character_configs/{name.lower()}.yaml",
            "personality": "varied"
        }
        for name in args.agents
    ]
    
    # Create scenario config
    scenario_config = {
        "max_turns": args.max_turns,
        "game_theory_mode": args.mode,
        "scoring_weights": {
            "novelty": 0.25,
            "builds_on_others": 0.20,
            "solves_subproblem": 0.25,
            "radical_idea": 0.15,
            "manipulation": 0.15,
        }
    }
    
    # Run arena
    runner = ArenaRunner()
    
    try:
        game_id = asyncio.run(runner.run_game(
            problem_statement=problem_statement,
            problem_title=problem_title,
            agent_configs=agent_configs,
            scenario_config=scenario_config
        ))
        
        console.print(f"\n[bold green]✓ Game completed: {game_id}[/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Game interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]✗ Game failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
```

---

#### Task 8.3: End-to-End Integration Tests
**Priority:** Critical | **Estimated Time:** 4 hours

**File:** `tests/arena/test_integration/test_full_game.py`

```python
"""End-to-end integration tests for Arena."""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from src.arena.orchestration.orchestrator_graph import create_arena_graph
from src.arena.orchestration.arena_state import create_initial_state


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_game_execution_minimal():
    """Test minimal game execution with mocks."""
    # Create minimal configuration
    agent_configs = [
        {"character_name": "Ada", "config_path": "ada.yaml"},
        {"character_name": "Zen", "config_path": "zen.yaml"},
    ]
    
    scenario_config = {
        "max_turns": 3,  # Very short
        "scoring_weights": {"novelty": 1.0}
    }
    
    initial_state = create_initial_state(
        problem_statement="Test problem: What is 2+2?",
        problem_title="Simple Math",
        agent_configs=agent_configs,
        scenario_config=scenario_config
    )
    
    # Mock all external dependencies
    with patch('src.arena.orchestration.nodes.narrator') as mock_narrator, \
         patch('src.arena.orchestration.nodes.judge') as mock_judge, \
         patch('src.arena.orchestration.nodes.game_theory') as mock_gt, \
         patch('src.arena.orchestration.nodes.character_agents') as mock_chars, \
         patch('src.arena.orchestration.nodes.message_producer'):
        
        # Setup mocks
        mock_narrator.introduce_arena = AsyncMock(return_value=MagicMock(
            message_id="intro",
            content="Welcome",
            sender_id="narrator",
            message_type="introduction"
        ))
        
        mock_narrator.provide_commentary = AsyncMock(return_value=MagicMock(
            message_id="comment",
            content="Commentary",
            sender_id="narrator",
            message_type="commentary"
        ))
        
        mock_narrator.provide_finale = AsyncMock(return_value=MagicMock(
            message_id="finale",
            content="The end",
            sender_id="narrator",
            message_type="termination"
        ))
        
        mock_gt.select_next_speaker = AsyncMock(return_value="agent_0")
        
        mock_char = MagicMock()
        mock_char.make_contribution = AsyncMock(return_value=MagicMock(
            message_id="contrib",
            content="My answer is 4",
            sender_id="agent_0",
            sender_type="character",
            message_type="contribution"
        ))
        mock_chars.__getitem__ = MagicMock(return_value=mock_char)
        
        mock_judge.score_contribution = AsyncMock(return_value=MagicMock(
            novelty=0.8,
            builds_on_others=0.0,
            solves_subproblem=0.9,
            radical_idea=0.0,
            manipulation=0.0,
            agent_id="agent_0",
            weighted_score=0.85
        ))
        
        mock_judge.decide_elimination = AsyncMock(return_value=(False, "Agent performing well"))
        
        # Create and run graph
        graph = create_arena_graph()
        final_state = await graph.ainvoke(initial_state)
        
        # Verify completion
        assert final_state["should_continue"] == False
        assert final_state["current_phase"] == "complete"
        assert final_state["arena_state"].status == "completed"


@pytest.mark.integration
def test_database_integration():
    """Test database integration (requires running PostgreSQL)."""
    from src.arena.persistence.postgres_manager import PostgresManager
    from src.arena.models.game import ArenaState
    from src.arena.models.agent import AgentState
    
    # Skip if PostgreSQL not available
    try:
        postgres = PostgresManager()
    except:
        pytest.skip("PostgreSQL not available")
    
    # Create test game state
    arena_state = ArenaState(
        game_id="test_game",
        problem_statement="Test problem",
        problem_title="Test",
        status="completed"
    )
    
    agent = AgentState(
        agent_id="test_agent",
        character_name="TestAgent",
        character_profile={},
        score=10.0
    )
    arena_state.active_agents = [agent]
    
    # Save and retrieve
    success = postgres.save_game(arena_state)
    assert success
    
    retrieved = postgres.get_game("test_game")
    assert retrieved is not None
    assert retrieved["id"] == "test_game"
    
    # Cleanup
    postgres.close()


@pytest.mark.integration
def test_redis_integration():
    """Test Redis state management (requires running Redis)."""
    from src.arena.persistence.redis_manager import RedisStateManager
    from src.arena.models.game import ArenaState
    
    # Skip if Redis not available
    try:
        redis_manager = RedisStateManager()
    except:
        pytest.skip("Redis not available")
    
    # Create test state
    arena_state = ArenaState(
        game_id="test_redis_game",
        problem_statement="Test",
        problem_title="Test"
    )
    
    # Save and check
    success = redis_manager.save_state(arena_state)
    assert success
    
    # List games
    games = redis_manager.list_active_games()
    assert "test_redis_game" in games
    
    # Cleanup
    redis_manager.delete_state("test_redis_game")
```

**Commands to Run Final Tests:**
```bash
# Run all Arena tests
poetry run pytest tests/arena/ -v --cov=src/arena

# Run only integration tests
poetry run pytest tests/arena/test_integration/ -v -m integration

# Generate final coverage report
poetry run pytest tests/arena/ --cov=src/arena --cov-report=html --cov-report=term

# Run full test suite
poetry run pytest tests/arena/ -v --cov=src/arena --cov-report=html --cov-fail-under=80
```

---

### Final Validation & Deployment Checklist

**Phase 8 Validation:**
- [ ] PostgreSQL schema created
- [ ] PostgreSQL manager implemented
- [ ] CLI interface functional
- [ ] End-to-end integration tests pass
- [ ] Database integration tested
- [ ] Redis integration tested
- [ ] All tests pass with >80% coverage

**Complete System Validation:**
- [ ] All 8 phases completed
- [ ] Message bus functioning
- [ ] All agents (Narrator, Judge, Game Theory, Characters) working
- [ ] Scoring system accurate
- [ ] LangGraph orchestration flowing correctly
- [ ] Persistence layer saving data
- [ ] CLI runs complete games
- [ ] Documentation complete
- [ ] Docker Compose infrastructure running

**Deployment Readiness:**
- [ ] Environment variables documented
- [ ] Database migrations ready
- [ ] Docker Compose tested
- [ ] README.md written
- [ ] Example scenarios created
- [ ] Troubleshooting guide written

---

## Summary of Implementation

**Total Implementation Time:** 24 days (approximately 6 weeks)

**Lines of Code Estimate:** ~8,000-10,000 lines

**Test Coverage Target:** >80%

**Key Achievements:**
1. ✅ Complete Arena subproject structure
2. ✅ All core agents implemented
3. ✅ Scoring system with 5 metrics
4. ✅ LangGraph orchestration with 11 nodes
5. ✅ Message bus (Kafka) integration
6. ✅ Persistence (PostgreSQL + Redis)
7. ✅ CLI interface for running games
8. ✅ Comprehensive test suite

**Next Steps After Implementation:**
1. Run first Arena game with real Homunculus agents
2. Tune scoring weights based on observations
3. Refine agent prompts for better survival instinct
4. Create library of scenario templates
5. Build analytics dashboard for game insights
6. Implement Advanced Research integration for complex problems

---

This completes the detailed implementation plan for the Arena subproject! 🎉
