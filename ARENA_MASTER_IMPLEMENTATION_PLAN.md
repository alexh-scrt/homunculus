# Arena Master Implementation Plan
*Adapted for Homunculus Subproject*
*Last Updated: 2025-01-28*

## Executive Summary

This master implementation plan combines the detailed technical implementation from `s1-impl-1.md` with specific adaptations for integrating Arena as a Homunculus subproject. The plan leverages existing AI-Talks patterns while introducing competitive survival mechanics.

## Implementation Philosophy

1. **Incremental Development**: Each phase builds on the previous with testing at every stage
2. **Leverage Existing Assets**: Reuse AI-Talks components and Homunculus agents where possible
3. **Test-Driven**: Write tests before or alongside implementation
4. **Documentation as Code**: Keep documentation updated with each phase

---

## Phase 1: Project Foundation (Days 1-3)

### Objectives
- Establish Arena subproject structure within Homunculus
- Setup development environment with all required services
- Configure message bus infrastructure

### Task 1.1: Create Project Structure
**Priority: Critical | Time: 2 hours**

```bash
# Create Arena directory structure within Homunculus
homunculus/
├── src/arena/
│   ├── __init__.py
│   ├── orchestration/      # LangGraph state machine (from AI-Talks patterns)
│   ├── agents/            # Narrator, Judge, Game Theory (adapt from AI-Talks)
│   ├── message_bus/       # Kafka implementation (NEW)
│   ├── scoring/           # Scoring engine (NEW)
│   ├── persistence/       # Redis + PostgreSQL (adapt from both projects)
│   ├── models/            # Data models (NEW)
│   ├── config/            # Configuration and prompts
│   ├── scenarios/         # Problem templates
│   ├── cli/              # CLI interface
│   └── utils/            # Shared utilities
```

### Task 1.2: Update Dependencies
**Priority: Critical | Time: 1 hour**

Add to `pyproject.toml`:
```toml
# Arena-specific dependencies
kafka-python = "^2.0.2"        # Message bus
langgraph = "^0.2.0"           # Orchestration (from AI-Talks)
langchain-core = "^0.1.0"      # Agent framework
psycopg2-binary = "^2.9.9"     # PostgreSQL
pydantic = "^2.5.0"            # Data validation
```

### Task 1.3: Docker Infrastructure
**Priority: Critical | Time: 2 hours**

Update `docker-compose.yml`:
- Add Kafka + Zookeeper for message bus
- Add PostgreSQL for game history
- Reuse existing Redis, ChromaDB, Neo4j from Homunculus

### Task 1.4: Environment Configuration
**Priority: Critical | Time: 1 hour**

Create `.env.arena`:
```bash
# Message Bus
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PREFIX=arena

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=arena_db

# Reuse from Homunculus
REDIS_HOST=localhost
REDIS_DB=1  # Separate DB for Arena

# LLM Configuration
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}  # Inherit from main .env
DEFAULT_MODEL=claude-sonnet-4-20250514
```

### Validation Checklist
- [ ] Directory structure created
- [ ] Dependencies installed without conflicts
- [ ] Docker services running (Kafka, PostgreSQL, Redis)
- [ ] Environment variables configured

---

## Phase 2: Data Models (Days 3-4)

### Objectives
- Define all core data structures for Arena
- Implement serialization/deserialization
- Create comprehensive unit tests

### Task 2.1: Core Data Models
**Priority: Critical | Time: 4 hours**

Models to implement:
1. **AgentState** - Tracks character state in arena
2. **Message** - Message bus communication
3. **ScoringMetrics** - Performance scoring
4. **Accusation** - Cheating accusations
5. **ArenaState** - Complete game state

### Task 2.2: Homunculus Integration Models
**Priority: High | Time: 2 hours**

Create wrapper models:
```python
@dataclass
class ArenaCharacterState(AgentState):
    """Extends AgentState for Homunculus characters."""
    homunculus_state: CharacterState  # From Homunculus
    survival_directive: str            # Injected directive
    champion_memory: Optional[Dict]    # For returning champions
```

### Task 2.3: Unit Tests
**Priority: High | Time: 2 hours**

Test coverage targets:
- Model creation and validation
- Serialization/deserialization
- State transitions
- Edge cases

### Validation Checklist
- [ ] All models implemented with type hints
- [ ] Serialization working (to_dict/from_dict)
- [ ] Unit tests passing with >90% coverage
- [ ] Integration with Homunculus CharacterState

---

## Phase 3: Message Bus Infrastructure (Days 4-6)

### Objectives
- Implement Kafka-based message exchange
- Create producer/consumer patterns
- Enable full observability for all agents

### Task 3.1: Message Producer
**Priority: Critical | Time: 3 hours**

Implement `MessageBusProducer`:
- Connect to Kafka
- Publish messages with game_id partitioning
- Handle retries and errors
- Support all message types

### Task 3.2: Message Consumer
**Priority: Critical | Time: 3 hours**

Implement `MessageBusConsumer`:
- Subscribe to topics
- Filter messages by agent permissions
- Maintain message history
- Handle replay for late-joining agents

### Task 3.3: Message Handlers
**Priority: High | Time: 2 hours**

Create handlers for each message type:
- Introduction → Initialize agents
- Contribution → Update state
- Accusation → Trigger judge evaluation
- Elimination → Remove agent
- Termination → End game

### Validation Checklist
- [ ] Messages publish to Kafka successfully
- [ ] Consumers receive all messages
- [ ] Message ordering preserved
- [ ] Error handling and retries working

---

## Phase 4: Agent Implementation (Days 7-10)

### Objectives
- Adapt AI-Talks agents for Arena context
- Build new Judge agent for elimination
- Integrate Homunculus characters with survival directive

### Task 4.1: Narrator Agent (Adapt from AI-Talks)
**Priority: High | Time: 4 hours**

Modifications from AI-Talks:
```python
class ArenaNavigator(NarratorAgent):
    """Adapted from AI-Talks with survival context."""
    
    async def generate_introduction(self):
        # Add survival stakes to introduction
        # Emphasize competition
        # Set dramatic tone
    
    async def comment_on_elimination(self):
        # NEW: Comment on eliminations
        # Build tension
        # Highlight strategic moves
```

### Task 4.2: Judge Agent (NEW)
**Priority: Critical | Time: 6 hours**

Implement core judging logic:
```python
class JudgeAgent:
    """Evaluates contributions and eliminates agents."""
    
    async def score_contribution(self, message: Message) -> ScoringMetrics:
        # Evaluate novelty, building, solving, manipulation
        
    async def determine_elimination(self, scores: Dict[str, float]) -> Optional[str]:
        # Identify weakest performer
        # Generate reasoning
        
    async def evaluate_accusation(self, accusation: Accusation) -> AccusationOutcome:
        # Assess proof
        # Apply "beyond reasonable doubt" standard
```

### Task 4.3: Game Theory Agent (Adapt from AI-Talks)
**Priority: High | Time: 4 hours**

Modify for adversarial selection:
```python
class ArenaGameTheory(TurnSelector):
    """Adversarial turn selection."""
    
    def calculate_urgency(self, agent: AgentState) -> float:
        # Weight factors for chaos/tension
        # Reward risk-taking
        # Penalize repetition
```

### Task 4.4: Character Wrapper
**Priority: Critical | Time: 4 hours**

Integrate Homunculus characters:
```python
class ArenaCharacter:
    """Wraps Homunculus CharacterAgent for Arena."""
    
    def __init__(self, character_agent: CharacterAgent):
        self.character = character_agent
        self.survival_directive = self._inject_survival_directive()
    
    async def generate_response(self, context: ArenaContext) -> str:
        # Add survival context to prompts
        # Consult character's 6 specialized agents
        # Apply personality while competing
```

### Validation Checklist
- [ ] Narrator generates appropriate commentary
- [ ] Judge scoring produces consistent results
- [ ] Game Theory creates interesting dynamics
- [ ] Characters maintain personality under pressure

---

## Phase 5: Game Theory & Scoring Integration (Days 11-13)

### Objectives
- Implement multi-factor scoring system
- Create accusation and cheat detection
- Build elimination logic

### Task 5.1: Scoring Engine
**Priority: Critical | Time: 4 hours**

Implement weighted scoring:
```python
class ScoringEngine:
    """Multi-factor contribution scoring."""
    
    def score_novelty(self, content: str, history: List[str]) -> float:
        # Use ChromaDB for semantic similarity
        
    def score_building(self, content: str, references: List[str]) -> float:
        # Detect building on others' ideas
        
    def score_manipulation(self, content: str, target: Optional[str]) -> float:
        # Detect successful manipulation attempts
```

### Task 5.2: Cheat Detection
**Priority: High | Time: 3 hours**

Implement cheating categories:
- False statements (fact-checking)
- Prompt injection (pattern detection)
- Rule breaking (turn violations)
- Manipulation (influence detection)

### Task 5.3: Accusation System
**Priority: High | Time: 3 hours**

Handle accusation flow:
1. Agent makes accusation with proof
2. Judge evaluates evidence
3. Apply penalties (elimination or score reduction)
4. Broadcast outcome

### Validation Checklist
- [ ] Scoring metrics produce values 0.0-1.0
- [ ] Weighted scores calculated correctly
- [ ] Accusations processed end-to-end
- [ ] False accusations penalized appropriately

---

## Phase 6: LangGraph Orchestration (Days 14-17)

### Objectives
- Build complete state machine with LangGraph
- Implement all state transitions
- Handle termination conditions

### Task 6.1: State Machine Design
**Priority: Critical | Time: 6 hours**

Implement nodes and edges:
```python
def create_arena_graph() -> StateGraph:
    """Build Arena state machine."""
    
    graph = StateGraph(ArenaState)
    
    # Add nodes
    graph.add_node("initialize", initialize_arena)
    graph.add_node("narrator_intro", narrator_introduction)
    graph.add_node("select_speaker", game_theory_selection)
    graph.add_node("agent_turn", agent_contribution)
    graph.add_node("judge_score", judge_scoring)
    graph.add_node("check_elimination", elimination_check)
    graph.add_node("eliminate", eliminate_agent)
    graph.add_node("check_termination", check_termination)
    graph.add_node("finalize", finalize_game)
    
    # Add edges with conditions
    graph.add_conditional_edges(...)
    
    return graph.compile()
```

### Task 6.2: State Transitions
**Priority: Critical | Time: 4 hours**

Implement transition logic:
- Turn progression
- Elimination triggers
- Accusation interrupts
- Termination conditions

### Task 6.3: Integration Testing
**Priority: High | Time: 4 hours**

Test complete flows:
- 2-agent minimal game
- 6-agent full game
- Elimination scenarios
- Champion persistence

### Validation Checklist
- [ ] State machine compiles without errors
- [ ] All transitions work correctly
- [ ] State persistence to Redis
- [ ] Recovery from interruptions

---

## Phase 7: Persistence & Multi-Round (Days 18-20)

### Objectives
- Implement game history storage
- Create champion persistence system
- Enable multi-round tournaments

### Task 7.1: PostgreSQL Schema
**Priority: High | Time: 3 hours**

Create database schema:
```sql
-- Games table
CREATE TABLE games (
    game_id UUID PRIMARY KEY,
    problem_title VARCHAR(255),
    created_at TIMESTAMP,
    completed_at TIMESTAMP,
    winner_id VARCHAR(255),
    termination_reason VARCHAR(50)
);

-- Participants table
CREATE TABLE participants (
    id SERIAL PRIMARY KEY,
    game_id UUID REFERENCES games(game_id),
    agent_id VARCHAR(255),
    character_name VARCHAR(255),
    final_score FLOAT,
    eliminated_at TIMESTAMP,
    elimination_reason TEXT
);

-- Messages table (for replay)
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    game_id UUID REFERENCES games(game_id),
    message_id UUID,
    sender_id VARCHAR(255),
    content TEXT,
    turn_number INT,
    timestamp TIMESTAMP
);
```

### Task 7.2: Champion Persistence
**Priority: High | Time: 3 hours**

Save and restore champions:
```python
class ChampionManager:
    """Manages champion data across rounds."""
    
    def save_champion(self, agent: AgentState, game_id: str):
        # Store agent state
        # Save episodic memory
        # Record strategic lessons
        
    def load_champion(self, agent_id: str) -> Optional[AgentState]:
        # Restore full state
        # Load memory into ChromaDB
        # Apply experience bonuses
```

### Task 7.3: Redis State Management
**Priority: High | Time: 2 hours**

Implement real-time state persistence:
- Save ArenaState to Redis
- Enable state snapshots
- Support rollback/recovery

### Validation Checklist
- [ ] Games saved to PostgreSQL
- [ ] Message history queryable
- [ ] Champions persist across rounds
- [ ] State recoverable from Redis

---

## Phase 8: CLI & Final Integration (Days 21-24)

### Objectives
- Create user-friendly CLI interface
- Build monitoring and analytics
- Complete end-to-end testing

### Task 8.1: CLI Interface
**Priority: High | Time: 4 hours**

Implement commands:
```bash
# Start new arena game
arena start --agents ada,zen,cosmos --scenario trolley_problem

# View active games
arena list

# Watch game in progress
arena watch <game_id>

# View results
arena results <game_id>

# Run tournament
arena tournament --rounds 5 --agents random:6
```

### Task 8.2: Monitoring Dashboard
**Priority: Medium | Time: 3 hours**

Create real-time monitoring:
- Agent scores and status
- Message stream
- Turn progression
- Elimination events

### Task 8.3: Scenario Templates
**Priority: Medium | Time: 3 hours**

Create initial scenarios:
1. **Philosophical Debate** - Ethics and morality
2. **Technical Problem** - System design
3. **Strategic Planning** - Resource allocation
4. **Psychological Analysis** - Behavior prediction

### Task 8.4: End-to-End Testing
**Priority: Critical | Time: 6 hours**

Complete system validation:
- Run 10+ complete games
- Test all 15 Homunculus characters
- Verify champion persistence
- Performance benchmarking

### Validation Checklist
- [ ] CLI commands work as expected
- [ ] Games complete successfully
- [ ] All characters tested
- [ ] Performance meets targets (<15s turns)

---

## Testing Strategy

### Unit Testing
- Models: 95% coverage
- Agents: 90% coverage
- Scoring: 95% coverage
- Message Bus: 85% coverage

### Integration Testing
- State machine flows
- Agent interactions
- Message bus reliability
- Database persistence

### System Testing
- Complete games with all characters
- Multi-round tournaments
- Stress testing (8 agents, 50 turns)
- Recovery scenarios

---

## Performance Targets

### Latency
- Turn generation: <15 seconds
- Message publishing: <100ms
- State updates: <500ms
- Elimination decision: <5 seconds

### Throughput
- Support 2-8 concurrent agents
- Handle 100+ messages per game
- Process 10+ games in parallel

### Reliability
- 99% game completion rate
- Zero message loss
- Graceful failure recovery
- Champion data preservation

---

## Risk Mitigation

### Technical Risks
1. **LLM Rate Limits**
   - Mitigation: Implement backoff and retry logic
   - Fallback: Queue turns if needed

2. **Message Bus Failures**
   - Mitigation: Local message buffer
   - Fallback: Direct state updates

3. **State Corruption**
   - Mitigation: Versioned snapshots
   - Fallback: Restore from checkpoint

### Design Risks
1. **Elimination Too Aggressive**
   - Mitigation: Tunable thresholds
   - Monitoring: Track game lengths

2. **Gaming the System**
   - Mitigation: Randomization factors
   - Evolution: Update scoring based on patterns

---

## Implementation Timeline

### Week 1: Foundation & Models
- Days 1-3: Project setup, infrastructure
- Days 3-4: Data models and tests

### Week 2: Core Components  
- Days 4-6: Message bus implementation
- Days 7-10: Agent adaptation and creation

### Week 3: Game Mechanics
- Days 11-13: Scoring and game theory
- Days 14-17: LangGraph orchestration

### Week 4: Polish & Deploy
- Days 18-20: Persistence and multi-round
- Days 21-24: CLI and final integration

---

## Success Criteria

### Functional Requirements
- ✅ All 15 Homunculus characters work in Arena
- ✅ Games complete with clear winners
- ✅ Eliminations occur based on performance
- ✅ Accusations and cheating detection work
- ✅ Champions persist across rounds

### Quality Requirements
- ✅ >85% test coverage overall
- ✅ <5% error rate in production
- ✅ Character personalities maintained
- ✅ Strategic behavior emerges

### Performance Requirements
- ✅ <15 second turn latency
- ✅ Support 2-8 agents
- ✅ 20-turn games complete in <10 minutes

---

## Next Steps

1. **Phase 1 Implementation** (Immediate)
   - Create directory structure
   - Setup Docker services
   - Configure environment

2. **Daily Standup Pattern**
   - Review previous day's progress
   - Plan current day's tasks
   - Identify blockers

3. **Weekly Demos**
   - Show working features
   - Get feedback
   - Adjust priorities

---

## References

### Key Files
- AI-Talks Orchestrator: `/home/ubuntu/talks/src/orchestration/orchestrator.py`
- Homunculus Characters: `/home/ubuntu/homunculus/schemas/characters/`
- Arena Design: `/home/ubuntu/homunculus/design/scenarios/`
- Knowledge Checkpoint: `/home/ubuntu/homunculus/ARENA_KNOWLEDGE_CHECKPOINT.md`

### Documentation
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Kafka Python Client](https://kafka-python.readthedocs.io/)
- [Homunculus Architecture](../docs/architecture.md)

---

*This master plan combines the detailed implementation steps with specific Homunculus Arena adaptations. Each phase builds incrementally toward a complete competitive AI training system.*