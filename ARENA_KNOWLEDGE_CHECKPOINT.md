# Arena Project Knowledge Checkpoint
*Generated: 2025-01-28*

## Executive Summary

Arena is a competitive AI agent training system designed as a subproject within Homunculus. It creates a survival-based environment where character agents compete to solve complex problems while ensuring their own survival through valuable contributions. This project leverages proven technologies from the AI-Talks POC while introducing competitive dynamics and elimination mechanics.

## Project Context

### Arena Vision
- **Core Innovation**: Combines Darwinian selection pressure with multi-agent coordination to create emergent strategic behavior
- **Training Methodology**: Agents must simultaneously solve problems, navigate social dynamics, and ensure survival
- **Multi-Round Learning**: Champion agents persist across rounds, accumulating expertise

### Key Objectives
1. Train specialized AI agents through competitive pressure
2. Validate Homunculus character psychology in high-stakes scenarios
3. Create reproducible training framework using evolutionary selection
4. Enable champion agents to accumulate expertise across rounds

## Architecture Overview

### System Components

#### 1. Orchestrator (from AI-Talks)
- Central coordinator managing all components
- LangGraph state machine for flow control
- Enforces turn-taking and message bus access
- Manages round transitions and champion persistence

#### 2. Message Bus (NEW)
- Fully observable by all agents
- Write access restricted by turn allocation
- Contains complete conversation history
- All announcements visible to all participants

#### 3. Character Agents (from Homunculus)
- 15 distinct characters with psychological modeling
- Multi-agent architecture per character:
  - Neurochemical Agent
  - Personality Agent
  - Mood Agent
  - Communication Style Agent
  - Goals Agent
  - Memory Agent
- Survival directive injection required

#### 4. Narrator (from AI-Talks)
- Makes public announcements and commentary
- Summarizes key developments
- Comments on interaction quality
- Adapted from AI-Talks narrator implementation

#### 5. Game Theory Engine (from AI-Talks)
- Selects next speaker each turn
- Modified for adversarial selection:
  - Rewards creativity and novelty
  - Rewards successful manipulation
  - Creates tension and chaos
- Selection reasoning kept private

#### 6. Judge (NEW)
- Silent continuous scoring
- Public elimination with reasoning
- Cheating adjudication
- Scoring metrics:
  - Novelty of ideas
  - Building on others
  - Sub-problem solving
  - Radical approaches
  - Successful manipulation

## AI-Talks Implementation Analysis

### Key Components Analyzed

#### Orchestrator (`/home/ubuntu/talks/src/orchestration/orchestrator.py`)
- **Pattern**: Centralized orchestration via `MultiAgentDiscussionOrchestrator`
- **State Management**: `GroupDiscussionState` and `ParticipantState`
- **Features**:
  - Turn-based message flow
  - Strategic coordinator for quality evaluation
  - Progression controller for managing discussion flow
  - Redundancy control system
  - Quote enrichment system
  - Cognitive coda generation

#### Turn Selector (`/home/ubuntu/talks/src/game_theory/turn_selector.py`)
- **Urgency Calculation**: Multi-factor weighted system
  - Personality baseline (30%)
  - Time since last spoke (20%)
  - Was addressed flag (40%)
  - Confidence level (10%)
  - Engagement level (20%)
- **Fairness Mechanisms**: Prevents domination
- **Randomization**: 80% game theory + 20% random

#### Narrator Agent (`/home/ubuntu/talks/src/agents/narrator_agent.py`)
- **Multi-phase Introduction**: Welcome, topic, participants, transition
- **Dynamic Coordination**: Contextual interjections
- **Closing Sequence**: Summary and remarks

#### Payoff Calculator (`/home/ubuntu/talks/src/game_theory/payoff_calculator.py`)
- **Move Types**: DEEPEN, CHALLENGE, SUPPORT, QUESTION, SYNTHESIZE, CONCLUDE
- **Context-aware Scoring**: Considers group state and relationships
- **Strategic Recommendations**: Move type + target participant

#### Participant Agent (`/home/ubuntu/talks/src/agents/participant_agent.py`)
- **Tool Integration**: Web search and other capabilities
- **RAG Style Transfer**: Voice adaptation
- **Dynamic Prompting**: Personality-aware, relationship context
- **State Tracking**: Updates confidence, relationships, aspects

#### Strategic Coordinator (`/home/ubuntu/talks/src/game_theory/strategic_coordinator.py`)
- **Meta-level Evaluation**: Strategic alignment scoring
- **Originality Detection**: Prevents redundancy
- **Analytics Tracking**: Historical evaluation data

### Key Insights
1. No traditional message bus - uses centralized orchestrator
2. Rich state management and tracking
3. Game theory at multiple levels
4. Modular agent architecture
5. Context-aware prompting
6. Multiple quality control layers

## Homunculus Character System

### Character Architecture
- **Main Class**: `CharacterAgent` orchestrates all components
- **Multi-Agent System** per character:
  - 6 specialized agents working in concert
  - Weighted consultation system
  - Parallel processing for efficiency

### Available Characters (15 total)
1. Ada Lovelace - Analytical genius
2. Zen Master - Philosophical wisdom
3. Captain Cosmos - Space adventurer
4. Grumpy Wizard - Cantankerous magic user
5. Creative Artist - Artistic expression
6. Friendly Teacher - Educational focus
7. Tech Enthusiast - Technology focused
8. Plus 8 personality variants (playful, sarcastic, serious, dumb × male/female)

### Character Configuration Structure
```yaml
name: Ada Lovelace
archetype: analytical_genius
demographics:
  age: 28
  background: Mathematician and early computer scientist
personality:
  big_five: [openness, conscientiousness, extraversion, agreeableness, neuroticism]
  behavioral_traits: [analytical, visionary, mathematical, curious]
  core_values: [intellectual pursuit, mathematical precision, innovation]
```

## Arena-Specific Modifications

### Game Mechanics

#### Survival Dynamics
- **Self-Preservation**: Core directive for all agents
- **Collaboration**: Allowed if it serves survival
- **Competition**: Natural result of elimination pressure
- **Cheating**: Permitted and rewarded if undetected

#### Accusation System
- Any agent can accuse others
- Proof required for adjudication
- Beyond reasonable doubt standard
- 50% score penalty for false accusations

#### Elimination Process
1. Judge determines elimination
2. Public announcement with reasoning
3. Final words from eliminated agent
4. Learning signal for survivors
5. Agent removed from pool

#### Multi-Round Structure
- Champion system for winners
- Fresh agents each round (except champion)
- Knowledge transfer through champion only

## Technical Stack

### From AI-Talks
- Python 3.11+
- LangGraph for orchestration
- LangChain for agent framework
- Redis for state management
- ChromaDB for vector storage
- PostgreSQL for persistence

### New for Arena
- Kafka/RabbitMQ for message bus
- Additional LangGraph components
- Enhanced scoring systems
- Champion persistence layer

## Implementation Roadmap

### Phase 1: Infrastructure (Week 1)
- Project setup and directory structure
- Message bus implementation
- Data model definitions

### Phase 2: Agent Adaptation (Week 2)
- Port AI-Talks components
- Build Judge agent
- Integrate Homunculus characters

### Phase 3: Scoring & Accusations (Week 3)
- Implement scoring engine
- Build cheat detection
- Create accusation flow

### Phase 4: Orchestration (Week 4)
- LangGraph state machine
- State persistence
- Integration testing

### Phase 5: Multi-Round & Polish (Week 5)
- Champion persistence
- Game history storage
- Results analysis tools

### Phase 6: Scenarios & Config (Week 6)
- Scenario templates
- Configuration management
- CLI interface

### Phase 7: Testing (Week 7)
- Comprehensive testing
- Character validation
- Bug fixes

### Phase 8: Documentation (Week 8)
- Complete documentation
- Deployment preparation
- Release

## Key Design Decisions

### Architecture Choices
- **Hybrid Approach**: Combine AI-Talks orchestration with true message bus
- **Dual Persistence**: Redis (ephemeral) + PostgreSQL (durable)
- **Multi-Factor Scoring**: Adapted from AI-Talks with survival focus
- **Character Integration**: Wrapper pattern for Homunculus agents

### Technical Considerations
- **Scalability**: Message bus enables distributed processing
- **Observability**: Full message history and state tracking
- **Modularity**: Clean separation of concerns
- **Extensibility**: Easy to add new agents or scenarios

## Critical Implementation Notes

### From AI-Talks Experience
1. State machine patterns work well with LangGraph
2. Centralized orchestration simplifies coordination
3. Rich context passing improves response quality
4. Redundancy control prevents repetitive content
5. Strategic scoring provides valuable metrics

### Arena-Specific Challenges
1. Balancing survival pressure with problem-solving
2. Preventing gaming of the elimination system
3. Ensuring fair accusation mechanics
4. Managing champion advantages
5. Creating meaningful learning signals

## Configuration Templates

### Scenario Configuration
```yaml
problem:
  title: "Ethical AI Design"
  description: "Design an ethical framework for AI systems"
  difficulty: 4
  expected_turns: 20

scoring:
  weights:
    novelty: 0.25
    builds_on_others: 0.20
    solves_subproblem: 0.25
    radical_idea: 0.15
    manipulation: 0.15

game_theory:
  mode: adversarial
  chaos_factor: 0.3

judge:
  elimination_threshold: -10.0
  reasoning_verbosity: high
```

## Resource Requirements

### Infrastructure
- Docker with 16GB+ RAM
- Kafka + Zookeeper
- Redis
- PostgreSQL
- ChromaDB
- Neo4j (from Homunculus)

### External Services
- Anthropic Claude API (primary)
- OpenAI API (backup)
- LangSmith (optional observability)

## Success Metrics

### Functional
- All 15 characters work in Arena
- Games complete successfully
- Eliminations occur correctly
- Accusations work end-to-end
- Champion persistence functions

### Performance
- <15 second turn latency
- Support 2-8 agents
- 20-turn game in <10 minutes
- Message bus >100 msg/sec

## Next Steps

1. Confirm architecture decisions
2. Setup development environment
3. Begin Phase 1 implementation
4. Create initial test scenarios
5. Validate with 2-3 character subset

## References

### File Paths
- AI-Talks Orchestrator: `/home/ubuntu/talks/src/orchestration/orchestrator.py`
- AI-Talks Game Theory: `/home/ubuntu/talks/src/game_theory/`
- Homunculus Characters: `/home/ubuntu/homunculus/schemas/characters/`
- Homunculus Agent: `/home/ubuntu/homunculus/src/character_agent.py`

### Design Documents
- Arena Scenario: `/home/ubuntu/homunculus/design/scenarios/scenario-1.md`
- Executive Summary: `/home/ubuntu/homunculus/design/scenarios/s1-summary.md`
- High-Level Design: `/home/ubuntu/homunculus/design/scenarios/s1-design.md`

---

*This checkpoint captures the complete understanding of the Arena project as of 2025-01-28. It serves as a reference for future implementation work and can be updated as the project evolves.*