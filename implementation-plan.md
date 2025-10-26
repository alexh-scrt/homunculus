# Implementation Plan for Character Agent Validation System

## Project Overview

**Goal**: Build a CLI-based chat system where users can interact with 8 distinct character agents. Each character remembers all interactions and uses this experience to inform future conversations.

**Success Criteria**:
- User can select a character and have a natural conversation
- Character's personality is consistent and recognizable
- Character remembers past interactions and references them
- Mood and hormones visibly affect responses
- Each character feels distinctly different from others

---

## Project Structure

```
character-agent-system/
├── pyproject.toml                 # Poetry project config
├── README.md                      # Setup and usage instructions
├── .env.example                   # Environment variables template
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py           # Global settings (Ollama, Redis, etc.)
│   │   └── character_configs/    # Character YAML definitions
│   │       ├── marcus_playful_male.yaml
│   │       ├── zoe_playful_female.yaml
│   │       ├── david_sarcastic_male.yaml
│   │       ├── rachel_sarcastic_female.yaml
│   │       ├── james_serious_male.yaml
│   │       ├── anita_serious_female.yaml
│   │       ├── tj_dumb_male.yaml
│   │       └── britt_dumb_female.yaml
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── character_state.py    # CharacterState dataclass
│   │   ├── agent_input.py        # AgentInput dataclass
│   │   └── experience.py         # Experience dataclass
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py         # BaseAgent abstract class
│   │   ├── personality_agent.py
│   │   ├── mood_agent.py
│   │   ├── neurochemical_agent.py
│   │   ├── goals_agent.py
│   │   ├── communication_style_agent.py
│   │   └── memory_agent.py
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── agent_orchestrator.py
│   │   ├── cognitive_module.py
│   │   ├── response_generator.py
│   │   └── state_updater.py
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── experience_module.py  # ChromaDB episodic memory
│   │   └── knowledge_graph.py    # Neo4j knowledge graph
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── ollama_client.py      # LangChain + Ollama wrapper
│   │
│   ├── character_agent.py        # Main CharacterAgent class
│   │
│   └── cli/
│       ├── __init__.py
│       ├── chat_interface.py     # Interactive CLI
│       └── debug_view.py         # Debug info display
│
├── tests/
│   ├── __init__.py
│   ├── test_agents/
│   ├── test_modules/
│   └── test_integration/
│
├── scripts/
│   ├── setup_databases.py        # Initialize ChromaDB, Neo4j
│   ├── load_character.py         # Load character from YAML
│   └── run_chat.py               # Main entry point
│
└── data/
    ├── chroma_db/                # ChromaDB storage
    └── logs/                     # Conversation logs
```

---

## Implementation Phases

### **Phase 1: Foundation Setup** (Day 1)

#### Tasks:

1. **Project Initialization**
   - Create project structure
   - Setup `pyproject.toml` with dependencies:
     ```toml
     [tool.poetry.dependencies]
     python = "^3.11"
     langchain = "^0.1.0"
     langchain-community = "^0.0.20"
     chromadb = "^0.4.22"
     neo4j = "^5.16.0"
     redis = "^5.0.1"
     pydantic = "^2.5.0"
     pyyaml = "^6.0.1"
     python-dotenv = "^1.0.0"
     rich = "^13.7.0"           # For beautiful CLI output
     typer = "^0.9.0"            # For CLI commands
     tavily-python = "^0.3.0"    # Web search (optional for Phase 1)
     ```

2. **Environment Configuration**
   - Create `.env` template:
     ```env
     # Ollama
     OLLAMA_BASE_URL=http://localhost:11434
     OLLAMA_MODEL=llama3.3:70b
     
     # ChromaDB
     CHROMA_PERSIST_DIRECTORY=./data/chroma_db
     
     # Neo4j
     NEO4J_URI=bolt://localhost:7687
     NEO4J_USER=neo4j
     NEO4J_PASSWORD=your_password
     
     # Redis
     REDIS_HOST=localhost
     REDIS_PORT=6379
     REDIS_DB=0
     
     # Tavily (optional for Phase 1)
     TAVILY_API_KEY=your_key_here
     ```

3. **Docker Compose Setup** (for databases)
   - Create `docker-compose.yml`:
     ```yaml
     version: '3.8'
     services:
       neo4j:
         image: neo4j:5.16
         ports:
           - "7474:7474"
           - "7687:7687"
         environment:
           NEO4J_AUTH: neo4j/your_password
         volumes:
           - neo4j_data:/data
       
       redis:
         image: redis:7-alpine
         ports:
           - "6379:6379"
         volumes:
           - redis_data:/data
     
     volumes:
       neo4j_data:
       redis_data:
     ```

4. **Core Data Classes**
   - Implement `src/core/character_state.py`
   - Implement `src/core/agent_input.py`
   - Implement `src/core/experience.py`

5. **Settings Management**
   - Implement `src/config/settings.py` using Pydantic

**Deliverables**:
- ✅ Project structure created
- ✅ Dependencies installed
- ✅ Databases running via Docker
- ✅ Core data structures defined
- ✅ Configuration system working

---

### **Phase 2: LLM Integration** (Day 1-2)

#### Tasks:

1. **Ollama Client Wrapper**
   - Implement `src/llm/ollama_client.py`
   - Wrap LangChain's Ollama integration
   - Add retry logic and error handling
   - Add streaming support (for future use)

2. **Test LLM Connection**
   - Create test script to verify Ollama is working
   - Test with simple prompt
   - Measure response time

**Implementation Details**:

```python
# src/llm/ollama_client.py
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from typing import Optional
import logging

class OllamaClient:
    """Wrapper for LangChain Ollama integration"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.3:70b",
        temperature: float = 0.7
    ):
        self.llm = Ollama(
            base_url=base_url,
            model=model,
            temperature=temperature
        )
        self.logger = logging.getLogger(__name__)
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text from prompt"""
        try:
            # Override temperature if provided
            if temperature is not None:
                llm = Ollama(
                    base_url=self.llm.base_url,
                    model=self.llm.model,
                    temperature=temperature
                )
            else:
                llm = self.llm
            
            response = llm.invoke(prompt)
            return response
        
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise
    
    def generate_with_template(
        self,
        template: str,
        variables: dict,
        temperature: Optional[float] = None
    ) -> str:
        """Generate using a template"""
        prompt_template = PromptTemplate.from_template(template)
        prompt = prompt_template.format(**variables)
        return self.generate(prompt, temperature)
```

**Deliverables**:
- ✅ Ollama client implemented
- ✅ LLM connection verified
- ✅ Basic generation working

---

### **Phase 3: Agent System** (Day 2-3)

#### Tasks:

1. **Base Agent**
   - Implement `src/agents/base_agent.py`
   - Define abstract interface
   - Add common LLM calling logic

2. **Core Agents Implementation**
   - Implement `PersonalityAgent`
   - Implement `MoodAgent`
   - Implement `NeurochemicalAgent`
   - Implement `GoalsAgent`
   - Implement `CommunicationStyleAgent`

3. **Agent Testing**
   - Unit test each agent in isolation
   - Verify prompt templates work
   - Check output format consistency

**Key Implementation Notes**:

For `NeurochemicalAgent`:
- This agent doesn't use LLM - pure calculation
- Implement decay formulas
- Implement hormone change calculations based on stimuli

For other agents:
- Focus on clear, concise prompts
- Temperature settings: 0.6-0.7 for most agents
- Max tokens: 150-250 for agent consultation

**Deliverables**:
- ✅ 5 core agents implemented
- ✅ Each agent has working `consult()` method
- ✅ Unit tests pass

---

### **Phase 4: Memory Systems** (Day 3-4)

#### Tasks:

1. **ChromaDB Experience Module**
   - Implement `src/memory/experience_module.py`
   - Setup ChromaDB client
   - Implement storage and retrieval
   - Test semantic search

2. **Neo4j Knowledge Graph**
   - Implement `src/memory/knowledge_graph.py`
   - Setup Neo4j driver
   - Implement entity/relationship storage
   - Test graph queries

3. **Memory Agent**
   - Implement `src/agents/memory_agent.py`
   - Integrate with ExperienceModule
   - Implement memory retrieval logic

**Implementation Details**:

```python
# Test ChromaDB setup
def test_chroma_setup():
    import chromadb
    
    client = chromadb.PersistentClient(path="./data/chroma_db")
    collection = client.get_or_create_collection("test")
    
    # Test storage
    collection.add(
        ids=["test_1"],
        documents=["This is a test memory"],
        metadatas=[{"type": "test"}]
    )
    
    # Test retrieval
    results = collection.query(
        query_texts=["test"],
        n_results=1
    )
    
    print(f"ChromaDB working: {len(results['ids'][0]) > 0}")

# Test Neo4j setup
def test_neo4j_setup():
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "your_password")
    )
    
    with driver.session() as session:
        result = session.run("RETURN 'Neo4j working!' as message")
        print(result.single()["message"])
    
    driver.close()
```

**Deliverables**:
- ✅ ChromaDB storing/retrieving experiences
- ✅ Neo4j storing/querying entities
- ✅ Memory agent retrieving relevant memories

---

### **Phase 5: Integration Modules** (Day 4-5)

#### Tasks:

1. **Agent Orchestrator**
   - Implement `src/modules/agent_orchestrator.py`
   - Sequential consultation of all agents
   - Collect all AgentInput objects

2. **Cognitive Module**
   - Implement `src/modules/cognitive_module.py`
   - Synthesize agent inputs using LLM
   - Parse structured output

3. **Response Generator**
   - Implement `src/modules/response_generator.py`
   - Convert intention to natural language
   - Apply communication style

4. **State Updater**
   - Implement `src/modules/state_updater.py`
   - Hormone decay logic
   - Mood recalculation
   - Memory creation
   - Knowledge extraction

**Integration Test**:

Create end-to-end test with mock character:

```python
def test_full_pipeline():
    # Initialize all components
    character = create_test_character()
    
    # User message
    user_msg = "Hey, how are you doing?"
    
    # Run through pipeline
    agent_inputs = orchestrator.consult_all_agents(context, user_msg)
    intention = cognitive_module.synthesize(agent_inputs, state, user_msg)
    response = response_generator.generate(intention, agent_inputs, state, user_msg)
    state = state_updater.update_after_response(state, user_msg, response, agent_inputs)
    
    # Verify
    assert len(response) > 0
    assert state.conversation_history[-1]['message'] == response
```

**Deliverables**:
- ✅ All modules implemented
- ✅ Integration test passes
- ✅ Pipeline runs end-to-end

---

### **Phase 6: Character Configuration** (Day 5)

#### Tasks:

1. **Character YAML Schemas**
   - Create detailed YAML for each of 8 characters
   - Validate schemas match our design specs
   - Store in `src/config/character_configs/`

2. **Character Loader**
   - Implement `scripts/load_character.py`
   - Parse YAML into character config dict
   - Validate required fields

**Example Character YAML**:

```yaml
# src/config/character_configs/marcus_playful_male.yaml
character_id: "char_001_marcus"
name: "Marcus Rivera"
archetype: "playful"

demographics:
  age: 29
  gender: "male"
  occupation: "Elementary School Teacher"
  education: "Bachelor's in Education"
  background: "Grew up in large family, middle child, class clown"

initial_agent_states:
  personality:
    big_five:
      openness: 0.8
      conscientiousness: 0.5
      extraversion: 0.85
      agreeableness: 0.75
      neuroticism: 0.3
    
    behavioral_traits:
      - trait: "spontaneous"
        intensity: 0.8
      - trait: "optimistic"
        intensity: 0.9
      - trait: "playful_teasing"
        intensity: 0.7
    
    core_values:
      - value: "joy"
        priority: 10
      - value: "connection"
        priority: 9
  
  specialty:
    domain: "child_education"
    expertise_level: 0.7
    subdomain_knowledge:
      - "child psychology"
      - "creative teaching methods"
  
  skills:
    intelligence:
      analytical: 0.6
      creative: 0.85
      practical: 0.65
    emotional_intelligence: 0.8
    physical_capability: 0.7
    problem_solving: 0.65
  
  mood_baseline:
    default_state: "happy"
    emotional_volatility: 0.4
    baseline_setpoint: 0.7
  
  communication_style:
    verbal_pattern: "verbose"
    social_comfort: "assertive"
    listening_preference: 0.4
    body_language: "expressive"
    quirks:
      - "Uses lots of emojis/emoticons in text"
      - "Makes pop culture references"
      - "Turns mundane topics into games"
      - "Self-deprecating humor"
  
  neurochemical_profile:
    baseline_sensitivities:
      dopamine: 1.3
      serotonin: 1.1
      oxytocin: 1.4
      endorphins: 1.2
      cortisol: 0.7
      adrenaline: 0.9
    
    baseline_levels:
      dopamine: 60.0
      serotonin: 55.0
      oxytocin: 58.0
      endorphins: 52.0
      cortisol: 40.0
      adrenaline: 48.0
    
    gender_modifiers:
      applies: true
      modifier_set: "male"

initial_goals:
  - goal_id: "goal_marcus_1"
    goal_type: "short_term"
    description: "Make this conversation fun and memorable"
    priority: 9
    progress: 0.0
  
  - goal_id: "goal_marcus_2"
    goal_type: "long_term"
    description: "Build genuine friendships"
    priority: 8
    progress: 0.2
  
  - goal_id: "goal_marcus_3"
    goal_type: "hidden"
    description: "Prove he's more than just 'the funny guy'"
    priority: 6
    progress: 0.1
```

**Deliverables**:
- ✅ 8 complete character YAML files
- ✅ Character loader implemented
- ✅ Validation logic working

---

### **Phase 7: Main Character Agent** (Day 6)

#### Tasks:

1. **CharacterAgent Class**
   - Implement `src/character_agent.py`
   - Integrate all components
   - Add initialization from config
   - Implement `chat()` method

2. **State Persistence**
   - Implement save/load character state
   - Use Redis for session state
   - Use filesystem for archival

**Key Implementation**:

```python
# src/character_agent.py

class CharacterAgent:
    def __init__(
        self,
        character_config: Dict[str, Any],
        llm_client: OllamaClient,
        chroma_client: chromadb.Client,
        neo4j_driver: neo4j.Driver
    ):
        # [Initialization code from design doc]
        pass
    
    def chat(
        self,
        user_message: str,
        context: Dict[str, Any] = None,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Main interaction method.
        Returns character response + optional debug info.
        """
        # [Implementation from design doc]
        pass
    
    def save_state(self, filepath: str):
        """Persist character state to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    def load_state(self, filepath: str):
        """Load character state from file"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.state = CharacterState.from_dict(data)
```

**Deliverables**:
- ✅ CharacterAgent fully implemented
- ✅ Can initialize from YAML config
- ✅ Chat method working
- ✅ State persistence working

---

### **Phase 8: CLI Interface** (Day 6-7)

#### Tasks:

1. **Interactive Chat CLI**
   - Implement `src/cli/chat_interface.py`
   - Use `rich` library for beautiful output
   - Character selection menu
   - Conversation loop
   - Commands: `/exit`, `/debug`, `/save`, `/load`, `/memory`

2. **Debug View**
   - Implement `src/cli/debug_view.py`
   - Display agent inputs
   - Show neurochemical levels
   - Show mood state
   - Show memory retrieval

**CLI Features**:

```python
# src/cli/chat_interface.py
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
import typer

app = typer.Typer()
console = Console()

def display_character_selection():
    """Show character selection menu"""
    table = Table(title="Select a Character")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Archetype", style="yellow")
    table.add_column("Age/Gender", style="magenta")
    
    characters = [
        ("1", "Marcus Rivera", "Playful", "29/M"),
        ("2", "Zoe Kim", "Playful", "26/F"),
        ("3", "David Okonkwo", "Sarcastic", "34/M"),
        ("4", "Rachel Stern", "Sarcastic", "31/F"),
        ("5", "Dr. James Morrison", "Serious", "52/M"),
        ("6", "Dr. Anita Patel", "Serious", "45/F"),
        ("7", "Tyler 'TJ' Johnson", "Dumb/Humorous", "24/M"),
        ("8", "Brittany 'Britt' Cooper", "Dumb/Humorous", "23/F"),
    ]
    
    for char in characters:
        table.add_row(*char)
    
    console.print(table)

def display_message(speaker: str, message: str, style: str = ""):
    """Display a conversation message"""
    panel = Panel(
        message,
        title=f"[bold]{speaker}[/bold]",
        border_style=style
    )
    console.print(panel)

def display_debug_info(debug_data: Dict[str, Any]):
    """Display debug information"""
    console.print("\n[bold yellow]═══ DEBUG INFO ═══[/bold yellow]")
    
    # Neurochemical levels
    neuro = debug_data.get('neurochemical_levels', {})
    table = Table(title="Neurochemical Levels")
    table.add_column("Hormone", style="cyan")
    table.add_column("Level", style="green")
    
    for hormone, level in neuro.items():
        color = "red" if level > 70 else "yellow" if level > 50 else "green"
        table.add_row(hormone.capitalize(), f"[{color}]{level:.1f}/100[/{color}]")
    
    console.print(table)
    
    # Mood
    mood = debug_data.get('mood', {})
    console.print(f"\n[bold]Current Mood:[/bold] {mood.get('current_state')} "
                  f"(intensity: {mood.get('intensity', 0):.2f})")
    
    # Agent inputs
    console.print("\n[bold]Agent Inputs:[/bold]")
    for agent_type, agent_data in debug_data.get('agent_inputs', {}).items():
        console.print(f"  [{agent_type}] {agent_data.get('content', '')[:100]}...")

@app.command()
def chat(character_id: int = None, debug: bool = False):
    """Start interactive chat with a character"""
    
    console.print("[bold green]Character Agent Chat System[/bold green]\n")
    
    # Character selection
    if character_id is None:
        display_character_selection()
        character_id = int(Prompt.ask("Select character", choices=[str(i) for i in range(1, 9)]))
    
    # Load character
    console.print(f"\n[cyan]Loading character {character_id}...[/cyan]")
    character = load_character_by_id(character_id)
    
    console.print(f"[green]Chatting with {character.character_name}[/green]")
    console.print("[dim]Commands: /exit, /debug, /save, /load, /memory[/dim]\n")
    
    # Conversation loop
    while True:
        user_input = Prompt.ask("[bold blue]You[/bold blue]")
        
        # Handle commands
        if user_input.startswith('/'):
            if user_input == '/exit':
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif user_input == '/debug':
                debug = not debug
                console.print(f"[yellow]Debug mode: {'ON' if debug else 'OFF'}[/yellow]")
                continue
            elif user_input == '/save':
                character.save_state(f"./data/saves/{character.character_id}.json")
                console.print("[green]State saved![/green]")
                continue
            elif user_input.startswith('/memory'):
                query = user_input.replace('/memory', '').strip()
                memories = character.recall_past_conversations(query or "recent experiences", 5)
                console.print(f"\n[yellow]Found {len(memories)} relevant memories:[/yellow]")
                for mem in memories:
                    console.print(f"  - {mem.description[:150]}...")
                continue
        
        # Get response
        result = character.chat(user_input, debug=debug)
        
        # Display response
        display_message(character.character_name, result['response'], style="green")
        
        # Display debug info if enabled
        if debug and 'debug_info' in result:
            display_debug_info(result['debug_info'])

if __name__ == "__main__":
    app()
```

**Deliverables**:
- ✅ Beautiful CLI interface
- ✅ Character selection working
- ✅ Conversation flow smooth
- ✅ Debug view functional
- ✅ Commands working

---

### **Phase 9: Testing & Validation** (Day 7-8)

#### Tasks:

1. **Validation Scenarios**
   - Test each of 8 characters
   - Run through validation scenarios from design doc:
     - Casual social interaction
     - Professional domain question
     - Personal/emotional topic
     - Stressful decision prompt
     - Repeat interaction (memory test)

2. **Consistency Testing**
   - Have 10-20 message conversations
   - Verify personality consistency
   - Check mood evolution
   - Confirm memory recall

3. **Cross-Character Comparison**
   - Chat with all 8 characters about same topic
   - Document behavioral differences
   - Verify archetypes are distinct

4. **Bug Fixes**
   - Fix any issues discovered
   - Tune prompts if needed
   - Adjust hormone sensitivity if too extreme

**Validation Checklist** (per character):

```markdown
## Character: [Name]

### Personality Consistency
- [ ] Big Five traits observable in responses
- [ ] Behavioral traits evident
- [ ] Core values reflected in decisions
- [ ] No contradictory behavior

### Mood Dynamics
- [ ] Mood changes appropriately with stimuli
- [ ] Energy levels affect response length/engagement
- [ ] Emotional states feel realistic
- [ ] Mood persists appropriately (not instant reset)

### Neurochemical System
- [ ] Hormones change based on interactions
- [ ] Decay happens over time
- [ ] Extreme levels affect behavior noticeably
- [ ] Different characters have different sensitivities

### Communication Style
- [ ] Verbal pattern matches config (concise/verbose/rambling)
- [ ] Quirks appear naturally
- [ ] Tone consistent with personality
- [ ] Style distinct from other characters

### Memory & Learning
- [ ] Remembers previous messages in conversation
- [ ] Recalls past interactions when relevant
- [ ] References shared history naturally
- [ ] Memory retrieval affects responses

### Goals
- [ ] Pursues stated goals subtly
- [ ] Hidden goals influence behavior
- [ ] Goal progress affects satisfaction/frustration

### Realism
- [ ] Feels like a specific person
- [ ] Surprises in believable ways
- [ ] Sometimes irrational but human
- [ ] Not generic chatbot responses
```

**Deliverables**:
- ✅ All 8 characters tested thoroughly
- ✅ Validation checklist completed for each
- ✅ Bugs fixed
- ✅ Documentation of findings

---

### **Phase 10: Documentation & Refinement** (Day 8)

#### Tasks:

1. **README Documentation**
   - Setup instructions
   - Usage guide
   - Architecture overview
   - Character descriptions

2. **Code Documentation**
   - Docstrings for all classes/methods
   - Type hints throughout
   - Inline comments for complex logic

3. **Configuration Guide**
   - How to create new characters
   - How to tune parameters
   - How to interpret debug output

4. **Demo Video/Screenshots**
   - Record conversation examples
   - Screenshot debug views
   - Show character differences

**README Structure**:

```markdown
# Character Agent System

AI-powered conversational agents with realistic personalities, memory, and emotional dynamics.

## Features

- 🎭 8 distinct character personalities (playful, sarcastic, serious, humorous)
- 🧠 Multi-agent architecture (personality, mood, goals, memory, etc.)
- 💊 Neurochemical simulation (dopamine, cortisol, oxytocin, etc.)
- 🧩 Episodic memory system (remembers all interactions)
- 📊 Knowledge graph (learns about world through conversations)
- 🎨 Beautiful CLI interface with debug mode

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Ollama with llama3.3:70b model

### Installation

1. Clone repository
2. Install dependencies: `poetry install`
3. Start databases: `docker-compose up -d`
4. Configure `.env` file
5. Run setup: `python scripts/setup_databases.py`

### Usage

```bash
# Start interactive chat
python scripts/run_chat.py

# Chat with specific character (with debug mode)
python scripts/run_chat.py --character-id 1 --debug
```

## Architecture

[High-level diagram and explanation]

## Characters

### Marcus Rivera (Playful Male)
29-year-old elementary school teacher...
[Description, traits, expected behavior]

[... 7 more characters ...]

## Creating New Characters

[Guide on YAML schema]

## Tuning Parameters

[Guide on adjusting personality, hormones, etc.]

## Development

[Contributing guide, testing, etc.]
```

**Deliverables**:
- ✅ Complete README
- ✅ All code documented
- ✅ Configuration guides written
- ✅ Demo materials ready

---

## File-by-File Implementation Order

For Claude Code, here's the exact order to implement files:

### Day 1: Foundation
1. `pyproject.toml` - Dependencies
2. `docker-compose.yml` - Database setup
3. `.env.example` - Configuration template
4. `src/config/settings.py` - Settings management
5. `src/core/agent_input.py` - AgentInput dataclass
6. `src/core/character_state.py` - CharacterState dataclass
7. `src/core/experience.py` - Experience dataclass
8. `src/llm/ollama_client.py` - LLM wrapper

### Day 2: Agents
9. `src/agents/base_agent.py` - Abstract base class
10. `src/agents/neurochemical_agent.py` - Hormone system (no LLM)
11. `src/agents/personality_agent.py` - Personality enforcement
12. `src/agents/mood_agent.py` - Mood tracking
13. `src/agents/communication_style_agent.py` - Style consistency
14. `src/agents/goals_agent.py` - Goal pursuit

### Day 3: Memory
15. `src/memory/experience_module.py` - ChromaDB integration
16. `src/memory/knowledge_graph.py` - Neo4j integration
17. `src/agents/memory_agent.py` - Memory retrieval agent

### Day 4: Integration
18. `src/modules/agent_orchestrator.py` - Agent coordination
19. `src/modules/cognitive_module.py` - Synthesis
20. `src/modules/response_generator.py` - Natural language output
21. `src/modules/state_updater.py` - State management + memory creation

### Day 5-6: Main System
22. `src/character_agent.py` - Main CharacterAgent class
23. `scripts/load_character.py` - YAML loader
24. `src/config/character_configs/*.yaml` - All 8 character configs

### Day 6-7: Interface
25. `src/cli/debug_view.py` - Debug display
26. `src/cli/chat_interface.py` - Interactive CLI
27. `scripts/run_chat.py` - Entry point
28. `scripts/setup_databases.py` - Database initialization

### Day 8: Polish
29. `README.md` - Documentation
30. `tests/*` - Test files
31. Bug fixes and refinements

---

## Testing Strategy

### Unit Tests
```python
# tests/test_agents/test_personality_agent.py
def test_personality_agent_extraversion():
    """High extraversion should produce more verbose responses"""
    pass

# tests/test_modules/test_neurochemical.py
def test_hormone_decay():
    """Hormones should decay toward baseline over time"""
    pass

# tests/test_memory/test_experience_module.py
def test_memory_storage_and_retrieval():
    """Experiences should be stored and retrieved semantically"""
    pass
```

### Integration Tests
```python
# tests/test_integration/test_full_conversation.py
def test_multi_turn_conversation():
    """Character should maintain consistency across multiple turns"""
    pass

def test_memory_recall():
    """Character should recall earlier conversation"""
    pass
```

### Manual Validation
- Conversation scripts for each character
- Checklist-based evaluation
- Cross-character comparison

---

## Success Metrics

### Technical Metrics
- ✅ All unit tests pass
- ✅ Integration tests pass
- ✅ No crashes during 50-message conversation
- ✅ Response time < 10 seconds per message
- ✅ Memory retrieval < 1 second

### Quality Metrics
- ✅ Character personality recognizable within 5 messages
- ✅ User can distinguish between all 8 characters blindly
- ✅ Mood changes observable and realistic
- ✅ Memory recall happens naturally
- ✅ No contradictory responses in 20-message conversation

---

## Risk Mitigation

### Risks & Mitigation:

1. **LLM Response Quality**
   - Risk: Prompts don't produce good agent inputs
   - Mitigation: Iteratively refine prompts, test with real conversations

2. **Performance**
   - Risk: Too slow with sequential agent calls
   - Mitigation: Start sequential, optimize later if needed

3. **Memory Overload**
   - Risk: Too many memories retrieved, slows down or confuses response
   - Mitigation: Limit to top 3-5 most relevant, tune similarity thresholds

4. **Personality Drift**
   - Risk: Character personality changes over time
   - Mitigation: Strong personality agent with high priority, periodic validation

5. **Database Setup Complexity**
   - Risk: Users struggle with Neo4j/ChromaDB setup
   - Mitigation: Docker Compose makes it one command, detailed docs

---

## Next Steps for Claude Code

**Immediate Action Items:**

1. **Create project structure** following the directory layout above
2. **Implement files in order** (1-28) as listed
3. **Test incrementally** after each phase
4. **Use validation checklist** to ensure quality
5. **Document as you go** (docstrings, comments)

**Key Principles:**
- ✅ Type everything (use type hints throughout)
- ✅ Log everything (use logging library extensively)
- ✅ Test early and often (don't wait until end)
- ✅ Keep it simple first (optimize later if needed)
- ✅ Follow the architecture exactly as designed
