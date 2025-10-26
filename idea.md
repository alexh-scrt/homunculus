
# Creative Writing AI Pipeline - Architecture Summary

## Core Philosophy
**Guided Emergence**: A novel unfolds through the interaction of structured planning (Blueprint + Narrator) and chaotic realism (God Engine + Character Agency), with Game Theory modeling human psychology through neurochemical reward systems.

---

## System Components

### 1. **Blueprint Layer** (Novel Initialization)
Defines the starting conditions and soft constraints:

**Schema Elements:**
- **Story Metadata**: Genre, tone, setting, themes
- **Character Blueprints**: Initial agent states (personality, skills, goals, mood baseline), character arc suggestions (not mandates)
- **Plot Skeleton**: Inciting incident, 3-5 major plot points (soft targets), climax structure suggestion
- **Narrative Constraints**: Minimal (e.g., "time period is 1990s", "setting is small town")

**Output**: A JSON/YAML blueprint that seeds the system

---

### 2. **Chapter Planning Layer**
For each chapter, generate:
- Chapter objectives (which plot points to approach)
- Scene breakdown (locations, characters, scene purposes)
- Initial character states for the chapter
- **Climax proximity indicator** (affects God Engine event frequency)

---

### 3. **Character Agent System**
Each character is composed of **specialized agents**:

**Agent Types:**
1. **Personality Agent** (introvert/extrovert, cautious/bold, etc.)
2. **Specialty Agent** (profession knowledge: doctor, lawyer, engineer)
3. **Skill Level Agent** (intelligence, emotional intelligence, physical abilities)
4. **Mood Agent** (current emotional state - dynamic)
5. **Communication Style Agent** (talkative, reserved, aggressive, diplomatic)
6. **Goals Agent** (active objectives this character wants to achieve)
7. **Character Development Agent** (tracks growth, remembers key experiences)
8. **Neurochemical Module** (NEW) - manages hormone levels and decay

**Cognitive Module**: 
- Single LLM call that takes all agent inputs
- Synthesizes into coherent character intention
- No voting - it's contextual integration

---

### 4. **Neurochemical Reward System**

**Hormone Types:**
- **Dopamine** (achievement, novelty) - decays fast
- **Serotonin** (status, pride) - medium decay
- **Oxytocin** (connection, trust) - slow decay
- **Endorphins** (comfort, pleasure) - fast decay
- **Cortisol** (stress, threat) - slow decay, accumulates
- **Adrenaline** (acute danger) - very fast decay

**Personality-Based Hormone Profiles:**
- Each personality type has **baseline sensitivity** (e.g., extroverts get more dopamine from social interaction)
- **Stimulus response curves** differ by gender, personality, past trauma
- Example: Female character receiving compliments → dopamine spike, but faster decay than male equivalent
- **Hormone levels change dynamically** through the story based on events and interactions
- Traumatic events → cortisol accumulation → character becomes more risk-averse over time

---

### 5. **Game Theory Engine**

**Purpose**: Evaluate character action options using neurochemical payoffs

**Process**:
1. Character's Cognitive Module proposes intended action
2. Game Theory Engine calculates **expected neurochemical outcomes** for this action
3. Considers: character's personality weights, current hormone levels, other characters' likely reactions
4. Generates **payoff score** for the action
5. May suggest alternative actions if payoff is poor
6. Final action is chosen (not purely optimization - personality can override rational choice)

**Strategic Modeling**: 
- Start with **Sequential Bayesian Games** (characters have incomplete information about each other's states)
- Characters optimize for neurochemical rewards based on their personality-weighted preferences

---

### 6. **God Engine** (Chaos Generator)

**Event Categories & Frequencies:**
- **Micro Events** (30% per scene): Personal disruptions, minor environmental
- **Meso Events** (15% per scene): Work issues, local news, health
- **Macro Events** (5% per chapter): Major news, economic, societal
- **Black Swan** (1% per novel): Life-altering, paradigm shifts

**Dynamic Frequency Scaling:**
- **Climax proximity multiplier**: As chapters approach climax, event probability increases (1.5x to 2x)
- **Event persistence**: Each event has a lifespan (scenes, chapters, or novel-duration)

**Genre Awareness:**
- Same event types across genres, but **weight distributions differ**
- Romance: Higher weight on social/emotional micro events
- Thriller: Higher weight on danger/conflict macro events
- Literary: More introspective meso events

**Event Impact Scope:**
- **Local**: Affects one character
- **Contextual**: Affects all characters in a scene/location
- **Global**: Affects all characters (news, weather, societal events)

---

### 7. **Narrator Agent** (Dual Mode)

**Mode A: Directive** (Proactive)
- Introduces external events when needed to push toward plot points
- Works WITH God Engine (can request certain event types if story stalls)
- Manages pacing

**Mode B: Reactive** (Adaptive)
- Observes character interactions for emergent narrative threads
- Maintains **Thread Registry** to track unexpected story developments
- **Pivot decision**: When emergent threads are richer than planned plot, follows the emergence
- **Abandons soft constraints** when reality (God events + character chemistry) dictates

**Narrator's Output:**
- Narrative prose weaving character actions and events
- Scene state updates
- Emergent thread tracking
- Pacing/tension monitoring

---

## Execution Flow (Per Scene)

```
1. SCENE INITIALIZATION
   - Load scene plan from chapter
   - Load character states
   - Set scene context

2. GOD ENGINE CHECK
   - Roll for event (probability based on climax proximity)
   - If event occurs:
     * Narrator evaluates (amplify/integrate/minimize)
     * Game Theory updates belief states
     * Character agents receive event

3. CHARACTER TURN LOOP (Sequential)
   For each character present:
   a) All agents provide input (parallel info gathering, but sequential LLM calls)
   b) Cognitive Module synthesizes intention
   c) Game Theory Engine evaluates neurochemical payoffs
   d) Character executes action
   e) Other characters observe and update states

4. NARRATOR NARRATION
   - Generate narrative prose
   - Evaluate for emergent threads
   - Update tension metrics
   - Decide: continue, pivot, or intervene

5. STATE UPDATES
   - Character agents update (mood shifts, knowledge gained, hormone decay)
   - World state updates (persistent God events)
   - Thread registry updates

6. SCENE END → Next scene or chapter
```

---

## Simplified Data Flow

```
BLUEPRINT (static)
    ↓
CHAPTER PLAN (semi-static, regenerated per chapter)
    ↓
SCENE EXECUTION (highly dynamic)
    ↓
    ├─→ GOD ENGINE → Random events
    ├─→ CHARACTERS → Actions via agents + game theory
    ├─→ NARRATOR → Prose + emergent thread tracking
    ↓
CHARACTER STATE UPDATES (hormone decay, memory, mood)
    ↓
NEXT SCENE (with updated context)
```

---

## POC Scope - Minimal Viable Architecture

**What to build first:**

1. **Blueprint Schema** (JSON/YAML format)
   - 2-3 character templates
   - Simple plot skeleton (3 plot points)
   - One genre (literary fiction - most flexible)

2. **Character Agent System**
   - 5 core agents per character (personality, specialty, mood, goals, communication)
   - Basic cognitive module (concatenate inputs + LLM synthesis)
   - Simple neurochemical module (track 3 hormones: dopamine, cortisol, oxytocin)

3. **God Engine**
   - Micro + Meso events only (skip macro/black swan for POC)
   - Fixed probabilities (no climax scaling yet)
   - 10-15 event templates

4. **Game Theory Engine**
   - Basic payoff calculation (simple weighted sum of expected hormone changes)
   - 2-3 action options per character turn

5. **Narrator Agent**
   - Reactive mode only (no proactive intervention yet)
   - Basic prose generation
   - Simple emergent thread detection (flag unexpected character combinations)

6. **Execution Engine**
   - Generate 1 chapter (3-5 scenes)
   - Sequential character turns
   - Basic state management (in-memory or simple JSON files)

**What to defer for later:**
- Advanced game theory (Bayesian games, Nash equilibrium)
- Narrator intervention toolkit
- Complex hormone decay curves
- Vector database for semantic search
- Multiple blueprint templates
- Post-processing/refinement layer

---

## Success Metrics for POC

1. **Coherence**: Does the chapter read like a story?
2. **Character consistency**: Do characters act according to their traits?
3. **Emergent behavior**: Did anything unexpected but interesting happen?
4. **God Engine impact**: Did random events create realistic complications?
5. **Readable prose**: Is the narrator's output natural?

---

## Next Steps

1. **Define Blueprint Schema** (detailed JSON structure)
2. **Design Character State Schema** (agent structure + neurochemical levels)
3. **Create Event Template Library** (God Engine event formats)
4. **Sketch LLM prompt templates** for each component
5. **Choose implementation stack** (Python + OpenAI/Anthropic? State management approach?)
