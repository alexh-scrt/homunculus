"""Agent orchestrator for coordinating all character agents."""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from ..core.agent_input import AgentInput
except ImportError:
    from core.agent_input import AgentInput

try:
    from ..core.character_state import CharacterState
except ImportError:
    from core.character_state import CharacterState

try:
    from ..agents.neurochemical_agent import NeurochemicalAgent
except ImportError:
    from agents.neurochemical_agent import NeurochemicalAgent

try:
    from ..agents.personality_agent import PersonalityAgent
except ImportError:
    from agents.personality_agent import PersonalityAgent

try:
    from ..agents.mood_agent import MoodAgent
except ImportError:
    from agents.mood_agent import MoodAgent

try:
    from ..agents.communication_style_agent import CommunicationStyleAgent
except ImportError:
    from agents.communication_style_agent import CommunicationStyleAgent

try:
    from ..agents.goals_agent import GoalsAgent
except ImportError:
    from agents.goals_agent import GoalsAgent

try:
    from ..agents.memory_agent import MemoryAgent
except ImportError:
    from agents.memory_agent import MemoryAgent


class AgentOrchestrator:
    """
    Orchestrates all character agents to produce cohesive responses.
    
    This class manages the multi-agent architecture by:
    1. Consulting all agents in parallel
    2. Collecting and prioritizing their inputs
    3. Resolving conflicts between agent recommendations
    4. Ensuring consistency across agent outputs
    """
    
    def __init__(
        self,
        character_id: str,
        llm_client: Any,
        character_config: Dict[str, Any]
    ):
        """Initialize the agent orchestrator."""
        self.character_id = character_id
        self.llm_client = llm_client
        self.character_config = character_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize all agents
        self.agents = self._initialize_agents()
        
        # Agent consultation weights (can be adjusted per character)
        self.agent_weights = {
            'neurochemical': 1.0,  # Always high priority - biological foundation
            'personality': 0.9,    # Core personality traits
            'mood': 0.95,         # Current emotional state
            'memory': 0.9,        # Past experiences and knowledge
            'goals': 0.8,         # Strategic considerations
            'communication_style': 0.85  # How to express the response
        }
        
        self.logger.info(f"AgentOrchestrator initialized for character {character_id}")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all character agents based on configuration."""
        agents = {}
        
        try:
            # Neurochemical agent (no LLM needed)
            agents['neurochemical'] = NeurochemicalAgent(
                agent_id=f"{self.character_id}_neuro",
                character_id=self.character_id,
                llm_client=self.llm_client,
                neurochemical_config=self.character_config.get('neurochemical', {})
            )
            
            # Personality agent
            agents['personality'] = PersonalityAgent(
                agent_id=f"{self.character_id}_personality",
                character_id=self.character_id,
                llm_client=self.llm_client,
                personality_config=self.character_config.get('personality', {})
            )
            
            # Mood agent
            agents['mood'] = MoodAgent(
                agent_id=f"{self.character_id}_mood",
                character_id=self.character_id,
                llm_client=self.llm_client,
                mood_config=self.character_config.get('mood_baseline', {})
            )
            
            # Communication style agent
            agents['communication_style'] = CommunicationStyleAgent(
                agent_id=f"{self.character_id}_style",
                character_id=self.character_id,
                llm_client=self.llm_client,
                style_config=self.character_config.get('communication_style', {})
            )
            
            # Goals agent
            agents['goals'] = GoalsAgent(
                agent_id=f"{self.character_id}_goals",
                character_id=self.character_id,
                llm_client=self.llm_client,
                initial_goals=self.character_config.get('initial_goals', [])
            )
            
            # Memory agent
            agents['memory'] = MemoryAgent(
                agent_id=f"{self.character_id}_memory",
                character_id=self.character_id,
                llm_client=self.llm_client
            )
            
            self.logger.info(f"Initialized {len(agents)} agents")
            return agents
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def consult_all_agents(
        self,
        context: Dict[str, Any],
        character_state: CharacterState,
        user_message: str
    ) -> Dict[str, AgentInput]:
        """
        Consult all agents in parallel and return their inputs.
        
        Args:
            context: Conversation context
            character_state: Current character state
            user_message: The user's message
            
        Returns:
            Dictionary mapping agent types to their AgentInput responses
        """
        # Create consultation tasks for all agents
        consultation_tasks = {}
        
        for agent_type, agent in self.agents.items():
            consultation_tasks[agent_type] = asyncio.create_task(
                self._safe_agent_consult(agent, context, character_state, user_message),
                name=f"consult_{agent_type}"
            )
        
        # Wait for all consultations to complete
        try:
            agent_inputs = {}
            completed_tasks = await asyncio.gather(*consultation_tasks.values(), return_exceptions=True)
            
            for agent_type, result in zip(consultation_tasks.keys(), completed_tasks):
                if isinstance(result, Exception):
                    self.logger.error(f"Agent {agent_type} consultation failed: {result}")
                    # Create a fallback input
                    agent_inputs[agent_type] = self._create_fallback_input(agent_type, str(result))
                else:
                    agent_inputs[agent_type] = result
            
            self.logger.debug(f"Consulted {len(agent_inputs)} agents")
            return agent_inputs
            
        except Exception as e:
            self.logger.error(f"Error during agent consultation: {e}")
            # Return empty inputs if everything fails
            return {agent_type: self._create_fallback_input(agent_type, str(e)) 
                   for agent_type in self.agents.keys()}
    
    async def _safe_agent_consult(
        self,
        agent: Any,
        context: Dict[str, Any],
        character_state: CharacterState,
        user_message: str
    ) -> AgentInput:
        """Safely consult an agent with error handling."""
        try:
            # Check if agent consult method is async
            if hasattr(agent, 'consult') and asyncio.iscoroutinefunction(agent.consult):
                return await agent.consult(context, character_state, user_message)
            else:
                return agent.consult(context, character_state, user_message)
        except Exception as e:
            self.logger.warning(f"Agent {agent.agent_type} consultation failed: {e}")
            return self._create_fallback_input(agent.agent_type, str(e))
    
    def _create_fallback_input(self, agent_type: str, error_message: str) -> AgentInput:
        """Create a fallback AgentInput when an agent fails."""
        return AgentInput(
            agent_type=agent_type,
            content=f"Agent {agent_type} unavailable: {error_message}",
            confidence=0.1,
            priority=0.1,
            emotional_tone="neutral",
            metadata={"error": error_message, "fallback": True}
        )
    
    def prioritize_agent_inputs(
        self,
        agent_inputs: Dict[str, AgentInput]
    ) -> List[AgentInput]:
        """
        Prioritize agent inputs based on confidence, priority, and weights.
        
        Args:
            agent_inputs: Dictionary of agent inputs
            
        Returns:
            List of AgentInputs sorted by effective priority
        """
        weighted_inputs = []
        
        for agent_type, agent_input in agent_inputs.items():
            # Calculate effective priority considering agent weights
            agent_weight = self.agent_weights.get(agent_type, 0.5)
            effective_priority = (
                agent_input.priority * 0.4 +
                agent_input.confidence * 0.3 +
                agent_weight * 0.3
            )
            
            weighted_inputs.append({
                'agent_input': agent_input,
                'effective_priority': effective_priority,
                'agent_type': agent_type
            })
        
        # Sort by effective priority (highest first)
        weighted_inputs.sort(key=lambda x: x['effective_priority'], reverse=True)
        
        # Return sorted AgentInputs
        prioritized = [item['agent_input'] for item in weighted_inputs]
        
        self.logger.debug(f"Prioritized {len(prioritized)} agent inputs")
        return prioritized
    
    def detect_agent_conflicts(
        self,
        agent_inputs: Dict[str, AgentInput]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between agent recommendations.
        
        Args:
            agent_inputs: Dictionary of agent inputs
            
        Returns:
            List of conflict descriptions
        """
        conflicts = []
        
        # Check for emotional tone conflicts
        emotional_tones = {}
        for agent_type, agent_input in agent_inputs.items():
            tone = agent_input.emotional_tone
            if tone not in emotional_tones:
                emotional_tones[tone] = []
            emotional_tones[tone].append(agent_type)
        
        if len(emotional_tones) > 2:  # More than 2 different emotional tones
            conflicts.append({
                'type': 'emotional_tone_conflict',
                'description': f"Multiple emotional tones: {list(emotional_tones.keys())}",
                'agents': emotional_tones
            })
        
        # Check for confidence conflicts (high confidence + low confidence on same priority)
        high_conf_agents = []
        low_conf_agents = []
        
        for agent_type, agent_input in agent_inputs.items():
            if agent_input.confidence > 0.8 and agent_input.priority > 0.7:
                high_conf_agents.append(agent_type)
            elif agent_input.confidence < 0.3 and agent_input.priority > 0.7:
                low_conf_agents.append(agent_type)
        
        if high_conf_agents and low_conf_agents:
            conflicts.append({
                'type': 'confidence_conflict',
                'description': "High confidence and low confidence agents both have high priority",
                'high_confidence': high_conf_agents,
                'low_confidence': low_conf_agents
            })
        
        # Check for specific agent conflicts
        neurochemical = agent_inputs.get('neurochemical')
        mood = agent_inputs.get('mood')
        
        if neurochemical and mood:
            neuro_tone = neurochemical.emotional_tone
            mood_tone = mood.emotional_tone
            
            # Check if neurochemical and mood suggest opposite emotional states
            opposing_pairs = [
                ('happy', 'sad'), ('excited', 'tired'), ('calm', 'anxious'),
                ('confident', 'insecure'), ('energetic', 'lethargic')
            ]
            
            for tone1, tone2 in opposing_pairs:
                if (neuro_tone == tone1 and mood_tone == tone2) or \
                   (neuro_tone == tone2 and mood_tone == tone1):
                    conflicts.append({
                        'type': 'neurochemical_mood_conflict',
                        'description': f"Neurochemical suggests {neuro_tone}, mood suggests {mood_tone}",
                        'neurochemical_tone': neuro_tone,
                        'mood_tone': mood_tone
                    })
        
        if conflicts:
            self.logger.warning(f"Detected {len(conflicts)} agent conflicts")
        
        return conflicts
    
    def resolve_conflicts(
        self,
        agent_inputs: Dict[str, AgentInput],
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Resolve conflicts between agent inputs.
        
        Args:
            agent_inputs: Dictionary of agent inputs
            conflicts: List of detected conflicts
            
        Returns:
            Resolution strategy and adjusted weights
        """
        resolution_strategy = {
            'method': 'weighted_average',
            'adjustments': {},
            'conflict_count': len(conflicts)
        }
        
        if not conflicts:
            return resolution_strategy
        
        for conflict in conflicts:
            conflict_type = conflict['type']
            
            if conflict_type == 'emotional_tone_conflict':
                # Use neurochemical agent as tie-breaker for emotional tone
                if 'neurochemical' in agent_inputs:
                    resolution_strategy['method'] = 'neurochemical_priority'
                    resolution_strategy['adjustments']['neurochemical'] = 1.2
                
            elif conflict_type == 'confidence_conflict':
                # Boost high-confidence agents, reduce low-confidence ones
                for agent_type in conflict.get('high_confidence', []):
                    resolution_strategy['adjustments'][agent_type] = 1.3
                for agent_type in conflict.get('low_confidence', []):
                    resolution_strategy['adjustments'][agent_type] = 0.7
                    
            elif conflict_type == 'neurochemical_mood_conflict':
                # Trust neurochemical state over mood interpretation
                resolution_strategy['adjustments']['neurochemical'] = 1.4
                resolution_strategy['adjustments']['mood'] = 0.8
        
        self.logger.debug(f"Resolved {len(conflicts)} conflicts using {resolution_strategy['method']}")
        return resolution_strategy
    
    def synthesize_agent_guidance(
        self,
        prioritized_inputs: List[AgentInput],
        resolution_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize all agent inputs into unified guidance.
        
        Args:
            prioritized_inputs: Prioritized list of agent inputs
            resolution_strategy: Conflict resolution strategy
            
        Returns:
            Synthesized guidance for response generation
        """
        synthesis = {
            'primary_guidance': prioritized_inputs[0] if prioritized_inputs else None,
            'supporting_guidance': prioritized_inputs[1:3] if len(prioritized_inputs) > 1 else [],
            'emotional_tone': self._determine_overall_emotional_tone(prioritized_inputs),
            'confidence_level': self._calculate_overall_confidence(prioritized_inputs),
            'key_considerations': self._extract_key_considerations(prioritized_inputs),
            'response_modifiers': self._collect_response_modifiers(prioritized_inputs),
            'agent_metadata': {inp.agent_type: inp.metadata for inp in prioritized_inputs},
            'resolution_applied': resolution_strategy
        }
        
        self.logger.debug("Synthesized agent guidance for response generation")
        return synthesis
    
    def _determine_overall_emotional_tone(self, inputs: List[AgentInput]) -> str:
        """Determine the overall emotional tone from agent inputs."""
        if not inputs:
            return "neutral"
        
        # Weighted voting based on priority and confidence
        tone_scores = {}
        
        for inp in inputs:
            tone = inp.emotional_tone
            weight = inp.priority * inp.confidence
            tone_scores[tone] = tone_scores.get(tone, 0) + weight
        
        # Return the highest-scoring tone
        return max(tone_scores, key=tone_scores.get) if tone_scores else "neutral"
    
    def _calculate_overall_confidence(self, inputs: List[AgentInput]) -> float:
        """Calculate overall confidence from agent inputs."""
        if not inputs:
            return 0.5
        
        # Weighted average of confidence levels
        total_weight = 0
        weighted_confidence = 0
        
        for inp in inputs:
            weight = inp.priority
            weighted_confidence += inp.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.5
    
    def _extract_key_considerations(self, inputs: List[AgentInput]) -> List[str]:
        """Extract key considerations from agent inputs."""
        considerations = []
        
        for inp in inputs:
            if inp.confidence > 0.6 and inp.priority > 0.6:
                # Extract key points from agent content
                content = inp.content[:100] + "..." if len(inp.content) > 100 else inp.content
                considerations.append(f"[{inp.agent_type}] {content}")
        
        return considerations[:5]  # Limit to top 5 considerations
    
    def _collect_response_modifiers(self, inputs: List[AgentInput]) -> Dict[str, Any]:
        """Collect response modifiers from agent metadata."""
        modifiers = {}
        
        for inp in inputs:
            if inp.metadata:
                # Collect specific modifier types
                if 'response_modifiers' in inp.metadata:
                    modifiers.update(inp.metadata['response_modifiers'])
                
                # Collect style information
                if inp.agent_type == 'communication_style' and 'style_config' in inp.metadata:
                    modifiers['communication_style'] = inp.metadata['style_config']
                
                # Collect mood information
                if inp.agent_type == 'mood' and 'mood_state' in inp.metadata:
                    modifiers['mood_state'] = inp.metadata['mood_state']
        
        return modifiers
    
    async def orchestrate_response(
        self,
        context: Dict[str, Any],
        character_state: CharacterState,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Complete orchestration process for response generation.
        
        This is the main entry point that coordinates all agents.
        
        Args:
            context: Conversation context
            character_state: Current character state
            user_message: User's message
            
        Returns:
            Complete orchestration result for response generation
        """
        try:
            # Step 1: Consult all agents
            agent_inputs = await self.consult_all_agents(context, character_state, user_message)
            
            # Step 2: Detect conflicts
            conflicts = self.detect_agent_conflicts(agent_inputs)
            
            # Step 3: Resolve conflicts
            resolution_strategy = self.resolve_conflicts(agent_inputs, conflicts)
            
            # Step 4: Prioritize inputs
            prioritized_inputs = self.prioritize_agent_inputs(agent_inputs)
            
            # Step 5: Synthesize guidance
            synthesis = self.synthesize_agent_guidance(prioritized_inputs, resolution_strategy)
            
            # Package complete orchestration result
            orchestration_result = {
                'agent_inputs': agent_inputs,
                'conflicts_detected': conflicts,
                'resolution_strategy': resolution_strategy,
                'prioritized_inputs': prioritized_inputs,
                'synthesis': synthesis,
                'orchestration_metadata': {
                    'character_id': self.character_id,
                    'agent_count': len(agent_inputs),
                    'conflict_count': len(conflicts),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"Orchestration completed for character {self.character_id}")
            return orchestration_result
            
        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            # Return minimal fallback result
            return {
                'agent_inputs': {},
                'conflicts_detected': [],
                'resolution_strategy': {'method': 'fallback'},
                'prioritized_inputs': [],
                'synthesis': {
                    'emotional_tone': 'neutral',
                    'confidence_level': 0.3,
                    'key_considerations': ['Orchestration system error'],
                    'response_modifiers': {}
                },
                'orchestration_metadata': {
                    'character_id': self.character_id,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about the agents."""
        return {
            'character_id': self.character_id,
            'agent_count': len(self.agents),
            'agent_types': list(self.agents.keys()),
            'agent_weights': self.agent_weights,
            'initialization_time': datetime.now().isoformat()
        }
    
    def close(self):
        """Clean up agent resources."""
        try:
            # Close memory agent specifically as it has database connections
            if 'memory' in self.agents:
                self.agents['memory'].close()
            
            self.logger.info(f"AgentOrchestrator closed for character {self.character_id}")
            
        except Exception as e:
            self.logger.error(f"Error closing AgentOrchestrator: {e}")