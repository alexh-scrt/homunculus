"""Neurochemical agent for hormone level management and behavior influence."""

from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from .base_agent import BaseAgent
except ImportError:
    from agents.base_agent import BaseAgent

try:
    from ..core.agent_input import AgentInput
except ImportError:
    from core.agent_input import AgentInput

try:
    from ..core.character_state import CharacterState
except ImportError:
    from core.character_state import CharacterState


class NeurochemicalAgent(BaseAgent):
    """
    Manages hormone levels and calculates how they influence behavior.
    
    This is the 'biological' layer of the character. Unlike other agents,
    this one doesn't use LLM - it performs pure calculation based on
    neurochemical state and provides quantified biological influences.
    """
    
    def __init__(
        self,
        agent_id: str,
        character_id: str,
        llm_client: Any,  # Not used by this agent
        neurochemical_config: Dict[str, Any]
    ):
        """Initialize neurochemical agent."""
        super().__init__(agent_id, "neurochemical", character_id, llm_client)
        
        self.config = neurochemical_config
        
        # Web search not needed for biological calculations
        self.web_search_enabled = False
        
        # Decay rates per hormone (per time step)
        self.decay_rates = {
            'dopamine': 0.15,      # Fast decay - rewards fade quickly
            'serotonin': 0.08,     # Medium decay - status feelings persist
            'oxytocin': 0.05,      # Slow decay - bonds are lasting
            'endorphins': 0.20,    # Very fast decay - pleasure is fleeting
            'cortisol': 0.03,      # Slow decay - stress accumulates and lingers
            'adrenaline': 0.25     # Very fast decay - acute stress response
        }
        
        # Baseline levels from character configuration
        self.baseline_levels = self.config.get('baseline_levels', {
            'dopamine': 50.0,
            'serotonin': 50.0,
            'oxytocin': 50.0,
            'endorphins': 50.0,
            'cortisol': 50.0,
            'adrenaline': 50.0
        })
        
        # Sensitivity profiles from character configuration
        self.baseline_sensitivities = self.config.get('baseline_sensitivities', {
            'dopamine': 1.0,
            'serotonin': 1.0,
            'oxytocin': 1.0,
            'endorphins': 1.0,
            'cortisol': 1.0,
            'adrenaline': 1.0
        })
        
        # Gender modifiers if applicable
        self.gender_modifiers = self._get_gender_modifiers()
    
    def consult(
        self,
        context: Dict[str, Any],
        character_state: CharacterState,
        user_message: str
    ) -> AgentInput:
        """
        Provide neurochemical analysis and behavioral guidance.
        
        This agent doesn't use LLM - it calculates biological state directly
        and provides quantified hormone levels that affect other agents.
        """
        levels = character_state.neurochemical_levels
        
        # Calculate drive states based on current levels and sensitivities
        drive_analysis = self._calculate_drive_states(levels)
        
        # Generate behavioral guidance based on hormone patterns
        behavioral_guidance = self._generate_behavioral_guidance(levels, drive_analysis)
        
        # Calculate overall energy/motivation level
        energy_level = self._calculate_energy_level(levels)
        
        # Determine dominant emotional influence
        dominant_influence = self._get_dominant_influence(levels)
        
        # Create comprehensive neurochemical analysis
        analysis = self._create_analysis_text(
            levels, drive_analysis, energy_level, dominant_influence
        )
        
        # High confidence since this is quantified data
        confidence = 1.0
        
        # Priority based on how extreme the hormone levels are
        priority = self._calculate_priority(levels)
        
        # Emotional tone based on hormone pattern
        emotional_tone = self._map_hormones_to_emotion(levels)
        
        # Package metadata for other agents to use
        metadata = {
            'hormone_levels': levels.copy(),
            'drive_states': drive_analysis,
            'energy_level': energy_level,
            'dominant_influence': dominant_influence,
            'behavioral_recommendations': self._get_behavioral_recommendations(levels)
        }
        
        return self._create_agent_input(
            content=analysis,
            confidence=confidence,
            priority=priority,
            emotional_tone=emotional_tone,
            metadata=metadata
        )
    
    def get_prompt_template(self) -> str:
        """Not used - this agent doesn't use LLM."""
        return ""
    
    def _calculate_drive_states(self, levels: Dict[str, float]) -> Dict[str, float]:
        """Calculate psychological drive states from hormone levels."""
        sensitivities = self.baseline_sensitivities
        
        return {
            'reward_seeking': (levels['dopamine'] * sensitivities['dopamine']) / 100,
            'status_seeking': (levels['serotonin'] * sensitivities['serotonin']) / 100,
            'connection_desire': (levels['oxytocin'] * sensitivities['oxytocin']) / 100,
            'pleasure_seeking': (levels['endorphins'] * sensitivities['endorphins']) / 100,
            'stress_level': (levels['cortisol'] * sensitivities['cortisol']) / 100,
            'arousal_level': (levels['adrenaline'] * sensitivities['adrenaline']) / 100
        }
    
    def _calculate_energy_level(self, levels: Dict[str, float]) -> float:
        """Calculate overall energy/motivation level (0-100)."""
        # Energy comes from dopamine and serotonin, reduced by cortisol
        energy_sources = (levels['dopamine'] + levels['serotonin']) / 2
        energy_drains = levels['cortisol'] * 0.5
        
        # Adrenaline can boost energy temporarily but also drain it
        adrenaline_effect = levels['adrenaline'] * 0.3
        
        raw_energy = energy_sources - energy_drains + adrenaline_effect
        
        # Clamp to 0-100 range
        return max(0, min(100, raw_energy))
    
    def _get_dominant_influence(self, levels: Dict[str, float]) -> str:
        """Determine which hormone has the strongest influence."""
        # Weight by distance from baseline
        influences = {}
        
        for hormone, level in levels.items():
            baseline = self.baseline_levels[hormone]
            distance = abs(level - baseline)
            # Weight extreme values more heavily
            weighted_distance = distance * (level / 100) if level > baseline else distance
            influences[hormone] = weighted_distance
        
        return max(influences, key=influences.get) if influences else 'balanced'
    
    def _generate_behavioral_guidance(
        self,
        levels: Dict[str, float],
        drives: Dict[str, float]
    ) -> List[str]:
        """Generate specific behavioral guidance based on hormone patterns."""
        guidance = []
        
        # High stress suppresses other drives
        if drives['stress_level'] > 0.7:
            guidance.append("High stress - prefer brief, safe interactions")
            guidance.append("Avoid complex decisions or commitments")
        
        # Low energy state
        if levels['dopamine'] < 40 and levels['cortisol'] > 60:
            guidance.append("Low energy and stressed - minimal engagement preferred")
        
        # High energy, motivated state
        if levels['dopamine'] > 70 and levels['cortisol'] < 50:
            guidance.append("High motivation - seek interesting topics and challenges")
        
        # Social connection drive
        if drives['connection_desire'] > 0.6:
            guidance.append("Strong connection desire - more open to bonding and sharing")
        elif drives['connection_desire'] < 0.3:
            guidance.append("Low social drive - prefer maintaining distance")
        
        # Reward/achievement motivation
        if drives['reward_seeking'] > 0.7:
            guidance.append("High reward motivation - seek accomplishment and progress")
        
        # Status and confidence
        if drives['status_seeking'] > 0.7:
            guidance.append("High status drive - want to demonstrate competence")
        elif levels['serotonin'] < 40:
            guidance.append("Low confidence - may be self-deprecating or hesitant")
        
        # Pleasure seeking
        if drives['pleasure_seeking'] > 0.6:
            guidance.append("Seeking pleasure/comfort - prefer enjoyable interactions")
        
        # High arousal/excitement
        if drives['arousal_level'] > 0.7:
            guidance.append("High arousal - may be more impulsive or reactive")
        
        return guidance if guidance else ["Neurochemical balance is relatively neutral"]
    
    def _create_analysis_text(
        self,
        levels: Dict[str, float],
        drives: Dict[str, float],
        energy: float,
        dominant: str
    ) -> str:
        """Create human-readable analysis of neurochemical state."""
        # Get behavioral guidance
        guidance = self._generate_behavioral_guidance(levels, drives)
        
        analysis_parts = [
            f"Energy level: {energy:.0f}/100",
            f"Dominant influence: {dominant} ({levels[dominant]:.0f}/100)",
            f"Behavioral guidance: {'; '.join(guidance)}"
        ]
        
        return " | ".join(analysis_parts)
    
    def _calculate_priority(self, levels: Dict[str, float]) -> float:
        """Calculate how important neurochemical state is for current response."""
        # Priority increases with how far levels are from baseline
        total_deviation = 0
        
        for hormone, level in levels.items():
            baseline = self.baseline_levels[hormone]
            deviation = abs(level - baseline) / 100  # Normalize to 0-1
            total_deviation += deviation
        
        # Average deviation, with bonus for extreme values
        avg_deviation = total_deviation / len(levels)
        
        # Check for extreme states that should override other considerations
        extreme_bonus = 0
        if levels['cortisol'] > 80 or levels['dopamine'] < 20:
            extreme_bonus = 0.3
        
        priority = min(1.0, avg_deviation + extreme_bonus + 0.3)  # Base priority of 0.3
        return priority
    
    def _map_hormones_to_emotion(self, levels: Dict[str, float]) -> str:
        """Map hormone pattern to emotional state description."""
        cortisol = levels['cortisol']
        dopamine = levels['dopamine']
        oxytocin = levels['oxytocin']
        serotonin = levels['serotonin']
        
        if cortisol > 70:
            return "stressed"
        elif dopamine > 70 and oxytocin > 60:
            return "happy_connected"
        elif dopamine > 70:
            return "excited"
        elif serotonin > 70:
            return "confident"
        elif oxytocin > 70:
            return "warm"
        elif dopamine < 40 and cortisol > 50:
            return "depleted"
        else:
            return "neutral"
    
    def _get_behavioral_recommendations(self, levels: Dict[str, float]) -> Dict[str, str]:
        """Get specific behavioral recommendations for other agents."""
        recommendations = {}
        
        # Response length recommendations
        energy = self._calculate_energy_level(levels)
        if energy > 70:
            recommendations['response_length'] = "verbose"
        elif energy < 40:
            recommendations['response_length'] = "brief"
        else:
            recommendations['response_length'] = "moderate"
        
        # Social engagement recommendations
        oxytocin = levels['oxytocin']
        cortisol = levels['cortisol']
        
        if oxytocin > 60 and cortisol < 60:
            recommendations['social_engagement'] = "high"
        elif cortisol > 70:
            recommendations['social_engagement'] = "low"
        else:
            recommendations['social_engagement'] = "moderate"
        
        # Risk-taking recommendations
        if levels['cortisol'] > 70:
            recommendations['risk_tolerance'] = "very_low"
        elif levels['dopamine'] > 70 and levels['adrenaline'] > 60:
            recommendations['risk_tolerance'] = "high"
        else:
            recommendations['risk_tolerance'] = "moderate"
        
        return recommendations
    
    def _get_gender_modifiers(self) -> Dict[str, float]:
        """Get gender-based hormone sensitivity modifiers."""
        gender_config = self.config.get('gender_modifiers', {})
        
        if not gender_config.get('applies', False):
            return {hormone: 1.0 for hormone in self.baseline_sensitivities.keys()}
        
        modifier_set = gender_config.get('modifier_set', 'neutral')
        
        # Default modifiers (these would be tuned based on research)
        if modifier_set == 'male':
            return {
                'dopamine': 1.1,    # Slightly higher reward sensitivity
                'serotonin': 0.9,   # Slightly lower status sensitivity
                'oxytocin': 0.9,    # Slightly lower bonding drive
                'endorphins': 1.0,
                'cortisol': 0.9,    # Slightly lower stress sensitivity
                'adrenaline': 1.1   # Slightly higher thrill tolerance
            }
        elif modifier_set == 'female':
            return {
                'dopamine': 0.9,
                'serotonin': 1.1,
                'oxytocin': 1.2,    # Higher bonding drive
                'endorphins': 1.1,
                'cortisol': 1.1,    # Higher stress sensitivity
                'adrenaline': 0.9
            }
        else:
            return {hormone: 1.0 for hormone in self.baseline_sensitivities.keys()}
    
    def apply_decay(self, current_levels: Dict[str, float]) -> Dict[str, float]:
        """Apply time-based decay to hormone levels."""
        new_levels = {}
        
        for hormone, level in current_levels.items():
            baseline = self.baseline_levels[hormone]
            decay_rate = self.decay_rates[hormone]
            
            # Move toward baseline with exponential decay
            if level > baseline:
                # Decay from elevated levels
                difference = level - baseline
                new_level = level - (difference * decay_rate)
                new_levels[hormone] = max(baseline, new_level)
            elif level < baseline:
                # Recovery toward baseline
                difference = baseline - level
                new_level = level + (difference * decay_rate)
                new_levels[hormone] = min(baseline, new_level)
            else:
                # Already at baseline
                new_levels[hormone] = baseline
        
        return new_levels
    
    def calculate_hormone_change(
        self,
        stimulus_type: str,
        intensity: float,
        character_response: str
    ) -> Dict[str, float]:
        """
        Calculate how a stimulus/interaction changes hormone levels.
        
        This models how events trigger neurochemical responses.
        """
        changes = {hormone: 0.0 for hormone in self.decay_rates.keys()}
        
        # Apply gender modifiers to sensitivity
        sensitivity_mods = self.gender_modifiers
        
        # Pattern matching for stimulus types and responses
        response_lower = character_response.lower()
        
        # Positive interactions
        if stimulus_type == "compliment" or any(word in response_lower for word in ["thank", "great", "wonderful"]):
            changes['dopamine'] += 15 * intensity * sensitivity_mods['dopamine']
            changes['serotonin'] += 10 * intensity * sensitivity_mods['serotonin']
        
        # Conflict or stress
        if stimulus_type == "conflict" or any(word in response_lower for word in ["angry", "upset", "disagree", "frustrated"]):
            changes['cortisol'] += 20 * intensity * sensitivity_mods['cortisol']
            changes['adrenaline'] += 15 * intensity * sensitivity_mods['adrenaline']
        
        # Social connection
        if stimulus_type == "connection" or any(word in response_lower for word in ["friend", "close", "trust", "love", "care"]):
            changes['oxytocin'] += 12 * intensity * sensitivity_mods['oxytocin']
        
        # Achievement or success
        if stimulus_type == "achievement" or any(word in response_lower for word in ["accomplish", "succeed", "win", "complete"]):
            changes['dopamine'] += 18 * intensity * sensitivity_mods['dopamine']
            changes['serotonin'] += 8 * intensity * sensitivity_mods['serotonin']
        
        # Humor and pleasure
        if stimulus_type == "humor" or any(word in response_lower for word in ["laugh", "funny", "haha", "lol", "joke"]):
            changes['endorphins'] += 15 * intensity * sensitivity_mods['endorphins']
            changes['dopamine'] += 8 * intensity * sensitivity_mods['dopamine']
        
        # Fear or threat
        if stimulus_type == "threat" or any(word in response_lower for word in ["scary", "afraid", "danger", "worried"]):
            changes['cortisol'] += 25 * intensity * sensitivity_mods['cortisol']
            changes['adrenaline'] += 20 * intensity * sensitivity_mods['adrenaline']
        
        # Learning or novelty
        if stimulus_type == "learning" or any(word in response_lower for word in ["interesting", "learn", "discover", "new"]):
            changes['dopamine'] += 10 * intensity * sensitivity_mods['dopamine']
        
        return changes