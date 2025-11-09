"""
Dynamic LLM Parameter Management for Arena

Inspired by the talks project's sophisticated parameter handling,
this module provides context-aware LLM parameter adjustment for
enhanced creativity and content richness in arena discussions.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from ..config.arena_settings import ArenaSettings

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent role types for parameter customization"""
    CHARACTER = "character"  # Main discussion participants (Jobs, Gates, etc.)
    NARRATOR = "narrator"    # Game narration and scene setting
    JUDGE = "judge"         # Scoring and evaluation
    SYSTEM = "system"       # System messages and interventions


class GamePhase(Enum):
    """Game phases that affect creativity parameters"""
    EARLY = "early"         # Opening moves, establish positions
    MID = "mid"            # Main competition phase
    LATE = "late"          # High stakes, elimination pressure
    FINAL = "final"        # End game, final arguments


@dataclass
class LLMParameters:
    """LLM parameters for a specific generation request"""
    temperature: float
    max_tokens: int
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    
    # Ollama-specific parameters for enhanced creativity
    repeat_penalty: Optional[float] = None
    top_k: Optional[int] = None
    num_predict: Optional[int] = None  # Ollama equivalent to max_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM client"""
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        
        # Ollama-specific parameters
        if self.repeat_penalty is not None:
            params["repeat_penalty"] = self.repeat_penalty
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.num_predict is not None:
            params["num_predict"] = self.num_predict
            
        return params


class DynamicParameterManager:
    """Manages dynamic LLM parameter adjustment based on context"""
    
    def __init__(self, settings: Optional[ArenaSettings] = None):
        """Initialize with arena settings"""
        self.settings = settings or ArenaSettings()
        
        # Base parameters by role with Ollama-specific optimizations
        self.role_base_params = {
            AgentRole.CHARACTER: LLMParameters(
                temperature=self.settings.character_agent_temperature,
                max_tokens=self.settings.character_agent_max_tokens,
                top_p=self.settings.ollama_top_p_base,           # From settings
                repeat_penalty=self.settings.ollama_repeat_penalty_base,   # From settings
                top_k=self.settings.ollama_top_k_base,            # From settings
                num_predict=self.settings.character_agent_max_tokens
            ),
            AgentRole.NARRATOR: LLMParameters(
                temperature=self.settings.narrator_temperature,
                max_tokens=self.settings.narrator_max_tokens,
                top_p=max(0.1, self.settings.ollama_top_p_base - 0.02),           # Slightly more focused
                repeat_penalty=max(1.0, self.settings.ollama_repeat_penalty_base - 0.05),  # Light repetition penalty
                top_k=max(10, self.settings.ollama_top_k_base - 5),            # Standard vocabulary
                num_predict=self.settings.narrator_max_tokens
            ),
            AgentRole.JUDGE: LLMParameters(
                temperature=self.settings.judge_temperature,
                max_tokens=self.settings.judge_max_tokens,
                top_p=max(0.1, self.settings.ollama_top_p_base - 0.07),          # More focused for consistency
                repeat_penalty=1.0,   # No repetition penalty for scoring
                top_k=max(10, self.settings.ollama_top_k_base - 10),            # Narrower vocabulary for precision
                num_predict=self.settings.judge_max_tokens
            ),
            AgentRole.SYSTEM: LLMParameters(
                temperature=0.7,     # Moderate for system interventions
                max_tokens=400,
                top_p=self.settings.ollama_top_p_base,          # From settings
                repeat_penalty=max(1.0, self.settings.ollama_repeat_penalty_base - 0.05), # Light penalty
                top_k=max(10, self.settings.ollama_top_k_base - 5),           # Standard choices
                num_predict=400
            )
        }
        
        # Phase-based adjustments
        self.phase_adjustments = {
            GamePhase.EARLY: {"temp_modifier": -0.05, "token_modifier": 0.9},   # Slightly more focused
            GamePhase.MID: {"temp_modifier": 0.0, "token_modifier": 1.0},      # Baseline
            GamePhase.LATE: {"temp_modifier": 0.05, "token_modifier": 1.1},    # More creative
            GamePhase.FINAL: {"temp_modifier": 0.1, "token_modifier": 1.2}     # Maximum creativity
        }
        
        # Context-based modifiers with Ollama enhancements
        self.context_modifiers = {
            "high_competition": {
                "temp_modifier": 0.1, 
                "add_presence_penalty": 0.1,
                "repeat_penalty_boost": 0.05,  # Slightly more anti-repetition
                "top_p_boost": 0.03            # More diverse sampling
            },
            "stale_discussion": {
                "temp_modifier": 0.15, 
                "add_frequency_penalty": 0.2,
                "repeat_penalty_boost": 0.1,   # Strong anti-repetition
                "top_p_boost": 0.05,           # More diverse vocabulary
                "top_k_boost": 10              # Wider vocabulary choices
            },
            "creative_burst": {
                "temp_modifier": 0.2, 
                "token_modifier": 1.3,
                "top_p_boost": 0.03,           # More creative sampling
                "top_k_boost": 15              # Wider vocabulary
            },
            "elimination_pressure": {
                "temp_modifier": 0.1, 
                "token_modifier": 1.1,
                "repeat_penalty_boost": 0.03,  # Light anti-repetition boost
                "top_p_boost": 0.02            # Slightly more diverse
            }
        }
    
    def get_parameters(
        self, 
        agent_role: AgentRole,
        game_context: Optional[Dict[str, Any]] = None
    ) -> LLMParameters:
        """
        Get optimized LLM parameters for the given context
        
        Args:
            agent_role: The role of the agent making the request
            game_context: Current game context including phase, turn, scores, etc.
            
        Returns:
            Optimized LLM parameters
        """
        # Start with base parameters for the role
        base_params = self.role_base_params[agent_role]
        
        # Create working copy with all Ollama parameters
        params = LLMParameters(
            temperature=base_params.temperature,
            max_tokens=base_params.max_tokens,
            top_p=base_params.top_p,
            frequency_penalty=base_params.frequency_penalty,
            presence_penalty=base_params.presence_penalty,
            repeat_penalty=base_params.repeat_penalty,
            top_k=base_params.top_k,
            num_predict=base_params.num_predict
        )
        
        if not game_context:
            return params
        
        # Apply dynamic adjustments if enabled
        if self.settings.use_dynamic_temperature:
            params = self._apply_phase_adjustments(params, game_context)
            params = self._apply_context_adjustments(params, game_context)
            params = self._apply_agent_specific_adjustments(params, agent_role, game_context)
        
        # Ensure parameters are within valid bounds
        params = self._clamp_parameters(params)
        
        logger.debug(f"Generated parameters for {agent_role.value}: temp={params.temperature:.2f}, tokens={params.max_tokens}")
        
        return params
    
    def _apply_phase_adjustments(self, params: LLMParameters, context: Dict[str, Any]) -> LLMParameters:
        """Apply game phase-based adjustments"""
        phase_str = context.get("phase", "mid")
        
        try:
            phase = GamePhase(phase_str.lower())
        except ValueError:
            phase = GamePhase.MID
        
        adjustments = self.phase_adjustments[phase]
        
        # Apply temperature modifier
        temp_modifier = adjustments["temp_modifier"]
        if phase == GamePhase.LATE or phase == GamePhase.FINAL:
            temp_modifier += self.settings.creativity_boost_late_game
        
        params.temperature += temp_modifier
        
        # Apply token modifier
        token_modifier = adjustments["token_modifier"]
        params.max_tokens = int(params.max_tokens * token_modifier)
        
        return params
    
    def _apply_context_adjustments(self, params: LLMParameters, context: Dict[str, Any]) -> LLMParameters:
        """Apply context-specific adjustments including Ollama parameters"""
        
        # Check for high competition (close scores)
        if self._is_high_competition(context):
            mods = self.context_modifiers["high_competition"]
            params.temperature += mods["temp_modifier"]
            if "add_presence_penalty" in mods:
                params.presence_penalty = (params.presence_penalty or 0) + mods["add_presence_penalty"]
            if "repeat_penalty_boost" in mods:
                params.repeat_penalty = (params.repeat_penalty or 1.0) + mods["repeat_penalty_boost"]
            if "top_p_boost" in mods:
                params.top_p = min(0.99, (params.top_p or 0.9) + mods["top_p_boost"])
        
        # Check for stale discussion (detected by anti-repetition system)
        if self._is_stale_discussion(context):
            mods = self.context_modifiers["stale_discussion"]
            params.temperature += mods["temp_modifier"]
            if "add_frequency_penalty" in mods:
                params.frequency_penalty = (params.frequency_penalty or 0) + mods["add_frequency_penalty"]
            if "repeat_penalty_boost" in mods:
                params.repeat_penalty = (params.repeat_penalty or 1.0) + mods["repeat_penalty_boost"]
            if "top_p_boost" in mods:
                params.top_p = min(0.99, (params.top_p or 0.9) + mods["top_p_boost"])
            if "top_k_boost" in mods:
                params.top_k = (params.top_k or 40) + mods["top_k_boost"]
        
        # Check for elimination pressure
        if self._is_elimination_pressure(context):
            mods = self.context_modifiers["elimination_pressure"]
            params.temperature += mods["temp_modifier"]
            params.max_tokens = int(params.max_tokens * mods["token_modifier"])
            params.num_predict = params.max_tokens  # Sync Ollama equivalent
            if "repeat_penalty_boost" in mods:
                params.repeat_penalty = (params.repeat_penalty or 1.0) + mods["repeat_penalty_boost"]
            if "top_p_boost" in mods:
                params.top_p = min(0.99, (params.top_p or 0.9) + mods["top_p_boost"])
        
        return params
    
    def _apply_agent_specific_adjustments(self, params: LLMParameters, role: AgentRole, context: Dict[str, Any]) -> LLMParameters:
        """Apply agent-specific adjustments based on performance and history"""
        
        current_speaker = context.get("current_speaker")
        if not current_speaker:
            return params
        
        # Boost creativity for underperforming agents
        scores = context.get("scores", {})
        if current_speaker in scores:
            speaker_score = scores[current_speaker]
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            
            # If significantly below average, boost creativity
            if speaker_score < avg_score - 5:
                params.temperature += 0.1
                params.max_tokens = int(params.max_tokens * 1.15)
                logger.debug(f"Applied underperformance boost for {current_speaker}")
        
        # Apply variety factor if enabled
        if self.settings.enable_diverse_response_sampling and role == AgentRole.CHARACTER:
            variety_boost = self.settings.response_variety_factor
            params.temperature += variety_boost * 0.5  # Scale down variety factor
            
            # Add small presence penalty to encourage diverse vocabulary
            params.presence_penalty = (params.presence_penalty or 0) + variety_boost * 0.3
        
        return params
    
    def _is_high_competition(self, context: Dict[str, Any]) -> bool:
        """Check if scores are very close (high competition)"""
        scores = context.get("scores", {})
        if len(scores) < 2:
            return False
        
        score_values = list(scores.values())
        score_range = max(score_values) - min(score_values)
        return score_range < 3  # Scores within 3 points = high competition
    
    def _is_stale_discussion(self, context: Dict[str, Any]) -> bool:
        """Check if discussion is stale (from anti-repetition system)"""
        # This would be set by the progression controller
        return context.get("stale_discussion", False) or context.get("needs_intervention", False)
    
    def _is_elimination_pressure(self, context: Dict[str, Any]) -> bool:
        """Check if elimination is imminent"""
        return context.get("elimination_pending", False) or context.get("phase") in ["late", "final"]
    
    def _clamp_parameters(self, params: LLMParameters) -> LLMParameters:
        """Ensure parameters are within valid bounds"""
        # Clamp temperature (Ollama supports up to 2.0, but we cap at 1.0 for stability)
        params.temperature = max(0.1, min(1.0, params.temperature))
        
        # Clamp max tokens
        params.max_tokens = max(50, min(2000, params.max_tokens))
        
        # Clamp penalties if they exist
        if params.frequency_penalty is not None:
            params.frequency_penalty = max(0.0, min(2.0, params.frequency_penalty))
        if params.presence_penalty is not None:
            params.presence_penalty = max(0.0, min(2.0, params.presence_penalty))
        
        # Clamp Ollama-specific parameters
        if params.repeat_penalty is not None:
            params.repeat_penalty = max(1.0, min(1.5, params.repeat_penalty))  # 1.0-1.5 is safe range
        if params.top_p is not None:
            params.top_p = max(0.1, min(0.99, params.top_p))
        if params.top_k is not None:
            params.top_k = max(10, min(100, params.top_k))  # Reasonable vocabulary range
        if params.num_predict is not None:
            params.num_predict = max(50, min(2000, params.num_predict))
        
        return params
    
    def get_role_from_agent_type(self, agent_type: str, agent_name: str = "") -> AgentRole:
        """Determine agent role from agent type string"""
        agent_type_lower = agent_type.lower()
        agent_name_lower = agent_name.lower()
        
        if "narrator" in agent_type_lower or "narrator" in agent_name_lower:
            return AgentRole.NARRATOR
        elif "judge" in agent_type_lower or "judge" in agent_name_lower:
            return AgentRole.JUDGE
        elif "system" in agent_type_lower or agent_name_lower == "system":
            return AgentRole.SYSTEM
        else:
            return AgentRole.CHARACTER  # Default to character for discussion participants


# Global instance for easy access
_parameter_manager = None

def get_parameter_manager() -> DynamicParameterManager:
    """Get global parameter manager instance"""
    global _parameter_manager
    if _parameter_manager is None:
        _parameter_manager = DynamicParameterManager()
    return _parameter_manager


def get_llm_parameters(agent_type: str, agent_name: str = "", game_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to get LLM parameters for an agent
    
    Args:
        agent_type: Type of agent (character, narrator, judge, system)
        agent_name: Name of the agent
        game_context: Current game context
        
    Returns:
        Dictionary of LLM parameters ready for use
    """
    manager = get_parameter_manager()
    role = manager.get_role_from_agent_type(agent_type, agent_name)
    params = manager.get_parameters(role, game_context)
    return params.to_dict()