"""Goals agent for strategic direction and goal pursuit."""

from typing import Dict, Any, Optional, List
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


class GoalsAgent(BaseAgent):
    """
    Tracks character goals and suggests actions that advance them.
    
    Provides strategic direction by analyzing how interactions relate to
    the character's short-term, long-term, and hidden goals. May use web search
    to find information relevant to goal achievement.
    """
    
    def __init__(
        self,
        agent_id: str,
        character_id: str,
        llm_client: Any,
        initial_goals: List[Dict[str, Any]]
    ):
        """Initialize goals agent."""
        super().__init__(agent_id, "goals", character_id, llm_client)
        
        self.initial_goals = initial_goals
        
        # Goals agents benefit from web search for achievement strategies
        self.web_search_enabled = True
        self.web_search_threshold = 0.4  # Moderately likely to search for goal-related info
    
    async def consult(
        self,
        context: Dict[str, Any],
        character_state: CharacterState,
        user_message: str
    ) -> AgentInput:
        """
        Analyze if/how this interaction can advance character goals.
        
        Provides strategic guidance on goal pursuit and may use web search
        for goal-achievement strategies when appropriate.
        """
        # Get current active goals
        active_goals = character_state.agent_states.get('goals', {}).get('active_goals', [])
        
        # Analyze goal relevance and opportunities
        goal_analysis = self._analyze_goal_opportunities(
            user_message,
            active_goals,
            context
        )
        
        # Check if web search could help with goal-related information
        should_search = self._should_search_for_goal_info(user_message, active_goals)
        
        # Build goals analysis prompt
        prompt = self._build_goals_prompt(
            user_message,
            active_goals,
            goal_analysis,
            context
        )
        
        # Generate goal-oriented guidance
        response = await self._call_llm(
            prompt=prompt,
            temperature=0.6,  # Balanced creativity and strategic focus
            max_tokens=220,
            use_web_search=should_search
        )
        
        # Goals have moderate confidence and priority
        confidence = 0.75
        priority = 0.65
        
        emotional_tone = "strategic"
        
        # Package goals metadata
        metadata = {
            'active_goals': active_goals,
            'goal_opportunities': goal_analysis,
            'goal_progress_impact': self._assess_progress_impact(goal_analysis),
            'strategic_recommendations': self._get_strategic_recommendations(goal_analysis)
        }
        
        return self._create_agent_input(
            content=response,
            confidence=confidence,
            priority=priority,
            emotional_tone=emotional_tone,
            metadata=metadata
        )
    
    def get_prompt_template(self) -> str:
        """Return goals analysis prompt template."""
        return """You are analyzing how this interaction relates to the character's goals and strategic objectives.

ACTIVE GOALS:
{active_goals}

GOAL OPPORTUNITY ANALYSIS:
{goal_analysis}

USER MESSAGE: "{user_message}"
CONVERSATION CONTEXT: {conversation_context}

Based on the character's goals, analyze strategic considerations:

1. GOAL ADVANCEMENT OPPORTUNITIES:
   - Does this interaction advance any active goals? Which ones and how?
   - Are there subtle ways to make progress without being obvious?
   - What information could be gathered that supports goal achievement?

2. STRATEGIC POSITIONING:
   - How should the character present themselves to further their objectives?
   - Should they reveal, conceal, or redirect based on their goals?
   - What impression would best serve their long-term interests?

3. RELATIONSHIP BUILDING:
   - Does this person seem useful for achieving goals?
   - Should the character invest in building this relationship?
   - How can they create value/rapport that might help later?

4. INFORMATION GATHERING:
   - What could the character learn that would help their goals?
   - Are there questions they should ask?
   - What topics should they steer toward or away from?

5. GOAL PROTECTION:
   - Should any goals be kept hidden or downplayed?
   - Are there risks to revealing too much ambition?
   - How to maintain authenticity while being strategic?

6. OPPORTUNITY RECOGNITION:
   - Are there unexpected opportunities emerging from this conversation?
   - Should the character adjust their approach based on new possibilities?
   - What follow-up actions might advance their goals?

Provide 2-3 sentences on goal-related strategic considerations for this response, focusing on how the character can authentically pursue their objectives."""
    
    def _analyze_goal_opportunities(
        self,
        user_message: str,
        active_goals: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze how this interaction relates to goals."""
        opportunities = {
            'direct_advancement': [],
            'information_gathering': [],
            'relationship_building': [],
            'skill_development': [],
            'resource_access': []
        }
        
        message_lower = user_message.lower()
        
        for goal in active_goals:
            goal_description = goal.get('description', '').lower()
            goal_type = goal.get('goal_type', 'unknown')
            goal_priority = goal.get('priority', 5)
            
            # Analyze different types of opportunities
            
            # Direct advancement - message directly relates to goal
            if any(keyword in message_lower for keyword in goal_description.split()):
                opportunities['direct_advancement'].append({
                    'goal_id': goal.get('goal_id'),
                    'description': goal.get('description'),
                    'relevance': 'high',
                    'opportunity_type': 'direct_discussion'
                })
            
            # Information gathering opportunities
            if self._is_information_opportunity(user_message, goal):
                opportunities['information_gathering'].append({
                    'goal_id': goal.get('goal_id'),
                    'description': goal.get('description'),
                    'info_type': 'domain_knowledge'
                })
            
            # Relationship building for goal support
            if goal_type in ['long_term', 'hidden'] and goal_priority >= 7:
                opportunities['relationship_building'].append({
                    'goal_id': goal.get('goal_id'),
                    'description': goal.get('description'),
                    'relationship_value': 'potential_ally'
                })
            
            # Skill development opportunities
            if self._is_skill_development_opportunity(user_message, goal):
                opportunities['skill_development'].append({
                    'goal_id': goal.get('goal_id'),
                    'skill_area': self._identify_skill_area(user_message)
                })
        
        return opportunities
    
    def _is_information_opportunity(
        self,
        user_message: str,
        goal: Dict[str, Any]
    ) -> bool:
        """Check if this interaction offers information gathering opportunities."""
        message_lower = user_message.lower()
        
        # Information gathering indicators
        info_indicators = [
            'experience', 'know', 'work', 'expert', 'background',
            'industry', 'field', 'profession', 'skill', 'advice'
        ]
        
        return any(indicator in message_lower for indicator in info_indicators)
    
    def _is_skill_development_opportunity(
        self,
        user_message: str,
        goal: Dict[str, Any]
    ) -> bool:
        """Check if this interaction offers skill development opportunities."""
        message_lower = user_message.lower()
        
        skill_indicators = [
            'learn', 'teach', 'show', 'practice', 'improve',
            'method', 'technique', 'approach', 'strategy'
        ]
        
        return any(indicator in message_lower for indicator in skill_indicators)
    
    def _identify_skill_area(self, user_message: str) -> str:
        """Identify what skill area this interaction might develop."""
        message_lower = user_message.lower()
        
        skill_areas = {
            'communication': ['talk', 'speak', 'communicate', 'present', 'explain'],
            'technical': ['code', 'program', 'technical', 'system', 'computer'],
            'creative': ['create', 'design', 'art', 'creative', 'innovation'],
            'leadership': ['lead', 'manage', 'team', 'leadership', 'organize'],
            'analytical': ['analyze', 'data', 'research', 'study', 'investigate']
        }
        
        for area, keywords in skill_areas.items():
            if any(keyword in message_lower for keyword in keywords):
                return area
        
        return 'general'
    
    def _should_search_for_goal_info(
        self,
        user_message: str,
        active_goals: List[Dict[str, Any]]
    ) -> bool:
        """Determine if web search would help with goal-related information."""
        if not self.should_search_web(user_message, {}):
            return False
        
        message_lower = user_message.lower()
        
        # Search for goal achievement strategies
        strategy_indicators = [
            'how to', 'best way', 'strategy', 'approach', 'method',
            'advice', 'tips', 'guide', 'steps', 'process'
        ]
        
        if any(indicator in message_lower for indicator in strategy_indicators):
            return True
        
        # Search for industry/domain information relevant to goals
        for goal in active_goals:
            goal_description = goal.get('description', '').lower()
            
            # If message relates to a goal domain, search might be helpful
            goal_keywords = goal_description.split()
            if any(keyword in message_lower for keyword in goal_keywords if len(keyword) > 3):
                return True
        
        return False
    
    def _assess_progress_impact(self, goal_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Assess how this interaction might impact goal progress."""
        impact = {}
        
        # Direct advancement has high impact
        if goal_analysis['direct_advancement']:
            impact['direct_goals'] = 'high_progress_potential'
        
        # Information gathering has medium impact
        if goal_analysis['information_gathering']:
            impact['knowledge_goals'] = 'medium_progress_potential'
        
        # Relationship building has long-term impact
        if goal_analysis['relationship_building']:
            impact['relationship_goals'] = 'long_term_progress_potential'
        
        # Skill development varies by goal type
        if goal_analysis['skill_development']:
            impact['skill_goals'] = 'moderate_progress_potential'
        
        return impact
    
    def _get_strategic_recommendations(
        self,
        goal_analysis: Dict[str, Any]
    ) -> List[str]:
        """Get specific strategic recommendations based on goal analysis."""
        recommendations = []
        
        if goal_analysis['direct_advancement']:
            recommendations.append("Engage deeply on goal-relevant topics")
            recommendations.append("Share appropriate level of ambition")
        
        if goal_analysis['information_gathering']:
            recommendations.append("Ask strategic questions to gather intel")
            recommendations.append("Listen for useful insights and connections")
        
        if goal_analysis['relationship_building']:
            recommendations.append("Invest in relationship development")
            recommendations.append("Demonstrate value and competence")
        
        if goal_analysis['skill_development']:
            recommendations.append("Seek learning opportunities")
            recommendations.append("Practice relevant skills in conversation")
        
        if not any(goal_analysis.values()):
            recommendations.append("Focus on general relationship building")
            recommendations.append("Keep conversation engaging and authentic")
        
        return recommendations
    
    def _build_goals_prompt(
        self,
        user_message: str,
        active_goals: List[Dict[str, Any]],
        goal_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build goals analysis prompt with current state."""
        # Format goals for prompt
        goals_text = self._format_goals_for_prompt(active_goals)
        
        # Format goal analysis
        analysis_text = self._format_goal_analysis(goal_analysis)
        
        # Get conversation context
        conversation_context = context.get('topic', 'general conversation')
        
        return self.get_prompt_template().format(
            active_goals=goals_text,
            goal_analysis=analysis_text,
            user_message=user_message,
            conversation_context=conversation_context
        )
    
    def _format_goals_for_prompt(self, goals: List[Dict[str, Any]]) -> str:
        """Format goals for inclusion in prompt."""
        if not goals:
            return "No specific active goals defined"
        
        formatted_goals = []
        for goal in goals:
            goal_id = goal.get('goal_id', 'unknown')
            goal_type = goal.get('goal_type', 'unknown')
            description = goal.get('description', 'No description')
            priority = goal.get('priority', 5)
            progress = goal.get('progress', 0)
            
            formatted_goals.append(
                f"[{goal_type.upper()}] {description} "
                f"(priority: {priority}/10, progress: {progress:.0%})"
            )
        
        return "\n".join(formatted_goals)
    
    def _format_goal_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format goal analysis for prompt."""
        analysis_parts = []
        
        for category, opportunities in analysis.items():
            if opportunities:
                category_name = category.replace('_', ' ').title()
                analysis_parts.append(f"{category_name}: {len(opportunities)} opportunities")
        
        if not analysis_parts:
            return "No specific goal opportunities identified"
        
        return "; ".join(analysis_parts)
    
    def update_goal_progress(
        self,
        goal_id: str,
        progress_delta: float,
        character_state: CharacterState
    ) -> None:
        """Update progress on a specific goal."""
        goals_state = character_state.agent_states.get('goals', {})
        active_goals = goals_state.get('active_goals', [])
        
        for goal in active_goals:
            if goal.get('goal_id') == goal_id:
                current_progress = goal.get('progress', 0.0)
                new_progress = max(0.0, min(1.0, current_progress + progress_delta))
                goal['progress'] = new_progress
                
                # Move to completed if fully achieved
                if new_progress >= 1.0:
                    self._move_goal_to_completed(goal, character_state)
                
                break
    
    def _move_goal_to_completed(
        self,
        goal: Dict[str, Any],
        character_state: CharacterState
    ) -> None:
        """Move a completed goal to the completed goals list."""
        goals_state = character_state.agent_states.get('goals', {})
        
        # Add to completed goals
        completed_goals = goals_state.get('completed_goals', [])
        goal['completed_at'] = character_state.last_updated.isoformat()
        completed_goals.append(goal)
        
        # Remove from active goals
        active_goals = goals_state.get('active_goals', [])
        goals_state['active_goals'] = [
            g for g in active_goals if g.get('goal_id') != goal.get('goal_id')
        ]
        
        goals_state['completed_goals'] = completed_goals
    
    def add_new_goal(
        self,
        goal_id: str,
        goal_type: str,
        description: str,
        priority: int,
        character_state: CharacterState
    ) -> None:
        """Add a new goal that emerged during conversation."""
        goals_state = character_state.agent_states.get('goals', {})
        active_goals = goals_state.get('active_goals', [])
        
        new_goal = {
            'goal_id': goal_id,
            'goal_type': goal_type,
            'description': description,
            'priority': priority,
            'progress': 0.0,
            'created_at': character_state.last_updated.isoformat()
        }
        
        active_goals.append(new_goal)
        goals_state['active_goals'] = active_goals
    
    def _should_consult_web_for_domain(self, topic: str, user_message: str) -> bool:
        """Override to check goal-specific research needs."""
        # Goals agent often benefits from researching achievement strategies
        strategy_keywords = [
            'how to achieve', 'best practices', 'success strategies',
            'goal achievement', 'progress methods', 'improvement techniques'
        ]
        
        message_lower = user_message.lower()
        return any(keyword in message_lower for keyword in strategy_keywords)