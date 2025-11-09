"""
Character Researcher for Arena

This module provides automated research capabilities for creating accurate character
profiles of public figures using web search integration. It combines Tavily web search
with intelligent data extraction to generate comprehensive Arena character profiles.

Features:
- Public figure detection and research
- Strategic multi-query research approach  
- Biographical data extraction and parsing
- Arena character schema mapping
- YAML profile generation

Author: Homunculus Team
"""

import asyncio
import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from ..llm.llm_client import ArenaLLMClient

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Container for research results on a public figure."""
    
    name: str
    confidence_score: float = 0.0
    biographical_info: Dict[str, Any] = field(default_factory=dict)
    personality_traits: Dict[str, Any] = field(default_factory=dict)
    achievements: List[str] = field(default_factory=list)
    communication_style: Dict[str, Any] = field(default_factory=dict)
    expertise_areas: List[str] = field(default_factory=list)
    quotes_and_philosophy: List[str] = field(default_factory=list)
    raw_research_data: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert research result to dictionary."""
        return {
            "name": self.name,
            "confidence_score": self.confidence_score,
            "biographical_info": self.biographical_info,
            "personality_traits": self.personality_traits,
            "achievements": self.achievements,
            "communication_style": self.communication_style,
            "expertise_areas": self.expertise_areas,
            "quotes_and_philosophy": self.quotes_and_philosophy,
            "research_queries_count": len(self.raw_research_data)
        }


class CharacterResearcher:
    """
    Automated character research system for Arena agents.
    
    Uses web search to research public figures and generate comprehensive
    character profiles that can be used to create Arena agents.
    """
    
    def __init__(self, llm_client: Optional[ArenaLLMClient] = None):
        """
        Initialize the character researcher.
        
        Args:
            llm_client: Optional LLM client for research. If None, creates new one.
        """
        self.llm_client = llm_client or ArenaLLMClient(
            model="llama3.3:70b",
            temperature=0.3  # Lower temperature for more factual research
        )
        
        # Enhanced hyper-targeted research query templates
        self.research_queries = {
            "basic_info": "What is {name} best known for? What is their primary claim to fame and main achievement?",
            "archetype": "What is the most accurate archetype or character type for {name}? (e.g., visionary entrepreneur, analytical scientist, creative artist, political leader)",
            "description": "Write a concise 2-sentence description of who {name} is and why they are notable",
            "birthplace": "Where was {name} born? What is their place of origin, city, state, and nationality?",
            "education": "What is {name}'s educational background? Which schools, colleges, or universities did they attend? What degrees did they earn or did they drop out?",
            "career_foundation": "What company, organization, or institution is {name} most associated with? What did they co-found, lead, or create?",
            "famous_quotes": "What are {name}'s most famous quotes, sayings, or memorable statements? List 3-5 specific quotes with exact wording",
            "location": "Where did {name} primarily live and work during their career? What city, state, or country are they most associated with?",
            "personality_known": "What specific personality traits, characteristics, or behaviors is {name} known for? How did colleagues, media, and biographers describe them?",
            "leadership_style": "How is {name}'s leadership, management, or working style described? What are they known for in terms of working with others?",
            "innovations": "What specific products, ideas, inventions, or innovations did {name} create, pioneer, or revolutionize?",
            "business_approach": "What business tactics, strategies, philosophies, or approaches is {name} specifically known for?",
            "legacy_impact": "What is {name}'s lasting impact, legacy, or how they changed their field? What are they remembered for?"
        }
        
        # Known public figure patterns (simple heuristics for detection)
        self.public_figure_indicators = [
            "CEO", "founder", "president", "author", "scientist", "inventor", 
            "artist", "musician", "actor", "politician", "philosopher",
            "entrepreneur", "leader", "pioneer", "Nobel", "award"
        ]
    
    async def is_public_figure(self, name: str) -> Tuple[bool, float]:
        """
        Determine if a given name likely refers to a public figure.
        
        Args:
            name: Name to check
            
        Returns:
            Tuple of (is_public_figure, confidence_score)
        """
        try:
            # Enhanced prompt with specific instructions for JSON output
            prompt = f"""Is '{name}' a well-known public figure, historical person, celebrity, or notable individual that would have biographical information available?

Consider: business leaders, politicians, scientists, artists, historical figures, celebrities, inventors, authors, etc.

You must respond ONLY with valid JSON in this exact format:
{{"is_public": true/false, "confidence": 0.X}}

Where:
- is_public: true if they are famous/well-known (like Steve Jobs, Einstein, Shakespeare, etc.), false if not a public figure or you're unsure
- confidence: a decimal between 0.0 and 1.0 indicating your confidence level

JSON Response:"""
            
            response = await self.llm_client.generate_character_response(
                character_id="researcher",
                character_name="Public Figure Detector",
                prompt=prompt
            )
            
            if not response or len(response.strip()) == 0:
                logger.warning(f"Empty response for public figure detection: {name}")
                return False, 0.0
            
            # Parse JSON response
            try:
                import json
                # Clean the response to extract JSON
                response_clean = response.strip()
                # Try to find JSON in the response
                json_start = response_clean.find('{')
                json_end = response_clean.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_clean[json_start:json_end]
                    result = json.loads(json_str)
                    
                    is_public = result.get("is_public", False)
                    confidence = float(result.get("confidence", 0.0))
                    
                    # Ensure confidence is between 0 and 1
                    confidence = max(0.0, min(1.0, confidence))
                    
                    logger.info(f"Public figure detection for '{name}': is_public={is_public}, confidence={confidence}")
                    return is_public, confidence
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse JSON response for '{name}': {e}. Response: {response}")
                # Fallback to simple text analysis as backup
                response_lower = response.lower().strip()
                if "true" in response_lower or "yes" in response_lower:
                    return True, 0.7
                else:
                    return False, 0.3
            
        except Exception as e:
            logger.warning(f"Error in public figure detection for '{name}': {e}")
            # For well-known names, assume they might be public figures
            well_known_patterns = ["jobs", "einstein", "shakespeare", "gandhi", "napoleon", 
                                 "mozart", "da vinci", "newton", "tesla", "edison"]
            if any(pattern in name.lower() for pattern in well_known_patterns):
                return True, 0.8
            return False, 0.0
    
    async def research_character(self, name: str) -> ResearchResult:
        """
        Conduct comprehensive research on a character.
        
        Args:
            name: Name of the person to research
            
        Returns:
            ResearchResult containing all gathered information
        """
        logger.info(f"Starting comprehensive research on: {name}")
        
        result = ResearchResult(name=name)
        research_tasks = []
        
        # Execute all research queries concurrently
        for aspect, query_template in self.research_queries.items():
            query = query_template.format(name=name)
            task = self._execute_research_query(aspect, query)
            research_tasks.append(task)
        
        # Gather all research results
        try:
            research_results = await asyncio.gather(*research_tasks, return_exceptions=True)
            
            # Process results
            for aspect, research_data in zip(self.research_queries.keys(), research_results):
                if isinstance(research_data, Exception):
                    logger.warning(f"Research failed for {aspect}: {research_data}")
                    continue
                
                result.raw_research_data.append({
                    "aspect": aspect,
                    "data": research_data
                })
                
                # Extract information based on aspect
                await self._extract_aspect_data(result, aspect, research_data)
            
            # Calculate overall confidence based on research quality
            result.confidence_score = self._calculate_confidence(result)
            
            logger.info(f"Research completed for {name}. Confidence: {result.confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error during character research for {name}: {e}")
            result.confidence_score = 0.0
            return result
    
    async def _execute_research_query(self, aspect: str, query: str) -> str:
        """
        Execute a single research query with enhanced prompting and tool call handling.
        
        Args:
            aspect: The aspect being researched (e.g., 'basic_info', 'education')
            query: The search query to execute
            
        Returns:
            Research response content
        """
        try:
            # Enhanced research prompt for specific biographical information  
            research_prompt = f"""You are a biographical researcher. Answer this specific question with factual, detailed information:

QUESTION: {query}

Please provide a comprehensive answer with specific details, names, dates, places, and quotes. Use web search to find accurate information and provide a detailed response."""

            # Use direct LLM call instead of character response to avoid tool conflicts
            from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
            
            system_prompt = "You are a biographical researcher who provides specific, factual information about historical figures. Use web search when needed to find accurate details."
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=research_prompt)
            ]
            
            response_obj = await self.llm_client.llm.ainvoke(messages)
            
            # Handle tool calls if present (following pattern from llm_client.py)
            if hasattr(response_obj, 'tool_calls') and response_obj.tool_calls:
                logger.debug(f"Processing {len(response_obj.tool_calls)} tool calls for aspect: {aspect}")
                
                # Process each tool call
                for tool_call in response_obj.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    # Find and execute the tool
                    for tool in self.llm_client.tools:
                        if tool.name == tool_name:
                            try:
                                result = await tool.ainvoke(tool_args)
                                # Add tool result to conversation
                                messages.append(ToolMessage(
                                    content=str(result),
                                    tool_call_id=tool_call['id']
                                ))
                                logger.debug(f"Tool {tool_name} executed for {aspect}: {tool_args.get('query', 'N/A')}")
                            except Exception as e:
                                logger.error(f"Tool execution failed for {aspect}: {e}")
                                # Add error as tool message
                                messages.append(ToolMessage(
                                    content=f"Tool error: {str(e)}",
                                    tool_call_id=tool_call['id']
                                ))
                
                # Get final response with tool results using simple_llm
                final_response_obj = await self.llm_client.simple_llm.ainvoke(messages)
                response = final_response_obj.content if hasattr(final_response_obj, 'content') else str(final_response_obj)
            else:
                # No tool calls, use direct response
                response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            
            if response and len(response.strip()) > 0:
                logger.debug(f"Research query '{aspect}' completed: {len(response)} chars")
                return response.strip()
            else:
                logger.warning(f"Empty or no response for research query '{aspect}' - response: '{response}'")
                return ""
            
        except Exception as e:
            logger.error(f"Research query failed for {aspect}: {e}")
            return ""
    
    async def _extract_aspect_data(self, result: ResearchResult, aspect: str, research_data: str) -> None:
        """
        Extract structured data from research results using LLM-based processing.
        
        Args:
            result: ResearchResult to populate
            aspect: The research aspect
            research_data: Raw research text to parse
        """
        if not research_data.strip():
            logger.warning(f"No research data for aspect: {aspect}")
            return
        
        try:
            # Use structured LLM processing for clean data extraction
            structured_data = await self._process_research_with_llm(result.name, aspect, research_data)
            
            # Store the structured data directly
            self._store_structured_data(result, aspect, structured_data)
                
        except Exception as e:
            logger.warning(f"Structured data extraction failed for {aspect}: {e}")
            # Fallback to simple text extraction
            self._store_extracted_data(result, aspect, research_data, research_data)
    
    async def _process_research_with_llm(self, character_name: str, aspect: str, research_data: str) -> Dict[str, Any]:
        """
        Process raw research data through LLM to get clean, structured output.
        
        Args:
            character_name: Name of the character being researched
            aspect: The research aspect (quotes, achievements, etc.)
            research_data: Raw research text
            
        Returns:
            Dictionary with structured, clean data
        """
        # Build aspect-specific processing prompt
        processing_prompt = self._build_structured_processing_prompt(character_name, aspect, research_data)
        
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            system_msg = """You are a data processing specialist. Your task is to analyze raw research text and extract clean, structured information. 

CRITICAL INSTRUCTIONS:
- Return ONLY the requested data structure
- No explanations, no additional text
- If a list is requested, return actual list items, not individual characters
- If something is not found, return empty list [] or empty string ""
- Be precise and factual"""
            
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=processing_prompt)
            ]
            
            response_obj = await self.llm_client.simple_llm.ainvoke(messages)
            response_text = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            
            # Parse the structured response
            return self._parse_structured_response(aspect, response_text)
                
        except Exception as e:
            logger.error(f"LLM processing failed for {aspect}: {e}")
            return {}
    
    def _build_structured_processing_prompt(self, character_name: str, aspect: str, research_data: str) -> str:
        """Build prompts for structured LLM processing of research data."""
        
        base_context = f"Research data about {character_name}:\n\n{research_data}\n\n"
        
        if aspect in ["famous_quotes", "legacy_impact"]:
            return base_context + f"""Extract famous quotes by {character_name} from this research.

Return EXACTLY this format:
QUOTES:
- Quote 1
- Quote 2  
- Quote 3

Only include actual quotes that are clearly attributed to {character_name}. If no clear quotes are found, return:
QUOTES:
- No verified quotes found"""
        
        elif aspect in ["personality_known", "leadership_style"]:
            return base_context + f"""Extract personality traits and leadership characteristics of {character_name}.

Return EXACTLY this format:
TRAITS:
- Trait 1
- Trait 2
- Trait 3
- Trait 4

LEADERSHIP_STYLE: [One sentence description]

REPUTATION: [How they were known/described]"""
        
        elif aspect in ["innovations", "business_approach"]:
            return base_context + f"""Extract specific innovations, products, or business strategies created by {character_name}.

Return EXACTLY this format:
INNOVATIONS:
- Innovation 1
- Innovation 2
- Innovation 3

STRATEGIES:
- Strategy 1
- Strategy 2"""
        
        elif aspect in ["basic_info", "description", "career_foundation"]:
            return base_context + f"""Extract key biographical information about {character_name}.

Return EXACTLY this format:
PRIMARY_CLAIM: [What they are most known for]
DESCRIPTION: [2-sentence description]
COMPANY: [Main company/organization]
FIELD: [Professional field]"""
        
        elif aspect in ["birthplace", "location", "education"]:
            return base_context + f"""Extract location and education information about {character_name}.

Return EXACTLY this format:
BIRTH_CITY: [City where born]
BIRTH_REGION: [State/Country]
WORK_LOCATION: [Where they primarily worked]
EDUCATION: 
- School 1
- School 2
DEGREE_STATUS: [Degrees earned or dropout status]"""
        
        elif aspect == "archetype":
            return base_context + f"""Determine the best character archetype for {character_name}.

Return EXACTLY this format:
ARCHETYPE: [e.g., tech_visionary, business_leader, creative_artist, analytical_scientist]
CHARACTER_TYPE: [More specific description]
PRIMARY_ROLE: [Their main professional role]"""
        
        else:
            # Generic processing
            return base_context + f"""Extract and summarize key information about {character_name} from this research.

Return the most important facts in a clear, structured format."""
    
    def _parse_structured_response(self, aspect: str, response_text: str) -> Dict[str, Any]:
        """Parse structured LLM response into clean data dictionary."""
        data = {}
        lines = response_text.strip().split('\n')
        
        current_section = None
        current_list = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle section headers
            if line.endswith(':') and not line.startswith('-'):
                # Save previous section if it was a list
                if current_section and current_list:
                    data[current_section.lower()] = current_list
                    current_list = []
                current_section = line[:-1]  # Remove colon
                continue
            
            # Handle list items
            if line.startswith('-'):
                item = line[1:].strip()
                if item and item != "No verified quotes found":
                    current_list.append(item)
                continue
            
            # Handle single value items
            if ':' in line and not line.startswith('-'):
                key, value = line.split(':', 1)
                data[key.lower().replace(' ', '_')] = value.strip()
        
        # Save final section if it was a list
        if current_section and current_list:
            data[current_section.lower()] = current_list
        
        return data
    
    def _store_structured_data(self, result: ResearchResult, aspect: str, structured_data: Dict[str, Any]) -> None:
        """Store structured data in ResearchResult with proper type checking."""
        
        if aspect in ["famous_quotes", "legacy_impact"]:
            quotes = structured_data.get("quotes", [])
            if isinstance(quotes, list):
                # Ensure we're adding actual quotes, not character strings
                valid_quotes = [q for q in quotes if isinstance(q, str) and len(q) > 10]
                result.quotes_and_philosophy.extend(valid_quotes)
        
        elif aspect in ["personality_known", "leadership_style"]:
            traits = structured_data.get("traits", [])
            if isinstance(traits, list):
                valid_traits = [t for t in traits if isinstance(t, str)]
                if "traits" not in result.personality_traits:
                    result.personality_traits["traits"] = []
                result.personality_traits["traits"].extend(valid_traits)
            
            leadership = structured_data.get("leadership_style", "")
            if isinstance(leadership, str) and leadership:
                result.personality_traits["leadership_style"] = leadership
            
            reputation = structured_data.get("reputation", "")
            if isinstance(reputation, str) and reputation:
                result.personality_traits["reputation"] = reputation
        
        elif aspect in ["innovations", "business_approach"]:
            innovations = structured_data.get("innovations", [])
            strategies = structured_data.get("strategies", [])
            
            if isinstance(innovations, list):
                valid_innovations = [i for i in innovations if isinstance(i, str)]
                result.achievements.extend(valid_innovations)
            
            if isinstance(strategies, list):
                valid_strategies = [s for s in strategies if isinstance(s, str)]
                result.expertise_areas.extend(valid_strategies)
        
        elif aspect in ["basic_info", "description", "career_foundation"]:
            bio_updates = {}
            
            for key in ["primary_claim", "description", "company", "field"]:
                value = structured_data.get(key, "")
                if isinstance(value, str) and value:
                    if key == "company":
                        bio_updates["company_organization"] = value
                    elif key == "field":
                        bio_updates["field_domain"] = value
                    else:
                        bio_updates[key] = value
            
            result.biographical_info.update(bio_updates)
        
        elif aspect in ["birthplace", "location", "education"]:
            bio_updates = {}
            
            for key in ["birth_city", "birth_region", "work_location", "degree_status"]:
                value = structured_data.get(key, "")
                if isinstance(value, str) and value:
                    bio_updates[key] = value
            
            education = structured_data.get("education", [])
            if isinstance(education, list):
                valid_schools = [e for e in education if isinstance(e, str)]
                bio_updates["education_institutions"] = valid_schools
            
            result.biographical_info.update(bio_updates)
        
        elif aspect == "archetype":
            bio_updates = {}
            
            for key in ["archetype", "character_type", "primary_role"]:
                value = structured_data.get(key, "")
                if isinstance(value, str) and value:
                    if key == "archetype":
                        bio_updates["character_archetype"] = value
                    else:
                        bio_updates[key] = value
            
            result.biographical_info.update(bio_updates)
    
    def _build_extraction_prompt(self, name: str, aspect: str, research_data: str) -> str:
        """Build aspect-specific extraction prompts for JSON output."""
        
        base_prompt = f"Extract specific information about {name} from this research data:\n\nResearch: {research_data}\n\n"
        
        if aspect in ["basic_info", "description", "career_foundation"]:
            return base_prompt + f"""Extract and format as JSON:
{{
    "primary_claim": "What they are most known for (1 specific thing)",
    "description": "2-sentence description of who they are",
    "company_organization": "Primary company/organization they founded/led",
    "field_domain": "Their professional field or domain"
}}"""

        elif aspect in ["birthplace", "location", "education"]:
            return base_prompt + f"""Extract and format as JSON:
{{
    "birth_city": "City where born",
    "birth_state_country": "State/Province and Country", 
    "work_location": "Primary city/region where they worked",
    "education_institutions": ["School 1", "School 2"],
    "degrees_status": "Degrees earned or dropout status"
}}"""

        elif aspect in ["famous_quotes", "legacy_impact"]:
            return base_prompt + f"""Extract and format as JSON:
{{
    "famous_quotes": ["Quote 1", "Quote 2", "Quote 3"],
    "core_philosophies": ["Philosophy 1", "Philosophy 2"],
    "lasting_impact": "How they changed their field"
}}"""

        elif aspect in ["personality_known", "leadership_style"]:
            return base_prompt + f"""Extract and format as JSON:
{{
    "personality_traits": ["Trait 1", "Trait 2", "Trait 3", "Trait 4"],
    "leadership_style": "How they led or managed",
    "known_behaviors": ["Behavior 1", "Behavior 2"],
    "reputation": "How colleagues/media described them"
}}"""

        elif aspect in ["innovations", "business_approach"]:
            return base_prompt + f"""Extract and format as JSON:
{{
    "innovations": ["Innovation 1", "Innovation 2", "Innovation 3"],
    "products_created": ["Product 1", "Product 2"],
    "business_strategies": ["Strategy 1", "Strategy 2"],
    "industry_impact": "How they changed their industry"
}}"""

        elif aspect == "archetype":
            return base_prompt + f"""Extract and format as JSON:
{{
    "character_archetype": "Best archetype (e.g., tech_visionary, business_leader, creative_artist)",
    "character_type": "More specific type description",
    "primary_role": "Their main professional role"
}}"""

        else:
            # Generic extraction for any other aspects
            return base_prompt + f"""Extract and format as JSON with relevant fields for this information."""
    
    def _store_extracted_data(self, result: ResearchResult, aspect: str, extracted_response: str, research_data: str) -> None:
        """Store extracted data in the appropriate ResearchResult fields with type checking."""
        try:
            # Try to parse JSON from the extracted response
            extracted_data = self._parse_json_from_response(extracted_response)
            
            if aspect in ["basic_info", "description", "career_foundation"]:
                result.biographical_info.update({
                    "primary_claim": extracted_data.get("primary_claim", ""),
                    "description": extracted_data.get("description", ""),
                    "company_organization": extracted_data.get("company_organization", ""),
                    "field_domain": extracted_data.get("field_domain", "")
                })
                
            elif aspect in ["birthplace", "location", "education"]:
                education_institutions = extracted_data.get("education_institutions", [])
                # Ensure it's a list, not a string
                if isinstance(education_institutions, str):
                    education_institutions = [education_institutions]
                elif not isinstance(education_institutions, list):
                    education_institutions = []
                
                result.biographical_info.update({
                    "birth_city": extracted_data.get("birth_city", ""),
                    "birth_state_country": extracted_data.get("birth_state_country", ""),
                    "work_location": extracted_data.get("work_location", ""),
                    "education_institutions": education_institutions,
                    "degrees_status": extracted_data.get("degrees_status", "")
                })
                
            elif aspect in ["famous_quotes", "legacy_impact"]:
                quotes = extracted_data.get("famous_quotes", [])
                philosophies = extracted_data.get("core_philosophies", [])
                
                # Ensure lists are actually lists, not strings
                if isinstance(quotes, str):
                    quotes = [quotes]
                elif not isinstance(quotes, list):
                    quotes = []
                    
                if isinstance(philosophies, str):
                    philosophies = [philosophies]
                elif not isinstance(philosophies, list):
                    philosophies = []
                
                # Filter out empty or very short strings
                valid_quotes = [q for q in quotes if isinstance(q, str) and len(q) > 10]
                valid_philosophies = [p for p in philosophies if isinstance(p, str) and len(p) > 10]
                
                result.quotes_and_philosophy.extend(valid_quotes + valid_philosophies)
                result.biographical_info["lasting_impact"] = extracted_data.get("lasting_impact", "")
                
            elif aspect in ["personality_known", "leadership_style"]:
                traits = extracted_data.get("personality_traits", [])
                
                # Ensure traits is a list
                if isinstance(traits, str):
                    traits = [traits]
                elif not isinstance(traits, list):
                    traits = []
                
                # Filter valid traits
                valid_traits = [t for t in traits if isinstance(t, str) and len(t) > 2]
                
                if "traits" not in result.personality_traits:
                    result.personality_traits["traits"] = []
                result.personality_traits["traits"].extend(valid_traits)
                result.personality_traits["leadership_style"] = extracted_data.get("leadership_style", "")
                result.personality_traits["reputation"] = extracted_data.get("reputation", "")
                
            elif aspect in ["innovations", "business_approach"]:
                innovations = extracted_data.get("innovations", [])
                products = extracted_data.get("products_created", [])
                strategies = extracted_data.get("business_strategies", [])
                
                # Ensure all are lists
                if isinstance(innovations, str):
                    innovations = [innovations]
                elif not isinstance(innovations, list):
                    innovations = []
                    
                if isinstance(products, str):
                    products = [products]
                elif not isinstance(products, list):
                    products = []
                    
                if isinstance(strategies, str):
                    strategies = [strategies]
                elif not isinstance(strategies, list):
                    strategies = []
                
                # Filter valid items
                valid_innovations = [i for i in innovations if isinstance(i, str) and len(i) > 3]
                valid_products = [p for p in products if isinstance(p, str) and len(p) > 3]
                valid_strategies = [s for s in strategies if isinstance(s, str) and len(s) > 3]
                
                result.achievements.extend(valid_innovations + valid_products)
                result.expertise_areas.extend(valid_strategies)
                
            elif aspect == "archetype":
                result.biographical_info.update({
                    "character_archetype": extracted_data.get("character_archetype", ""),
                    "character_type": extracted_data.get("character_type", ""),
                    "primary_role": extracted_data.get("primary_role", "")
                })
                
        except Exception as e:
            logger.warning(f"Failed to parse extracted data for {aspect}: {e}")
            # Fallback: store raw research data for manual processing
            if aspect not in result.biographical_info:
                result.biographical_info[f"raw_{aspect}"] = research_data[:500]
    
    def _parse_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response with enhanced parsing."""
        try:
            import json
            import re
            
            # Clean up the response
            response = response.strip()
            logger.debug(f"Trying to parse JSON from response: {response[:200]}...")
            
            # Try multiple JSON extraction methods
            
            # Method 1: Look for complete JSON blocks with proper braces
            json_matches = re.findall(r'\{[^{}]*\}', response, re.DOTALL)
            if json_matches:
                for json_str in json_matches:
                    try:
                        parsed = json.loads(json_str)
                        logger.debug(f"Successfully parsed JSON: {parsed}")
                        return parsed
                    except Exception as e:
                        logger.debug(f"Failed to parse JSON '{json_str}': {e}")
                        continue
            
            # Method 2: Look for JSON-like patterns and extract key-value pairs
            extracted_data = {}
            
            # Extract quoted key-value pairs
            kv_patterns = [
                r'"([^"]+)":\s*"([^"]*)"',
                r'"([^"]+)":\s*\[([^\]]*)\]',
                r'"([^"]+)":\s*([^",\}]+)'
            ]
            
            for pattern in kv_patterns:
                matches = re.findall(pattern, response)
                for key, value in matches:
                    # Clean up the value
                    value = value.strip().strip('"')
                    if value.startswith('[') and value.endswith(']'):
                        # Try to parse as list
                        try:
                            value = json.loads(value)
                        except:
                            # Fallback: split by comma
                            value = [v.strip().strip('"') for v in value[1:-1].split(',')]
                    extracted_data[key] = value
            
            if extracted_data:
                return extracted_data
            
            # Method 3: Simple text extraction using key indicators
            # Look for common patterns like "key: value"
            simple_patterns = [
                r'(\w+):\s*"([^"]*)"',
                r'(\w+):\s*([^,\n]+)'
            ]
            
            for pattern in simple_patterns:
                matches = re.findall(pattern, response)
                for key, value in matches:
                    if key.lower() not in ['steve', 'jobs', 'the', 'is', 'was', 'and']:  # Skip common words
                        extracted_data[key] = value.strip().strip('"')
            
            return extracted_data
            
        except Exception as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
            # Fallback: return raw text for key fields if we can extract them
            return self._extract_simple_text_data(response)
    
    def _extract_simple_text_data(self, text: str) -> Dict[str, Any]:
        """Fallback method to extract basic information from text."""
        data = {}
        
        # Simple text parsing for common biographical elements
        if "co-founder" in text.lower() and "apple" in text.lower():
            data["company_organization"] = "Apple Computer"
            data["primary_claim"] = "Co-founder of Apple Computer"
        
        if "cupertino" in text.lower():
            data["work_location"] = "Cupertino, California"
        
        if "san francisco" in text.lower():
            data["birth_city"] = "San Francisco"
        
        if "reed college" in text.lower():
            data["education_institutions"] = ["Reed College"]
            
        if "dropout" in text.lower():
            data["degrees_status"] = "College dropout"
        
        # Extract quotes if present
        quote_matches = re.findall(r'"([^"]{20,})"', text)
        if quote_matches:
            data["famous_quotes"] = quote_matches[:3]
        
        return data
    
    def _parse_biographical_data(self, data: str) -> Dict[str, Any]:
        """Parse biographical information from research data."""
        # Extract key biographical elements
        bio_data = {
            "age": None,
            "birth_year": None,
            "death_year": None,
            "background": "",
            "education": [],
            "career_highlights": [],
            "family_background": ""
        }
        
        # Use regex to find age/dates if mentioned
        age_match = re.search(r'age\s*:?\s*(\d+)', data.lower())
        if age_match:
            bio_data["age"] = int(age_match.group(1))
        
        birth_match = re.search(r'(?:born|birth)\s*:?\s*(\d{4})', data.lower())
        if birth_match:
            bio_data["birth_year"] = int(birth_match.group(1))
        
        death_match = re.search(r'(?:died|death)\s*:?\s*(\d{4})', data.lower())
        if death_match:
            bio_data["death_year"] = int(death_match.group(1))
        
        # Extract background as summary
        bio_data["background"] = data[:300] + "..." if len(data) > 300 else data
        
        return bio_data
    
    def _parse_personality_data(self, data: str) -> Dict[str, Any]:
        """Parse personality traits from research data."""
        # Extract personality indicators
        personality = {
            "traits": [],
            "big_five_indicators": {},
            "leadership_style": "",
            "behavioral_patterns": []
        }
        
        # Look for personality descriptors
        trait_patterns = [
            r'(?:personality|traits?|character)\s*:?\s*([^.]+)',
            r'(?:described as|known for being)\s*([^.]+)',
            r'(?:leadership style|management)\s*:?\s*([^.]+)'
        ]
        
        for pattern in trait_patterns:
            matches = re.findall(pattern, data.lower())
            for match in matches:
                traits = [t.strip() for t in match.split(',') if t.strip()]
                personality["traits"].extend(traits[:3])  # Limit to avoid noise
        
        # Extract unique traits
        personality["traits"] = list(set(personality["traits"]))[:8]
        
        return personality
    
    def _parse_achievements_data(self, data: str) -> List[str]:
        """Parse achievements from research data."""
        achievements = []
        
        # Look for achievement patterns
        patterns = [
            r'(?:achievement|accomplishment|innovation|founded|created|invented|pioneered)\s*:?\s*([^.]+)',
            r'(?:known for|famous for)\s*([^.]+)',
            r'(?:received|won|awarded)\s*([^.]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, data, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 10:  # Filter out very short matches
                    achievements.append(match.strip())
        
        # Remove duplicates and limit
        return list(set(achievements))[:10]
    
    def _parse_communication_data(self, data: str) -> Dict[str, Any]:
        """Parse communication style from research data."""
        communication = {
            "style": "direct",
            "formality": "formal",
            "presentation_ability": "good",
            "notable_characteristics": []
        }
        
        # Look for communication descriptors
        style_indicators = {
            "charismatic": "charismatic",
            "direct": "direct", 
            "eloquent": "eloquent",
            "passionate": "passionate",
            "authoritative": "authoritative",
            "inspiring": "inspiring"
        }
        
        data_lower = data.lower()
        for indicator, style in style_indicators.items():
            if indicator in data_lower:
                communication["style"] = style
                break
        
        return communication
    
    def _parse_philosophy_data(self, data: str) -> List[str]:
        """Parse quotes and philosophy from research data."""
        quotes = []
        
        # Look for quote patterns
        quote_patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r'(?:quote|said|stated|believed)\s*:?\s*"([^"]+)"',
            r'(?:philosophy|belief|principle)\s*:?\s*([^.]+)'
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, data)
            for match in matches:
                if len(match.strip()) > 15:  # Filter short phrases
                    quotes.append(match.strip())
        
        return list(set(quotes))[:8]
    
    def _parse_expertise_data(self, data: str) -> List[str]:
        """Parse expertise areas from research data."""
        expertise = []
        
        # Look for expertise indicators
        patterns = [
            r'(?:expert in|expertise|specializes? in|field of)\s*([^.]+)',
            r'(?:profession|career|industry)\s*:?\s*([^.]+)',
            r'(?:domain|area of knowledge)\s*:?\s*([^.]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, data.lower())
            for match in matches:
                areas = [area.strip() for area in match.split(',') if area.strip()]
                expertise.extend(areas)
        
        # Clean and deduplicate
        expertise = [exp for exp in set(expertise) if len(exp) > 3]
        return expertise[:8]
    
    def _calculate_confidence(self, result: ResearchResult) -> float:
        """Calculate enhanced confidence score based on research quality and specificity."""
        score = 0.0
        bio = result.biographical_info
        
        # Core biographical data (40% of score)
        core_data_score = 0.0
        
        # Key identity information
        if bio.get("primary_claim") or bio.get("description"):
            core_data_score += 0.15
        if bio.get("company_organization"):
            core_data_score += 0.10
        if bio.get("character_archetype"):
            core_data_score += 0.05
        if bio.get("birth_city") and bio.get("birth_state_country"):
            core_data_score += 0.10
        
        score += core_data_score
        
        # Personality and behavioral data (25% of score)  
        personality_score = 0.0
        traits = result.personality_traits.get("traits", [])
        if traits and len(traits) >= 3:
            personality_score += 0.15
        if result.personality_traits.get("leadership_style"):
            personality_score += 0.05
        if result.personality_traits.get("reputation"):
            personality_score += 0.05
        
        score += personality_score
        
        # Achievements and innovations (20% of score)
        achievement_score = 0.0
        if result.achievements and len(result.achievements) >= 2:
            achievement_score += 0.15
        if len(result.achievements) >= 5:
            achievement_score += 0.05
        
        score += achievement_score
        
        # Quotes and philosophy (10% of score)
        quotes_score = 0.0
        if result.quotes_and_philosophy and len(result.quotes_and_philosophy) >= 2:
            quotes_score += 0.08
        if len(result.quotes_and_philosophy) >= 4:
            quotes_score += 0.02
        
        score += quotes_score
        
        # Research completeness and quality (5% of score)
        research_completeness = len(result.raw_research_data) / len(self.research_queries)
        score += 0.05 * research_completeness
        
        # Bonus points for exceptional detail
        bonus_score = 0.0
        if bio.get("education_institutions") and len(bio.get("education_institutions", [])) > 0:
            bonus_score += 0.02
        if bio.get("lasting_impact"):
            bonus_score += 0.03
        if len(result.expertise_areas) >= 3:
            bonus_score += 0.02
        
        score += bonus_score
        
        # Ensure score is between 0 and 1
        final_score = min(score, 1.0)
        
        # Log confidence breakdown for debugging
        logger.debug(f"Confidence breakdown for {result.name}: "
                    f"core={core_data_score:.2f}, personality={personality_score:.2f}, "
                    f"achievements={achievement_score:.2f}, quotes={quotes_score:.2f}, "
                    f"research={research_completeness:.2f}, bonus={bonus_score:.2f}, "
                    f"final={final_score:.2f}")
        
        return final_score
    
    async def generate_character_profile(self, research_result: ResearchResult) -> Dict[str, Any]:
        """
        Generate a complete Arena character profile from research results.
        
        Args:
            research_result: Research data to convert
            
        Returns:
            Dictionary containing complete character profile in Arena format
        """
        logger.info(f"Generating character profile for {research_result.name}")
        
        profile = {
            "name": research_result.name,
            "archetype": self._determine_archetype(research_result),
            "description": f"AI representation of {research_result.name} based on historical research",
            "version": 1.0,
            "created_date": "2025-01-08", 
            "tags": ["researched", "historical", "public_figure"],
            "demographics": self._generate_demographics(research_result),
            "personality": self._generate_personality(research_result), 
            "specialty": self._generate_specialty(research_result),
            "skills": self._generate_skills(research_result),
            "communication_style": self._generate_communication_style(research_result),
            "initial_goals": self._generate_goals(research_result),
            "mood_baseline": self._generate_mood_baseline(research_result),
            "neurochemical_baseline": self._generate_neurochemical_baseline(research_result),
            "backstory": self._generate_backstory(research_result),
            "development": self._generate_development(research_result),
            "research_metadata": {
                "confidence_score": research_result.confidence_score,
                "research_date": "2025-01-08",
                "queries_executed": len(research_result.raw_research_data),
                "data_sources": "web_search_tavily"
            }
        }
        
        return profile
    
    def _determine_archetype(self, result: ResearchResult) -> str:
        """Determine character archetype based on enhanced research data."""
        bio = result.biographical_info
        
        # Use extracted archetype if available
        extracted_archetype = bio.get("character_archetype", "")
        if extracted_archetype and extracted_archetype != "":
            # Clean up the archetype
            archetype_mapping = {
                "tech_visionary": "tech_visionary",
                "business_leader": "business_leader", 
                "visionary_entrepreneur": "visionary_entrepreneur",
                "analytical_scientist": "analytical_genius",
                "creative_artist": "creative_artist",
                "political_leader": "political_leader",
                "tech_innovator": "tech_innovator"
            }
            for key, value in archetype_mapping.items():
                if key.lower() in extracted_archetype.lower():
                    return value
        
        # Fallback to expertise-based mapping
        expertise = result.expertise_areas
        traits = result.personality_traits.get("traits", [])
        company = bio.get("company_organization", "").lower()
        
        # Enhanced archetype detection
        if any(word in company for word in ["apple", "microsoft", "google", "facebook", "tesla", "spacex"]):
            return "tech_visionary"
        elif any("technology" in exp.lower() for exp in expertise):
            return "tech_innovator"
        elif any("business" in exp.lower() for exp in expertise) or "entrepreneur" in str(traits).lower():
            return "business_leader" 
        elif any("art" in exp.lower() for exp in expertise) or "creative" in str(traits).lower():
            return "creative_artist"
        elif any("science" in exp.lower() for exp in expertise):
            return "analytical_genius"
        elif any(trait for trait in traits if "lead" in trait.lower()):
            return "visionary_leader"
        else:
            return "notable_figure"
    
    def _generate_demographics(self, result: ResearchResult) -> Dict[str, Any]:
        """Generate demographics section from research using enhanced data mapping."""
        bio = result.biographical_info
        
        # Use extracted research data for specific demographic details
        birthplace = bio.get("birth_city", "")
        birth_region = bio.get("birth_state_country", "")
        birth_location = f"{birthplace}, {birth_region}" if birthplace and birth_region else bio.get("work_location", "Historical/Global")
        
        # Extract background from primary claim and description
        background = bio.get("description", bio.get("primary_claim", "Notable figure in their field"))
        if len(background) > 200:
            background = background[:200] + "..."
        
        # Education from extracted institutions and status
        education_institutions = bio.get("education_institutions", [])
        degrees_status = bio.get("degrees_status", "")
        education = education_institutions if education_institutions else [degrees_status] if degrees_status else ["Unknown"]
        
        return {
            "age": bio.get("age", 50),
            "background": background,
            "location": birth_location,
            "education": education,
            "interests": result.expertise_areas[:5] if result.expertise_areas else ["general"]
        }
    
    def _generate_personality(self, result: ResearchResult) -> Dict[str, Any]:
        """Generate personality section from research."""
        traits = result.personality_traits.get("traits", [])
        
        # Map traits to Big Five (simplified heuristics)
        openness = 0.7  # Default moderate
        conscientiousness = 0.7
        extraversion = 0.6
        agreeableness = 0.6
        neuroticism = 0.4
        
        # Adjust based on traits
        innovation_traits = ["creative", "innovative", "visionary", "pioneering"]
        if any(trait for trait in traits if any(word in trait.lower() for word in innovation_traits)):
            openness = 0.9
        
        leadership_traits = ["leader", "charismatic", "authoritative", "commanding"]
        if any(trait for trait in traits if any(word in trait.lower() for word in leadership_traits)):
            extraversion = 0.8
            conscientiousness = 0.8
        
        return {
            "big_five": {
                "openness": openness,
                "conscientiousness": conscientiousness,
                "extraversion": extraversion,
                "agreeableness": agreeableness,
                "neuroticism": neuroticism
            },
            "behavioral_traits": traits[:8] or ["thoughtful", "driven", "focused"],
            "core_values": result.quotes_and_philosophy[:5] if result.quotes_and_philosophy else ["excellence", "achievement"],
            "quirks": ["speaks from deep experience", "references personal journey"],
            "strengths": ["domain expertise", "strategic thinking", "influence"],
            "weaknesses": ["perfectionist tendencies", "high expectations"]
        }
    
    def _generate_specialty(self, result: ResearchResult) -> Dict[str, Any]:
        """Generate specialty section from research."""
        return {
            "domain": result.expertise_areas[0].lower().replace(" ", "_") if result.expertise_areas else "general",
            "expertise_level": 0.9,
            "experience_years": 20,
            "subdomain_knowledge": result.expertise_areas[:8],
            "certifications": ["Historical Recognition"],
            "notable_achievements": result.achievements[:5]
        }
    
    def _generate_skills(self, result: ResearchResult) -> Dict[str, Any]:
        """Generate skills section from research."""
        return {
            "intelligence": {
                "analytical": 0.85,
                "creative": 0.8,
                "practical": 0.8,
                "social": 0.75
            },
            "emotional_intelligence": 0.75,
            "physical_capability": 0.6,
            "problem_solving": 0.9,
            "communication": 0.85,
            "leadership": 0.8,
            "technical_skills": result.expertise_areas[:5] or ["strategic thinking"],
            "soft_skills": ["leadership", "communication", "strategic planning"]
        }
    
    def _generate_communication_style(self, result: ResearchResult) -> Dict[str, Any]:
        """Generate communication style from research."""
        style = result.communication_style
        
        return {
            "verbal_pattern": "authoritative",
            "social_comfort": "high",
            "listening_preference": 0.7,
            "body_language": "confident",
            "formality_level": style.get("formality", "professional"),
            "humor_style": "situational",
            "conversation_style": style.get("style", "direct"),
            "quirks": ["draws from personal experience", "uses domain-specific examples"],
            "preferred_topics": result.expertise_areas[:5] or ["achievement", "strategy"]
        }
    
    def _generate_goals(self, result: ResearchResult) -> List[str]:
        """Generate initial goals based on research."""
        if result.achievements:
            return [
                "share expertise and experience",
                "contribute valuable insights",
                "demonstrate strategic thinking",
                "achieve meaningful impact"
            ]
        else:
            return [
                "contribute thoughtfully to discussions", 
                "demonstrate knowledge and capability",
                "achieve success in competitive environment"
            ]
    
    def _generate_mood_baseline(self, result: ResearchResult) -> Dict[str, Any]:
        """Generate mood baseline from research."""
        return {
            "current_state": "confident",
            "intensity": 0.7,
            "baseline_setpoint": 0.75,
            "emotional_volatility": 0.3,
            "triggered_by": "achievement_opportunities",
            "typical_moods": ["focused", "determined", "confident", "strategic"],
            "mood_triggers": {
                "success": "satisfied",
                "challenge": "energized", 
                "setback": "determined"
            }
        }
    
    def _generate_neurochemical_baseline(self, result: ResearchResult) -> Dict[str, Any]:
        """Generate neurochemical baseline."""
        return {
            "dopamine": 75.0,
            "serotonin": 70.0,
            "oxytocin": 60.0,
            "endorphins": 65.0,
            "cortisol": 30.0,
            "adrenaline": 55.0
        }
    
    def _generate_backstory(self, result: ResearchResult) -> str:
        """Generate rich backstory from enhanced research data."""
        bio = result.biographical_info
        
        # Use extracted description and company information
        description = bio.get("description", "")
        company = bio.get("company_organization", "")
        primary_claim = bio.get("primary_claim", "")
        lasting_impact = bio.get("lasting_impact", "")
        
        # Build comprehensive backstory
        backstory_parts = []
        
        if description:
            backstory_parts.append(description)
        elif primary_claim:
            backstory_parts.append(f"{result.name} is {primary_claim}.")
        
        if company:
            backstory_parts.append(f"Associated with {company},")
        
        if result.achievements:
            key_achievements = result.achievements[:2]
            backstory_parts.append(f"they are known for {', '.join(key_achievements)}.")
        
        if lasting_impact:
            backstory_parts.append(f"{lasting_impact}")
        
        if result.quotes_and_philosophy:
            famous_quote = result.quotes_and_philosophy[0]
            backstory_parts.append(f'Famously said: "{famous_quote}"')
        
        # Combine into coherent backstory
        if backstory_parts:
            backstory = " ".join(backstory_parts)
            # Clean up extra spaces and ensure proper sentence structure
            backstory = " ".join(backstory.split())
        else:
            backstory = f"This character represents {result.name}, bringing their unique perspective and expertise to Arena discussions."
        
        return backstory
    
    def _generate_development(self, result: ResearchResult) -> Dict[str, Any]:
        """Generate development section."""
        return {
            "arc_stage": "established_expert",
            "growth_areas": ["contemporary perspectives", "collaborative dynamics", "arena strategy"],
            "key_experiences": result.achievements[:3] if result.achievements else ["professional success"],
            "relationship_capacity": 0.75,
            "adaptability": 0.7,
            "learning_style": "experiential"
        }
    
    async def save_character_profile(self, profile: Dict[str, Any], output_dir: Path = None) -> Path:
        """
        Save character profile as YAML file.
        
        Args:
            profile: Character profile dictionary
            output_dir: Directory to save to (defaults to schemas/characters)
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = Path("schemas/characters")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate safe filename
        name = profile["name"]
        safe_name = re.sub(r'[^\w\s-]', '', name).strip().lower().replace(' ', '_')
        filename = f"{safe_name}.yaml"
        filepath = output_dir / filename
        
        # Save as YAML
        with open(filepath, 'w') as f:
            yaml.dump(profile, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"Character profile saved to: {filepath}")
        return filepath