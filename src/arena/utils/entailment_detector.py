"""
Entailment Detection for Arena Conversations

Adapted from the talks project to detect logical entailments and ensure
agents contribute meaningful new implications rather than just restating ideas.
"""

import re
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EntailmentType(Enum):
    """Types of entailments detected in arena discussions"""
    IMPLICATION = "implication"
    APPLICATION = "application"
    PREDICTION = "prediction"
    COUNTEREXAMPLE = "counterexample"
    TEST = "test"
    CONSEQUENCE = "consequence"
    STRATEGY = "strategy"
    RISK = "risk"


class EntailmentDetector:
    """Detects logical entailments and implications in arena responses"""
    
    def __init__(self):
        """Initialize entailment detector with arena-specific patterns"""
        
        # Arena-focused entailment patterns
        self.patterns = {
            EntailmentType.IMPLICATION: [
                r'\bif\b.*\bthen\b',
                r'\bthis (leads to|results in|causes|means|implies)\b',
                r'\b(therefore|thus|hence|so|consequently)\b',
                r'\bwhen.*\bwe (will|would|can|could)\b',
                r'\bsince.*\b(this means|we can conclude)\b'
            ],
            
            EntailmentType.APPLICATION: [
                r'\bin practice\b.*\bwe (should|would|could|must)\b',
                r'\bthis means we (need to|should|must|can)\b',
                r'\bto implement this\b',
                r'\bin the real world\b.*\b(requires|demands|needs)\b',
                r'\bfor our company\b.*\b(this would|we would|we should)\b'
            ],
            
            EntailmentType.PREDICTION: [
                r'\bby \d{4}\b.*\bwill\b',
                r'\bi predict\b.*\bwill\b',
                r'\bwithin.*years\b.*\bwe (will see|expect)\b',
                r'\bthis will (lead to|result in|cause)\b',
                r'\bwe can expect\b.*\bto\b',
                r'\bthe market will\b'
            ],
            
            EntailmentType.COUNTEREXAMPLE: [
                r'\bhowever\b.*\b(if|when|unless)\b',
                r'\bunless\b.*\bthen\b',
                r'\bbut what if\b',
                r'\bexcept when\b',
                r'\bon the other hand\b.*\b(would|could|might)\b',
                r'\bthe challenge is\b.*\bwould\b'
            ],
            
            EntailmentType.TEST: [
                r'\bwe (could|should|can) test\b.*\bby\b',
                r'\ba key (metric|indicator|measure)\b.*\bwould be\b',
                r'\bto validate this\b.*\bwe (need|should|could)\b',
                r'\bproof of concept\b.*\bwould\b',
                r'\bthe criterion for success\b'
            ],
            
            EntailmentType.CONSEQUENCE: [
                r'\bthis would require\b',
                r'\bthe cost would be\b',
                r'\bwe would need to (invest|build|hire)\b',
                r'\bthis creates.*\b(opportunity|risk|challenge)\b',
                r'\bthe implication is\b',
                r'\bas a result\b.*\bwe (must|should|would)\b'
            ],
            
            EntailmentType.STRATEGY: [
                r'\bour strategy (should|would|could) be\b',
                r'\bto achieve this\b.*\bwe (need|must|should)\b',
                r'\bthe path forward\b.*\b(requires|involves)\b',
                r'\bour competitive advantage\b.*\b(would be|comes from)\b',
                r'\bto scale this\b.*\bwe (would|need to|must)\b'
            ],
            
            EntailmentType.RISK: [
                r'\bthe risk is\b.*\b(could|might|would)\b',
                r'\bif we don\'t\b.*\bthen\b',
                r'\bthis could (fail|backfire)\b.*\bif\b',
                r'\bthe downside\b.*\b(would be|is that)\b',
                r'\bwe risk\b.*\bby\b'
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for ent_type, pattern_list in self.patterns.items():
            self.compiled_patterns[ent_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in pattern_list
            ]
    
    def detect(self, text: str) -> List[Dict[str, any]]:
        """
        Detect entailments in the given text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of entailment dictionaries with type, pattern, and match info
        """
        entailments = []
        
        for ent_type, patterns in self.compiled_patterns.items():
            for i, pattern in enumerate(patterns):
                matches = pattern.finditer(text)
                for match in matches:
                    entailments.append({
                        "type": ent_type,
                        "pattern_index": i,
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": self._calculate_confidence(ent_type, match.group())
                    })
        
        # Sort by position in text
        entailments.sort(key=lambda x: x["start"])
        
        # Remove overlapping matches, keeping the highest confidence
        entailments = self._remove_overlaps(entailments)
        
        logger.debug(f"Detected {len(entailments)} entailments: {[e['type'].value for e in entailments]}")
        return entailments
    
    def _calculate_confidence(self, ent_type: EntailmentType, match_text: str) -> float:
        """Calculate confidence score for an entailment match"""
        
        # Base confidence by type
        base_confidence = {
            EntailmentType.IMPLICATION: 0.9,
            EntailmentType.APPLICATION: 0.8,
            EntailmentType.PREDICTION: 0.9,
            EntailmentType.COUNTEREXAMPLE: 0.7,
            EntailmentType.TEST: 0.8,
            EntailmentType.CONSEQUENCE: 0.8,
            EntailmentType.STRATEGY: 0.9,
            EntailmentType.RISK: 0.8
        }
        
        confidence = base_confidence.get(ent_type, 0.5)
        
        # Boost confidence for stronger language
        strong_indicators = [
            'must', 'will', 'requires', 'necessarily', 'always', 
            'never', 'certainly', 'definitely', 'clearly'
        ]
        
        for indicator in strong_indicators:
            if indicator in match_text.lower():
                confidence = min(1.0, confidence + 0.1)
                break
        
        return confidence
    
    def _remove_overlaps(self, entailments: List[Dict]) -> List[Dict]:
        """Remove overlapping entailment matches, keeping highest confidence"""
        if not entailments:
            return entailments
        
        result = []
        current = entailments[0]
        
        for next_ent in entailments[1:]:
            # Check if they overlap
            if (current["start"] <= next_ent["start"] <= current["end"] or 
                next_ent["start"] <= current["start"] <= next_ent["end"]):
                
                # Keep the one with higher confidence
                if next_ent["confidence"] > current["confidence"]:
                    current = next_ent
            else:
                result.append(current)
                current = next_ent
        
        result.append(current)
        return result
    
    def has_entailment(self, text: str, min_confidence: float = 0.6) -> bool:
        """
        Check if text has any meaningful entailments
        
        Args:
            text: Text to check
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if text contains entailments above threshold
        """
        entailments = self.detect(text)
        
        # Check for formulaic beginnings that should disqualify responses
        if self.has_formulaic_opening(text):
            logger.debug("Response has formulaic opening - reducing entailment confidence")
            # Reduce confidence of all entailments if response starts formulaically
            for ent in entailments:
                ent["confidence"] *= 0.7  # Reduce confidence by 30%
        
        return any(ent["confidence"] >= min_confidence for ent in entailments)
    
    def has_formulaic_opening(self, text: str) -> bool:
        """
        Check if response starts with formulaic/repetitive language
        
        Args:
            text: Text to check
            
        Returns:
            True if text starts with formulaic patterns
        """
        text_lower = text.lower().strip()
        
        # Patterns that indicate formulaic/repetitive beginnings
        formulaic_patterns = [
            r'^as we continue to explore the concept of',
            r'^as we (continue to )?(explore|delve|discuss)',
            r'^building on the points? made by',
            r'^building on.*insights?',
            r'^i\'d like to delve deeper into',
            r'^to move (the|this) discussion forward',
            r'^i\'d like to (build|add|emphasize)',
            r'^let me (build|add|emphasize)',
            r'^the research (suggests|highlights|shows)',
            r'^according to recent research',
            r'^it\'s clear that',
            r'^this clearly (shows|demonstrates)',
            r'^furthermore,?\s*(it\'s clear|we can see)',
            r'^notably,?\s*(ai|the research)',
            r'^i propose that we focus on',
            r'^(jobs|gates|musk|bezos)\'?s?\s*(insight|point)',
        ]
        
        for pattern in formulaic_patterns:
            if re.search(pattern, text_lower):
                logger.debug(f"Found formulaic pattern: {pattern[:40]}...")
                return True
        
        # Also check for repetitive trillion-dollar references at start
        trillion_patterns = [
            r'^.*creating the next trillion[- ]dollar company',
            r'^.*concept of.*trillion[- ]dollar',
            r'^.*trillion[- ]dollar.*potential',
        ]
        
        for pattern in trillion_patterns:
            if re.search(pattern, text_lower):
                logger.debug(f"Found repetitive trillion-dollar pattern")
                return True
        
        return False
    
    def get_entailment_summary(self, text: str) -> Dict[str, any]:
        """
        Get a summary of entailments in the text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with entailment analysis summary
        """
        entailments = self.detect(text)
        
        type_counts = {}
        for ent in entailments:
            ent_type = ent["type"].value
            type_counts[ent_type] = type_counts.get(ent_type, 0) + 1
        
        avg_confidence = sum(ent["confidence"] for ent in entailments) / len(entailments) if entailments else 0.0
        
        return {
            "total_entailments": len(entailments),
            "type_counts": type_counts,
            "average_confidence": avg_confidence,
            "has_entailment": len(entailments) > 0,
            "strong_entailments": len([e for e in entailments if e["confidence"] > 0.8])
        }
    
    def extract_specific_entailments(self, text: str, entailment_types: List[EntailmentType]) -> List[Dict]:
        """
        Extract only specific types of entailments
        
        Args:
            text: Text to analyze
            entailment_types: List of entailment types to look for
            
        Returns:
            List of matching entailments
        """
        all_entailments = self.detect(text)
        return [ent for ent in all_entailments if ent["type"] in entailment_types]


class BusinessEntailmentAnalyzer:
    """Specialized analyzer for business/startup context entailments"""
    
    def __init__(self):
        """Initialize with business-specific patterns"""
        self.business_keywords = {
            'market': ['market', 'customers', 'demand', 'competition', 'sector', 'industry'],
            'financial': ['revenue', 'profit', 'valuation', 'investment', 'funding', 'costs', 'roi'],
            'operational': ['scale', 'operations', 'infrastructure', 'team', 'resources', 'execution'],
            'strategic': ['competitive advantage', 'differentiation', 'positioning', 'growth', 'expansion'],
            'risk': ['risk', 'challenge', 'threat', 'uncertainty', 'failure', 'downside']
        }
    
    def analyze_business_implications(self, text: str, detector: EntailmentDetector) -> Dict[str, any]:
        """
        Analyze business-specific implications in the text
        
        Args:
            text: Text to analyze
            detector: EntailmentDetector instance
            
        Returns:
            Dictionary with business implication analysis
        """
        entailments = detector.detect(text)
        
        # Categorize by business domain
        business_entailments = {category: [] for category in self.business_keywords.keys()}
        
        text_lower = text.lower()
        
        for ent in entailments:
            match_text = ent["match"].lower()
            
            for category, keywords in self.business_keywords.items():
                if any(keyword in match_text or keyword in text_lower for keyword in keywords):
                    business_entailments[category].append(ent)
                    break
        
        return {
            "business_entailments": business_entailments,
            "market_focus": len(business_entailments['market']),
            "financial_implications": len(business_entailments['financial']),
            "operational_considerations": len(business_entailments['operational']),
            "strategic_elements": len(business_entailments['strategic']),
            "risk_awareness": len(business_entailments['risk']),
            "total_business_entailments": sum(len(ents) for ents in business_entailments.values())
        }