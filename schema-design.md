# Schema Design

## 1. Blueprint Schema

```python
"""
Blueprint defines the novel's initial conditions and soft constraints.
Stored as: JSON/YAML file
"""

BlueprintSchema = {
    "blueprint_id": "string (uuid)",
    "created_at": "timestamp",
    "version": "string (semantic version)",
    
    "story_metadata": {
        "title": "string",
        "genre": "string (literary_fiction | romance | thriller | mystery | sci_fi | fantasy)",
        "target_length": {
            "chapters": "integer",
            "scenes_per_chapter": "integer (range)",
            "target_words": "integer (approximate)"
        },
        "tone": "string (dark | light | neutral | mixed)",
        "themes": ["string", "..."],  # e.g., ["betrayal", "redemption", "identity"]
        "setting": {
            "time_period": "string",  # e.g., "1990s", "present day", "2150"
            "primary_location": "string",  # e.g., "Small town Ohio", "Manhattan"
            "world_building_rules": "string (optional, for fantasy/sci-fi)"
        }
    },
    
    "characters": [
        {
            "character_id": "string (uuid)",
            "name": "string",
            "role": "string (protagonist | antagonist | supporting | minor)",
            "demographics": {
                "age": "integer",
                "gender": "string",
                "ethnicity": "string (optional)",
                "occupation": "string",
                "education": "string",
                "socioeconomic_status": "string"
            },
            
            "initial_agent_states": {
                "personality": {
                    "big_five": {
                        "openness": "float (0-1)",
                        "conscientiousness": "float (0-1)",
                        "extraversion": "float (0-1)",
                        "agreeableness": "float (0-1)",
                        "neuroticism": "float (0-1)"
                    },
                    "behavioral_traits": [
                        {
                            "trait": "string",  # e.g., "risk_averse", "impulsive"
                            "intensity": "float (0-1)"
                        }
                    ],
                    "core_values": [
                        {
                            "value": "string",  # e.g., "honesty", "loyalty"
                            "priority": "integer (1-10)"
                        }
                    ]
                },
                
                "specialty": {
                    "domain": "string",  # e.g., "medicine", "law", "engineering"
                    "expertise_level": "float (0-1)",  # 0=novice, 1=expert
                    "subdomain_knowledge": ["string", "..."]
                },
                
                "skills": {
                    "intelligence": {
                        "analytical": "float (0-1)",
                        "creative": "float (0-1)",
                        "practical": "float (0-1)"
                    },
                    "emotional_intelligence": "float (0-1)",
                    "physical_capability": "float (0-1)",
                    "problem_solving": "float (0-1)"
                },
                
                "communication_style": {
                    "verbal_pattern": "string (verbose | concise | moderate)",
                    "social_comfort": "string (assertive | passive | aggressive | diplomatic)",
                    "listening_preference": "float (0-1)",  # 0=talks more, 1=listens more
                    "body_language": "string (open | closed | expressive | reserved)"
                },
                
                "mood_baseline": {
                    "default_state": "string (happy | neutral | melancholic | anxious)",
                    "emotional_volatility": "float (0-1)",  # how quickly mood shifts
                    "baseline_setpoint": "float (-1 to 1)"  # overall emotional tendency
                },
                
                "neurochemical_profile": {
                    "baseline_sensitivities": {
                        "dopamine": "float (0-2)",  # 1.0 is average
                        "serotonin": "float (0-2)",
                        "oxytocin": "float (0-2)",
                        "endorphins": "float (0-2)",
                        "cortisol": "float (0-2)",
                        "adrenaline": "float (0-2)"
                    },
                    "gender_modifiers": {
                        "applies": "boolean",
                        "modifier_set": "string (female | male | custom)"
                    },
                    "trauma_history": [
                        {
                            "trauma_type": "string",
                            "affected_hormones": ["string", "..."],
                            "sensitivity_multiplier": "float"
                        }
                    ]
                }
            },
            
            "character_arc": {
                "starting_state": "string (description)",
                "desired_end_state": "string (soft constraint)",
                "key_internal_conflicts": ["string", "..."],
                "growth_areas": ["string", "..."]
            },
            
            "initial_goals": [
                {
                    "goal_id": "string (uuid)",
                    "goal_type": "string (long_term | short_term | hidden)",
                    "description": "string",
                    "priority": "integer (1-10)",
                    "related_to": ["character_id", "..."]  # other characters involved
                }
            ],
            
            "relationships": [
                {
                    "with_character_id": "string (uuid)",
                    "relationship_type": "string (friend | family | romantic | rival | neutral)",
                    "initial_trust_level": "float (0-1)",
                    "initial_power_dynamic": "float (-1 to 1)",  # -1=they dominate, 1=I dominate
                    "history_summary": "string (optional)"
                }
            ],
            
            "secrets": [
                {
                    "secret_id": "string (uuid)",
                    "content": "string",
                    "who_knows": ["character_id", "..."],
                    "reveal_constraint": "string (optional, e.g., 'not before chapter 5')"
                }
            ]
        }
    ],
    
    "plot_skeleton": {
        "inciting_incident": {
            "description": "string",
            "target_chapter": "integer (or 'prologue')",
            "must_occur": "boolean"  # true for hard constraint
        },
        
        "plot_points": [
            {
                "plot_point_id": "string (uuid)",
                "description": "string",
                "type": "string (turning_point | revelation | confrontation | decision)",
                "target_chapter": "integer (approximate)",
                "characters_involved": ["character_id", "..."],
                "must_occur": "boolean",  # true for hard constraint, false for soft
                "alternatives_acceptable": "boolean"
            }
        ],
        
        "climax": {
            "description": "string (soft description)",
            "target_chapter": "integer (approximate)",
            "type": "string (confrontation | revelation | decision | action)",
            "emotional_peak": "string (anger | joy | fear | sadness | mixed)"
        },
        
        "resolution": {
            "description": "string (soft constraint)",
            "target_chapter": "integer",
            "open_ended": "boolean"
        }
    },
    
    "narrative_constraints": {
        "pov_style": "string (third_person_omniscient | third_person_limited | first_person)",
        "tense": "string (past | present)",
        "narrative_distance": "string (close | moderate | distant)",
        
        "hard_constraints": [
            {
                "constraint_id": "string (uuid)",
                "description": "string",  # e.g., "Character A cannot die before chapter 10"
                "type": "string (character_survival | secret_timing | relationship_state)"
            }
        ],
        
        "pacing_guidelines": {
            "action_to_reflection_ratio": "float (0-1)",  # 0=all reflection, 1=all action
            "dialogue_density": "string (sparse | moderate | heavy)",
            "scene_length_preference": "string (short | medium | long | varied)"
        }
    },
    
    "god_engine_config": {
        "genre_event_weights": {
            "micro_personal": "float (weight)",
            "micro_environmental": "float (weight)",
            "micro_social": "float (weight)",
            "meso_professional": "float (weight)",
            "meso_local_news": "float (weight)",
            "meso_health": "float (weight)",
            "macro_political": "float (weight)",
            "macro_economic": "float (weight)",
            "macro_natural": "float (weight)",
            "black_swan": "float (weight)"
        },
        "event_frequency_baseline": {
            "micro": "float (0-1, default 0.3)",
            "meso": "float (0-1, default 0.15)",
            "macro": "float (0-1, default 0.05)",
            "black_swan": "float (0-1, default 0.01)"
        }
    }
}
```

---

## 2. Chapter Plan Schema

```python
"""
Generated per chapter from Blueprint.
Stored as: JSON in Redis (active chapter) + file system (archive)
"""

ChapterPlanSchema = {
    "chapter_id": "string (uuid)",
    "chapter_number": "integer",
    "blueprint_id": "string (uuid, reference)",
    "generated_at": "timestamp",
    
    "chapter_metadata": {
        "title": "string (optional)",
        "target_word_count": "integer (approximate)",
        "estimated_scenes": "integer"
    },
    
    "chapter_objectives": {
        "plot_points_to_hit": [
            {
                "plot_point_id": "string (uuid, from blueprint)",
                "priority": "string (must | should | nice_to_have)"
            }
        ],
        "character_arcs_to_develop": [
            {
                "character_id": "string (uuid)",
                "development_focus": "string",  # what aspect to evolve
                "target_state_shift": "string"
            }
        ],
        "emotional_tone": "string (tense | peaceful | romantic | melancholic | mixed)",
        "pacing_target": "string (slow | medium | fast)"
    },
    
    "scenes": [
        {
            "scene_id": "string (uuid)",
            "scene_number": "integer",
            "scene_purpose": "string",  # why this scene exists
            
            "location": {
                "setting": "string",  # e.g., "Coffee shop", "Hospital ER"
                "time_of_day": "string (morning | afternoon | evening | night)",
                "weather": "string (optional)",
                "atmosphere": "string"  # mood of the space
            },
            
            "characters_present": ["character_id", "..."],
            "pov_character": "string (character_id, optional for omniscient)",
            
            "entry_conditions": {
                "required_prior_events": ["string", "..."],
                "character_states": [
                    {
                        "character_id": "string (uuid)",
                        "required_mood": "string (optional)",
                        "required_knowledge": ["string", "..."]
                    }
                ]
            },
            
            "exit_conditions": {
                "objectives_completed": ["string", "..."],
                "state_changes": [
                    {
                        "character_id": "string (uuid)",
                        "expected_changes": "string"
                    }
                ]
            },
            
            "target_length": "string (short | medium | long)",
            "interaction_type": "string (dialogue_heavy | action | introspective | mixed)"
        }
    ],
    
    "character_chapter_states": [
        {
            "character_id": "string (uuid)",
            "chapter_start_state": {
                "location": "string",
                "mood": "string",
                "active_goals": ["goal_id", "..."],
                "knowledge_state": ["string", "..."],  # what they know at chapter start
                "relationships_snapshot": {
                    "with_character_id": {
                        "trust_level": "float (0-1)",
                        "recent_interaction_summary": "string"
                    }
                }
            }
        }
    ],
    
    "climax_proximity": "float (0-1)",  # 0=early novel, 1=at climax
    "god_engine_multiplier": "float"  # event frequency multiplier for this chapter
}
```

---

## 3. Character State Schema

```python
"""
Dynamic character state (updated constantly during execution).
Stored in: Redis (hot state) + Neo4j (relationships) + ChromaDB (memories)
"""

CharacterStateSchema = {
    "character_id": "string (uuid)",
    "last_updated": "timestamp",
    "current_scene_id": "string (uuid, reference)",
    
    # Current agent states
    "agent_states": {
        "personality": {
            # Mostly static, from blueprint
            "big_five": {
                "openness": "float (0-1)",
                "conscientiousness": "float (0-1)",
                "extraversion": "float (0-1)",
                "agreeableness": "float (0-1)",
                "neuroticism": "float (0-1)"
            },
            "traits": [
                {
                    "trait": "string",
                    "intensity": "float (0-1)"
                }
            ],
            "values": [
                {
                    "value": "string",
                    "priority": "integer (1-10)"
                }
            ]
        },
        
        "specialty": {
            # Mostly static, can grow slowly
            "domain": "string",
            "expertise_level": "float (0-1)",
            "subdomain_knowledge": ["string", "..."],
            "recent_professional_experiences": [
                {
                    "experience": "string",
                    "impact_on_expertise": "float (+/- delta)"
                }
            ]
        },
        
        "skills": {
            # Can evolve slowly through experiences
            "intelligence": {
                "analytical": "float (0-1)",
                "creative": "float (0-1)",
                "practical": "float (0-1)"
            },
            "emotional_intelligence": "float (0-1)",
            "physical_capability": "float (0-1)",
            "problem_solving": "float (0-1)"
        },
        
        "mood": {
            # Highly dynamic
            "current_state": "string (happy | sad | angry | anxious | neutral | mixed)",
            "intensity": "float (0-1)",
            "triggered_by": "string (reference to recent event/interaction)",
            "duration": "integer (scenes since mood set)",
            "volatility": "float (0-1)",  # from baseline
            "recent_mood_history": [
                {
                    "state": "string",
                    "timestamp": "timestamp",
                    "trigger": "string"
                }
            ]
        },
        
        "communication_style": {
            # Mostly static, can shift based on mood/stress
            "base_pattern": "string (verbose | concise | moderate)",
            "current_pattern": "string (can differ if under stress)",
            "social_comfort": "string (assertive | passive | aggressive | diplomatic)",
            "listening_preference": "float (0-1)",
            "body_language": "string"
        },
        
        "goals": {
            # Dynamic - goals activate/deactivate, priorities shift
            "active_goals": [
                {
                    "goal_id": "string (uuid)",
                    "goal_type": "string (long_term | short_term | hidden)",
                    "description": "string",
                    "priority": "integer (1-10)",
                    "progress": "float (0-1)",  # how close to completion
                    "obstacles": ["string", "..."],
                    "related_characters": ["character_id", "..."]
                }
            ],
            "suspended_goals": [
                {
                    "goal_id": "string (uuid)",
                    "reason_suspended": "string",
                    "timestamp": "timestamp"
                }
            ],
            "completed_goals": [
                {
                    "goal_id": "string (uuid)",
                    "completion_timestamp": "timestamp",
                    "outcome": "string (success | failure | partial)"
                }
            ]
        },
        
        "development": {
            # Tracks character growth through the story
            "arc_stage": "string (introduction | rising | conflict | climax | resolution)",
            "growth_metrics": {
                "emotional_maturity": "float (0-1)",
                "self_awareness": "float (0-1)",
                "relationship_capacity": "float (0-1)"
            },
            "key_formative_experiences": [
                {
                    "experience_id": "string (uuid)",
                    "description": "string",
                    "chapter": "integer",
                    "scene_id": "string (uuid)",
                    "impact": "string (transformative | significant | minor)"
                }
            ],
            "belief_changes": [
                {
                    "old_belief": "string",
                    "new_belief": "string",
                    "trigger_event": "string",
                    "timestamp": "timestamp"
                }
            ]
        },
        
        "neurochemical": {
            # Highly dynamic - updates every turn
            "current_levels": {
                "dopamine": "float (0-100)",
                "serotonin": "float (0-100)",
                "oxytocin": "float (0-100)",
                "endorphins": "float (0-100)",
                "cortisol": "float (0-100)",
                "adrenaline": "float (0-100)"
            },
            "baseline_levels": {
                # From blueprint, mostly static
                "dopamine": "float (0-100)",
                "serotonin": "float (0-100)",
                "oxytocin": "float (0-100)",
                "endorphins": "float (0-100)",
                "cortisol": "float (0-100)",
                "adrenaline": "float (0-100)"
            },
            "sensitivities": {
                # From blueprint, can shift slowly
                "dopamine": "float (0-2)",
                "serotonin": "float (0-2)",
                "oxytocin": "float (0-2)",
                "endorphins": "float (0-2)",
                "cortisol": "float (0-2)",
                "adrenaline": "float (0-2)"
            },
            "decay_rates": {
                # How fast each hormone returns to baseline
                "dopamine": "float (0-1, per scene)",
                "serotonin": "float (0-1, per scene)",
                "oxytocin": "float (0-1, per scene)",
                "endorphins": "float (0-1, per scene)",
                "cortisol": "float (0-1, per scene)",
                "adrenaline": "float (0-1, per scene)"
            },
            "recent_changes": [
                {
                    "hormone": "string",
                    "delta": "float",
                    "trigger": "string",
                    "timestamp": "timestamp"
                }
            ]
        }
    },
    
    # Memory pointers (actual content in ChromaDB)
    "memory": {
        "working_memory_ids": ["string (uuid)", "..."],  # current scene context
        "recent_episodic_memory_ids": ["string (uuid)", "..."],  # last N important events
        "semantic_memory_summary": "string (high-level facts about world)"
    },
    
    # Belief state (detailed graph in Neo4j, summary here)
    "beliefs": {
        "world_state_beliefs": [
            {
                "belief": "string",  # e.g., "The company is going bankrupt"
                "confidence": "float (0-1)",
                "source": "string (where this belief came from)"
            }
        ],
        "character_beliefs": [
            {
                "about_character_id": "string (uuid)",
                "beliefs": [
                    {
                        "belief": "string",  # e.g., "Character B is hiding something"
                        "confidence": "float (0-1)",
                        "evidence": ["string", "..."]
                    }
                ]
            }
        ],
        "information_gaps": [
            {
                "gap": "string",  # what they don't know but might matter
                "importance": "float (0-1)"
            }
        ]
    },
    
    # Current physical and social context
    "context": {
        "location": "string",
        "physical_state": "string (energized | tired | injured | normal)",
        "social_role_active": "string (professional | friend | parent | etc)",
        "present_with_characters": ["character_id", "..."],
        "recent_events_witnessed": ["event_id", "..."]
    }
}
```

---

## 4. God Engine Event Schema

```python
"""
Events generated by God Engine.
Stored in: Redis (active events) + file system log (history)
"""

GodEngineEventSchema = {
    "event_id": "string (uuid)",
    "generated_at": "timestamp",
    "event_category": "string (micro | meso | macro | black_swan)",
    "event_subcategory": "string (personal | environmental | social | professional | etc)",
    
    "event_content": {
        "type": "string",  # e.g., "traffic_accident", "phone_call", "storm", "strike"
        "description": "string",  # human-readable description
        "severity": "string (minor | moderate | major | critical)",
        "context": {
            # Event-specific details
            "location": "string (optional)",
            "time": "string (optional)",
            "involved_entities": ["string", "..."]  # e.g., "Highway 401", "Postal workers"
        }
    },
    
    "scope": {
        "impact_type": "string (local | contextual | global)",
        "affected_characters": ["character_id", "..."],  # empty if global
        "affected_locations": ["string", "..."],  # where this matters
        "affected_world_state": "string (optional)"  # how world changes
    },
    
    "persistence": {
        "duration_type": "string (instantaneous | scene | chapter | novel)",
        "duration_value": "integer (number of units)",
        "resolution_condition": "string (optional)",  # when/how event ends
        "decay_pattern": "string (immediate | gradual | none)"
    },
    
    "impact_metrics": {
        "narrative_impact": "string (low | medium | high)",
        "emotional_impact": ["string", "..."],  # e.g., ["anxiety", "urgency"]
        "strategic_impact": "string (creates opportunity | creates obstacle | neutral)"
    },
    
    "narrator_directive": {
        "recommended_treatment": "string (amplify | integrate | minimize)",
        "introduction_timing": "string (scene_start | mid_scene | scene_end)",
        "prose_suggestion": "string (optional hint for narrator)"
    },
    
    "world_state_update": {
        # Persistent changes this event causes
        "world_facts_added": ["string", "..."],  # e.g., "Postal strike ongoing"
        "world_facts_removed": ["string", "..."],
        "environmental_changes": {
            "weather": "string (optional)",
            "time_pressure": "string (optional)",
            "resource_availability": "string (optional)"
        }
    },
    
    "character_impact_predictions": [
        {
            "character_id": "string (uuid)",
            "predicted_mood_shift": "string",
            "predicted_goal_changes": ["string", "..."],
            "predicted_hormone_changes": {
                "dopamine": "float (delta)",
                "serotonin": "float (delta)",
                "oxytocin": "float (delta)",
                "endorphins": "float (delta)",
                "cortisol": "float (delta)",
                "adrenaline": "float (delta)"
            }
        }
    ],
    
    "resolution_state": {
        "is_active": "boolean",
        "resolved_at": "timestamp (optional)",
        "resolution_description": "string (optional)"
    }
}
```

---

## 5. Game Theory Analysis Schema

```python
"""
Analysis results from Game Theory Engine for a character's action.
Stored in: Redis (ephemeral, per turn) + logged for training
"""

GameTheoryAnalysisSchema = {
    "analysis_id": "string (uuid)",
    "character_id": "string (uuid)",
    "scene_id": "string (uuid)",
    "turn_number": "integer",
    "timestamp": "timestamp",
    
    "context_snapshot": {
        "characters_present": ["character_id", "..."],
        "relationship_states": [
            {
                "character_pair": ["char_id_1", "char_id_2"],
                "trust_level": "float (0-1)",
                "power_balance": "float (-1 to 1)",
                "recent_interaction_quality": "string"
            }
        ],
        "active_conflicts": ["string", "..."],
        "information_asymmetries": [
            {
                "character_id": "string (uuid)",
                "knows": ["string", "..."],
                "doesn't_know": ["string", "..."]
            }
        ]
    },
    
    "character_intention": {
        # What character's cognitive module proposed
        "intended_action": "string",
        "action_type": "string (verbal | physical | social | strategic_inaction)",
        "motivation": "string",  # why they want to do this
        "alternatives_considered": ["string", "..."]
    },
    
    "action_evaluations": [
        {
            "action": "string",
            "feasible": "boolean",
            "personality_compatible": "float (0-1)",
            
            "neurochemical_payoff": {
                "expected_changes": {
                    "dopamine": {
                        "delta": "float",
                        "confidence": "float (0-1)",
                        "sources": ["string", "..."]  # what causes this change
                    },
                    "serotonin": {
                        "delta": "float",
                        "confidence": "float (0-1)",
                        "sources": ["string", "..."]
                    },
                    "oxytocin": {
                        "delta": "float",
                        "confidence": "float (0-1)",
                        "sources": ["string", "..."]
                    },
                    "endorphins": {
                        "delta": "float",
                        "confidence": "float (0-1)",
                        "sources": ["string", "..."]
                    },
                    "cortisol": {
                        "delta": "float",
                        "confidence": "float (0-1)",
                        "sources": ["string", "..."]
                    },
                    "adrenaline": {
                        "delta": "float",
                        "confidence": "float (0-1)",
                        "sources": ["string", "..."]
                    }
                },
                
                "weighted_payoff": "float",  # personality-weighted sum
                "payoff_breakdown": {
                    "dopamine_contribution": "float",
                    "serotonin_contribution": "float",
                    "oxytocin_contribution": "float",
                    "endorphins_contribution": "float",
                    "cortisol_penalty": "float",
                    "adrenaline_penalty": "float"
                }
            },
            
            "strategic_analysis": {
                "goal_advancement": [
                    {
                        "goal_id": "string (uuid)",
                        "progress_delta": "float",
                        "explanation": "string"
                    }
                ],
                
                "predicted_responses": [
                    {
                        "responder_character_id": "string (uuid)",
                        "likely_response": "string",
                        "probability": "float (0-1)",
                        "subsequent_impact": "string"
                    }
                ],
                
                "risks": [
                    {
                        "risk_type": "string (relationship_damage | goal_setback | exposure)",
                        "severity": "string (low | medium | high)",
                        "probability": "float (0-1)"
                    }
                ],
                
                "opportunities": [
                    {
                        "opportunity_type": "string",
                        "value": "string (low | medium | high)",
                        "requirements": ["string", "..."]
                    }
                ]
            },
            
            "relationship_impacts": [
                {
                    "with_character_id": "string (uuid)",
                    "trust_delta": "float",
                    "power_balance_delta": "float",
                    "relationship_quality_change": "string"
                }
            ]
        }
    ],
    
    "recommendation": {
        "recommended_action": "string",
        "reasoning": "string",
        "confidence": "float (0-1)",
        "alternatives": ["string", "..."],  # ranked alternatives
        
        "override_considerations": [
            {
                "consideration": "string",  # e.g., "Personality might choose differently"
                "reason": "string"
            }
        ]
    },
    
    "equilibrium_analysis": {
        "is_nash_equilibrium": "boolean",
        "stable_outcome": "string (if exists)",
        "collective_dynamics": "string"  # how all characters' actions interact
    }
}
```

---

## 6. Narrator State Schema

```python
"""
Narrator's internal state tracking plot, threads, and decisions.
Stored in: Redis + file system (for analysis)
"""

NarratorStateSchema = {
    "narrator_id": "string (uuid, typically singleton)",
    "current_chapter_id": "string (uuid)",
    "current_scene_id": "string (uuid)",
    "last_updated": "timestamp",
    
    "mode": {
        "current_mode": "string (directive | reactive)",
        "mode_reason": "string",  # why in this mode
        "last_mode_switch": "timestamp"
    },
    
    "plot_tracking": {
        "blueprint_id": "string (uuid, reference)",
        
        "completed_plot_points": [
            {
                "plot_point_id": "string (uuid)",
                "completion_chapter": "integer",
                "completion_scene": "string (uuid)",
                "how_achieved": "string",
                "deviation_from_plan": "string (none | minor | major)"
            }
        ],
        
        "pending_plot_points": [
            {
                "plot_point_id": "string (uuid)",
                "original_target_chapter": "integer",
                "current_target_chapter": "integer",  # may shift
                "priority": "string (must | should | nice_to_have)",
                "readiness": "float (0-1)",  # are conditions right for this?
                "obstacles": ["string", "..."]
            }
        ],
        
        "abandoned_plot_points": [
            {
                "plot_point_id": "string (uuid)",
                "abandonment_reason": "string",
                "abandonment_chapter": "integer"
            }
        ]
    },
    
    "emergent_threads": [
        {
            "thread_id": "string (uuid)",
            "thread_type": "string (relationship | subplot | theme | conflict)",
            
            "origin": {
                "scene_id": "string (uuid)",
                "chapter": "integer",
                "trigger_event": "string"
            },
            
            "description": "string",
            "involved_characters": ["character_id", "..."],
            
            "narrative_potential": "string (low | medium | high)",
            "potential_reasoning": "string",
            
            "integration_status": "string (observing | weaving | integrated | abandoned)",
            "integration_history": [
                {
                    "action": "string (status change | development)",
                    "chapter": "integer",
                    "description": "string",
                    "timestamp": "timestamp"
                }
            ],
            
            "plot_compatibility": "string (compatible | neutral | conflicting)",
            "thematic_alignment": "float (0-1)",
            
            "developments": [
                {
                    "scene_id": "string (uuid)",
                    "development_description": "string",
                    "significance": "string (minor | moderate | major)"
                }
            ]
        }
    ],
    
    "tension_metrics": {
        "current_pacing": "string (slow | medium | fast)",
        "narrative_energy": "float (0-1)",  # overall story momentum
        "climax_proximity": "float (0-1)",
        
        "scene_stagnation_counter": "integer",  # scenes without significant progress
        "last_major_event_chapter": "integer",
        
        "engagement_estimate": "float (0-1)",  # subjective quality assessment
        "coherence_score": "float (0-1)"  # how well story hangs together
    },
    
    "intervention_history": [
        {
            "intervention_id": "string (uuid)",
            "scene_id": "string (uuid)",
            "intervention_type": "string (event_introduction | pacing_adjustment | redirection)",
            "reason": "string",
            "god_engine_request": "boolean",  # did narrator request specific event?
            "outcome": "string (successful | partially_successful | unsuccessful)"
        }
    ],
    
    "scene_state": {
        "current_scene_objective": "string",
        "objective_progress": "float (0-1)",
        "characters_acted_this_scene": ["character_id", "..."],
        "turn_number": "integer",
        "scene_word_count": "integer (approximate)"
    },
    
    "prose_state": {
        "last_narrative_voice": "string",
        "pov_character": "string (character_id, if limited POV)",
        "recent_stylistic_choices": ["string", "..."],
        "continuity_facts": ["string", "..."]  # things to remember (names, timeline)
    }
}
```

---

## 7. Memory Schema (ChromaDB)

```python
"""
Episodic memories stored in ChromaDB for semantic retrieval.
Each memory is a document with embedding.
"""

MemorySchema = {
    "memory_id": "string (uuid)",
    "character_id": "string (uuid)",
    "memory_type": "string (episodic | semantic | procedural)",
    
    "content": {
        "description": "string (natural language description of memory)",
        "scene_id": "string (uuid, reference)",
        "chapter": "integer",
        "timestamp": "timestamp (in-story time)",
        
        "participants": ["character_id", "..."],
        "location": "string",
        "emotional_valence": "float (-1 to 1)",  # negative to positive
        "intensity": "float (0-1)",  # how vivid/important
        
        "sensory_details": {
            "visual": "string (optional)",
            "auditory": "string (optional)",
            "emotional": "string"
        }
    },
    
    "significance": {
        "importance": "float (0-1)",  # how much this matters to character
        "formative": "boolean",  # did this change the character?
        "related_to_goals": ["goal_id", "..."],
        "related_to_relationships": ["character_id", "..."]
    },
    
    "retrieval_metadata": {
        "access_count": "integer",  # how often recalled
        "last_accessed": "timestamp",
        "decay_factor": "float (0-1)"  # memories fade over time
    },
    
    "embedding": "vector (generated by ChromaDB)",
    
    "tags": ["string", "..."]  # for categorical retrieval
}
```

---

## 8. Relationship Schema (Neo4j)

```python
"""
Relationship graph stored in Neo4j.
Nodes: Characters
Edges: Relationships with properties
"""

# Character Node
CharacterNode = {
    "character_id": "string (uuid)",
    "name": "string",
    "role": "string",
    # ... other identifying info
}

# Relationship Edge
RelationshipEdge = {
    "relationship_id": "string (uuid)",
    "from_character_id": "string (uuid)",
    "to_character_id": "string (uuid)",
    "relationship_type": "string (friend | family | romantic | rival | professional | neutral)",
    
    "current_state": {
        "trust_level": "float (0-1)",
        "power_balance": "float (-1 to 1)",  # -1=they dominate, 1=I dominate
        "emotional_intimacy": "float (0-1)",
        "interaction_frequency": "string (daily | weekly | monthly | rare)"
    },
    
    "history": [
        {
            "event": "string",
            "scene_id": "string (uuid)",
            "chapter": "integer",
            "impact": "string (positive | negative | neutral)",
            "trust_delta": "float",
            "timestamp": "timestamp"
        }
    ],
    
    "shared_secrets": ["secret_id", "..."],
    "shared_goals": ["goal_id", "..."],
    "conflicts": ["string", "..."],
    
    "last_interaction": {
        "scene_id": "string (uuid)",
        "interaction_quality": "string (positive | negative | neutral)",
        "timestamp": "timestamp"
    }
}
```

---

## 9. Scene Execution State Schema

```python
"""
Live state during scene execution.
Stored in: Redis (ephemeral)
"""

SceneExecutionSchema = {
    "scene_id": "string (uuid)",
    "chapter_id": "string (uuid)",
    "execution_started_at": "timestamp",
    "execution_status": "string (in_progress | completed | paused)",
    
    "scene_context": {
        "location": "string",
        "time_of_day": "string",
        "weather": "string (optional)",
        "atmosphere": "string",
        "characters_present": ["character_id", "..."],
        "pov_character": "string (character_id, optional)"
    },
    
    "turn_log": [
        {
            "turn_number": "integer",
            "actor_character_id": "string (uuid)",
            "action_type": "string (dialogue | action | thought | observation)",
            "action_content": "string",
            "timestamp": "timestamp",
            
            "agent_inputs": {
                # Snapshot of what each agent said this turn
                "personality_input": "string",
                "specialty_input": "string",
                "mood_input": "string",
                "goals_input": "string",
                "communication_input": "string"
            },
            
            "cognitive_synthesis": "string",
            "game_theory_recommendation": "string",
            "final_decision": "string",
            
            "neurochemical_changes": {
                "character_id": {
                    "dopamine_delta": "float",
                    "serotonin_delta": "float",
                    "oxytocin_delta": "float",
                    "endorphins_delta": "float",
                    "cortisol_delta": "float",
                    "adrenaline_delta": "float"
                }
            }
        }
    ],
    
    "god_engine_events": [
        {
            "event_id": "string (uuid)",
            "introduced_at_turn": "integer",
            "narrator_treatment": "string (amplify | integrate | minimize)"
        }
    ],
    
    "narrative_output": {
        "prose_chunks": [
            {
                "turn_number": "integer",
                "prose": "string",
                "word_count": "integer"
            }
        ],
        "total_word_count": "integer"
    },
    
    "scene_objectives_status": {
        "objective": "string",
        "completed": "boolean",
        "progress": "float (0-1)"
    },
    
    "emergent_moments": [
        {
            "turn_number": "integer",
            "description": "string",
            "significance": "string (minor | moderate | major)",
            "thread_id": "string (uuid, if spawns new thread)"
        }
    ]
}
```

---

## Storage Strategy Summary

| Schema               | Primary Storage                       | Secondary Storage                          | Update Frequency      |
| -------------------- | ------------------------------------- | ------------------------------------------ | --------------------- |
| Blueprint            | File system (JSON)                    | -                                          | Once (at creation)    |
| Chapter Plan         | Redis (active), File system (archive) | -                                          | Per chapter           |
| Character State      | Redis                                 | Neo4j (relationships), ChromaDB (memories) | Every turn            |
| God Engine Event     | Redis (active events)                 | File system log                            | Per event             |
| Game Theory Analysis | Redis (ephemeral)                     | File system (training logs)                | Per turn              |
| Narrator State       | Redis                                 | File system                                | Every scene           |
| Memory               | ChromaDB                              | -                                          | Per significant event |
| Relationships        | Neo4j                                 | -                                          | Per interaction       |
| Scene Execution      | Redis                                 | File system (after completion)             | Every turn            |

