"""
Arena Configuration Management

Configuration class for Homunculus Arena system, following the pattern
from the talks project for configurable recursion limits and other settings.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass 
class ArenaSystemConfig:
    """System configuration class for Homunculus Arena"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from arena.yml file"""
        # Look for config file in project root and common locations
        config_paths = [
            Path("arena.yml"),
            Path(__file__).parent.parent.parent.parent / "arena.yml",
            Path.cwd() / "arena.yml",
            Path(__file__).parent / "arena.yml"
        ]
        
        config_file = None
        for path in config_paths:
            if path.exists():
                config_file = path
                break
        
        if not config_file:
            # Use default configuration if no file found
            self._config = self._get_default_config()
            return
        
        try:
            with open(config_file, 'r') as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_file}: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "orchestration": {
                "recursion_limit": 250,
                "max_turns": 100,
                "min_agents": 2,
                "checkpoint_frequency": 5,
                "enable_recovery": True,
                "parallel_execution": True,
                "timeout_seconds": 300,
                "retry_attempts": 3
            },
            "termination": {
                "safety_turn_limit": 50,
                "phase_transition_limits": {
                    "early": 20,
                    "mid": 50,
                    "late": 80
                },
                "auto_terminate": {
                    "max_turns_reached": True,
                    "single_survivor": True,
                    "all_eliminated": True,
                    "safety_limit_reached": True
                }
            },
            "elimination": {
                "enabled": True,
                "min_turn_threshold": 20,
                "frequency": 10,
                "rate": 0.2
            },
            "scoring": {
                "weights": {
                    "novelty": 0.25,
                    "builds_on_others": 0.20,
                    "solves_subproblem": 0.25,
                    "radical_idea": 0.15,
                    "manipulation": 0.15
                }
            },
            "logging": {
                "level": "INFO",
                "event_streaming": True,
                "performance_metrics": True,
                "game_statistics": True
            }
        }
    
    # Orchestration settings
    @property
    def recursion_limit(self) -> int:
        """Get LangGraph recursion limit"""
        # Check environment variable first, then config file, then default
        env_limit = os.getenv("LANGGRAPH_RECURSION_LIMIT")
        if env_limit is not None:
            try:
                return int(env_limit)
            except ValueError:
                pass  # Fall back to config file/default
        return self.get("orchestration.recursion_limit", 250)
    
    @property
    def max_turns(self) -> int:
        """Get maximum number of turns per game"""
        return self.get("orchestration.max_turns", 100)
    
    @property
    def min_agents(self) -> int:
        """Get minimum number of agents required"""
        return self.get("orchestration.min_agents", 2)
    
    @property
    def checkpoint_frequency(self) -> int:
        """Get checkpoint frequency (turns)"""
        return self.get("orchestration.checkpoint_frequency", 5)
    
    @property
    def enable_recovery(self) -> bool:
        """Check if game state recovery is enabled"""
        return self.get("orchestration.enable_recovery", True)
    
    @property
    def parallel_execution(self) -> bool:
        """Check if parallel agent execution is enabled"""
        return self.get("orchestration.parallel_execution", True)
    
    @property
    def timeout_seconds(self) -> int:
        """Get game timeout in seconds"""
        return self.get("orchestration.timeout_seconds", 300)
    
    # Termination settings
    @property
    def safety_turn_limit(self) -> int:
        """Get safety turn limit to prevent infinite games"""
        return self.get("termination.safety_turn_limit", 50)
    
    @property
    def early_phase_limit(self) -> int:
        """Get turn limit for early phase"""
        return self.get("termination.phase_transition_limits.early", 20)
    
    @property
    def mid_phase_limit(self) -> int:
        """Get turn limit for mid phase"""
        return self.get("termination.phase_transition_limits.mid", 50)
    
    @property
    def late_phase_limit(self) -> int:
        """Get turn limit for late phase"""
        return self.get("termination.phase_transition_limits.late", 80)
    
    # Elimination settings
    @property
    def elimination_enabled(self) -> bool:
        """Check if elimination is enabled"""
        return self.get("elimination.enabled", True)
    
    @property
    def elimination_min_turns(self) -> int:
        """Get minimum turns before elimination can occur"""
        return self.get("elimination.min_turn_threshold", 20)
    
    @property
    def elimination_frequency(self) -> int:
        """Get elimination check frequency"""
        return self.get("elimination.frequency", 10)
    
    @property
    def elimination_rate(self) -> float:
        """Get elimination rate"""
        return self.get("elimination.rate", 0.2)
    
    # Scoring weights
    @property
    def scoring_weights(self) -> Dict[str, float]:
        """Get scoring weights"""
        return self.get("scoring.weights", {
            "novelty": 0.25,
            "builds_on_others": 0.20,
            "solves_subproblem": 0.25,
            "radical_idea": 0.15,
            "manipulation": 0.15
        })
    
    # Logging settings
    @property
    def log_level(self) -> str:
        """Get logging level"""
        return self.get("logging.level", "INFO")
    
    @property
    def event_streaming_enabled(self) -> bool:
        """Check if event streaming is enabled"""
        return self.get("logging.event_streaming", True)
    
    @property
    def performance_metrics_enabled(self) -> bool:
        """Check if performance metrics are enabled"""
        return self.get("logging.performance_metrics", True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dots)"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()


# Singleton instance
arena_config = ArenaSystemConfig()