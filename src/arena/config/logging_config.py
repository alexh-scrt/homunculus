"""
Arena Logging Configuration

Configures logging for Arena games to provide:
1. Clean console output showing only "Agent Name: <response>"
2. Detailed file logging to logs/game_id_timestamp.log

Author: Homunculus Team
"""

import os
import logging
import logging.handlers
import sys
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, AsyncGenerator
from pathlib import Path


# ANSI color codes for agent names
AGENT_COLORS = {
    'Ada Lovelace': '\033[96m',      # Cyan
    'Captain Cosmos': '\033[95m',    # Magenta
    'Tech Enthusiast': '\033[92m',   # Green
    'Zen Master': '\033[93m',        # Yellow
    'Grumpy Wizard': '\033[91m',     # Red
    'Friendly Teacher': '\033[94m',  # Blue
    'Creative Artist': '\033[97m',   # White
    # Fallback for IDs
    'ada_lovelace': '\033[96m',      # Cyan
    'captain_cosmos': '\033[95m',    # Magenta
    'tech_enthusiast': '\033[92m',   # Green
    'zen_master': '\033[93m',        # Yellow
    'grumpy_wizard': '\033[91m',     # Red
    'friendly_teacher': '\033[94m',  # Blue
    'creative_artist': '\033[97m',   # White
}
RESET_COLOR = '\033[0m'


def get_agent_color(agent_name: str) -> str:
    """Get color code for agent name."""
    # Try exact match first
    if agent_name in AGENT_COLORS:
        return AGENT_COLORS[agent_name]
    
    # Try case-insensitive match
    for key, color in AGENT_COLORS.items():
        if key.lower() == agent_name.lower():
            return color
    
    # Try partial match
    for key, color in AGENT_COLORS.items():
        if key.lower() in agent_name.lower() or agent_name.lower() in key.lower():
            return color
    
    # Fallback: use hash of name for consistent color
    colors = ['\033[96m', '\033[95m', '\033[92m', '\033[93m', '\033[94m', '\033[97m']
    return colors[hash(agent_name.lower()) % len(colors)]


class ArenaConsoleFormatter(logging.Formatter):
    """Clean console formatter that shows only agent responses."""
    
    def __init__(self):
        super().__init__()
        self.last_was_agent = False
    
    def format(self, record):
        # Only show clean agent responses in console
        if hasattr(record, 'agent_name') and hasattr(record, 'agent_response'):
            # Add extra spacing between agent outputs
            spacing = "\n\n" if self.last_was_agent else ""
            self.last_was_agent = True
            
            # Color the agent name only
            agent_name = record.agent_name
            color = get_agent_color(agent_name)
            colored_name = f"{color}{agent_name}{RESET_COLOR}"
            
            # Process agent response to handle newlines properly for paragraph breaks
            processed_response = record.agent_response.replace('\\n', '\n')
            return f"{spacing}{colored_name}: {processed_response}\n\n---"
        
        # For non-agent messages, show minimal info
        if record.levelno >= logging.WARNING:
            self.last_was_agent = False
            return f"[{record.levelname}] {record.getMessage()}"
        
        # This should never be reached due to filter, but just in case
        self.last_was_agent = False
        return f"{record.getMessage()}"


class ArenaFileFormatter(logging.Formatter):
    """Detailed file formatter with timestamps and full context."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class ArenaLogFilter(logging.Filter):
    """Filter to hide verbose log messages from console."""
    
    def filter(self, record):
        # Allow agent responses through
        if hasattr(record, 'agent_name') and hasattr(record, 'agent_response'):
            return True
            
        # Allow warnings and errors
        if record.levelno >= logging.WARNING:
            return True
            
        # Hide all other messages in console (they go to file)
        return False


class ArenaLogger:
    """Arena-specific logger that manages game logging."""
    
    def __init__(self, game_id: str, log_dir: Optional[str] = None):
        """
        Initialize Arena logger.
        
        Args:
            game_id: Unique game identifier
            log_dir: Directory for log files (default: logs/)
        """
        self.game_id = game_id
        self.log_dir = Path(log_dir or "logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{game_id}_{timestamp}.log"
        
        # Create agent-specific logger first
        self.logger = logging.getLogger(f"arena.game.{game_id}")
        
        # Streaming configuration
        self.streaming_enabled = os.getenv("ENABLE_STREAMING_RESPONSE", "false").lower() == "true"
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up file and console logging."""
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler with clean formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = ArenaConsoleFormatter()
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(ArenaLogFilter())
        root_logger.addHandler(console_handler)
        
        # File handler with detailed logging
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = ArenaFileFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        self.logger.info(f"Arena logging initialized - Game: {self.game_id}")
        self.logger.info(f"Log file: {self.log_file}")
    
    def log_agent_response(self, agent_name: str, response: str, agent_id: Optional[str] = None):
        """
        Log an agent response with clean console output.
        
        Args:
            agent_name: Name of the agent
            response: Agent's response text
            agent_id: Optional agent ID for file logging
        """
        # Create a log record with agent info
        record = logging.LogRecord(
            name=f"arena.agent.{agent_id or agent_name.lower()}",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"Agent response: {response}",
            args=(),
            exc_info=None
        )
        
        # Add custom attributes for console formatting
        record.agent_name = agent_name
        record.agent_response = response
        record.agent_id = agent_id
        
        # Log the record
        self.logger.handle(record)
    
    def log_agent_response_streaming(
        self, 
        agent_name: str, 
        response_stream, 
        agent_id: Optional[str] = None
    ) -> str:
        """
        Log an agent response with streaming console output.
        
        Args:
            agent_name: Name of the agent
            response_stream: Stream of response tokens (sync generator)
            agent_id: Optional agent ID for file logging
            
        Returns:
            Complete accumulated response
        """
        if not self.streaming_enabled:
            # If streaming is disabled, collect all tokens first
            complete_response = ""
            for token in response_stream:
                complete_response += token
            self.log_agent_response(agent_name, complete_response, agent_id)
            return complete_response
        
        # Streaming output
        accumulated_response = ""
        first_token = True
        
        # Start the agent name with color and spacing
        console_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)]
        if console_handlers:
            formatter = console_handlers[0].formatter
            if hasattr(formatter, 'last_was_agent') and formatter.last_was_agent:
                # Add extra spacing before agent name
                sys.stdout.write("\n\n")
                sys.stdout.flush()
            
            # Write colored agent name
            color = get_agent_color(agent_name)
            colored_name = f"{color}{agent_name}{RESET_COLOR}"
            sys.stdout.write(f"{colored_name}: ")
            sys.stdout.flush()
            
            # Mark that we just output an agent
            formatter.last_was_agent = True
        
        # Stream the response tokens immediately (simple approach)
        try:
            for token in response_stream:
                if token:
                    accumulated_response += token
                    # Process token to handle newlines properly for paragraph breaks
                    processed_token = token.replace('\\n', '\n')
                    # Output token immediately
                    sys.stdout.write(processed_token)
                    sys.stdout.flush()
            
            # Add separator after complete response
            sys.stdout.write("\n\n---\n")
            sys.stdout.flush()
            
        except Exception as e:
            self.logger.error(f"Error during streaming output: {e}")
        
        # Log complete response to file
        if accumulated_response:
            # Create log record for file logging (without console output)
            record = logging.LogRecord(
                name=f"arena.agent.{agent_id or agent_name.lower()}",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"Agent response: {accumulated_response}",
                args=(),
                exc_info=None
            )
            
            # Add custom attributes but don't mark as agent (to avoid console output)
            record.agent_id = agent_id
            record.streaming_logged = True  # Mark as already output to console
            
            # Only log to file handlers
            for handler in self.logger.handlers:
                if not isinstance(handler, logging.StreamHandler):
                    handler.emit(record)
        
        return accumulated_response
    
    def log_game_event(self, event_type: str, message: str, **kwargs):
        """
        Log a game event with additional context.
        
        Args:
            event_type: Type of event (turn_start, scoring, elimination, etc.)
            message: Event message
            **kwargs: Additional context
        """
        context = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        full_message = f"[{event_type.upper()}] {message}"
        if context:
            full_message += f" | {context}"
        
        self.logger.info(full_message)
    
    def log_turn_start(self, turn: int, current_speaker: str, active_agents: list):
        """Log turn start information."""
        self.log_game_event(
            "turn_start",
            f"Turn {turn} - Speaker: {current_speaker}",
            turn=turn,
            speaker=current_speaker,
            active_agents=len(active_agents)
        )
    
    def log_scoring(self, agent_id: str, score: float, total_score: float):
        """Log scoring information."""
        self.log_game_event(
            "scoring",
            f"Agent {agent_id} scored {score:.3f} (total: {total_score:.3f})",
            agent=agent_id,
            score=score,
            total=total_score
        )
    
    def log_elimination(self, agent_id: str, reason: str):
        """Log agent elimination."""
        self.log_game_event(
            "elimination",
            f"Agent {agent_id} eliminated: {reason}",
            agent=agent_id,
            reason=reason
        )
    
    def log_game_end(self, winner: Optional[str], final_scores: Dict[str, float], total_turns: int):
        """Log game completion."""
        winner_text = f"Winner: {winner}" if winner else "No winner"
        self.log_game_event(
            "game_end",
            f"Game completed - {winner_text}",
            winner=winner,
            turns=total_turns,
            final_scores=final_scores
        )
    
    def get_log_file_path(self) -> Path:
        """Get the path to the current log file."""
        return self.log_file


def setup_arena_logging(game_id: str, log_dir: Optional[str] = None) -> ArenaLogger:
    """
    Set up Arena logging for a game session.
    
    Args:
        game_id: Unique game identifier
        log_dir: Optional log directory (default: logs/)
        
    Returns:
        ArenaLogger instance
    """
    return ArenaLogger(game_id, log_dir)


# Global logger instance for current game
_current_arena_logger: Optional[ArenaLogger] = None


def get_arena_logger() -> Optional[ArenaLogger]:
    """Get the current arena logger instance."""
    return _current_arena_logger


def set_arena_logger(logger: ArenaLogger):
    """Set the current arena logger instance."""
    global _current_arena_logger
    _current_arena_logger = logger


def log_agent_response(agent_name: str, response: str, agent_id: Optional[str] = None):
    """Convenience function to log agent response using current logger."""
    if _current_arena_logger:
        _current_arena_logger.log_agent_response(agent_name, response, agent_id)


def log_game_event(event_type: str, message: str, **kwargs):
    """Convenience function to log game event using current logger."""
    if _current_arena_logger:
        _current_arena_logger.log_game_event(event_type, message, **kwargs)