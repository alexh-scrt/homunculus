"""
CLI Utility Functions

This module provides utility functions for the Arena CLI,
including formatting, display, and user interaction helpers.

Author: Homunculus Team
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import shutil
import time


# ANSI color codes
COLORS = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "gold": "\033[93m",
    "silver": "\033[37m",
    "bronze": "\033[33m",
    "reset": "\033[0m",
    "bold": "\033[1m",
    "underline": "\033[4m"
}


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Log level
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def colored_text(text: str, color: str, bold: bool = False) -> str:
    """
    Return colored text for terminal output.
    
    Args:
        text: Text to color
        color: Color name
        bold: Whether to make text bold
        
    Returns:
        Colored text string
    """
    if not sys.stdout.isatty():
        return text
    
    color_code = COLORS.get(color, "")
    bold_code = COLORS["bold"] if bold else ""
    reset_code = COLORS["reset"]
    
    return f"{bold_code}{color_code}{text}{reset_code}"


def print_banner() -> None:
    """Print Arena CLI banner."""
    banner = r"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     █████╗ ██████╗ ███████╗███╗   ██╗ █████╗            ║
    ║    ██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗           ║
    ║    ███████║██████╔╝█████╗  ██╔██╗ ██║███████║           ║
    ║    ██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║██╔══██║           ║
    ║    ██║  ██║██║  ██║███████╗██║ ╚████║██║  ██║           ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝           ║
    ║                                                           ║
    ║         Competitive AI Agent Training System              ║
    ║                    Version 0.1.0                          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(colored_text(banner, "cyan", bold=True))


def print_table(data: List[Dict[str, Any]], headers: Optional[List[str]] = None) -> None:
    """
    Print data as formatted table.
    
    Args:
        data: List of dictionaries
        headers: Optional custom headers
    """
    if not data:
        print("No data to display")
        return
    
    # Get headers
    if not headers:
        headers = list(data[0].keys())
    
    # Calculate column widths
    widths = {}
    for header in headers:
        widths[header] = len(str(header))
        for row in data:
            value = str(row.get(header, ""))
            widths[header] = max(widths[header], len(value))
    
    # Print header
    header_line = " | ".join(
        str(header).ljust(widths[header])
        for header in headers
    )
    print("\n" + colored_text(header_line, "cyan", bold=True))
    print("-" * len(header_line))
    
    # Print rows
    for row in data:
        row_line = " | ".join(
            str(row.get(header, "")).ljust(widths[header])
            for header in headers
        )
        print(row_line)
    
    print()


def confirm_action(prompt: str, default: bool = False) -> bool:
    """
    Ask user for confirmation.
    
    Args:
        prompt: Confirmation prompt
        default: Default answer
        
    Returns:
        User's confirmation
    """
    if default:
        prompt += " [Y/n]: "
    else:
        prompt += " [y/N]: "
    
    while True:
        answer = input(colored_text(prompt, "yellow")).strip().lower()
        
        if not answer:
            return default
        
        if answer in ["y", "yes"]:
            return True
        elif answer in ["n", "no"]:
            return False
        else:
            print("Please answer 'yes' or 'no'")


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def progress_bar(
    current: int,
    total: int,
    width: int = 50,
    prefix: str = "",
    suffix: str = ""
) -> None:
    """
    Display a progress bar.
    
    Args:
        current: Current progress
        total: Total items
        width: Bar width
        prefix: Prefix text
        suffix: Suffix text
    """
    if total == 0:
        percent = 100.0
    else:
        percent = (current / total) * 100
    
    filled = int(width * current // max(total, 1))
    bar = "█" * filled + "░" * (width - filled)
    
    # Clear line and print progress
    sys.stdout.write('\r')
    sys.stdout.write(f"{prefix} [{bar}] {percent:.1f}% {suffix}")
    sys.stdout.flush()
    
    if current >= total:
        sys.stdout.write('\n')


def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_terminal_size() -> tuple:
    """
    Get terminal size.
    
    Returns:
        (columns, rows) tuple
    """
    return shutil.get_terminal_size((80, 24))


def print_error(message: str) -> None:
    """
    Print error message.
    
    Args:
        message: Error message
    """
    print(colored_text(f"✗ Error: {message}", "red", bold=True))


def print_success(message: str) -> None:
    """
    Print success message.
    
    Args:
        message: Success message
    """
    print(colored_text(f"✓ {message}", "green", bold=True))


def print_warning(message: str) -> None:
    """
    Print warning message.
    
    Args:
        message: Warning message
    """
    print(colored_text(f"⚠ Warning: {message}", "yellow"))


def print_info(message: str) -> None:
    """
    Print info message.
    
    Args:
        message: Info message
    """
    print(colored_text(f"ℹ {message}", "blue"))


def format_list(items: List[str], bullet: str = "•") -> str:
    """
    Format list with bullets.
    
    Args:
        items: List items
        bullet: Bullet character
        
    Returns:
        Formatted list string
    """
    return "\n".join(f"  {bullet} {item}" for item in items)


def truncate_text(text: str, max_length: int = 80, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_timestamp(timestamp: datetime) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: Datetime object
        
    Returns:
        Formatted timestamp
    """
    now = datetime.now()
    diff = now - timestamp
    
    if diff.days > 7:
        return timestamp.strftime("%Y-%m-%d")
    elif diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds > 3600:
        return f"{diff.seconds // 3600}h ago"
    elif diff.seconds > 60:
        return f"{diff.seconds // 60}m ago"
    else:
        return "just now"


def animated_spinner(duration: float = 2.0, message: str = "Processing") -> None:
    """
    Show animated spinner.
    
    Args:
        duration: Duration to show spinner
        message: Message to display
    """
    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    start_time = time.time()
    
    while time.time() - start_time < duration:
        for char in spinner_chars:
            sys.stdout.write(f"\r{char} {message}...")
            sys.stdout.flush()
            time.sleep(0.1)
            
            if time.time() - start_time >= duration:
                break
    
    sys.stdout.write("\r" + " " * (len(message) + 10) + "\r")
    sys.stdout.flush()


def format_score(score: float, precision: int = 2) -> str:
    """
    Format score for display.
    
    Args:
        score: Score value
        precision: Decimal precision
        
    Returns:
        Formatted score
    """
    formatted = f"{score:.{precision}f}"
    
    # Color based on value
    if score >= 100:
        return colored_text(formatted, "gold", bold=True)
    elif score >= 80:
        return colored_text(formatted, "green")
    elif score >= 60:
        return colored_text(formatted, "yellow")
    else:
        return colored_text(formatted, "red")


def create_ascii_chart(
    data: List[float],
    width: int = 60,
    height: int = 10,
    title: Optional[str] = None
) -> str:
    """
    Create simple ASCII chart.
    
    Args:
        data: Data points
        width: Chart width
        height: Chart height
        title: Chart title
        
    Returns:
        ASCII chart string
    """
    if not data:
        return "No data to display"
    
    # Normalize data
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val != min_val else 1
    
    normalized = [(v - min_val) / range_val * height for v in data]
    
    # Create chart
    chart = []
    
    if title:
        chart.append(title.center(width))
        chart.append("")
    
    # Build rows from top to bottom
    for row in range(height, -1, -1):
        line = []
        for i, val in enumerate(normalized):
            if val >= row:
                line.append("█")
            else:
                line.append(" ")
        
        # Add axis
        if row == height:
            label = f"{max_val:.1f}"
        elif row == 0:
            label = f"{min_val:.1f}"
        else:
            label = ""
        
        chart.append(f"{label:>6} |{''.join(line)}")
    
    # Add bottom axis
    chart.append(" " * 7 + "+" + "-" * len(data))
    
    return "\n".join(chart)


def format_game_state(state: Dict[str, Any]) -> str:
    """
    Format game state for display.
    
    Args:
        state: Game state dictionary
        
    Returns:
        Formatted state string
    """
    lines = []
    lines.append(colored_text("═" * 60, "cyan"))
    lines.append(colored_text(f"Game State - Turn {state.get('current_turn', 0)}", "cyan", bold=True))
    lines.append(colored_text("═" * 60, "cyan"))
    
    # Phase and status
    lines.append(f"Phase: {state.get('phase', 'Unknown')}")
    lines.append(f"Active Agents: {len(state.get('active_agents', []))}")
    
    # Scores
    scores = state.get('scores', {})
    if scores:
        lines.append("\nTop Scores:")
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for agent, score in sorted_scores[:5]:
            lines.append(f"  {agent}: {format_score(score)}")
    
    # Recent events
    events = state.get('recent_events', [])
    if events:
        lines.append("\nRecent Events:")
        for event in events[:3]:
            lines.append(f"  • {event}")
    
    return "\n".join(lines)