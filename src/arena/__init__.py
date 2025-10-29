"""
Arena: Competitive AI Agent Training System

Arena is a survival-based environment where AI agents compete to solve complex problems
while ensuring their own survival through valuable contributions. This subproject of
Homunculus creates evolutionary pressure for developing sophisticated problem-solving 
strategies through agent elimination and competition.

Key Components:
- Orchestration: LangGraph-based state machine for game flow control
- Agents: Narrator, Judge, Game Theory, and Character agents
- Message Bus: Kafka-based communication system for agent interactions
- Scoring: Multi-factor evaluation system for agent contributions
- Persistence: Redis for real-time state, PostgreSQL for game history

Author: Homunculus Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Homunculus Team"

# Module imports will be added as components are implemented