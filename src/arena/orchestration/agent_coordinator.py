"""
Agent Coordinator for Arena

This module manages agent coordination, scheduling, and parallel execution.

Features:
- Agent scheduling and ordering
- Parallel execution management
- Dependency resolution
- Resource allocation
- Execution monitoring

Author: Homunculus Team  
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from ..agents import BaseAgent
from ..models import Message

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Agent execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BATCH = "batch"
    PRIORITY = "priority"


class CoordinationStrategy(Enum):
    """Agent coordination strategies."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    PERFORMANCE_BASED = "performance_based"
    ADAPTIVE = "adaptive"


@dataclass
class AgentTask:
    """Task for an agent to execute."""
    agent_id: str
    task_type: str
    priority: int = 0
    dependencies: Set[str] = field(default_factory=set)
    timeout: int = 30
    retry_count: int = 0
    max_retries: int = 3
    
    def can_execute(self, completed: Set[str]) -> bool:
        """Check if task can execute."""
        return self.dependencies.issubset(completed)


@dataclass
class ExecutionPlan:
    """Execution plan for agent tasks."""
    tasks: List[AgentTask]
    strategy: ExecutionStrategy
    max_parallel: int = 5
    total_timeout: int = 300
    
    def get_ready_tasks(self, completed: Set[str]) -> List[AgentTask]:
        """Get tasks ready for execution."""
        return [
            task for task in self.tasks
            if task.agent_id not in completed and task.can_execute(completed)
        ]
    
    def get_batches(self) -> List[List[AgentTask]]:
        """Get task batches for batch execution."""
        batches = []
        completed = set()
        remaining = self.tasks.copy()
        
        while remaining:
            batch = []
            for task in remaining[:]:
                if task.can_execute(completed):
                    batch.append(task)
                    remaining.remove(task)
            
            if not batch:
                # Circular dependency or error
                logger.error("Cannot resolve task dependencies")
                break
            
            batches.append(batch)
            completed.update(task.agent_id for task in batch)
        
        return batches


class AgentScheduler:
    """
    Schedules agent execution based on various strategies.
    """
    
    def __init__(
        self,
        strategy: CoordinationStrategy = CoordinationStrategy.ROUND_ROBIN
    ):
        """
        Initialize scheduler.
        
        Args:
            strategy: Coordination strategy
        """
        self.strategy = strategy
        self.agent_queue: List[str] = []
        self.agent_priorities: Dict[str, int] = {}
        self.agent_performance: Dict[str, float] = {}
        self.execution_history: List[Tuple[str, datetime]] = []
    
    def schedule_agents(
        self,
        agents: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Schedule agents for execution.
        
        Args:
            agents: List of agent IDs
            context: Optional context for scheduling
            
        Returns:
            Ordered list of agent IDs
        """
        if self.strategy == CoordinationStrategy.ROUND_ROBIN:
            return self._round_robin_schedule(agents)
        elif self.strategy == CoordinationStrategy.RANDOM:
            return self._random_schedule(agents)
        elif self.strategy == CoordinationStrategy.PERFORMANCE_BASED:
            return self._performance_schedule(agents)
        elif self.strategy == CoordinationStrategy.ADAPTIVE:
            return self._adaptive_schedule(agents, context)
        else:
            return agents
    
    def _round_robin_schedule(self, agents: List[str]) -> List[str]:
        """Round-robin scheduling."""
        # Maintain queue order
        if not self.agent_queue:
            self.agent_queue = agents.copy()
        
        # Rotate queue
        if self.agent_queue:
            self.agent_queue = self.agent_queue[1:] + [self.agent_queue[0]]
        
        # Filter to active agents
        scheduled = []
        for agent in self.agent_queue:
            if agent in agents:
                scheduled.append(agent)
        
        # Add any new agents
        for agent in agents:
            if agent not in scheduled:
                scheduled.append(agent)
        
        return scheduled
    
    def _random_schedule(self, agents: List[str]) -> List[str]:
        """Random scheduling."""
        import random
        scheduled = agents.copy()
        random.shuffle(scheduled)
        return scheduled
    
    def _performance_schedule(self, agents: List[str]) -> List[str]:
        """Performance-based scheduling."""
        # Sort by performance (higher performance first)
        return sorted(
            agents,
            key=lambda a: self.agent_performance.get(a, 0),
            reverse=True
        )
    
    def _adaptive_schedule(
        self,
        agents: List[str],
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Adaptive scheduling based on context."""
        if not context:
            return self._round_robin_schedule(agents)
        
        # Adapt based on game phase
        phase = context.get("phase", "early")
        
        if phase == "early":
            # Random for exploration
            return self._random_schedule(agents)
        elif phase in ["mid", "late"]:
            # Mix of performance and fairness
            perf_order = self._performance_schedule(agents)
            rr_order = self._round_robin_schedule(agents)
            
            # Interleave
            scheduled = []
            for i in range(len(agents)):
                if i % 2 == 0 and i < len(perf_order):
                    scheduled.append(perf_order[i // 2])
                elif i < len(rr_order):
                    scheduled.append(rr_order[i // 2])
            
            return scheduled
        else:
            # Final phase - performance based
            return self._performance_schedule(agents)
    
    def update_performance(self, agent_id: str, score: float) -> None:
        """Update agent performance score."""
        self.agent_performance[agent_id] = score
    
    def record_execution(self, agent_id: str) -> None:
        """Record agent execution."""
        self.execution_history.append((agent_id, datetime.utcnow()))


class AgentCoordinator:
    """
    Main agent coordinator for managing agent execution.
    """
    
    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        max_parallel: int = 5,
        default_timeout: int = 30
    ):
        """
        Initialize coordinator.
        
        Args:
            agents: Dictionary of agents
            max_parallel: Maximum parallel executions
            default_timeout: Default task timeout
        """
        self.agents = agents
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout
        
        # Scheduler
        self.scheduler = AgentScheduler()
        
        # Execution tracking
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Dict[str, str] = {}
        
        # Resource management
        self.semaphore = asyncio.Semaphore(max_parallel)
    
    async def execute_plan(
        self,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """
        Execute an execution plan.
        
        Args:
            plan: Execution plan
            
        Returns:
            Execution results
        """
        logger.info(f"Executing plan with {len(plan.tasks)} tasks")
        
        results = {}
        
        if plan.strategy == ExecutionStrategy.SEQUENTIAL:
            results = await self._execute_sequential(plan)
        elif plan.strategy == ExecutionStrategy.PARALLEL:
            results = await self._execute_parallel(plan)
        elif plan.strategy == ExecutionStrategy.BATCH:
            results = await self._execute_batch(plan)
        elif plan.strategy == ExecutionStrategy.PRIORITY:
            results = await self._execute_priority(plan)
        
        return {
            "results": results,
            "completed": list(self.completed_tasks),
            "failed": self.failed_tasks.copy()
        }
    
    async def _execute_sequential(
        self,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Execute tasks sequentially."""
        results = {}
        
        for task in plan.tasks:
            if task.agent_id in self.completed_tasks:
                continue
            
            try:
                result = await self._execute_task(task)
                results[task.agent_id] = result
                self.completed_tasks.add(task.agent_id)
            except Exception as e:
                logger.error(f"Task failed for {task.agent_id}: {e}")
                self.failed_tasks[task.agent_id] = str(e)
                
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    # Retry later
                    plan.tasks.append(task)
        
        return results
    
    async def _execute_parallel(
        self,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Execute tasks in parallel."""
        results = {}
        tasks = []
        
        # Create all tasks
        for task in plan.tasks:
            if task.agent_id not in self.completed_tasks:
                async_task = asyncio.create_task(
                    self._execute_task_with_semaphore(task)
                )
                tasks.append((task.agent_id, async_task))
                self.active_tasks[task.agent_id] = async_task
        
        # Wait for completion
        for agent_id, async_task in tasks:
            try:
                result = await async_task
                results[agent_id] = result
                self.completed_tasks.add(agent_id)
            except Exception as e:
                logger.error(f"Parallel task failed for {agent_id}: {e}")
                self.failed_tasks[agent_id] = str(e)
            finally:
                self.active_tasks.pop(agent_id, None)
        
        return results
    
    async def _execute_batch(
        self,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Execute tasks in batches."""
        results = {}
        batches = plan.get_batches()
        
        for i, batch in enumerate(batches):
            logger.info(f"Executing batch {i+1}/{len(batches)}")
            
            # Execute batch in parallel
            batch_plan = ExecutionPlan(
                tasks=batch,
                strategy=ExecutionStrategy.PARALLEL,
                max_parallel=plan.max_parallel
            )
            
            batch_results = await self._execute_parallel(batch_plan)
            results.update(batch_results)
        
        return results
    
    async def _execute_priority(
        self,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Execute tasks by priority."""
        # Sort by priority (higher first)
        sorted_tasks = sorted(
            plan.tasks,
            key=lambda t: t.priority,
            reverse=True
        )
        
        # Execute in priority order with parallelism
        priority_plan = ExecutionPlan(
            tasks=sorted_tasks,
            strategy=ExecutionStrategy.PARALLEL,
            max_parallel=plan.max_parallel
        )
        
        return await self._execute_parallel(priority_plan)
    
    async def _execute_task(self, task: AgentTask) -> Any:
        """Execute a single task."""
        agent = self.agents.get(task.agent_id)
        if not agent:
            raise ValueError(f"Agent {task.agent_id} not found")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._run_agent_task(agent, task),
                timeout=task.timeout
            )
            
            self.scheduler.record_execution(task.agent_id)
            return result
        
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task timed out for {task.agent_id}")
    
    async def _execute_task_with_semaphore(
        self,
        task: AgentTask
    ) -> Any:
        """Execute task with semaphore for parallel limit."""
        async with self.semaphore:
            return await self._execute_task(task)
    
    async def _run_agent_task(
        self,
        agent: BaseAgent,
        task: AgentTask
    ) -> Any:
        """Run specific agent task."""
        logger.debug(f"Running task {task.task_type} for {agent.agent_id}")
        
        if task.task_type == "initialize":
            return await agent.initialize()
        elif task.task_type == "process_message":
            # Would need message from context
            return None
        elif task.task_type == "generate_action":
            # Would need context
            return await agent.generate_action({})
        elif task.task_type == "update_state":
            # Would need state from context
            return await agent.update_state({})
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    async def coordinate_turn(
        self,
        active_agents: List[str],
        message: Message
    ) -> Dict[str, Any]:
        """
        Coordinate agent actions for a turn.
        
        Args:
            active_agents: List of active agent IDs
            message: Message to process
            
        Returns:
            Coordination results
        """
        # Schedule agents
        scheduled = self.scheduler.schedule_agents(active_agents)
        
        # Create tasks for message processing
        tasks = []
        for agent_id in scheduled:
            if agent_id != message.sender_id:
                task = AgentTask(
                    agent_id=agent_id,
                    task_type="process_message",
                    priority=self.scheduler.agent_priorities.get(agent_id, 0)
                )
                tasks.append(task)
        
        # Create execution plan
        plan = ExecutionPlan(
            tasks=tasks,
            strategy=ExecutionStrategy.PARALLEL,
            max_parallel=self.max_parallel
        )
        
        # Execute
        return await self.execute_plan(plan)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        return {
            "total_agents": len(self.agents),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "scheduler_history": len(self.scheduler.execution_history),
            "max_parallel": self.max_parallel
        }