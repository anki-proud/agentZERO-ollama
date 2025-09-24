"""
Executor Agent for agentZERO
Handles task execution based on plans from the Planner Agent
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

from core.ollama_client import OllamaClient, GenerationConfig
from core.prompt_engine import PromptEngine
from agents.planner import TaskPlan, TaskStep, TaskType

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ExecutionResult:
    """Result of executing a single task step"""
    step_id: int
    status: ExecutionStatus
    output: str
    error_message: Optional[str] = None
    execution_time: float = 0.0
    model_used: Optional[str] = None
    tokens_used: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class TaskExecution:
    """Complete execution context for a task"""
    task_id: str
    plan: TaskPlan
    results: Dict[int, ExecutionResult] = field(default_factory=dict)
    current_step: Optional[int] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_execution_time: float = 0.0

class ExecutorAgent:
    """Agent responsible for executing task plans"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_client = OllamaClient(ollama_url)
        self.prompt_engine = PromptEngine()
        
        # Execution configuration
        self.max_retries = 3
        self.timeout_per_step = 120  # seconds
        self.parallel_execution = True
        
        # Active executions
        self.active_executions: Dict[str, TaskExecution] = {}
        
        # Model selection strategy
        self.model_selection_strategy = "auto"  # auto, fastest, balanced, most_capable
        
        # System prompts for different task types
        self.system_prompts = {
            TaskType.ANALYSIS: """You are an expert analyst. Provide thorough, objective analysis based on the given information. Structure your response clearly with key findings, insights, and conclusions.""",
            
            TaskType.CREATION: """You are a creative assistant. Generate original, high-quality content that meets the specified requirements. Be innovative while staying practical and useful.""",
            
            TaskType.PROBLEM_SOLVING: """You are a problem-solving expert. Approach problems systematically, break them down into manageable parts, and provide clear, actionable solutions.""",
            
            TaskType.RESEARCH: """You are a research specialist. Gather, organize, and synthesize information comprehensively. Provide well-structured findings with proper context and implications.""",
            
            TaskType.CODING: """You are a programming expert. Write clean, efficient, well-documented code. Explain your approach and include error handling where appropriate.""",
            
            TaskType.WRITING: """You are a skilled writer. Create clear, engaging, and well-structured content appropriate for the target audience and purpose.""",
            
            TaskType.GENERAL: """You are a helpful AI assistant. Provide accurate, useful responses that directly address the user's needs. Be clear, concise, and practical."""
        }
    
    async def execute_plan(self, plan: TaskPlan, context: Dict[str, Any] = None) -> TaskExecution:
        """Execute a complete task plan"""
        logger.info(f"Starting execution of plan {plan.task_id}")
        
        # Create execution context
        execution = TaskExecution(
            task_id=plan.task_id,
            plan=plan,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now()
        )
        
        self.active_executions[plan.task_id] = execution
        
        try:
            # Get execution order from planner
            from agents.planner import PlannerAgent
            planner = PlannerAgent()
            execution_levels = planner.get_execution_order(plan)
            
            # Execute steps level by level
            for level, step_ids in enumerate(execution_levels):
                logger.info(f"Executing level {level + 1} with steps: {step_ids}")
                
                if self.parallel_execution and len(step_ids) > 1:
                    # Execute steps in parallel
                    tasks = []
                    for step_id in step_ids:
                        step = next(s for s in plan.steps if s.id == step_id)
                        task = self._execute_step(step, execution, context)
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for i, result in enumerate(results):
                        step_id = step_ids[i]
                        if isinstance(result, Exception):
                            execution.results[step_id] = ExecutionResult(
                                step_id=step_id,
                                status=ExecutionStatus.FAILED,
                                output="",
                                error_message=str(result)
                            )
                        else:
                            execution.results[step_id] = result
                else:
                    # Execute steps sequentially
                    for step_id in step_ids:
                        step = next(s for s in plan.steps if s.id == step_id)
                        result = await self._execute_step(step, execution, context)
                        execution.results[step_id] = result
                        
                        # Stop if step failed and no fallback
                        if result.status == ExecutionStatus.FAILED:
                            logger.error(f"Step {step_id} failed: {result.error_message}")
                            # Could implement fallback strategies here
            
            # Determine overall execution status
            failed_steps = [r for r in execution.results.values() if r.status == ExecutionStatus.FAILED]
            if failed_steps:
                execution.status = ExecutionStatus.FAILED
            else:
                execution.status = ExecutionStatus.COMPLETED
            
            execution.completed_at = datetime.now()
            execution.total_execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            logger.info(f"Plan execution completed with status: {execution.status.value}")
            return execution
            
        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            execution.status = ExecutionStatus.FAILED
            execution.completed_at = datetime.now()
            return execution
        finally:
            # Clean up
            if plan.task_id in self.active_executions:
                del self.active_executions[plan.task_id]
    
    async def _execute_step(
        self, 
        step: TaskStep, 
        execution: TaskExecution, 
        context: Dict[str, Any] = None
    ) -> ExecutionResult:
        """Execute a single task step"""
        logger.info(f"Executing step {step.id}: {step.description}")
        
        start_time = datetime.now()
        execution.current_step = step.id
        
        try:
            # Select appropriate model
            model = self._select_model_for_step(step)
            
            # Build execution prompt
            prompt = self._build_execution_prompt(step, execution, context)
            
            # Get system prompt for task type
            system_prompt = self.system_prompts.get(step.type, self.system_prompts[TaskType.GENERAL])
            
            # Execute with selected model
            config = GenerationConfig(
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9
            )
            
            messages = self.prompt_engine.create_chat_messages(
                system_prompt,
                prompt
            )
            
            async with self.ollama_client as client:
                response = await client.chat(model, messages, config)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Estimate tokens used (rough approximation)
            tokens_used = len(prompt.split()) + len(response.split())
            
            return ExecutionResult(
                step_id=step.id,
                status=ExecutionStatus.COMPLETED,
                output=response,
                execution_time=execution_time,
                model_used=model,
                tokens_used=tokens_used,
                started_at=start_time,
                completed_at=end_time
            )
            
        except Exception as e:
            logger.error(f"Error executing step {step.id}: {e}")
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return ExecutionResult(
                step_id=step.id,
                status=ExecutionStatus.FAILED,
                output="",
                error_message=str(e),
                execution_time=execution_time,
                started_at=start_time,
                completed_at=end_time
            )
    
    def _select_model_for_step(self, step: TaskStep) -> str:
        """Select the most appropriate model for a step"""
        # If step specifies a model, use it
        if step.required_model:
            return step.required_model
        
        # Use model selection strategy
        if self.model_selection_strategy == "fastest":
            return "smollm:135m"
        elif self.model_selection_strategy == "most_capable":
            return "smollm:1.7b"
        elif self.model_selection_strategy == "balanced":
            return "smollm:360m"
        else:  # auto
            return self._auto_select_model(step)
    
    def _auto_select_model(self, step: TaskStep) -> str:
        """Automatically select model based on step characteristics"""
        # Analyze step complexity
        description_length = len(step.description)
        estimated_complexity = 0
        
        # Simple heuristics for complexity
        complex_keywords = ["analyze", "detailed", "comprehensive", "complex", "advanced", "expert"]
        simple_keywords = ["simple", "quick", "basic", "brief", "short"]
        
        description_lower = step.description.lower()
        
        for keyword in complex_keywords:
            if keyword in description_lower:
                estimated_complexity += 1
        
        for keyword in simple_keywords:
            if keyword in description_lower:
                estimated_complexity -= 1
        
        # Consider task type
        complex_types = [TaskType.ANALYSIS, TaskType.PROBLEM_SOLVING, TaskType.RESEARCH, TaskType.CODING]
        if step.type in complex_types:
            estimated_complexity += 1
        
        # Consider estimated time
        if step.estimated_time > 60:  # More than 1 minute
            estimated_complexity += 1
        elif step.estimated_time < 15:  # Less than 15 seconds
            estimated_complexity -= 1
        
        # Select model based on complexity
        if estimated_complexity >= 2:
            return "smollm:1.7b"
        elif estimated_complexity <= -1:
            return "smollm:135m"
        else:
            return "smollm:360m"
    
    def _build_execution_prompt(
        self, 
        step: TaskStep, 
        execution: TaskExecution, 
        context: Dict[str, Any] = None
    ) -> str:
        """Build the execution prompt for a step"""
        prompt_parts = []
        
        # Add task context
        prompt_parts.append(f"Task: {execution.plan.original_request}")
        prompt_parts.append(f"Current Step: {step.description}")
        
        # Add step-specific information
        if step.success_criteria:
            prompt_parts.append(f"Success Criteria: {step.success_criteria}")
        
        if step.resources_needed:
            prompt_parts.append(f"Required Resources: {', '.join(step.resources_needed)}")
        
        # Add context from previous steps
        if step.dependencies:
            prompt_parts.append("\nPrevious Step Results:")
            for dep_id in step.dependencies:
                if dep_id in execution.results:
                    result = execution.results[dep_id]
                    if result.status == ExecutionStatus.COMPLETED:
                        prompt_parts.append(f"Step {dep_id}: {result.output[:200]}...")
        
        # Add additional context
        if context:
            prompt_parts.append(f"\nAdditional Context: {json.dumps(context, indent=2)}")
        
        # Add specific instructions based on task type
        type_instructions = {
            TaskType.ANALYSIS: "Provide a thorough analysis with clear findings and insights.",
            TaskType.CREATION: "Create original content that meets the specified requirements.",
            TaskType.PROBLEM_SOLVING: "Solve the problem step by step with clear reasoning.",
            TaskType.RESEARCH: "Research and synthesize information comprehensively.",
            TaskType.CODING: "Write clean, well-documented code with explanations.",
            TaskType.WRITING: "Write clear, engaging content for the target audience.",
            TaskType.GENERAL: "Complete the task effectively and efficiently."
        }
        
        instruction = type_instructions.get(step.type, type_instructions[TaskType.GENERAL])
        prompt_parts.append(f"\nInstructions: {instruction}")
        
        return "\n".join(prompt_parts)
    
    def get_execution_status(self, task_id: str) -> Optional[TaskExecution]:
        """Get the current status of a task execution"""
        return self.active_executions.get(task_id)
    
    def cancel_execution(self, task_id: str) -> bool:
        """Cancel an active execution"""
        if task_id in self.active_executions:
            execution = self.active_executions[task_id]
            execution.status = ExecutionStatus.FAILED
            execution.completed_at = datetime.now()
            del self.active_executions[task_id]
            logger.info(f"Cancelled execution of task {task_id}")
            return True
        return False
    
    def get_execution_summary(self, execution: TaskExecution) -> Dict[str, Any]:
        """Generate a summary of task execution"""
        total_steps = len(execution.plan.steps)
        completed_steps = len([r for r in execution.results.values() if r.status == ExecutionStatus.COMPLETED])
        failed_steps = len([r for r in execution.results.values() if r.status == ExecutionStatus.FAILED])
        
        total_tokens = sum(r.tokens_used for r in execution.results.values())
        
        models_used = {}
        for result in execution.results.values():
            if result.model_used:
                models_used[result.model_used] = models_used.get(result.model_used, 0) + 1
        
        return {
            "task_id": execution.task_id,
            "status": execution.status.value,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "success_rate": completed_steps / total_steps if total_steps > 0 else 0,
            "total_execution_time": execution.total_execution_time,
            "total_tokens_used": total_tokens,
            "models_used": models_used,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None
        }

# Example usage
if __name__ == "__main__":
    async def test_executor():
        from agents.planner import PlannerAgent
        
        # Create a simple plan
        planner = PlannerAgent()
        plan = await planner.create_plan("Write a short poem about artificial intelligence")
        
        # Execute the plan
        executor = ExecutorAgent()
        execution = await executor.execute_plan(plan)
        
        # Print results
        summary = executor.get_execution_summary(execution)
        print(f"Execution Summary: {json.dumps(summary, indent=2)}")
        
        for step_id, result in execution.results.items():
            print(f"\nStep {step_id}:")
            print(f"Status: {result.status.value}")
            print(f"Output: {result.output[:200]}...")
            print(f"Time: {result.execution_time:.2f}s")
            print(f"Model: {result.model_used}")
    
    asyncio.run(test_executor())

