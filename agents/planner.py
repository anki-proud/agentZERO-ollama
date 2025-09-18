"""
Planner Agent for agentZERO
Handles task planning, decomposition, and strategy formulation
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from core.ollama_client import OllamaClient, GenerationConfig
from core.prompt_engine import PromptEngine

logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class TaskType(Enum):
    ANALYSIS = "analysis"
    CREATION = "creation"
    PROBLEM_SOLVING = "problem_solving"
    RESEARCH = "research"
    CODING = "coding"
    WRITING = "writing"
    GENERAL = "general"

@dataclass
class TaskStep:
    """Individual step in a task plan"""
    id: int
    description: str
    type: TaskType
    estimated_time: int  # in seconds
    dependencies: List[int] = field(default_factory=list)
    required_model: Optional[str] = None
    success_criteria: str = ""
    resources_needed: List[str] = field(default_factory=list)

@dataclass
class TaskPlan:
    """Complete plan for executing a task"""
    task_id: str
    original_request: str
    complexity: TaskComplexity
    estimated_total_time: int
    steps: List[TaskStep]
    success_metrics: List[str] = field(default_factory=list)
    fallback_strategies: List[str] = field(default_factory=list)
    created_at: str = ""

class PlannerAgent:
    """Agent responsible for task planning and decomposition"""
    
    def __init__(self, model: str = "smollm:1.7b", ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.ollama_client = OllamaClient(ollama_url)
        self.prompt_engine = PromptEngine()
        
        # Planning configuration
        self.max_steps = 10
        self.max_planning_time = 60  # seconds
        
        # System prompt for planning
        self.system_prompt = """You are an expert task planner and strategist. Your role is to:

1. Analyze incoming tasks and break them down into manageable steps
2. Estimate complexity and time requirements
3. Identify dependencies between steps
4. Suggest appropriate models/tools for each step
5. Create fallback strategies for potential failures

Always respond in valid JSON format with the following structure:
{
    "complexity": "simple|moderate|complex|expert",
    "estimated_total_time": <seconds>,
    "steps": [
        {
            "id": <number>,
            "description": "<clear description>",
            "type": "<task_type>",
            "estimated_time": <seconds>,
            "dependencies": [<step_ids>],
            "required_model": "<model_name>",
            "success_criteria": "<how to measure success>",
            "resources_needed": ["<resource1>", "<resource2>"]
        }
    ],
    "success_metrics": ["<metric1>", "<metric2>"],
    "fallback_strategies": ["<strategy1>", "<strategy2>"]
}

Be practical, specific, and consider the limitations of small language models."""
    
    async def create_plan(self, task_description: str, context: Dict[str, Any] = None) -> TaskPlan:
        """Create a comprehensive plan for the given task"""
        logger.info(f"Creating plan for task: {task_description}")
        
        # Prepare context information
        context = context or {}
        available_models = ["smollm:135m", "smollm:360m", "smollm:1.7b"]
        
        # Build planning prompt
        planning_prompt = f"""
Task to plan: {task_description}

Available models and their capabilities:
- smollm:135m: Fast, efficient, good for simple tasks and quick responses
- smollm:360m: Balanced performance, suitable for general tasks
- smollm:1.7b: Best reasoning, complex tasks, slower but more capable

Context information:
{json.dumps(context, indent=2) if context else "No additional context provided"}

Create a detailed execution plan for this task. Consider:
- What steps are needed to complete this task?
- Which model is best suited for each step?
- What are the dependencies between steps?
- How long might each step take?
- What could go wrong and how to handle it?

Respond with a valid JSON plan following the specified format.
"""
        
        try:
            # Generate plan using the planner model
            config = GenerationConfig(
                temperature=0.3,  # Lower temperature for more consistent planning
                max_tokens=2048
            )
            
            messages = self.prompt_engine.create_chat_messages(
                self.system_prompt,
                planning_prompt
            )
            
            async with self.ollama_client as client:
                response = await client.chat(self.model, messages, config)
            
            # Parse the JSON response
            plan_data = self._parse_plan_response(response)
            
            # Create TaskPlan object
            task_plan = self._create_task_plan(task_description, plan_data)
            
            logger.info(f"Created plan with {len(task_plan.steps)} steps")
            return task_plan
            
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            # Return a fallback simple plan
            return self._create_fallback_plan(task_description)
    
    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from the planning model"""
        try:
            # Clean up the response - remove any markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            # Parse JSON
            plan_data = json.loads(cleaned_response)
            
            # Validate required fields
            required_fields = ["complexity", "estimated_total_time", "steps"]
            for field in required_fields:
                if field not in plan_data:
                    raise ValueError(f"Missing required field: {field}")
            
            return plan_data
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing plan response: {e}")
            logger.debug(f"Raw response: {response}")
            raise ValueError(f"Invalid plan format: {e}")
    
    def _create_task_plan(self, task_description: str, plan_data: Dict[str, Any]) -> TaskPlan:
        """Create TaskPlan object from parsed data"""
        import uuid
        from datetime import datetime
        
        # Parse complexity
        complexity_str = plan_data.get("complexity", "moderate").lower()
        complexity = TaskComplexity(complexity_str)
        
        # Create task steps
        steps = []
        for step_data in plan_data.get("steps", []):
            # Parse task type
            task_type_str = step_data.get("type", "general").lower()
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.GENERAL
            
            step = TaskStep(
                id=step_data.get("id", len(steps) + 1),
                description=step_data.get("description", ""),
                type=task_type,
                estimated_time=step_data.get("estimated_time", 30),
                dependencies=step_data.get("dependencies", []),
                required_model=step_data.get("required_model"),
                success_criteria=step_data.get("success_criteria", ""),
                resources_needed=step_data.get("resources_needed", [])
            )
            steps.append(step)
        
        return TaskPlan(
            task_id=str(uuid.uuid4()),
            original_request=task_description,
            complexity=complexity,
            estimated_total_time=plan_data.get("estimated_total_time", 300),
            steps=steps,
            success_metrics=plan_data.get("success_metrics", []),
            fallback_strategies=plan_data.get("fallback_strategies", []),
            created_at=datetime.now().isoformat()
        )
    
    def _create_fallback_plan(self, task_description: str) -> TaskPlan:
        """Create a simple fallback plan when planning fails"""
        import uuid
        from datetime import datetime
        
        logger.warning("Creating fallback plan due to planning failure")
        
        # Create a simple single-step plan
        step = TaskStep(
            id=1,
            description=f"Complete the task: {task_description}",
            type=TaskType.GENERAL,
            estimated_time=120,
            dependencies=[],
            required_model="smollm:360m",
            success_criteria="Task completed successfully",
            resources_needed=[]
        )
        
        return TaskPlan(
            task_id=str(uuid.uuid4()),
            original_request=task_description,
            complexity=TaskComplexity.MODERATE,
            estimated_total_time=120,
            steps=[step],
            success_metrics=["Task completion"],
            fallback_strategies=["Break down into smaller parts", "Use simpler approach"],
            created_at=datetime.now().isoformat()
        )
    
    async def refine_plan(self, plan: TaskPlan, feedback: str) -> TaskPlan:
        """Refine an existing plan based on feedback"""
        logger.info(f"Refining plan {plan.task_id} based on feedback")
        
        refinement_prompt = f"""
Original task: {plan.original_request}

Current plan:
{self._plan_to_text(plan)}

Feedback received: {feedback}

Please refine the plan based on this feedback. Consider:
- Are there steps that need to be added, removed, or modified?
- Should the complexity or time estimates be adjusted?
- Are there better model choices for any steps?
- Should the success criteria be updated?

Respond with an improved JSON plan following the same format.
"""
        
        try:
            config = GenerationConfig(temperature=0.3, max_tokens=2048)
            
            messages = self.prompt_engine.create_chat_messages(
                self.system_prompt,
                refinement_prompt
            )
            
            async with self.ollama_client as client:
                response = await client.chat(self.model, messages, config)
            
            plan_data = self._parse_plan_response(response)
            refined_plan = self._create_task_plan(plan.original_request, plan_data)
            
            # Preserve original task_id
            refined_plan.task_id = plan.task_id
            
            logger.info(f"Refined plan now has {len(refined_plan.steps)} steps")
            return refined_plan
            
        except Exception as e:
            logger.error(f"Error refining plan: {e}")
            return plan  # Return original plan if refinement fails
    
    def _plan_to_text(self, plan: TaskPlan) -> str:
        """Convert plan to readable text format"""
        text = f"Complexity: {plan.complexity.value}\n"
        text += f"Estimated time: {plan.estimated_total_time} seconds\n\n"
        text += "Steps:\n"
        
        for step in plan.steps:
            text += f"{step.id}. {step.description}\n"
            text += f"   Type: {step.type.value}\n"
            text += f"   Time: {step.estimated_time}s\n"
            text += f"   Model: {step.required_model}\n"
            if step.dependencies:
                text += f"   Depends on: {step.dependencies}\n"
            text += "\n"
        
        return text
    
    def validate_plan(self, plan: TaskPlan) -> List[str]:
        """Validate a plan and return list of issues"""
        issues = []
        
        # Check for circular dependencies
        if self._has_circular_dependencies(plan.steps):
            issues.append("Circular dependencies detected in plan")
        
        # Check step dependencies exist
        step_ids = {step.id for step in plan.steps}
        for step in plan.steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    issues.append(f"Step {step.id} depends on non-existent step {dep_id}")
        
        # Check time estimates are reasonable
        if plan.estimated_total_time > 3600:  # 1 hour
            issues.append("Plan estimated time exceeds 1 hour - consider breaking down further")
        
        # Check model assignments
        available_models = ["smollm:135m", "smollm:360m", "smollm:1.7b"]
        for step in plan.steps:
            if step.required_model and step.required_model not in available_models:
                issues.append(f"Step {step.id} requires unavailable model: {step.required_model}")
        
        return issues
    
    def _has_circular_dependencies(self, steps: List[TaskStep]) -> bool:
        """Check for circular dependencies in the plan"""
        # Create adjacency list
        graph = {step.id: step.dependencies for step in steps}
        
        # Use DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for step_id in graph:
            if step_id not in visited:
                if has_cycle(step_id):
                    return True
        
        return False
    
    def get_execution_order(self, plan: TaskPlan) -> List[List[int]]:
        """Get the optimal execution order considering dependencies"""
        steps_by_id = {step.id: step for step in plan.steps}
        
        # Topological sort to get execution order
        in_degree = {step.id: 0 for step in plan.steps}
        
        # Calculate in-degrees
        for step in plan.steps:
            for dep_id in step.dependencies:
                if dep_id in in_degree:
                    in_degree[step.id] += 1
        
        # Find steps that can be executed in parallel
        execution_levels = []
        remaining_steps = set(step.id for step in plan.steps)
        
        while remaining_steps:
            # Find steps with no dependencies
            ready_steps = [
                step_id for step_id in remaining_steps 
                if in_degree[step_id] == 0
            ]
            
            if not ready_steps:
                # This shouldn't happen with a valid plan
                logger.error("No ready steps found - possible circular dependency")
                break
            
            execution_levels.append(ready_steps)
            
            # Remove ready steps and update in-degrees
            for step_id in ready_steps:
                remaining_steps.remove(step_id)
                for other_step in plan.steps:
                    if step_id in other_step.dependencies:
                        in_degree[other_step.id] -= 1
        
        return execution_levels

# Example usage
if __name__ == "__main__":
    async def test_planner():
        planner = PlannerAgent()
        
        # Test planning
        task = "Create a simple web application that displays weather information"
        plan = await planner.create_plan(task)
        
        print(f"Created plan for: {task}")
        print(f"Complexity: {plan.complexity.value}")
        print(f"Total time: {plan.estimated_total_time}s")
        print(f"Steps: {len(plan.steps)}")
        
        for step in plan.steps:
            print(f"  {step.id}. {step.description} ({step.type.value})")
        
        # Test validation
        issues = planner.validate_plan(plan)
        if issues:
            print(f"Plan issues: {issues}")
        else:
            print("Plan validation passed")
        
        # Test execution order
        execution_order = planner.get_execution_order(plan)
        print(f"Execution order: {execution_order}")
    
    asyncio.run(test_planner())

