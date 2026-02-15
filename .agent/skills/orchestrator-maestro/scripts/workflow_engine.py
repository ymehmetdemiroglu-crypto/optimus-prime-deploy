"""
Workflow Engine - Core orchestration logic for multi-skill execution
"""
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

class WorkflowEngine:
    """
    Coordinates execution of multiple skills with dependency resolution
    """
    
    def __init__(self, skill_registry_path: Optional[str] = None):
        """
        Initialize workflow engine
        
        Args:
            skill_registry_path: Path to skill registry JSON
        """
        self.skill_registry = self._load_skill_registry(skill_registry_path)
        self.execution_log = []
        
    def _load_skill_registry(self, path: Optional[str]) -> Dict[str, Any]:
        """Load available skills from registry"""
        if not path:
            # Default path
            path = Path(__file__).parent.parent / "resources" / "skill_registry.json"
        
        if Path(path).exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    async def execute_workflow(
        self,
        workflow_template: Dict[str, Any],
        parameters: Dict[str, Any],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a workflow from template
        
        Args:
            workflow_template: Workflow definition
            parameters: Runtime parameters
            dry_run: If True, simulate execution without calling skills
            
        Returns:
            Workflow execution results
        """
        workflow_id = f"wf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.utcnow()
        
        results = {}
        execution_log = []
        
        try:
            skills_sequence = workflow_template.get("skills", [])
            
            for skill_step in skills_sequence:
                step_start = datetime.utcnow()
                
                # Resolve input parameters
                resolved_input = self._resolve_parameters(
                    skill_step.get("input", {}),
                    parameters,
                    results
                )
                
                if dry_run:
                    # Simulate execution
                    step_result = {"status": "simulated", "input": resolved_input}
                else:
                    # Actually invoke skill
                    step_result = await self._invoke_skill(
                        skill_step["skill"],
                        skill_step.get("action"),
                        resolved_input
                    )
                
                # Store result with output key
                output_key = skill_step.get("output_key", skill_step["skill"])
                results[output_key] = step_result
                
                # Log execution
                step_end = datetime.utcnow()
                execution_log.append({
                    "skill": skill_step["skill"],
                    "action": skill_step.get("action"),
                    "status": "completed" if not dry_run else "simulated",
                    "duration_seconds": (step_end - step_start).total_seconds(),
                    "timestamp": step_start.isoformat()
                })
        
        except Exception as e:
            execution_log.append({
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            raise
        
        end_time = datetime.utcnow()
        
        return {
            "workflow_id": workflow_id,
            "status": "completed" if not dry_run else "simulated",
            "execution_time_seconds": (end_time - start_time).total_seconds(),
            "skills_executed": len(skills_sequence),
            "results": results,
            "execution_log": execution_log
        }
    
    def _resolve_parameters(
        self,
        param_template: Dict[str, Any],
        runtime_params: Dict[str, Any],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve {{variable}} references in parameters
        
        Args:
            param_template: Parameter template with {{vars}}
            runtime_params: User-provided parameters
            previous_results: Results from previous skill executions
            
        Returns:
            Resolved parameters
        """
        resolved = {}
        
        for key, value in param_template.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                # Extract variable path
                var_path = value[2:-2].strip()
                
                # Check runtime params first
                if var_path in runtime_params:
                    resolved[key] = runtime_params[var_path]
                # Then check previous results
                elif "." in var_path:
                    # Nested access like "market_data.keywords"
                    parts = var_path.split(".")
                    current = previous_results
                    for part in parts:
                        if isinstance(current, dict) and part in current:
                            current = current[part]
                        else:
                            current = None
                            break
                    resolved[key] = current
                else:
                    resolved[key] = previous_results.get(var_path)
            else:
                resolved[key] = value
        
        return resolved
    
    async def _invoke_skill(
        self,
        skill_name: str,
        action: Optional[str],
        input_data: Dict[str, Any]
    ) -> Any:
        """
        Invoke a skill (placeholder - actual implementation would call real skills)
        
        Args:
            skill_name: Name of skill to invoke
            action: Specific action within skill
            input_data: Input parameters
            
        Returns:
            Skill execution result
        """
        # This is a placeholder. In actual implementation, this would:
        # 1. Load the skill's SKILL.md
        # 2. Parse its API/operations
        # 3. Call the appropriate endpoint or function
        # 4. Return the result
        
        print(f"INVOKE: {skill_name}.{action}({input_data})")
        
        # Simulated response
        return {
            "skill": skill_name,
            "action": action,
            "input_received": input_data,
            "simulated": True
        }
    
    async def execute_parallel(
        self,
        skill_steps: List[Dict[str, Any]],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute multiple skills in parallel
        
        Args:
            skill_steps: List of skill execution steps
            parameters: Runtime parameters
            
        Returns:
            Combined results from all parallel executions
        """
        tasks = []
        for step in skill_steps:
            task = self._invoke_skill(
                step["skill"],
                step.get("action"),
                self._resolve_parameters(step.get("input", {}), parameters, {})
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        combined = {}
        for i, step in enumerate(skill_steps):
            output_key = step.get("output_key", step["skill"])
            combined[output_key] = results[i]
        
        return combined


# Example usage
if __name__ == "__main__":
    async def test_workflow():
        engine = WorkflowEngine()
        
        workflow = {
            "name": "test_workflow",
            "skills": [
                {
                    "skill": "market-researcher",
                    "input": {"asin": "{{product_asin}}"},
                    "output_key": "market_data"
                },
                {
                    "skill": "grok-admaster-operator",
                    "action": "create_campaign",
                    "input": {
                        "asin": "{{product_asin}}",
                        "keywords": "{{market_data.keywords}}"
                    }
                }
            ]
        }
        
        result = await engine.execute_workflow(
            workflow,
            parameters={"product_asin": "B0ABC123"},
            dry_run=True
        )
        
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_workflow())
