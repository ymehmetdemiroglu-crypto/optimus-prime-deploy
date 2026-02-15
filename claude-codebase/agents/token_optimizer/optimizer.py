"""
Token Optimizer - Monitors and optimizes token usage across the system.

Purpose:
- Track token usage per agent and operation
- Detect token waste and inefficiencies
- Recommend optimizations
- Enforce token budgets
- Generate usage reports

Token Optimization:
- Real-time monitoring
- Anomaly detection (unexpected high usage)
- Optimization recommendations
- Budget alerts
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from ..base_agent import BaseAgent


@dataclass
class TokenUsageRecord:
    """
    Record of token usage for a single operation.
    
    Attributes:
        agent: Agent name that used the tokens
        operation: Operation type
        tokens: Number of tokens used
        timestamp: When the operation occurred
        metadata: Additional context
    """
    agent: str
    operation: str
    tokens: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """
    Suggestion for reducing token usage.
    
    Attributes:
        priority: Priority level (1=high, 3=low)
        category: Type of optimization (caching, compression, etc.)
        description: What to optimize
        potential_savings: Estimated tokens that could be saved
    """
    priority: int
    category: str
    description: str
    potential_savings: int


class TokenOptimizer(BaseAgent):
    """
    Monitors and optimizes token usage across all agents.
    
    Expected Operations:
        - track(agent, operation, tokens): Record token usage
        - get_stats(): Get usage statistics
        - get_recommendations(): Get optimization suggestions
        - check_budget(tokens): Verify if within budget
    
    Returns:
        - Statistics, recommendations, or budget status
    
    Best Practices:
        1. Track all token usage centrally
        2. Set realistic budgets per operation type
        3. Review recommendations regularly
        4. Alert on budget overruns
        5. Generate periodic reports for analysis
    
    Token Insights:
        - Identify most expensive operations
        - Find caching opportunities
        - Detect redundant API calls
        - Monitor compression effectiveness
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        
        # Configuration
        self.total_budget = config.get('total_budget', 200000)
        self.operation_budgets = config.get('operation_budgets', {
            'input_processing': 500,
            'intent_classification': 1000,
            'context_management': 50000,
            'response_generation': 8000,
        })
        
        self.alert_threshold = config.get('alert_threshold', 0.8)  # 80% of budget
        
        # State
        self.usage_records: List[TokenUsageRecord] = []
        self.usage_by_agent: Dict[str, int] = defaultdict(int)
        self.usage_by_operation: Dict[str, int] = defaultdict(int)
        self.total_usage = 0
    
    def process(self, operation: str, **kwargs) -> Any:
        """
        Process optimizer operation.
        
        Args:
            operation: Operation type ('track', 'stats', 'recommendations', 'check_budget')
            **kwargs: Operation-specific arguments
        
        Returns:
            Result depends on operation
        """
        if operation == 'track':
            return self.track_usage(
                kwargs['agent'],
                kwargs['operation_type'],
                kwargs['tokens'],
                kwargs.get('metadata', {})
            )
        elif operation == 'stats':
            return self.get_stats()
        elif operation == 'recommendations':
            return self.get_recommendations()
        elif operation == 'check_budget':
            return self.check_budget(kwargs['tokens'])
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def track_usage(self, agent: str, operation_type: str, 
                   tokens: int, metadata: Dict[str, Any] = None) -> None:
        """
        Record token usage for an operation.
        
        Args:
            agent: Name of the agent that used tokens
            operation_type: Type of operation
            tokens: Number of tokens used
            metadata: Optional additional context
        """
        record = TokenUsageRecord(
            agent=agent,
            operation=operation_type,
            tokens=tokens,
            metadata=metadata or {}
        )
        
        self.usage_records.append(record)
        self.usage_by_agent[agent] += tokens
        self.usage_by_operation[operation_type] += tokens
        self.total_usage += tokens
        
        # Track for base agent stats
        self.track_tokens(tokens)
        
        # Check if approaching budget
        budget_usage = self.total_usage / self.total_budget
        if budget_usage >= self.alert_threshold:
            self.logger.warning(
                f"Token usage at {budget_usage:.1%} of budget "
                f"({self.total_usage}/{self.total_budget})"
            )
        
        # Check operation-specific budget
        if operation_type in self.operation_budgets:
            op_budget = self.operation_budgets[operation_type]
            op_usage = self.usage_by_operation[operation_type]
            if op_usage > op_budget:
                self.logger.warning(
                    f"{operation_type} exceeded budget: "
                    f"{op_usage}/{op_budget} tokens"
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics.
        
        Returns:
            Dictionary with usage breakdown and metrics
        """
        stats = {
            'total_usage': self.total_usage,
            'total_budget': self.total_budget,
            'budget_remaining': self.total_budget - self.total_usage,
            'usage_percentage': (self.total_usage / self.total_budget * 100),
            'operation_count': len(self.usage_records),
            
            'by_agent': dict(self.usage_by_agent),
            'by_operation': dict(self.usage_by_operation),
            
            'avg_per_operation': (
                self.total_usage / len(self.usage_records)
                if self.usage_records else 0
            ),
            
            'most_expensive_agent': (
                max(self.usage_by_agent.items(), key=lambda x: x[1])[0]
                if self.usage_by_agent else None
            ),
            
            'most_expensive_operation': (
                max(self.usage_by_operation.items(), key=lambda x: x[1])[0]
                if self.usage_by_operation else None
            ),
        }
        
        return stats
    
    def get_recommendations(self) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations based on usage patterns.
        
        Returns:
            List of optimization suggestions
        """
        recommendations = []
        
        # Check for repeated operations (caching opportunity)
        operation_counts = defaultdict(int)
        for record in self.usage_records:
            key = f"{record.agent}:{record.operation}"
            operation_counts[key] += 1
        
        # Recommend caching for frequently repeated operations
        for op_key, count in operation_counts.items():
            if count > 5:  # Repeated 5+ times
                agent, operation = op_key.split(':', 1)
                tokens_per_op = self.usage_by_operation.get(operation, 0) / count
                potential_savings = int(tokens_per_op * count * 0.8)  # 80% cache hit rate
                
                recommendations.append(OptimizationRecommendation(
                    priority=1,
                    category='caching',
                    description=(
                        f"Enable caching for {agent}/{operation} "
                        f"(repeated {count} times)"
                    ),
                    potential_savings=potential_savings
                ))
        
        # Check for budget overruns
        for op_type, budget in self.operation_budgets.items():
            usage = self.usage_by_operation.get(op_type, 0)
            if usage > budget:
                recommendations.append(OptimizationRecommendation(
                    priority=1,
                    category='budget',
                    description=(
                        f"{op_type} exceeded budget by {usage - budget} tokens"
                    ),
                    potential_savings=0
                ))
        
        # Check for large context (compression opportunity)
        context_usage = self.usage_by_operation.get('context_management', 0)
        if context_usage > 30000:
            potential_savings = int(context_usage * 0.4)  # 40% compression
            recommendations.append(OptimizationRecommendation(
                priority=2,
                category='compression',
                description=(
                    f"Context size is large ({context_usage} tokens). "
                    f"Consider more aggressive compression."
                ),
                potential_savings=potential_savings
            ))
        
        # Check for long responses (truncation opportunity)
        response_usage = self.usage_by_operation.get('response_generation', 0)
        response_count = sum(
            1 for r in self.usage_records 
            if r.operation == 'response_generation'
        )
        if response_count > 0:
            avg_response = response_usage / response_count
            if avg_response > 2000:
                potential_savings = int((avg_response - 1500) * response_count)
                recommendations.append(OptimizationRecommendation(
                    priority=3,
                    category='response_length',
                    description=(
                        f"Average response is {avg_response:.0f} tokens. "
                        f"Consider reducing max_tokens or using more concise prompts."
                    ),
                    potential_savings=potential_savings
                ))
        
        # Sort by priority
        recommendations.sort(key=lambda r: (r.priority, -r.potential_savings))
        
        return recommendations
    
    def check_budget(self, additional_tokens: int) -> Dict[str, Any]:
        """
        Check if additional tokens would exceed budget.
        
        Args:
            additional_tokens: Tokens about to be used
        
        Returns:
            Dictionary with budget status and recommendation
        """
        projected_usage = self.total_usage + additional_tokens
        within_budget = projected_usage <= self.total_budget
        
        return {
            'within_budget': within_budget,
            'current_usage': self.total_usage,
            'additional_tokens': additional_tokens,
            'projected_usage': projected_usage,
            'budget': self.total_budget,
            'remaining': self.total_budget - projected_usage,
            'recommendation': (
                'proceed' if within_budget 
                else 'compress_context_or_reduce_response'
            )
        }
    
    def reset(self) -> None:
        """Reset all usage tracking."""
        self.usage_records.clear()
        self.usage_by_agent.clear()
        self.usage_by_operation.clear()
        self.total_usage = 0
        self.reset_stats()
        self.logger.info("Token usage reset")
