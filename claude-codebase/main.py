"""
Main application - Orchestrates all agents and services.

This is the entry point that ties everything together with
token-efficient processing.
"""

import logging
from typing import List, Dict, Any

from agents.input_processor import InputProcessor
from agents.intent_classifier import IntentClassifier
from agents.context_manager import ContextManager
from agents.response_generator import ResponseGenerator
from agents.token_optimizer import TokenOptimizer
from configs import get_config
from services import ClaudeAPIService, StorageService
from utils import setup_logging
from models import Message, MessageRole


class ClaudeAssistant:
    """
    Main assistant class that coordinates all agents.
    
    Features:
    - Token-optimized conversation handling
    - Intelligent context management
    - Caching and optimization
    - Cost tracking
    
    Usage:
        # Initialize
        assistant = ClaudeAssistant()
        
        # Process user input
        response = assistant.process("What is machine learning?")
        
        # Get statistics
        stats = assistant.get_stats()
    """
    
    def __init__(self):
        """Initialize the assistant with all agents and services."""
        
        # Load configuration
        self.config = get_config()
        
        # Setup logging
        setup_logging(
            level=self.config.log_level,
            log_file=self.config.log_file,
            json_format=self.config.log_json_format
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.api_service = ClaudeAPIService(
            api_key=self.config.anthropic_api_key,
            default_model=self.config.default_model
        )
        self.storage_service = StorageService()
        
        # Initialize agents with config from YAML
        input_config = {
            'remove_filler_words': False,
            'cache_enabled': self.config.caching_enabled
        }
        self.input_processor = InputProcessor(input_config, self.logger)
        
        intent_config = {
            'cache_enabled': self.config.caching_enabled,
            'confidence_threshold': 0.7
        }
        self.intent_classifier = IntentClassifier(intent_config, self.logger)
        
        context_config = {
            'max_tokens': self.config.context_max_tokens,
            'window_size': self.config.context_window_size,
            'summarization_threshold': self.config.context_summarization_threshold,
            'auto_compress': self.config.context_auto_compress
        }
        self.context_manager = ContextManager(context_config, self.logger)
        
        response_config = {
            'model': self.config.default_model,
            'default_max_tokens': 4096,
            'streaming_enabled': True
        }
        self.response_generator = ResponseGenerator(response_config, self.logger)
        
        optimizer_config = {
            'total_budget': 200000,
            'operation_budgets': self.config.operation_budgets,
            'alert_threshold': 0.8
        }
        self.token_optimizer = TokenOptimizer(optimizer_config, self.logger)
        
        self.logger.info("ClaudeAssistant initialized successfully")
    
    def process(self, user_input: str, conversation_id: str = None) -> str:
        """
        Process user input and generate response.
        
        This is the main entry point for conversation handling.
        
        Args:
            user_input: Raw user input text
            conversation_id: Optional conversation ID for context tracking
        
        Returns:
            Assistant's response text
        
        Pipeline:
        1. Input Processing: Clean and normalize input
        2. Intent Classification: Determine user's goal
        3. Context Management: Retrieve relevant context
        4. Response Generation: Create response with Claude
        5. Token Optimization: Track usage and optimize
        """
        self.logger.info(f"Processing user input: {user_input[:50]}...")
        
        # Step 1: Process input
        processed_input = self.input_processor.process(user_input)
        self.token_optimizer.track_usage(
            agent='input_processor',
            operation_type='input_processing',
            tokens=processed_input.processed_tokens
        )
        
        # Step 2: Classify intent
        intent = self.intent_classifier.process(processed_input)
        self.token_optimizer.track_usage(
            agent='intent_classifier',
            operation_type='intent_classification',
            tokens=processed_input.processed_tokens
        )
        
        # Step 3: Add to context
        self.context_manager.add_message('user', processed_input.text)
        context = self.context_manager.get_context()
        self.token_optimizer.track_usage(
            agent='context_manager',
            operation_type='context_management',
            tokens=context.total_tokens
        )
        
        # Step 4: Generate response
        response = self.response_generator.process(intent, context)
        self.token_optimizer.track_usage(
            agent='response_generator',
            operation_type='response_generation',
            tokens=response.tokens_used
        )
        
        # Step 5: Add response to context
        self.context_manager.add_message('assistant', response.content)
        
        # Log completion
        self.logger.info(
            f"Request completed: {response.tokens_used} tokens, "
            f"intent={intent.type.value}"
        )
        
        return response.content
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics.
        
        Returns:
            Dictionary with stats from all agents
        """
        return {
            'input_processor': self.input_processor.get_stats(),
            'intent_classifier': self.intent_classifier.get_stats(),
            'context_manager': self.context_manager.get_stats(),
            'response_generator': self.response_generator.get_stats(),
            'token_optimizer': self.token_optimizer.get_stats()
        }
    
    def get_recommendations(self) -> List[Any]:
        """
        Get optimization recommendations.
        
        Returns:
            List of recommendations from token optimizer
        """
        return self.token_optimizer.process('recommendations')
    
    def reset(self) -> None:
        """Reset all agents and clear context."""
        self.context_manager.reset()
        self.token_optimizer.reset()
        self.logger.info("Assistant reset")


def main():
    """
    Main function for running the assistant interactively.
    
    Provides a simple CLI for testing the assistant.
    """
    print("=" * 60)
    print("Claude-Based AI Assistant")
    print("Token-Optimized Conversation System")
    print("=" * 60)
    print()
    
    # Initialize assistant
    assistant = ClaudeAssistant()
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'stats':
                stats = assistant.get_stats()
                print("\n" + "=" * 40)
                print("Statistics:")
                print("=" * 40)
                for agent, agent_stats in stats.items():
                    print(f"\n{agent}:")
                    for key, value in agent_stats.items():
                        print(f"  {key}: {value}")
                continue
            
            if user_input.lower() == 'recommend':
                recommendations = assistant.get_recommendations()
                print("\n" + "=" * 40)
                print("Optimization Recommendations:")
                print("=" * 40)
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i}. [{rec.category}] (Priority: {rec.priority})")
                    print(f"   {rec.description}")
                    if rec.potential_savings > 0:
                        print(f"   Potential savings: {rec.potential_savings} tokens")
                continue
            
            if user_input.lower() == 'reset':
                assistant.reset()
                print("\nContext reset successfully.")
                continue
            
            # Process input
            response = assistant.process(user_input)
            print(f"\nAssistant: {response}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
