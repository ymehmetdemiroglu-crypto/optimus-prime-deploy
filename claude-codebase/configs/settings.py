"""
Application settings and configuration.

Load configuration from YAML files and environment variables.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    max_context_tokens: int
    default_output_tokens: int
    cost_per_million_input: float
    cost_per_million_output: float


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Model settings
    default_model: str = 'claude-sonnet-4-5-20250929'
    models: Dict[str, ModelConfig] = None
    
    # Operation budgets
    operation_budgets: Dict[str, int] = None
    
    # Context settings
    context_max_tokens: int = 50000
    context_window_size: int = 20
    context_summarization_threshold: int = 40000
    context_auto_compress: bool = True
    
    # Caching settings
    caching_enabled: bool = True
    cache_max_size: int = 1000
    cache_default_ttl: int = 3600
    
    # Logging
    log_level: str = 'INFO'
    log_file: str = None
    log_json_format: bool = False
    
    # API keys (from environment)
    anthropic_api_key: str = None
    
    @classmethod
    def load_from_yaml(cls, config_path: str = 'configs/token_limits.yaml') -> 'AppConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
        
        Returns:
            AppConfig instance
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Parse model configs
        models = {}
        for model_name, model_data in config_data.get('models', {}).items():
            models[model_name] = ModelConfig(
                max_context_tokens=model_data['max_context_tokens'],
                default_output_tokens=model_data['default_output_tokens'],
                cost_per_million_input=model_data['cost_per_million_input'],
                cost_per_million_output=model_data['cost_per_million_output']
            )
        
        # Get API key from environment
        api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Create config instance
        context_config = config_data.get('context', {})
        caching_config = config_data.get('caching', {})
        
        return cls(
            models=models,
            operation_budgets=config_data.get('operation_budgets', {}),
            context_max_tokens=context_config.get('max_tokens', 50000),
            context_window_size=context_config.get('window_size', 20),
            context_summarization_threshold=context_config.get('summarization_threshold', 40000),
            context_auto_compress=context_config.get('auto_compress', True),
            caching_enabled=caching_config.get('enabled', True),
            cache_max_size=caching_config.get('max_cache_size', 1000),
            cache_default_ttl=caching_config.get('default_ttl_seconds', 3600),
            anthropic_api_key=api_key
        )


# Global config instance
_config: AppConfig = None


def get_config() -> AppConfig:
    """
    Get global configuration instance.
    
    Loads config on first call, returns cached instance on subsequent calls.
    
    Returns:
        AppConfig instance
    """
    global _config
    
    if _config is None:
        _config = AppConfig.load_from_yaml()
    
    return _config


def reload_config() -> AppConfig:
    """
    Force reload configuration from file.
    
    Returns:
        New AppConfig instance
    """
    global _config
    _config = AppConfig.load_from_yaml()
    return _config
