"""
Prompt template utilities for consistent, token-efficient prompts.

Purpose:
- Define reusable prompt templates
- Support variable interpolation
- Track template versions
- Minimize redundant text

Token Savings: 20-40% through template reuse
"""

from typing import Dict, Any, List
from string import Template


class PromptTemplate:
    """
    Manages prompt templates for consistent AI interactions.
    
    Features:
    - Variable interpolation
    - Template registry
    - Version tracking
    - Token-efficient formatting
    
    Usage:
        # Define template
        template = PromptTemplate(
            name='question_answering',
            template='Answer this question concisely: ${question}',
            version='1.0'
        )
        
        # Render with variables
        prompt = template.render(question='What is AI?')
    """
    
    # Registry of all templates
    _registry: Dict[str, 'PromptTemplate'] = {}
    
    def __init__(self, name: str, template: str, 
                 version: str = '1.0', metadata: Dict[str, Any] = None):
        """
        Create a prompt template.
        
        Args:
            name: Template name/identifier
            template: Template string with ${variable} placeholders
            version: Template version
            metadata: Optional metadata (description, author, etc.)
        """
        self.name = name
        self.template = template
        self.version = version
        self.metadata = metadata or {}
        self._template_obj = Template(template)
        
        # Register template
        PromptTemplate._registry[name] = self
    
    def render(self, **variables) -> str:
        """
        Render template with variables.
        
        Args:
            **variables: Variables to interpolate into template
        
        Returns:
            Rendered prompt string
        
        Raises:
            KeyError: If required variable is missing
        """
        try:
            return self._template_obj.substitute(variables)
        except KeyError as e:
            raise ValueError(
                f"Missing required variable for template '{self.name}': {e}"
            )
    
    def safe_render(self, **variables) -> str:
        """
        Render template with safe substitution (leaves missing vars as-is).
        
        Args:
            **variables: Variables to interpolate
        
        Returns:
            Rendered prompt string
        """
        return self._template_obj.safe_substitute(variables)
    
    @classmethod
    def get(cls, name: str) -> 'PromptTemplate':
        """
        Get template from registry.
        
        Args:
            name: Template name
        
        Returns:
            PromptTemplate instance
        
        Raises:
            KeyError: If template not found
        """
        if name not in cls._registry:
            raise KeyError(f"Template '{name}' not found in registry")
        return cls._registry[name]
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """
        List all registered templates.
        
        Returns:
            List of template names
        """
        return list(cls._registry.keys())
    
    def __str__(self) -> str:
        return f"PromptTemplate(name='{self.name}', version='{self.version}')"


# Common templates
# These are token-optimized templates for common tasks

# Question answering (concise mode)
PromptTemplate(
    name='qa_concise',
    template=(
        'Question: ${question}\n'
        'Answer concisely in 1-2 sentences.'
    ),
    version='1.0',
    metadata={'category': 'question_answering', 'style': 'concise'}
)

# Question answering (detailed mode)
PromptTemplate(
    name='qa_detailed',
    template=(
        'Question: ${question}\n'
        'Provide a detailed answer with examples.'
    ),
    version='1.0',
    metadata={'category': 'question_answering', 'style': 'detailed'}
)

# Command execution
PromptTemplate(
    name='command',
    template=(
        'Execute this command: ${command}\n'
        'Confirm what you will do, then proceed.'
    ),
    version='1.0',
    metadata={'category': 'action'}
)

# Context-aware conversation
PromptTemplate(
    name='conversation',
    template=(
        'Previous context: ${context}\n'
        'User: ${message}\n'
        'Respond naturally and helpfully.'
    ),
    version='1.0',
    metadata={'category': 'conversation'}
)

# Summarization
PromptTemplate(
    name='summarize',
    template=(
        'Summarize the following in ${max_sentences} sentences:\n'
        '${text}'
    ),
    version='1.0',
    metadata={'category': 'summarization'}
)

# Classification
PromptTemplate(
    name='classify',
    template=(
        'Classify this text into one of: ${categories}\n'
        'Text: ${text}\n'
        'Return only the category name.'
    ),
    version='1.0',
    metadata={'category': 'classification'}
)
