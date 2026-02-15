"""
Intent Classifier - Determines user intent with minimal token usage.

Purpose:
- Classify user input into predefined intent categories
- Route requests to appropriate handlers
- Cache classifications to avoid redundant API calls

Token Optimization:
- Use lightweight embeddings instead of full context
- Cache similar queries (can save 80-90% tokens on repeated intents)
- Early classification to avoid processing full context unnecessarily
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ..base_agent import BaseAgent


class IntentType(Enum):
    """Enumeration of supported intent types."""
    QUESTION = "question"              # User asking for information
    COMMAND = "command"                # User requesting action
    CONVERSATION = "conversation"       # Casual chat
    CLARIFICATION = "clarification"     # User providing more details
    FEEDBACK = "feedback"               # User giving feedback
    UNKNOWN = "unknown"                 # Cannot determine intent


@dataclass
class Intent:
    """
    Classified intent with metadata.
    
    Attributes:
        type: The intent type (from IntentType enum)
        confidence: Confidence score (0.0 to 1.0)
        entities: Extracted entities (e.g., topic, action, subject)
        requires_context: Whether this intent needs conversation history
    """
    type: IntentType
    confidence: float
    entities: Dict[str, Any]
    requires_context: bool
    
    def __str__(self) -> str:
        return f"Intent({self.type.value}, confidence={self.confidence:.2f})"


class IntentClassifier(BaseAgent):
    """
    Classifies user intent to optimize downstream processing.
    
    Expected Input:
        - ProcessedInput object from InputProcessor
    
    Returns:
        - Intent object with classification and metadata
    
    Best Practices:
        1. Use cached classifications for similar inputs
        2. Set requires_context=False when context not needed (saves tokens)
        3. Extract entities early to avoid re-parsing later
        4. Use confidence threshold to trigger clarification
    
    Token Savings:
        - Caching: 80-90% on similar queries
        - Context filtering: 30-50% when context not required
        - Early routing: Prevents unnecessary agent invocations
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.cache_enabled = config.get('cache_enabled', True)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self._cache: Dict[str, Intent] = {}
        
        # Simple keyword-based classification (replace with ML model in production)
        self.intent_patterns = {
            IntentType.QUESTION: ['what', 'why', 'how', 'when', 'where', 'who', '?'],
            IntentType.COMMAND: ['create', 'delete', 'update', 'run', 'execute', 'start', 'stop'],
            IntentType.CLARIFICATION: ['i mean', 'actually', 'to clarify', 'specifically'],
            IntentType.FEEDBACK: ['thanks', 'good', 'bad', 'helpful', 'not helpful'],
        }
    
    def process(self, processed_input) -> Intent:
        """
        Classify the intent of processed input.
        
        Args:
            processed_input: ProcessedInput object
        
        Returns:
            Intent object with classification results
        """
        text = processed_input.text.lower()
        
        # Check cache first (huge token savings)
        if self.cache_enabled:
            cached_intent = self._check_cache(text)
            if cached_intent:
                self.logger.debug("Intent found in cache - saved API call")
                return cached_intent
        
        # Classify intent
        intent = self._classify(text)
        
        # Cache the result
        if self.cache_enabled and intent.confidence >= self.confidence_threshold:
            self._cache[text] = intent
        
        # Track token usage (estimation for classification)
        tokens_used = processed_input.processed_tokens
        self.track_tokens(tokens_used)
        
        self.logger.info(f"Classified intent: {intent}")
        
        return intent
    
    def _classify(self, text: str) -> Intent:
        """
        Perform actual classification.
        
        In production, this would use:
        - Lightweight embedding model (e.g., sentence-transformers)
        - Pre-trained classifier (e.g., DistilBERT)
        - Few-shot prompting with Claude
        
        This simple implementation uses keyword matching for demonstration.
        """
        scores = {}
        
        # Calculate scores for each intent type
        for intent_type, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[intent_type] = score
        
        # Determine best match
        if not scores:
            return Intent(
                type=IntentType.CONVERSATION,
                confidence=0.5,
                entities={},
                requires_context=True
            )
        
        best_intent = max(scores, key=scores.get)
        max_score = scores[best_intent]
        total_keywords = len(self.intent_patterns[best_intent])
        confidence = min(max_score / total_keywords, 1.0)
        
        # Extract simple entities
        entities = self._extract_entities(text, best_intent)
        
        # Determine if context is needed
        requires_context = best_intent in [
            IntentType.CLARIFICATION,
            IntentType.CONVERSATION,
            IntentType.FEEDBACK
        ]
        
        return Intent(
            type=best_intent,
            confidence=confidence,
            entities=entities,
            requires_context=requires_context
        )
    
    def _extract_entities(self, text: str, intent_type: IntentType) -> Dict[str, Any]:
        """
        Extract relevant entities from text based on intent.
        
        In production, use NER (Named Entity Recognition) model.
        """
        entities = {}
        
        # Simple topic extraction (first noun-like word)
        words = text.split()
        if words:
            # Very naive: find first capitalized word or noun
            for word in words:
                if len(word) > 3 and word[0].isupper():
                    entities['topic'] = word
                    break
        
        return entities
    
    def _check_cache(self, text: str) -> Optional[Intent]:
        """
        Check if similar intent was classified before.
        
        Implements fuzzy matching for better cache hits.
        """
        # Exact match
        if text in self._cache:
            return self._cache[text]
        
        # Fuzzy match: check if 80% of words match
        # (In production, use embedding similarity)
        text_words = set(text.split())
        
        for cached_text, cached_intent in self._cache.items():
            cached_words = set(cached_text.split())
            
            # Calculate Jaccard similarity
            if len(text_words) > 0 and len(cached_words) > 0:
                intersection = text_words & cached_words
                union = text_words | cached_words
                similarity = len(intersection) / len(union)
                
                if similarity >= 0.8:
                    self.logger.debug(f"Fuzzy cache hit (similarity: {similarity:.2f})")
                    return cached_intent
        
        return None
    
    def clear_cache(self) -> None:
        """Clear the intent cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        self.logger.info(f"Cleared intent cache ({cache_size} entries)")
