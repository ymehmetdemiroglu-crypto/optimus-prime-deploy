"""
Search Term Analysis with NLP.
Extracts insights from search terms using text mining techniques.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchTermInsight:
    """Insight derived from search term analysis."""
    term: str
    pattern: str
    frequency: int
    avg_acos: float
    total_spend: float
    total_sales: float
    recommendation: str
    confidence: float


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    # Common English stop words
    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    # Amazon-specific terms to filter
    AMAZON_STOP_WORDS = {
        'amazon', 'prime', 'buy', 'purchase', 'order', 'cheap', 'best',
        'top', 'rated', 'review', 'reviews', 'free', 'shipping'
    }
    
    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenize text into words."""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]
    
    @classmethod
    def remove_stop_words(cls, tokens: List[str]) -> List[str]:
        """Remove stop words from tokens."""
        all_stops = cls.STOP_WORDS | cls.AMAZON_STOP_WORDS
        return [t for t in tokens if t not in all_stops]
    
    @classmethod
    def extract_ngrams(cls, tokens: List[str], n: int = 2) -> List[str]:
        """Extract n-grams from tokens."""
        if len(tokens) < n:
            return []
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


class TFIDFVectorizer:
    """
    Simple TF-IDF implementation for search term analysis.
    """
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
    
    def fit(self, documents: List[str]):
        """Fit TF-IDF on documents."""
        # Build vocabulary
        word_doc_count = Counter()
        all_words = Counter()
        
        for doc in documents:
            tokens = set(TextPreprocessor.tokenize(doc))
            for token in tokens:
                word_doc_count[token] += 1
            all_words.update(TextPreprocessor.tokenize(doc))
        
        # Select top features
        top_words = [w for w, _ in all_words.most_common(self.max_features)]
        self.vocabulary = {word: idx for idx, word in enumerate(top_words)}
        
        # Calculate IDF
        n_docs = len(documents)
        for word, count in word_doc_count.items():
            if word in self.vocabulary:
                self.idf[word] = np.log(n_docs / (1 + count))
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF vectors."""
        vectors = np.zeros((len(documents), len(self.vocabulary)))
        
        for i, doc in enumerate(documents):
            tokens = TextPreprocessor.tokenize(doc)
            token_counts = Counter(tokens)
            doc_length = len(tokens) or 1
            
            for token, count in token_counts.items():
                if token in self.vocabulary:
                    tf = count / doc_length
                    idf = self.idf.get(token, 0)
                    vectors[i, self.vocabulary[token]] = tf * idf
        
        return vectors
    
    def get_top_terms(self, vector: np.ndarray, n: int = 10) -> List[Tuple[str, float]]:
        """Get top terms from a TF-IDF vector."""
        idx_to_word = {idx: word for word, idx in self.vocabulary.items()}
        top_indices = np.argsort(vector)[-n:][::-1]
        return [(idx_to_word[idx], vector[idx]) for idx in top_indices if vector[idx] > 0]


class SearchTermAnalyzer:
    """
    Comprehensive search term analysis for PPC optimization.
    """

    def __init__(self):
        self.tfidf = TFIDFVectorizer(max_features=500)
        self._intent_classifier = None

    @property
    def intent_classifier(self):
        """Lazy-load the intent classifier to avoid circular imports."""
        if self._intent_classifier is None:
            try:
                from app.services.ml.intent_classifier import get_intent_classifier
                self._intent_classifier = get_intent_classifier()
            except Exception as e:
                logger.warning(f"Intent classifier unavailable: {e}")
        return self._intent_classifier
    
    def analyze_search_terms(
        self,
        search_term_data: List[Dict[str, Any]],
        target_acos: float = 25.0
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of search terms.
        
        Args:
            search_term_data: List of {term, impressions, clicks, spend, sales, orders}
            target_acos: Target ACoS for recommendations
        """
        if not search_term_data:
            return {'error': 'No search term data provided'}
        
        # Basic statistics
        terms = [d.get('term', '') for d in search_term_data]
        
        # Extract patterns
        word_performance = self._analyze_word_performance(search_term_data)
        ngram_performance = self._analyze_ngram_performance(search_term_data)

        # Categorize terms
        categories = self._categorize_terms(search_term_data, target_acos)

        # Find patterns
        patterns = self._find_patterns(search_term_data)

        # Intent analysis (Rufus/Cosmo)
        intent_breakdown = self._analyze_intents(search_term_data)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            word_performance, categories, target_acos
        )

        return {
            'summary': {
                'total_terms': len(terms),
                'unique_words': len(word_performance),
                'avg_term_length': round(np.mean([len(t.split()) for t in terms]), 1)
            },
            'word_performance': dict(sorted(
                word_performance.items(),
                key=lambda x: x[1].get('total_sales', 0),
                reverse=True
            )[:50]),
            'ngram_performance': dict(sorted(
                ngram_performance.items(),
                key=lambda x: x[1].get('total_sales', 0),
                reverse=True
            )[:30]),
            'categories': categories,
            'patterns': patterns,
            'intent_breakdown': intent_breakdown,
            'recommendations': recommendations
        }
    
    def _analyze_word_performance(
        self,
        data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance by individual words."""
        word_stats = defaultdict(lambda: {
            'impressions': 0, 'clicks': 0, 'spend': 0, 'sales': 0, 'count': 0
        })
        
        for record in data:
            term = record.get('term', '')
            tokens = TextPreprocessor.tokenize(term)
            tokens = TextPreprocessor.remove_stop_words(tokens)
            
            for token in tokens:
                word_stats[token]['impressions'] += record.get('impressions', 0)
                word_stats[token]['clicks'] += record.get('clicks', 0)
                word_stats[token]['spend'] += record.get('spend', 0)
                word_stats[token]['sales'] += record.get('sales', 0)
                word_stats[token]['count'] += 1
        
        # Calculate derived metrics
        for word, stats in word_stats.items():
            if stats['sales'] > 0:
                stats['acos'] = round(stats['spend'] / stats['sales'] * 100, 2)
            else:
                stats['acos'] = float('inf') if stats['spend'] > 0 else 0
            
            if stats['impressions'] > 0:
                stats['ctr'] = round(stats['clicks'] / stats['impressions'] * 100, 2)
            else:
                stats['ctr'] = 0
            
            stats['total_sales'] = round(stats['sales'], 2)
            stats['total_spend'] = round(stats['spend'], 2)
        
        return dict(word_stats)
    
    def _analyze_ngram_performance(
        self,
        data: List[Dict[str, Any]],
        n: int = 2
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance by n-grams."""
        ngram_stats = defaultdict(lambda: {
            'impressions': 0, 'clicks': 0, 'spend': 0, 'sales': 0, 'count': 0
        })
        
        for record in data:
            term = record.get('term', '')
            tokens = TextPreprocessor.tokenize(term)
            ngrams = TextPreprocessor.extract_ngrams(tokens, n)
            
            for ngram in ngrams:
                ngram_stats[ngram]['impressions'] += record.get('impressions', 0)
                ngram_stats[ngram]['clicks'] += record.get('clicks', 0)
                ngram_stats[ngram]['spend'] += record.get('spend', 0)
                ngram_stats[ngram]['sales'] += record.get('sales', 0)
                ngram_stats[ngram]['count'] += 1
        
        # Calculate ACoS
        for ngram, stats in ngram_stats.items():
            if stats['sales'] > 0:
                stats['acos'] = round(stats['spend'] / stats['sales'] * 100, 2)
            else:
                stats['acos'] = float('inf') if stats['spend'] > 0 else 0
            
            stats['total_sales'] = round(stats['sales'], 2)
        
        return dict(ngram_stats)
    
    def _categorize_terms(
        self,
        data: List[Dict[str, Any]],
        target_acos: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize search terms by performance."""
        categories = {
            'winners': [],      # Low ACoS, good volume
            'potential': [],    # Low ACoS, low volume
            'test': [],         # No conversions, low spend
            'losers': [],       # High ACoS
            'expensive': []     # High spend, no sales
        }
        
        for record in data:
            term = record.get('term', '')
            spend = record.get('spend', 0)
            sales = record.get('sales', 0)
            clicks = record.get('clicks', 0)
            
            acos = (spend / sales * 100) if sales > 0 else float('inf')
            
            term_info = {
                'term': term,
                'spend': round(spend, 2),
                'sales': round(sales, 2),
                'clicks': clicks,
                'acos': round(acos, 2) if acos != float('inf') else None
            }
            
            if sales > 0 and acos < target_acos:
                if clicks >= 10:
                    categories['winners'].append(term_info)
                else:
                    categories['potential'].append(term_info)
            elif sales == 0 and spend < 10:
                categories['test'].append(term_info)
            elif sales == 0 and spend >= 10:
                categories['expensive'].append(term_info)
            elif acos >= target_acos * 1.5:
                categories['losers'].append(term_info)
        
        # Sort each category
        for cat in categories:
            if cat in ['winners', 'potential']:
                categories[cat].sort(key=lambda x: x['sales'], reverse=True)
            else:
                categories[cat].sort(key=lambda x: x['spend'], reverse=True)
            
            categories[cat] = categories[cat][:20]  # Top 20
        
        return categories
    
    def _find_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find patterns in search terms."""
        terms = [d.get('term', '') for d in data]
        
        # Word length patterns
        word_counts = [len(t.split()) for t in terms]
        
        # Brand/Product indicators
        brand_indicators = ['brand', 'official', 'genuine', 'authentic', 'original']
        product_indicators = ['for', 'with', 'size', 'color', 'pack']
        
        brand_terms = sum(1 for t in terms if any(b in t.lower() for b in brand_indicators))
        product_terms = sum(1 for t in terms if any(p in t.lower() for p in product_indicators))
        
        # Question patterns
        question_terms = sum(1 for t in terms if any(q in t.lower() for q in ['how', 'what', 'where', 'why', 'which', 'best']))
        
        return {
            'avg_word_count': round(np.mean(word_counts), 1),
            'single_word_percentage': round(sum(1 for c in word_counts if c == 1) / len(terms) * 100, 1),
            'long_tail_percentage': round(sum(1 for c in word_counts if c >= 4) / len(terms) * 100, 1),
            'brand_related': brand_terms,
            'product_modifier': product_terms,
            'question_based': question_terms
        }
    
    def _analyze_intents(
        self,
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Classify search terms by shopping intent (Rufus/Cosmo optimization)."""
        classifier = self.intent_classifier
        if classifier is None:
            return {'available': False, 'reason': 'Intent classifier not loaded'}

        intent_buckets: Dict[str, List[Dict[str, Any]]] = {
            'transactional': [],
            'informational_rufus': [],
            'navigational': [],
            'discovery': [],
        }
        intent_spend: Dict[str, float] = {k: 0.0 for k in intent_buckets}
        intent_sales: Dict[str, float] = {k: 0.0 for k in intent_buckets}

        for record in data:
            term = record.get('term', '')
            result = classifier.classify(term)
            intent_key = result.intent.value

            entry = {
                'term': term,
                'intent': intent_key,
                'confidence': round(result.confidence, 3),
                'spend': round(record.get('spend', 0), 2),
                'sales': round(record.get('sales', 0), 2),
                'clicks': record.get('clicks', 0),
            }
            intent_buckets[intent_key].append(entry)
            intent_spend[intent_key] += record.get('spend', 0)
            intent_sales[intent_key] += record.get('sales', 0)

        # Build summary
        total_spend = sum(intent_spend.values()) or 1
        total_sales = sum(intent_sales.values()) or 1

        summary = {}
        for intent_key in intent_buckets:
            count = len(intent_buckets[intent_key])
            spend = intent_spend[intent_key]
            sales = intent_sales[intent_key]
            summary[intent_key] = {
                'count': count,
                'spend': round(spend, 2),
                'sales': round(sales, 2),
                'acos': round(spend / sales * 100, 2) if sales > 0 else None,
                'spend_share': round(spend / total_spend * 100, 1),
                'sales_share': round(sales / total_sales * 100, 1),
            }

        # Top 5 Rufus terms by spend (these need special threshold treatment)
        rufus_top = sorted(
            intent_buckets['informational_rufus'],
            key=lambda x: x['spend'],
            reverse=True
        )[:10]

        return {
            'available': True,
            'summary': summary,
            'rufus_top_terms': rufus_top,
            'total_rufus_spend': round(intent_spend['informational_rufus'], 2),
            'total_discovery_spend': round(intent_spend['discovery'], 2),
        }

    def _generate_recommendations(
        self,
        word_performance: Dict[str, Dict[str, float]],
        categories: Dict[str, List[Dict[str, Any]]],
        target_acos: float
    ) -> List[SearchTermInsight]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # High-performing words to target
        for word, stats in sorted(
            word_performance.items(),
            key=lambda x: x[1].get('total_sales', 0),
            reverse=True
        )[:10]:
            if stats.get('acos', 100) < target_acos and stats.get('total_sales', 0) > 0:
                recommendations.append(SearchTermInsight(
                    term=word,
                    pattern='high_performer',
                    frequency=stats['count'],
                    avg_acos=stats.get('acos', 0),
                    total_spend=stats['total_spend'],
                    total_sales=stats['total_sales'],
                    recommendation=f"Add '{word}' as exact match keyword",
                    confidence=0.85
                ))
        
        # Words to negate
        for word, stats in sorted(
            word_performance.items(),
            key=lambda x: x[1].get('total_spend', 0),
            reverse=True
        )[:20]:
            if stats.get('acos', 0) > target_acos * 2 and stats.get('total_spend', 0) > 20:
                recommendations.append(SearchTermInsight(
                    term=word,
                    pattern='negative_candidate',
                    frequency=stats['count'],
                    avg_acos=stats.get('acos', 0),
                    total_spend=stats['total_spend'],
                    total_sales=stats['total_sales'],
                    recommendation=f"Consider adding '{word}' as negative keyword",
                    confidence=0.75
                ))
        
        return recommendations
    
    def find_negative_keywords(
        self,
        search_term_data: List[Dict[str, Any]],
        min_spend: float = 10.0,
        max_conversions: int = 0
    ) -> List[Dict[str, Any]]:
        """Find potential negative keywords."""
        negatives = []
        
        for record in search_term_data:
            spend = record.get('spend', 0)
            orders = record.get('orders', 0)
            
            if spend >= min_spend and orders <= max_conversions:
                negatives.append({
                    'term': record.get('term'),
                    'spend': round(spend, 2),
                    'clicks': record.get('clicks', 0),
                    'impressions': record.get('impressions', 0),
                    'reason': 'High spend with no conversions'
                })
        
        return sorted(negatives, key=lambda x: x['spend'], reverse=True)
    
    def find_exact_match_candidates(
        self,
        search_term_data: List[Dict[str, Any]],
        target_acos: float = 25.0,
        min_orders: int = 2
    ) -> List[Dict[str, Any]]:
        """Find search terms that should be exact match keywords."""
        candidates = []
        
        for record in search_term_data:
            spend = record.get('spend', 0)
            sales = record.get('sales', 0)
            orders = record.get('orders', 0)
            
            if orders >= min_orders and sales > 0:
                acos = spend / sales * 100
                
                if acos < target_acos:
                    candidates.append({
                        'term': record.get('term'),
                        'orders': orders,
                        'sales': round(sales, 2),
                        'acos': round(acos, 2),
                        'clicks': record.get('clicks', 0),
                        'reason': 'Strong performer - promote to exact match'
                    })
        
        return sorted(candidates, key=lambda x: x['sales'], reverse=True)
