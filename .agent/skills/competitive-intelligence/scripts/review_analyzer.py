"""
Review Analyzer for Competitive Intelligence
Analyzes competitor reviews for sentiment and actionable insights.
"""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import Counter

@dataclass
class Review:
    review_id: str
    asin: str
    rating: int  # 1-5
    title: str
    body: str
    date: datetime
    verified_purchase: bool
    helpful_votes: int = 0

class ReviewAnalyzer:
    def __init__(self):
        # Simple sentiment keywords (in production, use NLTK/transformers)
        self.positive_words = {
            "great", "excellent", "amazing", "love", "perfect", "best", 
            "fantastic", "wonderful", "awesome", "recommend", "quality",
            "comfortable", "works", "easy", "fast", "value", "worth"
        }
        self.negative_words = {
            "bad", "terrible", "worst", "hate", "broken", "cheap", "poor",
            "disappointed", "waste", "return", "refund", "defective",
            "uncomfortable", "difficult", "slow", "expensive", "overpriced"
        }
        
        # Feature extraction patterns
        self.feature_patterns = {
            "battery": r"battery|charge|charging|hours|life",
            "sound": r"sound|audio|bass|treble|volume|music",
            "comfort": r"comfort|fit|ear|head|wear|wearing",
            "build": r"build|quality|material|plastic|metal|durable",
            "connection": r"bluetooth|connect|pair|wireless|range",
            "price": r"price|cost|value|money|worth|expensive|cheap"
        }
    
    def analyze_review(self, review: Review) -> Dict[str, Any]:
        """Analyze a single review."""
        text = f"{review.title} {review.body}".lower()
        words = set(re.findall(r'\b\w+\b', text))
        
        # Sentiment scoring
        pos_count = len(words & self.positive_words)
        neg_count = len(words & self.negative_words)
        
        if pos_count > neg_count:
            sentiment = "positive"
            sentiment_score = min(1.0, (pos_count - neg_count) / 5)
        elif neg_count > pos_count:
            sentiment = "negative"
            sentiment_score = max(-1.0, (pos_count - neg_count) / 5)
        else:
            sentiment = "neutral"
            sentiment_score = 0
        
        # Feature mentions
        features_mentioned = []
        for feature, pattern in self.feature_patterns.items():
            if re.search(pattern, text):
                features_mentioned.append(feature)
        
        return {
            "review_id": review.review_id,
            "rating": review.rating,
            "sentiment": sentiment,
            "sentiment_score": round(sentiment_score, 2),
            "features_mentioned": features_mentioned,
            "positive_words_found": list(words & self.positive_words),
            "negative_words_found": list(words & self.negative_words),
            "is_helpful": review.helpful_votes > 5
        }
    
    def analyze_product_reviews(self, reviews: List[Review]) -> Dict[str, Any]:
        """Analyze all reviews for a product."""
        if not reviews:
            return {"error": "No reviews to analyze"}
        
        analyses = [self.analyze_review(r) for r in reviews]
        
        # Aggregate metrics
        avg_rating = sum(r.rating for r in reviews) / len(reviews)
        sentiment_dist = Counter(a["sentiment"] for a in analyses)
        
        # Feature sentiment breakdown
        feature_sentiment = {}
        for feature in self.feature_patterns.keys():
            feature_reviews = [a for a in analyses if feature in a["features_mentioned"]]
            if feature_reviews:
                avg_sent = sum(a["sentiment_score"] for a in feature_reviews) / len(feature_reviews)
                feature_sentiment[feature] = {
                    "mentions": len(feature_reviews),
                    "avg_sentiment": round(avg_sent, 2),
                    "assessment": "positive" if avg_sent > 0.2 else "negative" if avg_sent < -0.2 else "mixed"
                }
        
        # Extract common complaints and praise
        all_negative_words = []
        all_positive_words = []
        for a in analyses:
            all_negative_words.extend(a["negative_words_found"])
            all_positive_words.extend(a["positive_words_found"])
        
        top_complaints = Counter(all_negative_words).most_common(5)
        top_praise = Counter(all_positive_words).most_common(5)
        
        # Review velocity (reviews per month approximation)
        if len(reviews) >= 2:
            date_range = (max(r.date for r in reviews) - min(r.date for r in reviews)).days
            velocity = len(reviews) / max(date_range / 30, 1)
        else:
            velocity = len(reviews)
        
        return {
            "total_reviews": len(reviews),
            "average_rating": round(avg_rating, 2),
            "rating_distribution": {
                5: sum(1 for r in reviews if r.rating == 5),
                4: sum(1 for r in reviews if r.rating == 4),
                3: sum(1 for r in reviews if r.rating == 3),
                2: sum(1 for r in reviews if r.rating == 2),
                1: sum(1 for r in reviews if r.rating == 1)
            },
            "sentiment_distribution": {
                "positive": sentiment_dist.get("positive", 0),
                "neutral": sentiment_dist.get("neutral", 0),
                "negative": sentiment_dist.get("negative", 0)
            },
            "sentiment_percentage": {
                "positive": round(sentiment_dist.get("positive", 0) / len(analyses) * 100, 1),
                "negative": round(sentiment_dist.get("negative", 0) / len(analyses) * 100, 1)
            },
            "feature_analysis": feature_sentiment,
            "top_complaints": [{"word": w, "count": c} for w, c in top_complaints],
            "top_praise": [{"word": w, "count": c} for w, c in top_praise],
            "review_velocity": round(velocity, 1),
            "verified_purchase_rate": round(sum(1 for r in reviews if r.verified_purchase) / len(reviews) * 100, 1)
        }
    
    def compare_products(self, product_reviews: Dict[str, List[Review]]) -> Dict[str, Any]:
        """Compare reviews across multiple products."""
        comparisons = {}
        
        for asin, reviews in product_reviews.items():
            analysis = self.analyze_product_reviews(reviews)
            comparisons[asin] = {
                "avg_rating": analysis.get("average_rating", 0),
                "total_reviews": analysis.get("total_reviews", 0),
                "positive_pct": analysis.get("sentiment_percentage", {}).get("positive", 0),
                "top_complaint": analysis.get("top_complaints", [{}])[0].get("word", "none") if analysis.get("top_complaints") else "none",
                "top_praise": analysis.get("top_praise", [{}])[0].get("word", "none") if analysis.get("top_praise") else "none"
            }
        
        # Rank by rating
        ranked = sorted(comparisons.items(), key=lambda x: x[1]["avg_rating"], reverse=True)
        
        return {
            "comparison": comparisons,
            "ranking": [{"asin": asin, "rank": i+1, **data} for i, (asin, data) in enumerate(ranked)],
            "best_rated": ranked[0][0] if ranked else None,
            "most_reviewed": max(comparisons.items(), key=lambda x: x[1]["total_reviews"])[0] if comparisons else None
        }
    
    def extract_actionable_insights(self, reviews: List[Review], your_asin: str) -> Dict[str, Any]:
        """Extract actionable competitive insights from reviews."""
        analysis = self.analyze_product_reviews(reviews)
        
        insights = []
        
        # Check feature weaknesses
        for feature, data in analysis.get("feature_analysis", {}).items():
            if data["assessment"] == "negative":
                insights.append({
                    "type": "competitor_weakness",
                    "feature": feature,
                    "sentiment": data["avg_sentiment"],
                    "action": f"Highlight your {feature} advantage in advertising",
                    "priority": "high" if data["mentions"] > 5 else "medium"
                })
        
        # Check common complaints
        for complaint in analysis.get("top_complaints", [])[:3]:
            insights.append({
                "type": "market_pain_point",
                "keyword": complaint["word"],
                "frequency": complaint["count"],
                "action": f"Address '{complaint['word']}' concern in your listing",
                "priority": "high" if complaint["count"] > 10 else "medium"
            })
        
        return {
            "competitor_asin": your_asin,
            "insights_count": len(insights),
            "actionable_insights": insights,
            "summary": self._generate_insight_summary(insights)
        }
    
    def _generate_insight_summary(self, insights: List[Dict]) -> str:
        weaknesses = [i for i in insights if i["type"] == "competitor_weakness"]
        pain_points = [i for i in insights if i["type"] == "market_pain_point"]
        
        summary_parts = []
        if weaknesses:
            summary_parts.append(f"Found {len(weaknesses)} feature weaknesses to exploit")
        if pain_points:
            summary_parts.append(f"Identified {len(pain_points)} market pain points to address")
        
        return ". ".join(summary_parts) if summary_parts else "No significant insights found"


if __name__ == "__main__":
    analyzer = ReviewAnalyzer()
    
    # Create sample reviews
    sample_reviews = [
        Review("R001", "B0TEST001", 5, "Great headphones!", 
               "These are amazing! The sound quality is fantastic and battery lasts forever.",
               datetime.now(), True, 15),
        Review("R002", "B0TEST001", 2, "Disappointed", 
               "The bluetooth connection is terrible. Keeps disconnecting. Returning for refund.",
               datetime.now(), True, 8),
        Review("R003", "B0TEST001", 4, "Good value", 
               "Comfortable fit and good sound for the price. Battery could be better.",
               datetime.now(), True, 3),
        Review("R004", "B0TEST001", 1, "Broke after 2 weeks", 
               "Cheap plastic build quality. Very disappointed with this purchase.",
               datetime.now(), False, 20),
        Review("R005", "B0TEST001", 5, "Love these!", 
               "Best headphones I've owned. Sound is excellent and very comfortable.",
               datetime.now(), True, 5),
    ]
    
    # Analyze
    analysis = analyzer.analyze_product_reviews(sample_reviews)
    
    print("Review Analysis:")
    print(f"  Average Rating: {analysis['average_rating']}")
    print(f"  Sentiment: {analysis['sentiment_percentage']['positive']}% positive, {analysis['sentiment_percentage']['negative']}% negative")
    print(f"  Top Complaints: {[c['word'] for c in analysis['top_complaints'][:3]]}")
    print(f"  Top Praise: {[p['word'] for p in analysis['top_praise'][:3]]}")
    
    # Extract insights
    insights = analyzer.extract_actionable_insights(sample_reviews, "B0COMP001")
    print(f"\nActionable Insights: {insights['summary']}")
