"""
Customer Segmenter for Data Scientist Skill
RFM analysis and behavioral clustering for customer segmentation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import math

@dataclass
class CustomerOrder:
    customer_id: str
    order_id: str
    order_date: datetime
    order_value: float
    products: List[str]
    is_repeat: bool = False

@dataclass
class CustomerSegment:
    segment_name: str
    customer_ids: List[str]
    count: int
    percentage: float
    avg_recency_days: float
    avg_frequency: float
    avg_monetary: float
    characteristics: Dict[str, Any]
    recommended_strategy: str
    ad_targeting: str

class CustomerSegmenter:
    """
    Customer segmentation using RFM (Recency, Frequency, Monetary) analysis.
    """
    
    def __init__(self, analysis_date: Optional[datetime] = None):
        self.analysis_date = analysis_date or datetime.now()
        
        # RFM segment mappings
        self.segment_definitions = {
            (5, 5, 5): "Champions",
            (5, 5, 4): "Champions",
            (5, 4, 5): "Champions",
            (4, 5, 5): "Loyal Customers",
            (4, 5, 4): "Loyal Customers",
            (4, 4, 5): "Loyal Customers",
            (4, 4, 4): "Loyal Customers",
            (5, 1, 1): "New Customers",
            (5, 1, 2): "New Customers",
            (5, 2, 1): "New Customers",
            (5, 2, 2): "New Customers",
            (3, 3, 3): "Need Attention",
            (3, 2, 3): "Need Attention",
            (2, 3, 3): "Need Attention",
            (2, 2, 2): "At Risk",
            (2, 1, 2): "At Risk",
            (1, 2, 2): "At Risk",
            (1, 1, 1): "Lost",
            (1, 1, 2): "Lost",
            (1, 2, 1): "Lost",
        }
    
    def calculate_rfm(self, orders: List[CustomerOrder]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate RFM scores for all customers.
        """
        customer_data = defaultdict(lambda: {
            "orders": [],
            "total_value": 0,
            "last_order": None
        })
        
        for order in orders:
            customer_data[order.customer_id]["orders"].append(order)
            customer_data[order.customer_id]["total_value"] += order.order_value
            
            if (customer_data[order.customer_id]["last_order"] is None or 
                order.order_date > customer_data[order.customer_id]["last_order"]):
                customer_data[order.customer_id]["last_order"] = order.order_date
        
        # Calculate raw RFM values
        rfm_data = {}
        for customer_id, data in customer_data.items():
            recency = (self.analysis_date - data["last_order"]).days if data["last_order"] else 999
            frequency = len(data["orders"])
            monetary = data["total_value"]
            
            rfm_data[customer_id] = {
                "recency_days": recency,
                "frequency": frequency,
                "monetary": round(monetary, 2)
            }
        
        # Calculate quintile scores (1-5)
        if rfm_data:
            recency_values = sorted([d["recency_days"] for d in rfm_data.values()])
            frequency_values = sorted([d["frequency"] for d in rfm_data.values()])
            monetary_values = sorted([d["monetary"] for d in rfm_data.values()])
            
            for customer_id in rfm_data:
                rfm_data[customer_id]["r_score"] = self._get_quintile_score(
                    rfm_data[customer_id]["recency_days"], recency_values, reverse=True
                )
                rfm_data[customer_id]["f_score"] = self._get_quintile_score(
                    rfm_data[customer_id]["frequency"], frequency_values
                )
                rfm_data[customer_id]["m_score"] = self._get_quintile_score(
                    rfm_data[customer_id]["monetary"], monetary_values
                )
                
                # Composite RFM score
                rfm_data[customer_id]["rfm_score"] = (
                    f"{rfm_data[customer_id]['r_score']}-"
                    f"{rfm_data[customer_id]['f_score']}-"
                    f"{rfm_data[customer_id]['m_score']}"
                )
        
        return rfm_data
    
    def _get_quintile_score(self, value: float, sorted_values: List[float], 
                           reverse: bool = False) -> int:
        """Assign quintile score (1-5)."""
        n = len(sorted_values)
        if n == 0:
            return 3
        
        # Find position
        position = 0
        for i, v in enumerate(sorted_values):
            if value <= v:
                position = i
                break
            position = i
        
        percentile = position / n
        
        if reverse:
            percentile = 1 - percentile
        
        if percentile >= 0.8:
            return 5
        elif percentile >= 0.6:
            return 4
        elif percentile >= 0.4:
            return 3
        elif percentile >= 0.2:
            return 2
        else:
            return 1
    
    def segment_customers(self, orders: List[CustomerOrder]) -> Dict[str, CustomerSegment]:
        """
        Segment customers based on RFM scores.
        """
        rfm_data = self.calculate_rfm(orders)
        
        # Group by segment
        segments: Dict[str, List[str]] = defaultdict(list)
        segment_rfm: Dict[str, List[Dict]] = defaultdict(list)
        
        for customer_id, data in rfm_data.items():
            rfm_tuple = (data["r_score"], data["f_score"], data["m_score"])
            
            # Find matching segment or default
            segment_name = self._get_segment_name(rfm_tuple)
            segments[segment_name].append(customer_id)
            segment_rfm[segment_name].append(data)
        
        # Create segment objects
        total_customers = len(rfm_data)
        result = {}
        
        for segment_name, customer_ids in segments.items():
            rfm_values = segment_rfm[segment_name]
            
            avg_recency = sum(d["recency_days"] for d in rfm_values) / len(rfm_values)
            avg_frequency = sum(d["frequency"] for d in rfm_values) / len(rfm_values)
            avg_monetary = sum(d["monetary"] for d in rfm_values) / len(rfm_values)
            
            result[segment_name] = CustomerSegment(
                segment_name=segment_name,
                customer_ids=customer_ids,
                count=len(customer_ids),
                percentage=round((len(customer_ids) / total_customers) * 100, 1),
                avg_recency_days=round(avg_recency, 1),
                avg_frequency=round(avg_frequency, 2),
                avg_monetary=round(avg_monetary, 2),
                characteristics=self._get_segment_characteristics(segment_name),
                recommended_strategy=self._get_segment_strategy(segment_name),
                ad_targeting=self._get_ad_targeting(segment_name)
            )
        
        return result
    
    def _get_segment_name(self, rfm_tuple: tuple) -> str:
        """Map RFM tuple to segment name."""
        if rfm_tuple in self.segment_definitions:
            return self.segment_definitions[rfm_tuple]
        
        # Fallback logic based on average score
        avg_score = sum(rfm_tuple) / 3
        if avg_score >= 4:
            return "Potential Loyalists"
        elif avg_score >= 3:
            return "Promising"
        elif avg_score >= 2:
            return "Hibernating"
        else:
            return "About to Sleep"
    
    def _get_segment_characteristics(self, segment: str) -> Dict[str, Any]:
        """Get characteristics for a segment."""
        characteristics = {
            "Champions": {
                "behavior": "Bought recently, buy often, spend the most",
                "value_tier": "highest",
                "churn_risk": "very_low"
            },
            "Loyal Customers": {
                "behavior": "Spend good money often, responsive to promotions",
                "value_tier": "high",
                "churn_risk": "low"
            },
            "New Customers": {
                "behavior": "Bought recently, but not frequently",
                "value_tier": "potential",
                "churn_risk": "medium"
            },
            "Need Attention": {
                "behavior": "Above average recency, frequency, monetary",
                "value_tier": "medium",
                "churn_risk": "medium"
            },
            "At Risk": {
                "behavior": "Spent big money and purchased often, but long time ago",
                "value_tier": "high_declining",
                "churn_risk": "high"
            },
            "Lost": {
                "behavior": "Lowest recency, frequency, and monetary",
                "value_tier": "low",
                "churn_risk": "very_high"
            }
        }
        return characteristics.get(segment, {"behavior": "Mixed signals", "value_tier": "medium", "churn_risk": "medium"})
    
    def _get_segment_strategy(self, segment: str) -> str:
        """Get recommended strategy for segment."""
        strategies = {
            "Champions": "Reward them! Early access to new products, VIP treatment, loyalty program",
            "Loyal Customers": "Upsell premium products, request reviews, referral program",
            "New Customers": "Nurture with onboarding, encourage second purchase, bundle offers",
            "Need Attention": "Reactivation campaigns, limited time offers, personalized recommendations",
            "At Risk": "Win-back campaigns, special discounts, survey to understand issues",
            "Potential Loyalists": "Offer membership/loyalty program, recommend products",
            "Promising": "Create brand awareness, free trials",
            "Hibernating": "Reconnect with relevant offers, share reviews",
            "About to Sleep": "Reconnect with personalized reach-out",
            "Lost": "Revive interest with heavy discounts or write-off"
        }
        return strategies.get(segment, "Monitor and analyze behavior")
    
    def _get_ad_targeting(self, segment: str) -> str:
        """Get ad targeting strategy for segment."""
        targeting = {
            "Champions": "Exclude from acquisition ads, retarget for upsells and new products",
            "Loyal Customers": "Cross-sell campaigns, new product announcements",
            "New Customers": "Onboarding sequences, second purchase incentives",
            "Need Attention": "Re-engagement campaigns, reminder ads",
            "At Risk": "Heavy retargeting, discount messaging, urgency ads",
            "Lost": "Aggressive win-back or exclude from campaigns"
        }
        return targeting.get(segment, "Standard retargeting")
    
    def get_segment_summary(self, segments: Dict[str, CustomerSegment]) -> Dict[str, Any]:
        """Generate summary statistics across all segments."""
        total = sum(s.count for s in segments.values())
        total_value = sum(s.avg_monetary * s.count for s in segments.values())
        
        # Sort by value
        sorted_segments = sorted(segments.values(), key=lambda x: x.avg_monetary * x.count, reverse=True)
        
        # Calculate which segments drive revenue
        cumulative_value = 0
        top_revenue_segments = []
        for seg in sorted_segments:
            seg_value = seg.avg_monetary * seg.count
            cumulative_value += seg_value
            top_revenue_segments.append(seg.segment_name)
            if cumulative_value >= total_value * 0.8:
                break
        
        return {
            "total_customers": total,
            "segments_count": len(segments),
            "top_value_segments": top_revenue_segments,
            "at_risk_customers": sum(s.count for s in segments.values() 
                                    if s.segment_name in ["At Risk", "Lost", "Hibernating"]),
            "high_value_customers": sum(s.count for s in segments.values() 
                                       if s.segment_name in ["Champions", "Loyal Customers"]),
            "segment_breakdown": [
                {
                    "name": s.segment_name,
                    "count": s.count,
                    "percentage": s.percentage,
                    "avg_value": s.avg_monetary
                }
                for s in sorted_segments
            ]
        }


def generate_sample_orders(n_customers: int = 100, n_orders: int = 500) -> List[CustomerOrder]:
    """Generate sample order data for demonstration."""
    import random
    
    orders = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(n_orders):
        customer_id = f"CUST_{random.randint(1, n_customers):04d}"
        order_date = start_date + timedelta(days=random.randint(0, 365))
        order_value = random.uniform(20, 200) * (1 + random.random())
        
        orders.append(CustomerOrder(
            customer_id=customer_id,
            order_id=f"ORD_{i:06d}",
            order_date=order_date,
            order_value=round(order_value, 2),
            products=[f"PROD_{random.randint(1, 20):03d}"],
            is_repeat=random.random() < 0.3
        ))
    
    return orders


if __name__ == "__main__":
    # Demo
    segmenter = CustomerSegmenter()
    
    # Generate sample data
    print("Generating sample order data...")
    orders = generate_sample_orders(100, 500)
    
    # Segment customers
    print("Segmenting customers...")
    segments = segmenter.segment_customers(orders)
    
    # Print results
    print(f"\nFound {len(segments)} segments:")
    for name, segment in sorted(segments.items(), key=lambda x: x[1].count, reverse=True):
        print(f"\n{name} ({segment.count} customers, {segment.percentage}%)")
        print(f"  Avg Recency: {segment.avg_recency_days} days")
        print(f"  Avg Frequency: {segment.avg_frequency} orders")
        print(f"  Avg Value: ${segment.avg_monetary}")
        print(f"  Strategy: {segment.recommended_strategy[:60]}...")
    
    # Summary
    summary = segmenter.get_segment_summary(segments)
    print(f"\n=== SUMMARY ===")
    print(f"Total Customers: {summary['total_customers']}")
    print(f"High Value: {summary['high_value_customers']}")
    print(f"At Risk: {summary['at_risk_customers']}")
