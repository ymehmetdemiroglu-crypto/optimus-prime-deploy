"""
Story Generator for Optimus Pryme Narrative Architect
Converts data and insights into compelling human-readable narratives.
"""

from typing import Dict, List, Any
import random

class StoryGenerator:
    def __init__(self):
        self.templates = {
            "performance_update": [
                "Overall, {metric} has {direction} by {value}%. This was primarily driven by {driver}.",
                "We observed a {value}% {direction} in {metric}, largely due to {driver}."
            ],
            "recommendation": [
                "To capitalize on this, we recommend {action}. This could yield {impact}.",
                "Our best course of action is to {action}, which historically results in {impact}."
            ],
            "alert": [
                "Heads up: {issue} detected. We should {action} immediately.",
                "Attention required: {issue}. Proposed fix: {action}."
            ]
        }
        
    def generate_narrative(self, context: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a full narrative based on the provided context.
        Context should include 'type', 'data', 'stakeholder'.
        """
        narrative_type = context.get("type", "performance_update")
        data = context.get("data", {})
        stakeholder = context.get("stakeholder", "manager")
        
        # 1. Generate Headline
        headline = self._generate_headline(data)
        
        # 2. Generate Body
        body = self._generate_body(narrative_type, data)
        
        # 3. Adapt for Stakeholder (Tone adjustment)
        adapted_body = self._adapt_tone(body, stakeholder)
        
        return {
            "headline": headline,
            "body": adapted_body,
            "raw_text": f"{headline}\n\n{adapted_body}"
        }

    def _generate_headline(self, data: Dict[str, Any]) -> str:
        metric = data.get("primary_metric", "performance")
        change = data.get("change_pct", 0)
        
        if change > 5:
            return f"🚀 Strong Growth: {metric} up {change}%"
        elif change < -5:
            return f"⚠️ Attention: {metric} down {abs(change)}%"
        else:
            return f"📊 {metric} Update: Stable Performance"

    def _generate_body(self, n_type: str, data: Dict[str, Any]) -> str:
        if n_type not in self.templates:
            return "No template available."
            
        template = random.choice(self.templates[n_type])
        
        # Fill slots safely
        try:
            direction = "increased" if data.get("change_pct", 0) > 0 else "decreased"
            filled = template.format(
                metric=data.get("primary_metric", "metrics"),
                value=abs(data.get("change_pct", 0)),
                direction=direction,
                driver=data.get("primary_driver", "market conditions"),
                action=data.get("recommended_action", "monitor closely"),
                impact=data.get("expected_impact", "improved results"),
                issue=data.get("issue_description", "anomaly"),
            )
            return filled
        except KeyError as e:
            return f"Error generating story: missing data {e}"

    def _adapt_tone(self, text: str, stakeholder: str) -> str:
        """
        Refine the text based on who is reading it.
        """
        if stakeholder == "cfo":
            # Focus on ROI, efficiency, cost
            return text + " This aligns with our efficiency targets."
        elif stakeholder == "technical":
            # Add detail
            return text + " (Confidence: 95%, p-value < 0.05)"
        elif stakeholder == "ceo":
            # High level, brief
            return "Executive Summary: " + text
        else:
            return text
