"""
Architecture Designer for Optimus Pryme
Audits existing campaign structures and designs optimal hierarchies.
"""

from typing import Dict, List, Any
import json

class ArchitectureDesigner:
    def __init__(self):
        pass

    def audit_structure(self, campaigns: List[Dict]) -> Dict:
        """
        Audit current account structure for best practices.
        """
        issues = []
        score = 100
        
        # 1. Analyze Naming Conventions
        bad_names = [c['name'] for c in campaigns if not self._is_naming_valid(c['name'])]
        if bad_names:
            score -= 15
            issues.append({
                "type": "Naming Convention",
                "severity": "Medium",
                "description": f"Found {len(bad_names)} campaigns with unclear naming.",
                "examples": bad_names[:3],
                "recommendation": "Adopt standard: [Product] - [Type] - [Targeting] - [Strategy]"
            })
            
        # 2. Check for Missing Campaign Types
        types_present = set(c.get('type', 'SP') for c in campaigns)
        if "SB" not in types_present and "Sponsored Brands" not in types_present:
            score -= 10
            issues.append({
                "type": "Missing Ad Type",
                "severity": "High",
                "description": "No Sponsored Brands campaigns detected.",
                "recommendation": "Launch SB Video and Header Search ads for brand awareness."
            })
            
        if "SD" not in types_present and "Sponsored Display" not in types_present:
            score -= 5
            issues.append({
                "type": "Missing Ad Type",
                "severity": "Medium",
                "description": "No Sponsored Display campaigns detected.",
                "recommendation": "Launch SD retargeting to capture abandoners."
            })
            
        # 3. Check Granularity (Keywords per Ad Group)
        # (Simulated check as we only have campaign level here usually, but assuming we had ad group data)
        
        # 4. Check for Cannibalization
        # (Checking for duplicate targeting types across same product)
        
        return {
            "audit_score": score,
            "grade": self._get_grade(score),
            "campaign_count": len(campaigns),
            "issues": issues,
            "recommended_structure": self._generate_ideal_structure_template()
        }

    def design_structure(self, product_name: str, asin: str) -> Dict:
        """
        Generate an ideal campaign structure for a product.
        """
        structure = {
            "product": product_name,
            "asin": asin,
            "portfolios": ["Brand Defense", "Offensive / Competitor", "Core / Generic", "Discovery"],
            "campaigns": []
        }
        
        # 1. Auto Discovery
        structure["campaigns"].append({
            "name": f"{product_name} - SP - Auto - Discovery",
            "type": "Sponsored Products",
            "targeting": "Automatic",
            "strategy": "Low bid, catch-all for keyword harvesting",
            "bidding": "Dynamic Down Only"
        })
        
        # 2. Manual Exact (Core)
        structure["campaigns"].append({
            "name": f"{product_name} - SP - Manual - Exact - Core",
            "type": "Sponsored Products",
            "targeting": "Manual Exact",
            "strategy": "High bid on main volume keywords",
            "bidding": "Fixed Bids or Dynamic Up/Down"
        })
        
        # 3. Competitor Conquesting
        structure["campaigns"].append({
            "name": f"{product_name} - SP - Product - Competitors",
            "type": "Sponsored Products",
            "targeting": "Product Targeting (ASINs)",
            "strategy": "Target weaker competitors (higher price/lower rating)",
            "bidding": "Dynamic Down Only"
        })
        
        # 4. Brand Defense
        structure["campaigns"].append({
            "name": f"{product_name} - SB - Video - Brand",
            "type": "Sponsored Brands",
            "targeting": "Keyword (Brand Name)",
            "strategy": "Defend real estate",
            "format": "Video"
        })
        
        return structure

    def _is_naming_valid(self, name: str) -> bool:
        # Simple heuristic: look for delimiters
        return "-" in name or "|" in name or "_" in name

    def _get_grade(self, score: int) -> str:
        if score >= 90: return "A"
        if score >= 80: return "B"
        if score >= 70: return "C"
        if score >= 60: return "D"
        return "F"

    def _generate_ideal_structure_template(self) -> List[str]:
        return [
            "1. SP - Auto (Discovery)",
            "2. SP - Manual Exact (Ranking/Core)",
            "3. SP - Manual Broad (Reach)",
            "4. SP - Product Targeting (Competitors)",
            "5. SP - Product Targeting (Own Brand/Cross-sell)",
            "6. SB - Video (Top Keywords)",
            "7. SD - Retargeting (Views)"
        ]

if __name__ == "__main__":
    designer = ArchitectureDesigner()
    
    # Test Audit
    sample_campaigns = [
        {"name": "Campaign 1", "type": "SP"},
        {"name": "Audio_Headphones_Exact", "type": "SP"},
        {"name": "test_camp", "type": "SP"}
    ]
    audit = designer.audit_structure(sample_campaigns)
    print("AUDIT RESULTS:")
    print(json.dumps(audit, indent=2))
    
    # Test Design
    print("\nDESIGN RESULTS:")
    design = designer.design_structure("Gaming Mouse", "B0MOUSE123")
    print(json.dumps(design, indent=2))
