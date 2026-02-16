import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PriceChangeDetector:
    """
    Implements Change-Point Detection (CPD) for price monitoring.
    Uses a simplified Binary Segmentation algorithm to detect structural breaks.
    """
    
    def __init__(self, min_segment_size: int = 3, threshold: float = 0.05):
        self.min_segment_size = min_segment_size
        self.threshold = threshold  # Normalized cost reduction threshold

    def detect_changes(self, prices: List[float], dates: List[datetime]) -> List[Dict]:
        """
        Detect change points in a price series.
        Returns a list of change events.
        """
        if len(prices) < self.min_segment_size * 2:
            return []

        # Normalize prices for stable detection
        try:
            prices_array = np.array(prices)
            # Simple binary segmentation
            change_points = self._binary_segmentation(prices_array, 0, len(prices_array))
            
            results = []
            if change_points:
                # Filter and format results
                # Sorted unique points
                change_points = sorted(list(set(change_points)))
                
                for cp_idx in change_points:
                    if cp_idx >= len(prices): continue
                    
                    # Calculate magnitude
                    prev_segment = prices[:cp_idx]
                    curr_segment = prices[cp_idx:]
                    
                    if len(prev_segment) == 0 or len(curr_segment) == 0:
                        continue

                    old_price = float(np.mean(prev_segment[-self.min_segment_size:]))
                    new_price = float(np.mean(curr_segment[:self.min_segment_size]))
                    
                    diff_pct = (new_price - old_price) / old_price if old_price != 0 else 0
                    
                    # Only report if change is significant (> 1% usually)
                    if abs(diff_pct) > 0.01:
                        results.append({
                            "change_date": dates[cp_idx],
                            "old_price": round(old_price, 2),
                            "new_price": round(new_price, 2),
                            "change_percent": round(diff_pct * 100, 2),
                            "change_type": "drop" if diff_pct < 0 else "hike",
                            "confidence": self._calculate_confidence(prices_array, cp_idx)
                        })
            
            return results
        except Exception as e:
            logger.error(f"Error in price change detection: {str(e)}")
            return []

    def _cost_function(self, segment: np.ndarray) -> float:
        """Sum of squared errors cost function (standard for mean-shift)."""
        if len(segment) == 0: return 0.0
        return np.sum((segment - np.mean(segment)) ** 2)

    def _binary_segmentation(self, signal: np.ndarray, start: int, end: int) -> List[int]:
        """Recursive binary segmentation."""
        n = end - start
        if n < self.min_segment_size * 2:
            return []

        segment = signal[start:end]
        total_cost = self._cost_function(segment)
        
        best_split_cost = float('inf')
        best_idx = -1

        # Check all possible split points
        # Optimization: Don't check every point, check every stride or smart search
        # For small N (30-90 days), exhaustive search is fine
        for i in range(self.min_segment_size, n - self.min_segment_size):
            left = segment[:i]
            right = segment[i:]
            cost = self._cost_function(left) + self._cost_function(right)
            
            if cost < best_split_cost:
                best_split_cost = cost
                best_idx = i

        # If splitting improves cost significantly
        # Gain computation
        gain = total_cost - best_split_cost
        
        # Simple thresholding logic for "significant gain"
        # In a real library like ruptures, this is more complex (BIC/AIC)
        # Here we use a heuristic relative to variance
        if gain > self.threshold * total_cost:
            global_idx = start + best_idx
            # Recursively search left and right
            left_points = self._binary_segmentation(signal, start, global_idx)
            right_points = self._binary_segmentation(signal, global_idx, end)
            return left_points + [global_idx] + right_points
        
        return []

    def _calculate_confidence(self, signal: np.ndarray, cp_idx: int) -> float:
        """Calculate a confidence score (0-1) based on signal-to-noise ratio around change.

        Returns:
            Confidence score between 0.1 and 0.99
            Default 0.5 if calculation fails
        """
        try:
            window = 5
            start = max(0, cp_idx - window)
            end = min(len(signal), cp_idx + window)

            local_variance = np.var(signal[start:end])
            if local_variance == 0:
                return 1.0  # Perfect step

            # Higher variance = lower confidence
            # Inverse mapping
            confidence = 1.0 / (1.0 + local_variance)
            return min(max(confidence, 0.1), 0.99)
        except (IndexError, ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Confidence calculation failed at index {cp_idx}: {e}")
            return 0.5  # Default confidence when calculation fails

class CannibalizationDetector:
    """
    Detects SEO keyword cannibalization (multiple pages ranking for same intent).
    """
    
    def detect_conflicts(self, google_search_console_data: List[Dict]) -> List[Dict]:
        """
        Input: List of {query, page, clicks, impressions, position}
        Output: List of cannibalization groups
        """
        # Group by keyword
        keyword_map = {}
        for row in google_search_console_data:
            kw = row['query'].lower().strip()
            if kw not in keyword_map:
                keyword_map[kw] = []
            keyword_map[kw].append(row)

        conflicts = []
        
        for kw, entries in keyword_map.items():
            if len(entries) > 1:
                # We have multiple pages ranking for this keyword (strict cannibalization)
                # Filter for "real" conflicts (both getting meaningful traffic)
                
                # Sort by clicks/impressions
                sorted_entries = sorted(entries, key=lambda x: x.get('clicks', 0), reverse=True)
                
                primary_page = sorted_entries[0]
                secondary_pages = sorted_entries[1:]
                
                # Logic: If secondary page has > 10% of primary page's impacts, it's a conflict
                significant_secondary = [
                    p for p in secondary_pages 
                    if p.get('impressions', 0) > (primary_page.get('impressions', 0) * 0.1)
                ]
                
                if significant_secondary:
                    conflicts.append({
                        "keyword": kw,
                        "primary_url": primary_page['page'],
                        "conflicting_urls": [p['page'] for p in significant_secondary],
                        "total_volume": sum(x.get('impressions', 0) for x in entries),
                        "ctr_loss_estimate": self._estimate_loss(entries),
                        "status": "detected"
                    })
        
        return conflicts

    def _estimate_loss(self, entries: List[Dict]) -> float:
        """Estimate traffic lost due to splitting optimization."""
        # Simple heuristic: Split ranking lowers authority.
        # If consolidated, rank typically improves.
        # Loss = (Potential CTR of Top Page) - (Sum of Current CTRs)
        total_clicks = sum(x.get('clicks', 0) for x in entries)
        total_imps = sum(x.get('impressions', 0) for x in entries)
        
        if total_imps == 0: return 0.0
        
        current_ctr = total_clicks / total_imps
        
        # Assume consolidation improves CTR by 20% relative (conservative)
        potential_ctr = current_ctr * 1.2
        
        loss = (potential_ctr - current_ctr) * total_imps
        return round(loss, 2)
