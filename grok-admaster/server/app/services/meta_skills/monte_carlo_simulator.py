"""
Monte Carlo Simulator for Optimus Prime Simulation Lab
Performs probabilistic forecasting for campaign performance.
"""

import numpy as np
import random
from typing import Dict, List, Any, Tuple
import math

class MonteCarloSimulator:
    def __init__(self, iterations: int = 10000):
        self.iterations = iterations

    def run_simulation(self, variables: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation based on input variable distributions.
        
        variables format:
        {
            "variable_name": {
                "distribution": "normal" | "lognormal" | "uniform" | "triangular",
                "params": { ... }
            }
        }
        """
        results = {}
        
        # Generate random samples for each variable
        samples = {}
        for name, config in variables.items():
            dist_type = config.get("distribution", "normal")
            params = config.get("params", {})
            
            if dist_type == "normal":
                mu = params.get("mean", 0)
                sigma = params.get("std", 1)
                samples[name] = np.random.normal(mu, sigma, self.iterations)
                
            elif dist_type == "lognormal":
                # Convert mean/std of underlying normal distribution
                mean = params.get("mean", 1)
                std = params.get("std", 0.5)
                # Helper to convert to mu/sigma for lognormal
                phi = math.sqrt(std**2 + mean**2)
                mu = math.log(mean**2 / phi)
                sigma = math.sqrt(math.log(phi**2 / mean**2))
                samples[name] = np.random.lognormal(mu, sigma, self.iterations)
                
            elif dist_type == "uniform":
                low = params.get("min", 0)
                high = params.get("max", 1)
                samples[name] = np.random.uniform(low, high, self.iterations)
                
            elif dist_type == "fixed":
                val = params.get("value", 0)
                samples[name] = np.full(self.iterations, val)

        # Calculate derived metrics (e.g., Sales = Clicks * CPC * ...) - Logic would be dynamic in real app
        # For this generic simulator, we assume we want to calculate revenue and profit
        # calculating typical ecommerce metrics if present
        
        if "clicks" in samples and "cpc" in samples:
            samples["spend"] = samples["clicks"] * samples["cpc"]
            
        if "clicks" in samples and "conversion_rate" in samples and "aov" in samples:
            samples["conversions"] = samples["clicks"] * samples["conversion_rate"]
            samples["revenue"] = samples["conversions"] * samples["aov"]
            
        if "revenue" in samples and "spend" in samples:
            samples["profit"] = samples["revenue"] - samples["spend"]
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                 samples["acos"] = np.where(samples["revenue"] > 0, samples["spend"] / samples["revenue"], 0)
                 samples["roas"] = np.where(samples["spend"] > 0, samples["revenue"] / samples["spend"], 0)

        # Aggregate results
        for metric, values in samples.items():
            results[metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "p5": float(np.percentile(values, 5)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
            
        return {
            "iterations": self.iterations,
            "metrics": results
        }

    def calculate_var(self, profit_distribution: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) at a given confidence level.
        Returns the absolute loss amount that will not be exceeded with confidence_level probability.
        """
        # Convert profit to loss (negative profit)
        losses = -np.array(profit_distribution)
        
        # We look for the percentile of the loss distribution
        percentile = confidence_level * 100
        var = np.percentile(losses, percentile)
        
        # If VaR is negative, it means we expect a profit even in the worst case
        return float(var)
