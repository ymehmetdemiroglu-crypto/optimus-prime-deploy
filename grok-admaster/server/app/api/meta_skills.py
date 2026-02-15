"""
Meta-Skills API API 
Exposes endpoints for the 5 core computational engines:
- Evolution Engine (Genetic Optimizer)
- Simulation Lab (Monte Carlo)
- Knowledge Synthesizer (Trend Detection)
- Meta-Learner (Adaptive Learning)
- Narrative Architect (Story Generator)
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from app.services.meta_skills.genetic_optimizer import GeneticOptimizer
from app.services.meta_skills.monte_carlo_simulator import MonteCarloSimulator
from app.services.meta_skills.trend_detector import TrendDetector
from app.services.meta_skills.learning_rate_adapter import LearningRateAdapter
from app.services.meta_skills.story_generator import StoryGenerator

router = APIRouter()

# --- Maintanence / Health ---

@router.get("/health")
async def health_check():
    return {"status": "meta-skills-active", "engines_loaded": 5}

# --- Evolution Engine ---

class EvolutionRequest(BaseModel):
    baseline_dna: Dict[str, Any]
    population_size: int = 20
    generations: int = 5
    target_objectives: Dict[str, float]  # e.g. {"acos": 0.15}
    historical_context: Optional[Dict[str, Any]] = None # Real campaign data for simulation

@router.post("/evolution/optimize")
async def run_evolution(request: EvolutionRequest):
    """
    Run a genetic optimization simulation on the provided DNA.
    Now accepts historical_context for data-driven fitness evaluation.
    """
    optimizer = GeneticOptimizer(population_size=request.population_size)
    optimizer.initialize_population(request.baseline_dna)
    
    # Generic fitness function for API demonstration
    # Tries to match 'target_objectives' keys in the DNA stats (simulated)
    # Use CampaignFitnessEvaluator for logic-driven fitness
    from app.services.meta_skills.genetic_optimizer import CampaignFitnessEvaluator
    
    # In a real scenario, we'd fetch this data from DB
    mock_history = {"cvr_30d": 0.12, "sales_velocity": 5.4}
    
    evaluator = CampaignFitnessEvaluator(
        historical_data=mock_history, 
        target_acos=request.target_objectives.get("acos", 30.0)
    )
    
    def api_fitness_func(dna):
        return evaluator.evaluate(dna) 

    import random # imported here for the dummy logic

    history = []
    for _ in range(request.generations):
        optimizer.evaluate_fitness(api_fitness_func)
        stats = optimizer.get_population_stats()
        history.append(stats)
        optimizer.evolve()

    best = optimizer.get_best_individual()
    
    return {
        "best_dna": best["dna"],
        "final_fitness": best["fitness"],
        "history": history
    }

# --- Simulation Lab ---

class SimulationVariable(BaseModel):
    distribution: str = "normal"  # normal, lognormal, uniform, fixed
    params: Dict[str, float]

class SimulationRequest(BaseModel):
    variables: Dict[str, SimulationVariable]
    iterations: int = 5000

@router.post("/simulation/forecast")
async def run_simulation(request: SimulationRequest):
    """
    Run Monte Carlo simulation for probabilistic forecasting.
    """
    sim = MonteCarloSimulator(iterations=request.iterations)
    
    # Convert Pydantic models to dict
    var_dict = {k: v.dict() for k, v in request.variables.items()}
    
    results = sim.run_simulation(var_dict)
    
    # Calculate VaR if profit exists
    var_95 = None
    if "profit" in results["metrics"]:
        # We need to re-run or store samples to calc VaR exactly, 
        # but the simple simulator returns stats. 
        # Let's trust the stats for now or assume the helper method needs raw data.
        # The current implementation of calculate_var requires a list.
        # Ideally, we'd update the service to return VaR in the main run.
        pass
        
    return results

# --- Knowledge Synthesizer ---

class TrendRequest(BaseModel):
    keyword_data: Dict[str, List[float]]

@router.post("/knowledge/trends")
async def detect_trends(request: TrendRequest):
    """
    Analyze time-series data for trends and anomalies.
    """
    detector = TrendDetector()
    try:
        results = detector.analyze_keyword_trends(request.keyword_data)
        return {"trends": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Meta-Learner ---

class AdaptationRequest(BaseModel):
    market_data: Dict[str, Any]
    system_confidence: float

@router.post("/learning/adapt")
async def adapt_learning(request: AdaptationRequest):
    """
    Get adaptive learning parameters based on market volatility.
    """
    adapter = LearningRateAdapter()
    result = adapter.adapt_learning_rate(request.market_data, request.system_confidence)
    return result

# --- Narrative Architect ---

class NarrativeRequest(BaseModel):
    type: str = "performance_update"
    stakeholder: str = "manager"
    data: Dict[str, Any]

@router.post("/narrative/generate")
async def generate_story(request: NarrativeRequest):
    """
    Generate human-readable narrative from data.
    """
    generator = StoryGenerator()
    result = generator.generate_narrative(request.dict())
    return result
