"""
Genetic Optimizer for Optimus Pryme Evolution Engine
Implements genetic algorithms for strategy optimization.
"""

import random
import json
import copy
import uuid
from datetime import datetime
from typing import List, Dict, Any, Callable

class GeneticOptimizer:
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism_count: int = 2
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.generation = 0
        self.population = []
        self.history = []

    def initialize_population(self, baseline_dna: Dict[str, Any], variation_range: float = 0.2):
        """
        Create initial population based on a baseline strategy with random variations.
        """
        self.population = []
        
        # Keep the baseline as is (elitism for the start)
        baseline_individual = {
            "id": str(uuid.uuid4()),
            "dna": copy.deepcopy(baseline_dna),
            "fitness": 0.0,
            "generation": 0,
            "parents": [],
            "source": "baseline"
        }
        self.population.append(baseline_individual)
        
        # generate rest of population
        for _ in range(self.population_size - 1):
            new_dna = self._mutate_dna(baseline_dna, variation_range)
            self.population.append({
                "id": str(uuid.uuid4()),
                "dna": new_dna,
                "fitness": 0.0,
                "generation": 0,
                "parents": ["baseline"],
                "source": "mutation"
            })
            
    def _mutate_dna(self, dna: Dict[str, Any], rate: float) -> Dict[str, Any]:
        """
        Apply random mutations to DNA parameters.
        Handles numeric values (int, float) and booleans.
        """
        mutated = copy.deepcopy(dna)
        
        for key, value in mutated.items():
            if random.random() < rate:
                if isinstance(value, bool):
                    mutated[key] = not value
                elif isinstance(value, float):
                    # Mutate by +/- 20% by default or specified rate
                    change = random.uniform(1.0 - rate, 1.0 + rate)
                    mutated[key] = value * change
                elif isinstance(value, int):
                    change = random.uniform(1.0 - rate, 1.0 + rate)
                    mutated[key] = int(value * change)
                    
        return mutated

    def evaluate_fitness(self, fitness_function: Callable[[Dict[str, Any]], float]):
        """
        Evaluate fitness for the entire population using a provided function.
        """
        for individual in self.population:
            individual["fitness"] = fitness_function(individual["dna"])
            
        # Sort by fitness descending
        self.population.sort(key=lambda x: x["fitness"], reverse=True)

    def select_parent(self) -> Dict[str, Any]:
        """
        Tournament selection.
        """
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x["fitness"])

    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uniform crossover: randomly merge genes from both parents.
        """
        child_dna = {}
        for key in parent1["dna"].keys():
            if random.random() < 0.5:
                child_dna[key] = parent1["dna"][key]
            else:
                child_dna[key] = parent2["dna"][key]
        return child_dna

    def evolve(self):
        """
        Run one generation of evolution.
        """
        next_gen = []
        
        # 1. Elitism: Keep best performers
        next_gen.extend(copy.deepcopy(self.population[:self.elitism_count]))
        
        # 2. Generate offspring
        while len(next_gen) < self.population_size:
            if random.random() < self.crossover_rate:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child_dna = self.crossover(parent1, parent2)
                source = "crossover"
                parent_ids = [parent1["id"], parent2["id"]]
            else:
                parent = self.select_parent()
                child_dna = copy.deepcopy(parent["dna"])
                source = "selection" # Cloned
                parent_ids = [parent["id"]]
            
            # 3. Mutation
            child_dna = self._mutate_dna(child_dna, self.mutation_rate)
            if source == "selection":
                source = "mutation" # If we just cloned and mutated
                
            next_gen.append({
                "id": str(uuid.uuid4()),
                "dna": child_dna,
                "fitness": 0.0,
                "generation": self.generation + 1,
                "parents": parent_ids,
                "source": source
            })
            
        self.population = next_gen
        self.generation += 1
        return self.get_best_individual()

    def get_best_individual(self) -> Dict[str, Any]:
        """Return the highest fitness individual."""
        return self.population[0] if self.population else None

    def get_population_stats(self) -> Dict[str, float]:
        """Return fitness statistics."""
        if not self.population:
            return {}
        
        fitnesses = [ind["fitness"] for ind in self.population]
        return {
            "generation": self.generation,
            "max_fitness": max(fitnesses),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "min_fitness": min(fitnesses)
        }

# Example usage
if __name__ == "__main__":
    # Define a dummy fitness function
    # Goal: Optimize parameters to match a target sum
    target_params = {"bid": 1.50, "budget": 100, "active": True}
    
    def dummy_fitness(dna):
        score = 0
        score -= abs(dna["bid"] - target_params["bid"]) * 10
        score -= abs(dna["budget"] - target_params["budget"]) * 0.1
        if dna["active"] == target_params["active"]:
            score += 50
        return max(0, 100 + score) # Normalized 0-100ish

    optimizer = GeneticOptimizer(population_size=10)
    
    baseline = {"bid": 1.0, "budget": 50, "active": False}
    print(f"Baseline DNA: {baseline}")
    
    optimizer.initialize_population(baseline)
    
    print("\nStarting Evolution...")
    for i in range(5):
        optimizer.evaluate_fitness(dummy_fitness)
        stats = optimizer.get_population_stats()
        best = optimizer.get_best_individual()
        print(f"Gen {i}: Max Fitness {stats['max_fitness']:.2f} | Best DNA: {best['dna']}")
        optimizer.evolve()
