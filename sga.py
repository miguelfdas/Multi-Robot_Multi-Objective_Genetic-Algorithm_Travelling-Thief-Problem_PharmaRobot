"""
PharmaRobot - Standard Genetic Algorithm (SGA) Baseline

A pure, standalone implementation of a Standard Single-Population Genetic Algorithm.
Designed for high performance, modularity, and strict adherence to foundational literature.

Characteristics:
    - Independent architecture (no external GA/Individual dependencies).
    - Single-Objective Scalar Fitness via static linear aggregation.
    - Pure random initialisation (no greedy or K-Means seeding).
    - Strict penalty-based constraint handling (no heuristic repairs).
    - Standard evolutionary operators (Tournament, OX, Uniform Crossover, Swap, Random Resetting).
"""

import time
import numpy as np
from joblib import Parallel, delayed


class SGAIndividual:
    """
    Encodes a single solution for the Standard Genetic Algorithm.
    Constraint violations are strictly penalised mathematically.
    """
    
    def __init__(self, problem, tours=None, item_assignment=None, charge_aggressiveness=None):
        self.problem = problem
        self.tours = tours
        self.item_assignment = item_assignment
        self.charge_aggressiveness = charge_aggressiveness
        
        self.fitness = None  # [Makespan, TWT]
        self.scalar_fitness = np.inf

    @classmethod
    def random_initialisation(cls, problem, rng):
        """
        Pure random initialisation (Goldberg, 1989).
        Generates stochastic routes and assignments without repair mechanisms.
        """
        tours = []
        for k in range(problem.num_robots):
            start = problem.position[k]
            cities = [c for c in range(problem.num_cities) if c != start and c != 0]
            perm = rng.permutation(cities).tolist()
            tours.append([start] + perm + [0])

        # Random assignment (50% chance of being picked, then uniform random robot)
        item_assignment = np.full(problem.num_items, -1, dtype=int)
        mask = rng.random(problem.num_items) < 0.5
        item_assignment[mask] = rng.randint(0, problem.num_robots, size=np.sum(mask))

        charge_aggressiveness = rng.uniform(0.0, 1.0, size=problem.num_robots)

        return cls(problem, tours, item_assignment, charge_aggressiveness)

    def evaluate_fitness(self):
        """
        Calculates objectives and applies heavy penalties for physical constraint violations.
        Static linear aggregation of Makespan and TWT.
        """
        time_matrix = self.problem.time_matrix
        delivery_time = np.full(self.problem.num_items, np.inf)
        robot_makespan = np.zeros(self.problem.num_robots)
        
        virtual_battery = [b for b in self.problem.battery]
        penalty = 0.0
        PENALTY_WEIGHT = 1e6  # Massive penalty for invalid solutions

        for k, tour in enumerate(self.tours):
            assigned_items = np.where(self.item_assignment == k)[0]
            
            # Constraint 1: Knapsack Capacity
            if len(assigned_items) > self.problem.knapsack_capacity:
                penalty += (len(assigned_items) - self.problem.knapsack_capacity) * PENALTY_WEIGHT

            # Map required destinations for this robot
            city_drop = {}
            for i in assigned_items:
                dest = int(self.problem.items[i, 1])
                city_drop.setdefault(dest, []).append(i)
                
                # Constraint 2: Route Synchronisation (Must visit the assigned item's destination)
                if dest != 0 and dest not in tour:
                    penalty += PENALTY_WEIGHT
 
            current_time = 0.0
            has_visited_depot = False
            
            # Simulate the route execution
            for pos, city in enumerate(tour):
                if city == 0:
                    has_visited_depot = True
                
                # Register delivery times if the depot was already visited
                for i in city_drop.get(city, []):
                    if delivery_time[i] == np.inf and has_visited_depot:
                        delivery_time[i] = current_time
 
                # Move to next city
                if pos < len(tour) - 1:
                    next_city = tour[pos + 1]
                    travel_time = time_matrix[city, next_city]
                    current_time += travel_time
                    virtual_battery[k] -= travel_time
            
            # Constraint 3: Battery Capacity
            if virtual_battery[k] < 0:
                penalty += abs(virtual_battery[k]) * PENALTY_WEIGHT

            robot_makespan[k] = current_time
 
        makespan = float(np.max(robot_makespan))
        
        # Calculate Total Weighted Tardiness (TWT)
        trip_start = self.problem.trip_start_time 
        arrival_times = self.problem.items[:, 4]
        delivered_mask = (delivery_time != np.inf)
        
        # Constraint 4: Unassigned or undelivered items
        num_unassigned = np.sum(~delivered_mask)
        penalty += num_unassigned * PENALTY_WEIGHT
        
        twt = 0.0
        if np.any(delivered_mask):
            tct_items_valid = (trip_start + delivery_time[delivered_mask]) - arrival_times[delivered_mask]
            priorities = self.problem.items[delivered_mask, 2]
            deadlines = self.problem.items[delivered_mask, 3]
            twt = float(np.sum((4.0 - priorities) * np.maximum(0.0, tct_items_valid - deadlines)))
                     
        # Store raw objectives
        self.fitness = np.array([makespan, twt], dtype=float)
        
        # Static Linear Aggregation (w1 = 0.5, w2 = 0.5)
        self.scalar_fitness = (0.5 * makespan) + (0.5 * twt) + penalty
        
        return self.scalar_fitness

    def copy(self):
        new_ind = SGAIndividual(
            self.problem, 
            [t.copy() for t in self.tours], 
            self.item_assignment.copy(), 
            self.charge_aggressiveness.copy()
        )
        new_ind.scalar_fitness = self.scalar_fitness
        if self.fitness is not None:
            new_ind.fitness = self.fitness.copy()
        return new_ind


class StandardGeneticAlgorithm:
    """
    Standard Genetic Algorithm (SGA)
    Built from scratch for maximum modularity and performance.
    """
    
    def __init__(self, problem, pop_size=200, generations=100, cx_pb_tour=0.8, cx_pb_pack=0.8, mut_pb_tour=0.2, mut_pb_pack=0.1, tournament_size=2, elitism=2, n_jobs=-1, seed=None):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.cx_pb_tour = cx_pb_tour
        self.cx_pb_pack = cx_pb_pack
        self.mut_pb_tour = mut_pb_tour
        self.mut_pb_pack = mut_pb_pack
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.n_jobs = n_jobs
        
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.population = []
        self.best_individual = None
        self.best_fitness = np.inf
        self.current_gen = 0

    # ==========================================
    # Initialization & Evaluation Lifecycle
    # ==========================================
    
    def initialize_population(self):
        """Creates the initial population using pure random bounds."""
        seeds = self.rng.randint(0, int(1e9), self.pop_size)
        self.population = Parallel(n_jobs=self.n_jobs)(
            delayed(SGAIndividual.random_initialisation)(self.problem, np.random.RandomState(s)) for s in seeds
        )

    def evaluate_population(self):
        """Parallel evaluation of the population."""
        fitness_results = Parallel(n_jobs=self.n_jobs)(
            delayed(ind.evaluate_fitness)() for ind in self.population
        )
        for ind, fit in zip(self.population, fitness_results):
            ind.scalar_fitness = fit
            
    def update_best_individual(self):
        """Tracks the best global solution found so far."""
        for ind in self.population:
            if ind.scalar_fitness < self.best_fitness:
                self.best_fitness = ind.scalar_fitness
                self.best_individual = ind.copy()

    # ==========================================
    # Genetic Operators (Modularly defined)
    # ==========================================

    def tournament_selection(self, k=2):
        """Standard Tournament Selection (Miller & Goldberg, 1995)."""
        selected = []
        for _ in range(k):
            indices = self.rng.choice(len(self.population), size=self.tournament_size, replace=False)
            tournament = [self.population[i] for i in indices]
            winner = min(tournament, key=lambda ind: ind.scalar_fitness)
            selected.append(winner)
        return selected

    def crossover_ox(self, parent1_tour, parent2_tour):
        """Order Crossover (OX) for TSP routes (Davis, 1985)."""
        if len(parent1_tour) <= 3 or len(parent2_tour) <= 3:
            return parent1_tour.copy(), parent2_tour.copy()
        
        size1 = len(parent1_tour)
        p1, p2 = sorted(self.rng.choice(range(1, size1 - 1), size=2))
        if p1 == p2:
            p2 = min(p1 + 1, size1 - 1)
        
        offspring1 = [-1] * size1
        offspring1[0], offspring1[-1] = 0, 0
        offspring1[p1:p2] = parent1_tour[p1:p2]
        
        donor1 = [c for c in parent2_tour if c not in offspring1 and c != 0]
        donor1.extend([c for c in parent1_tour if c not in offspring1 and c not in donor1 and c != 0])
        
        idx = 0
        for i in range(1, size1 - 1):
            if offspring1[i] == -1:
                offspring1[i] = donor1[idx] if idx < len(donor1) else parent1_tour[i]
                idx += 1
                
        size2 = len(parent2_tour)
        p1_2, p2_2 = sorted(self.rng.choice(range(1, size2 - 1), size=2))
        if p1_2 == p2_2:
            p2_2 = min(p1_2 + 1, size2 - 1)
            
        offspring2 = [-1] * size2
        offspring2[0], offspring2[-1] = 0, 0
        offspring2[p1_2:p2_2] = parent2_tour[p1_2:p2_2]
        
        donor2 = [c for c in parent1_tour if c not in offspring2 and c != 0]
        donor2.extend([c for c in parent2_tour if c not in offspring2 and c not in donor2 and c != 0])
        
        idx = 0
        for i in range(1, size2 - 1):
            if offspring2[i] == -1:
                offspring2[i] = donor2[idx] if idx < len(donor2) else parent2_tour[i]
                idx += 1
        
        return offspring1, offspring2

    def crossover_uniform(self, parent1_assign, parent2_assign):
        """Uniform Crossover for integer arrays (Syswerda, 1989)."""
        mask = self.rng.rand(len(parent1_assign)) < 0.5
        offspring1 = np.where(mask, parent1_assign, parent2_assign)
        offspring2 = np.where(mask, parent2_assign, parent1_assign)
        return offspring1, offspring2

    def mutate_swap(self, tour):
        """Standard 2-opt Swap Mutation (Banzhaf, 1990)."""
        if self.rng.rand() < self.mut_pb_tour:
            size = len(tour)
            if size > 3:
                i, j = sorted(self.rng.randint(1, size-1, size=2))
                if i != j:
                    tour[i:j] = tour[i:j][::-1]

    def mutate_random_resetting(self, assignment):
        """Pure Random Resetting Mutation for knapsack assignments."""
        for i in range(len(assignment)):
            if self.rng.rand() < self.mut_pb_pack:
                assignment[i] = self.rng.randint(-1, self.problem.num_robots)

    # ==========================================
    # Main Evolutionary Loop
    # ==========================================

    def evolve_generation(self):
        """Standard generational step ensuring absolute size synchronisation."""
        
        target_len = self.problem.num_items
        for ind in self.population:
            if len(ind.item_assignment) != target_len:
                new_assign = np.full(target_len, -1, dtype=int)
                min_len = min(len(ind.item_assignment), target_len)
                new_assign[:min_len] = ind.item_assignment[:min_len]
                ind.item_assignment = new_assign
        
        self.population.sort(key=lambda x: x.scalar_fitness)
        offspring = [ind.copy() for ind in self.population[:self.elitism]]
        
        while len(offspring) < self.pop_size:
            parents = self.tournament_selection(k=2)
            child1, child2 = parents[0].copy(), parents[1].copy()
            
            # Crossover Phase
            if self.rng.rand() < self.cx_pb_tour:
                for k in range(self.problem.num_robots):
                    child1.tours[k], child2.tours[k] = self.crossover_ox(parents[0].tours[k], parents[1].tours[k])

            if self.rng.rand() < self.cx_pb_pack:
                child1.item_assignment, child2.item_assignment = self.crossover_uniform(child1.item_assignment, child2.item_assignment)
            
            alpha = self.rng.rand()
            c1_charge = alpha * parents[0].charge_aggressiveness + (1 - alpha) * parents[1].charge_aggressiveness
            c2_charge = (1 - alpha) * parents[0].charge_aggressiveness + alpha * parents[1].charge_aggressiveness
            child1.charge_aggressiveness = np.clip(c1_charge, 0.0, 1.0)
            child2.charge_aggressiveness = np.clip(c2_charge, 0.0, 1.0)
            
            # Mutation Phase
            for child in [child1, child2]:
                for k in range(self.problem.num_robots):
                    self.mutate_swap(child.tours[k])
                
                self.mutate_random_resetting(child.item_assignment)
                
                # Standard Gaussian mutation for continuous parameters
                if self.rng.rand() < self.mut_pb_pack:
                    noise = self.rng.normal(0.0, 0.1, size=self.problem.num_robots)
                    child.charge_aggressiveness = np.clip(child.charge_aggressiveness + noise, 0.0, 1.0)
            
            offspring.append(child1)
            if len(offspring) < self.pop_size:
                offspring.append(child2)
        
        self.population = offspring[:self.pop_size]
    
    def evolve(self):
        """
        Executes the evolution process.
        Returns: (elapsed_time, genotypic_div, phenotypic_div, priority_div)
        Diversity metrics return 0.0 as they are unnecessary in standard implementations.
        """
        start_time = time.time()
        
        self.initialize_population()
        self.evaluate_population()
        self.update_best_individual()
        
        for gen in range(1, self.generations + 1):
            self.current_gen = gen
            
            self.evolve_generation()
            self.evaluate_population()
            self.update_best_individual()
            
        elapsed = time.time() - start_time
        
        # Returned as (elapsed, 0.0, 0.0, 0.0) to maintain plug-and-play compatibility with your logger
        return elapsed, 0.0, 0.0, 0.0