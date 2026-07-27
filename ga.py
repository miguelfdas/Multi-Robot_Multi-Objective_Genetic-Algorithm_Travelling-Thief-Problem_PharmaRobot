"""
PharmaRobot - Multi-Objective Genetic Algorithm

A single GA with Adaptive Weighted Tchebycheff Aggregation from Non-dominated Sorting Genetic Algorithm II (NSGA-II)

GA Lifecycle

Population initialization:

Evolution:

Tchebycheff Weight Adaptation
Fitness Evaluation
Diversity Evaluation
Mutation Rate Adaptation

Survivor Selection
    - Elitism 
    - Elit size = 2

Mate Selection:
- Tournament Selection [Miller & Goldberg, 1995]
- Tournament size = 2

Crossover:

    - Order Crossover (OX) for tours [Davis, 1985]
        - Preserves relative order of cities
        - Prevents invalid tours (duplicate cities)
        - Good for permutation problems

    - Uniform Crossover for medication assignment [Syswerda, 1989]
        - Each item independently inherits from parent
        - 50% probability from each parent
        - Good for binary/integer encodings

Mutation:

    - 2-opt Swap Mutation for tours [Banzhaf, 1990]
        - With probability mut_pb_tour:
            - Select two random positions
            - Reverses the tour segment [i:j]
            - Maintains the validity of the tour (no duplicates)
        - Exchange two random cities
        - Maintains tour validity
        - 2-opt local search for tours [Croes, 1958]

    - Priority-guided bit-flip Mutation for medication assignment [Back, 1996]
        - Priority-guided reassign items to different robots
        - Instead of flipping every item with the same probability
        - Each item gets an individual flip probability scaled by its priority
        - Currently unassigned (assignment == -1): flip probability is boosted for high-priority items
        - Currently assigned: flip probability is boosted for low-priority items

Repair [Michalewicz & Schoenauer, 1996]
- Greedy priority-based repair [Michalewicz & Schoenauer, 1996]
- Removes low-priority items when capacity exceeded
"""

import numpy as np
import time
from joblib import Parallel, delayed
from individual import Individual

class GeneticAlgorithm:
    """
    Genetic Algorithm (GA)
    Dynamic weight adjustment (Multi-Objective)
    """
    
    def __init__(self, problem, pop_size=200, generations=100, cx_pb_tour=0.8, cx_pb_pack=0.8, mut_pb_tour=0.2, mut_pb_pack=0.1, tournament_size=2, elitism=2, n_jobs=-1, seed=None):
        """
        Initialize Genetic Algorithm for PharmaRobot Hospital Delivery Problem as Multi-Robot Multi-Objective Travelling Thief Problem (TTP)
        
        Args:
            problem: TTProblem/HospitalProblem instance
            cx_pb_tour: Tour crossover probability
            cx_pb_pack: Packing crossover probability
            mut_pb_tour: Tour mutation probability
            mut_pb_pack: Packing mutation probability
            n_jobs: Number of parallel workers (-1 = all CPUs)
        """

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
        
        self.num_objectives = 2
        
        self.population = []
        self.current_gen = 0
        
        # Bounds for adaptive mutation (never go below base / 5 or above base * 5)
        self.mut_tour_base = mut_pb_tour
        self.mut_pack_base = max(mut_pb_pack, 3.0 / max(problem.num_items, 1))
        self.mut_tour_min  = mut_pb_tour / 5.0
        self.mut_tour_max  = min(mut_pb_tour * 5.0, 1.0)
        self.mut_pack_min  = self.mut_pack_base / 5.0
        self.mut_pack_max  = min(self.mut_pack_base * 10.0, 1.0)
        
        # Tchebycheff weights equal initial importance across 4 objectives
        self.objective_weights = np.ones(2) / 2.0 # [1/2, 1/2] [Makespan, TWT]
        self.pareto_front = None

        self.rng = np.random.RandomState(seed)
        
        self.best_individual = None # best solution found
        self.best_fitness = np.inf # fitness of best solution found
        self.best_found_at_gen = 0 # generation when best solution was found
    
    def normalize_objectives(self, objectives):
        # (f_i - 0) / (nadir_i - 0)  == f_i / nadir_i
        normalized = objectives / np.maximum(self.problem.fitness_bounds.copy(), 1e-10)
        return normalized
        
    def calculate_scalar_fitness(self, objectives):
        """
        Weighted Tchebycheff Aggregation:
            - Multi-objective optimization via adaptive Weighted Tchebycheff Aggregation from NSGA-II
            - Convert Multi-Objective Fitness to Scalar Fitness (Zhang & Li, 2007 MOEA/D)
            - scalar_fitness = max {w_i x normalized(f_i)}
            - w_i: objective weight
            - normalized(f_i) = f_i / nadir_i
            - nadir_i: Worst-case objective fitness, derived from problem structure, population-independent.
            - worst_makespan: all items of the robot are delivered by the worst possible path.
            - worst_tct: all items penalized at 2x worst path.
            - worst_twt: all items at max weight (P1=3), fully late (delivered at worst_tct and deadline=0).
        """
        
        # Normalize objectives first
        norm_obj = self.normalize_objectives(objectives)
        # Weighted Tchebycheff
        weighted_distances = self.objective_weights * norm_obj
        return np.max(weighted_distances)
    
    def adapt_weights(self, generation):        
        """
        Tchebycheff Weight Adaptation (Zhang & Li, 2007 MOEA/D).
        
        Adapt objectives weights during evolution to better exploration and exploitation
                
        Early - f0=0.75, f1=0.25
        Late - f0=0.25, f1=0.75
        
        Early generations: Emphasize Makespan (build good routes)
        Later generations: Emphasize TWT (minimise tardiness once routes are good)
        Weights sum to 1.0 throughout.
        """
        
        # Progress ratio: 0 at start, 1 at end
        progress = generation / self.generations
        
        # Adaptive weighting schedule:
        # w_Makespan = 0.75 * (1 - progress) + 0.25 * progress 
        # w_TWT = 0.25 * (1 - progress) + 0.75 * progress
        
        w_Makespan = 0.5
        w_TWT = 0.5

        self.objective_weights = np.array([w_Makespan, w_TWT])

    def initialize_population(self):
        """        
        Initialize population with Hybrid Strategy (Faulkner et al., 2015):

        50% random: 
            - Random with Repair
            - Generate K stochastic tours one per robot
            - Stochastic medication assignment to robots - 50% chance to assign each medication (promotes diversity)

        50% greedy: 
            - Greedy with Repair
            - Build tour via Nearest-Neighbors Travelling Salesman Problem heuristic, end at pharmacy (0)
            - Priority Packing with Repair:
                - Assign items greedily by priority and distribute items across robots to balance load
        """
        
        n_random = self.pop_size // 2
        n_greedy = self.pop_size - n_random
        
        # Generate unique seeds for each individual
        seeds_random = self.rng.randint(0, int(1e9), n_random)
        seeds_greedy = self.rng.randint(0, int(1e9), n_greedy)
        
        # Random individuals
        random_inds = Parallel(n_jobs=self.n_jobs)(delayed(Individual.random)(self.problem, seed=int(s)) for s in seeds_random)
        
        # Greedy individuals with variation
        greedy_inds = Parallel(n_jobs=self.n_jobs)(delayed(Individual.greedy)(self.problem, seed=int(s)) for s in seeds_greedy)
        
        self.population = random_inds + greedy_inds
    
    def evaluate_population(self):
        """
        Evaluate all individuals in parallel
        - Each individual evaluates multi-objective fitness
        - Update reference points
        - Scalar fitness computed via Tchebycheff aggregation
        """
        
        # Evaluation of multi-objective fitness            
        fitness_results = Parallel(n_jobs=self.n_jobs)(delayed(ind.evaluate_fitness)() for ind in self.population)
        for ind, fit in zip(self.population, fitness_results):
            ind.fitness = fit
                    
        for ind in self.population:
            ind.scalar_fitness = self.calculate_scalar_fitness(ind.fitness)
    
    def tournament_selection(self, k=1):
        # Tournament selection for parent selection (Miller & Goldberg, 1995):
        
        selected = []
        for _ in range(k):
            # Random tournament
            indices = self.rng.choice(len(self.population), size=self.tournament_size, replace=False)
            tournament = [self.population[i] for i in indices]
            winner = min(tournament, key=lambda ind: ind.scalar_fitness)
            selected.append(winner)
        
        return selected
    
    def crossover_ox(self, parent1_tour, parent2_tour):
        """    
        Order Crossover (OX) for delivery routes (Davis (1985)
            
        - Select random substring from parent_1
        - Copy substring to offspring_1
        - Fill remaining positions with parent_2 order
        - Symmetric for offspring_2
        
        - Cut points strictly exclude both index 0 and the final index (start and return to pharmacy)
        - Preserves relative order of cities
        - Prevents duplicate cities (valid tour)
        - Adjacent cities matter
        """
        
        if len(parent1_tour) <= 3 or len(parent2_tour) <= 3:
            return parent1_tour.copy(), parent2_tour.copy()
        
        size1 = len(parent1_tour)
        
        # Select two crossover points (exclude position 0 (origin) and last position (depot again) and ensure they are different      
        point1_1, point2_1 = sorted(self.rng.choice(range(1, size1 - 1), size=2))
        if point1_1 == point2_1:
            point2_1 = min(point1_1 + 1, size1 - 1)
        
        # Initialize offspring
        offspring1 = [-1] * size1
        offspring1[0] = 0  # Depot always first
        offspring1[-1] = 0 # Depot always last
        
        offspring1[point1_1:point2_1] = parent1_tour[point1_1:point2_1]
        donor_order1 = [city for city in parent2_tour if city not in offspring1 and city != 0]
        fallback1 = [city for city in parent1_tour if city not in offspring1 and city not in donor_order1 and city != 0]
        donor_order1.extend(fallback1)
        
        idx1 = 0
        for i in range(1, size1 - 1):
            if offspring1[i] == -1:
                if idx1 < len(donor_order1):
                    offspring1[i] = donor_order1[idx1]
                    idx1 += 1
                else:
                    offspring1[i] = parent1_tour[i]
                
        size2 = len(parent2_tour)
        
        point1_2, point2_2 = sorted(self.rng.choice(range(1, size2 - 1), size=2))
        if point1_2 == point2_2:
            point2_2 = min(point1_2 + 1, size2 - 1)
            
        offspring2 = [-1] * size2
        offspring2[0] = 0
        offspring2[-1] = 0
        
        offspring2[point1_2:point2_2] = parent2_tour[point1_2:point2_2]
        donor_order2 = [city for city in parent1_tour if city not in offspring2 and city != 0]
        fallback2 = [city for city in parent2_tour if city not in offspring2 and city not in donor_order2 and city != 0]
        donor_order2.extend(fallback2)
        
        idx2 = 0
        for i in range(1, size2 - 1):
            if offspring2[i] == -1:
                if idx2 < len(donor_order2):
                    offspring2[i] = donor_order2[idx2]
                    idx2 += 1
                else:
                    offspring2[i] = parent2_tour[i]
        
        return offspring1, offspring2
    
    def crossover_uniform(self, parent1_assign, parent2_assign):
        """
        Uniform Crossover for medication assignment (Syswerda, 1989) 
        Each medication independently inherits from one parent (50% each).
        
        For each gene position:
        - Flip fair coin
        - Heads: inherit from parent1
        - Face: inherit from parent2
        
        Parent1:    [0, 1, -1, 2, 0, 1]
        Parent2:    [1, 0, 2, -1, 1, 0]
        Coin:       [H, T, H, T, H, T]
        Offspring1: [0, 0, -1, -1, 0, 0]
        Offspring2: [1, 1, 2, 2, 1, 1]
        """
        size = len(parent1_assign)
        
        # Random mask: True = take from parent1
        mask = self.rng.rand(size) < 0.5
        
        # Apply mask
        offspring1 = np.where(mask, parent1_assign, parent2_assign)
        offspring2 = np.where(mask, parent2_assign, parent1_assign)
        
        return offspring1, offspring2

    def mutate_swap(self, tour):
        """
        2-opt Swap Mutation for tours [Banzhaf, 1990]
        
        With probability mut_pb_tour:
        - Select two random positions (excluding origin)
        - Reverses the tour segment [i:j]
        - Maintains the validity of the tour (no duplicates)
        
        
        - Exchange two random cities
        - Maintains tour validity
        - 2-opt local search for tours [Croes, 1958]
        """
        
        if self.rng.rand() < self.mut_pb_tour:
            size = len(tour)
            if size > 3:
                # Select two cut points (excluding position 0 and last position)
                i, j = sorted(self.rng.randint(1, size-1, size=2))
                if i != j:
                    # Reverse the segment between i and j
                    tour[i:j] = tour[i:j][::-1]
    
    def mutate_bitflip(self, tour_city_sets, assignment):
        """
        Priority-Guided Bit-Flip Mutation for item assignment (Back, 1996)

        Priority-guided reassign items to different robots
        Instead of flipping every item with the same probability
        Each item gets an individual flip probability scaled by its priority

        - Currently unassigned (assignment == -1):
            flip probability is boosted for high-priority items
        - Currently assigned:
            flip probability is boosted for low-priority items

        After mutation, Repair        
        """

        for i in range(len(assignment)):        
            item_city = int(self.problem.items[i, 1])
            valid_robots = [r for r, city_set in enumerate(tour_city_sets) if item_city in city_set]
            
            priority = self.problem.items[i, 2]

            if assignment[i] == -1:
                p_flip = self.mut_pb_pack * (4 - priority) 
            else:
                p_flip = self.mut_pb_pack * priority
                
            if self.rng.rand() < p_flip:
                if assignment[i] == -1:
                    # Item currently unassigned -> try to assign to a valid robot
                    if valid_robots:
                        assignment[i] = self.rng.choice(valid_robots)
                else:
                    # Item currently assigned -> unassign or reassign
                    choices = valid_robots + [-1] if valid_robots else [-1]
                    assignment[i] = self.rng.choice(choices)
    
    def calculate_diversity(self):
        """
        Calculate population diversity.

        Genotypic diversity:
            - 0.5 × Hamming(packing) + 0.5 × Jaccard(tour edges)
            - Measures structural differences in chromosomes
            - Combines:
                - Pairwise Hamming distance on item_assignment vectors.
                - Pairwise Jaccard edge-disagreement on tours.
            Both components live in [0, 1]

        Phenotypic diversity:
            - 0.5 × Shannon entropy(tour edges) + 0.5 × entropy(meds)
            - Behavioural diversity in the solution space
            - Combines:
                - Binary Shannon entropy of tour-edge usage frequencies.
                - Binary entropy of item-selection frequencies.
            Both components live in [0, 1]

        Priority diversity:
            Measures whether individuals differ in their tendency to select prioritys.
            Mean per-bin standard deviation of priority-fraction distributions. 
        """

        if len(self.population) < 2:
            return 0.0, 0.0, 0.0

        n_inds = len(self.population)

        # Item assignment matrix (n_inds × n_items)
        # Auto syncronization
        assignments = np.full((n_inds, self.problem.num_items), -1, dtype=int)
        for i, ind in enumerate(self.population):
            min_len = min(len(ind.item_assignment), self.problem.num_items)
            assignments[i, :min_len] = ind.item_assignment[:min_len]

        # Per-individual edge sets  (list of sets, one per individual)
        # Edges are undirected: stored as (min(u,v), max(u,v))
        individual_edge_sets = []
        for ind in self.population:
            edges = set()
            for tour in ind.tours:
                tour_len = len(tour)
                for pos in range(tour_len):
                    u = tour[pos]
                    v = tour[(pos + 1) % tour_len]
                    edges.add((min(u, v), max(u, v)))
            individual_edge_sets.append(edges)

        # Genotypic diversity
        n_samples = len(self.population) // 2
        sampled_pairs = [self.rng.choice(n_inds, size=2, replace=False) for _ in range(n_samples)]

        pair_distances = []
        for idx_i, idx_j in sampled_pairs:

            # Packing component: normalised Hamming on item_assignment
            hamming_pack = float(np.mean(assignments[idx_i] != assignments[idx_j]))

            # Routing component: Jaccard distance on tour-edge sets
            #   = 1 - |intersection| / |union|
            #   = 0 when tours are identical, 1 when completely different
            edges_i = individual_edge_sets[idx_i]
            edges_j = individual_edge_sets[idx_j]
            union_size = len(edges_i | edges_j)
            inter_size = len(edges_i & edges_j)
            hamming_tour = 1.0 - inter_size / union_size if union_size > 0 else 0.0

            pair_distances.append(0.5 * hamming_pack + 0.5 * hamming_tour)

        genotypic = float(np.mean(pair_distances))

        # Phenotypic diversity

        # Tour edge entropy
        edge_counts: dict = {}
        for edge_set in individual_edge_sets:
            for edge in edge_set:
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        if edge_counts:
            fracs_tour = np.array(list(edge_counts.values()), dtype=float) / n_inds
            mask_tour = (fracs_tour > 1e-10) & (fracs_tour < 1.0 - 1e-10)
            if np.any(mask_tour):
                f = fracs_tour[mask_tour]
                entropy_tour = -f * np.log2(f) - (1.0 - f) * np.log2(1.0 - f)
                tour_diversity = float(np.mean(entropy_tour))
            else:
                tour_diversity = 0.0 # all edges unanimous - population has converged
        else:
            tour_diversity = 0.0

        # Item selection entropy
        frac_picked = np.mean(assignments >= 0, axis=0) # shape (n_items,)
        mask_item = (frac_picked > 1e-10) & (frac_picked < 1.0 - 1e-10)
        item_entropy = np.zeros(self.problem.num_items)
        item_entropy[mask_item] = (
            - frac_picked[mask_item] * np.log2(frac_picked[mask_item])
            - (1.0 - frac_picked[mask_item]) * np.log2(1.0 - frac_picked[mask_item])
        )
        item_diversity = float(np.mean(item_entropy))

        phenotypic = 0.5 * tour_diversity + 0.5 * item_diversity
        
        priority_distributions = []
        for i in range(n_inds):
            selected = assignments[i] >= 0
            if np.any(selected):
                priorities = self.problem.items[selected, 2]
                dist = np.bincount(priorities.astype(int), minlength=4)[1:].astype(float) 
                dist /= dist.sum() + 1e-10
                priority_distributions.append(dist)

        if len(priority_distributions) >= 2:
            priority_distributions = np.array(priority_distributions)
            raw = float(np.mean(np.std(priority_distributions, axis=0, ddof=1)))
            priority_div = max(0.0, raw)
        else:
            priority_div = 0.0

        return genotypic, phenotypic, priority_div
    
    def update_best_individual(self):
        # Track the best individual found across all generations based on scalar fitness
        
        # Recompute current best_individual scalar with current reference/nadir
        if self.best_individual is not None:
            self.best_individual.scalar_fitness = self.calculate_scalar_fitness(self.best_individual.fitness)
            self.best_fitness = self.best_individual.scalar_fitness

        for ind in self.population:
            if ind.scalar_fitness < self.best_fitness:
                self.best_fitness = ind.scalar_fitness
                self.best_individual = ind.copy()
                self.best_found_at_gen = self.current_gen
        
    def adapt_mutation_rate(self, genotypic_diversity):
        """
        Adaptive Mutation Rate based on Population Diversity (Eiben et al., 1999).

        The intuition is:
          - When diversity is *low*  the population has converged - increase mutation to escape local optima and re-introduce variation.
          - When diversity is *high* exploration is already adequate - reduce mutation to avoid disrupting good solutions.

        A normalised diversity signal d ∈ [0, 1] is mapped to a mutation scale factor via a monotonically *decreasing* linear schedule:
            - scale(d) = scale_max - d x (scale_max - scale_min) where scale_max = 3.0 and scale_min = 0.5.
        
        The raw rates are clamped to [_mut_*_min, _mut_*_max] so they never degenerate to zero or become excessively disruptive.

        Eiben, A. E., Hinterding, R., & Michalewicz, Z. (1999).
        "Parameter control in evolutionary algorithms."
        IEEE Transactions on Evolutionary Computation, 3(2), 124–141.
        """
        # Target diversity for "neutral" (no change) mutation rate
        scale_high = 5.0 # multiplier when diversity - 0
        scale_low = 0.5 # multiplier when diversity - 1

        d_norm = float(np.clip(genotypic_diversity, 0.0, 1.0))

        # Linear interpolation: high scale when low diversity, low scale when high
        # At d=0: scale = scale_high
        # At d=target: scale ≈ 1  (neutral)
        # At d=1: scale = scale_low
        scale = scale_high - d_norm * (scale_high - scale_low)

        self.mut_pb_tour = float(np.clip(self.mut_tour_base * scale, self.mut_tour_min, self.mut_tour_max))
        self.mut_pb_pack = float(np.clip(self.mut_pack_base * scale, self.mut_pack_min, self.mut_pack_max))
    
    def compute_pareto_front(self):
        """
        Extract non-dominated solutions (approximate Pareto front) from the current population
        A solution is non-dominated if no other solution is better or equal in all objectives and strictly better in at least one
        """
        
        objectives = np.array([ind.fitness for ind in self.population])
        
        n = len(objectives)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if j is <= in all and < in at least one
                if (np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i])):
                    is_dominated[i] = True
                    break
        
        self.pareto_front = objectives[~is_dominated]
    
    def evolve_generation(self):
        """
        Evolves the population by one generation.
        """
        # ABSOLUTE SYNCHRONISATION GUARANTEE:
        # Forces all parents and offspring to possess exactly the same dimension,
        # ensuring the genome directly maps to the currently available items.
        target_len = self.problem.num_items
        for ind in self.population:
            if len(ind.item_assignment) != target_len:
                new_assign = np.full(target_len, -1, dtype=int)
                min_len = min(len(ind.item_assignment), target_len)
                new_assign[:min_len] = ind.item_assignment[:min_len]
                ind.item_assignment = new_assign
                ind.repair()
        
        geno_div, pheno_div, priority_div = self.calculate_diversity()
        self.adapt_mutation_rate(geno_div)
        
        self.population.sort(key=lambda x: x.scalar_fitness)
        offspring = [ind.copy() for ind in self.population[:self.elitism]]
        
        # Force elites to synchronise their routes with the current physical environment
        for ind in offspring:
            ind.repair()
        
        while len(offspring) < self.pop_size:
            parents = self.tournament_selection(k=2)
            parent1, parent2 = parents[0], parents[1]
                        
            child1 = parent1.copy()
            child2 = parent2.copy()
            
            if self.rng.rand() < self.cx_pb_tour:
                for k in range(self.problem.num_robots):
                    child1.tours[k], child2.tours[k] = self.crossover_ox(parent1.tours[k], parent2.tours[k])

            if self.rng.rand() < self.cx_pb_pack:
                a1, a2 = self.crossover_uniform(child1.item_assignment, child2.item_assignment)
                child1.item_assignment = a1
                child2.item_assignment = a2
                
            alpha = self.rng.rand()
            c1_charge = alpha * parent1.charge_aggressiveness + (1 - alpha) * parent2.charge_aggressiveness
            c2_charge = (1 - alpha) * parent1.charge_aggressiveness + alpha * parent2.charge_aggressiveness
            
            child1.charge_aggressiveness = np.clip(c1_charge, 0.0, 1.0)
            child2.charge_aggressiveness = np.clip(c2_charge, 0.0, 1.0)
            
            for child in [child1, child2]:
                for k in range(self.problem.num_robots):
                    self.mutate_swap(child.tours[k])
                tour_city_sets = [set(t) for t in child.tours]
                self.mutate_bitflip(tour_city_sets, child.item_assignment)

                if self.rng.rand() < self.mut_pb_pack:
                    noise = self.rng.normal(0, 0.1, size=self.problem.num_robots)
                    child.charge_aggressiveness += noise
                    child.charge_aggressiveness = np.clip(child.charge_aggressiveness, 0.0, 1.0)

            child1.repair()
            child2.repair()
            
            offspring.append(child1)
            if len(offspring) < self.pop_size:
                offspring.append(child2)
        
        self.population = offspring[:self.pop_size]
        return geno_div, pheno_div, priority_div
    
    def evolve(self):
        """        
        - Population initialization -> Hybrid (Random + Greedy)
        - While generation < max_gen:
            - Adapt Tchebycheff weights
            - Update scalar fitness
            - Survivor Selection -> Elitism
            - Offspring Generation:
                - Mate Selection -> Tournament
                - Crossover for tours (OX) and items (Uniform)
                - Mutation for tours (Swap) and items (Bit-flip)
                - Repair
            - Evaluation
            - Update best individual
        """
        start_time = time.time()
       
        # Initialize new run
        self.initialize_population()
        self.evaluate_population()
        self.update_best_individual()
        geno_div, pheno_div, priority_div = self.calculate_diversity()
        start_gen = 1
        
        # Evolution loop
        for gen in range(start_gen, self.generations + 1):
            self.current_gen = gen
            
            # Adapt weights based on progress
            self.adapt_weights(gen)
            
            for ind in self.population:
                ind.scalar_fitness = self.calculate_scalar_fitness(ind.fitness)
            
            # Evolve one generation
            geno_div, pheno_div, priority_div = self.evolve_generation()
            self.evaluate_population()
            self.update_best_individual()
            
            self.compute_pareto_front()
            
            step = max(1, self.generations // 4)
            if gen % step == 0 or gen == self.generations:
                mksp = self.best_individual.fitness[0]
                twt = self.best_individual.fitness[1]
            
            elapsed = time.time() - start_time
            
        return elapsed, geno_div, pheno_div, priority_div