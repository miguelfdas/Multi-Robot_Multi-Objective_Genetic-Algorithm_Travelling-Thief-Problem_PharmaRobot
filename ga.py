"""
Genetic Algorithm

Multi-Objective Weighted Tchebycheff Aggregation:

- Repair always, never penalize
- Greedy repair preserves high-value items

Parent Selection:
- Tournament Selection [Miller & Goldberg, 1995]
- Tournament size = 2

Crossover:

Order Crossover (OX) for tours [Davis, 1985]
- Preserves relative order of cities
- Prevents invalid tours (duplicate cities)
- Good for permutation problems

Uniform Crossover for item assignment [Syswerda, 1989]
- Each item independently inherits from parent
- 50% probability from each parent
- Good for binary/integer encodings

Mutation:

Swap Mutation for tours [Banzhaf, 1990]
- Exchange two random cities
- Maintains tour validity

Bit-flip Mutation for items
- Randomly reassign items to different robots

Repair:
- Greedy weight-based repair [Michalewicz & Schoenauer, 1996]
- Removes low-value items when capacity exceeded
"""

import numpy as np
import pandas as pd
import time
import pickle
import os
from joblib import Parallel, delayed
from individual import Individual

class GeneticAlgorithm:
    """
    Genetic Algorithm
    Dynamic weight adjustment (multi-objective)
    """
    
    def __init__(self, problem, pop_size=100, generations=500, cx_pb_tour=0.8, cx_pb_pack=0.8, mut_pb_tour=0.2, mut_pb_pack=0.02, tournament_size=2, elitism=2, n_jobs=-1, seed=None):
        """
        Initialize Genetic Algorithm.
        
        Args:
            problem: TTProblem instance
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
        self.mut_pb_pack = mut_pb_pack # mut_pb_pack = 0.002 = 1/n = 1/495 where n is the num_items
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.n_jobs = n_jobs
        self.seed = seed
        
        self.num_objectives = 5
        
        self.population = []
        self.metrics = []
        self.current_gen = 0
        
        # Bounds for adaptive mutation (never go below base / 5 or above base * 5)
        self.mut_tour_base = mut_pb_tour
        self.mut_pack_base = max(mut_pb_pack, 3.0 / max(problem.num_items, 1))
        self.mut_tour_min  = mut_pb_tour / 5.0
        self.mut_tour_max  = min(mut_pb_tour * 5.0, 1.0)
        self.mut_pack_min  = self.mut_pack_base / 5.0
        self.mut_pack_max  = min(self.mut_pack_base * 10.0, 1.0)
        
        # Multi-objective weights (adaptive)
        # Initial weights: equal importance to all objectives
        self.objective_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # [tct, Makespan, profit, priority_profit, load_imbalance]
        
        # Reference point (ideal point in objective space)
        # Updated during evolution to track best found values
        self.reference_point = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
        
        # Nadir point (worst values)
        self.nadir_point = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        
        # Hypervolume reference point (worse than any expected solution) for hypervolume calculation
        # Hipervolume is a common multi-objective performance metric that measures the volume of objective space dominated by the Pareto front and bounded by a reference point.
        self.hv_reference_point = np.array([5000.0, 200000.0, 300000.0, 0.0, 100.0]) # [tct, Makespan, profit, priority_profit, load_imbalance]
        
        self.rng = np.random.RandomState(seed)
        
        self.best_individual = None # best solution found
        self.best_fitness = np.inf # fitness of best solution found
        self.best_found_at_gen = 0 # generation when best solution was found
    
    def normalize_objectives(self, objectives):
        """
        Normalize objectives to [0, 1] (Li & Zhang, 2009):
        To all objectives be comparable and to prevent domination by any single objective
        Use ideal and nadir points to normalize each objective:
        
        normalized_f_i = (f_i - z_i*) / (z_nad_i - z_i*)        
        - f_i = raw objective value
        - z_i* = ideal point (best value found)
        - z_nad_i = nadir point (worst value found)
        """
        
        normalized = np.zeros(self.num_objectives)
        for i in range(self.num_objectives):
            range_i = self.nadir_point[i] - self.reference_point[i]
            if range_i > 1e-10:
                normalized[i] = (objectives[i] - self.reference_point[i]) / range_i
            else:
                normalized[i] = 0.0
        return np.clip(normalized, 0, 1)
    
    def calculate_scalar_fitness(self, objectives):
        """
        Convert multi-objective fitness to scalar using Weighted Tchebycheff Aggregation (Zhang & Li, 2007):
        fitness = max_i { w_i x |f_i - z_i*| }
        """
                
        if self.problem.mode == 'benchmark':
            # For benchmark mode, we want to maintain the original TTP objective
            return objectives
        
        else:
            # Normalize objectives first
            norm_obj = self.normalize_objectives(objectives)
            # Weighted Tchebycheff
            weighted_distances = self.objective_weights * norm_obj
            return np.max(weighted_distances)
    
    def calculate_fixed_scalar_fitness(self, objectives):
        """
        Scalar fitness with fixed weights [1/5, 1/5, 1/5, 1/5, 1/5] for all objectives, for logging
        It is not affected by weight adaptation, allowing consistent comparison between generations
        """  
        if self.problem.mode == 'benchmark':
            return objectives
        else:  
            fixed_weights = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
            norm_obj = self.normalize_objectives(objectives)
            weighted_distances = fixed_weights * norm_obj
            return np.max(weighted_distances)
    
    def update_reference_points(self):
        """
        Pareto-front to Multi-objective
        Update ideal and nadir points based on current population based on Adaptive Reference Point (Li & Zhang, 2009):
        - Ideal point: min value per objective (best found)
        - Nadir point: max value per objective (worst found)        
        """      
        if not self.problem.mode == 'benchmark':
            if len(self.population) == 0:
                return
            
            # Extract objectives from population
            objectives_array = np.array([ind.fitness for ind in self.population])
            
            # Update ideal point (minimize all objectives)
            current_ideal = np.min(objectives_array, axis=0)
            self.reference_point = np.minimum(self.reference_point, current_ideal)
            
            # Update nadir point (maximize all objectives)
            current_nadir = np.max(objectives_array, axis=0)
            self.nadir_point = np.maximum(self.nadir_point, current_nadir)
        else:
            return
    
    def adapt_weights(self, generation):        
        """
        Adapt objective weights during evolution to better exploration and trade-offs
        
        Early generations: Emphasize Makespan and tct
        Later generations: Emphasize profit and imbalance
        """
        
        if not self.problem.mode == 'benchmark':
            # Progress ratio: 0 at start, 1 at end
            progress = generation / self.generations
            
            # Adaptive weighting schedule:
            
            # Early: [0.25, 0.25, 0.3, 0.1, 0.1] - focus on Makespan and tct
            # Late:  [0.1, 0.1, 0.3, 0.25, 0.25] - focus on profit and priority_profit
            
            w_tct = 0.25 * (1 - progress) + 0.1 * progress # decrease from 0.25 to 0.1
            w_Makespan = 0.25 * (1 - progress) + 0.1 * progress # decrease from 0.25 to 0.1
            w_G = 0.3
            w_profit = 0.1 * (1 - progress) + 0.25 * progress # increase from 0.1 to 0.25
            w_imbalance = 0.1 * (1 - progress) + 0.25 * progress # increase from 0.1 to 0.25
            
            self.objective_weights = np.array([w_tct, w_Makespan, w_G, w_profit, w_imbalance])
        else:
            return
        
    def initialize_population(self):
        """        
        Initialize population with Hybrid Strategy (Faulkner et al., 2015):

        50% random: Random with Repair
        50% greedy: Greedy Nearest-Neighbor + Profit-Ratio Packing with Repair
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
        - Scalar fitness computed via Tchebycheff aggregation
        """
        
        # Evaluation of multi-objective fitness            
        fitness_results = Parallel(n_jobs=self.n_jobs)(delayed(ind.evaluate)() for ind in self.population)
        for ind, fit in zip(self.population, fitness_results):
            ind.fitness = fit
            
        self.update_reference_points()   
        
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
        Order Crossover (OX) (Davis (1985))
        
        - Select random substring from parent1
        - Copy substring to offspring1
        - Fill remaining positions with parent2's order
        - Symmetric for offspring2
        
        - Preserves relative order of cities
        - Prevents duplicate cities (valid tour)
        - Adjacent cities matter
        """
        
        size = len(parent1_tour)
        
        # Select two crossover points (exclude position 0, origin) and ensure they are different      
        point1, point2 = sorted(self.rng.choice(range(1, size), size=2, replace=False))
        if point1 == point2:
            point2 = point1 + 1
            if point2 >= size:
                point2 = size
        
        # Initialize offspring
        offspring1 = [-1] * size
        offspring2 = [-1] * size
        offspring1[0] = 0  # Depot always first
        offspring2[0] = 0
        
        # Copy substring
        offspring1[point1:point2] = parent1_tour[point1:point2]
        offspring2[point1:point2] = parent2_tour[point1:point2]
        
        # Fill remaining positions            
        donor_order = [city for city in parent2_tour if city not in offspring1]
        idx = 0
        for i in range(1, size):
            if offspring1[i] == -1:
                offspring1[i] = donor_order[idx]
                idx += 1
                
        donor_order = [city for city in parent1_tour if city not in offspring2]
        idx = 0
        for i in range(1, size):
            if offspring2[i] == -1:
                offspring2[i] = donor_order[idx]
                idx += 1
        
        return offspring1, offspring2
    
    def crossover_uniform(self, parent1_assign, parent2_assign):
        """
        Uniform Crossover for item assignment (Syswerda, 1989) 
        
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
        2-opt Swap Mutation for tours

        Selects two positions and reverses the segment between them to reduce route crossings and improve tour quality.

        With probability mut_pb_tour:
        - Select two random positions (excluding origin)
        - Reverses the tour segment [i:j]
        - Maintains the validity of the tour (no duplicates)
        """
        
        if self.rng.rand() < self.mut_pb_tour:
            size = len(tour)
            if size > 3:
                # Select two cut points (excluding position 0, origin)
                i, j = sorted(self.rng.randint(1, size, size=2))
                if i != j:
                    # Reverse the segment between i and j
                    tour[i:j] = tour[i:j][::-1]
    
    def mutate_bitflip(self, tour_city_sets, assignment):
        """
        Profit-Guided Bit-Flip Mutation for item assignment (Back, 1996)

        Instead of flipping every item with the same probability
        Each item gets an individual flip probability scaled by its profit/weight ratio:

        p_i = mut_pb_pack * (ratio_i / mean_ratio)

        - High value-density items  -> higher probability of being flipped IN
        - Low  value-density items  -> higher probability of being flipped OUT

        The directional bias is applied only for the choice of action:
        - Currently unassigned (assignment == -1):
            flip probability is boosted for high-ratio items  (try to pick them)
        - Currently assigned:
            flip probability is boosted for low-ratio  items  (try to drop them)

        After mutation, Repair
        """
        # Compute profit/weight ratio for every item (once per call)
        profits  = self.problem.items[:, 1]
        weights  = self.problem.items[:, 2]
        ratios   = profits / np.maximum(weights, 1e-10)
        mean_ratio = ratios.mean() if ratios.mean() > 0 else 1.0

        for i in range(len(assignment)):
            item_city = int(self.problem.items[i, 0])
            valid_robots = [r for r, city_set in enumerate(tour_city_sets) if item_city in city_set]
            if not valid_robots:
                continue

            # Scale mut_pb_pack by relative value density
            scaled_prob = self.mut_pb_pack * (ratios[i] / mean_ratio)
            scaled_prob = float(np.clip(scaled_prob, 0.0, 1.0))

            if self.rng.rand() < scaled_prob:
                if assignment[i] == -1:
                    # Item currently unassigned -> try to assign to a valid robot
                    assignment[i] = self.rng.choice(valid_robots)
                else:
                    # Item currently assigned -> unassign or reassign
                    choices = valid_robots + [-1]
                    assignment[i] = self.rng.choice(choices)
    
    def calculate_diversity(self):
        """
        Calculate population diversity.

        Genotypic diversity:
            - Measures structural differences in chromosomes
            - Combines:
                - Pairwise Hamming distance on item_assignment vectors.
                - Pairwise Jaccard edge-disagreement on tours.
            Both components live in [0, 1]

        Phenotypic diversity:
            - Behavioural diversity in the solution space
            - Combines:
                - Binary Shannon entropy of tour-edge usage frequencies.
                - Binary Shannon entropy of item-selection frequencies.
            Both components live in [0, 1]

        Priority diversity:
            Measures whether individuals differ in their tendency to select prioritys.
            Mean per-bin standard deviation of priority-fraction distributions. 
        """

        if len(self.population) < 2:
            return 0.0, 0.0, 0.0

        n_inds = len(self.population)

        # Item assignment matrix  (n_inds × n_items)
        assignments = np.array([ind.item_assignment for ind in self.population])

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
                tour_diversity = 0.0 # all edges unanimous → population has converged
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

        if self.problem.mode == 'benchmark':
            # All benchmark items share priority = 1; metric is identically 0.
            priority_div = 0.0
        else:
            priority_distributions = []
            for ind in self.population:
                selected = ind.item_assignment >= 0
                if np.any(selected):
                    priorities = self.problem.items[selected, 3]
                    dist = np.bincount(
                        priorities.astype(int), minlength=4
                    )[1:].astype(float)          # bins for priorities 1, 2, 3
                    dist /= dist.sum() + 1e-10
                    priority_distributions.append(dist)

            if len(priority_distributions) >= 2:
                priority_distributions = np.array(priority_distributions)   # (N, 3)
                raw = float(np.mean(np.std(priority_distributions, axis=0, ddof=1)))
                priority_div = max(0.0, raw)   # clip fp residual to zero
            else:
                priority_div = 0.0

        return genotypic, phenotypic, priority_div
    
    def update_best_individual(self):
        for ind in self.population:
            if ind.scalar_fitness < self.best_fitness:
                self.best_fitness = ind.scalar_fitness
                self.best_individual = ind.copy()
                self.best_found_at_gen = self.current_gen
                
    def adapt_mutation_rate(self, genotypic_diversity):
        """
        Adaptive Mutation Rate based on Population Diversity (Eiben et al., 1999).

        The intuition is:
          - When diversity is *low*  the population has converged → increase
            mutation to escape local optima and re-introduce variation.
          - When diversity is *high* exploration is already adequate → reduce
            mutation to avoid disrupting good solutions.

        A normalised diversity signal d ∈ [0, 1] is mapped to a mutation
        scale factor via a monotonically *decreasing* linear schedule:

        - scale(d) = scale_max - d x (scale_max - scale_min)

        where scale_max = 3.0 and scale_min = 0.5.
        The raw rates are clamped to [_mut_*_min, _mut_*_max] so they never
        degenerate to zero or become excessively disruptive.

        Eiben, A. E., Hinterding, R., & Michalewicz, Z. (1999).
        "Parameter control in evolutionary algorithms."
        IEEE Transactions on Evolutionary Computation, 3(2), 124–141.
        """
        # Target diversity for "neutral" (no change) mutation rate
        scale_high = 3.0   # multiplier when diversity → 0
        scale_low = 0.5    # multiplier when diversity → 1

        d_norm = float(np.clip(genotypic_diversity, 0.0, 1.0))

        # Linear interpolation: high scale when low diversity, low scale when high
        # At d=0: scale = scale_high
        # At d=target: scale ≈ 1  (neutral)
        # At d=1: scale = scale_low
        scale = scale_high - d_norm * (scale_high - scale_low)

        self.mut_pb_tour = float(np.clip(self.mut_tour_base * scale, self.mut_tour_min, self.mut_tour_max))
        self.mut_pb_pack = float(np.clip(self.mut_pack_base * scale, self.mut_pack_min, self.mut_pack_max))
    
    def compute_pareto_front(self, objectives_array):
        """
        Extract the approximate Pareto front from the current population
        A solution is non-dominated if no other solution is better or equal in all objectives and strictly better in at least one
        """
        
        n = len(objectives_array)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if j is <= in all and < in at least one
                if (np.all(objectives_array[j] <= objectives_array[i]) and np.any(objectives_array[j] < objectives_array[i])):
                    is_dominated[i] = True
                    break
        
        return objectives_array[~is_dominated]

    def compute_hypervolume_nd(self, points: np.ndarray, ref: np.ndarray) -> float:
        """
        Exact hypervolume for n-dimensional objective spaces.

        Implements the HSO (Hypervolume by Slicing Objectives) algorithm recursively (Zitzler & Thiele, 1998; While et al., 2012).

        HV_n(S, r) = Σ_k  HV_{n-1}(S_k, r[:-1]) × (prev_last − s_k[−1])
        """
        if len(points) == 0:
            return 0.0

        n_dims = points.shape[1]

        # Keep only points that strictly dominate ref in ALL objectives
        dominated_mask = np.all(points < ref, axis=1)
        pts = points[dominated_mask]
        if len(pts) == 0:
            return 0.0

        # Base case 1-D
        if n_dims == 1:
            return float(ref[0] - pts[:, 0].min())

        # Base case 2-D: standard sweep-line
        if n_dims == 2:
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            area = 0.0
            current_f2 = ref[1]
            for p in pts:
                if p[1] < current_f2:
                    area += (ref[0] - p[0]) * (current_f2 - p[1])
                    current_f2 = p[1]
            return area

        # Recursive n-D: slice along the last objective
        idx = np.argsort(pts[:, -1])
        pts = pts[idx]

        hv = 0.0
        prev_last = ref[-1]

        for k in range(len(pts) - 1, -1, -1):
            slab_height = prev_last - pts[k, -1]   # ≥ 0
            if slab_height > 0:
                # Recurse on the (n-1)-D projection of all points up to index k
                hv += self.compute_hypervolume_nd(pts[:k + 1, :-1], ref[:-1]) * slab_height
            prev_last = pts[k, -1]

        return hv
    
    def record_metrics(self, generation, elapsed, geno_div, pheno_div, prior_div):
        """
        Record metrics in each generation.
        """
        
        # Extract fitness values
        scalar_fits = np.array([ind.scalar_fitness for ind in self.population])
        objectives = np.array([ind.fitness for ind in self.population])
                        
        best_idx = np.argmin(scalar_fits)
                        
        if not self.problem.mode == 'benchmark':
            # Pareto front
            pareto_front = self.compute_pareto_front(objectives)
            
            # Normalise Pareto front to [0, 1] using current ideal/nadir, then compute HV with a fixed reference point of [1.1, 1.1, 1.1].
            norm_front = np.zeros_like(pareto_front)
            for i in range(self.num_objectives):
                r_i = self.nadir_point[i] - self.reference_point[i]
                if r_i > 1e-10:
                    norm_front[:, i] = (pareto_front[:, i] - self.reference_point[i]) / r_i
                else:
                    norm_front[:, i] = 0.0
            norm_front = np.clip(norm_front, 0.0, 1.1)
            hv_ref_norm = np.array([1.1, 1.1, 1.1, 1.1, 1.1])

            # Hypervolume
            hv = self.compute_hypervolume_nd(norm_front, hv_ref_norm)
        
        # Record metrics
        self.metrics.append({
            'generation': generation,
            
            # Scalar fitness
            'best_fitness': self.calculate_fixed_scalar_fitness(objectives[best_idx]),
            'avg_fitness': np.mean([self.calculate_fixed_scalar_fitness(objectives[i]) for i in range(len(self.population))]),
            'std_fitness': np.std([self.calculate_fixed_scalar_fitness(objectives[i]) for i in range(len(self.population))]),
            
            # Individual objectives (best individual)
                        
            'best_G': -self.best_fitness if self.problem.mode == 'benchmark' else -objectives[best_idx, 2],
            'best_tct': np.nan if self.problem.mode == 'benchmark' else objectives[best_idx, 0],
            'best_Makespan': np.nan if self.problem.mode == 'benchmark' else objectives[best_idx, 1],
            'best_profit': np.nan if self.problem.mode == 'benchmark' else objectives[best_idx, 3],
            'best_imbalance': np.nan if self.problem.mode == 'benchmark' else objectives[best_idx, 4],
            
            # Multi-objective tracking
            'ideal_G': np.nan if self.problem.mode == 'benchmark' else -self.reference_point[2],
            'ideal_tct': np.nan if self.problem.mode == 'benchmark' else self.reference_point[0],
            'ideal_Makespan': np.nan if self.problem.mode == 'benchmark' else self.reference_point[1],
            'ideal_profit': np.nan if self.problem.mode == 'benchmark' else self.reference_point[3],
            'ideal_imbalance': np.nan if self.problem.mode == 'benchmark' else self.reference_point[4],
            
            # Weights
            'weight_G': np.nan if self.problem.mode == 'benchmark' else self.objective_weights[2],
            'weight_travel': np.nan if self.problem.mode == 'benchmark' else self.objective_weights[0],
            'weight_Makespan': np.nan if self.problem.mode == 'benchmark' else self.objective_weights[1],
            'weight_profit': np.nan if self.problem.mode == 'benchmark' else self.objective_weights[3],
            'weight_imbalance': np.nan if self.problem.mode == 'benchmark' else self.objective_weights[4],
            
            'hypervolume': np.nan if self.problem.mode == 'benchmark' else hv,
            'pareto_front_size': np.nan if self.problem.mode == 'benchmark' else len(pareto_front),
            
            # Diversity
            'genotypic_diversity': geno_div,
            'phenotypic_diversity': pheno_div,
            'priority_diversity': prior_div,
            
            # Other
            'elapsed_time_seconds': elapsed,
        })
    
    def evolve_generation(self):
        """
        - Elitism (Goldberg & Deb, 1991)
        - Generate offspring:
            - Tournament selection
            - Crossover (OX for tours, Uniform for items)
            - Mutation (Swap for tours, Bit-flip for items)
            - Repair
        - Replace population
        """
        
        geno_div, pheno_div, prior_div = self.calculate_diversity()
        self.adapt_mutation_rate(geno_div)
        
        # Elitism: sort by scalar fitness (minimization)
        self.population.sort(key=lambda x: x.scalar_fitness)
        offspring = [ind.copy() for ind in self.population[:self.elitism]]
        
        # Generate offspring
        while len(offspring) < self.pop_size:
            parents = self.tournament_selection(k=2)
            parent1, parent2 = parents[0], parents[1]
                        
            child1 = parent1.copy()
            child2 = parent2.copy()

            if self.rng.rand() < self.cx_pb_tour:
                for k in range(self.problem.num_robots):
                    child1.tours[k], child2.tours[k] = self.crossover_ox(parent1.tours[k], parent2.tours[k])
                child1.repair()
                child2.repair()

            if self.rng.rand() < self.cx_pb_pack:
                a1, a2 = self.crossover_uniform(child1.item_assignment, child2.item_assignment)
                child1.item_assignment = a1
                child2.item_assignment = a2
                child1.repair()
                child2.repair()
            
            for child in [child1, child2]:
                for k in range(self.problem.num_robots):
                    self.mutate_swap(child.tours[k])
                tour_city_sets = [set(t) for t in child.tours]
                self.mutate_bitflip(tour_city_sets, child.item_assignment)
                child.repair()
            
            # Add to offspring
            offspring.append(child1)
            if len(offspring) < self.pop_size:
                offspring.append(child2)
        
        self.population = offspring[:self.pop_size]
        
        return geno_div, pheno_div, prior_div
    
    def save_checkpoint(self, filename):
        """
        Save state.
        """
        
        state = {
            'population': [ind.to_dict() for ind in self.population],
            'current_gen': self.current_gen,
            'metrics': self.metrics,
            'seed': self.seed,
            'rng_state': self.rng.get_state(),
            'objective_weights': self.objective_weights.tolist(),
            'reference_point': self.reference_point.tolist(),
            'nadir_point': self.nadir_point.tolist(),
            'best_individual': self.best_individual.to_dict() if self.best_individual is not None else None,
            'best_fitness': float(self.best_fitness),
            'best_found_at_gen': int(self.best_found_at_gen),
        }
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
    
    def load_checkpoint(self, filename):
        """
        Load state.
        """
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            
            # Restore population
            self.population = [Individual.from_dict(self.problem, d) for d in state['population']]
            self.current_gen = state['current_gen']
            self.metrics = state['metrics']
            
            # Restore RNG
            if 'rng_state' in state:
                self.rng.set_state(state['rng_state'])
            # Restore adaptive components
            if 'objective_weights' in state:
                self.objective_weights = np.array(state['objective_weights'])
            if 'reference_point' in state:
                self.reference_point = np.array(state['reference_point'])
            if 'nadir_point' in state:
                self.nadir_point = np.array(state['nadir_point'])  
            if state.get('best_individual') is not None:
                self.best_individual = Individual.from_dict(self.problem, state['best_individual'])
            self.best_fitness = float(state.get('best_fitness', np.inf))
            self.best_found_at_gen = int(  state.get('best_found_at_gen', 0))
            
            return True
        
        except Exception as e:
            return False
    
    def evolve(self, checkpoint_interval=1, checkpoint_file=None):
        """        
        - Population initialization -> Hybrid (Random + Greedy)
        - While generation < max_gen:
            - Adapt weights
            - Survivor Selection -> Elitism
            - Offspring Generation:
                - Mate Selection -> Tournament
                - Crossover for tours (OX) and items (Uniform)
                - Mutation for tours (Swap) and items (Bit-flip)
                - Repair
            - Evaluation
        """
        start_time = time.time()
        
        # Try to load checkpoint
        if checkpoint_file and self.load_checkpoint(checkpoint_file):
            start_gen = self.current_gen + 1
            self.update_reference_points()
            for ind in self.population:
                ind.scalar_fitness = self.calculate_scalar_fitness(ind.fitness)
        else:
            # Initialize new run
            self.initialize_population()
            self.evaluate_population()
            self.update_best_individual()
            geno_div, pheno_div, prior_div = self.calculate_diversity()
            self.record_metrics(0, 0, geno_div, pheno_div, prior_div)
            start_gen = 1
            if checkpoint_file:
                self.save_checkpoint(checkpoint_file)
        
        # Evolution loop
        for gen in range(start_gen, self.generations + 1):
            self.current_gen = gen
            
            # Adapt weights based on progress
            self.adapt_weights(gen)
            
            for ind in self.population:
                ind.scalar_fitness = self.calculate_scalar_fitness(ind.fitness)
            
            # Evolve one generation
            geno_div, pheno_div, prior_div = self.evolve_generation()
            self.evaluate_population()
            self.update_best_individual()
            
            # Record metrics
            elapsed = time.time() - start_time
            self.record_metrics(gen, elapsed, geno_div, pheno_div, prior_div)
            
            # Save checkpoint
            if checkpoint_file and gen % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_file)
            
            # Print progress
            if gen == 1 or gen % 50 == 0 or gen == self.generations:
                m = self.metrics[-1]
                if self.problem.mode == 'benchmark':
                    print(f"Gen {gen} - Best Fitness: {-m['best_fitness']:.4f}")
                else:
                    print(f"Gen {gen} - Best Fitness: {m['best_fitness']:.4f}")
        
        # Final checkpoint
        if checkpoint_file:
            self.save_checkpoint(checkpoint_file)
                
    def save_metrics(self, filename):
        # Save metrics to CSV
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        pd.DataFrame(self.metrics).to_csv(filename, index=False)