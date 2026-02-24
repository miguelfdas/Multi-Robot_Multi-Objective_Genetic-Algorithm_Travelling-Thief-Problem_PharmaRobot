"""
S5 Baseline Algorithm for the Travelling Thief Problem (TTP)
S5 is the best single-component algorithm in the 9720-instance of TTP

Most of the code come from LLM-generated implementation of the S5 algorithm 

Algorithm overview:
- Construct a TSP tour using the Nearest-Neighbour heuristic improved with 2-opt local search
- Apply PackIterative to produce a packing plan:
    - Score items by profit / (weight x distance_remaining_from_city)
    - Greedily select items in score order respecting capacity
    - Iteratively flip each bit; keep change if it improves G
- Restart step 2 with shuffled item ordering until the time budget is exhausted
- Return the individual (Individual instance) with the best fitness

References:

Faulkner, H., Polyakovskiy, S., Schultz, T., & Wagner, M. (2015).
"Approximate Approaches to the Travelling Thief Problem."
GECCO 2015, ACM, pp. 799-806.

Polyakovskiy, S., & Neumann, F. (2017).
"Packing While Travelling: Mixed Integer Programming for a Class of
Nonlinear Knapsack Problems."
European Journal of Operational Research, 258(2), 424-441.

Lin & Kernighan (1973) - Nearest-Neighbour TSP heuristic origin

Croes, G. A. (1958) - 2-opt local search for TSP
"""

import time
import numpy as np
from ttp_problem import load_ttp_benchmark
from individual import Individual

def _nearest_neighbour_tour(problem):
    """
    Construct a greedy TSP tour using the Nearest-Neighbour heuristic.

    Algorithm (Lin & Kernighan, 1973 - greedy TSP initialisation):
    - Start at city 0 (depot)
    - Repeatedly visit the closest unvisited city
    - Stop when all cities have been visited

    The tour is returned in the format expected by Individual:
        [0, c1, c2, ..., cn]

    Args:
        problem: TTProblem instance with precomputed distance_matrix.

    Returns:
        List of city indices forming the tour.

    Complexity: O(n²) - acceptable for benchmark instances up to ~1000 cities.
    """
    
    n = problem.num_cities
    dist = problem.distance_matrix

    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True

    for _ in range(n - 1):
        current = tour[-1] # last visited city
        row = dist[current].copy()
        row[visited] = np.inf # mask already-visited cities
        nearest = int(np.argmin(row)) # index of nearest unvisited city
        tour.append(nearest)
        visited[nearest] = True

    return tour # length = num_cities

def two_opt_improve(tour, dist, max_iter=500):
    """
    Improve a TSP tour with 2-opt local search (Croes, 1958).

    Repeatedly reverses sub-segments of the tour when doing so reduces
    total tour length.  Stops after `max_iter` passes with no improvement
    to keep runtime bounded.

    Args:
        tour: List of city indices (no trailing depot)
        dist: Ceiling distance matrix (n x n numpy array)
        max_iter: Maximum number of improvement passes

    Returns:
        Improved tour as a list.

    Reference:
        Croes, G. A. (1958). "A method for solving traveling-salesman
        problems." Operations Research, 6(6), 791-812.
    """
    best = tour[:]
    n = len(best)
    improved = True
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Cities involved in the two edges being swapped
                a, b = best[i - 1], best[i]
                c, d = best[j], best[(j + 1) % n]

                # Current cost of the two edges
                current_cost = dist[a, b] + dist[c, d]
                # Cost if we reconnect: a-c and b-d (reverse segment i..j)
                new_cost = dist[a, c] + dist[b, d]

                if new_cost < current_cost - 1e-10:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    improved = True

    return best

def _compute_distance_remaining(tour, dist):
    """
    Precompute, for each city in the tour, the remaining travel distance
    from that city to the end of the tour (returning to the depot, city 0).

    This is a key component of the PackIterative scoring function.

    Faulkner et al. (2015) use this to reward picking items near the end of
    the tour, since carrying them incurs less speed penalty over fewer edges.

    The tour is traversed in reverse:
        - dist_remaining[last_city]  = dist(last_city → depot)
        - dist_remaining[city_i]     = dist(city_i → city_{i+1}) + dist_remaining[city_{i+1}]

    Args:
        tour: List of city indices [0, c1, ..., cn].
        dist: Distance matrix (n × n).

    Returns:
        dict mapping city_index → remaining distance (float).
    """
    n = len(tour)
    remaining = {}
    cumulative = 0.0

    # Walk backwards: from last city → depot is the first edge we add
    # last city is tour[n-1]; the wrap-around edge is tour[n-1] → tour[0]
    for pos in range(n - 1, -1, -1):
        city = tour[pos]
        next_city = tour[(pos + 1) % n] # wrap-around for last city → depot
        cumulative += dist[city, next_city]
        remaining[city] = cumulative

    return remaining


def _pack_iterative(individual, rng):
    """
    PackIterative packing heuristic (Faulkner et al., 2015).

    Operates in two phases on a fixed tour stored in individual.tours[0]:

    Phase A - Greedy initialisation:
        Score each item:  score_i = profit_i / (weight_i x dist_remaining[city_i])
        Items closer to the end of the tour score higher (less speed penalty
        from carrying them).  Select items greedily in score order, respecting
        the knapsack capacity constraint.

    Phase B - Iterative bit-flip improvement:
        Shuffle item order (randomised restart behaviour).
        For each item:
            - If currently picked  → try un-picking it; keep if G improves.
            - If currently skipped → try picking it (if capacity allows);
              keep if G improves.
        Repeat until a full pass over all items yields no improvement.

    The method modifies individual.item_assignment and individual.fitness
    in-place.

    Args:
        individual: Individual instance (benchmark mode, single robot).
        rng:        NumPy RandomState for reproducible shuffling.

    Reference:
        Faulkner et al. (2015), Algorithm 2 - PackIterative.
    """
    problem = individual.problem
    tour = individual.tours[0]
    dist = problem.distance_matrix
    capacity = problem.knapsack_capacity
    n_items = problem.num_items

    items = problem.items # (n_items × 4): [city, profit, weight, prio]

    dist_remaining = _compute_distance_remaining(tour, dist)

    # score_i = profit_i / (weight_i × dist_remaining[city_i])
    # Higher score → prefer this item  (cheap to carry, high value)
    scores = np.zeros(n_items)
    for i in range(n_items):
        city_i = int(items[i, 0])
        d_rem = dist_remaining.get(city_i, 1e-10)
        scores[i] = items[i, 1] / (items[i, 2] * max(d_rem, 1e-10))

    sorted_idx = np.argsort(-scores) # descending

    assignment  = np.full(n_items, -1, dtype=int)
    total_weight = 0.0

    for idx in sorted_idx:
        w = items[idx, 2]
        if total_weight + w <= capacity:
            assignment[idx] = 0 # robot 0 (single-robot benchmark)
            total_weight += w

    individual.item_assignment = assignment

    # Evaluate after greedy init
    individual.evaluate()
    best_fitness = individual.fitness # = -G; lower is better

    improved = True
    while improved:
        improved = False
        item_order = rng.permutation(n_items) # randomised order each pass

        current_weight = float(np.sum(items[individual.item_assignment == 0, 2]))

        for idx in item_order:
            w = items[idx, 2]
            picked = (individual.item_assignment[idx] == 0)

            if picked:
                # Try un-picking
                individual.item_assignment[idx] = -1
                individual.evaluate()
                if individual.fitness < best_fitness:
                    best_fitness = individual.fitness
                    current_weight -= w
                    improved = True
                else:
                    # Revert
                    individual.item_assignment[idx] = 0
                    individual.fitness = best_fitness

            else:
                # Try picking (only if capacity allows)
                if current_weight + w <= capacity:
                    individual.item_assignment[idx] = 0
                    individual.evaluate()
                    if individual.fitness < best_fitness:
                        best_fitness = individual.fitness
                        current_weight += w
                        improved = True
                    else:
                        # Revert
                        individual.item_assignment[idx] = -1
                        individual.fitness = best_fitness


def run_s5(filepath, time_budget_seconds=60.0, seed=42):
    """
    Run the S5 algorithm on a standard TTP benchmark instance.

    S5 strategy (Faulkner et al., 2015):
        1. Build a deterministic TSP tour: Nearest-Neighbour + 2-opt.
           The tour is fixed throughout all restarts.
        2. Apply PackIterative with a fresh random seed each restart.
        3. Track the best Individual found across all restarts.
        4. Return when the time budget is exhausted.

    The tour is fixed because the original S5 paper found that a good fixed
    tour combined with repeated packing restarts outperforms jointly
    re-optimising both components on most benchmark instances.

    Args:
        filepath: Path to a .ttp benchmark file.
        time_budget_seconds: Wall-clock time limit in seconds.

    Returns:
        best_individual: Individual instance with best fitness found.
                         best_individual.fitness == -G  (benchmark mode)
                         So G = -best_individual.fitness
    """

    problem = load_ttp_benchmark(filepath, mode='benchmark', seed=seed)

    nn_tour = _nearest_neighbour_tour(problem)
    tour_2opt = two_opt_improve(nn_tour, problem.distance_matrix)

    master_rng = np.random.RandomState(seed)
    best_ind = None
    restart_num = 0
    start_time = time.time()

    while (time.time() - start_time) < time_budget_seconds:
        restart_num += 1

        # Each restart gets its own RNG child seed for reproducibility
        child_seed = master_rng.randint(0, 2**31 - 1)
        child_rng = np.random.RandomState(child_seed)

        # Build a fresh Individual with the fixed tour and empty packing
        candidate = Individual(problem, tours=[tour_2opt[:]],item_assignment=np.full(problem.num_items, -1, dtype=int),)

        # Run PackIterative (modifies candidate in-place)
        _pack_iterative(candidate, child_rng)

        # Update best
        if best_ind is None or candidate.fitness < best_ind.fitness:
            best_ind = candidate.copy()

    return best_ind