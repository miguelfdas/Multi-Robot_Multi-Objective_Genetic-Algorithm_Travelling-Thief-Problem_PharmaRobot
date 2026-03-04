"""
TTP Individual Representation (Chromosome Encoding)

Travel Salesman Problem (Permutation Encoding) (Davis, 1985):

Ordered sequence of cities to visit
Fixed start at city 0 (origin)
Example: [0, 23, 45, 12, ..., 0]
   
KnapSack Problem (Integer Encoding) Multi-Robot not Bynary because we need to track which robot picks which item):

Which robot picks which items
Binary: item_assignment[i] = robot_id ∈ {0, 1, ..., K-1} or -1 (not picked)
Respects capacity
Example: [-1, 0, 1, 0, -1, 2, ...] (item[0]->not picker, item[1]->robot 0, item[2]->robot 1, ...)

Individual = {
    'tours': [[0, 23, 45, 12, ..., 0], tour_robot_1, ..., tour_robot_K],
    'item_assignment': [assignment for each item] == [-1, 0, 1, 0, -1, 2, ...]
}

[1] Davis, L. (1985).
    "Applying adaptive algorithms to epistatic domains."
    IJCAI, 162-164.
    Permutation encoding for TSP

[2] Michalewicz, Z., & Schoenauer, M. (1996).
    "Evolutionary algorithms for constrained parameter optimization problems."
    Evolutionary Computation, 4(1), 1-32.
    Repair mechanisms for constraint handling

[3] Bonyadi et al. (2013). TTP evaluation function.

[4] Gerkey, B. P., & Matarić, M. J. (2004).
    Multi-robot task allocation taxonomy.
"""

import numpy as np

class Individual:
    """    
    Encodes:
    - K tours (one per robot)
    - Item assignment to robots
    - Multi-objective fitness values
    
    Constraints:
    - Each robot starts/ends at origin (city 0)
    - SR -> Each item assigned to at most one robot
    - Each robot respects capacity constraint
    """
    
    def __init__(self, problem, tours=None, item_assignment=None):
        """        
        Args:
            problem: TTProblem instance
            tours: List of K tours (one per robot)
            item_assignment: Array mapping items to robots
        """
        self.problem = problem
        self.num_robots = problem.num_robots
        
        # Initialize tours
        if tours is None:
            # Partition cities N-1 evenly among robots; each robot starts at origin 0
            # All cities are covered, no duplicates, and each robot visit the same number of cities
            cities = list(range(1, problem.num_cities))
            chunks = np.array_split(cities, self.num_robots)
            self.tours = [[0] + list(chunk) for chunk in chunks]
        else:
            self.tours = tours
        
        # Initialize item assignment
        if item_assignment is None:
            # No items
            self.item_assignment = np.full(problem.num_items, -1, dtype=int) # No item is assigned
        else:
            self.item_assignment = item_assignment
        
        # Multi-objective fitness (3 objectives)
        self.fitness = None  # [f1, f2, f3]
        self.scalar_fitness = np.inf
    
    @classmethod #classmethod is to allow calling Individual.random() without needing an instance
    def random(cls, problem, seed=None):
        """
        Create random individual.
        
        Generate K random tours (permutations)
        Randomly assign items to robots
        Repair capacity
        """
        
        rng = np.random.RandomState(seed)
        
        # Randomly partition cities N-1 and assign one subset per robot
        shuffled = list(rng.permutation(range(1, problem.num_cities)))
        chunks = np.array_split(shuffled, problem.num_robots)
        tours = [[0] + list(chunk) for chunk in chunks]
        
        # Build city -> robot mapping from the partitioned tours
        city_to_robot = {}
        for robot_id, tour in enumerate(tours):
            for city in tour:
                city_to_robot[city] = robot_id

        item_assignment = np.full(problem.num_items, -1, dtype=int)
        for i in range(problem.num_items):
            city = int(problem.items[i, 0])
            robot_id = city_to_robot.get(city, -1)
            if robot_id >= 0 and rng.random() < 0.8:  # 80% chance of picking item
                item_assignment[i] = robot_id
        
        ind = cls(problem, tours, item_assignment)
        ind.repair()  # Repair capacity
        return ind
    
    @classmethod
    def greedy(cls, problem, seed=None):
        """
        Create greedy individual using nearest-neighbor heuristic.
        
        Build tour via nearest-neighbor TSP heuristic
        Assign items greedily by profit/weight ratio
        Distribute items across robots to balance load
        """
        
        rng = np.random.RandomState(seed)
        
        unvisited = set(range(1, problem.num_cities))
        nn_order = []
        current = 0
        while unvisited:
            distances = [(c, problem.get_distance(current, c)) for c in unvisited]
            distances.sort(key=lambda x: (x[1], rng.random()))
            nearest = distances[0][0]
            nn_order.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        chunks = np.array_split(nn_order, problem.num_robots)
        tours = [[0] + list(chunk) for chunk in chunks]
        
        # Greedy item assignment by profit/weight ratio
        ratios = problem.items[:, 1] / np.maximum(problem.items[:, 2], 1e-10)
        
        # Add small noise for variation, don't want all individuals to have the same item assignment
        if seed is not None:
            ratios = ratios + rng.normal(0, 0.01, len(ratios))
        
        sorted_items = np.argsort(-ratios)  # Sort descending
        
        # Round-robin assignment to balance load
        item_assignment = np.full(problem.num_items, -1, dtype=int)
        robot_weights = np.zeros(problem.num_robots)
        
        city_to_robot = {}
        for robot_id, tour in enumerate(tours):
            for city in tour:
                city_to_robot[city] = robot_id
        
        for idx in sorted_items:
            city = int(problem.items[idx, 0])
            robot_id = city_to_robot.get(city, -1)   # Only the robot that visits this city
            if robot_id >= 0:
                weight = problem.items[idx, 2]
                if robot_weights[robot_id] + weight <= problem.knapsack_capacity:
                    item_assignment[idx] = robot_id
                    robot_weights[robot_id] += weight
        
        return cls(problem, tours, item_assignment)
    
    def repair(self):
        """
        Repair constraint violations

        - If an item is assigned to a robot that does not visit the item's city, unassign it (set to -1).

        - For each robot that exceeds its weight capacity:
        - Calculate removal score = priority / (profit/weight)
        - Remove items with the highest score until feasible.
        """

        # Build city sets for each robot for quick lookup
        robot_city_sets = [set(tour) for tour in self.tours]

        for item_idx in range(self.problem.num_items):
            robot_id = self.item_assignment[item_idx]
            if robot_id == -1:
                continue  # Already unassigned, skip

            item_city = int(self.problem.items[item_idx, 0])

            # If the assigned robot does not visit this item's city, unassign
            if item_city not in robot_city_sets[robot_id]:
                self.item_assignment[item_idx] = -1

        for robot_id in range(self.num_robots):
            # Get items assigned to this robot (after feasibility pass)
            robot_items = np.where(self.item_assignment == robot_id)[0]

            if len(robot_items) == 0:
                continue

            # Check capacity
            total_weight = np.sum(self.problem.items[robot_items, 2])

            if total_weight <= self.problem.knapsack_capacity:
                continue  # Feasible

            # Calculate removal scores (higher = remove first)
            priorities = self.problem.items[robot_items, 3]
            # Avoid division by zero by adding a small constant to weight
            # ratios = profit/weight, higher means more profitable per unit weight
            ratios = (self.problem.items[robot_items, 1] /np.maximum(self.problem.items[robot_items, 2], 1e-10))

            removal_scores = priorities / (ratios + 1e-10)
            sorted_indices = robot_items[np.argsort(-removal_scores)]

            # Remove items until feasible
            for idx in sorted_indices:
                self.item_assignment[idx] = -1
                total_weight -= self.problem.items[idx, 2]
                if total_weight <= self.problem.knapsack_capacity:
                    break
    
    def evaluate_fitness(self):
        """
        5-Objective fitness for custom (multi-robot, multi-objective) mode.

        - f1:
            - Task Completion Time (TCT) = Makespan / items_delivered [minimize]
            - Penalises routes that deliver fewer items per unit cost.

        - f2:
            - Makespan = renting_ratio x Σ_k travel_time_k [minimize]
            - Sum of all robots travel costs:
                - travel costs = time_k x renting_ratio
                - time_k = Sum of edges [distance / speed(weight)]
                - Speed decreases with weight
            - Total monetary cost of operating all robots.

        - f3:
            - -G_TTP = Σ profit  -  renting_ratio x Σ travel_time [maximize G ->  minimize -G]
            - TTP standard objective

        - f4  
            - -Weighted Priority Profit = -Σ_{collected items} profit x priority_weight [minimize]
            - Priority weights:
                - Priority 1 (High Urgency): weight = 2.0
                - Priority 2 (Medium Urgency): weight = 1.0
                - Priority 3 (Low Urgency): weight = 0.5
            - Drives the packing plan to prefer clinically urgent medications.

        - f5  
            - Load Imbalance = alpha(Σ_weight per robot) [minimize]
            - Standard deviation of total weight carried by each robot.
            - Promotes balanced workload across the fleet.

        References
        ----------
        [1] Bonyadi et al. (2013) – original TTP evaluation function.
        [2] Deb et al. (2002)    – NSGA-II dominance and diversity.
        [3] Zhang & Li (2007)    – MOEA/D Tchebycheff scalarisation.
        """

        #  Per-robot accumulators
        robot_travel_times = np.zeros(self.num_robots)
        robot_priority_profit = np.zeros(self.num_robots)
        robot_weight_totals = np.zeros(self.num_robots)
        total_raw_profit = 0.0

        for robot_id in range(self.num_robots):
            tour = self.tours[robot_id]
            robot_items_idx = np.where(self.item_assignment == robot_id)[0]
            robot_items_set = set(robot_items_idx)

            priority_weights_map = {1: 2.0, 2: 1.0, 3: 0.5}
            for item_idx in robot_items_idx:
                profit = self.problem.items[item_idx, 1]
                weight = self.problem.items[item_idx, 2]
                priority = int(self.problem.items[item_idx, 3])
                pw = priority_weights_map.get(priority, 1.0)
                
                robot_priority_profit[robot_id] += profit * pw
                robot_weight_totals[robot_id] += weight
                total_raw_profit += profit

            travel_time = 0.0
            current_weight = 0.0

            for i in range(len(tour)):
                current_city = tour[i]

                # Pick up items that belong to this robot at this city
                for item_idx in self.problem.get_items_at_city(current_city):
                    if item_idx in robot_items_set:
                        current_weight += self.problem.items[item_idx, 2]

                next_city = tour[(i + 1) % len(tour)]
                distance = self.problem.get_distance(current_city, next_city)
                weight_ratio = current_weight / self.problem.knapsack_capacity
                speed = self.problem.calculate_speed(weight_ratio)
                travel_time += distance / speed

            robot_travel_times[robot_id] = travel_time

        total_items_delivered = int(np.sum(self.item_assignment >= 0))
        total_travel_time = float(np.sum(robot_travel_times))

        # f2 – Makespan
        makespan = total_travel_time * self.problem.renting_ratio

        # f1 – Task Completion Time
        tct = makespan / max(1, total_items_delivered)

        # f3 – TTP standard objective
        neg_G = -(total_raw_profit - self.problem.renting_ratio * total_travel_time)

        # f4 – Weighted Priority Profit
        neg_priority_profit = -float(np.sum(robot_priority_profit))

        # f5 – Load Imbalance
        load_imbalance = float(np.std(robot_weight_totals))

        return tct, makespan, neg_G, neg_priority_profit, load_imbalance
    
    def evaluate_standard_ttp(self):
        """
        Evaluates using the standard TTP formula (single-robot, single-objective).
        G = total_profit - renting_ratio x total_travel_time
        Bonyadi et al. (2013) - original TTP formula.
        """
        
        total_profit = 0.0
        total_travel_time = 0.0
        
        for robot_id in range(self.num_robots):
            tour = self.tours[robot_id]
            robot_items = set(np.where(self.item_assignment == robot_id)[0])
            
            current_weight = 0.0
            
            for i in range(len(tour)):
                current_city = tour[i]
                
                # Recolhe itens
                for item_idx in self.problem.get_items_at_city(current_city):
                    if item_idx in robot_items:
                        current_weight += self.problem.items[item_idx, 2]
                        total_profit += self.problem.items[item_idx, 1]
                
                next_city = tour[(i + 1) % len(tour)]
                distance = self.problem.get_distance(current_city, next_city)
                weight_ratio = current_weight / self.problem.knapsack_capacity
                speed = self.problem.calculate_speed(weight_ratio)
                total_travel_time += distance / speed
        
        G = total_profit - self.problem.renting_ratio * total_travel_time
        return G
    
    def evaluate(self):
        if self.problem.mode == 'benchmark':
            G = self.evaluate_standard_ttp()
            self.fitness = - G # Negative because GA minimizes fitness        
            self.scalar_fitness = self.fitness
        else:
            self.fitness = self.evaluate_fitness()
        return self.fitness
    
    def copy(self):
        # Deep copy of individual.
        new_tours = [tour.copy() for tour in self.tours]
        new_assignment = self.item_assignment.copy()
        new_ind = Individual(self.problem, new_tours, new_assignment)
        new_ind.fitness = self.fitness
        new_ind.scalar_fitness = self.scalar_fitness
        return new_ind
    
    def to_dict(self):
        return {
            'tours': [tour.copy() for tour in self.tours],
            'item_assignment': self.item_assignment.tolist(),
            'fitness': self.fitness,
        }
    
    @classmethod
    def from_dict(cls, problem, data):
        ind = cls(problem, data['tours'], np.array(data['item_assignment']))
        ind.fitness = data['fitness']
        return ind