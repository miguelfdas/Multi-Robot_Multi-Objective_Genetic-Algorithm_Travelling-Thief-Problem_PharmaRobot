"""
PharmaRobot - Hospital Delivery Problem as Multi-Robot Multi-Objective Dynamic Travelling Thief Problem (TTP) - Problem Individual Representation - Chromosome Encoding

Multi-Robot Multi-Objective Dynamic Travelling Thief Problem (TTP):
    - Travel Salesman Problem (Permutation Encoding) (Davis, 1985):
        - Ordered sequence of A&E rooms to visit
        - Ending at pharmacy/depot
        - Example: [23, 45, 12, ..., 0]
  
    - KnapSack Problem (Assignment Encoding) (Michalewicz & Schoenauer, 1996):
        - Integer Encoding
            - Multi-Robot - Not Bynary because we need to track which robot picks which item:
            - Not Bynary: item_assignment[i] = robot_id ∈ {0, 1, ..., K-1} or -1 (not picked)
            - Respects capacity
            - Example: [-1, 0, 1, 0, -1, 2, ...] (item[0]->not pick, item[1]->robot 0, item[2]->robot 1, ...)

Individual = {
    'tours': [[0, 23, 45, 12, ..., 0], tour_robot_1, ..., tour_robot_K],
    'item_assignment': [assignment for each medicine] == [-1, 0, 1, 0, -1, 2, ...]
}

Each individual encodes a complete fleet plan:
    tours[k] - ordered A&E room sequence for robot k
    item_assignment - which robot carries each medication (-1 = unassigned)

Delivery semantics: robots load at pharmacy, drop medications en route

Population initialization:
    - Initialize population with Hybrid Strategy (Faulkner et al., 2015):

    - 50% random: 
        - Random with Repair
        - Generate K stochastic tours one per robot
        - Stochastic medication assignment to robots - 50% chance to assign each medication (promotes diversity)

    - 50% greedy: 
        - Greedy with Repair
        - Build tour via Nearest-Neighbors Travelling Salesman Problem heuristic, end at pharmacy (0)
        - Priority Packing with Repair:
            - Assign items greedily by priority and distribute items across robots to balance load

Repair:
    Greedy Repair constraint violations
    
    Knapsack constraints: 
        - For each overloaded robot remove the least priority medications until capacity is respected.

    Route Synchronization: 
        - Ensure all required destinations are visited and remove useless cities from the tour
        - Guarantee the tour concludes at the pharmacy

    Battery constraint: 
        - For each robot, calculate the total distance of its tour. 
        - If it exceeds battery capacity, truncate the tour to fit within the battery limit and unassign any items whose destination is cut from the truncated tour.


Multi-objective fitness evaluation:
    - Makespan (minimize): Total time required to complete a trip
    - Total Weighted Tardiness (TWT) (minimize): 
        - Penalises late deliveries by priority and lateness, zero for on-time deliveries
        - TWT = sum{(4 - priority) * max(0, TCT - deadline)}
        - Task Completion Time (TCT) (minimize): Time from medication entry in the batch until it is delivered (delivery sim time - arrival time)
    
    - This creates a Pareto front

References:
    OX permutation encoding for TSP
    Davis, L. (1985). "Applying adaptive algorithms to epistatic domains." IJCAI, 162-164.

    Repair-based mechanisms for constraint handling
    Michalewicz, Z., & Schoenauer, M. (1996). "Evolutionary algorithms for constrained parameter optimization problems." Evolutionary Computation, 4(1), 1-32.

    TTP evaluation function and speed model
    Bonyadi et al. (2013).

    Multi-robot task allocation taxonomy
    Gerkey, B. P., & Matarić, M. J. (2004).
"""


import numpy as np

class Individual:
    """
    Encodes a single solution:
    - K tours (one per robot)
    - Item assignment to robots
    - Multi-objective fitness values
    
    Constraints:
    - Each robot ends at pharmacy (city 0)
    - SR -> Each item assigned to at most one robot
    - Each robot respects capacity constraint
    """

    def __init__(self, problem, tours=None, item_assignment=None, charge_aggressiveness = None):
        """        
        Args:
            problem: TTProblem/HospitalProblem instance
            tours: List of K tours (one per robot)
            item_assignment: Array mapping medicines to robots
        """        
        
        self.problem = problem
            
        self.tours = tours
        self.item_assignment = item_assignment
        
        self.charge_aggressiveness = charge_aggressiveness
        
        
        # Multi-objective fitness
        self.fitness = None # 2-objective [f0, f1]
        self.scalar_fitness = np.inf # for selection purposes (e.g. NSGA-II crowding distance or MOEA/D scalarisation)

    @classmethod #classmethod is to allow calling Individual.random() without needing an instance
    def random(cls, problem, seed=None):
        """
        Create a random individual
        Generate K tours one per robot
        Stochastic medication assignment to robots
        Repair capacity violations, tour synchronisation, and battery constraints after random generation
        - 50% chance to assign each medication (promotes diversity)
        """
        
        rng = np.random.RandomState(seed)

        tours = []
        for k in range(problem.num_robots):
            start = problem.position[k]
            cities = [c for c in range(problem.num_cities) if c != start and c != 0]
            perm = rng.permutation(cities).tolist()
            tours.append([start] + perm + [0])

        # Assign items to robots with some randomness
        item_assignment = np.full(problem.num_items, -1, dtype=int)
        for i in range(problem.num_items):
            if rng.random() < 0.5: # 50% chance of picking item
                robot_id = rng.randint(0, problem.num_robots)
                item_assignment[i] = robot_id

        charge_aggressiveness = np.random.uniform(0.0, 1.0, size=problem.num_robots)

        ind = cls(problem, tours, item_assignment, charge_aggressiveness)
        ind.repair() # Repair
        return ind

    @classmethod
    def greedy(cls, problem, seed=None):
        """
        Create greedy individual using nearest-neighbor heuristic
        Build tour via nearest-neighbor TSP heuristic
        Tour: always move to the closest unvisited room and end at pharmacy (0)
        Assign items greedly by priority
        Distribute items across robots to balance load
        """
        
        rng = np.random.RandomState(seed)

        tours = []
        for k in range(problem.num_robots):
            start = problem.position[k]
            unvisited = set(range(problem.num_cities)) - {start} - {0}
            nn_order = []
            current = start
            while unvisited:
                dists = sorted([(r, problem.time_matrix[current, r]) for r in unvisited], key=lambda x: (x[1], rng.random()))
                nearest = dists[0][0]
                nn_order.append(nearest)
                unvisited.discard(nearest)
                current = nearest
            tours.append([start] + nn_order + [0])


        # [problem.items[0, 2] = 1, problem.items[1, 2] = 2, problem.items[2, 2] = 3, problem.items[3, 2] = 1, problem.items[4, 2] = 2, problem.items[5, 2] = 3]
        # [problem.items[0, 2] = 1, problem.items[3, 2] = 1, problem.items[1, 2] = 2, problem.items[4, 2] = 2, problem.items[5, 2] = 3, problem.items[2, 2] = 3]
        priority = problem.items[:, 2]     
        sorted_meds = np.argsort(-priority) # highest priority first

        # Round-robin assignment to balance load
        item_assignment = np.full(problem.num_items, -1, dtype=int)
        robot_counts = np.zeros(problem.num_robots)

        for med_idx in sorted_meds:
            robot_id = int(np.argmin(robot_counts))
            if robot_counts[robot_id] < problem.knapsack_capacity:
                item_assignment[med_idx] = robot_id
                robot_counts[robot_id] += 1

        charge_aggressiveness = np.random.uniform(0.0, 1.0, size=problem.num_robots)

        ind = cls(problem, tours, item_assignment, charge_aggressiveness)
        ind.repair()

        return ind
    
    def repair_old(self):
        """
        Greedy Repair constraint violations
        
        Knapsack constraints: 
            - For each overloaded robot remove the least priority medications until capacity is respected.
            - Knapsack physical capacity is not exceeded (limit = 6 orders/sleeves, rather than the sum of boxes).

        Route Synchronization: 
            - Ensure all required destinations are visited and remove useless cities from the tour
            - Guarantee the tour concludes at the pharmacy
            - Route synchronisation includes all mandatory destinations and starts/ends at the correct depot.
            
        Flexible Multi-Stage Route Construction 
            - Allowing GA to interrupt and mix deliveries

        Battery constraint: 
            - For each robot, calculate the total distance of its tour. If it exceeds battery capacity, 
            truncate the tour to fit within the battery limit and unassign any items whose destination is cut from the truncated tour.        
        """
        
        for k in range(self.problem.num_robots):
            assigned_items = np.where(self.item_assignment == k)[0]            
            if len(assigned_items) > self.problem.knapsack_capacity:
                items_info = [(idx, self.problem.items[idx, 2]) for idx in assigned_items]
                items_info.sort(key=lambda x: x[1], reverse=True)
                num_to_remove = len(assigned_items) - self.problem.knapsack_capacity
                for i in range(num_to_remove):
                    item_idx = items_info[i][0]
                    self.item_assignment[item_idx] = -1

            current_pos = int(self.problem.position[k])
                
            valid_assigned_items = np.where(self.item_assignment == k)[0]
            required_dests = set()
            for item_idx in valid_assigned_items:
                dest = int(self.problem.items[item_idx, 1])
                if dest != 0:
                    required_dests.add(dest)
                    
            new_tour = [current_pos]
            for node in self.tours[k]:
                node = int(node)
                if node in required_dests and node not in new_tour:
                    new_tour.append(node)
                    required_dests.remove(node)
                    
            for dest in required_dests:
                new_tour.append(dest)
                
            if new_tour[-1] != 0:
                new_tour.append(0)
                
            current_battery = self.problem.battery[k]
            time_matrix = self.problem.time_matrix
            
            truncated_tour = [current_pos]
            accumulated_cost = 0.0
            
            for i in range(1, len(new_tour) - 1):
                next_node = new_tour[i]
                cost_to_next = time_matrix[truncated_tour[-1], next_node]
                cost_to_return = time_matrix[next_node, 0]
                
                if accumulated_cost + cost_to_next + cost_to_return <= current_battery:
                    accumulated_cost += cost_to_next
                    truncated_tour.append(next_node)
                else:
                    # Battery constraint violated, truncate tour here and unassign items whose destination is cut
                    break
            
            if truncated_tour[-1] != 0:
                truncated_tour.append(0)
                
            self.tours[k] = truncated_tour
            
            valid_destinations = set(truncated_tour)
            final_assigned_items = np.where(self.item_assignment == k)[0]
            
            for item_idx in final_assigned_items:
                dest = int(self.problem.items[item_idx, 1])
                if dest not in valid_destinations and dest != 0:
                    self.item_assignment[item_idx] = -1
                            
        self.fitness = None
    
    def repair(self):
        
        """
        Greedy Repair constraint violations
        
        Knapsack constraints: 
            - For each overloaded robot remove the least priority medications until capacity is respected.
            - Knapsack physical capacity is not exceeded (limit = 6 orders/sleeves, rather than the sum of boxes).

        Route Synchronization: 
            - Ensure all required destinations are visited and remove useless cities from the tour
            - Guarantee the tour concludes at the pharmacy
            - Route synchronisation includes all mandatory destinations and starts/ends at the correct depot.
            
        Flexible Multi-Stage Route Construction 
            - Allowing GA to interrupt and mix deliveries

        Battery constraint: 
            - For each robot, calculate the total distance of its tour. If it exceeds battery capacity, 
            truncate the tour to fit within the battery limit and unassign any items whose destination is cut from the truncated tour.        
        """
        
        for k in range(self.problem.num_robots):
            local_cargo = self.problem.robots_cargo[k] if hasattr(self.problem, 'robots_cargo') else []
            
            for item_idx in local_cargo:
                self.item_assignment[item_idx] = k
            
            for other_k in range(self.problem.num_robots):
                if other_k != k:
                    other_cargo = self.problem.robots_cargo[other_k] if hasattr(self.problem, 'robots_cargo') else []
                    for item_idx in other_cargo:
                        if self.item_assignment[item_idx] == k:
                            self.item_assignment[item_idx] = other_k

            # 1. Knapsack constraints
            assigned_items = np.where(self.item_assignment == k)[0]            
            if len(assigned_items) > self.problem.knapsack_capacity:
                new_items = [idx for idx in assigned_items if idx not in local_cargo]
                new_items.sort(key=lambda x: self.problem.items[x, 2], reverse=True) 
                
                num_to_remove = len(assigned_items) - self.problem.knapsack_capacity
                items_removed = 0
                for item_idx in new_items:
                    if items_removed < num_to_remove:
                        self.item_assignment[item_idx] = -1
                        items_removed += 1
                    else:
                        break

            # 3. Flexible Multi-Stage Route Construction
            current_pos = int(self.problem.position[k])
            valid_assigned_items = np.where(self.item_assignment == k)[0]
            
            dests_in_cargo = set()
            dests_need_pickup = set()
            
            for item_idx in valid_assigned_items:
                dest = int(self.problem.items[item_idx, 1])
                if dest != 0:
                    if item_idx in local_cargo:
                        dests_in_cargo.add(dest)
                    else:
                        dests_need_pickup.add(dest)
            
            new_tour = [current_pos]
            has_visited_depot = (current_pos == 0)
            
            # Read the GA's raw DNA and respect its chosen sequence, applying Just-In-Time physical corrections
            for node in self.tours[k]:
                node = int(node)
                
                # Depot visit
                if node == 0:
                    if new_tour[-1] != 0:
                        new_tour.append(0)
                    has_visited_depot = True
                    continue
                
                # If the GA targets a room that fulfills BOTH an existing cargo item and a new pickup
                if node in dests_in_cargo and node in dests_need_pickup:
                    if has_visited_depot:
                        if new_tour[-1] != node: new_tour.append(node)
                        dests_in_cargo.discard(node)
                        dests_need_pickup.discard(node)
                    else:
                        # Can only drop existing cargo. Will have to return later for the new item.
                        if new_tour[-1] != node: new_tour.append(node)
                        dests_in_cargo.discard(node)
                        
                # If the GA targets a room to deliver an item already in the knapsack (can do anytime)
                elif node in dests_in_cargo:
                    if new_tour[-1] != node: new_tour.append(node)
                    dests_in_cargo.discard(node)
                    
                # If the GA targets a room for a NEW item, it MUST have visited the pharmacy first
                elif node in dests_need_pickup:
                    if not has_visited_depot:
                        # Just-In-Time intervention: Force a detour to the pharmacy right now
                        if new_tour[-1] != 0: new_tour.append(0)
                        has_visited_depot = True
                        
                    if new_tour[-1] != node: new_tour.append(node)
                    dests_need_pickup.discard(node)

            # Cleanup Phase: Enforce any missed mandatory destinations
            for dest in list(dests_in_cargo):
                if new_tour[-1] != dest: new_tour.append(dest)
                
            if dests_need_pickup:
                if not has_visited_depot:
                    if new_tour[-1] != 0: new_tour.append(0)
                    has_visited_depot = True
                for dest in list(dests_need_pickup):
                    if new_tour[-1] != dest: new_tour.append(dest)
            
            # Conclude the route back at the central depot
            if new_tour[-1] != 0:
                new_tour.append(0)

            # 4. Battery Constraints Management
            current_battery = self.problem.battery[k]
            time_matrix = self.problem.time_matrix
            gene = self.charge_aggressiveness[k] if hasattr(self, 'charge_aggressiveness') else 0.5
            
            truncated_tour = [current_pos]
            virtual_battery = current_battery
            accumulated_cost = 0.0
            
            for i in range(1, len(new_tour) - 1):
                prev_node = truncated_tour[-1]
                next_node = new_tour[i]
                
                if prev_node == 0:
                    cost_until_next_pharmacy = 0.0
                    
                    temp_idx = i - 1
                    while temp_idx < len(new_tour) - 1:
                        c1 = new_tour[temp_idx]
                        c2 = new_tour[temp_idx + 1]
                        cost_until_next_pharmacy += time_matrix[c1, c2]
                        if c2 == 0:
                            break
                        temp_idx += 1
                        
                    min_battery_required = cost_until_next_pharmacy
                    target_battery = min_battery_required + gene * (self.problem.battery_capacity - min_battery_required)
                    target_battery = min(target_battery, self.problem.battery_capacity)
                    
                    if virtual_battery < target_battery:
                        virtual_battery = target_battery

                cost_to_next = time_matrix[prev_node, next_node]
                cost_to_return = time_matrix[next_node, 0]
                
                if next_node == 0:
                    virtual_battery = self.problem.battery_capacity # Simula o reabastecimento na farmácia
                    truncated_tour.append(next_node)
                elif cost_to_next + cost_to_return <= virtual_battery:
                    virtual_battery -= cost_to_next
                    truncated_tour.append(next_node)
                else:
                    break
            
            if truncated_tour[-1] != 0:
                truncated_tour.append(0)
                
            self.tours[k] = truncated_tour
            
            valid_destinations = set(truncated_tour)
            final_assigned_items = np.where(self.item_assignment == k)[0]
            
            for item_idx in final_assigned_items:
                dest = int(self.problem.items[item_idx, 1])
                if dest not in valid_destinations and dest != 0:
                    if item_idx not in local_cargo:
                        self.item_assignment[item_idx] = -1

        self.fitness = None
    
    def evaluate_fitness(self):
        """
        Evaluates Multi-Objective Fitness: 
            - f0: Makespan (minimize): Total time required to complete a trip
            - f1: Total Weighted Tardiness (TWT) (minimize): 
                - Penalises late deliveries by priority and lateness, zero for on-time deliveries
                - TWT = sum{(4 - priority) * max(0, TCT - deadline)}
                - Task Completion Time (TCT) (minimize): Time from medication entry in the batch until it is delivered (delivery sim time - arrival time)
            
            - This creates a Pareto front
        """
        
        time_matrix = self.problem.time_matrix
        delivery_time = np.full(self.problem.num_items, np.inf)
        robot_makespan = np.zeros(self.problem.num_robots)
        
        virtual_battery = [b for b in self.problem.battery]

        for k, tour in enumerate(self.tours):
            assigned = np.where(self.item_assignment == k)[0]
            local_cargo = self.problem.robots_cargo[k]
            
            city_drop = {}
            for i in assigned:
                dest = int(self.problem.items[i, 1])
                city_drop.setdefault(dest, []).append(i)
 
            current_time = 0.0
            has_visited_depot = False
            
            for pos, city in enumerate(tour):
                if city == 0:
                    has_visited_depot = True
                
                # Evaluate deliveries at the current room location
                for i in city_drop.get(city, []):
                    if delivery_time[i] == np.inf: 
                        # Delivery is valid if the item was already in the knapsack, OR if the depot has been visited
                        if has_visited_depot or (i in local_cargo):
                            delivery_time[i] = current_time
 
                if pos < len(tour) - 1:
                    next_city = tour[pos + 1]
                    
                    if city == 0 and next_city != 0:
                        cost_until_next_pharmacy = 0.0
                        temp_pos = pos
                        while temp_pos < len(tour) - 1:
                            c1 = tour[temp_pos]
                            c2 = tour[temp_pos + 1]
                            cost_until_next_pharmacy += time_matrix[c1, c2]
                            if c2 == 0:
                                break
                            temp_pos += 1
                            
                        min_battery_required = cost_until_next_pharmacy
                        gene = self.charge_aggressiveness[k]
                        
                        target_battery = min_battery_required + gene * (self.problem.battery_capacity - min_battery_required)
                        target_battery = min(target_battery, self.problem.battery_capacity)
                        
                        if virtual_battery[k] < target_battery:
                            charge_time = target_battery - virtual_battery[k]
                            current_time += charge_time
                            virtual_battery[k] = target_battery
                                
                    travel_time = time_matrix[city, next_city]
                    current_time += travel_time
                    virtual_battery[k] -= travel_time
                
            robot_makespan[k] = current_time
 
        makespan = float(np.max(robot_makespan))
        trip_start = self.problem.trip_start_time 
        arrival_times = self.problem.items[:, 4]
        
        if not hasattr(self.problem, 'dynamic_penalty_per_item'):
            max_edge_cost = float(np.max(time_matrix[time_matrix < np.inf]))
            worst_case_run_time = (self.problem.num_cities * max_edge_cost) + self.problem.battery_capacity
            self.problem.dynamic_penalty_per_item = worst_case_run_time + (3.0 * worst_case_run_time)

        delivered_mask = (delivery_time != np.inf)
        
        num_unassigned = np.sum(~delivered_mask)
        unassigned_penalty = num_unassigned * self.problem.dynamic_penalty_per_item
        
        twt_real = 0.0

        if np.any(delivered_mask):
            tct_items_valid = (trip_start + delivery_time[delivered_mask]) - arrival_times[delivered_mask]
            
            priorities = self.problem.items[delivered_mask, 2]
            deadlines = self.problem.items[delivered_mask, 3]
            
            twt_real = float(np.sum((4.0 - priorities) * np.maximum(0.0, tct_items_valid - deadlines)))
                
        makespan += unassigned_penalty
        twt = twt_real + unassigned_penalty
                     
        self.fitness = np.array([makespan, twt], dtype=float)
        return self.fitness
    
    def copy(self):
        new_ind = Individual(self.problem, [t.copy() for t in self.tours], self.item_assignment.copy(), self.charge_aggressiveness.copy())
        new_ind.scalar_fitness = self.scalar_fitness
        if self.fitness is not None:
            new_ind.fitness = self.fitness.copy()
        return new_ind