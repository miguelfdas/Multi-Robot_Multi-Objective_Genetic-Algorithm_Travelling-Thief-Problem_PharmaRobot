"""
Multi-Robot Multi-Objective Travelling Thief Problem (TTP)

Travelling Thief Problem:
Single agent travels through cities, collecting items
Speed decreases with carried weight

Multi-Robot
K agents operate in parallel
Each agent has its own tour and knapsack
Items can only be picked by one agent

Multi-Objective:
Task Completion Time (minimize)
Makespan (minimize)
Weighted Profit Vs Priorities (minimize)
This creates a Pareto front

[1] Bonyadi, M. R., Michalewicz, Z., & Barone, L. (2013).
    "The travelling thief problem: The first step in the transition from
    theoretical problems to realistic problems."
    IEEE Congress on Evolutionary Computation (CEC), 1037-1044.
    Original TTP formulation

[2] Polyakovskiy, S., Bonyadi, M. R., Wagner, M., Michalewicz, Z., & Neumann, F. (2014).
    "A comprehensive benchmark set for the travelling thief problem."
    GECCO, 477-484.
    Standard benchmark instances
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

class TTProblem:
    """
    Multi-Robot Travelling Thief Problem instance.
    
    K agents must visit cities
    Each city has items with profit, weight, and priority
    Each agent has limited capacity (knapsack constraint)
    Speed decreases with weight
    SR (Single-Robot) Each items is collected by only one agent
    Time-extended: Tasks have temporal ordering (tour sequence)
    """
    
    def __init__(self, num_cities, num_items, num_robots, knapsack_capacity, min_speed, max_speed, renting_ratio, mode, seed = None, ):

        self.num_cities = num_cities # Number of cities (nodes in graph)
        self.num_items = num_items # Total number of items
        self.num_robots = num_robots # Multi-Robot number of agents
        
        self.knapsack_capacity = knapsack_capacity # Capacity per agent
        self.min_speed = min_speed # Minimum speed (fully loaded)
        self.max_speed = max_speed # Maximum speed (empty)
        self.renting_ratio = renting_ratio # Cost per time unit
        self.speed_range = max_speed - min_speed # Delta v
        
        self.seed = seed
        
        self.cities = None # City coordinates (num_cities × 2)
        self.distance_matrix = None # Distance matrix (num_cities × num_cities)
        self.items = None # Item properties (num_items × 4): [city, profit, weight, priority]
        self.items_by_city = None # Item Indexing: city -> list of item indexs
        self.indices = None # Set of all item indices (0 to num_items-1)
                
        self.mode = mode
            
    def load_data(self, cities, items):
        """
        Computes distance matrix (Euclidean metric)
        Indexes items
        
        Args:
            cities: Array of shape (num_cities, 2) with (x, y) coordinates
            items: Array of shape (num_items, 4) with columns:
                   [city_index, profit, weight, priority]
        """
        self.cities = cities
        self.items = items
        
        # Compute pairwise Euclidean distances
        # Euclidean distance: d(i,j) = ||city_i - city_j||_2
        # Shape: (num_cities, num_cities)
        self.distance_matrix = squareform(pdist(cities, metric='euclidean'))
        
        # Build inverse index: city_id -> list of item indices (items to that city)        
        self.items_by_city = [[] for _ in range(self.num_cities)]
        for idx, (city, profit, weight, priority) in enumerate(items):
            self.items_by_city[int(city)].append(idx)
        
        self.indices = set(range(self.num_items))
    
    def get_distance(self, city_i, city_j):
        # Get Euclidean distance between two cities.
        return self.distance_matrix[city_i, city_j]
    
    def get_items_at_city(self, city_index):
        """
        Get list of item indices at a given city.
        Robot arrives at city -> checks available items -> decides what to pick
        """
        return self.items_by_city[city_index]
    
    def calculate_speed(self, weight_ratio):
        """
        Calculate robot speed based on current load.
        
        Speed Model:
        v(w) = v_max - weight_ratio * (v_max - v_min)
        weight_ratio: w / W_cap
        
        w = current carried weight
        W_cap = knapsack capacity
        """
        
        return self.max_speed - weight_ratio * self.speed_range

def load_ttp_benchmark(filepath, mode, seed):
    """
    Load a .ttp file (Polyakovskiy et al., 2014).
    
    Adaptations:
    - Keeps city 0 as the origin (converts from 1-based to 0-based)
    """
    
    params = {}
    cities = []
    items = []
    
    mode_ttp = None  # 'coords' or 'items'
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('DIMENSION'):
                params['num_cities'] = int(line.split(':')[1].strip())
            elif line.startswith('NUMBER OF ITEMS'):
                params['num_items'] = int(line.split(':')[1].strip())
            elif line.startswith('CAPACITY OF KNAPSACK'):
                params['knapsack_capacity'] = float(line.split(':')[1].strip())
            elif line.startswith('MIN SPEED'):
                params['min_speed'] = float(line.split(':')[1].strip())
            elif line.startswith('MAX SPEED'):
                params['max_speed'] = float(line.split(':')[1].strip())
            elif line.startswith('RENTING RATIO'):
                params['renting_ratio'] = float(line.split(':')[1].strip())
            
            elif line.startswith('NODE_COORD_SECTION'):
                mode_ttp = 'coords'
            elif line.startswith('ITEMS SECTION'):
                mode_ttp = 'items'
            
            elif mode_ttp == 'coords':
                parts = line.split()
                if len(parts) == 3:
                    # Convert from 1-based to 0-based
                    cities.append([float(parts[1]), float(parts[2])])
            
            elif mode_ttp == 'items':
                parts = line.split()
                if len(parts) == 4:
                    # [index, profit, weight, city_index]
                    profit = float(parts[1])
                    weight = float(parts[2])
                    city = int(parts[3]) - 1  # 1-based -> 0-based

                    items.append([city, profit, weight, 1]) # Aloways priority = 1
    
    cities_array = np.array(cities)
    items_array = np.array(items, dtype=float)
    
    # Remove any items assigned to the depot (city 0)
    depot_mask = items_array[:, 0] != 0
    if not np.all(depot_mask):
        n_removed = np.sum(~depot_mask)
        items_array = items_array[depot_mask]
        
    params['num_items'] = len(items_array)
    
    # Standard TTP is single-robot
    # Mark as benchmark for standart evaluation (not Multi-Objective)
    problem = TTProblem(num_cities=params['num_cities'], num_items=params['num_items'], num_robots=1, knapsack_capacity=params['knapsack_capacity'], min_speed=params['min_speed'], max_speed=params['max_speed'], renting_ratio=params['renting_ratio'], mode=mode, seed=seed)
    
    problem.load_data(cities_array, items_array)
    problem.distance_matrix = np.ceil(problem.distance_matrix).astype(float) 
    
    return problem

def create_ttp_instance(num_robots, mode, seed):
    """    
    Cities arranged in a circle
    Items randomly distributed across cities
    Three priority levels (1=high, 2=medium, 3=low)
    5 items per city
    Random profits/weights
        
    - Capacity: 100 units per robot
    - Speed range: [0.01, 1.0] (slow when loaded)
    - Renting ratio: 5.0 (high time penalty)

    - Item distribution across cities
    - Knapsack capacity relative to total weight
    - Speed variation range
    """
    
    num_cities = 100
    items_per_city = 5
    num_items = (num_cities - 1) * items_per_city  # City 0 has no items (origin), 495 items total
    
    rng = np.random.RandomState(seed)
    
    # Generate cities in circular arrangement
    # x = r·cos(θ), y = r·sin(θ)
    angles = np.linspace(0, 2*np.pi, num_cities, endpoint=False)
    radius = 10.0
    cities = np.column_stack([radius * np.cos(angles),radius * np.sin(angles)])
    
    items = []
    for city in range(1, num_cities):
        for _ in range(items_per_city):
            profit = rng.randint(10, 50)
            weight = rng.randint(5, 15)
            # Distribution: 45% critical, 50% important, 5% routine
            priority = rng.choice([1, 2, 3], p=[0.45, 0.5, 0.05])            
            items.append([city, profit, weight, priority])
    items = np.array(items, dtype=float)
    
    problem = TTProblem(num_cities=num_cities, num_items=num_items, num_robots=num_robots, knapsack_capacity=100.0, min_speed=0.01, max_speed=1.0, renting_ratio=5.0, mode=mode, seed=seed)
    
    problem.load_data(cities, items)
    
    return problem