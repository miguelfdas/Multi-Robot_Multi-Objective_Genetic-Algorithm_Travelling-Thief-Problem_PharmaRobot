"""
Simulation of Hospital Delivery Problem as a Multi-Robot Multi-Objective Dynamic Travelling Thief Problem (TTP)

Motivation: To evaluate the model’s performance in a realistic and dynamic environment

Hospital: Pharmacy (city 0) + 4 A&E rooms (cities 1-4):
    1 - HUC-URGÊNCIA
    2 - Urgência Geral - Polo HUC
    3 - HUC-URGÊNCIA PSIQUIATRIA
    4 - HUC-URG MEDICA VIA

Medication catalog built from real hospital data (final.csv):
    - Frequency follows n_pedidos per medication
    - Destination follows historical descr_serv per medication
    - Priority follows cluster_medicamento: 0/3 -> 1 (Urgent), 2/4/5 -> 2 (Medium), 1 -> 3 (Routine)
    - Deadline = minim delivery time from distribuicao_tempos_medicamento.csv matched by nome_med

Priority levels:
    P1: Urgent (priority = 1)
    P2: Standard (priority = 2)
    P3: Low (priority = 3)

Each batch:
    - batch_id
    - items - (n, 7): [order_number, destination, priority, deadline, arrival_time]
    - arrival_time:
        - time since simulation start 
        - first 20% items at t=0
        - rest follow Poisson arrivals with mean_interarrival_min (5 min)
        - sorted by arrival_time to simulate real-time arrivals
        - to simulate the arrival of new items during the delivery process, allowing testing of dynamic decision-making in the TTP context
        - 120 items in 7h30
"""

import os
import numpy as np
import pandas as pd
import csv
import json
from imoga import GeneticAlgorithm

# Constants for mapping real hospital data to TTP problem format
ROOM_MAP = {
    'HUC-URGÊNCIA': 1,
    'Urgência Geral - Polo HUC': 2,
    'HUC-URGÊNCIA PSIQUIATRIA': 3,
    'HUC-URG MEDICA VIA': 4,
}
 
CLUSTER_TO_PRIORITY = {0: 1, 3: 1, 2: 2, 4: 2, 5: 2, 1: 3}
 
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_exploration')

def load_catalog():
    """
    Build medication catalog from real hospital data
 
    Returns:
        catalog (DataFrame): one row per unique medication code with columns: medicamento, nome_med, n_pedidos, ent_req, priority, deadline
        per_med_data (dict): medicamento - {'cities': list}
        Historical city indices for sampling
    """
    
    final = pd.read_csv(os.path.join(DATA_DIR, 'final.csv'))
 
    # Map room names to city indices
    final['city_idx'] = final['descr_serv'].map(ROOM_MAP).fillna(1).astype(int)
 
    # Map cluster to priority
    final['priority'] = final['cluster_medicamento'].map(CLUSTER_TO_PRIORITY).fillna(2).astype(int)
 
    # Deadline: minimum delivery time (hours) matched by nome_med
    PRIORITY_TO_DEADLINE_MIN = {1: 30.0, 2: 60.0, 3: 120.0}
    final['deadline'] = final['priority'].map(PRIORITY_TO_DEADLINE_MIN)
 
    # Aggregate catalog: one row per medication code
    catalog = (final.groupby('medicamento').agg(
            nome_med=('nome_med', 'first'),
            n_pedidos=('num_pedido', 'count'),
            ent_req=('ent_req', lambda x: x.mode()[0]),
            priority=('priority', lambda x: x.mode()[0]),
            deadline=('deadline', 'first'),
        ).reset_index()
    )
 
    # Historical quantities and cities per medication for sampling
    per_med_data = {}
    for med, grp in final.groupby('medicamento'):
        cities = grp['city_idx'].tolist()
        per_med_data[med] = {'cities': cities or [1]}
 
    return catalog, per_med_data

CATALOG, PER_MED = load_catalog()

def generate_batches(items_per_batch = 120, seed = 42, mean_interarrival_min=5.0):
    """
    Returns a batch of orders

    Batch:
        - batch_id
        - items - (n, 5): [order_number, city_idx, priority, deadline, arrival_time]  
    """
    
    rng = np.random.default_rng(seed)
    
    catalog = CATALOG
    # Sample medications with probabilities proportional to number of times this medication was ordered (n_pedidos)
    weights = catalog['n_pedidos'].values.astype(float)
    # Avoid zero probabilities for medications with no orders
    weights /= weights.sum()
    med_codes = catalog['medicamento'].values
        
    # First 20% of items available at t=0; remaining follow Poisson arrivals
    n_initial = int(items_per_batch * 0.2)
    # To simulate the arrival of new items during the delivery process, allowing testing of dynamic decision-making in the TTP context
    arrival_times = np.zeros(items_per_batch)
    arrival_times[n_initial:] = np.cumsum(rng.exponential(mean_interarrival_min, items_per_batch - n_initial))

    n_p1 = int(items_per_batch * 0.35)
    n_p2 = int(items_per_batch * 0.60)
    n_p3 = items_per_batch - n_p1 - n_p2

    sampled_codes = []

    for priority_level, count in [(1, n_p1), (2, n_p2), (3, n_p3)]:
        if count > 0:
            cat_p = catalog[catalog['priority'] == priority_level]
            weights_p = cat_p['n_pedidos'].values.astype(float)
            if weights_p.sum() > 0:
                weights_p /= weights_p.sum()
            else:
                weights_p = np.ones(len(cat_p)) / len(cat_p)
            med_codes_p = cat_p['medicamento'].values
            codes = rng.choice(med_codes_p, size=count, p=weights_p)
            sampled_codes.extend(codes)

    sampled_codes = np.array(sampled_codes)
    rng.shuffle(sampled_codes)

    order_number, city_idxs, priorities, deadlines = [], [], [], []

    # Get a order from catalog to the batch
    for code in sampled_codes:
        row = catalog.loc[catalog['medicamento'] == code].iloc[0]
        hist = PER_MED[code]

        # Sample city and quantity from historical data
        city_idxs.append(int(rng.choice(hist['cities'])))
        priorities.append(int(row['priority']))
        deadlines.append(float(row['deadline']))
        order_number.append(int(row['n_pedidos']))

    items = np.column_stack([order_number, city_idxs, priorities, deadlines, arrival_times])
    
    items = items[np.argsort(items[:,4])]

    batche = ({
        'batch_id': 0,
        'items': items,
    })
 
    return batche

class BatchSimulator:
    """
    Manages the simulation state of a single batch.
    
    Batch have a size.
    Batch built from an exploratory analysis of real hospital data.
    Sample medications with probabilities proportional to number of times this medication was ordered.
    Destination follows historical data.
    Priority follows historical data cluster.
    Deadline: Minimum delivery time for this medicine based on an exploratory analysis of hospital data.

    Arrival time:
        - First 20% of medicines available at the start of the simulation.
        - The rest follows a Poisson distribution with an average interarrival of 5 minutes.
        
    High-stress case: 120 medicines in 7h30

    Each medicine has:
        - Order number
        - Destination: A&E room
        - Priority level: (P1: Urgent, P2: Standard, P3: Low)
        - Deadline: Minimum delivery time for this medicine based on an exploratory analysis of hospital data.
        - Arrival time: Time elapsed from the start of the simulation until the medicine entered the system (to simulate a dynamic environment)

    items: (N, 5) — full batch sorted by arrival_time (col 4)
    
    delivered: set of indices already delivered

    - An item is available when arrival_time <= simulation_time and it has not yet been delivered
    
    Priority levels:
        P1: Urgent (priority = 1)
        P2: Standard (priority = 2)
        P3: Low (priority = 3)
    """

    def __init__(self, items):
        self.items = items
        self.delivered = set()

    def get_available(self, sim_time):
        # Indices of items arrived by sim_time and not yet delivered
        return [i for i in range(len(self.items)) if i not in self.delivered and self.items[i, 4] <= sim_time]
    
    def time_to_available(self, sim_time):
        next_available = self.items - self.delivered
        return next_available[0, 4] - sim_time

    def mark_delivered(self, indices):
        self.delivered.update(indices)

    def is_empty(self):
        return len(self.delivered) == len(self.items)
    
class RobotState:
    """
    Physical state of a single robot
    
    Robot entities have:
        - Position(x,y): The current point on the map where the robot is .
        - Knapsack capacity: Number of different medicines the robot can carry.
        - Robot speed.
        - Battery capacity: Battery capacity is expressed in terms of operating time.
        - Battery level at the moment
    
    Assumptions:
        - The battery charging time is the same as the battery discharge time.
        - K robots operate in parallel.
        - Each K robot visit N A&E rooms to deliver M medications
        - Each k robot has its own tour and knapsack.
        - Each medicine can only be picked by one robot.
    """
    
    def __init__(self, position, battery, knapsack_capacity, speed, battery_capacity):
        self.position = position
        self.battery = battery
        self.tours = None
        
        self.knapsack_capacity = knapsack_capacity
        self.speed = speed
        self.battery_capacity = battery_capacity
        
        self.cargo = []
        self.current_tour = []

    def deplete(self, cost):
        # Deplete battery by cost, ensuring it doesn't go below zero
        self.battery = max(0.0, self.battery - cost)
        
    def charge(self, amount):
        # Charge battery by amount, ensuring it doesn't exceed capacity
        self.battery = min(self.battery_capacity, self.battery + amount)
        
class GADecisionService:
    """
    Model
    
    Independent service that handles all GA evaluations, entirely decoupled from the simulation.
    It reads the simulation state dictionary
    Executes the model/GA
    Returns the elapsed time, the routs and item alocation for each robot and inputs and outputs logs
    
    Integration:
        Read simulation state (Minimal input):
            - Available medicines from the batch (An item is available when arrival time <= simulation time)
                - With order number, destination and priority
            - Robot states:
                - Position (x,y)
                - Battery level
                - Alocated medicines at the moment
            - Hospital map state 
        Output:
            - Elapsed time: Time of Genetic Algorithm (GA) running + trip
            - Individual (route and medicines allocation decisions)
    """
    def __init__(self, params, seed, log_dir="results_sim"):
        self.params = params
        self.seed = seed
        self.log_file = os.path.join(log_dir, "ga_log.csv")
        self.rng = np.random.RandomState(seed)
        
        os.makedirs(log_dir, exist_ok=True)

    def get_decision(self, state_dict, decision_id):
        """        
        Reads the simulation state
        Executes the GA
        Returns decisions
        """
                      
        problem = state_dict['problem']
        
        ga = GeneticAlgorithm(problem, seed=self.seed, **self.params)
          
        elapsed, geno_div, pheno_div, priority_div = ga.evolve()
          
        best = ga.best_individual

        self.log_metrics(state_dict, ga, elapsed, geno_div, pheno_div, priority_div, decision_id)
        
        return best, elapsed

    def log_metrics(self, state_dict, ga, elapsed, geno_div, pheno_div, priority_div, decision_id):        
        sim = state_dict['sim']
        problem = state_dict['problem']
        
        metrics = {
            'decision_id': decision_id,
            'sim_time (before GA and trip)': state_dict['sim_time'],
            'elapsed_time_minutes': float(elapsed) / 60.0,
            'best_fitness': ga.best_individual.scalar_fitness,
            'avg_fitness': float(np.mean([ind.scalar_fitness for ind in ga.population])),
            'std_fitness': float(np.std([ind.scalar_fitness for ind in ga.population])),
            'best_Makespan': float(ga.best_individual.fitness[0]),
            'best_TWT': float(ga.best_individual.fitness[1]),
            'mean_Makespan': float(np.mean([ind.fitness[0] for ind in ga.population])),
            'mean_TWT': float(np.mean([ind.fitness[1] for ind in ga.population])),
            'genotypic_diversity': float(geno_div),
            'phenotypic_diversity': float(pheno_div),
            'priority_diversity': float(priority_div),
            'robots_positions': json.dumps([int(p) for p in problem.position]),
            'robots_batterys': json.dumps([float(b) for b in problem.battery]),
            'robots_knapsack_capacity': problem.knapsack_capacity,
            'robots_speed': problem.speed, 
            'time_matrix': problem.time_matrix,
            'batch_items': json.dumps(sim.items.tolist()),
            'items_indices': json.dumps([int(idx) for idx in sim.get_available(state_dict['sim_time'])]),
            'best_tours': json.dumps([t.copy() for t in ga.best_individual.tours]),
            'item_assignment': json.dumps(ga.best_individual.item_assignment.tolist()),
            'pop_size': ga.pop_size,
            'generations': ga.generations,
        }
                
        file_exists = os.path.isfile(self.log_file) and os.path.getsize(self.log_file) > 0
        
        with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
            # Force fieldnames into a concrete list to prevent CSV writing errors
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)