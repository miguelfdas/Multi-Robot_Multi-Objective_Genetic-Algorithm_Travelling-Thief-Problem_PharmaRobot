"""
PharmaRobot - Hospital Delivery Problem Definition as Multi-Robot Multi-Objective Dynamic Travelling Thief Problem (TTP)

Entities:
    Location 0 - pharmacy (depot)
    Location n: A&E rooms  (1 - HUC-URGÊNCIA, 2 - Urgência Geral - Polo HUC, 3 - HUC-URGÊNCIA PSIQUIATRIA, 4 - HUC-URG MEDICA VIA) - delivery points
    Waypoints: corridor intersections for micro-routing and congestion updates - Layer 3
    Corridors: edges between nodes (pharmacy, A&E rooms and Waypoints) with physical distance and congestion factor
    Medication m - [order_number, destination, priority, deadline, arrival_time]
            
Time/Distance Model:
    
    time_matrix[N,N]:
    
    [time(0,0), ..., time(0,N-1)]
    [time(1,0), ..., time(1,N-1)]
    
    time[i,j] = sum((d * (1 + (b_c + d_c))) / rs)
    time[i,j] = sum(edge_cost) along A* optimal path (Layer 3)
    i and j: 2 A&E rooms/Pharmacy
    edge_cost = effective_dist / rs
    effective_dist = d * (1 + c_e) (higher congestion = slower)
    d = corridor distance (Euclidean distance)
    c_e = congestion
    c_e = b_c + d_c
    base_congestion (b_c):
        - 0.0 - service corridor
        - 0.5 - general corridor
    dynamic_congestion (d_c): random(0.0, 0.5) sampled when robot arrives at corridors intersections (waypoints)
    lower congestion = faster
    rs = robot speed

    A* minimises time[i,j]

Real hospital graph based on the CHUC floor plan.
Scale: 1cm (A4 page) to 480cm (simulation/real) (hospital door wise)
A 5 minutes travel time is added between the pharmacy and the 'grid_0' waypoint to simulate the elevator.

Units:
    - Distances: cm
    - Times: minutes

References:
    Bonyadi, M. R., Michalewicz, Z., & Barone, L. (2013).
        "The travelling thief problem: the first step in the transition from theoretical problems to realistic problems."
        IEEE CEC, 1037–1044.

    Polyakovskiy, S. et al. (2014).
        "A comprehensive benchmark set for the travelling thief problem."
        GECCO, 477–484.

    Hart, P. E., Nilsson, N. J., & Raphael, B. (1968).
        "A formal basis for the heuristic determination of minimum cost paths."
        IEEE Transactions on Systems Science and Cybernetics, 4(2), 100–107.
"""

import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
Labels:
    - HUC-URGÊNCIA - 1
    - Urgência Geral - Polo HUC - 2
    - HUC-URGÊNCIA PSIQUIATRIA - 3
    - HUC-URG MEDICA VIA - 4
"""

HOSPITAL_NODES = [
    # Main nodes 
    # Pharmacy
    {'node_id': 'main_0', 'x': 0, 'y': 0, 'is_main': True, 'label': 0, 'descr_serv': 'Pharmacy'},
    # A&E rooms
    {'node_id': 'main_1', 'x': 12672, 'y': 2016, 'is_main': True, 'label': 4, 'descr_serv': 'HUC-URG MEDICA VIA'},
    {'node_id': 'main_2', 'x': 13056, 'y': -576, 'is_main': True, 'label': 1, 'descr_serv': 'HUC-URGÊNCIA'},
    {'node_id': 'main_3', 'x': 20592, 'y': -2448, 'is_main': True, 'label': 2, 'descr_serv': 'Urgência Geral - Polo HUC'},
    {'node_id': 'main_4', 'x': 16728, 'y': 912, 'is_main': True, 'label': 3, 'descr_serv': 'HUC-URGÊNCIA PSIQUIATRIA'},    

    # Waypoints
    {'node_id': 'grid_0', 'x': 9024, 'y': 0, 'is_main': False},
    {'node_id': 'grid_1', 'x': 9360, 'y': 0, 'is_main': False},
    {'node_id': 'grid_2', 'x': 10704, 'y': 0, 'is_main': False},
    {'node_id': 'grid_3', 'x': 9360, 'y': 912, 'is_main': False},
    {'node_id': 'grid_4', 'x': 10704, 'y': 912, 'is_main': False},
    {'node_id': 'grid_5', 'x': 11568, 'y': 912, 'is_main': False},
    {'node_id': 'grid_6', 'x': 12672, 'y': 912, 'is_main': False},
    {'node_id': 'grid_7', 'x': 13056, 'y': 912, 'is_main': False},
    {'node_id': 'grid_8', 'x': 11568, 'y': -576, 'is_main': False},
    {'node_id': 'grid_9', 'x': 15360, 'y': 912, 'is_main': False},
    {'node_id': 'grid_10', 'x': 11568, 'y': -2448, 'is_main': False},
    {'node_id': 'grid_11', 'x': 13056, 'y': -2448, 'is_main': False},
    {'node_id': 'grid_12', 'x': 15360, 'y': -2448, 'is_main': False},
    {'node_id': 'grid_13', 'x': 15360, 'y': -2064, 'is_main': False},
    {'node_id': 'grid_14', 'x': 15360, 'y': 48, 'is_main': False},
    {'node_id': 'grid_15', 'x': 17568, 'y': 48, 'is_main': False},
    {'node_id': 'grid_16', 'x': 17568, 'y': -2064, 'is_main': False},
    {'node_id': 'grid_17', 'x': 17568, 'y': -816, 'is_main': False},
]

HOSPITAL_EDGES = [
    # {'from': node_id, 'to': node_id, 'base_congestion': 0.0 | 0.5}
    # All edges are bidirectional.
    {'from': 'main_0', 'to': 'grid_0', 'base_congestion': 0.0},
    {'from': 'grid_0', 'to': 'grid_1', 'base_congestion': 0.5},
    {'from': 'grid_1', 'to': 'grid_2', 'base_congestion': 0.5},
    {'from': 'grid_1', 'to': 'grid_3', 'base_congestion': 0.5},
    {'from': 'grid_2', 'to': 'grid_4', 'base_congestion': 0.5},
    {'from': 'grid_3', 'to': 'grid_4', 'base_congestion': 0.5},
    {'from': 'grid_4', 'to': 'grid_5', 'base_congestion': 0.5},
    {'from': 'grid_5', 'to': 'grid_6', 'base_congestion': 0.5},
    {'from': 'grid_5', 'to': 'grid_8', 'base_congestion': 0.5},
    {'from': 'grid_6', 'to': 'main_1', 'base_congestion': 0.0},
    {'from': 'grid_6', 'to': 'grid_7', 'base_congestion': 0.5},
    {'from': 'grid_7', 'to': 'main_2', 'base_congestion': 0.0},
    {'from': 'grid_8', 'to': 'main_2', 'base_congestion': 0.0},
    {'from': 'grid_8', 'to': 'grid_10', 'base_congestion': 0.5},
    {'from': 'grid_10', 'to': 'grid_11', 'base_congestion': 0.5},
    {'from': 'grid_11', 'to': 'main_2', 'base_congestion': 0.0},
    {'from': 'grid_11', 'to': 'grid_12', 'base_congestion': 0.5},
    {'from': 'grid_12', 'to': 'main_3', 'base_congestion': 0.0},
    {'from': 'grid_12', 'to': 'grid_13', 'base_congestion': 0.5},
    {'from': 'grid_7', 'to': 'grid_9', 'base_congestion': 0.5},
    {'from': 'grid_9', 'to': 'main_4', 'base_congestion': 0.0},
    {'from': 'grid_9', 'to': 'grid_14', 'base_congestion': 0.5},
    {'from': 'grid_14', 'to': 'grid_13', 'base_congestion': 0.5},
    {'from': 'grid_14', 'to': 'grid_15', 'base_congestion': 0.0},
    {'from': 'grid_13', 'to': 'grid_16', 'base_congestion': 0.0},
    {'from': 'grid_16', 'to': 'grid_17', 'base_congestion': 0.0},
    {'from': 'grid_15', 'to': 'grid_17', 'base_congestion': 0.0},
]

class HospitalProblem:
    """
    Hospital Delivery Problem Instance as Multi-Robot Multi-Objective Dynamic Travelling Thief Problem (TTP) Instance
    
    Hospital topology (N locations (Pharmacy, A&E rooms, Waypoints), corridors, distances, congestion)
    Locations are nodes in a graph with coordinates and effective travel times (A* with congestion) between them
    Waypoints are corridor intersections; main nodes are pharmacy and A&E rooms
    
    Robot fleet (K robots, capacity, speed, positions, battery)
    Medications (order_number, destination, priority)
    To extend Medications simulation: [order_number, destination, priority, deadline, arrival_time]

    K robots must visit N locations (0 pharmacy + N A&E rooms) to deliver M medications
    Each location has medications to be delivered
    Each medication has a order number, a destination (A&E room), a priority and a deadline
    The deadline can be given by the historical data or deducted by the priority
    Each robot has limited capacity (knapsack constraint)
    Each robot has a battery capacity (max travel time) and a speed (cm/min)
    SR (Single-Robot) Each medication is collected and delivered by only one robot
    Time-extended: Tasks have temporal ordering (tour sequence)
    """

    def __init__(self, num_cities, num_robots, knapsack_capacity, speed, battery_capacity, seed = None):

        self.num_cities = num_cities # total number of locations (pharmacy + A&E rooms) - N
        self.num_items = None # total number of medications - M
        self.num_robots = num_robots # total number of robots in parallel - K

        self.seed = seed

        self.trip_start_time = 0.0 # simulation time when the robot starts its trip
        
        self.position = None # list[(x,y)], one point label per robot
        self.battery = None # list[float], one battery level per robot
        self.knapsack_capacity = knapsack_capacity 
        self.speed = speed 
        self.battery_capacity = battery_capacity   
        
        self.cities = None # (num_cities, 2): (x, y) coordinates of main locations (pharmacy + A&E rooms)
        self.time_matrix = None # = HospitalMap.time_matrix (num_cities × num_cities): effective travel times between locations (A* with congestion) time[i,j] = sum((d_e * (1 + (b_c + d_c))) / rs) in minutes along A* path        
        
        self.items = None # (num_items, 5): [order_number, destination, priority, deadline, arrival_time]
        self.items_by_city = None # items_by_city[room] = [items, ...]: list of medication to deliver at each room
        self.indices = None # full set of medication indices: {0, ..., num_items-1}
                
        self.fitness_bounds = None
        self.hospital_map = None # HospitalMap instance with A* routing and congestion model

    def get_distance(self, room_i, room_j):
        # Get the distance (effective travel time) from room_i to room_j using the precomputed time_matrix (A* with congestion)
        return self.time_matrix[room_i, room_j]

    def get_items_at_city(self, room_index):
        # Medications to deliver at a n room
        # Get list of item indices at a given room
        # Robot arrives at city -> checks items_by_city[room_index] to see which medications to deliver there
        return self.items_by_city[room_index]

class HospitalMap:
    """
    Hospital corridor graph with A* routing and and per-edge congestion model

    Nodes:
        - Main nodes: pharmacy and A&E rooms
        - Waypoints: corridor intersections

    Corridor list: (distance, congestion)

    time_matrix: (N, N) - A* travel times in minutes between main nodes
    micro_routes: dict (li, lj) - list[node_id] – full node path for each pair
    """

    def __init__(self):
        self.nodes = {} # node_id: (x, y) coordinates (waypoints and main nodes), positions in cm
        self.node_meta = {} # node_id - {'label', 'descr_serv', 'is_main'}
        self.corridor = {} # node_id: [(start_node,  base_dist_cm, eff_dist_cm, congestion), ...]: bidirectional
        self.main_node_ids = [] # main_node_ids[0] = pharmacy; main_node_ids[k] = A&E room k ({'label', 'descr_serv', 'is_main'})
        self.micro_routes = {} # (i, j): list of node_ids along A* path from main_node_ids[i] to main_node_ids[j]
        self.time_matrix = None # time_matrix[i,j]: sum((d * (1 + (b_c + d_c))) / rs) along A* path

    def add_node(self, node_id, x, y, is_main=False, label=None, descr_serv=None):
        """
        Add a node to the map

        Args:
            node_id: Unique identifier ('main_0', 'grid_3_4')
            x, y: Coordinates
            is_main: True - pharmacy or A&E room; False - waypoint
        """
        self.nodes[node_id] = (x, y)
        if node_id not in self.corridor: # initialize corridor list for this node
            self.corridor[node_id] = []
        if is_main:
            self.main_node_ids.append(node_id)
        self.node_meta[node_id] = {'label': label, 'descr_serv': descr_serv, 'is_main': is_main}

    def add_edge(self, u, v, congestion):
        """
        Add a bidirectional edge between two nodes.

        Args:
            u, v: node IDs to connect
            distance: computed using Euclidean distance from coordinates in cm
            base_congestion (b_c): 
                - 0.0 - service corridor
                - 0.5 - general corridor
                - lower = faster
            dynamic_congestion (d_c) - for now 0 sampled when robots get to nodes (waypoints)
            effective distance = d * (1 + (b_c + d_c))            
        """
        
        x1, y1 = self.nodes[u]
        x2, y2 = self.nodes[v]
        # Euclidean distance between u and v acepted because is biger than the real distance through the corridors
        base_dist = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        
        eff_dist  = base_dist * (1.0 + congestion)
        # Relate nodes u and v in the corridor graph with distance and congestion
        self.corridor[u].append((v, base_dist, eff_dist, congestion))
        self.corridor[v].append((u, base_dist, eff_dist, congestion))

    def astar(self, start_id, goal_id, robot_speed):
        """
        A* search to find the optimal path from main node start_id to main node goal_id
                
        edge_cost = (d_e * (1 + (b_c + d_c))) / rs        
        
        Returns: (total_time, path)
            total_time: sum(edge_cost) along optimal path
            path: list of node IDs from start to goal

        Reference: Hart et al. (1968).
        """
        if start_id == goal_id:
            return 0.0, [start_id]

        # A* with a priority queue (min-heap) for the open set
        open_set = [(float(np.sqrt((self.nodes[goal_id][0] - self.nodes[start_id][0])**2 + (self.nodes[goal_id][1] - self.nodes[start_id][1])**2)), 0.0, start_id)]
        g_time = {start_id: 0.0} # best time cost to each node
        parent = {} # for path reconstruction
        closed = set() # explored nodes

        while open_set: # while there are nodes to explore            
            _, gt, current = heappop(open_set) # get node with lowest total time

            if current == goal_id:
                # reconstruct path from goal back to start
                path = [current]
                while current in parent:
                    current = parent[current]
                    path.append(current)
                path.reverse()
                return gt, path

            if current in closed:
                continue
            closed.add(current)

            for neighbour, base_d, eff_d, cong in self.corridor.get(current, []):
                if neighbour in closed:
                    continue
                
                # t = (d * (1 + (b_c + d_c))) / rs
                t = gt + eff_d / robot_speed
                
                # Add 5 minutes to the time between main_0 and grid_0 (and vice versa) to simulate elevator wait time
                if (current == 'main_0' and neighbour == 'grid_0') or (current == 'grid_0' and neighbour == 'main_0'):
                    t += 5.0

                if neighbour not in g_time or t < g_time[neighbour]:
                    g_time[neighbour] = t
                    parent[neighbour] = current
                    # f_score = g_score + heuristic (estimated time to goal)
                    # g_score: best time cost from start to current node
                    # heuristic: straight-line euclidean distance to goal divided by max speed (best case time)
                    dist_restante = float(np.sqrt((self.nodes[goal_id][0] - self.nodes[neighbour][0])**2 + (self.nodes[goal_id][1] - self.nodes[neighbour][1])**2))
                    f_score = t + (dist_restante / robot_speed)
                    heappush(open_set, (f_score, t, neighbour))

        return np.inf, []

    def precompute_effective_distances(self, robot_speed):
        """
        Pre-compute A* travel times between all pairs of main nodes
        
                                [time(0,0), ..., time(0,N-1)]
        Fills time_matrix[N,N]: [...      , ...,        ... ]
                                [time(1,0), ..., time(1,N-1)]
    
        time[i,j] = sum((d_e * (1 + (b_c + d_c))) / rs) in minutes along A* path        
        """

        label_to_id = {
            meta['label']: nid
            for nid, meta in self.node_meta.items()
            if meta.get('is_main') and meta.get('label') is not None
        }

        n = max(label_to_id) + 1 # num_cities = max_label + 1
        self.main_node_ids = [label_to_id.get(i) for i in range(n)]
        self.time_matrix = np.full((n, n), np.inf) # initialize with infinity
        self.micro_routes = {}

        for li in range(n):
            u_id = self.main_node_ids[li]
            if u_id is None:
                continue
            self.time_matrix[li, li] = 0.0
            self.micro_routes[(li, li)] = [u_id]

            for lj in range(n):
                if li == lj:
                    continue
                v_id = self.main_node_ids[lj]
                if v_id is None:
                    continue
                t, path = self.astar(u_id, v_id, robot_speed)
                self.time_matrix[li, lj] = t
                self.micro_routes[(li, lj)] = path
                
    def update_corridor_congestion(self, waypoint_id, rng, robot_speed):
        """
        Robot arrived at waypoint_id: re-sample congestion for every adjacent edge.
        Updates self.corridor (both directions) then rebuilds time_matrix via A*.

        Args:
            waypoint_id: node_id string (e.g. 'grid_3')
            rng: np.random.RandomState
        """
        if waypoint_id not in self.corridor:
            return
                
        updated = []
        for nbr, base_d, _, _ in self.corridor[waypoint_id]:
            new_cong = float(np.clip(rng.uniform(0.0, 1.0), 0.0, 1.0))
            new_eff = base_d * (1.0 + new_cong)
            updated.append((nbr, base_d, new_eff, new_cong))
            # Mirror: update the reverse edge on the neighbour
            for j, (u, bd, ed, cg) in enumerate(self.corridor[nbr]):
                if u == waypoint_id:
                    self.corridor[nbr][j] = (u, bd, new_eff, new_cong)
                    break
        self.corridor[waypoint_id] = updated
        self.precompute_effective_distances(robot_speed) # rebuilds time_matrix and micro_routes

def create_hospital_map(nodes_def, edges_def):
    """
    Generate a hospital graph from CHUC floor plan
    With a grid of waypoints, A&E rooms and pharmacy
    Points of the graph conected by congestion-weighted corridors
    """
    
    hmap = HospitalMap()

    for n in nodes_def:
        hmap.add_node(n['node_id'], n['x'], n['y'], is_main = n.get('is_main', False), label = n.get('label'), descr_serv = n.get('descr_serv'))

    for e in edges_def:
        hmap.add_edge(e['from'], e['to'], e['base_congestion'])
    
    return hmap

def plot_hospital_map(hmap, title='HUC Hospital Map', ax=None):
    """
    Draw the hospital graph.

    Edges colour-coded by congestion
        green : < 0.35
        orange: 0.35 – 0.70
        red: > 0.70

    Nodes
        blue: pharmacy (label 0)
        red: A&E room (label 1..N)
        gray: waypoint

    Parameters:
        hmap: HospitalMap
        title: str
        ax: matplotlib Axes or None  (creates a new figure if None)
    """

    show = ax is None
    if show:
        fig, ax = plt.subplots(figsize=(16, 10))

    seen_edges = set()
    for u, edges in hmap.corridor.items():
        x1, y1 = hmap.nodes[u]
        for v, base_d, eff_d, cong in edges:
            key = tuple(sorted([u, v]))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            x2, y2 = hmap.nodes[v]
            color = ('green' if cong < 0.35 else 'orange' if cong < 0.70 else'red')
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.8, alpha=0.75, zorder=1)

            # Congestion value at edge midpoint
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f'{cong:.2f}', fontsize=6, ha='center', va='center', color='dimgray', zorder=2)

            # Base distance label (cm) slightly offset
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 - 60, f'{base_d:.0f} cm', fontsize=5, ha='center', va='top', color='steelblue', zorder=2)

    for node_id, (x, y) in hmap.nodes.items():
        meta = hmap.node_meta.get(node_id, {})
        if meta.get('is_main'):
            label = meta.get('label', '')
            descr = meta.get('descr_serv', '')
            color = 'royalblue' if label == 0 else 'crimson'
            ax.scatter(x, y, s=350, c=color, zorder=5, edgecolors='black', linewidths=1.0)
            ax.annotate(f'[{label}] {descr}', (x, y), textcoords='offset points', xytext=(0, 14), ha='center', fontsize=8, fontweight='bold', zorder=6)
        else:
            ax.scatter(x, y, s=35, c='lightgray', zorder=4, edgecolors='gray', linewidths=0.5)
            ax.annotate(node_id, (x, y), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=5, color='gray', zorder=4)

    legend_handles = [
        mpatches.Patch(color='royalblue', label='Pharmacy (depot)'),
        mpatches.Patch(color='crimson', label='A&E Room'),
        mpatches.Patch(color='lightgray', label='Waypoint'),
        mpatches.Patch(color='green', label='Low congestion  (< 0.35)'),
        mpatches.Patch(color='orange', label='Med congestion  (0.35–0.70)'),
        mpatches.Patch(color='red', label='High congestion (> 0.70)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

def compute_fitness_bounds(problem):
    """
    Weighted Tchebycheff Aggregation:
        - Multi-objective optimization via adaptive Weighted Tchebycheff Aggregation 
        - Convert Multi-Objective Fitness to Scalar Fitness 
        - scalar_fitness = max {w_i x normalized(f_i)}
        - w_i: objective weight
        - normalized(f_i) = f_i / nadir_i
    
    
    nadir_i: 
        - Worst-case objective fitness, derived from problem structure, population-independent.
        
        - worst_makespan: all items of the robot are delivered by the worst possible path.
        - worst_twt: all items at max weight (P1=3), fully late (delivered at worst_tct and deadline=0).
    """
    
    d = problem.time_matrix
    worst_trip = float(np.max(d[d < np.inf]))
    # worst_makespan = Exit from the pharmacy, visit all A&E rooms by the worst possible path and return to pharmacy    
    worst_makespan = worst_trip * problem.num_cities
    worst_tct = problem.trip_start_time + worst_makespan  
    worst_twt = 3.0 * problem.num_items * worst_tct  # weight = 3 (P1), all items at worst_tct, deadline = 0

    return np.array([worst_makespan, worst_twt])

def create_hospital_problem(num_robots, knapsack_capacity, speed, battery_capacity, seed):
    """
    Build a HospitalProblem from a batch using the real hospital graph
    Medications randomly assigned to rooms with priority levels from the batch
    Three priority levels (1=high, 2=medium, 3=low)
    Probability of congestion on corridors from the robot, is randomly assigned to edges
    5 locations: pharmacy + 4 A&E rooms.
    """
   
    hmap = create_hospital_map(HOSPITAL_NODES, HOSPITAL_EDGES) # generates hospital map with A* routing and congestion
    hmap.precompute_effective_distances(speed)
    
    num_cities = len(hmap.time_matrix)
    
    
    problem = HospitalProblem(num_cities = num_cities, num_robots = num_robots, knapsack_capacity = knapsack_capacity, speed = speed, battery_capacity = battery_capacity, seed = seed)
    
    # (num_cities, 2): (x, y) coordinates of main locations (pharmacy + A&E rooms)
    problem.cities = np.array([hmap.nodes[nid] if nid is not None else (0.0, 0.0) for nid in hmap.main_node_ids])

    problem.num_robots = num_robots
    
    problem.knapsack_capacity = knapsack_capacity
    problem.speed = speed
    problem.battery_capacity = battery_capacity
        
    problem.hospital_map = hmap # Complete hospital map with nodes, corridors, and precomputed A* paths
    problem.time_matrix = hmap.time_matrix # (num_cities × num_cities): effective travel times between locations (A* with congestion)     

    return problem

def update_hospital_problem(state_dict):
        
    problem = state_dict['problem']
    sim = state_dict['sim']
    robots = state_dict['robots']
    
    problem.trip_start_time = state_dict['sim_time']

    available_idx = sim.get_available(state_dict['sim_time'])
    items = sim.items[available_idx].copy()    
    problem.items = items # (num_items, 5): [order_number, destination, priority, deadline, arrival_time]
    problem.num_items = len(problem.items)
    
    # builds reverse room - medications index: room - list of medication indices
    problem.items_by_city = [[] for _ in range(problem.num_cities)]
    for idx in range(problem.num_items):
        room = int(items[idx, 1]) 
        if 0 <= room <  problem.num_cities:
            problem.items_by_city[room].append(idx)

    problem.indices = set(range(problem.num_items))
    
    # Synchronise real-time robot inventory and routes
    problem.robots_cargo = []
    problem.robots_current_tour = []
    problem.robots_tour_idx = []
    
    for r in robots:
        local_cargo = []
        for global_idx in r.cargo:
            if global_idx in available_idx:
                local_cargo.append(available_idx.index(global_idx))
        problem.robots_cargo.append(local_cargo)
        problem.robots_current_tour.append(r.current_tour.copy())
        problem.robots_tour_idx.append(r.tours)
        
    problem.fitness_bounds = compute_fitness_bounds(problem)
    
    problem.position = [rs.position for rs in robots]
    problem.battery = [rs.battery for rs in robots]
        
    problem.hospital_map.precompute_effective_distances(problem.speed)
    problem.time_matrix = problem.hospital_map.time_matrix