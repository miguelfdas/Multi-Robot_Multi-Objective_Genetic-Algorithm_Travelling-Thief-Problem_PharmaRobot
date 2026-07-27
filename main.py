"""
PharmaRobot - Flash

Alpha Version -> One robot

Hospital Delivery Problem Definition as Multi-Robot Multi-Objective Dynamic Travelling Thief Problem (TTP)
    - Identify interdependencies between the Medicine Allocation and the Path Planning.

Classification of the problem - Multi-Robot Task Allocation (MRTA) according to Gerkey and Mataric:
    - MT (Multi-Task): The robot transports multiple items to different destinations
    - SR (Single-Robot): Each task is performed by a single robot
    - TA (Time-Extended): Task management within a time window
    
3-layer architecture:
    Layer 1: 
        - Medicine Allocation (KnapSack Problem) - Integer representation to determine which medicines each robot caries.
        - [-1 (unassigned), 0, 1, 0, 2, -1, …, K (assigned to robot k)]
    Layer 2: 
        - Global Path Planning (Travelling Salesman Problem)
        - Permutation representation to determine the optimal delivery sequence of A&E rooms.
        - [2, 3, 0 (pick medicines), 3, 1, 4, 2, 0  (pharmacy)]
        - Robot alloways pick up medications at the pharmacy, but can start anywhere in the map
        - The GA can interrupt a trip and return to the pharmacy if a high-priority medicine with a tight deadline becomes available.
    Layer 3: 
        - Local Path Planning - Micro-Routing (A*) 
        - Graph-based search algorithm to find the optimal corridor path between consecutive A&E rooms considering congestion.
        - Robot gets toptimal corridor path between consecutive A&E rooms
        - Robot gets to a waypoint or main node, re-samples congestion for adjacent corridors, and updates its route accordingly (dynamic environment)
        - Between pharmacy and waypoint 0, add a fixed 5 minutes to simulate elevator wait time

Pipeline:
    1. Generate a batche of items with realistic arrival times and priorities based on historical data
    2. Set the robot start position and initial battery
    3. Build a fix HospitalProblem
    4. Integrate the batch and robot in the hospital map
    5. Start the GA process
    6. GA read simulation state, output a GA individual (tour and medicines allocation)
    7. Simulate batch with the GA's best solution, advancing the simulation clock and marking delivered items
    8. Save metrics
    
    - The GA hyperparameters are fixed

Assumptions:  
    - Number of robots: 1
    - Robots’ speed: 0.25 m/s  == 0.9 km/h
    - Robots' battery capacity: 240 min == 4h
    - Batch size: 120 Medicines (From historical data, an average of 60 medicines per day, with a maximum of 340)
    - Knapsack capacity: 6 (number of difrent orders per robot)

Simulation features :
    - Task Allocation and Route Optimisation Model
    - Assign orders to robots and optimise their tours. 
    - If a high-priority medicine with a tight deadline becomes available, the GA should be able to interrupt a trip and return to the pharmacy to realize this order
    - Battery constraints are respected and the GA should be able to decide when charge and or how long stay in charging.
    - Delivery time = Time available in the batch without being selected for delivery + trip time

"""

import os
import numpy as np
import pandas as pd
import shutil
import json
from simulation_batches import BatchSimulator, generate_batches, RobotState, GADecisionService
from ttp_problem import create_hospital_problem, update_hospital_problem

def summary(results_dir='results_sim'):
    log_continuous_file = f'{results_dir}/log_batch.csv'
    log_ga_file = f'{results_dir}/ga_log.csv'

    df_trip = pd.read_csv(log_continuous_file)
    df_ga = pd.read_csv(log_ga_file)

    global_kpis = {
        'Total_Decisions_Made_by_GA': df_ga['decision_id'].nunique(),
        'Total_Physical_Deliveries': df_trip['delivered_count'].sum(),
        'Total_Simulation_Time_Hours': df_trip['sim_time_end'].max() / 60.0,
        'Avg_GA_Computation_Time_sec': df_ga['elapsed_time_minutes'].mean(),
        'Final_System_TWT_Expected': df_ga['best_TWT'].iloc[-1],
        'Average_Genotypic_Diversity': df_ga['genotypic_diversity'].mean(),
        'Average_Phenotypic_Diversity': df_ga['phenotypic_diversity'].mean(),
        'Max_Backlog_Size': df_ga['items_indices'].apply(lambda x: len(json.loads(x))).max()
    }

    summary_df = pd.DataFrame([global_kpis])
    summary_df.to_csv(f'{results_dir}/summary_simulation_global.csv', index=False)

def simulate_batch(params, num_robots, items_per_batch, mean_interarrival_min, num_main, knapsack_capacity, battery_capacity, speed, seed):    
    # Alpha version: 1 batch, 1 run, 1 robot, fixed GA hyperparameters, fixed seed
    

    results_dir = 'results_sim'    
    
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    
    os.makedirs(results_dir, exist_ok=True)
    
    rng_sim = np.random.RandomState(seed)
        
    speed = speed * 100 * 60 # m\s to cm/min
    
    sim_time = 0.0
    time_step = 1.0 
    
    # 1. Generate a batche of items with realistic arrival times and priorities based on historical data
    # Generate a simulation batch with realistic parameters based on historical data
    # Batch size very large than robot capacity to force multiple trips
    batch = generate_batches(items_per_batch = items_per_batch, seed = seed, mean_interarrival_min = mean_interarrival_min)
    items = batch['items']
    sim = BatchSimulator(items)
    
    # 2. Set the robot start position and initial battery
    position = [int(rng_sim.randint(0, num_main)) for _ in range(num_robots)]
    battery = [float(rng_sim.uniform(0.3 * battery_capacity, battery_capacity)) for _ in range(num_robots)]
    
    # knapsack_capacity - the robot have 6 sleeves - the robot can carry 6 different orders at the same time
    robots = [RobotState(position[i], battery[i], knapsack_capacity, speed, battery_capacity) for i in range(num_robots)]
    
    # Continuous motion tracking variables
    robot_travel_times = [0.0 for _ in range(num_robots)]
    robot_destinations = [pos for pos in position]
    
    # Physical tracking variables in the corridors
    robot_initial_travel_times = [0.0 for _ in range(num_robots)]
    robot_active_path = [[] for _ in range(num_robots)]
    robot_path_progress = [0 for _ in range(num_robots)]
    
    for r in robots:
        r.tours = 0
    
    # 3. Build a fix HospitalProblem
    # 4. Integrate the robots in the hospital map
    problem = create_hospital_problem(num_robots, knapsack_capacity, speed, battery_capacity, seed)    
    
    ga = GADecisionService(params, seed, log_dir=results_dir)    
    
    trip_log = []
    item_delivery_log = []
    prev_available = []
    best = None
    
    decision_id = 0
    robot_current_decision = [-1 for _ in range(num_robots)]
    ga_cooldown = 0.0

    while not sim.is_empty():
        available = sim.get_available(sim_time)
        environment_changed = (len(available) != len(prev_available))
        robot_idle_and_charged = False
        
        if available and best is not None:
            for k in range(num_robots):
                if robot_travel_times[k] == 0 and robots[k].position == 0:
                    tour = robots[k].current_tour
                    idx = robots[k].tours
                    next_node = tour[idx + 1] if tour and idx + 1 < len(tour) else 0
                    
                    if next_node == 0 and ga_cooldown <= 0:
                        robot_idle_and_charged = True
                        ga_cooldown = time_step * 5 
                        break

        if environment_changed or robot_idle_and_charged:
            decision_id += 1
                        
            state_dict = {
                'sim_time': sim_time,
                'problem': problem,
                'robots': robots,
                'sim': sim
            }
                      
            # 4. Integrate the batch and robot in the hospital map
            update_hospital_problem(state_dict)            
            
            best, elapsed_seconds = ga.get_decision(state_dict, decision_id)
            
            if best is not None:
                for k in range(num_robots):
                    robots[k].current_tour = best.tours[k].copy()
                    robots[k].tours = 0  # Reset active checkpoint index for the new plan
            
            elapsed_mins = elapsed_seconds / 60.0
            
            sim_time += elapsed_mins
            
            # Atualization of robot states during GA computation time
            for k in range(num_robots):
                # If in transit during GA computation, update position and deliveries based on elapsed time
                if robot_travel_times[k] > 0:
                    time_moved = min(elapsed_mins, robot_travel_times[k])
                    robot_travel_times[k] -= time_moved
                    
                    # If in the corridors
                    if robot_initial_travel_times[k] > 0:
                        progress = 1.0 - (robot_travel_times[k] / robot_initial_travel_times[k])
                        total_nodes = len(robot_active_path[k])
                        if total_nodes > 0:
                            expected_idx = min(int(progress * total_nodes), total_nodes - 1)
                            
                            while robot_path_progress[k] <= expected_idx:
                                current_node_id = robot_active_path[k][robot_path_progress[k]]
                                problem.hospital_map.update_corridor_congestion(current_node_id, rng_sim, speed)
                                problem.time_matrix = problem.hospital_map.time_matrix # Sincroniza a Layer 2
                                robot_path_progress[k] += 1
                                
                    if robot_travel_times[k] == 0:
                        robots[k].position = robot_destinations[k]
                        # Mark items as delivered if the robot reached their destination during GA computation
                        remaining_cargo = []
                        delivered_this_node = []
                        for global_idx in robots[k].cargo:
                            if int(items[global_idx, 1]) == robots[k].position:
                                delivered_this_node.append(global_idx)
                            else:
                                remaining_cargo.append(global_idx)
                        
                        robots[k].cargo = remaining_cargo     
                                           
                        if delivered_this_node:
                            sim.mark_delivered(delivered_this_node)
                            
                            for g_idx in delivered_this_node:
                                order_number = items[g_idx, 0]
                                destination = items[g_idx, 1]
                                priority = items[g_idx, 2]
                                deadline = items[g_idx, 3]
                                arrival_time = items[g_idx, 4]
                                
                                tct = sim_time - arrival_time 
                                twt_item = (4.0 - priority) * max(0.0, tct - deadline)
                                
                                item_delivery_log.append({
                                    'order_number': int(order_number),
                                    'decision_id': robot_current_decision[k],
                                    'destination': int(destination),
                                    'priority': int(priority),
                                    'deadline': round(deadline, 2),
                                    'arrival_time': round(arrival_time, 2),
                                    'delivered_time': round(sim_time, 2),
                                    'tct': round(tct, 2),
                                    'twt': round(twt_item, 2)
                                })
                            
                            f0, f1 = (best.fitness[0], best.fitness[1]) if best and best.fitness is not None else (0.0, 0.0)
                            
                            trip_log.append({
                                'decision_id': robot_current_decision[k],
                                'sim_time_end': sim_time,
                                'robot_id': k,
                                'node': robots[k].position,
                                'delivered_items_indices': json.dumps([int(idx) for idx in delivered_this_node]),
                                'delivered_count': len(delivered_this_node),
                                'remaining_in_batch': len(items) - len(sim.delivered),
                                'makespan': f0,
                                'twt': f1,
                                'robot_starts': json.dumps([rs.position for rs in robots]),
                                'robot_batteries': json.dumps([round(rs.battery, 2) for rs in robots]),
                            })
                    
                    remaining_dt = elapsed_mins - time_moved
                    if remaining_dt > 0 and robots[k].position == 0:
                        robots[k].charge(remaining_dt)
                            
                elif robots[k].position == 0:
                    # If stationary at the pharmacy during GA computation, charge the robot
                    robots[k].charge(elapsed_mins)
            
            for r in robots:
                    r.tours = 0
                    
            prev_available = available.copy()
         
        # Continuous Physical State Machine
        for k in range(num_robots):
            # Stationary
            if robot_travel_times[k] == 0:
                # If the environment changed and the GA provided a new tour and item assignment
                if len(available) > 0 and robots[k].current_tour:
                    robot_current_decision[k] = decision_id
                    tour = robots[k].current_tour
                    idx = robots[k].tours
                    next_node = tour[idx + 1] if idx + 1 < len(tour) else 0

                    # At Pharmacy                 
                    if robots[k].position == 0:
                        # Load newly assigned items into the physical robot knapsack state
                        if best is not None and robots[k].tours == 0:
                            assigned_local_indices = np.where(best.item_assignment == k)[0]
                            for local_idx in assigned_local_indices:
                                global_idx = available[local_idx]
                                if global_idx not in robots[k].cargo:
                                    robots[k].cargo.append(global_idx)
                        if next_node == 0:
                            robots[k].current_tour = []
                            robots[k].charge(time_step)
                        # Have assigned items and a next node to deliver to
                        elif robots[k].cargo and next_node != 0:
                            cost_until_next_pharmacy = 0.0
                            temp_idx = idx
                            while temp_idx < len(tour) - 1:
                                c1 = tour[temp_idx]
                                c2 = tour[temp_idx + 1]
                                cost_until_next_pharmacy += problem.time_matrix[c1, c2]
                                if c2 == 0:
                                    break
                                temp_idx += 1
                            
                            gene = best.charge_aggressiveness[k] if best else 0.5
                            target_bat = cost_until_next_pharmacy + gene * (robots[k].battery_capacity - cost_until_next_pharmacy)
                            target_bat = min(target_bat, robots[k].battery_capacity)
                            
                            # Have enough battery to complete the tour, including returning to the pharmacy if needed
                            if robots[k].battery >= target_bat:
                                travel_cost = problem.time_matrix[robots[k].position, next_node]                                                                    
                                robot_destinations[k] = next_node
                                robot_travel_times[k] = travel_cost
                                robots[k].deplete(travel_cost)
                                robots[k].tours += 1
                                robot_initial_travel_times[k] = travel_cost
                                robot_active_path[k] = problem.hospital_map.micro_routes[(robots[k].position, next_node)].copy()
                                robot_path_progress[k] = 0
                            # Do not have enough battery to complete the tour, need to charge
                            else:
                                robots[k].charge(time_step)
                        # No assigned items or no next node, just charge at the pharmacy
                        else:
                            robots[k].charge(time_step)
                    # At A&E rooms
                    else:
                        # If have a next node to deliver to, move towards it
                        if next_node == robots[k].position:
                            # Safeguard: if the algorithm targets the same room, skip to maintain progress sequence
                            robots[k].tours += 1
                        elif next_node != 0:
                            travel_cost = problem.time_matrix[robots[k].position, next_node]
                            robot_destinations[k] = next_node
                            robot_travel_times[k] = travel_cost
                            robots[k].deplete(travel_cost)
                            robots[k].tours += 1
                            robot_initial_travel_times[k] = travel_cost
                            robot_active_path[k] = problem.hospital_map.micro_routes[(robots[k].position, next_node)].copy()
                            robot_path_progress[k] = 0
                        # If end of tour, return to pharmacy
                        elif next_node == 0:
                            travel_cost = problem.time_matrix[robots[k].position, 0]
                            robot_destinations[k] = 0
                            robot_travel_times[k] = travel_cost
                            robots[k].deplete(travel_cost)
                            robots[k].tours = 0
                            robot_initial_travel_times[k] = travel_cost
                            robot_active_path[k] = problem.hospital_map.micro_routes[(robots[k].position, 0)].copy()
                            robot_path_progress[k] = 0
                            robots[k].current_tour = []
                # If no assigned items or no next node
                else:
                    # If at the pharmacy, charge
                    if robots[k].position == 0:
                        robots[k].charge(time_step)
                    # If at an A&E or in corridors, return to pharmacy
                    else:
                        travel_cost = problem.time_matrix[robots[k].position, 0]
                        robot_destinations[k] = 0
                        robot_travel_times[k] = travel_cost
                        robots[k].deplete(travel_cost)
                        robots[k].tours = 0
                        robot_initial_travel_times[k] = travel_cost
                        robot_active_path[k] = problem.hospital_map.micro_routes[(robots[k].position, 0)].copy()
                        robot_path_progress[k] = 0
            # In transit
            else:
                time_moved = min(time_step, robot_travel_times[k])
                robot_travel_times[k] -= time_moved

                if robot_initial_travel_times[k] > 0:
                    progress = 1.0 - (robot_travel_times[k] / robot_initial_travel_times[k])
                    total_nodes = len(robot_active_path[k])
                    
                    if total_nodes > 0:
                        expected_idx = min(int(progress * total_nodes), total_nodes - 1)
                        
                        while robot_path_progress[k] <= expected_idx:
                            current_node_id = robot_active_path[k][robot_path_progress[k]]
                            problem.hospital_map.update_corridor_congestion(current_node_id, rng_sim, speed)
                            problem.time_matrix = problem.hospital_map.time_matrix # Sincroniza a Layer 2
                            robot_path_progress[k] += 1
                            
                # If the robot has reached its destination
                if robot_travel_times[k] == 0:
                    robots[k].position = robot_destinations[k]
                    # Mark items as delivered if the robot reached their destination                    
                    remaining_cargo = []

                    delivered_this_node = []
                    for global_idx in robots[k].cargo:
                        if int(items[global_idx, 1]) == robots[k].position:
                            delivered_this_node.append(global_idx)
                        else:
                            remaining_cargo.append(global_idx)
                    
                    robots[k].cargo = remaining_cargo  
                    
                    if delivered_this_node:
                        sim.mark_delivered(delivered_this_node)
                        
                        for g_idx in delivered_this_node:
                            order_num = items[g_idx, 0]
                            dest = items[g_idx, 1]
                            prioridade = items[g_idx, 2]
                            deadline = items[g_idx, 3]
                            chegada = items[g_idx, 4]
                            
                            tct = (sim_time + time_step) - chegada 
                            twt_item = (4.0 - prioridade) * max(0.0, tct - deadline)
                            
                            item_delivery_log.append({
                                'decision_id': robot_current_decision[k],
                                'sim_time_delivery': round(sim_time + time_step, 2),
                                'item_index': int(g_idx),
                                'order_number': int(order_num),
                                'destination': int(dest),
                                'priority': int(prioridade),
                                'deadline': round(deadline, 2),
                                'arrival_time': round(chegada, 2),
                                'tct': round(tct, 2),
                                'twt_real': round(twt_item, 2)
                            })
                        
                        f0, f1 = (best.fitness[0], best.fitness[1]) if best and best.fitness is not None else (0.0, 0.0)
                        
                        trip_log.append({
                            'decision_id': robot_current_decision[k],
                            'sim_time_end': sim_time + time_step,
                            'robot_id': k,
                            'node': robots[k].position,
                            'delivered_items_indices': json.dumps([int(idx) for idx in delivered_this_node]),
                            'delivered_count': len(delivered_this_node),
                            'remaining_in_batch': len(items) - len(sim.delivered),
                            'makespan': f0,
                            'twt': f1,
                            'robot_starts': json.dumps([rs.position for rs in robots]),
                            'robot_batteries': json.dumps([round(rs.battery, 2) for rs in robots]),
                        })
                        
                    remaining_dt = time_step - time_moved
                    if remaining_dt > 0 and robots[k].position == 0:
                        robots[k].charge(remaining_dt)
        
        if ga_cooldown > 0:
            ga_cooldown -= time_step
                      
        sim_time += time_step
    
    pd.DataFrame(trip_log).to_csv(f'{results_dir}/log_batch.csv', index=False)
    pd.DataFrame(item_delivery_log).to_csv(f'{results_dir}/log_items_delivered.csv', index=False)
    
    summary(results_dir)
    
    return trip_log

def main():
    num_robots = 3
    # num_robots = 3
    items_per_batch = 120
    mean_interarrival_min = 5.0
    num_main = 5 
    knapsack_capacity = 6 # The robot have 6 sleeves
    battery_capacity = 240
    speed = 1
    # speed = 0.5
    seed = 42
 
    params = {
        'pop_size': 200,
        'generations': 10,
        'cx_pb_tour': 0.8,
        'cx_pb_pack': 0.8,
        'mut_pb_tour': 0.2,
        'mut_pb_pack': 0.1,
        'tournament_size': 2,
        'elitism': 2,
        'n_jobs': 1,
    }
 
    # Simulate the problem
    # Run the GA
    simulate_batch(params, num_robots, items_per_batch, mean_interarrival_min, num_main, knapsack_capacity, battery_capacity, speed, seed)    

if __name__ == "__main__":
    main()