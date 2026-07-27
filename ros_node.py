"""
PharmaRobot - ROS2 Integration Node
Multi-Robot Multi-Objective Dynamic Travelling Thief Problem (TTP)
"""

import json
import numpy as np
from typing import Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from simulation_batches import RobotState, GADecisionService, BatchSimulator
from ttp_problem import create_hospital_problem, update_hospital_problem


class PharmaRobotDecisionNode(Node):

    NODE_CONFIG = {
        'topics': {
            'subscriber': '/pharma_robot/current_state',
            'publisher': '/pharma_robot/ga_decisions'
        },
        'physical_constraints': {
            'num_robots': 1,
            'knapsack_capacity': 6,
            'speed': 0.25,
            'battery_capacity': 240.0
        },
        'ga_hyperparameters': {
            'pop_size': 200,
            'generations': 100,
            'cx_pb_tour': 0.8,
            'cx_pb_pack': 0.8,
            'mut_pb_tour': 0.2,
            'mut_pb_pack': 0.1,
            'tournament_size': 2,
            'elitism': 2,
            'n_jobs': -1,
        },
        'runtime': {
            'seed': 42,
            'queue_size': 10,
        }
    }

    def __init__(self):
        super().__init__('pharma_robot_decision_node')
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.NODE_CONFIG['runtime']['queue_size']
        )

        self.state_subscriber = self.create_subscription(
            String, 
            self.NODE_CONFIG['topics']['subscriber'], 
            self.state_cb, 
            qos_profile
        )
        
        self.decision_publisher = self.create_publisher(
            String, 
            self.NODE_CONFIG['topics']['publisher'], 
            10
        )
        
        self.decision_id = 0
        
        self.problem = create_hospital_problem(
            num_robots=self.NODE_CONFIG['physical_constraints']['num_robots'], 
            knapsack_capacity=self.NODE_CONFIG['physical_constraints']['knapsack_capacity'],
            speed=self.NODE_CONFIG['physical_constraints']['speed'],
            battery_capacity=self.NODE_CONFIG['physical_constraints']['battery_capacity'],
            seed=self.NODE_CONFIG['runtime']['seed']
        )
        
        self.ga_service = GADecisionService(
            self.NODE_CONFIG['ga_hyperparameters'], 
            self.NODE_CONFIG['runtime']['seed'], 
            log_dir="/dev/null"
        )

        self.ga_service.log_metrics = lambda *args, **kwargs: None
        
        self.get_logger().info("PharmaRobot Decision Node operational.")

    def state_cb(self, msg: String):
        """
        Triggered when a new physical state is published. 
        """
        # Input parsed from JSON string to dictionary
        try:
            input_dict: Dict[str, Any] = json.loads(msg.data)
        except json.JSONDecodeError:
            return
            
        medications = input_dict.get('medications', [])
        robots_data = input_dict.get('robots', [])
        
        if not medications or not robots_data:
            return
            
        self.decision_id += 1
            
        # Phisical stat mapping
        items_array = np.zeros((len(medications), 5), dtype=float)
        for i, m in enumerate(medications):
            items_array[i, :4] = m[:4]
            
        robots = []
        for r_data in robots_data:
            r = RobotState(
                position=r_data.get('position', 0),
                battery=r_data.get('battery', self.NODE_CONFIG['physical_constraints']['battery_capacity']),
                knapsack_capacity=self.NODE_CONFIG['physical_constraints']['knapsack_capacity'],
                speed=self.NODE_CONFIG['physical_constraints']['speed'],
                battery_capacity=self.NODE_CONFIG['physical_constraints']['battery_capacity']
            )
            r.cargo = r_data.get('cargo', [])
            r.current_tour = r_data.get('current_tour', [])
            r.tours = r_data.get('tours', 0)
            robots.append(r)
            
        state_dict = {
            'sim_time': 0.0,
            'problem': self.problem,
            'robots': robots,
            'sim': BatchSimulator(items_array)
        }
        
        update_hospital_problem(state_dict)
        best_ind, elapsed = self.ga_service.get_decision(state_dict, self.decision_id)
        
        if best_ind is None:
            return
            
        # Output dictionary to be serialized and published
        output_dict: Dict[str, Any] = {
            'decision_id': self.decision_id,
            'elapsed_time_seconds': elapsed,
            'decisions': [{
                'robot_id': k,
                'tour': best_ind.tours[k].copy() if isinstance(best_ind.tours[k], list) else best_ind.tours[k].tolist(),
                'assigned_items': [int(idx) for idx in np.where(best_ind.item_assignment == k)[0]],
                'charge_aggressiveness': float(best_ind.charge_aggressiveness[k])
            } for k in range(len(robots))]
        }
        
        self.decision_publisher.publish(String(data=json.dumps(output_dict)))


def main(args=None):
    rclpy.init(args=args)
    node = PharmaRobotDecisionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()