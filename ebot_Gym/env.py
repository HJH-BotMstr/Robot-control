'''
@Author: HJH-BotMstr
@Date: 2025-9-5
@Description: An ASCF Path Planning Method for a Hybrid Robot in Coating Inspection Scenarios
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import time
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from loguru import logger
import random

class JointBasedAPFPlanner:
    def __init__(self):
        self.katt = 1.25    
        self.krep = 1.25      
        self.dm = 0.5       
        self.lambda_step = 0.2  
        self.L1 = 0.425    
        self.L2 = 0.39225  
        self.shoulder_height = 0.089159  
        self.shoulder_offset = 0.13585   
        self.elbow_offset = 0.1197      
        self.joint_weights = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    def forward_kinematics_with_joints(self, pose):
        theta1, theta2, theta3 = pose[:3]
        
        base_height = 0.15 + 0.25 + self.shoulder_height
        base_pos = [0, 0, base_height]
        joint_positions = []

        shoulder_pan_pos = base_pos.copy()
        joint_positions.append(shoulder_pan_pos)
        shoulder_lift_pos = [
            base_pos[0],
            base_pos[1], 
            base_pos[2]  
        ]
        joint_positions.append(shoulder_lift_pos)
        elbow_pos = [
            self.L1 * np.cos(theta2) * np.cos(theta1),
            self.shoulder_offset * np.cos(theta1) + self.L1 * np.cos(theta2) * np.sin(theta1),
            base_pos[2] + self.L1 * np.sin(theta2)
        ]
        joint_positions.append(elbow_pos)
        
        wrist1_pos = [
            (self.L1 * np.cos(theta2) + self.L2 * np.cos(theta2 + theta3)) * np.cos(theta1),
            (self.shoulder_offset - self.elbow_offset * np.cos(theta2) + 
             self.L2 * np.sin(theta2 + theta3) * np.sin(theta1)) * np.cos(theta1) + \
            (self.L1 * np.cos(theta2) + self.L2 * np.cos(theta2 + theta3)) * np.sin(theta1),
            base_pos[2] + self.L1 * np.sin(theta2) + self.L2 * np.sin(theta2 + theta3)
        ]
        joint_positions.append(wrist1_pos)
        end_extension = 0.1
        end_pos = [
            wrist1_pos[0] + end_extension * np.cos(theta2 + theta3) * np.cos(theta1),
            wrist1_pos[1] + end_extension * np.cos(theta2 + theta3) * np.sin(theta1),
            wrist1_pos[2] + end_extension * np.sin(theta2 + theta3)
        ]
        
        return end_pos, joint_positions
    
    def forward_kinematics_3d(self, pose):
        end_pos, _ = self.forward_kinematics_with_joints(pose)
        return end_pos[0], end_pos[1], end_pos[2]

    def plan(self, start_pose, target_pos_3d, obstacle_pos_3d):
        path = [start_pose.copy()]
        current_pose = start_pose[:4].copy()  # 前4关节
        flag = False
        max_iterations = 30
        
        for iteration in range(max_iterations):
            if flag:
                break
            U = []
            J = []
            for i in range(4):  
                for delta in [-self.lambda_step, 0, self.lambda_step]:
                    new_pose = current_pose.copy()
                    new_pose[i] += delta
                    
                    joint_limits = [
                        [-np.pi, np.pi],      #  shoulder_pan
                        [-2.2, 0],            # shoulder_lift  
                        [-2.8, 2.8],          # elbow
                        [-2.2, 2.2],          # θwrist_1
                    ]
                    new_pose[i] = np.clip(new_pose[i], joint_limits[i][0], joint_limits[i][1])
                    u = self.calculate_joint_based_potential(new_pose, target_pos_3d, obstacle_pos_3d)
                    U.append(u)
                    J.append(new_pose.copy())
            
            min_idx = np.argmin(U)
            j_min = J[min_idx]
            
            if np.allclose(j_min, current_pose, atol=1e-3):
                flag = True
            else:
                current_pose = j_min.copy()
                path.append(current_pose.copy())
        
        return current_pose
    
    def calculate_joint_based_potential(self, pose, target_pos, obstacle_pos):
        end_pos, joint_positions = self.forward_kinematics_with_joints(pose[:3])
        
        dist_to_target = np.sqrt(
            (end_pos[0] - target_pos[0])**2 + 
            (end_pos[1] - target_pos[1])**2 + 
            (end_pos[2] - target_pos[2])**2
        )
        uatt = 0.5 * self.katt * dist_to_target**2
        
        urep_total = 0.0
        
        for i, joint_pos in enumerate(joint_positions):
            dist_to_obstacle = np.sqrt(
                (joint_pos[0] - obstacle_pos[0])**2 + 
                (joint_pos[1] - obstacle_pos[1])**2 + 
                (joint_pos[2] - obstacle_pos[2])**2
            )
            
            if dist_to_obstacle <= self.dm:
                weight = self.joint_weights[min(i, len(self.joint_weights)-1)]
                urep = weight * 0.5 * self.krep * (1/max(dist_to_obstacle, 0.01) - 1/self.dm)**2
                urep_total += urep
        
        if len(pose) >= 4:
            if len(joint_positions) >= 4: 
                wrist_pos = joint_positions[3]
            else:
                wrist_pos = end_pos  
            
            wrist_dist_to_obstacle = np.sqrt(
                (wrist_pos[0] - obstacle_pos[0])**2 + 
                (wrist_pos[1] - obstacle_pos[1])**2 + 
                (wrist_pos[2] - obstacle_pos[2])**2
            )
            
            if wrist_dist_to_obstacle <= self.dm:
                weight_4 = self.joint_weights[4] if len(self.joint_weights) > 4 else 3.0
                urep_4 = weight_4 * 0.5 * self.krep * (1/max(wrist_dist_to_obstacle, 0.01) - 1/self.dm)**2
                urep_total += urep_4
        
        end_dist_to_obstacle = np.sqrt(
            (end_pos[0] - obstacle_pos[0])**2 + 
            (end_pos[1] - obstacle_pos[1])**2 + 
            (end_pos[2] - obstacle_pos[2])**2
        )
        
        if end_dist_to_obstacle <= self.dm:
            end_weight = max(self.joint_weights) * 1.5
            end_urep = end_weight * 0.5 * self.krep * (1/max(end_dist_to_obstacle, 0.01) - 1/self.dm)**2
            urep_total += end_urep
        
        return uatt + urep_total

class MobileManipulatorEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, gui=False, fixed_scene=False,complex_scene=False, channel_width=1.3):
        super(MobileManipulatorEnv, self).__init__()
        self.fixed_scene = fixed_scene  # True: 固定场景, False: 随机场景
        self.complex_scene = complex_scene
        self.step_num = 0
        self.max_steps = 200  # 单轮最大步数T
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), 
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(29,), dtype=np.float32
        )
        
        self.channel_width = channel_width  
        self.obstacle_radius = 0.1  
        self.target_radius = 0.03   
        self.d_min = 0.15           
        # 引导机制参数
        self.omega = 3.0  
        
        if gui:
            self.p = bullet_client.BulletClient(connection_mode=p.GUI)
        else:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)
            
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.apf_planner = JointBasedAPFPlanner()
        self.robot_id = None
        self.target_id = None
        self.obstacle_id = None
        self.obstacle2_id = None
        self.wall_ids = [] 
        self.arm_joint_indices = [] 
        self.end_effector_index = None
        self.target_pos = None
        self.obstacle_pos = None
        self.obstacle2_pos = None
        self.apf_target_pose = None
        
        self.init_env()
        self.apf_activation_count = 0
        self.total_guidance_steps = 0
    
    def init_env(self):
        self.plane_id = self.p.loadURDF("plane.urdf")
        
        # 加载机器人,用绝对路径
        self.robot_id = self.p.loadURDF(
            "............../src/ebot_description/urdf/ebot_ur5.urdf",
            basePosition=[0, 0, 0.15], 
            baseOrientation=self.p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )
        self.setup_robot_joints()
        self.setup_scene()
    
    def setup_robot_joints(self):
        num_joints = self.p.getNumJoints(self.robot_id)
        target_joints = {
            'shoulder_pan_joint': None,   # θ1
            'shoulder_lift_joint': None,  # θ2
            'elbow_joint': None,          # θ3  
            'wrist_1_joint': None,        # θ4
            'wrist_2_joint': None,        # θ5
            'wrist_3_joint': None         # θ6
        }
        
        print("机器人关节信息:")
        for i in range(num_joints):
            joint_info = self.p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            print(f"关节 {i}: {joint_name}")
            
            if joint_name in target_joints:
                target_joints[joint_name] = i
        
        self.arm_joint_indices = [
            target_joints['shoulder_pan_joint'],   # θ1
            target_joints['shoulder_lift_joint'],  # θ2
            target_joints['elbow_joint'],          # θ3
            target_joints['wrist_1_joint'],        # θ4
            target_joints['wrist_2_joint'],        # θ5
            target_joints['wrist_3_joint']         # θ6
        ]
        
        self.arm_joint_indices = [idx for idx in self.arm_joint_indices if idx is not None]
        self.end_effector_index = None
        for i in range(num_joints):
            joint_info = self.p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8').lower()
            if any(keyword in joint_name for keyword in ['ee', 'end', 'tool', 'gripper']):
                self.end_effector_index = i
                break
        if self.end_effector_index is None:
            self.end_effector_index = num_joints - 1
            
        print(f"使用的6个机械臂关节索引: {self.arm_joint_indices}")
        print(f"末端执行器索引: {self.end_effector_index}")
    
    def setup_scene(self):
        if self.fixed_scene:
            self.setup_fixed_scene()
        elif self.complex_scene:
            self.setup_complex_scene()
        else:
            self.setup_random_scene()
        self.create_channel_walls()
    
    def setup_fixed_scene(self):
        self.target_pos = [2.45, 0.015, 1.05]  
        self.obstacle_pos = [1.35, 0.0, 1.095]  
        self.create_target_and_obstacles()
    
    def setup_random_scene(self):
        self.target_pos = [
            np.random.uniform(2.4, 2.5),   # x
            np.random.uniform(0.01, 0.02),  # y
            np.random.uniform(1, 1.1)    # z
        ]
        
        self.obstacle_pos = [
            np.random.uniform(1.3, 1.5),   # x
            np.random.uniform(-0.15, 0.15),  # y
            np.random.uniform(0.85, 1.1)    # z 
        ]
        
        self.create_target_and_obstacles()
    def setup_complex_scene(self):
        self.target_pos = [
            np.random.uniform(2.6, 2.7),
            np.random.uniform(0.01, 0.02),
            np.random.uniform(1, 1.1)
        ]
        
        self.obstacle_pos = [
            np.random.uniform(1.2, 1.3),
            np.random.uniform(-0.15, 0.15),
            np.random.uniform(0.85, 1.1)
        ]
        
        self.obstacle2_pos = [
            np.random.uniform(1.5, 1.7),
            np.random.uniform(-0.25, 0.25),
            np.random.uniform(0.8, 1.2)
        ]
        
        self.create_target_and_obstacles()  
        
    def create_target_and_obstacles(self):
        target_visual = self.p.createVisualShape(
            self.p.GEOM_SPHERE, 
            radius=self.target_radius * 2,
            rgbaColor=[0, 1, 0, 1]
        )
        self.target_id = self.p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=self.target_pos
        )
        
        obstacle_visual = self.p.createVisualShape(
            self.p.GEOM_SPHERE,
            radius=self.obstacle_radius,
            rgbaColor=[0.5, 0.5, 0.5, 1]
        )
        obstacle_collision = self.p.createCollisionShape(
            self.p.GEOM_SPHERE,
            radius=self.obstacle_radius
        )
        self.obstacle_id = self.p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=obstacle_visual,
            baseCollisionShapeIndex=obstacle_collision,
            basePosition=self.obstacle_pos
        )
        
        if self.complex_scene and self.obstacle2_pos is not None:
            obstacle2_visual = self.p.createVisualShape(
                self.p.GEOM_SPHERE,
                radius=self.obstacle_radius,
                rgbaColor=[0.7, 0.4, 0.4, 1]  
            )
            obstacle2_collision = self.p.createCollisionShape(
                self.p.GEOM_SPHERE,
                radius=self.obstacle_radius
            )
            self.obstacle2_id = self.p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=obstacle2_visual,
                baseCollisionShapeIndex=obstacle2_collision,
                basePosition=self.obstacle2_pos
            )
    
    def create_channel_walls(self):
        wall_height = 0.6
        wall_thickness = 0.1
        wall_length = 3.0
        
        # 左墙 
        left_wall_pos = [0.5, self.channel_width/2 + wall_thickness/2, wall_height/2]
        left_wall_visual = self.p.createVisualShape(
            self.p.GEOM_BOX,
            halfExtents=[wall_length/2, wall_thickness/2, wall_height/2],  
            rgbaColor=[0.8, 0.8, 0.8, 0.7]
        )
        left_wall_collision = self.p.createCollisionShape(
            self.p.GEOM_BOX,
            halfExtents=[wall_length/2, wall_thickness/2, wall_height/2]
        )
        left_wall_id = self.p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=left_wall_visual,
            baseCollisionShapeIndex=left_wall_collision,
            basePosition=left_wall_pos
        )
        
        # 右墙
        right_wall_pos = [0.5, -self.channel_width/2 - wall_thickness/2, wall_height/2]
        right_wall_visual = self.p.createVisualShape(
            self.p.GEOM_BOX,
            halfExtents=[wall_length/2, wall_thickness/2, wall_height/2],
            rgbaColor=[0.8, 0.8, 0.8, 0.7]
        )
        right_wall_collision = self.p.createCollisionShape(
            self.p.GEOM_BOX,
            halfExtents=[wall_length/2, wall_thickness/2, wall_height/2]
        )
        right_wall_id = self.p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=right_wall_visual,
            baseCollisionShapeIndex=right_wall_collision,
            basePosition=right_wall_pos
        )
        
        self.wall_ids = [left_wall_id, right_wall_id]
    
    def check_environment_stability(self):
        try:
            if not self.p.isConnected():
                return False, "PyBullet连接断开"
            try:
                self.p.getBasePositionAndOrientation(self.robot_id)
            except:
                return False, "机器人对象丢失"
            for joint_idx in self.arm_joint_indices:
                if joint_idx is not None:
                    try:
                        joint_state = self.p.getJointState(self.robot_id, joint_idx)
                        if any(np.isnan(joint_state[:2])):  
                            return False, f"关节{joint_idx}状态异常"
                    except:
                        return False, f"无法获取关节{joint_idx}状态"
            
            return True, "环境正常"
            
        except Exception as e:
            return False, f"环境检查异常: {e}"
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_num = 0
        self.terminated = False
        self.truncated = False

        if self.target_id is not None:
            self.p.removeBody(self.target_id)
        if self.obstacle_id is not None:
            self.p.removeBody(self.obstacle_id)
        if self.obstacle2_id is not None: 
            self.p.removeBody(self.obstacle2_id)
            self.obstacle2_id = None
            self.obstacle2_pos = None
        
        self.setup_scene()
        self.reset_robot_pose()
        self.get_joint_based_apf_approximation()
        
        for _ in range(50):
            self.p.stepSimulation()
        
        observation = self.get_observation()
        
        info = {
            'is_success': False,
            'reward': 0,
            'step_num': 0
        }
        
        return observation, info
    
    def reset_robot_pose(self):
        base_pos = [0, 0, 0.125]
        base_orn = self.p.getQuaternionFromEuler([0, 0, 0])
        self.p.resetBasePositionAndOrientation(self.robot_id, base_pos, base_orn)
        initial_angles = [0.0, -2.2, 1.57, -0.5, 0.0, 0.0]  
        for i, angle in enumerate(initial_angles):
            if i < len(self.arm_joint_indices) and self.arm_joint_indices[i] is not None:
                self.p.resetJointState(self.robot_id, self.arm_joint_indices[i], angle)
    
    def get_joint_based_apf_approximation(self):
        current_pose = np.zeros(6)
        for i, joint_idx in enumerate(self.arm_joint_indices):
            if self.complex_scene and self.obstacle2_pos is not None:
                end_pos = self.get_end_effector_position()
                dist1 = np.linalg.norm(np.array(end_pos) - np.array(self.obstacle_pos))
                dist2 = np.linalg.norm(np.array(end_pos) - np.array(self.obstacle2_pos))
                
                primary_obstacle = self.obstacle_pos if dist1 <= dist2 else self.obstacle2_pos
            else:
                primary_obstacle = self.obstacle_pos
        
        apf_result = self.apf_planner.plan(
            current_pose,           
            self.target_pos,       
            primary_obstacle       
        )
        
        self.apf_target_pose = np.zeros(6)
        self.apf_target_pose[:4] = apf_result  
        self.apf_target_pose[4:] = current_pose[4:]  
    
    def step(self, action):
        info = {}
        try:
            is_stable, msg = self.check_environment_stability()
            if not is_stable:
                print(f"环境不稳定: {msg}")
                return (np.zeros(29, dtype=np.float32), -1000, True, False, 
                        {'error': msg, 'step_num': self.step_num})
       
            self.get_joint_based_apf_approximation()

            delta_joints = np.clip(action, -0.15, 0.15)  
            
            guided_action = self.apply_joint_based_guidance_mechanism(action)
        
            self.execute_action(guided_action)
            
            try:
                for _ in range(5):
                    self.p.stepSimulation()
                    if not self.p.isConnected():
                        raise RuntimeError("PyBullet连接断开")
            except Exception as e:
                print(f"仿真步进错误: {e}")
                self.terminated = True
                return (np.zeros(29, dtype=np.float32), -1000, True, False, 
                        {'error': str(e), 'step_num': self.step_num})
            
            observation = self.get_observation()
            reward = self.calculate_reward()

            self.terminated, self.truncated = self.check_done()
            
            self.step_num += 1
            
            info.update({
                'is_success': self.check_success(),
                'collision': self.check_collision(),
                'step_num': self.step_num
            })
            
            return observation, reward, self.terminated, self.truncated, info
            
        except Exception as e:
            print(f"Step执行错误: {e}")
            return (np.zeros(29, dtype=np.float32), -1000, True, False, 
                    {'error': str(e), 'step_num': self.step_num})
    
    def apply_joint_based_guidance_mechanism(self, action):
        current_angles = []
        for i in range(6):
            if i < len(self.arm_joint_indices):
                joint_state = self.p.getJointState(self.robot_id, self.arm_joint_indices[i])
                current_angles.append(joint_state[0])
        
        if len(current_angles) < 4:  
            return action
        
        base_pos, _ = self.p.getBasePositionAndOrientation(self.robot_id)
        
        dt = abs(base_pos[0] - self.target_pos[0])
        
        delta_d_star = abs(self.target_pos[0] - self.obstacle_pos[0])
        
        base_pos, _ = self.p.getBasePositionAndOrientation(self.robot_id)

        base_to_obstacle_dist = np.sqrt(
            (base_pos[0] - self.obstacle_pos[0])**2 +
            (base_pos[1] - self.obstacle_pos[1])**2
        )

        if (dt < delta_d_star or base_to_obstacle_dist<0.5) and self.apf_target_pose is not None:
            self.apf_activation_count += 1
        
            dt_safe = max(dt, 0.001)
            G_dt = self.omega / (self.omega * dt_safe + 1)
            
            guided_action = action.copy()
            
            for i in range(4): 
                if i < len(self.apf_target_pose):
                    delta_q = self.apf_target_pose[i] - current_angles[i]
                    dt_time = 0.1  
                    guidance_gain = 0.18  
                    guided_action[i] += G_dt * delta_q * guidance_gain
            
            self.total_guidance_steps += 1
            
            return guided_action
        
        return action
    
    def execute_action(self, action):
        try:
            current_angles = []
            for joint_idx in self.arm_joint_indices:
                if joint_idx is not None:
                    joint_state = self.p.getJointState(self.robot_id, joint_idx)
                    current_angles.append(joint_state[0])
            
            if len(current_angles) >= 6:
      
                max_delta = 0.1  
         
                new_angles = []
                for i in range(6):
                    if i < len(current_angles):
                        delta = np.clip(action[i], -max_delta, max_delta)
                        new_angles.append(current_angles[i] + delta)
                    else:
                        new_angles.append(0.0)
                
                joint_limits = [
                    [-np.pi, np.pi],      
                    [-2.2, 0],          
                    [-2.8, 2.8],        
                    [-2.2, 2.2],         
                    [-np.pi, np.pi],      
                    [-np.pi, np.pi]       
                ]
                
                for i in range(len(new_angles)):
                    if i < len(joint_limits):
                        new_angles[i] = np.clip(new_angles[i], 
                                              joint_limits[i][0], 
                                              joint_limits[i][1])
                for i, angle in enumerate(new_angles):
                    if i < len(self.arm_joint_indices) and self.arm_joint_indices[i] is not None:
                        self.p.setJointMotorControl2(
                            self.robot_id, 
                            self.arm_joint_indices[i],
                            self.p.POSITION_CONTROL, 
                            targetPosition=angle,
                            force=100,
                            maxVelocity=1.0
                        )
            
            base_pos, base_orn = self.p.getBasePositionAndOrientation(self.robot_id)
            target_x = self.target_pos[0]
            current_x = base_pos[0]
            
            x_diff = target_x - current_x
            
            max_move_per_step = 0.015  
            
            if abs(x_diff) > max_move_per_step:
                move_distance = max_move_per_step if x_diff > 0 else -max_move_per_step
            else:
                move_distance = x_diff
            
            new_x = current_x + move_distance
            
            new_x = np.clip(new_x, -0.5, 2.5)  
            
            new_base_pos = [new_x, base_pos[1], base_pos[2]]
            self.p.resetBasePositionAndOrientation(self.robot_id, new_base_pos, base_orn)
        
        except Exception as e:
            print(f"动作执行错误: {e}")

    def get_obstacle_position(self):
        try:
            if hasattr(self, 'obstacle_pos') and self.obstacle_pos is not None:
                return self.obstacle_pos
            elif hasattr(self, 'obstacle_id') and self.obstacle_id is not None:
                try:
                    obstacle_pos, _ = self.p.getBasePositionAndOrientation(self.obstacle_id)
                    return list(obstacle_pos)
                except Exception as e:
                    print(f"从PyBullet获取障碍物位置失败: {e}")
                    return [0.5, 0.0, 0.8]
            
            else:
                print("警告: 无法获取障碍物位置，使用默认位置")
                return [0.5, 0.0, 0.8]
                
        except Exception as e:
            print(f"获取障碍物位置时发生错误: {e}")
            return [0.5, 0.0, 0.8]
    
    def get_end_effector_position(self):
        try:
            if self.end_effector_index is not None:
                link_state = self.p.getLinkState(self.robot_id, self.end_effector_index)
                end_pos = list(link_state[0])  
                return end_pos
            else:
                base_pos, _ = self.p.getBasePositionAndOrientation(self.robot_id)
                return list(base_pos)
        except Exception as e:
            print(f"获取末端执行器位置错误: {e}")
            return [0, 0, 0]
    
    def get_observation(self):
        obs = np.zeros(29, dtype=np.float32)
        
        try:
            base_pos, base_orn = self.p.getBasePositionAndOrientation(self.robot_id)
            
            joint_states = []
            for joint_idx in self.arm_joint_indices:
                if joint_idx is not None:
                    joint_state = self.p.getJointState(self.robot_id, joint_idx)
                    joint_states.append(joint_state[0])
            
            end_pos = self.get_end_effector_position()
            
            idx = 0
            
            for i in range(6):
                obs[idx] = joint_states[i] if i < len(joint_states) else 0.0
                idx += 1
            for i in range(6):
                obs[idx] = joint_states[i] if i < len(joint_states) else 0.0  
                idx += 1
            
            for i in range(6):
                if i < 3:
                    dy = end_pos[1] - self.target_pos[1]
                    dz = end_pos[2] - self.target_pos[2]
                    obs[idx] = np.sqrt(dy**2 + dz**2)
                else:  
                    dy = end_pos[1] - self.obstacle_pos[1] 
                    dz = end_pos[2] - self.obstacle_pos[2]
                    obs[idx] = np.sqrt(dy**2 + dz**2)
                idx += 1
            
            obs[idx] = abs(base_pos[0] - self.target_pos[0])
            idx += 1
            obs[idx] = abs(base_pos[0] - self.obstacle_pos[0])
            idx += 1
            
            obs[idx] = 1.0 if self.check_success() else 0.0  # freach
            idx += 1
            obs[idx] = 1.0 if self.check_collision() else 0.0  # fcollision
            idx += 1
            obs[idx] = 1.0 if self.obstacle_pos[1] > 0 else 0.0  # fobs
            idx += 1
            
            while idx < 29:
                obs[idx] = 0.0
                idx += 1
                
        except Exception as e:
            print(f"获取观测错误: {e}")
            obs = np.zeros(29, dtype=np.float32)
        
        return obs
    
    def calculate_reward(self):
        try:
            from reward import calculate_apf_sac_reward
            total_reward, info = calculate_apf_sac_reward(self)
            return total_reward
        except ImportError:
            return self.calculate_builtin_reward()
    
    def check_collision(self):
        try:
            collision_threshold = 0.01
            

            if self.obstacle_id is not None:
                contact_points = self.p.getContactPoints(self.robot_id, self.obstacle_id)
                for contact in contact_points:
                    if contact[8] < -collision_threshold:
                        return True
            
            if self.complex_scene and self.obstacle2_id is not None:
                contact_points = self.p.getContactPoints(self.robot_id, self.obstacle2_id)
                for contact in contact_points:
                    if contact[8] < -collision_threshold:
                        return True
            
            for wall_id in self.wall_ids:
                contact_points = self.p.getContactPoints(self.robot_id, wall_id)
                for contact in contact_points:
                    if contact[8] < -collision_threshold:
                        return True
            
            return False
            
        except Exception as e:
            print(f"碰撞检测错误: {e}")
            return False
    def check_success(self):
        try:
            end_pos = self.get_end_effector_position()
            target_dist = np.linalg.norm(np.array(end_pos) - np.array(self.target_pos))
            success = target_dist < self.d_min
            if success:
                print(f"成功！末端位置: {end_pos}, 目标位置: {self.target_pos}, 距离: {target_dist:.4f}")
            return success
        except Exception as e:
            print(f"成功检测错误: {e}")
            return False
    
    def check_done(self):
        terminated = False
        truncated = False
        
        try:
            if self.check_success():
                terminated = True
                return terminated, truncated  
            
            if self.check_collision():
                terminated = True
                return terminated, truncated
            
            if self.step_num >= self.max_steps:
                truncated = True
                
        except Exception as e:
            print(f"终止检测错误: {e}")
            terminated = True 
        
        return terminated, truncated
    
    def render(self):
        try:
            self.p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.5, 0, 0.5]
            )
        except Exception as e:
            print(f"渲染错误: {e}")
    
    def close(self):
        try:
            self.p.disconnect()
        except Exception as e:
            print(f"关闭环境错误: {e}")

# 测试环境
if __name__ == "__main__":
    try:
        env = MobileManipulatorEnv(gui=True, fixed_scene=True, channel_width=1.2)
        
        print("基于关节势能作用点的6-DOF环境初始化完成")
        print(f"动作空间: {env.action_space} (6维)")
        print(f"状态空间: {env.observation_space}")
        print(f"使用的关节数量: {len(env.arm_joint_indices)}")
        print("已启用基于关节势能作用点的APF")
        
        obs, info = env.reset()
        print(f"初始观测维度: {obs.shape}")
        print(f"目标位置: {env.target_pos}")
        print(f"障碍物位置: {env.obstacle_pos}")
        print(f"初始末端位置: {env.get_end_effector_position()}")
        
        current_pose = [0, -2.2, 1.57, -0.5, 0.0, 0.0]
        end_pos, joint_positions = env.apf_planner.forward_kinematics_with_joints(current_pose[:3])
        print(f"末端执行器位置: [{end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f}]")
        print(f"关节势能作用点数量: {len(joint_positions)}")
        print(f"  关节位置: {[f'[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]' for p in joint_positions]}")
        
        env.render()
        
        for i in range(100):
            is_stable, msg = env.check_environment_stability()
            if not is_stable:
                print(f"环境不稳定: {msg}")
                break
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            success = info.get('is_success', False)
            collision = info.get('collision', False)
            
            if i % 10 == 0:
                end_pos = env.get_end_effector_position()
                target_dist = np.linalg.norm(np.array(end_pos) - np.array(env.target_pos))

                current_angles = [env.p.getJointState(env.robot_id, idx)[0] for idx in env.arm_joint_indices[:3]]
                end_pos_calc, joint_positions = env.apf_planner.forward_kinematics_with_joints(current_angles)
                
                min_obs_dist = np.linalg.norm(np.array(end_pos_calc) - np.array(env.obstacle_pos))
                for joint_pos in joint_positions:
                    joint_dist = np.linalg.norm(np.array(joint_pos) - np.array(env.obstacle_pos))
                    min_obs_dist = min(min_obs_dist, joint_dist)
                
                print(f"Step {i}: 6D动作={[f'{a:.2f}' for a in action]}, "
                      f"reward={reward:.3f}, target_dist={target_dist:.4f}, "
                      f"min_obs_dist={min_obs_dist:.4f}, success={success}, collision={collision}")
            
            if terminated or truncated:
                result = "成功到达" if success else ("碰撞" if collision else "超时")
                print(f"环境结束 ({result}): terminated={terminated}, truncated={truncated}")
                obs, info = env.reset()
                print("环境重置")
            
            time.sleep(0.1)
        
        print("6-DOF测试完成，准备关闭...")
        time.sleep(2)
        env.close()
        
    except Exception as e:
        print(f"6-DOF测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
