'''
@Author: HJH-BotMstr
@Date: 2025-10-31
@Description: An ASCF Path Planning Method for a Hybrid Robot in Coating Inspection Scenarios
'''
import gymnasium as gym
import numpy as np
import pybullet as p
import math
import time
from loguru import logger

class SACRewardConfig:
    C_TIME = 1.0        
    C_COLLISION = 250.0   
    C_REACH = 150.0     
    ZETA = 10.0         
    D_MIN = 0.1         
    D_SAFE = 0.15      
    D_CRITICAL = 0.08   
    MAX_STEPS = 200     
    COLLISION_THRESHOLD = 0.01  
    C_APF_CONSISTENCY = 30.0   
    C_BARRIER = 100.0           
    C_SMOOTH = 50.0             
    C_PROGRESS = 40.0           
    C_AVOIDANCE = 50.0          
    KAPPA = 5.0                
    RHO_0 = 0.12            

def calculate_apf_sac_reward(env):
    info = {}
    end_pos = get_end_effector_position(env)
    target_pos = get_target_position(env)
    obstacle_pos = get_obstacle_position(env)
    r_time = calculate_time_penalty()
    r_dis = calculate_distance_penalty(end_pos, target_pos)
    r_collision = calculate_collision_penalty(env)
    r_reach = calculate_reach_reward(end_pos, target_pos)
    r_apf_consistency = calculate_apf_consistency_reward(env, end_pos, target_pos, obstacle_pos)
    r_barrier = calculate_barrier_reward(end_pos, obstacle_pos)
    r_smooth = calculate_smoothness_reward(env)
    r_progress = calculate_progress_reward(env, end_pos, target_pos)
    r_avoidance = calculate_enhanced_avoidance_reward(end_pos, obstacle_pos)
    r_posture = calculate_posture_stability_reward(env)
    total_reward = (r_time + r_dis + r_collision + r_reach + 
                   r_apf_consistency + r_barrier + r_smooth + 
                   r_progress + r_avoidance + r_posture)
    
    success = judge_success(end_pos, target_pos)
    collision = judge_collision(env)
    timeout = judge_timeout(env)
    apf_activated = is_apf_activated(end_pos, obstacle_pos)
    if success:
        env.terminated = True
        env.success = True
    elif collision:
        env.terminated = True
        env.success = False
    elif timeout:
        env.truncated = True
        env.success = False
    info.update({
        'total_reward': total_reward,
        'time_penalty': r_time,
        'distance_penalty': r_dis, 
        'collision_penalty': r_collision,
        'reach_reward': r_reach,
        'apf_consistency_reward': r_apf_consistency,
        'barrier_reward': r_barrier,
        'smoothness_reward': r_smooth,
        'progress_reward': r_progress,
        'avoidance_reward': r_avoidance,
        'posture_reward': r_posture,
        'is_success': success,
        'collision': collision,
        'timeout': timeout,
        'apf_activated': apf_activated,
        'step_num': env.step_num if hasattr(env, 'step_num') else 0,
        'distance_to_target': get_distance_to_target(end_pos, target_pos),
        'distance_to_obstacle': np.linalg.norm(np.array(end_pos) - np.array(obstacle_pos)),
        'end_effector_pos': end_pos,
        'target_pos': target_pos,
    })
    return total_reward, info

def calculate_apf_consistency_reward(env, end_pos, target_pos, obstacle_pos):
    try:
        if hasattr(env, 'last_action') and env.last_action is not None:
            action = env.last_action
        else:
            return 0.0  
        
        apf_gradient = calculate_apf_gradient(env, end_pos, target_pos, obstacle_pos)
        
        if np.linalg.norm(action) < 1e-6 or np.linalg.norm(apf_gradient) < 1e-6:
            return 0.0
        
        cosine_similarity = np.dot(action, apf_gradient) / (
            np.linalg.norm(action) * np.linalg.norm(apf_gradient)
        )
        
        consistency_reward = SACRewardConfig.C_APF_CONSISTENCY * max(0, cosine_similarity)
        
        return consistency_reward
        
    except Exception as e:
        logger.warning(f"APF一致性奖励计算失败: {e}")
        return 0.0

def calculate_apf_gradient(env, end_pos, target_pos, obstacle_pos):
    try:
        if not hasattr(env, 'apf_planner'):
            return np.zeros(6)
        
        current_angles = []
        for i in range(6):
            if i < len(env.arm_joint_indices) and env.arm_joint_indices[i] is not None:
                joint_state = env.p.getJointState(env.robot_id, env.arm_joint_indices[i])
                current_angles.append(joint_state[0])
            else:
                current_angles.append(0.0)
        
        gradient = np.zeros(6)
        delta = 0.01 
        
        for i in range(6):
            pose_forward = current_angles.copy()
            pose_forward[i] += delta
            
            pose_backward = current_angles.copy() 
            pose_backward[i] -= delta
            u_forward = env.apf_planner.calculate_joint_based_potential(
                pose_forward, target_pos, obstacle_pos
            )
            u_backward = env.apf_planner.calculate_joint_based_potential(
                pose_backward, target_pos, obstacle_pos
            )
            gradient[i] = -(u_forward - u_backward) / (2 * delta)
        
        return gradient
        
    except Exception as e:
        logger.warning(f"APF梯度计算失败: {e}")
        return np.zeros(6)

def calculate_barrier_reward(end_pos, obstacle_pos):
    rho = np.linalg.norm(np.array(end_pos) - np.array(obstacle_pos))
    
    barrier_value = 1 - np.tanh(SACRewardConfig.KAPPA * (rho - SACRewardConfig.RHO_0))
    barrier_reward = -SACRewardConfig.C_BARRIER * max(0, barrier_value)
    
    return barrier_reward

def calculate_smoothness_reward(env):
    try:
        if hasattr(env, 'last_action') and env.last_action is not None:
            action_magnitude = np.linalg.norm(env.last_action)
            smoothness_reward = -SACRewardConfig.C_SMOOTH * action_magnitude**2
            return smoothness_reward
        return 0.0
    except:
        return 0.0

def calculate_progress_reward(env, end_pos, target_pos):
    try:
        current_dist = get_distance_to_target(end_pos, target_pos)
        
        if hasattr(env, 'last_distance_to_target'):
            previous_dist = env.last_distance_to_target
            progress = previous_dist - current_dist  
            progress_reward = SACRewardConfig.C_PROGRESS * progress
        else:
            progress_reward = 0.0
        
        env.last_distance_to_target = current_dist
        
        return progress_reward
        
    except:
        return 0.0

def calculate_enhanced_avoidance_reward(end_pos, obstacle_pos):
    distance = np.linalg.norm(np.array(end_pos) - np.array(obstacle_pos))
    
    if distance < SACRewardConfig.D_CRITICAL:
        reward = -SACRewardConfig.C_AVOIDANCE * 2.0
    elif distance < SACRewardConfig.D_SAFE:
        ratio = (SACRewardConfig.D_SAFE - distance) / (SACRewardConfig.D_SAFE - SACRewardConfig.D_CRITICAL)
        reward = -SACRewardConfig.C_AVOIDANCE * ratio
    elif distance < SACRewardConfig.D_SAFE * 1.5:
        reward = SACRewardConfig.C_AVOIDANCE * 0.3
    else:
        reward = 0
    
    return reward

def is_apf_activated(end_pos, obstacle_pos):
    distance = np.linalg.norm(np.array(end_pos) - np.array(obstacle_pos))
    return distance < SACRewardConfig.D_SAFE

def calculate_time_penalty():
    return -SACRewardConfig.C_TIME

def calculate_distance_penalty(end_pos, target_pos):
    distance_3d = math.sqrt(
        (end_pos[0] - target_pos[0])**2 + 
        (end_pos[1] - target_pos[1])**2 + 
        (end_pos[2] - target_pos[2])**2
    )
    distance_penalty = -SACRewardConfig.ZETA * distance_3d
    return distance_penalty

def calculate_collision_penalty(env):
    if judge_collision(env):
        return -SACRewardConfig.C_COLLISION
    else:
        return 0.0

def calculate_reach_reward(end_pos, target_pos):
    distance = get_distance_to_target(end_pos, target_pos)
    if distance < SACRewardConfig.D_MIN:
        return SACRewardConfig.C_REACH
    else:
        return 0.0

def calculate_posture_stability_reward(env):
    try:
        wrist_velocities = []
        for i in range(3, 6): 
            if i < len(env.arm_joint_indices):
                joint_state = env.p.getJointState(env.robot_id, env.arm_joint_indices[i])
                wrist_velocities.append(abs(joint_state[1])) 
        if wrist_velocities:
            avg_velocity = np.mean(wrist_velocities)
            stability_reward = 5.0 * np.exp(-avg_velocity)  
            return stability_reward
        return 0.0
    except:
        return 0.0

def get_obstacle_position(env):
    try:
        if hasattr(env, 'get_obstacle_position') and callable(env.get_obstacle_position):
            return env.get_obstacle_position()
        
        if hasattr(env, 'obstacle_pos') and env.obstacle_pos is not None:
            return env.obstacle_pos
        elif hasattr(env, 'obstacle_id') and env.obstacle_id is not None:
            obstacle_pos, _ = p.getBasePositionAndOrientation(env.obstacle_id)
            return list(obstacle_pos)
        else:
            logger.warning("无法获取障碍物位置，使用默认位置")
            return [0.5, 0, 0.8]
    except Exception as e:
        logger.error(f"获取障碍物位置失败: {e}")
        return [0.5, 0, 0.8]

def get_end_effector_position(env):
    try:
        if hasattr(env, 'get_end_effector_position') and callable(env.get_end_effector_position):
            return env.get_end_effector_position()
        
        if hasattr(env, 'end_effector_index') and env.end_effector_index is not None:
            link_state = p.getLinkState(env.robot_id, env.end_effector_index)
            return list(link_state[0])
        else:
            base_pos, _ = p.getBasePositionAndOrientation(env.robot_id)
            return list(base_pos)
    except Exception as e:
        logger.error(f"获取末端执行器位置失败: {e}")
        return [0, 0, 0]

def get_target_position(env):
    try:
        if hasattr(env, 'target_pos') and env.target_pos is not None:
            return env.target_pos
        elif hasattr(env, 'target_id') and env.target_id is not None:
            target_pos, _ = p.getBasePositionAndOrientation(env.target_id)
            return list(target_pos)
        else:
            logger.warning("无法获取目标位置，使用默认位置")
            return [0.6, 0, 0.5]
    except Exception as e:
        logger.error(f"获取目标位置失败: {e}")
        return [0.6, 0, 0.5]

def get_distance_to_target(end_pos, target_pos):
    distance = math.sqrt(
        (end_pos[0] - target_pos[0])**2 + 
        (end_pos[1] - target_pos[1])**2 + 
        (end_pos[2] - target_pos[2])**2
    )
    return distance

def judge_success(end_pos, target_pos):
    distance = get_distance_to_target(end_pos, target_pos)
    return distance < SACRewardConfig.D_MIN

def judge_collision(env):
    collision_detected = False
    
    try:
        threshold = SACRewardConfig.COLLISION_THRESHOLD
        
        if hasattr(env, 'obstacle_id') and env.obstacle_id is not None:
            contact_points = p.getContactPoints(env.robot_id, env.obstacle_id)
            for contact in contact_points:
                if contact[8] < -threshold:
                    collision_detected = True
                    break
        
    except Exception as e:
        logger.error(f"碰撞检测失败: {e}")
        
    return collision_detected

def judge_timeout(env):
    """判断是否超时"""
    return hasattr(env, 'step_num') and env.step_num >= SACRewardConfig.MAX_STEPS

def grasp_reward(env):
    return calculate_apf_sac_reward(env)

def test_sac_reward_function():
    print("测试APF-SAC奖励函数（移除课程退火）...")
    time_penalty = calculate_time_penalty()
    print(f"时间惩罚: {time_penalty}")
    end_pos = [0.5, 0.1, 0.4]
    target_pos = [0.6, 0.0, 0.5]
    obstacle_pos = [0.55, 0.05, 0.45]

    distance_penalty = calculate_distance_penalty(end_pos, target_pos)
    print(f"距离惩罚 (3D): {distance_penalty}")
    
    reach_reward = calculate_reach_reward(end_pos, target_pos)
    print(f"到达奖励: {reach_reward}")
    
    barrier_reward = calculate_barrier_reward(end_pos, obstacle_pos)
    print(f"屏障奖励: {barrier_reward}")
    
    avoidance_reward = calculate_enhanced_avoidance_reward(end_pos, obstacle_pos)
    print(f"增强避障奖励: {avoidance_reward}")
    
    apf_activated = is_apf_activated(end_pos, obstacle_pos)
    print(f"APF激活状态: {apf_activated}")

    distance_3d = get_distance_to_target(end_pos, target_pos)
    print(f"3D距离: {distance_3d:.4f}")
    
    print("APF-SAC奖励函数测试完成!")
    print("SAC新增特性（固定权重，无课程退火）:")
    print("1. APF一致性奖励: 鼓励动作与APF梯度一致")
    print("2. 屏障奖励: 连续惩罚函数，避免硬阈值")
    print("3. 平滑奖励: 惩罚动作幅度过大")
    print("4. 进步奖励: 奖励每步向目标靠近")
    print("5. APF激活统计: 监控APF使用频率")
    print("6. 距离动态式引导: 基于距离的动态激活条件")

if __name__ == "__main__":
    test_sac_reward_function()