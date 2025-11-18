'''
@Author: HJH-BotMstr
@Date: 2025-10-31
@Description: An ASCF Path Planning Method for a Hybrid Robot in Coating Inspection Scenarios
'''

import numpy as np
import pybullet as p
from cbf_qp import CBF_QP
from loguru import logger
import time

class CBFSafetyFilter:
    def __init__(self, env, safety_radius=0.15, kappa1=1.0, kappa2=1.0):
        self.env = env
        self.p = env.p
        self.robot_id = env.robot_id
        self.cbf = CBF_QP(
            safety_radius=safety_radius,
            kappa1=kappa1,
            kappa2=kappa2,
            obstacle_radius=env.obstacle_radius,
            end_effector_radius=0.05
        )
        self.control_point_links = {
            'end_effector': env.end_effector_index,  
            'elbow': None,  
            'wrist': None   
        }
        self._find_control_point_links()
        self.last_states = {
            'end_effector': {'pos': None, 'vel': None, 'time': None},
            'elbow': {'pos': None, 'vel': None, 'time': None},
            'wrist': {'pos': None, 'vel': None, 'time': None}
        }

        self.cbf_active = False
        self.h_activate = 0.0    
        self.h_deactivate = 0.03   
        self.cbf_activation_count = 0
        self.total_filter_calls = 0
        self.avg_solve_time = 0.0
        self.point_activation_stats = {
            'end_effector': 0,
            'elbow': 0,
            'wrist': 0
        }
        
        logger.info("="*60)
        logger.info("CBF安全过滤器初始化完成（部署模式 - 多控制点）")
        logger.info(f"  安全半径: {safety_radius} m")
        logger.info(f"  CBF参数: κ1={kappa1}, κ2={kappa2}")
        logger.info(f"  激活阈值: h ≤ {self.h_activate} m")
        logger.info(f"  退出阈值: h > {self.h_deactivate} m")
        logger.info(f"  保护控制点:")
        for cp_name, link_idx in self.control_point_links.items():
            logger.info(f"    - {cp_name}: link_index={link_idx}")
        logger.info("="*60)
    
    def _find_control_point_links(self):
        try:
            num_joints = self.p.getNumJoints(self.robot_id)
            
            for i in range(num_joints):
                joint_info = self.p.getJointInfo(self.robot_id, i)
                link_name = joint_info[12].decode('utf-8').lower()  

                if 'elbow' in link_name or 'upper_arm' in link_name:
                    if self.control_point_links['elbow'] is None:
                        self.control_point_links['elbow'] = i
                        logger.info(f"找到肘部link: {joint_info[12].decode('utf-8')} (index={i})")

                if 'wrist_1' in link_name or 'wrist1' in link_name:
                    if self.control_point_links['wrist'] is None:
                        self.control_point_links['wrist'] = i
                        logger.info(f"找到腕部link: {joint_info[12].decode('utf-8')} (index={i})")
            
            if self.control_point_links['elbow'] is None:
                if len(self.env.arm_joint_indices) >= 3:
                    self.control_point_links['elbow'] = self.env.arm_joint_indices[2]
                    logger.warning(f"使用备用肘部link: index={self.control_point_links['elbow']}")
            
            if self.control_point_links['wrist'] is None:
                if len(self.env.arm_joint_indices) >= 4:
                    self.control_point_links['wrist'] = self.env.arm_joint_indices[3]
                    logger.warning(f"使用备用腕部link: index={self.control_point_links['wrist']}")
                    
        except Exception as e:
            logger.error(f"查找控制点link失败: {e}")
    
    def get_robot_state(self):
        try:

            q = np.zeros(6)
            q_dot = np.zeros(6)
            
            for i, joint_idx in enumerate(self.env.arm_joint_indices[:6]):
                if joint_idx is not None:
                    joint_state = self.p.getJointState(self.robot_id, joint_idx)
                    q[i] = joint_state[0]       
                    q_dot[i] = joint_state[1]   
            current_time = time.time()
            control_points_states = {}
            for cp_name, link_idx in self.control_point_links.items():
                if link_idx is None:
                    control_points_states[cp_name] = {
                        'pos': np.zeros(3),
                        'vel': np.zeros(3),
                        'acc': np.zeros(3)
                    }
                    continue
                link_state = self.p.getLinkState(
                    self.robot_id,
                    link_idx,
                    computeLinkVelocity=1
                )
                
                cp_pos = np.array(link_state[0])  
                cp_vel = np.array(link_state[6])  
                last_state = self.last_states[cp_name]
                if last_state['vel'] is not None and last_state['time'] is not None:
                    dt = current_time - last_state['time']
                    if dt > 0:
                        cp_acc = (cp_vel - last_state['vel']) / dt
                    else:
                       cp_acc = np.zeros(3)
                else:
                    cp_acc = np.zeros(3)
                self.last_states[cp_name]['pos'] = cp_pos.copy()
                self.last_states[cp_name]['vel'] = cp_vel.copy()
                self.last_states[cp_name]['time'] = current_time
                
                control_points_states[cp_name] = {
                    'pos': cp_pos,
                    'vel': cp_vel,
                    'acc': cp_acc
                }
            
            return {
                'q': q,
                'q_dot': q_dot,
                'control_points': control_points_states
            }
            
        except Exception as e:
            logger.error(f"状态获取失败: {e}")
            return None
    
    def compute_dynamics(self, q, q_dot):
        try:
            num_joints = self.p.getNumJoints(self.robot_id)
            full_q = []
            full_q_dot = []
            joint_idx_to_full_q_idx = {}
            
            full_q_counter = 0
            for i in range(num_joints):
                joint_info = self.p.getJointInfo(self.robot_id, i)
                joint_type = joint_info[2]
                
                if joint_type != self.p.JOINT_FIXED:
                    joint_state = self.p.getJointState(self.robot_id, i)
                    full_q.append(joint_state[0])
                    full_q_dot.append(joint_state[1])
                    joint_idx_to_full_q_idx[i] = full_q_counter
                    full_q_counter += 1
            
            if len(full_q) == 0:
                return None
            
            zero_vec = [0.0] * len(full_q)
            J_dict = {}
            for cp_name, link_idx in self.control_point_links.items():
                if link_idx is None:
                    J_dict[cp_name] = np.zeros((3, 6))
                    continue
                
                jac_t, jac_r = self.p.calculateJacobian(
                    self.robot_id, link_idx,
                    localPosition=[0, 0, 0],
                    objPositions=full_q,
                    objVelocities=zero_vec,
                    objAccelerations=zero_vec
                )
                
                jac_full = np.array(jac_t)
                J_arm = np.zeros((3, 6))
                
                for i, urdf_idx in enumerate(self.env.arm_joint_indices[:6]):
                    if urdf_idx in joint_idx_to_full_q_idx:
                        full_q_idx = joint_idx_to_full_q_idx[urdf_idx]
                        if full_q_idx < jac_full.shape[1]:
                            J_arm[:, i] = jac_full[:, full_q_idx]
                
                J_dict[cp_name] = J_arm
            
            M_full = self.p.calculateMassMatrix(self.robot_id, full_q)
            M_full = np.array(M_full)
            
            M = np.zeros((6, 6))
            for i, urdf_idx_i in enumerate(self.env.arm_joint_indices[:6]):
                for j, urdf_idx_j in enumerate(self.env.arm_joint_indices[:6]):
                    if urdf_idx_i in joint_idx_to_full_q_idx and urdf_idx_j in joint_idx_to_full_q_idx:
                        full_q_i = joint_idx_to_full_q_idx[urdf_idx_i]
                        full_q_j = joint_idx_to_full_q_idx[urdf_idx_j]
                        if full_q_i < M_full.shape[0] and full_q_j < M_full.shape[1]:
                            M[i, j] = M_full[full_q_i, full_q_j]

            if np.linalg.cond(M) > 1e10:
                logger.warning(f"质量矩阵条件数过大: {np.linalg.cond(M):.2e}")
                M += np.eye(6) * 1e-6
            
            M_inv = np.linalg.inv(M)
            zero_acc = [0.0] * len(full_q)
            tau_cg_full = self.p.calculateInverseDynamics(
                self.robot_id, full_q, full_q_dot, zero_acc
            )
            
            C_plus_G = np.zeros(6)
            for i, urdf_idx in enumerate(self.env.arm_joint_indices[:6]):
                if urdf_idx in joint_idx_to_full_q_idx:
                    full_q_idx = joint_idx_to_full_q_idx[urdf_idx]
                    if full_q_idx < len(tau_cg_full):
                        C_plus_G[i] = tau_cg_full[full_q_idx]
            
            zero_vel = [0.0] * len(full_q)
            tau_g_full = self.p.calculateInverseDynamics(
                self.robot_id, full_q, zero_vel, zero_acc
            )
            
            G = np.zeros(6)
            for i, urdf_idx in enumerate(self.env.arm_joint_indices[:6]):
                if urdf_idx in joint_idx_to_full_q_idx:
                    full_q_idx = joint_idx_to_full_q_idx[urdf_idx]
                    if full_q_idx < len(tau_g_full):
                        G[i] = tau_g_full[full_q_idx]
            
            C = C_plus_G - G
            D = np.zeros(6)
            
            return {
                'J_dict': J_dict,
                'M': M,
                'M_inv': M_inv,
                'C': C,
                'G': G,
                'D': D
            }
            
        except Exception as e:
            logger.error(f"动力学计算失败: {e}")
            return None
    
    def filter_action(self, tau_nom):
        solve_start = time.time()
        self.total_filter_calls += 1
        
        try:
            state = self.get_robot_state()
            if state is None:
                return tau_nom, {'error': 'state_failure', 'cbf_active': False}
            obstacle_pos = self.env.obstacle_pos
            if self.env.complex_scene and self.env.obstacle2_pos is not None:
                min_dist1 = np.inf
                min_dist2 = np.inf
                end_h, _ = self.cbf.compute_safety_function(
                    state['control_points']['end_effector']['pos'], 
                    obstacle_pos
                )
                if end_h > self.cbf.r_m * 2:  
                    control_points_to_check = ['end_effector']
                else:
                    control_points_to_check = state['control_points'].keys()

                for cp_name in control_points_to_check:
                    cp_state = state['control_points'][cp_name]

                for cp_name, cp_state in state['control_points'].items():
                    dist1 = np.linalg.norm(cp_state['pos'] - np.array(self.env.obstacle_pos))
                    dist2 = np.linalg.norm(cp_state['pos'] - np.array(self.env.obstacle2_pos))
                    min_dist1 = min(min_dist1, dist1)
                    min_dist2 = min(min_dist2, dist2)
                
                obstacle_pos = self.env.obstacle_pos if min_dist1 <= min_dist2 else self.env.obstacle2_pos
            min_h = np.inf
            control_points_h = {}
            for cp_name, cp_state in state['control_points'].items():
                h, sep = self.cbf.compute_safety_function(cp_state['pos'], obstacle_pos)
                control_points_h[cp_name] = h
                min_h = min(min_h, h)
            if not self.cbf_active:
                if min_h <= self.h_activate:
                    self.cbf_active = True
                    self.cbf_activation_count += 1
                    #logger.debug(f"多点CBF激活: min_h={min_h:.4f} ≤ {self.h_activate}, h_values={control_points_h}")
            else:
                if min_h > self.h_deactivate:
                    self.cbf_active = False
            if not self.cbf_active:
                solve_time = time.time() - solve_start
                return tau_nom, {
                    'cbf_active': False,
                    'min_h': min_h,
                    'h_values': control_points_h,
                    'tau_qp_norm': 0.0,
                    'solve_time': solve_time,
                    'active_points': []
                }
            dynamics = self.compute_dynamics(state['q'], state['q_dot'])
            if dynamics is None:
                return tau_nom, {'error': 'dynamics_failure', 'cbf_active': True}
            tau_safe, cbf_activated, active_points = self.cbf.compute_safe_control_multi_point(
                tau_nom=tau_nom,
                control_points_states=state['control_points'],
                obstacle_pos=obstacle_pos,
                J_dict=dynamics['J_dict'],
                M_inv=dynamics['M_inv'],
                C=dynamics['C'],
                D=dynamics['D'],
                G=dynamics['G'],
                q_dot=state['q_dot']
            )
            for cp_name in active_points:
                self.point_activation_stats[cp_name] += 1
            solve_time = time.time() - solve_start
            self.avg_solve_time = (self.avg_solve_time * (self.total_filter_calls - 1) + 
                                  solve_time) / self.total_filter_calls
            
            tau_qp = tau_nom - tau_safe
            tau_qp_norm = np.linalg.norm(tau_qp)
            return tau_safe, {
                'cbf_active': True,
                'min_h': min_h,
                'h_values': control_points_h,
                'tau_qp_norm': tau_qp_norm,
                'solve_time': solve_time,
                'active_points': active_points
            }
            
        except Exception as e:
            logger.error(f"CBF过滤失败: {e}")
            return tau_nom, {'error': str(e), 'cbf_active': self.cbf_active}
    
    def get_statistics(self):
        if self.total_filter_calls == 0:
            return {}
        
        activation_rate = self.cbf_activation_count / self.total_filter_calls * 100
        
        point_activation_rates = {}
        for cp_name, count in self.point_activation_stats.items():
            rate = count / self.total_filter_calls * 100
            point_activation_rates[cp_name] = rate
        
        return {
            'total_calls': self.total_filter_calls,
            'activation_count': self.cbf_activation_count,
            'activation_rate': activation_rate,
            'avg_solve_time_ms': self.avg_solve_time * 1000,
            'current_active': self.cbf_active,
            'point_activation_rates': point_activation_rates,
            'point_activation_counts': self.point_activation_stats.copy()
        }
    
    def reset_statistics(self):
        self.cbf_activation_count = 0
        self.total_filter_calls = 0
        self.avg_solve_time = 0.0
        self.cbf_active = False
        
        for cp_name in self.last_states:
            self.last_states[cp_name] = {'pos': None, 'vel': None, 'time': None}

        for cp_name in self.point_activation_stats:
            self.point_activation_stats[cp_name] = 0


def create_cbf_filter(env, safety_config=None):
    if safety_config is None:
        safety_config = {
            'safety_radius': 0.15,  # r_pad = 0.15 m
            'kappa1': 1.0,          # κ̂1 = 1.0
            'kappa2': 1.0           # κ̂2 = 1.0
        }
    
    return CBFSafetyFilter(
        env=env,
        safety_radius=safety_config['safety_radius'],
        kappa1=safety_config['kappa1'],
        kappa2=safety_config['kappa2']
    )
if __name__ == "__main__":
    from env_test import MobileManipulatorEnv
    from stable_baselines3 import SAC
    
    print("="*60)
    print("CBF安全过滤器使用示例 (多控制点)")
    print("="*60)
    env = MobileManipulatorEnv(gui=True, fixed_scene=False, complex_scene=True)
    model_path = "./models_6dof_sac/apf_sac_6dof_sb3_final_XXXXX.zip"
    try:
        model = SAC.load(model_path)
        print("✓ SAC模型加载成功")
    except:
        print("✗ 模型文件不存在，使用随机动作")
        model = None
    
    cbf_filter = create_cbf_filter(env)
    print("✓ CBF过滤器创建成功 (保护末端/肘部/腕部)")
    
    print("\n开始部署测试（带多点CBF安全保护）...")
    state, _ = env.reset()
    
    for step in range(100):
        if model is not None:
            action, _ = model.predict(state, deterministic=True)
        else:
            action = env.action_space.sample()
        tau_nom = action * 10.0 
        tau_safe, cbf_info = cbf_filter.filter_action(tau_nom)

        state, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0 and cbf_info.get('cbf_active', False):
            active_points_str = ', '.join(cbf_info.get('active_points', []))
            h_values_str = ', '.join([f"{k}={v:.4f}" for k, v in cbf_info.get('h_values', {}).items()])
            print(f"Step {step}: 多点CBF激活")
            print(f"  激活控制点: [{active_points_str}]")
            print(f"  安全函数值: {h_values_str}")
            print(f"  ‖τ_qp‖={cbf_info['tau_qp_norm']:.3f}")
            print(f"  求解时间={cbf_info['solve_time']*1000:.2f}ms")
        
        if terminated or truncated:
            break
    stats = cbf_filter.get_statistics()
    print("\n" + "="*60)
    print("多点CBF统计信息:")
    print(f"  总调用次数: {stats['total_calls']}")
    print(f"  激活次数: {stats['activation_count']}")
    print(f"  激活率: {stats['activation_rate']:.2f}%")
    print(f"  平均求解时间: {stats['avg_solve_time_ms']:.2f} ms")
    print("\n各控制点激活统计:")
    for cp_name, rate in stats['point_activation_rates'].items():
        count = stats['point_activation_counts'][cp_name]
        print(f"  {cp_name}: {count}次 ({rate:.2f}%)")
    print("="*60)
    
    env.close()