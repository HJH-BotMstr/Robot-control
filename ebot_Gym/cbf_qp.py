'''
@Author: CBF-QP Implementation based on Paper (Extended to Multiple Control Points)
@Date: 2025-01-XX
@Description: Control Barrier Function with Quadratic Programming for safe control
@Reference: "Reinforcement Learning-Enhanced Control Barrier Functions for Robot Manipulators"
@Extension: Now protects end-effector, elbow, and wrist joints simultaneously
@Fix: Added constraint relaxation to handle infeasible cases
'''

import numpy as np
from scipy.optimize import minimize
import cvxopt
from cvxopt import matrix, solvers
from loguru import logger

solvers.options['show_progress'] = False  # 关闭cvxopt输出
solvers.options['maxiters'] = 100  # 增加最大迭代次数
solvers.options['abstol'] = 1e-7  # 绝对容差
solvers.options['reltol'] = 1e-6  # 相对容差
solvers.options['feastol'] = 1e-7  # 可行性容差

class CBF_QP:
    """
    基于论文的完整CBF-QP实现 - 扩展到多控制点
    论文Section 3: CBF Filter Design
    扩展: 同时保护末端执行器、肘部、腕部三个关键点
    修复: 添加松弛变量处理不可行约束
    """
    def __init__(self, safety_radius=0.15, kappa1=1.0, kappa2=1.0, 
                 obstacle_radius=0.1, end_effector_radius=0.05):
        """
        初始化CBF-QP参数
        
        Args:
            safety_radius: 安全间距 r_pad (论文公式10)
            kappa1: CBF参数 κ̂1 (论文公式17)
            kappa2: CBF参数 κ̂2 (论文公式17)
            obstacle_radius: 障碍物半径 r_o
            end_effector_radius: 末端执行器半径 r_ee
        """
        self.r_o = obstacle_radius
        self.r_ee = end_effector_radius
        self.r_pad = safety_radius
        
        # 论文公式(10): r_m = r_o + r_ee + r_pad
        self.r_m = self.r_o + self.r_ee + self.r_pad
        
        # 论文公式(17)中的CBF参数
        self.kappa1 = kappa1  # κ̂1
        self.kappa2 = kappa2  # κ̂2
        
        logger.info(f"CBF-QP初始化 (多控制点): r_m={self.r_m:.3f}, κ1={kappa1}, κ2={kappa2}")
        logger.info("保护点: 末端执行器 + 肘部 + 腕部")
    
    def compute_safety_function(self, control_point_pos, obstacle_pos):
        """
        计算安全函数 h(x) - 论文公式(11)
        h(x) = (x_cp - x_o)² + (y_cp - y_o)² + (z_cp - z_o)² - r_m² ≥ 0
        
        Args:
            control_point_pos: 控制点位置 [x, y, z] (可以是末端/肘部/腕部)
            obstacle_pos: 障碍物位置 [x_o, y_o, z_o]
        
        Returns:
            h: 安全函数值
            sep: 分离向量 [x_cp-x_o, y_cp-y_o, z_cp-z_o]
        """
        sep = np.array(control_point_pos) - np.array(obstacle_pos)
        h = np.dot(sep, sep) - self.r_m**2
        return h, sep
    
    def compute_lie_derivatives(self, sep, control_point_vel, control_point_acc):
        """
        计算Lie导数 - 论文公式(12)(13)
        
        L_f h(x) = 2(sep_x)ẋ + 2(sep_y)ẏ + 2(sep_z)ż  (公式12)
        L²_f h(x) = 2(sep_x)ẍ + 2ẋ² + 2(sep_y)ÿ + 2ẏ² + 2(sep_z)z̈ + 2ż²  (公式13)
        
        Args:
            sep: 分离向量 [sep_x, sep_y, sep_z]
            control_point_vel: 控制点速度 [ẋ, ẏ, ż]
            control_point_acc: 控制点加速度 [ẍ, ÿ, z̈]
        
        Returns:
            Lf_h: 一阶Lie导数
            L2f_h_nom: 二阶Lie导数的名义部分
            S_n: 向量 [2sep_x, 2sep_y, 2sep_z]
        """
        # 论文公式(12)
        S_n = 2 * sep  # [2sep_x, 2sep_y, 2sep_z]
        Lf_h = np.dot(S_n, control_point_vel)
        
        # 论文公式(13) - 名义控制器部分
        Gamma_1 = 2 * (control_point_vel[0]**2 + control_point_vel[1]**2 + control_point_vel[2]**2)
        L2f_h_nom = np.dot(S_n, control_point_acc) + Gamma_1
        
        return Lf_h, L2f_h_nom, S_n
    
    def formulate_qp_single_point(self, tau_nom, S_n, J_cp, M_inv, C, D, G, q_dot, 
                                   L2f_h_nom, Lf_h, h):
        """
        为单个控制点构建CBF-QP约束 - 论文公式(14)-(18)
        
        目标: 最小化 ||τ_qp||²
        约束: -L²_f h_qp ≤ L²_f h_nom + κ̂2·L_f h + κ̂1·h
        
        其中 L²_f h_qp = S_n·J_cp·M^{-1}·(-τ_qp)  (公式16)
        
        Args:
            J_cp: 控制点的Jacobian矩阵 (3×6)
        
        Returns:
            A: QP约束矩阵 (1×6)
            b: QP约束向量 (标量)
        """
        try:
            # 论文公式(14): 计算 Σ_CDG = Cq̇ + Dq̇ + Gq
            Sigma_CDG = C + D + G
            
            # 论文公式(15): Γ_3 = Γ_2 + S_n(J_cp·M^{-1}(τ_nom - Σ_CDG))
            Gamma_3 = L2f_h_nom + np.dot(S_n, np.dot(J_cp, np.dot(M_inv, tau_nom - Sigma_CDG)))
            
            # 论文公式(16): A = S_n·J_cp·M^{-1}
            A = np.dot(S_n, np.dot(J_cp, M_inv))
            
            # 论文公式(17-18): 构建约束 A·τ_qp ≤ b
            # -L²_f h_qp ≤ L²_f h_nom + κ̂2·L_f h + κ̂1·h
            # 即: A·τ_qp ≤ Γ_3 + κ̂2·L_f h + κ̂1·h
            b = Gamma_3 + self.kappa2 * Lf_h + self.kappa1 * h
            
            return A, b
                
        except Exception as e:
            logger.error(f"单点CBF-QP约束构建失败: {e}")
            return None, None
    
    def formulate_qp_multi_point(self, tau_nom, control_points_data, M_inv, C, D, G, q_dot):
        """
        为多个控制点构建联合CBF-QP优化问题 - 带松弛变量
        
        目标: 最小化 ||τ_qp||² + ρ||ξ||² (带松弛变量)
        约束: 对每个控制点 i，-L²_f h_qp^i ≤ L²_f h_nom^i + κ̂2·L_f h^i + κ̂1·h^i + ξ_i
              ξ_i ≥ 0 (松弛变量非负)
        
        Args:
            tau_nom: 名义控制器输出 (6维)
            control_points_data: 包含多个控制点数据的列表
            M_inv: 质量矩阵的逆 (6×6)
            C: Coriolis力 (6维)
            D: 摩擦力 (6维)
            G: 重力 (6维)
            q_dot: 关节速度 (6维)
        
        Returns:
            tau_qp: QP求解得到的安全修正力矩
        """
        try:
            n_joints = tau_nom.shape[0]
            n_constraints = len(control_points_data)
            
            # 为每个控制点构建CBF约束
            A_list = []
            b_list = []
            
            for cp_data in control_points_data:
                # 计算Lie导数
                Lf_h, L2f_h_nom, S_n = self.compute_lie_derivatives(
                    cp_data['sep'], 
                    cp_data['vel'], 
                    cp_data['acc']
                )
                
                # 构建单个控制点的约束
                A_i, b_i = self.formulate_qp_single_point(
                    tau_nom, S_n, cp_data['J'], M_inv, C, D, G, q_dot,
                    L2f_h_nom, Lf_h, cp_data['h']
                )
                
                if A_i is not None and b_i is not None:
                    A_list.append(A_i)
                    b_list.append(b_i)
            
            if len(A_list) == 0:
                logger.warning("没有有效的CBF约束")
                return np.zeros(n_joints)
            
            # ===== 关键修复: 添加松弛变量处理不可行约束 =====
            # 变量: [τ_qp; ξ] 其中 ξ 是松弛变量
            # minimize: (1/2)||τ_qp||² + (ρ/2)||ξ||²
            # subject to: A·τ_qp - ξ ≤ b, ξ ≥ 0
            
            n_slack = len(A_list)
            n_vars = n_joints + n_slack
            
            # 松弛变量惩罚系数 (大值鼓励可行解)
            rho = 1e4  # 强惩罚，优先满足约束
            
            # P矩阵: diag([1, 1, ..., ρ, ρ, ...])
            P_np = np.zeros((n_vars, n_vars))
            P_np[:n_joints, :n_joints] = np.eye(n_joints)
            P_np[n_joints:, n_joints:] = rho * np.eye(n_slack)
            P = matrix(P_np, tc='d')
            
            # q向量: 全零
            q = matrix(np.zeros(n_vars), tc='d')
            
            # 不等式约束: [A, -I]·[τ_qp; ξ] ≤ b
            G_np = np.zeros((n_slack, n_vars))
            G_np[:, :n_joints] = np.vstack(A_list)
            G_np[:, n_joints:] = -np.eye(n_slack)
            
            # 添加松弛变量非负约束: -ξ ≤ 0
            G_slack = np.zeros((n_slack, n_vars))
            G_slack[:, n_joints:] = -np.eye(n_slack)
            
            G_combined = np.vstack([G_np, G_slack])
            h_combined = np.hstack([b_list, np.zeros(n_slack)])
            
            G = matrix(G_combined, tc='d')
            h_cvx = matrix(h_combined, tc='d')
            
            # 求解QP
            sol = solvers.qp(P, q, G, h_cvx)
            
            if sol['status'] == 'optimal':
                solution = np.array(sol['x']).flatten()
                tau_qp = solution[:n_joints]
                slack = solution[n_joints:]
                
                # 检查松弛变量是否被激活 (调试用)
                max_slack = np.max(np.abs(slack))
                if max_slack > 1e-3:
                    #logger.debug(f"松弛变量激活: max(ξ)={max_slack:.4f} (约束可能冲突)")
                    pass
                return tau_qp
                
            elif sol['status'] in ['primal infeasible', 'dual infeasible', 'unknown']:
                # 如果带松弛变量仍然失败，尝试只用最严格的约束
                logger.warning(f"带松弛变量的QP失败 ({sol['status']})，尝试单约束降级")
                return self._fallback_single_constraint(A_list, b_list, n_joints)
            else:
                logger.warning(f"多点CBF-QP求解失败: {sol['status']}")
                return np.zeros(n_joints)
                
        except Exception as e:
            logger.error(f"多点CBF-QP构建失败: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(tau_nom.shape[0])
    
    def _fallback_single_constraint(self, A_list, b_list, n_joints):
        """
        降级策略: 只使用最严格的单个约束
        当多约束QP不可行时使用
        """
        try:
            # 找到最严格的约束 (最小的b值)
            min_idx = np.argmin(b_list)
            A_single = A_list[min_idx].reshape(1, -1)
            b_single = [b_list[min_idx]]
            
            # 求解单约束QP
            P = matrix(np.eye(n_joints), tc='d')
            q = matrix(np.zeros(n_joints), tc='d')
            G = matrix(A_single, tc='d')
            h_cvx = matrix(b_single, tc='d')
            
            sol = solvers.qp(P, q, G, h_cvx)
            
            if sol['status'] == 'optimal':
                tau_qp = np.array(sol['x']).flatten()
                logger.debug(f"降级成功: 使用约束 {min_idx}")
                return tau_qp
            else:
                logger.warning("降级策略也失败，返回零修正")
                return np.zeros(n_joints)
                
        except Exception as e:
            logger.error(f"降级策略失败: {e}")
            return np.zeros(n_joints)
    
    def compute_safe_control_multi_point(self, tau_nom, control_points_states, obstacle_pos,
                                         J_dict, M_inv, C, D, G, q_dot):
        """
        计算安全控制输入 - 多控制点版本
        
        论文公式: τ = τ_nom - τ_qp
        
        Args:
            tau_nom: 名义控制器输出 (6维)
            control_points_states: 字典，包含多个控制点的状态
            obstacle_pos: 障碍物位置
            J_dict: 各控制点的Jacobian矩阵字典
            M_inv: 质量矩阵的逆
            C, D, G: 动力学项
            q_dot: 关节速度
        
        Returns:
            tau_safe: 安全控制输入
            cbf_active: CBF是否激活
            active_points: 激活CBF的控制点列表
        """
        # 1. 检查每个控制点的安全函数
        control_points_data = []
        active_points = []
        max_h = -np.inf
        
        for cp_name, cp_state in control_points_states.items():
            # 计算安全函数 h(x) - 论文公式(11)
            h, sep = self.compute_safety_function(cp_state['pos'], obstacle_pos)
            max_h = max(max_h, h)
            
            # 如果距离足够远，跳过此控制点
            if h > self.r_m * 3:
                continue
            
            # 收集需要保护的控制点数据
            control_points_data.append({
                'J': J_dict[cp_name],
                'sep': sep,
                'vel': cp_state['vel'],
                'acc': cp_state['acc'],
                'h': h,
                'name': cp_name
            })
            active_points.append(cp_name)
        
        # 2. 如果所有控制点都足够远，不需要CBF介入
        if len(control_points_data) == 0:
            return tau_nom, False, []
        
        # 3. 构建并求解多点QP - 论文公式(14)-(18)
        tau_qp = self.formulate_qp_multi_point(
            tau_nom, control_points_data, M_inv, C, D, G, q_dot
        )
        
        # 4. 计算最终安全控制 - 论文公式(15)
        tau_safe = tau_nom - tau_qp
        
        cbf_active = np.linalg.norm(tau_qp) > 1e-3
        
        return tau_safe, cbf_active, active_points
    
    def compute_safe_control(self, tau_nom, end_pos, end_vel, end_acc,
                            obstacle_pos, J, M_inv, C, D, G, q_dot):
        """
        单点版本 - 保持向后兼容性
        仅保护末端执行器
        """
        # 1. 计算安全函数 h(x) - 论文公式(11)
        h, sep = self.compute_safety_function(end_pos, obstacle_pos)
        
        # 如果距离足够远，不需要CBF介入
        if h > self.r_m * 3:
            return tau_nom, False
        
        # 2. 计算Lie导数 - 论文公式(12)(13)
        Lf_h, L2f_h_nom, S_n = self.compute_lie_derivatives(sep, end_vel, end_acc)
        
        # 3. 构建并求解QP - 论文公式(14)-(18)
        A, b = self.formulate_qp_single_point(
            tau_nom, S_n, J, M_inv, C, D, G, q_dot,
            L2f_h_nom, Lf_h, h
        )
        
        if A is None or b is None:
            return tau_nom, False
        
        # 使用cvxopt求解单约束QP
        try:
            n_joints = tau_nom.shape[0]
            P = matrix(np.eye(n_joints), tc='d')
            q = matrix(np.zeros(n_joints), tc='d')
            G = matrix(A.reshape(1, -1), tc='d')
            h_cvx = matrix([b], tc='d')
            
            sol = solvers.qp(P, q, G, h_cvx)
            
            if sol['status'] == 'optimal':
                tau_qp = np.array(sol['x']).flatten()
                tau_safe = tau_nom - tau_qp
                cbf_active = np.linalg.norm(tau_qp) > 1e-3
                return tau_safe, cbf_active
            else:
                return tau_nom, False
        except Exception as e:
            logger.error(f"单点QP求解失败: {e}")
            return tau_nom, False