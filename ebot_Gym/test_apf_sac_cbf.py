'''
@Author: HJH-BotMstr
@Date: 2025-10-31
@Description: An ASCF Path Planning Method for a Hybrid Robot in Coating Inspection Scenarios
'''

import sys
import os
import time
import numpy as np
from stable_baselines3 import SAC
from env_test import MobileManipulatorEnv
from deploy_filter import create_cbf_filter

def test_apf_sac_model():
    # 模型绝对路径
    model_path = "..........."
    # 测试参数
    test_episodes = 1000
    print("="*60)
    print(f"APF-SAC 6-DOF 复杂环境模型测试 (完整多点CBF-QP)")
    print("="*60)
    print(f"模型路径: {model_path}")
    print(f"测试回合数: {test_episodes}")
    print(f"环境模式: 复杂场景 (双随机障碍物)")
    print(f"CBF安全过滤器: 完整QP求解 - 多控制点版本")
    print(f"保护控制点: 末端执行器 + 肘部 + 腕部")
    print("="*60)
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return
    
    try:
        env = MobileManipulatorEnv(
            gui=True,
            fixed_scene=False,
            complex_scene=True
        )
        model = SAC.load(model_path)
        print("✓ SAC模型加载成功")
        
        cbf_filter = create_cbf_filter(
            env,
            safety_config={
                'safety_radius': 0.2,  
                'kappa1': 1.25,          
                'kappa2': 1.25           
            }
        )
        print("✓ 完整多点CBF-QP过滤器初始化完成")
        print("  保护点: 末端执行器 + 肘部 + 腕部")
        print("  使用cvxopt QP求解器 (无简化)")
        print("✓ 复杂环境初始化完成")
        
        success_count = 0
        collision_count = 0
        total_rewards = []
        episode_lengths = []
        apf_activation_count = 0
    
        cbf_activation_episodes = 0 
        total_cbf_calls = 0
        total_cbf_activations = 0
        all_solve_times = []
        all_tau_qp_norms = []
        
        point_activation_total = {
            'end_effector': 0,
            'elbow': 0,
            'wrist': 0
        }
        
        print("\n开始测试...")
        start_time = time.time()
        
        for episode in range(test_episodes):
            state, _ = env.reset()
            cbf_filter.reset_statistics()
            
            episode_reward = 0
            episode_steps = 0
            done = False
            cbf_activated_this_episode = False
            while not done:

                action, _ = model.predict(observation=state, deterministic=True)
                K_p = 10.0  # 比例增益
                tau_nom = action * K_p
                tau_safe, cbf_info = cbf_filter.filter_action(tau_nom)
                
                total_cbf_calls += 1
                if cbf_info.get('cbf_active', False):
                    total_cbf_activations += 1
                    cbf_activated_this_episode = True
                    solve_time = cbf_info.get('solve_time', 0)
                    all_solve_times.append(solve_time * 1000)  # 转换为ms
                    tau_qp_norm = cbf_info.get('tau_qp_norm', 0)
                    all_tau_qp_norms.append(tau_qp_norm)
                    active_points = cbf_info.get('active_points', [])
                    for cp_name in active_points:
                        if cp_name in point_activation_total:
                            point_activation_total[cp_name] += 1
                
                safe_action = tau_safe / K_p
                safe_action = np.clip(safe_action, 
                                     env.action_space.low, 
                                     env.action_space.high)  
                state, reward, terminated, truncated, info = env.step(safe_action)
                episode_reward += reward
                episode_steps += 1
                done = terminated or truncated
                
                if episode_steps >= 180:
                    done = True

            total_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            
            if cbf_activated_this_episode:
                cbf_activation_episodes += 1
            if info.get('is_success', False):
                success_count += 1
            if info.get('collision', False):
                collision_count += 1
            if hasattr(env, 'apf_activation_count') and env.apf_activation_count > 0:
                apf_activation_count += 1
            if (episode + 1) % 100 == 0:
                current_success_rate = success_count / (episode + 1) * 100
                current_collision_rate = collision_count / (episode + 1) * 100
                avg_reward = np.mean(total_rewards)
                avg_steps = np.mean(episode_lengths)
                
                total_steps_so_far = sum(episode_lengths)
                cbf_rate = total_cbf_activations / total_steps_so_far * 100 if total_steps_so_far > 0 else 0
                
                progress_str = (f"进度: {episode + 1:4d}/{test_episodes} | "
                               f"成功率: {current_success_rate:5.1f}% | "
                               f"碰撞率: {current_collision_rate:5.1f}% | "
                               f"平均奖励: {avg_reward:7.2f} | "
                               f"平均步数: {avg_steps:5.1f} | "
                               f"CBF激活: {cbf_rate:.1f}%")
                
                print(progress_str)
        
        total_time = time.time() - start_time
        final_success_rate = success_count / test_episodes * 100
        final_collision_rate = collision_count / test_episodes * 100
        final_avoidance_rate = (test_episodes - collision_count) / test_episodes * 100
        apf_usage_rate = apf_activation_count / test_episodes * 100
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        max_reward = np.max(total_rewards)
        min_reward = np.min(total_rewards)
        
        avg_steps = np.mean(episode_lengths)
        std_steps = np.std(episode_lengths)
        total_steps = sum(episode_lengths)
        
        cbf_episode_rate = cbf_activation_episodes / test_episodes * 100
        cbf_step_rate = total_cbf_activations / total_steps * 100
        
        avg_solve_time = np.mean(all_solve_times) if all_solve_times else 0
        max_solve_time = np.max(all_solve_times) if all_solve_times else 0
        min_solve_time = np.min(all_solve_times) if all_solve_times else 0
        
        avg_tau_qp = np.mean(all_tau_qp_norms) if all_tau_qp_norms else 0
        max_tau_qp = np.max(all_tau_qp_norms) if all_tau_qp_norms else 0

        point_rates = {}
        for cp_name, count in point_activation_total.items():
            point_rates[cp_name] = count / total_steps * 100 if total_steps > 0 else 0

        print("\n" + "="*60)
        print(f"APF-SAC 6-DOF 完整多点CBF-QP测试结果")
        print("="*60)
        print(f"测试总时间: {total_time/60:.2f} 分钟")
        print(f"测试回合数: {test_episodes}")
        print()
        print("任务完成统计:")
        print(f"  成功回合数: {success_count}")
        print(f"  成功率: {final_success_rate:.2f}%")
        print(f"  碰撞回合数: {collision_count}")
        print(f"  碰撞率: {final_collision_rate:.2f}%")
        print(f"  避障率: {final_avoidance_rate:.2f}%")
        print()
        print("奖励统计:")
        print(f"  平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  最高奖励: {max_reward:.2f}")
        print(f"  最低奖励: {min_reward:.2f}")
        print()
        print("回合长度统计:")
        print(f"  平均步数: {avg_steps:.1f} ± {std_steps:.1f}")
        print(f"  总步数: {total_steps}")
        print()
        print("APF引导统计:")
        print(f"  APF激活回合数: {apf_activation_count}")
        print(f"  APF使用率: {apf_usage_rate:.2f}%")
        print()
        print("="*60)
        print("完整多点CBF-QP统计:")
        print("="*60)
        print(f"  CBF总调用次数: {total_cbf_calls}")
        print(f"  CBF激活次数: {total_cbf_activations}")
        print(f"  CBF激活回合数: {cbf_activation_episodes} ({cbf_episode_rate:.2f}%)")
        print(f"  CBF激活率 (按步数): {cbf_step_rate:.2f}%")
        print()
        print("QP求解性能:")
        print(f"  平均求解时间: {avg_solve_time:.3f} ms")
        print(f"  最大求解时间: {max_solve_time:.3f} ms")
        print(f"  最小求解时间: {min_solve_time:.3f} ms")
        print()
        print("修正力矩统计:")
        print(f"  平均 ‖τ_qp‖: {avg_tau_qp:.4f}")
        print(f"  最大 ‖τ_qp‖: {max_tau_qp:.4f}")
        print()
        print("各控制点激活统计:")
        print(f"  末端执行器: {point_activation_total['end_effector']} 次 ({point_rates['end_effector']:.2f}%)")
        print(f"  肘部: {point_activation_total['elbow']} 次 ({point_rates['elbow']:.2f}%)")
        print(f"  腕部: {point_activation_total['wrist']} 次 ({point_rates['wrist']:.2f}%)")
        print("="*60)
        
        result_file = f"test_results_full_cbf_qp_{int(time.time())}.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"APF-SAC 6-DOF 完整多点CBF-QP测试结果\n")
            f.write("="*60 + "\n")
            f.write(f"模型路径: {model_path}\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试回合数: {test_episodes}\n")
            f.write(f"环境模式: 复杂场景 (双随机障碍物)\n")
            f.write(f"CBF实现: 完整QP求解 (cvxopt) - 多控制点\n")
            f.write(f"保护控制点: 末端执行器 + 肘部 + 腕部\n\n")
            
            f.write("任务完成统计:\n")
            f.write(f"  成功率: {final_success_rate:.2f}% ({success_count}/{test_episodes})\n")
            f.write(f"  碰撞率: {final_collision_rate:.2f}% ({collision_count}/{test_episodes})\n")
            f.write(f"  避障率: {final_avoidance_rate:.2f}%\n\n")
            
            f.write("性能统计:\n")
            f.write(f"  平均奖励: {avg_reward:.2f} ± {std_reward:.2f}\n")
            f.write(f"  奖励范围: [{min_reward:.2f}, {max_reward:.2f}]\n")
            f.write(f"  平均步数: {avg_steps:.1f} ± {std_steps:.1f}\n")
            f.write(f"  APF使用率: {apf_usage_rate:.2f}%\n\n")
            
            f.write("完整CBF-QP统计:\n")
            f.write(f"  CBF激活次数: {total_cbf_activations}\n")
            f.write(f"  CBF激活率: {cbf_step_rate:.2f}%\n")
            f.write(f"  CBF激活回合率: {cbf_episode_rate:.2f}%\n")
            f.write(f"  平均求解时间: {avg_solve_time:.3f} ms\n")
            f.write(f"  求解时间范围: [{min_solve_time:.3f}, {max_solve_time:.3f}] ms\n")
            f.write(f"  平均修正力矩: {avg_tau_qp:.4f}\n")
            f.write(f"  最大修正力矩: {max_tau_qp:.4f}\n\n")
            
            f.write("各控制点激活统计:\n")
            for cp_name in ['end_effector', 'elbow', 'wrist']:
                count = point_activation_total[cp_name]
                rate = point_rates[cp_name]
                f.write(f"  {cp_name}: {count} 次 ({rate:.2f}%)\n")
            
            f.write(f"\n总测试时间: {total_time/60:.2f} 分钟\n")
        
        print(f"\n详细测试结果已保存到: {result_file}")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            env.close()
            print("环境已关闭")
        except:
            pass

if __name__ == '__main__':
    """
    运行完整CBF-QP测试
    
    关键改进:
    1. 使用完整的deploy_filter.CBFSafetyFilter.filter_action()
    2. 正确的动作->力矩->安全力矩->安全动作转换
    3. 完整的多点QP求解 (无简化)
    4. 详细的CBF性能统计
    """
    print("\n" + "="*60)
    print("开始测试：SAC + 完整多点CBF-QP")
    print("保护点: 末端执行器 + 肘部 + 腕部")
    print("QP求解器: cvxopt (无简化)")
    print("="*60)
    
    test_apf_sac_model()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("结果文件: test_results_full_cbf_qp_*.txt")
    print("="*60)