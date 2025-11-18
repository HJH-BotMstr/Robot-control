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
from env import MobileManipulatorEnv

def test_apf_sac_model():
    # 模型绝对路径
    model_path = "............."
    test_episodes = 1000
    print("="*60)
    print("APF-SAC 6-DOF 复杂环境模型测试")
    print("="*60)
    print(f"模型路径: {model_path}")
    print(f"测试回合数: {test_episodes}")
    print(f"环境模式: 复杂场景 (双随机障碍物)")
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
        print("✓ 模型加载成功")
        print("✓ 复杂环境初始化完成")
        
        success_count = 0
        collision_count = 0
        total_rewards = []
        episode_lengths = []
        apf_activation_count = 0
        
        print("\n开始测试...")
        start_time = time.time()
        
        # 执行测试回合
        for episode in range(test_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            while not done:
                action, _ = model.predict(observation=state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1
                done = terminated or truncated
                if episode_steps >= 180:
                    done = True
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            if info.get('is_success', False):
                success_count += 1
            if info.get('collision', False):
                collision_count += 1
            if info.get('apf_activated', False):
                apf_activation_count += 1
            if (episode + 1) % 100 == 0:
                current_success_rate = success_count / (episode + 1) * 100
                current_collision_rate = collision_count / (episode + 1) * 100
                avg_reward = np.mean(total_rewards)
                avg_steps = np.mean(episode_lengths)
                
                print(f"进度: {episode + 1:4d}/{test_episodes} | "
                      f"成功率: {current_success_rate:5.1f}% | "
                      f"碰撞率: {current_collision_rate:5.1f}% | "
                      f"平均奖励: {avg_reward:7.2f} | "
                      f"平均步数: {avg_steps:5.1f}")
        
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
        
        print("\n" + "="*60)
        print("APF-SAC 6-DOF 复杂环境测试结果")
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
        print()
        print("APF引导统计:")
        print(f"  APF激活回合数: {apf_activation_count}")
        print(f"  APF使用率: {apf_usage_rate:.2f}%")
        print("="*60)
        
        result_file = f"test_results_complex_{int(time.time())}.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("APF-SAC 6-DOF 复杂环境测试结果\n")
            f.write("="*50 + "\n")
            f.write(f"模型路径: {model_path}\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试回合数: {test_episodes}\n")
            f.write(f"环境模式: 复杂场景 (双随机障碍物)\n\n")
            
            f.write("任务完成统计:\n")
            f.write(f"  成功率: {final_success_rate:.2f}% ({success_count}/{test_episodes})\n")
            f.write(f"  碰撞率: {final_collision_rate:.2f}% ({collision_count}/{test_episodes})\n")
            f.write(f"  避障率: {final_avoidance_rate:.2f}%\n\n")
            
            f.write("性能统计:\n")
            f.write(f"  平均奖励: {avg_reward:.2f} ± {std_reward:.2f}\n")
            f.write(f"  奖励范围: [{min_reward:.2f}, {max_reward:.2f}]\n")
            f.write(f"  平均步数: {avg_steps:.1f} ± {std_steps:.1f}\n")
            f.write(f"  APF使用率: {apf_usage_rate:.2f}%\n")
            f.write(f"  总测试时间: {total_time/60:.2f} 分钟\n")
        
        print(f"\n测试结果已保存到: {result_file}")
        
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
    test_apf_sac_model()