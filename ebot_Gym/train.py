'''
@Author: HJH-BotMstr
@Date: 2025-10-31
@Description: An ASCF Path Planning Method for a Hybrid Robot in Coating Inspection Scenarios
'''
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
import random
from loguru import logger
import argparse

# Stable Baselines3 imports
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 导入6-DOF环境
from env import MobileManipulatorEnv
try:
    from reward import calculate_apf_sac_reward
except ImportError:
    print("Warning: reward_sac.py not found, using built-in reward function")

class Config:
    # 训练参数
    M_RANDOM = 6000      
    T = 200             
    GAMMA = 0.99        
    
    # 网络参数 
    CRITIC_LR = 3e-4   
    ACTOR_LR = 3e-4     
    HIDDEN_SIZE = 400   
    
    # SAC特有参数
    BUFFER_SIZE = int(2e5)  
    BATCH_SIZE = 256        
    TAU = 0.005             
    LEARNING_STARTS = 10000 
    
    # 动作和状态空间 
    STATE_DIM = 35      
    ACTION_DIM = 6      
    
    # SAC熵调节参数
    ENT_COEF = "auto_0.1"   
    
    # 引导机制参数
    OMEGA = 4.0         
    
    # 保存参数
    SAVE_FREQ = 100     
    LOG_FREQ = 10       
    EVAL_FREQ = 100     

class CustomSACCallback(BaseCallback):
    def __init__(self, trainer, verbose=0):
        super(CustomSACCallback, self).__init__(verbose)
        self.trainer = trainer
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.apf_activation_count = 0
        self.switch_point_1 = Config.M_RANDOM * 3 // 10  
        self.switch_point_2 = Config.M_RANDOM * 6 // 10  
        self.timestamp = time.strftime('%m%d-%H%M%S', time.localtime())
        
    def _on_step(self) -> bool:
        # 累积当前episode的奖励
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # 检查是否需要切换场景（switch_point）
        if self.episode_count == self.switch_point_1 and not hasattr(self, 'switched_to_random'):
            self.switched_to_random = True
            logger.info(f"\n切换到随机场景训练模式 (Episode {self.episode_count})")
            env = self.training_env.envs[0].env
            env.fixed_scene = False
            env.complex_scene = False
            env.reset()
        elif self.episode_count == self.switch_point_2 and not hasattr(self, 'switched_to_complex'):
            self.switched_to_complex = True
            logger.info(f"\n切换到复杂场景训练模式 (Episode {self.episode_count})")
            env = self.training_env.envs[0].env
            env.fixed_scene = False
            env.complex_scene = True
            env.reset()
        
        # Episode结束时的处理
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_count += 1
            self.trainer.training_rewards.append(self.current_episode_reward)

            # 更新训练器统计信息
            if self.episode_count <= self.switch_point_1:
                self.trainer.fixed_scene_rewards.append(self.current_episode_reward)
            elif self.episode_count <= self.switch_point_2:
                self.trainer.random_scene_rewards.append(self.current_episode_reward)
            else:
                self.trainer.complex_scene_rewards.append(self.current_episode_reward)
            
            # 获取info
            info = self.locals.get('infos', [{}])[0]
            if info.get('is_success', False):
                self.success_count += 1
            if info.get('collision', False):
                self.collision_count += 1
            if info.get('apf_activated', False):
                self.apf_activation_count += 1
            
            # 更新累积统计
            current_success_rate = self.success_count / self.episode_count * 100
            current_collision_rate = self.collision_count / self.episode_count * 100
            current_avoidance_rate = 100.0 - current_collision_rate
            apf_activation_rate = self.apf_activation_count / self.episode_count * 100
            
            self.trainer.cumulative_success_rates.append(current_success_rate)
            self.trainer.cumulative_collision_rates.append(current_collision_rate)
            self.trainer.cumulative_avoidance_rates.append(current_avoidance_rate)
            self.trainer.apf_activation_rates.append(apf_activation_rate)
            
            # 日志记录
            if self.episode_count % Config.LOG_FREQ == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                success_rate = self.success_count / self.episode_count * 100
                collision_rate = self.collision_count / self.episode_count * 100
                if self.episode_count <= self.switch_point_1:
                    scene_type = "固定"
                elif self.episode_count <= self.switch_point_2:
                    scene_type = "随机"
                else:
                    scene_type = "复杂"
                
                logger.info(
                    f"6-DOF SAC [{scene_type}] Episode {self.episode_count:4d} | "
                    f"Reward: {self.current_episode_reward:7.2f} | "
                    f"Avg100: {avg_reward:7.2f} | "
                    f"Success: {success_rate:5.1f}% | "
                    f"Collision: {collision_rate:5.1f}% | "
                    f"Steps: {self.current_episode_length:3d}"
                )
            
           # 保存模型
            if self.episode_count % Config.SAVE_FREQ == 0 and self.episode_count > 0:
                if self.episode_count <= self.switch_point_1:
                    scene_suffix = "fixed"
                elif self.episode_count <= self.switch_point_2:
                    scene_suffix = "random"
                else:
                    scene_suffix = "complex"
                model_path = f"{self.trainer.save_dir}/apf_sac_6dof_sb3_{scene_suffix}_{self.timestamp}_ep{self.episode_count}"
                self.model.save(model_path)
                logger.info(f"6-DOF SAC模型已保存到: {model_path}")
            
            # 重置episode统计
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
        return True

class APFSACTrainingManager:
    def __init__(self, save_dir="./models_6dof_sac", log_dir="./logs_6dof_sac"):
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.training_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.collision_rates = []
        self.cumulative_collision_rates = []  
        self.cumulative_avoidance_rates = []
        self.cumulative_success_rates = []
        self.apf_activation_rates = []  
        self.fixed_scene_rewards = []
        self.random_scene_rewards = []
        self.complex_scene_rewards = []  

        self.timestamp = time.strftime('%m%d-%H%M%S', time.localtime())
        
        # SAC模型
        self.model = None
    
    def create_sac_model(self, env):
        # 定义网络架构，与原代码保持一致
        policy_kwargs = dict(
            net_arch=dict(
                pi=[Config.HIDDEN_SIZE, Config.HIDDEN_SIZE],  # Actor网络
                qf=[Config.HIDDEN_SIZE, Config.HIDDEN_SIZE]   # Critic网络
            ),
            activation_fn=nn.ReLU,
        )
        
        # 创建SAC模型
        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=Config.ACTOR_LR,
            buffer_size=Config.BUFFER_SIZE,
            learning_starts=Config.LEARNING_STARTS,  
            batch_size=Config.BATCH_SIZE,
            tau=Config.TAU,
            gamma=Config.GAMMA,
            train_freq=1,
            gradient_steps=2,  
            target_update_interval=2,  
            ent_coef=Config.ENT_COEF,  
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="auto",
            tensorboard_log=f"{self.log_dir}/tensorboard/"
        )
        
        logger.info(f"创建SB3 SAC模型完成")
        logger.info(f"  - Actor学习率: {Config.ACTOR_LR}")
        logger.info(f"  - Critic学习率: {Config.CRITIC_LR}")
        logger.info(f"  - 缓冲区大小: {Config.BUFFER_SIZE}")
        logger.info(f"  - 批次大小: {Config.BATCH_SIZE}")
        logger.info(f"  - 软更新系数: {Config.TAU}")
        logger.info(f"  - 学习开始步数: {Config.LEARNING_STARTS}")
        logger.info(f"  - 熵系数: {Config.ENT_COEF}")
        
        return self.model
    
    def train_mixed_mode(self, episodes=Config.M_RANDOM):
        logger.info("="*50)
        logger.info("开始6-DOF混合模式训练 (Stable Baselines3 SAC)")
        logger.info(f"总训练轮次: {episodes}")
        logger.info(f"前{episodes*3//10}轮：固定场景训练")
        logger.info(f"中{episodes*3//10}轮：随机场景训练")
        logger.info(f"后{episodes*4//10}轮：复杂场景训练（双障碍物）")
        logger.info("控制6个机械臂关节 + 底座自动移动")
        logger.info("="*50)
        
        # 初始化环境
        env = MobileManipulatorEnv(gui=False, fixed_scene=True)
        env = Monitor(env)  
        
        # 创建SAC模型
        self.create_sac_model(env)
        
        # 创建自定义回调
        callback = CustomSACCallback(trainer=self)
        
        # 计算总训练步数
        total_timesteps = episodes * Config.T
        
        # 开始训练
        start_time = time.time()
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=Config.LOG_FREQ,
                progress_bar=False  
            )
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
        finally:
            env.close()
        
        final_model_path = f"{self.save_dir}/apf_sac_6dof_sb3_final_{self.timestamp}"
        self.model.save(final_model_path)
        logger.info(f"最终模型已保存到: {final_model_path}")

        total_time = time.time() - start_time
        actual_episodes = callback.episode_count
        final_success_rate = callback.success_count / actual_episodes * 100 if actual_episodes > 0 else 0
        final_collision_rate = callback.collision_count / actual_episodes * 100 if actual_episodes > 0 else 0
        final_avoidance_rate = (actual_episodes - callback.collision_count) / actual_episodes * 100 if actual_episodes > 0 else 0
        final_apf_rate = callback.apf_activation_count / episodes * 100 if episodes > 0 else 0
        
        fixed_avg_reward = np.mean(self.fixed_scene_rewards) if self.fixed_scene_rewards else 0
        random_avg_reward = np.mean(self.random_scene_rewards) if self.random_scene_rewards else 0
        
        logger.info("="*50)
        logger.info("6-DOF混合模式训练完成 (Stable Baselines3 SAC)")
        logger.info(f"训练总耗时: {total_time/3600:.2f} 小时")
        logger.info(f"固定场景平均奖励: {fixed_avg_reward:.2f}")
        logger.info(f"随机场景平均奖励: {random_avg_reward:.2f}")
        logger.info(f"总体最终成功率: {final_success_rate:.2f}%")
        logger.info(f"总体最终避障率: {final_avoidance_rate:.2f}%")
        logger.info(f"总体平均奖励: {np.mean(self.training_rewards):.2f}")
        logger.info("="*50)
        
        return self.training_rewards, final_success_rate
    
    def evaluate_model(self, model_path, num_episodes=100, render=True):
        """评估训练好的6-DOF模型（SAC版本）"""
        logger.info("="*50)
        logger.info("开始6-DOF模型评估 (Stable Baselines3 SAC)")
        logger.info("="*50)
        
        env = MobileManipulatorEnv(gui=render, fixed_scene=False)
        
        self.model = SAC.load(model_path)
        logger.info(f"6-DOF SAC模型已从 {model_path} 加载")
        
        test_rewards = []
        success_count = 0
        collision_count = 0
        apf_activation_count = 0
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(Config.T):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                if terminated or truncated:
                    break
            
            test_rewards.append(episode_reward)
            if info.get('is_success', False):
                success_count += 1
            if info.get('collision', False):
                collision_count += 1
            if info.get('apf_activated', False):
                apf_activation_count += 1
            
            if episode % 10 == 0:
                logger.info(f"6-DOF SAC Test Episode {episode:3d} | Reward: {episode_reward:7.2f} | Steps: {episode_length:3d}")
        env.close()
        avg_reward = np.mean(test_rewards)
        success_rate = success_count / num_episodes * 100
        collision_rate = collision_count / num_episodes * 100
        apf_rate = apf_activation_count / num_episodes * 100
        logger.info("="*50)
        logger.info("6-DOF模型评估完成 (Stable Baselines3 SAC)")
        logger.info(f"测试集数: {num_episodes}")
        logger.info(f"平均奖励: {avg_reward:.2f}")
        logger.info(f"成功率: {success_rate:.2f}%")
        logger.info(f"碰撞率: {collision_rate:.2f}%")
        logger.info("="*50)
        return {
            'average_reward': avg_reward,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'rewards': test_rewards
        }
    
    def plot_training_curves(self):
        """绘制6-DOF训练曲线（包括混合模式的对比 + SAC特有指标）"""
        if not self.training_rewards:
            logger.warning("没有训练数据可绘制")
            return
        plt.figure(figsize=(20, 10))
        
        # 图1：归一化奖励分布直方图
        plt.subplot(2, 3, 1)
        if self.training_rewards:
            rewards_array = np.array(self.training_rewards)
            min_reward = np.min(rewards_array)
            max_reward = np.max(rewards_array)
            if max_reward > min_reward:  # 避免除以零
                normalized_rewards = (rewards_array - min_reward) / (max_reward - min_reward)
            else:
                normalized_rewards = np.zeros_like(rewards_array)
            
            plt.hist(normalized_rewards, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            plt.title('6-DOF Mixed Training (SAC) - Normalized Reward Distribution')
            plt.xlabel('Normalized Reward')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
        
        # 图2：每回合奖励散点图
        plt.subplot(2, 3, 2)
        episodes = np.arange(len(self.training_rewards))
        switch_point_1 = len(self.training_rewards) * 3 // 10
        switch_point_2 = len(self.training_rewards) * 6 // 10
        
        # 固定场景
        plt.scatter(episodes[:switch_point_1], self.training_rewards[:switch_point_1], 
                s=8, color='tab:blue', alpha=0.6, label='Fixed Scene')
        # 随机场景
        plt.scatter(episodes[switch_point_1:switch_point_2], self.training_rewards[switch_point_1:switch_point_2], 
                s=8, color='tab:orange', alpha=0.6, label='Random Scene')
        # 复杂场景
        plt.scatter(episodes[switch_point_2:], self.training_rewards[switch_point_2:], 
                s=8, color='tab:red', alpha=0.6, label='Complex Scene')
        
        plt.axvline(x=switch_point_1, color='gray', linestyle='--', linewidth=1, label='Switch Point 1')
        plt.axvline(x=switch_point_2, color='gray', linestyle='--', linewidth=1, label='Switch Point 2')
        plt.title('6-DOF Three-Stage Training (SAC) - Reward Scatter Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 图3：成功率随回合数变化曲线
        plt.subplot(2, 3, 3)
        if self.cumulative_success_rates:
            episodes = range(len(self.cumulative_success_rates))
            plt.plot(episodes, self.cumulative_success_rates,
                    color='red', linewidth=2)
            plt.axvline(x=switch_point_1, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Switch to Random')
            plt.axvline(x=switch_point_2, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Switch to Complex')
            plt.title('6-DOF Three-Stage Training (SAC) - Cumulative Success Rate')
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Success Rate (%)')
            plt.legend() 
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
        
        # 图4：避障率随回合数变化曲线
        plt.subplot(2, 3, 4)
        if self.cumulative_avoidance_rates:
            episodes = range(len(self.cumulative_avoidance_rates))
            plt.plot(episodes, self.cumulative_avoidance_rates,
                    color='blue', linewidth=2)
            plt.axvline(x=switch_point_1, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Switch to Random')
            plt.axvline(x=switch_point_2, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Switch to Complex')
            plt.title('6-DOF Three-Stage Training (SAC) - Cumulative Avoidance Rate')
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Avoidance Rate (%)')
            plt.legend()  
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
        
        # 图5：固定场景vs随机场景奖励对比
        plt.subplot(2, 3, 5)
        if self.fixed_scene_rewards and self.random_scene_rewards and self.complex_scene_rewards:
            window = 50
            if len(self.fixed_scene_rewards) >= window:
                fixed_avg = np.convolve(self.fixed_scene_rewards, np.ones(window)/window, mode='valid')
                plt.plot(range(len(fixed_avg)), fixed_avg, 
                        color='blue', linewidth=2, label='Fixed Scene (MA)')
            if len(self.random_scene_rewards) >= window:
                random_avg = np.convolve(self.random_scene_rewards, np.ones(window)/window, mode='valid')
                plt.plot(range(len(self.fixed_scene_rewards), 
                            len(self.fixed_scene_rewards) + len(random_avg)), 
                        random_avg, color='orange', linewidth=2, label='Random Scene (MA)')
            if len(self.complex_scene_rewards) >= window:
                complex_avg = np.convolve(self.complex_scene_rewards, np.ones(window)/window, mode='valid')
                plt.plot(range(len(self.fixed_scene_rewards) + len(self.random_scene_rewards), 
                            len(self.fixed_scene_rewards) + len(self.random_scene_rewards) + len(complex_avg)), 
                        complex_avg, color='red', linewidth=2, label='Complex Scene (MA)')
            
            plt.title(f'6-DOF Three-Stage Comparison (SAC) - Moving Average (window={window})')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
       # 图6：留空或者显示简单的总结信息
        plt.subplot(2, 3, 6)
        plt.axis('off')
        if self.training_rewards:
            total_episodes = len(self.training_rewards)
            avg_reward = np.mean(self.training_rewards)
            final_success_rate = self.cumulative_success_rates[-1] if self.cumulative_success_rates else 0
            final_avoidance_rate = self.cumulative_avoidance_rates[-1] if self.cumulative_avoidance_rates else 0
            
            summary_text = f"6-DOF SAC Training Summary\n\n"
            summary_text += f"Total Episodes: {total_episodes}\n"
            summary_text += f"Average Reward: {avg_reward:.2f}\n"
            summary_text += f"Final Success Rate: {final_success_rate:.1f}%\n"
            summary_text += f"Final Avoidance Rate: {final_avoidance_rate:.1f}%\n\n"
            
            plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        save_path = f"{self.log_dir}/training_curves_6dof_sac_{self.timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"6-DOF SAC混合训练曲线已保存到: {save_path}")
        plt.show()

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='APF-SAC 6-DOF Mobile Manipulator Mixed Training (Stable Baselines3)')
    
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'], default='both',
                        help='运行模式: train(仅训练), eval(仅评估), both(训练+评估)')
    parser.add_argument('--episodes', type=int, default=Config.M_RANDOM,
                        help='总训练轮次（包括固定场景和随机场景）')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='评估轮次')
    parser.add_argument('--model_path', type=str, default=None,
                        help='评估时使用的模型路径')
    parser.add_argument('--save_dir', type=str, default='./models_6dof_sac',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs_6dof_sac',
                        help='日志保存目录')
    parser.add_argument('--gui', action='store_true',
                        help='是否显示GUI(仅评估时生效)')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    logger.add(f"{args.log_dir}/apf_sac_6dof_sb3_training.log", rotation="100 MB")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    logger.info("="*70)
    logger.info("APF-SAC移动机械臂协同避障轨迹规划训练开始（6-DOF混合模式 + SB3）")
    logger.info("="*70)
    
    logger.info(f"6-DOF SAC混合训练配置信息 (Stable Baselines3):")
    logger.info(f"  - 状态空间维度: {Config.STATE_DIM} (原29维 + 6维APF梯度)")
    logger.info(f"  - 动作空间维度: {Config.ACTION_DIM} (6个关节速度)")
    logger.info(f"  - 总训练轮次: {args.episodes}")
    logger.info(f"  - 前{args.episodes*3//10}轮: 固定场景")
    logger.info(f"  - 中{args.episodes*3//10}轮: 随机场景")
    logger.info(f"  - 后{args.episodes*4//10}轮: 复杂场景（双随机障碍物）")
    logger.info(f"  - Critic学习率: {Config.CRITIC_LR}")
    logger.info(f"  - Actor学习率: {Config.ACTOR_LR}")
    logger.info(f"  - 批次大小: {Config.BATCH_SIZE}")
    logger.info(f"  - 缓冲池大小: {Config.BUFFER_SIZE}")
    logger.info(f"  - 折扣因子: {Config.GAMMA}")
    logger.info(f"  - 软更新系数: {Config.TAU}")
    logger.info(f"  - 学习开始步数: {Config.LEARNING_STARTS}")
    logger.info(f"  - 熵系数: {Config.ENT_COEF}")
    logger.info(f"  - 引导增益上限: {Config.OMEGA}")
    logger.info(f"  - 引导模式: 距离动态式（无课程退火）")
    logger.info(f"  - 使用框架: Stable Baselines3 SAC")
    logger.info("="*70)
    trainer = APFSACTrainingManager(args.save_dir, args.log_dir)
    
    if args.mode in ['train', 'both']:
        start_time = time.time()
        logger.info("开始6-DOF SAC混合模式训练 (Stable Baselines3)")
        mixed_rewards, mixed_success_rate = trainer.train_mixed_mode(args.episodes)
        total_time = time.time() - start_time
        logger.info(f"6-DOF SAC混合训练总耗时: {total_time/3600:.2f} 小时")
        trainer.plot_training_curves()
        # 保存最终模型路径
        final_model = f"{args.save_dir}/apf_sac_6dof_sb3_final_{trainer.timestamp}"
        
    if args.mode in ['eval', 'both']:
        if args.mode == 'eval' and args.model_path:
            model_path = args.model_path
        elif args.mode == 'both':
            model_path = final_model
        else:
            logger.error("评估模式需要指定模型路径 --model_path")
            return
        
        if os.path.exists(model_path):
            eval_results = trainer.evaluate_model(
                model_path, 
                num_episodes=args.eval_episodes,
                render=args.gui
            )
            results_path = f"{args.log_dir}/eval_results_6dof_sac_{trainer.timestamp}.txt"
            with open(results_path, 'w') as f:
                f.write(f"APF-SAC 6-DOF混合模式模型评估结果 (Stable Baselines3)\n")
                f.write(f"模型路径: {model_path}\n")
                f.write(f"测试轮次: {args.eval_episodes}\n")
                f.write(f"平均奖励: {eval_results['average_reward']:.2f}\n")
                f.write(f"成功率: {eval_results['success_rate']:.2f}%\n")
                f.write(f"碰撞率: {eval_results['collision_rate']:.2f}%\n")
                
            logger.info(f"6-DOF SAC评估结果已保存到: {results_path}")
        else:
            logger.error(f"模型文件不存在: {model_path}")
    
    logger.info("="*70)
    logger.info("APF-SAC 6-DOF混合模式训练/评估完成 (Stable Baselines3)")
    logger.info("="*70)

if __name__ == "__main__":
    main()
