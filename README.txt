基于人工势场引导的软演员-评论家算法(APF-SAC)的6自由度移动机械臂避障轨迹规划系统
An ASCF Path Planning Method for a Hybrid Robot in Coating Inspection Scenarios
注意：
    运行train和test前修改URDF路径/模型路径为你自己的绝对路径
    
训练阶段 (Pure SAC)
	Stable Baselines3 SAC
	引导机制: 距离动态式APF引导
	训练模式: 三阶段渐进式
			前30%: 固定场景
			中30%: 随机场景  
			后40%: 复杂场景(双障碍物)
		状态空间: 29维 (6关节 + APF梯度信息)
		动作空间: 6维连续 (6个关节速度增量)

	部署阶段 (SAC + CBF-QP)
		安全过滤: 完整多点CBF-QP (100Hz)
		保护点: 末端执行器 + 肘部 + 腕部
		求解器: cvxopt QP优化
		实时性: 平均求解时间 < 5ms

项目结构

├── APF_SAC_train.py          # 训练脚本 
├── env.py                    # 训练环境定义 
├── env_test.py               # 测试环境定义 
├── reward.py                 # 奖励函数 
├── test_apf_sac.py           # 测试脚本 
├── test_apf_sac_cbf.py       # 含cbf约束的测试脚本 
├── deploy_filter.py          # CBF安全过滤器 
├── cbf_qp.py                 # CBF约束
└── env_test.py               # 测试环境 

快速开始

1. 环境配置:
		pip install stable-baselines3
		pip install pybullet
		pip install gymnasium
		pip install numpy scipy
		pip install loguru matplotlib
		pip install cvxopt  # CBF-QP求解器

2. 训练模型

自定义参数
python APF_SAC_train.py \
    mode train \
    episodes 6000 \
    save_dir ./models \
    log_dir ./logs

关键训练参数*(Config类):
	CRITIC_LR = 3e-4` - Critic学习率
	ACTOR_LR = 3e-4` - Actor学习率  
	BUFFER_SIZE = 2e5` - 经验回放池
	BATCH_SIZE = 256` - 批次大小
	ENT_COEF = "auto_0.125"` - 自动熵调节
	OMEGA = 4.0` - APF引导增益上限

3. 部署测试
测试配置:
	测试episodes: 1000
	环境模式: 复杂场景(双随机障碍物)
	CBF参数:
	  safety_radius = 0.2` (r_pad)
	  kappa1 = 1.25` (κ̂₁)
	  kappa2 = 1.25` (κ̂₂)


如使用本代码,请引用原论文:
@article{apf_sac_manipulator,
  title={An ASCF Path Planning Method for a Hybrid Robot in Coating Inspection Scenarios},
  author={Junhao Hu},
  journal={控制与决策(CCDC)},
  year={2025}
}
License：MIT License

