# 环境配置

```cmd
conda create -n YYYRL python=3.8
conda activate YYYRL
pip install gym==0.19.0
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# 0.8.4
pip install tyro
# 0.29.37
pip install "cython<3"
# 2.7.0
pip install glfw
# 2.34.1
pip install imageio
# 3.7.5
pip install matplotlib

# 2024-8-15 19:37:04
# env.render() 时出错，解决方案：
pip install pyglet==1.5.0

# 2024-9-27 16:16:38
# gym - Box2D - LunarLander-v2
# AttributeError: module ‘gym.envs.box2d‘ has no attribute ‘LunarLander‘ 
pip install gym[box2d]
```

并将 `mujoco_py` 文件夹放入 `S:\Users\YYYXB\anaconda3\envs\YYYRL\Lib\site-packages` 中。

# 代码来源

| 算法        | 代码来源                                                                                                                                 |
| ----------- |--------------------------------------------------------------------------------------------------------------------------------------|
| Sarsa       | [动手学强化学习](https://hrl.boyuai.com/chapter/1/%E6%97%B6%E5%BA%8F%E5%B7%AE%E5%88%86%E7%AE%97%E6%B3%95/#53-sarsa-%E7%AE%97%E6%B3%95)      |
| Qlearning   | [动手学强化学习](https://hrl.boyuai.com/chapter/1/%E6%97%B6%E5%BA%8F%E5%B7%AE%E5%88%86%E7%AE%97%E6%B3%95/#55-q-learning-%E7%AE%97%E6%B3%95) |
| DQN         | [动手学强化学习](https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95)                                                                    |
| REINFORCE   | [动手学强化学习](https://hrl.boyuai.com/chapter/2/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E7%AE%97%E6%B3%95)                                   |
| ActorCritic | [动手学强化学习](https://hrl.boyuai.com/chapter/2/actor-critic%E7%AE%97%E6%B3%95)                                                           |
| PPO         | [动手学强化学习](https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95)                                                                    |
| DDPG        | [动手学强化学习](https://hrl.boyuai.com/chapter/2/ddpg%E7%AE%97%E6%B3%95)                                                                   |
| TD3         | [Official - GitHub](https://github.com/sfujim/TD3)                                                                                   |
| SAC         | [动手学强化学习](https://hrl.boyuai.com/chapter/2/sac%E7%AE%97%E6%B3%95)                                                                    |
|             |                                                                                                                                      |
|             |                                                                                                                                      |

