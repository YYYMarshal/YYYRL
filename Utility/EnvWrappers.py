import gym


class FrozenLakeWrapper(gym.Wrapper):
    """
    环境参考：
    https://www.bilibili.com/video/BV1X94y1Y7hS?t=41.3&p=3
    """

    def __init__(self):
        # is_slippery 控制会不会滑
        env = gym.make('FrozenLake-v0', is_slippery=False)
        super().__init__(env)
        self.env = env

    def step(self, action):
        """
        Reward schedule:
        Reach goal(G): +1
        Reach hole(H): 0
        Reach frozen(F): 0
        """
        next_state, reward, done, info = self.env.step(action)
        # 走一步扣一分,逼迫智能体尽快结束游戏
        if not done:
            reward = -1
        # 掉坑(H, hole)扣100分
        if done and reward == 0:
            reward = -100
        # 走到终点，得一分，修改为得100分。
        if reward == 1:
            reward = 100
        return next_state, reward, done, info
