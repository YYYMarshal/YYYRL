import random
import gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Utility.ReplayBuffer import ReplayBuffer
from Utility.TrainingProcess import train_off_policy_agent
from Utility.Plot import plot, moving_average
from Utility import Timer


class QNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        只有一层隐藏层（fc1）的Q网络，fc2是输出层
        """
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        定义了前向传播方法，用于定义模型的正向计算
        """
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class VANet(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        只有一层隐藏层的 A网络 和 V网络
        """
        super(VANet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        value_a = self.fc_A(F.relu(self.fc1(x)))
        value_v = self.fc_V(F.relu(self.fc1(x)))
        # Q值 由 V值 和 A值 计算得到
        value_q = value_v + value_a - value_a.mean(1).view(-1, 1)
        return value_q


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate,
                 gamma, epsilon, target_update, device, dqn_type="DQN"):
        self.action_dim = action_dim
        if dqn_type == "DuelingDQN":
            """ Dueling DQN 采取不一样的网络框架 """
            self.q_net = VANet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = VANet(state_dim, hidden_dim, self.action_dim).to(device)
        else:
            # Q网络
            self.q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)
            # 目标网络
            self.target_q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.dqn_type = dqn_type

    def take_action(self, state):
        """
        epsilon-贪婪策略采取动作
        """
        # 探索
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        # 利用
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == "DoubleDQN":
            """ DQN 与 Double DQN 的区别 """
            """
            也就是选取q_net中n个状态的最大值对应的索引（0或1，因为 action_dim = 2），
            然后摊平变成n行1列的二维数组的类型，
            再将获得的索引放入对应的target_q_net的n个状态中，获得对应的q值。
            """
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            """ DQN的情况 """
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # 如果 dones 的某个值为 1，也就是代表该 Episode 结束，那么 q_targets = rewards
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1


def train(env_name: str, dqn_type: str):
    lr = 2e-3
    num_episodes = 500
    # 当 hidden_dim = 128 时：mean_DQN = 141.316
    # 当 hidden_dim = 512 时：mean_DQN = 153.216
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    # 当 buffer 数据的数量超过 minimal_size 后,才进行Q网络训练
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"dqn_type = {dqn_type}, device: {device}\n"
          f"env_name = {env_name}, state_dim = {state_dim}, action_dim = {action_dim}")

    agent = DQN(state_dim, hidden_dim, action_dim, lr,
                gamma, epsilon, target_update, device, dqn_type=dqn_type)
    replay_buffer = ReplayBuffer(buffer_size)
    episode_reward_list = train_off_policy_agent(
        env, agent, num_episodes,
        replay_buffer, minimal_size, batch_size)
    return episode_reward_list


def main_single_dqn(dqn_type: str):
    start_time = Timer.get_current_time()
    # env_name = "CartPole-v0"
    env_name = "Acrobot-v1"
    episode_reward_list = train(env_name, dqn_type)
    Timer.time_difference(start_time)
    plot(episode_reward_list, dqn_type, env_name)


def main_all_dqn():
    env_name = "CartPole-v0"

    start_time = Timer.get_current_time()
    episode_reward_list_dqn = train(env_name, "DQN")
    Timer.time_difference(start_time)
    print("---------------------")

    start_time = Timer.get_current_time()
    episode_reward_list_doubledqn = train(env_name, "DoubleDQN")
    Timer.time_difference(start_time)
    print("---------------------")

    start_time = Timer.get_current_time()
    episode_reward_list_duelingdqn = train(env_name, "DuelingDQN")
    Timer.time_difference(start_time)
    print("---------------------")

    xlabel = "Episodes"
    ylabel = "Returns"
    title = f"DQN/DoubleDQN/DuelingDQN on {env_name}"

    episodes_list_dqn = list(range(len(episode_reward_list_dqn)))
    plt.plot(episodes_list_dqn, episode_reward_list_dqn,
             label="DQN", color="blue")

    episodes_list_doubledqn = list(range(len(episode_reward_list_doubledqn)))
    plt.plot(episodes_list_doubledqn, episode_reward_list_doubledqn,
             label="DoubleDQN", color="red")

    episodes_list_duelingdqn = list(range(len(episode_reward_list_duelingdqn)))
    plt.plot(episodes_list_duelingdqn, episode_reward_list_duelingdqn,
             label="DuelingDQN", color="green")

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

    mv_return_dqn = moving_average(episode_reward_list_dqn, 9)
    plt.plot(episodes_list_dqn, mv_return_dqn,
             label="DQN", color="blue")

    mv_return_doubledqn = moving_average(episode_reward_list_doubledqn, 9)
    plt.plot(episodes_list_doubledqn, mv_return_doubledqn,
             label="DoubleDQN", color="red")

    mv_return_duelingdqn = moving_average(episode_reward_list_duelingdqn, 9)
    plt.plot(episodes_list_duelingdqn, mv_return_duelingdqn,
             label="DuelingDQN", color="green")

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # CartPole, Reward 的平均值 = 141.316
    # Acrobot, Episode Reward List 的平均值 = -122.096
    main_single_dqn("DQN")
    # main_single_dqn("DoubleDQN")
    # main_single_dqn("DuelingDQN")
    # main_all_dqn()
