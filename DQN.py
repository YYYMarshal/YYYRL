import random
import gym
import numpy as np
import torch
import torch.nn.functional as F
from Utility import ReplayBuffer, TrainingProcess, Timer
from dataclasses import dataclass
import tyro


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
            # self.q_net(state).argmax()：选取Q网络中对应state时的值最大的索引，类型是tensor，比如tensor(2)
            # self.q_net(state).argmax().item()：取具体的值，比如上一步是tensor(2)，则.item()会返回2
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        # 示例中，batch_size = 64，state_dim = 4，所以 states.shape 是 torch.Size([64, 4])
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # states的形状是 [64, 4]，self.q_net(states)的形状是torch.Size([64, 2])，可以理解为 64*2 的矩阵
        # 下面这行代码的意思就是取每一行中 self.q_net(states)[i, j] 的值，其中 j = actions[i]
        # 也就是取某个具体状态下的某个动作的Q值，最后形成一个 64*1 的 tensor
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


@dataclass
class Args:
    agent_name: str = "DQN"
    env_name: str = "CartPole-v0"
    num_episodes: int = 500
    seed: int = 0
    # learning rate
    learning_rate: int = 2e-3
    hidden_dim: int = 128
    gamma: float = 0.98
    epsilon: float = 0.01
    # 经过 target_update 个 step 后目标网络进行更新
    target_update: int = 10
    buffer_size: int = 10000
    # 当 buffer 数据的数量超过 minimal_size 后,才进行Q网络训练
    minimal_size: int = 500
    batch_size: int = 64


def train(args: Args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = gym.make(args.env_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"dqn_type = {args.agent_name}, device: {device}\n"
          f"env_name = {args.env_name}, state_dim = {state_dim}, action_dim = {action_dim}")

    agent = DQN(state_dim, args.hidden_dim, action_dim, args.learning_rate,
                args.gamma, args.epsilon, args.target_update, device, dqn_type=args.agent_name)
    replay_buffer = ReplayBuffer.ReplayBuffer(args.buffer_size)
    episode_reward_list = TrainingProcess.train_off_policy_agent(
        env, agent, args.num_episodes,
        replay_buffer, args.minimal_size, args.batch_size)

    file_name = f"{args.agent_name}_{args.env_name}_{args.seed}"
    target_folder_name = "results"
    np.save(f"{target_folder_name}/{file_name}", episode_reward_list)

    return episode_reward_list


def main():
    start_time = Timer.get_current_time()
    args = tyro.cli(Args)
    train(args)
    Timer.time_difference(start_time)


if __name__ == '__main__':
    main()
