import gym
import torch
import torch.nn.functional as F
import numpy as np
from Utility.TrainingProcess import train_on_policy_agent
from Utility.Plot import plot
from Utility import Timer


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


"""
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
"""


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        state_list = transition_dict["states"]
        action_list = transition_dict["actions"]
        reward_list = transition_dict["rewards"]

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor(np.array([state_list[i]]), dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降


def train(env_name: str):
    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"device: {device}\n"
          f"env_name = {env_name}, state_dim = {state_dim}, action_dim = {action_dim}")
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

    episode_reward_list = train_on_policy_agent(env, agent, num_episodes)
    # np.mean(episode_reward_list) = 141.86
    return episode_reward_list


def main():
    start_time = Timer.get_current_time()
    env_name = "CartPole-v0"
    return_list = train(env_name)
    Timer.time_difference(start_time)
    plot(return_list, "REINFORCE", env_name, True)


if __name__ == "__main__":
    main()
