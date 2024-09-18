import gym
import torch
import torch.nn.functional as F
import numpy as np
from Utility import TrainingProcess, Timer
from dataclasses import dataclass
import tyro


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


@dataclass
class Args:
    agent_name: str = "REINFORCE"
    env_name: str = "CartPole-v0"
    seed: int = 0
    learning_rate: int = 1e-3
    num_episodes: int = 1000
    hidden_dim: int = 128
    gamma: float = 0.98


def train(args: Args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"agent_name = {args.agent_name}, device: {device}\n"
          f"env_name = {args.env_name}, state_dim = {state_dim}, action_dim = {action_dim}")
    agent = REINFORCE(state_dim, args.hidden_dim, action_dim, args.learning_rate, args.gamma, device)

    episode_reward_list = TrainingProcess.train_on_policy_agent(env, agent, args.num_episodes)
    file_name = f"{args.agent_name}_{args.env_name}_{args.seed}"
    target_folder_name = "results"
    np.save(f"{target_folder_name}/{file_name}", episode_reward_list)
    return episode_reward_list


def main():
    start_time = Timer.get_current_time()
    args = tyro.cli(Args)
    train(args)
    Timer.time_difference(start_time)


if __name__ == "__main__":
    main()
