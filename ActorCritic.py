import gym
import numpy as np
import torch
import torch.nn.functional as F
from Utility import TrainingProcess, Timer
from dataclasses import dataclass
import tyro


class PolicyNet(torch.nn.Module):
    # 与 REINFORCE 算法一样
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    """
    Actor-Critic 算法中额外引入一个价值网络
    接下来的代码定义价值网络ValueNet，其输入是某个状态，输出则是状态的价值。
    """

    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic:
    """
    ActorCritic算法，主要包含采取动作（take_action()）和更新网络参数（update()）两个函数。
    """

    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        # 与 REINFORCE 算法一样，self.actor 也是 PolicyNet
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict["states"]), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict["next_states"]), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        # 时序差分误差
        td_error = td_target - self.critic(states)

        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_error.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数


@dataclass
class Args:
    agent_name: str = "ActorCritic"
    env_name: str = "CartPole-v0"
    seed: int = 0
    actor_lr: float = 1e-3
    critic_lr: float = 1e-2
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

    agent = ActorCritic(state_dim, args.hidden_dim, action_dim, args.actor_lr, args.critic_lr, args.gamma, device)
    episode_reward_list = TrainingProcess.train_on_policy_agent(env, agent, args.num_episodes)

    TrainingProcess.save_npy(args.agent_name, args.env_name, args.seed,
                             episode_reward_list)


def main():
    start_time = Timer.get_current_time()
    args = tyro.cli(Args)
    train(args)
    Timer.time_difference(start_time)


if __name__ == "__main__":
    main()
