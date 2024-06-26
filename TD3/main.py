"""
HalfCheetah-v2:
OurDDPG, seed=0, 1h:05m
TD3, seed=0, 1h
"""
import numpy as np
import torch
import gym
import argparse

import utils
import TD3
import OurDDPG
import DDPG
from dataclasses import dataclass
import tyro
import os
import sys

# Windows下运行mujoco
os.add_dll_directory(os.environ["USERPROFILE"] + "/.mujoco/mjpro150/bin")
# （Windows下）运行.sh时，调用相对于当前代码文件夹所在的其他文件夹的代码文件时所需，
# 然后将所调用的代码文件放在该行代码的下面。
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utility.Timer import get_current_time, time_difference


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


@dataclass
class Args:
    # Policy name (TD3, DDPG or OurDDPG)
    policy: str = "TD3"
    # OpenAI gym environment name
    env: str = "HalfCheetah-v2"
    # Sets Gym, PyTorch and Numpy seeds
    seed: int = 7
    # Time steps initial random policy is used
    start_timesteps: int = 25e3
    # How often (time steps) we evaluate
    eval_freq: int = 5e3
    # Max time steps to run environment
    max_timesteps: int = 1e6
    # Std of Gaussian exploration noise
    expl_noise: float = 0.1
    # Batch size for both actor and critic
    batch_size: int = 256
    # Discount factor
    discount: float = 0.99
    # Target network update rate
    tau: float = 0.005
    # Noise added to target policy during critic update
    policy_noise: float = 0.2
    # Range to clip target policy noise
    noise_clip: float = 0.5
    # Frequency of delayed policy updates
    policy_freq: int = 2
    # Save model and optimizer parameters
    # save_model: str = "store_true"
    save_model: bool = False
    # Model load file name, "" doesn't load, "default" uses file_name
    load_model: str = ""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    return args


def main():
    # args = get_args()
    args = tyro.cli(Args)
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    policy = None
    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    # state, done = env.reset(), False
    state = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state)) +
                    np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        # print("-------------------")
        # print(episode_timesteps, env._max_episode_steps)
        # print(done, done_bool)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t + 1} "
                  f"Episode Num: {episode_num + 1} "
                  f"Episode T: {episode_timesteps} "
                  f"Reward: {episode_reward:.3f}")
            # Reset environment
            # state, done = env.reset(), False
            state = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}")


if __name__ == "__main__":
    start_time = get_current_time()
    main()
    time_difference(start_time)
