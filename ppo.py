"""
# Enviroment

Before we begin we must understand how `MountainCarContinuous-v0` enviorment works. In simple term it is just a car trying to climb a mountain and the goal is just this.

Observation Space:
The observation is a ndarray with shape (2,) where the elements correspond to the following:
* 0: position of the car along the x-axis, minimum -1.2 maximum 0.6 (position)
* 1: velocity of the car, minimum -0.07 maximum 0.07 (velocity)

Action Space:
The action is clipped in the range [-1,1] and multiplied by a power of 0.0015.

Reward:
Agent will get positive reward when it reaches the mountain, else negative reward.

Episode End:
The episode ends if either of the following happens:
1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
2. Truncation: The length of the episode is 999.

[read more](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/)

# What we'll impeliment
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# Global parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "MountainCarContinuous-v0"

MAX_EPISODES = 5000
ROLLOUT_LEN = 1024
MINI_BATCH_SIZE = 64
K_EPOCHS = 10

LR = 2e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.3
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
GRAD_CLIP = 0.6
HIDDEN_DIM = 64
PRINT_EVERY = 1


# this is possibile that the observation space get diffrent mean and variance so we are going to
# use this class to update mean and variance. this one is a little new so i use so mch comments here
class RunningMeanStd:
    def __init__(self, shape=()):
        # keeping first eman 0 and variance 1
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        # the count of samples we saw, we dont use 0 tho we stop devision by zero
        self.count = 1e-4

    def update(self, x):
        # increase the accuracy by turning the input tensor to float64
        x = np.array(x, dtype=np.float64)
        # calc mean and variancefor comming batch
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        # batch_count defindes the count of seen samples, at least it must be 1.0
        batch_count = x.shape[0] if x.ndim > 1 else 1.0
        # let's GO FOT UPDATE
        self._update_from_moments(batch_mean, batch_var, batch_count)

    # In this function, we used Welford/Chan algorithm to update mean and var
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        # delta = diffrence between old mean and new mean
        delta = batch_mean - self.mean
        # adding new count with old count so we have now total count
        total_count = self.count + batch_count
        # this is a weighted average between old mean and new batch mean (new_mean)
        new_mean = self.mean + delta * (batch_count / total_count)
        # sum of squared deviations for old data
        m_a = self.var * self.count
        # sum of squared deviations for new batch
        m_b = batch_var * batch_count
        # and this is variance combination using Welford/Chan formula, M2 is sum of square variance
        M2 = m_a + m_b + delta**2 * (self.count * batch_count / total_count)
        # just calvulate new variance
        new_var = M2 / total_count
        # just assign new stuff which we calculated!
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        # remember standard scalare?
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# I use 2 diffrent networks for Actor and Critic
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # one learned log_std per action dimension
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def get_action_and_value(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(state).squeeze(-1)
        return action, log_prob, value, dist.entropy().sum(dim=-1)

    def evaluate_actions(self, states, actions):
        mu, std = self.forward(states)
        dist = torch.distributions.Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(states).squeeze(-1)
        return log_probs, entropy, values


# Now LETS GO FOR PPO AGENT!
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.net = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)

    def compute_gae(self, rewards, values, dones, last_value):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        values_extended = np.append(values, last_value)
        for t in reversed(range(len(rewards))):
            mask = 0.0 if dones[t] else 1.0
            delta = (
                rewards[t] + GAMMA * values_extended[t + 1] * mask - values_extended[t]
            )
            gae = delta + GAMMA * LAMBDA * mask * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, memory):
        states = torch.tensor(
            np.array(memory["states"]), dtype=torch.float32, device=DEVICE
        )
        actions = torch.tensor(
            np.array(memory["actions"]), dtype=torch.float32, device=DEVICE
        )
        old_log_probs = torch.tensor(
            np.array(memory["log_probs"]), dtype=torch.float32, device=DEVICE
        )
        rewards = np.array(memory["rewards"], dtype=np.float32)
        dones = np.array(memory["dones"], dtype=np.bool_)
        values = np.array(memory["values"], dtype=np.float32)

        # compute last value for next state (use 0 if done at last step)
        with torch.no_grad():
            last_state = torch.tensor(
                memory["last_state"], dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            last_value = self.net.critic(last_state).squeeze(-1).cpu().item()

        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(rewards)
        indices = np.arange(dataset_size)

        # multi-epoch minibatch update
        for _ in range(K_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, MINI_BATCH_SIZE):
                mb_idx = indices[start : start + MINI_BATCH_SIZE]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = torch.tensor(
                    advantages[mb_idx], dtype=torch.float32, device=DEVICE
                )
                mb_returns = torch.tensor(
                    returns[mb_idx], dtype=torch.float32, device=DEVICE
                )

                log_probs, entropy, value_preds = self.net.evaluate_actions(
                    mb_states, mb_actions
                )
                ratios = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (mb_returns - value_preds).pow(2).mean()
                loss = (
                    actor_loss
                    + VALUE_LOSS_COEF * critic_loss
                    - ENTROPY_COEF * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP)
                self.optimizer.step()


# we came all here to see this part, the train loop!
def train():
    env = gym.make(ENV_NAME)

    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    agent = PPOAgent(state_dim, action_dim)

    rewards_history = []
    avg_history = []

    # main loop: collect rollouts and update
    episode = 0
    total_steps = 0
    while episode < MAX_EPISODES:
        memory = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "last_state": None,
        }
        steps = 0
        ep_rewards = []
        # collect ROLLOUT_LEN timesteps (may span multiple episodes)
        state, _ = env.reset()
        while steps < ROLLOUT_LEN:
            # normalize observation
            obs_rms.update(state.reshape(1, -1))  # keep updating running stats online
            state_norm = obs_rms.normalize(state)

            state_tensor = torch.tensor(
                state_norm, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor, entropy_tensor = (
                    agent.net.get_action_and_value(state_tensor)
                )
            action = action_tensor.squeeze(0).cpu().numpy()
            log_prob = log_prob_tensor.cpu().item()
            value = value_tensor.cpu().item()

            # add some exploration noise by slightly increasing std via log_std param is already learned,
            # but we clip action to environment bounds:
            action_clipped = np.clip(action, action_low, action_high)
            next_state, reward, terminated, truncated, _ = env.step(action_clipped)
            done = bool(terminated or truncated)

            memory["states"].append(state_norm)  # store normalized state
            memory["actions"].append(action_clipped)
            memory["log_probs"].append(log_prob)
            memory["rewards"].append(float(reward))
            memory["dones"].append(done)
            memory["values"].append(value)

            state = next_state
            steps += 1
            total_steps += 1

            if done:
                episode += 1
                # track episode reward (optional: compute per-episode)
                # we don't break rollout on episode end, we continue collecting
                # but to report we might want to sample episode reward by running until done
                # so we reset environment
                state, _ = env.reset()

        # after rollout, remember last state for bootstrap
        memory["last_state"] = (
            state  # raw (unnormalized) next state before normalization -> will be normalized inside update if needed
        )

        # update policy
        agent.update(memory)

        # optionally evaluate a full episode reward for logging
        # run one evaluation episode (deterministic: use mu) for stable logging (cheap)
        eval_state, _ = env.reset(seed=9999)
        done_eval = False
        ep_reward = 0.0
        while not done_eval:
            state_norm = obs_rms.normalize(eval_state)
            state_tensor = torch.tensor(
                state_norm, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            with torch.no_grad():
                mu, _ = agent.net.forward(state_tensor)
            action = mu.squeeze(0).cpu().numpy()
            action = np.clip(action, action_low, action_high)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done_eval = bool(terminated or truncated)
            ep_reward += float(reward)
            eval_state = next_state

        rewards_history.append(ep_reward)
        avg20 = np.mean(rewards_history[-20:])
        avg_history.append(avg20)

        if len(rewards_history) % PRINT_EVERY == 0:
            print(
                f"[PPO] Ep {len(rewards_history):4d} | Eval reward: {ep_reward:7.2f} | Avg20: {avg20:7.2f}"
            )

    env.close()

    # plot
    plt.figure(figsize=(10, 4))
    plt.plot(rewards_history, label="eval episode reward")
    plt.plot(avg_history, label="avg20", linewidth=2)
    plt.xlabel("Update number")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.title(f"PPO on {ENV_NAME}")
    plt.show()


if __name__ == "__main__":
    train()
