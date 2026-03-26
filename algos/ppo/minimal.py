from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class LineWorld:
    def __init__(self, length: int = 7) -> None:
        assert length >= 3
        self.length = length
        self.start_state = length // 2
        self.state = self.start_state

    @property
    def num_states(self) -> int:
        return self.length

    @property
    def num_actions(self) -> int:
        return 2

    def encode(self, state: int) -> np.ndarray:
        obs = np.zeros(self.length, dtype=np.float32)
        obs[state] = 1.0
        return obs

    def reset(self) -> np.ndarray:
        self.state = self.start_state
        return self.encode(self.state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        move = -1 if action == 0 else 1
        next_state = min(max(self.state + move, 0), self.length - 1)
        self.state = next_state

        if next_state == 0:
            return self.encode(next_state), -1.0, True
        if next_state == self.length - 1:
            return self.encode(next_state), 1.0, True
        return self.encode(next_state), -0.02, False


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


def collect_rollout(
    env: LineWorld,
    model: PolicyValueNet,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states: list[np.ndarray] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    rewards: list[float] = []
    values: list[float] = []
    next_values: list[float] = []
    dones: list[float] = []

    obs = env.reset()
    for _ in range(rollout_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_tensor)
            dist = Categorical(logits=logits)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)

        action = int(action_tensor.item())
        next_obs, reward, done = env.step(action)

        with torch.no_grad():
            if done:
                next_value = 0.0
            else:
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
                _, next_value_tensor = model(next_obs_tensor)
                next_value = float(next_value_tensor.item())

        states.append(obs)
        actions.append(action)
        old_log_probs.append(float(log_prob.item()))
        rewards.append(reward)
        values.append(float(value.item()))
        next_values.append(next_value)
        dones.append(float(done))

        obs = env.reset() if done else next_obs

    advantages = np.zeros(rollout_steps, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(rollout_steps)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae

    returns = advantages + np.asarray(values, dtype=np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return (
        torch.tensor(np.stack(states), dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(old_log_probs, dtype=torch.float32),
        torch.tensor(advantages, dtype=torch.float32),
        torch.tensor(returns, dtype=torch.float32),
    )


def ppo_update(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
) -> tuple[float, float, float]:
    states, actions, old_log_probs, advantages, returns = batch
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    for _ in range(epochs):
        logits, values = model(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        ratios = torch.exp(log_probs - old_log_probs)

        unclipped = ratios * advantages
        clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = F.mse_loss(values, returns)
        entropy = dist.entropy().mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_policy_loss = float(policy_loss.item())
        last_value_loss = float(value_loss.item())
        last_entropy = float(entropy.item())

    return last_policy_loss, last_value_loss, last_entropy


def evaluate(env: LineWorld, model: PolicyValueNet) -> tuple[list[int], float]:
    obs = env.reset()
    trajectory = [int(np.argmax(obs))]
    total_reward = 0.0
    done = False

    while not done and len(trajectory) < env.length + 2:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_tensor)
            action = int(logits.argmax(dim=-1).item())

        obs, reward, done = env.step(action)
        total_reward += reward
        trajectory.append(int(np.argmax(obs)))

    return trajectory, total_reward


def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False)

    env = LineWorld(length=7)
    model = PolicyValueNet(
        input_dim=env.num_states,
        hidden_dim=32,
        num_actions=env.num_actions,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _ in range(60):
        batch = collect_rollout(
            env=env,
            model=model,
            rollout_steps=128,
            gamma=0.95,
            gae_lambda=0.95,
        )
        ppo_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            epochs=4,
        )

    trajectory, total_reward = evaluate(env, model)
    with torch.no_grad():
        logits, values = model(torch.eye(env.num_states))
        action_probs = torch.softmax(logits, dim=-1)

    print("Policy probabilities:")
    print(action_probs)
    print("\nState values:")
    print(values)
    print("\nGreedy trajectory:")
    print(trajectory)
    print("Greedy return:", round(total_reward, 3))


if __name__ == "__main__":
    main()
